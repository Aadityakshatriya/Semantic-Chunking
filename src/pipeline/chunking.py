import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Heavy deps imported lazily inside functions where possible (faster startup).


# ----------------------------
# YouTube -> WAV (16k mono)
# ----------------------------
def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nstderr:\n{p.stderr}")


def download_youtube_to_wav_16k(youtube_url: str, wav_path: str) -> None:
    """
    Requires:
      - yt-dlp installed and on PATH
      - ffmpeg installed and on PATH
    """
    with tempfile.TemporaryDirectory() as td:
        in_template = os.path.join(td, "input.%(ext)s")
        _run(["yt-dlp", "-f", "bestaudio", "-o", in_template, youtube_url])

        # find downloaded file
        downloaded = None
        for name in os.listdir(td):
            if name.startswith("input."):
                downloaded = os.path.join(td, name)
                break
        if not downloaded:
            raise RuntimeError("yt-dlp did not produce an input file.")

        _run(["ffmpeg", "-y", "-i", downloaded, "-ac", "1", "-ar", "16000", wav_path])


# ----------------------------
# VAD -> silence midpoints
# ----------------------------
def compute_vad_silence_midpoints(wav_path: str, min_silence_s: float = 0.1) -> List[float]:
    import torchaudio
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    audio = waveform.squeeze().to(torch.float32)

    vad_model = load_silero_vad()
    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=16000,
        min_speech_duration_ms=300,
        min_silence_duration_ms=int(min_silence_s * 1000),
    )

    speech_segments = sorted(
        [
            {"start": ts["start"] / 16000.0, "end": ts["end"] / 16000.0}
            for ts in speech_timestamps
        ],
        key=lambda x: x["start"],
    )

    silences = []
    for i in range(len(speech_segments) - 1):
        s_end = speech_segments[i]["end"]
        n_start = speech_segments[i + 1]["start"]
        dur = n_start - s_end
        if dur >= min_silence_s:
            silences.append(0.5 * (s_end + n_start))

    return silences


# ----------------------------
# Diarization (optional)
# ----------------------------
def run_diarization_segments(wav_path: str, hf_token: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not hf_token:
        return None

    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    diarization = pipeline(wav_path)
    diar_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append(
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        )
    diar_segments.sort(key=lambda x: x["start"])
    return diar_segments


def assign_speakers_to_words(
    words: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
    default_speaker: str = "UNK",
    min_overlap: float = 0.2,
    max_snap: float = 0.25,
) -> List[Dict[str, Any]]:
    diar_segments = sorted(diar_segments, key=lambda x: x["start"])

    for w in words:
        ws = float(w["start"])
        we = float(w["end"])
        if we <= ws:
            w["speaker"] = default_speaker
            continue

        best_spk = default_speaker
        best_ov = 0.0

        for seg in diar_segments:
            ss = float(seg["start"])
            se = float(seg["end"])
            ov = max(0.0, min(we, se) - max(ws, ss))
            if ov > best_ov:
                best_ov = ov
                best_spk = seg["speaker"]

        if best_ov >= min_overlap:
            w["speaker"] = best_spk
            continue

        mid = 0.5 * (ws + we)
        nearest_spk = default_speaker
        nearest_dist = float("inf")
        for seg in diar_segments:
            ss = float(seg["start"])
            se = float(seg["end"])
            dist = 0.0 if (ss <= mid <= se) else min(abs(mid - ss), abs(mid - se))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_spk = seg["speaker"]

        w["speaker"] = nearest_spk if nearest_dist <= max_snap else default_speaker

    return words


def speaker_stats_between(words, start, end, default_speaker="UNK"):
    spks = []
    for w in words:
        mid = 0.5 * (w["start"] + w["end"])
        if start <= mid < end:
            spks.append(w.get("speaker", default_speaker))

    if not spks:
        return {"dominant": default_speaker, "purity": 0.0, "counts": Counter()}

    counts = Counter(spks)
    dominant, dom_n = counts.most_common(1)[0]
    return {"dominant": dominant, "purity": dom_n / len(spks), "counts": counts}


# ----------------------------
# Transcription -> words
# ----------------------------
def transcribe_words_faster_whisper(
    wav_path: str,
    model_size: str = "large-v3",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from faster_whisper import WhisperModel

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        wav_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=False,
    )

    words: List[Dict[str, Any]] = []
    for seg in segments:
        if seg.words is None:
            continue
        for w in seg.words:
            words.append(
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "text": str(w.word).strip(),
                }
            )

    words.sort(key=lambda x: x["start"])
    return words


# ----------------------------
# Embeddings + similarity
# ----------------------------
_embedder = None


def embed(text: str) -> np.ndarray:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if not text.strip():
        return np.zeros(_embedder.get_sentence_embedding_dimension(), dtype=np.float32)

    v = _embedder.encode(text, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ----------------------------
# Chunking helpers (match Sarvam_Assignment.ipynb)
# ----------------------------
FILLER_WORDS = {"um", "uh", "umm", "uhh"}
PUNCTUATION = {".", "?", "!", "..."}

MAX_LEN = 15.0          # seconds
MIN_SIDE = 4.0          # seconds (notebook output shows ~4s minimum)
BALANCE_LAMBDA = 0.2

SPEAKER_PURITY_MIN = 0.9
SPEAKER_BONUS = 0.2
SPEAKER_PENALTY = 0.1


def speaker_change_candidates(
    words,
    segment,
    ignore_speakers={"UNK"},
    min_gap=0.2,          # seconds
    min_run_words=1,      # require this many words on each side
):
    seg_start, seg_end = float(segment["start"]), float(segment["end"])

    seg_words = []
    for w in words:
        mid = 0.5 * (float(w["start"]) + float(w["end"]))
        if seg_start <= mid < seg_end:
            seg_words.append(w)

    if len(seg_words) < (2 * min_run_words + 1):
        return []

    candidates = []
    for i in range(min_run_words, len(seg_words) - min_run_words):
        w1 = seg_words[i - 1]
        w2 = seg_words[i]
        spk1 = w1.get("speaker", "UNK")
        spk2 = w2.get("speaker", "UNK")

        if spk1 in ignore_speakers or spk2 in ignore_speakers:
            continue
        if spk1 == spk2:
            continue

        gap = float(w2["start"]) - float(w1["end"])
        if gap < min_gap:
            continue

        t = 0.5 * (float(w1["end"]) + float(w2["start"]))
        if seg_start < t < seg_end:
            candidates.append(t)

    return sorted(set(round(float(t), 3) for t in candidates))


def word_boundary_candidates(words, segment, min_gap=0.05):
    """
    Notebook fallback candidates: acoustic gaps + filler words + punctuation + speaker-change cues.
    """
    seg_start, seg_end = float(segment["start"]), float(segment["end"])

    span = []
    for w in words:
        mid = 0.5 * (float(w["start"]) + float(w["end"]))
        if seg_start <= mid < seg_end:
            span.append(w)

    candidates = []
    for w1, w2 in zip(span, span[1:]):
        gap = float(w2["start"]) - float(w1["end"])

        # Small acoustic gap (even < VAD threshold)
        if gap >= min_gap:
            candidates.append(0.5 * (float(w1["end"]) + float(w2["start"])))

        t1 = str(w1.get("text", "")).strip()
        if t1.lower() in FILLER_WORDS:
            candidates.append(float(w1["end"]))

        if any(p in t1 for p in PUNCTUATION):
            candidates.append(float(w1["end"]))

    # NEW: speaker-change cues
    candidates += speaker_change_candidates(
        words,
        segment,
        ignore_speakers={"UNK"},
        min_gap=0.0,  # create more potential avenues for chunking
        min_run_words=1,
    )

    return sorted(set(round(float(t), 3) for t in candidates))


def imbalance_penalty(left_dur: float, right_dur: float) -> float:
    total = left_dur + right_dur
    if total <= 0:
        return 0.0
    return abs(left_dur - right_dur) / total  # 0 (balanced) .. ~1 (very imbalanced)


def score_all_splits(
    current_chunk,
    all_chunks,
    words,
    silence_times,
    embed,
    left_context_size=1,
    right_context_size=1,
    include_speaker_candidates=True,
    ignore_speakers={"UNK"},
):
    scored = []

    # --- build candidate times ---
    candidate_times = list(silence_times) if silence_times is not None else []

    if include_speaker_candidates:
        candidate_times += speaker_change_candidates(
            words,
            current_chunk,
            ignore_speakers=ignore_speakers,
            min_gap=0.0,
            min_run_words=1,
        )

    candidate_times = sorted(set(round(float(t), 3) for t in candidate_times))

    # find index for context windows
    try:
        idx = all_chunks.index(current_chunk)
    except ValueError:
        idx = [
            i
            for i, ch in enumerate(all_chunks)
            if ch["start"] == current_chunk["start"] and ch["end"] == current_chunk["end"]
        ][0]

    for t in candidate_times:
        if not (float(current_chunk["start"]) < float(t) < float(current_chunk["end"])):
            continue

        left_dur = float(t) - float(current_chunk["start"])
        right_dur = float(current_chunk["end"]) - float(t)

        # hard constraint to prevent tiny chunks
        if left_dur < MIN_SIDE or right_dur < MIN_SIDE:
            continue

        left_chunk = {"start": float(current_chunk["start"]), "end": float(t)}
        right_chunk = {"start": float(t), "end": float(current_chunk["end"])}

        left_text = text_between(words, left_chunk["start"], left_chunk["end"])
        right_text = text_between(words, right_chunk["start"], right_chunk["end"])
        if not left_text.strip() or not right_text.strip():
            continue

        # --- semantic score ---
        E_left = embed(left_text)
        E_right = embed(right_text)

        left_neighbors = all_chunks[max(0, idx - left_context_size) : idx]
        extended_left = left_neighbors + [left_chunk]
        ext_left_text = text_for_chunks(words, extended_left)
        if not ext_left_text.strip():
            continue
        E_ext_left = embed(ext_left_text)

        right_neighbors = all_chunks[idx + 1 : idx + 1 + right_context_size]
        extended_right = [right_chunk] + right_neighbors
        ext_right_text = text_for_chunks(words, extended_right)
        if not ext_right_text.strip():
            continue
        E_ext_right = embed(ext_right_text)

        S_local = 1.0 - cosine_similarity(E_left, E_right)
        S_ext = 1.0 - cosine_similarity(E_ext_left, E_ext_right)
        C_left = cosine_similarity(E_left, E_ext_left)
        C_right = cosine_similarity(E_right, E_ext_right)
        C_int = C_left + C_right

        score = 1.0 * S_local + 0.5 * S_ext + 0.5 * C_int

        # soft penalty for uneven splits
        score -= BALANCE_LAMBDA * imbalance_penalty(left_dur, right_dur)

        # speaker-aware scoring
        Ls = speaker_stats_between(words, left_chunk["start"], left_chunk["end"], default_speaker="UNK")
        Rs = speaker_stats_between(words, right_chunk["start"], right_chunk["end"], default_speaker="UNK")

        L_dom, R_dom = Ls.get("dominant", "UNK"), Rs.get("dominant", "UNK")
        L_pur, R_pur = float(Ls.get("purity", 0.0)), float(Rs.get("purity", 0.0))

        if (
            L_dom not in ignore_speakers
            and R_dom not in ignore_speakers
            and L_pur >= SPEAKER_PURITY_MIN
            and R_pur >= SPEAKER_PURITY_MIN
        ):
            if L_dom != R_dom:
                score += SPEAKER_BONUS
            else:
                score -= SPEAKER_PENALTY

        scored.append((float(t), float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def best_fallback_split(
    segment,
    all_chunks,
    words,
    embed,
    left_context_size=1,
    right_context_size=1,
    ignore_speakers={"UNK"},
):
    """
    Notebook fallback: generate word/punctuation candidates, then score via score_all_splits.
    """
    fallback_times = word_boundary_candidates(words, segment)
    if not fallback_times:
        return None

    scored = score_all_splits(
        current_chunk=segment,
        all_chunks=all_chunks,
        words=words,
        silence_times=fallback_times,
        embed=embed,
        left_context_size=left_context_size,
        right_context_size=right_context_size,
        include_speaker_candidates=False,  # IMPORTANT: word_boundary_candidates already adds these
        ignore_speakers=ignore_speakers,
    )

    return float(scored[0][0]) if scored else None


def run_segmentation(
    words,
    silence_times,
    embed,
    local_context=1,
    right_context=1,
    include_speaker_candidates=True,
    ignore_speakers={"UNK"},
):
    current_chunks = [{"start": 0.0, "end": float(words[-1]["end"])}]

    i = 0
    while i < len(current_chunks):
        segment = current_chunks[i]
        start, end = float(segment["start"]), float(segment["end"])
        duration = end - start

        if duration <= MAX_LEN:
            i += 1
            continue

        if duration < 2 * MIN_SIDE:
            i += 1
            continue

        # 1) Build per-segment combined candidates: VAD silences (inside segment) + optional speaker changes
        seg_candidate_times = []
        if silence_times is not None:
            seg_candidate_times.extend(float(t) for t in silence_times if start < float(t) < end)

        if include_speaker_candidates:
            seg_candidate_times.extend(
                speaker_change_candidates(
                    words,
                    segment,
                    ignore_speakers=ignore_speakers,
                    min_gap=0.0,
                    min_run_words=1,
                )
            )

        seg_candidate_times = sorted(set(round(float(t), 3) for t in seg_candidate_times))

        # 2) Score the combined candidates (IMPORTANT: avoid double-adding speaker candidates)
        scored = score_all_splits(
            current_chunk=segment,
            all_chunks=current_chunks,
            words=words,
            silence_times=seg_candidate_times,
            embed=embed,
            left_context_size=local_context,
            right_context_size=right_context,
            include_speaker_candidates=False,
            ignore_speakers=ignore_speakers,
        )

        if scored:
            cut_time = float(scored[0][0])
        else:
            # 3) Fallback: word-based candidates + same scorer
            cut_time = best_fallback_split(
                segment,
                all_chunks=current_chunks,
                words=words,
                embed=embed,
                left_context_size=local_context,
                right_context_size=right_context,
                ignore_speakers=ignore_speakers,
            )

        # 4) Absolute last resort
        if cut_time is None or not (start < float(cut_time) < end):
            cut_time = 0.5 * (start + end)

        # 5) Clamp to respect MIN_SIDE
        cut_time = float(cut_time)
        cut_time = max(start + MIN_SIDE, min(cut_time, end - MIN_SIDE))

        if not (start + MIN_SIDE < cut_time < end - MIN_SIDE):
            i += 1
            continue

        left_chunk = {"start": start, "end": cut_time}
        right_chunk = {"start": cut_time, "end": end}

        current_chunks[i : i + 1] = [left_chunk, right_chunk]
        # Do NOT increment i: re-check the new left chunk first

    return current_chunks


# ----------------------------
# Public entrypoint for Gradio
# ----------------------------
def chunk_youtube(
    youtube_url: str,
    hf_token: Optional[str] = None,
    diarize: bool = True,
) -> List[Dict[str, Any]]:
    youtube_url = (youtube_url or "").strip()
    if not youtube_url:
        raise ValueError("Please provide a YouTube URL.")

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")
        download_youtube_to_wav_16k(youtube_url, wav_path)

        silence_times = compute_vad_silence_midpoints(wav_path)

        words = transcribe_words_faster_whisper(wav_path)

        diar_segments = None
        if diarize:
            diar_segments = run_diarization_segments(wav_path, hf_token=hf_token)

        if diar_segments:
            words = assign_speakers_to_words(words, diar_segments)
        else:
            for w in words:
                w["speaker"] = "UNK"

        chunks = run_segmentation(
            words=words,
            silence_times=vad_silence_times,
            embed=embed,
            local_context=5,
            right_context=5,
            include_speaker_candidates=bool(diar_segments),
            ignore_speakers={"UNK"},
        )

        chunks_sorted = sorted(chunks, key=lambda c: float(c["start"]))
        output_chunks = []
        for i, c in enumerate(chunks_sorted, start=1):
            start = float(c["start"])
            end = float(c["end"])
            output_chunks.append(
                {
                    "chunk_id": i,
                    "chunk_length": float(end - start),
                    "text": text_between(words, start, end).strip(),
                    "start_time": start,
                    "end_time": end,
                }
            )

        return output_chunks


def output_chunks_to_pretty_json(output_chunks: List[Dict[str, Any]]) -> str:
    return json.dumps(output_chunks, ensure_ascii=False, indent=2)