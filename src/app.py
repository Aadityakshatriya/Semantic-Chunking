import os
import gradio as gr

from pipeline.chunking import chunk_youtube, output_chunks_to_pretty_json


def run_pipeline(youtube_url: str, diarize: bool) -> str:
    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    # If diarization is enabled but token missing, just proceed without diarization.
    if diarize and not hf_token:
        diarize = False

    output_chunks = chunk_youtube(
        youtube_url=youtube_url,
        hf_token=hf_token,
        diarize=diarize,
    )
    return output_chunks_to_pretty_json(output_chunks)


with gr.Blocks(title="YouTube Semantic Chunker") as demo:
    gr.Markdown("# YouTube Semantic Chunker")
    gr.Markdown("Paste a YouTube link → get `output_chunks` back as JSON.")

    url = gr.Textbox(label="YouTube Link", placeholder="https://www.youtube.com/watch?v=...")
    diarize = gr.Checkbox(label="Enable speaker diarization (requires HF_TOKEN)", value=False)

    out = gr.Textbox(label="Chunks (output_chunks)", lines=22)

    btn = gr.Button("Run")
    btn.click(run_pipeline, inputs=[url, diarize], outputs=[out])

if __name__ == "__main__":
    demo.queue().launch()