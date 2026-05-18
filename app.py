import os
import gradio as gr

def run_demo():
    return "Autodrive AI app is running. main.py logic should be connected here."

demo=gr.Interface(
    fn=run_demo,
    inputs=[],
    outputs="text",
    title="Autodrive AI"
)

if __name__=="__main__":
    port=int(os.environ.get("PORT",7860))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port
    )