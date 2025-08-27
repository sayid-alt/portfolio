import gradio as gr
from functions import respond, reset_memory

with gr.Blocks() as demo:
    gr.Markdown("# Meet Jungler!")
    gr.Markdown("Get to know more about Junge Leiter")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your question here...", label="Your Message")
    clear_btn = gr.Button("Reset Memory")
    status = gr.Label()

    msg.submit(respond, inputs=msg, outputs=[chatbot, msg])
    clear_btn.click(reset_memory, outputs=[chatbot, status])


if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch(share=True)