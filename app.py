from dotenv import load_dotenv
load_dotenv(override=True)

import gradio as gr
from career_manager import CareerManager


async def respond(message: str, history: list):
    """Async generator for Gradio to call directly."""
    async for updated_history in manager.run(message, history):
        yield updated_history


if __name__ == "__main__":
    manager = CareerManager()

    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="sky"),
        fill_height=True,
        fill_width=True,
        css="""
            .gradio-container { padding: 0 !important; }
            #chatbot { flex-grow: 1; }
        """
    ) as demo:
        chatbot = gr.Chatbot(
            type="messages",
            height="100%",
            scale=1,
            container=False,
            elem_id="chatbot",
            resizable=False,
            placeholder="Ask me anything about Sam's career, skills, or experience!",
            examples=[
                {"text": "How did you make this?"},
                {"text": "What is your professional background?"},
                {"text": "What are your key skills?"},
                {"text": "Tell me about your experience"},
                {"text": "How can I get in touch with you?"},
            ]
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                scale=9,
                container=False,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        msg.submit(respond, [msg, chatbot], chatbot).then(
            lambda: "", outputs=msg
        )
        submit_btn.click(respond, [msg, chatbot], chatbot).then(
            lambda: "", outputs=msg
        )

    demo.launch()
