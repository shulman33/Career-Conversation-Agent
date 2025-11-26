from dotenv import load_dotenv
load_dotenv(override=True)

import gradio as gr
from career_manager import CareerManager


async def respond(message: str, history: list):
    """Async generator for Gradio to call directly."""
    if not message or not message.strip():
        yield history
        return
    async for updated_history in manager.run(message, history):
        yield updated_history


def handle_example_select(evt: gr.SelectData, history: list):
    """Handle when user clicks an example in the chatbot."""
    message = evt.value.get("text", "") if evt.value else ""
    return message, history


if __name__ == "__main__":
    manager = CareerManager()

    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="sky"),
    ) as demo:
        chatbot = gr.Chatbot(
            type="messages",
            height="calc(100vh - 120px)",
            elem_id="chatbot",
            examples=[
                {"text": "Tell me about yourself"},
                {"text": "What side projects are you working on?"},
                {"text": "What's your experience with AWS?"},
                {"text": "Are you interested in startup opportunities?"},
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

        chatbot.example_select(handle_example_select, [chatbot], [msg, chatbot]).then(
            respond, [msg, chatbot], chatbot
        ).then(lambda: "", outputs=msg)

    demo.launch()
