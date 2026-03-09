"""Gradio UI for the flight ticket booking agent."""

from __future__ import annotations

import os

import gradio as gr

from flight_booking_agent import (
    ANSWER_MODEL_NAME,
    REASONING_MODEL_NAME,
    FlightTicketBookingAgent,
)


def create_chatbot() -> gr.Blocks:
    """Create a custom Gradio interface with chat + postcard display blocks."""
    agent = FlightTicketBookingAgent()
    initial_message = "Hello! How can I assist with your flight today?"

    def respond(
        message: str,
        history: list[dict[str, str]],
        current_postcard_path: str | None,
    ) -> tuple[list[dict[str, str]], str, str | None, str | None, str | None]:
        """Handle one chat turn and update chat, postcard panel, and audio strip.

        Parameters:
            message: Current user message from the chat input.
            history: Prior chat turns supplied by Gradio.
            current_postcard_path: Existing postcard path shown in the UI.
        """
        if not message.strip():
            return history, "", current_postcard_path, current_postcard_path, None

        answer = agent.ask(message)
        updated_history = list(history)
        updated_history.append({"role": "user", "content": message})
        updated_history.append({"role": "assistant", "content": answer})

        generated_postcard = agent.get_latest_postcard_path()
        if generated_postcard and os.path.exists(generated_postcard):
            postcard_path = generated_postcard
        else:
            postcard_path = current_postcard_path

        return updated_history, "", postcard_path, postcard_path, None

    with gr.Blocks(title="Flight Ticket Booking Agent") as app:
        gr.Markdown("# Flight Ticket Booking Agent")
        gr.Markdown(
            "Specialized assistant for flight booking only. "
            f"Model routing: reasoning={REASONING_MODEL_NAME}, "
            f"answering={ANSWER_MODEL_NAME}"
        )

        chat_history_state = gr.State(
            [{"role": "assistant", "content": initial_message}]
        )
        postcard_state = gr.State(None)

        with gr.Row():
            chatbot = gr.Chatbot(
                label="Chatbot",
                height=420,
                value=[{"role": "assistant", "content": initial_message}],
            )
            postcard_image = gr.Image(
                label="Image",
                type="filepath",
                height=420,
                interactive=False,
            )

        with gr.Row():
            message_box = gr.Textbox(
                label="Chat with our AI Assistant:",
                placeholder="Ask about flights, booking, or destinations...",
                scale=8,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)

        send_button.click(
            fn=respond,
            inputs=[message_box, chat_history_state, postcard_state],
            outputs=[chatbot, message_box, postcard_image, postcard_state],
        ).then(
            fn=lambda chat_data: chat_data,
            inputs=[chatbot],
            outputs=[chat_history_state],
        )

        message_box.submit(
            fn=respond,
            inputs=[message_box, chat_history_state, postcard_state],
            outputs=[chatbot, message_box, postcard_image, postcard_state],
        ).then(
            fn=lambda chat_data: chat_data,
            inputs=[chatbot],
            outputs=[chat_history_state],
        )

    return app


def run_gradio() -> None:
    """Launch the Gradio chat application."""
    app = create_chatbot()
    app.launch(inbrowser=True)


if __name__ == "__main__":
    run_gradio()
