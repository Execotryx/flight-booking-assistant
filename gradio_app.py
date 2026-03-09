"""Gradio UI for the flight ticket booking agent."""

from __future__ import annotations

import os

import gradio as gr
from gradio.components.chatbot import Message, MessageDict

ChatbotMessage = MessageDict | Message

from flight_booking_agent import (
    ANSWER_MODEL_NAME,
    REASONING_MODEL_NAME,
    FlightTicketBookingAgent,
)


class FlightAssistantUI:
    """Class-based Gradio UI wrapper for the flight booking assistant."""

    def __init__(self, agent: FlightTicketBookingAgent | None = None) -> None:
        """Create a UI instance.

        Parameters:
            agent: Optional existing booking agent instance. When omitted,
                a new `FlightTicketBookingAgent` is created.
        """
        self._agent: FlightTicketBookingAgent = (
            agent if agent is not None else FlightTicketBookingAgent()
        )
        self._initial_message: str = "Hello! How can I assist with your flight today?"

    def _respond(
        self,
        message: str,
        history: list[ChatbotMessage],
        current_postcard_path: str | None,
    ) -> tuple[
        list[ChatbotMessage],
        list[ChatbotMessage],
        str,
        str | None,
        str | None,
    ]:
        """Handle one chat turn and update all UI state outputs.

        Parameters:
            message: The latest user text from the input box.
            history: Current chat history state from Gradio.
            current_postcard_path: Existing image path currently shown in the postcard panel.
        """
        if not message.strip():
            return (
                history,
                history,
                "",
                current_postcard_path,
                current_postcard_path,
            )

        answer = self._agent.ask(message)
        updated_history = list(history)
        updated_history.append({"role": "user", "content": message})
        updated_history.append({"role": "assistant", "content": answer})

        generated_postcard = self._agent.get_latest_postcard_path()
        if generated_postcard and os.path.exists(generated_postcard):
            postcard_path = generated_postcard
        else:
            postcard_path = current_postcard_path

        return (
            updated_history,
            updated_history,
            "",
            postcard_path,
            postcard_path,
        )

    def create_app(self) -> gr.Blocks:
        """Build and return the Gradio Blocks application.

        Returns:
            A fully wired Gradio app instance with chat and postcard panels.
        """
        with gr.Blocks(title="Flight Ticket Booking Agent") as app:
            gr.Markdown("# Flight Ticket Booking Agent")
            gr.Markdown(
                "Specialized assistant for flight booking only. "
                f"Model routing: reasoning={REASONING_MODEL_NAME}, "
                f"answering={ANSWER_MODEL_NAME}"
            )

            initial_history: list[ChatbotMessage] = [
                {"role": "assistant", "content": self._initial_message}
            ]
            chat_history_state = gr.State(initial_history)
            postcard_state = gr.State(None)

            with gr.Row():
                chatbot = gr.Chatbot(
                    label="Chatbot",
                    height=420,
                    value=initial_history,
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
                fn=self._respond,
                inputs=[message_box, chat_history_state, postcard_state],
                outputs=[
                    chatbot,
                    chat_history_state,
                    message_box,
                    postcard_image,
                    postcard_state,
                ],
            )

            message_box.submit(
                fn=self._respond,
                inputs=[message_box, chat_history_state, postcard_state],
                outputs=[
                    chatbot,
                    chat_history_state,
                    message_box,
                    postcard_image,
                    postcard_state,
                ],
            )

        return app

    def run(self) -> None:
        """Launch the Gradio application in the default browser."""
        app = self.create_app()
        app.launch(inbrowser=True)


def create_chatbot() -> gr.Blocks:
    """Create a Gradio app instance for integration points expecting a function.

    Returns:
        A configured `gr.Blocks` application.
    """
    return FlightAssistantUI().create_app()


def run_gradio() -> None:
    """Launch the Gradio chat application."""
    FlightAssistantUI().run()


if __name__ == "__main__":
    run_gradio()
