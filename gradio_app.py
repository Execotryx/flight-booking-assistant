"""Gradio UI for the flight ticket booking agent."""

from __future__ import annotations

import gradio as gr

from flight_booking_agent import (
    ANSWER_MODEL_NAME,
    REASONING_MODEL_NAME,
    FlightTicketBookingAgent,
)


def create_chatbot() -> gr.ChatInterface:
    """Create the Gradio chat interface bound to one agent instance."""
    agent = FlightTicketBookingAgent()

    def respond(
        message: str,
        history: list[tuple[str, str]],
    ) -> str:
        """Handle one chat turn from Gradio and return assistant text.

        Parameters:
            message: Current user message from the chat input.
            history: Prior chat turns supplied by Gradio.
        """
        _ = history
        return agent.ask(message)

    return gr.ChatInterface(
        fn=respond,
        title="Flight Ticket Booking Agent",
        description=(
            "Specialized assistant for flight booking only. "
            f"Model routing: reasoning={REASONING_MODEL_NAME}, "
            f"answering={ANSWER_MODEL_NAME}"
        ),
    )


def run_gradio() -> None:
    """Launch the Gradio chat application."""
    app = create_chatbot()
    app.launch(inbrowser=True)


if __name__ == "__main__":
    run_gradio()
