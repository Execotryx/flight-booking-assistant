"""Flight ticket booking agent built on top of OllamaAICore with tool calling."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_config import AIConfig
from ollama_core import OllamaAICore

REASONING_MODEL_NAME: str = "lfm2.5-thinking:1.2b-q8_0"
ANSWER_MODEL_NAME: str = "smollm2:latest"

REASONING_SYSTEM_PROMPT: str = """
You are a strict routing classifier.
Task: classify whether a user request is related to flight travel or airline ticket booking.

Return exactly one token:
- FLIGHT_RELATED
- NOT_FLIGHT_RELATED

Rules:
- FLIGHT_RELATED: asks to search/book/cancel/change flights, airports, airlines, baggage, check-in,
  boarding, fares, itineraries, or passenger details for air travel.
- NOT_FLIGHT_RELATED: everything else.
- If uncertain, return NOT_FLIGHT_RELATED.
- Do not add explanations, punctuation, or extra words.
""".strip()


BOOKING_SYSTEM_PROMPT: str = """
You are FlightTicketAgent, a specialized assistant for airline ticket booking only.

Scope and policy:
- You MUST only answer topics directly related to flight travel and booking.
- Allowed topics: searching flights, fares, seat classes, baggage policies, booking, cancellation,
  rescheduling, airport/airline details relevant to an itinerary, and payment steps for a booking.
- Disallowed topics: programming, politics, health, law, finance advice, general trivia, or any
  unrelated conversation.
- If the user asks anything outside flight booking/travel, refuse briefly and steer back to booking:
  "I can only help with flight booking and flight-travel questions."

Booking behavior:
- Be concise and practical.
- Ask for missing required details before booking: origin, destination, date, passenger name.
- Never invent available flights. Always use the provided tools to search, quote, create, cancel,
  or inspect bookings.
- For prices and IDs, trust and cite tool outputs.
- If a tool fails or returns no result, explain plainly and offer next booking-related step.
- Never claim a booking is confirmed unless the booking tool returns a booking_id.
""".strip()

BookingRecord = dict[str, Any]


@dataclass(frozen=True)
class Flight:
    """Immutable in-memory flight inventory entry."""

    flight_id: str
    airline: str
    origin: str
    destination: str
    date: str
    depart_time: str
    arrive_time: str
    base_fare_usd: int
    seats_left: int


class FlightTicketBookingAgent:
    """Domain-restricted booking assistant powered by OllamaAICore."""

    def __init__(self, config: AIConfig | None = None) -> None:
        """Initialize reasoning and answer models with in-memory booking state."""
        self._reasoning_core = OllamaAICore(
            config=config,
            system_behavior=REASONING_SYSTEM_PROMPT,
            model_name=REASONING_MODEL_NAME,
            temperature=0.0,
            max_tokens=8,
        )
        self._answer_core = OllamaAICore(
            config=config,
            system_behavior=BOOKING_SYSTEM_PROMPT,
            model_name=ANSWER_MODEL_NAME,
        )

        self._flights: dict[str, Flight] = {
            "FL-1001": Flight(
                flight_id="FL-1001",
                airline="SkyJet",
                origin="NYC",
                destination="LON",
                date="2026-03-20",
                depart_time="09:10",
                arrive_time="21:45",
                base_fare_usd=640,
                seats_left=4,
            ),
            "FL-1002": Flight(
                flight_id="FL-1002",
                airline="AeroWays",
                origin="NYC",
                destination="LON",
                date="2026-03-20",
                depart_time="18:35",
                arrive_time="07:05",
                base_fare_usd=560,
                seats_left=7,
            ),
            "FL-2001": Flight(
                flight_id="FL-2001",
                airline="PacificAir",
                origin="LAX",
                destination="NRT",
                date="2026-03-22",
                depart_time="11:40",
                arrive_time="16:10",
                base_fare_usd=790,
                seats_left=5,
            ),
            "FL-3001": Flight(
                flight_id="FL-3001",
                airline="EuroConnect",
                origin="LON",
                destination="CDG",
                date="2026-03-18",
                depart_time="14:15",
                arrive_time="16:20",
                base_fare_usd=120,
                seats_left=9,
            ),
        }

        self._bookings: dict[str, BookingRecord] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        """Register flight-booking tools exposed to the answer model."""
        self._answer_core.register_tool(
            name="search_flights",
            description="Search available flights by origin, destination, and date.",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {
                        "type": "string",
                        "description": "Travel date in YYYY-MM-DD format.",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
            handler=self._tool_search_flights,
        )

        self._answer_core.register_tool(
            name="quote_fare",
            description="Get a fare quote for a flight and cabin class.",
            parameters={
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"},
                    "cabin_class": {
                        "type": "string",
                        "enum": ["economy", "premium_economy", "business"],
                    },
                },
                "required": ["flight_id", "cabin_class"],
            },
            handler=self._tool_quote_fare,
        )

        self._answer_core.register_tool(
            name="create_booking",
            description="Create a flight booking with passenger details.",
            parameters={
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"},
                    "passenger_name": {"type": "string"},
                    "cabin_class": {
                        "type": "string",
                        "enum": ["economy", "premium_economy", "business"],
                    },
                },
                "required": ["flight_id", "passenger_name", "cabin_class"],
            },
            handler=self._tool_create_booking,
        )

        self._answer_core.register_tool(
            name="get_booking",
            description="Retrieve details for an existing booking.",
            parameters={
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string"},
                },
                "required": ["booking_id"],
            },
            handler=self._tool_get_booking,
        )

        self._answer_core.register_tool(
            name="cancel_booking",
            description="Cancel an existing booking by booking ID.",
            parameters={
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string"},
                },
                "required": ["booking_id"],
            },
            handler=self._tool_cancel_booking,
        )

    def ask(self, user_input: str) -> str:
        """Handle one user message with domain-guarded booking support."""
        if not self._is_flight_related(user_input):
            return (
                "I can only help with flight booking and flight-travel questions. "
                "Please share your route, travel date, and passenger details."
            )
        return self._answer_core.ask(user_input)

    def _is_flight_related(self, text: str) -> bool:
        """Classify whether input is flight related using the reasoning model."""
        classification_prompt = (
            "Classify this user request. Return exactly FLIGHT_RELATED or "
            f"NOT_FLIGHT_RELATED.\n\nUser request:\n{text}"
        )
        try:
            decision = self._reasoning_core.ask(classification_prompt).strip().upper()
        except Exception:
            return False
        return decision == "FLIGHT_RELATED"

    def _tool_search_flights(self, origin: str, destination: str, date: str) -> dict[str, Any]:
        """Return available flights for route/date with remaining seats."""
        origin_code = origin.strip().upper()
        destination_code = destination.strip().upper()
        self._validate_date(date)

        matches = [
            flight
            for flight in self._flights.values()
            if flight.origin == origin_code
            and flight.destination == destination_code
            and flight.date == date
            and flight.seats_left > 0
        ]

        return {
            "origin": origin_code,
            "destination": destination_code,
            "date": date,
            "count": len(matches),
            "flights": [
                {
                    "flight_id": flight.flight_id,
                    "airline": flight.airline,
                    "depart_time": flight.depart_time,
                    "arrive_time": flight.arrive_time,
                    "base_fare_usd": flight.base_fare_usd,
                    "seats_left": flight.seats_left,
                }
                for flight in matches
            ],
        }

    def _tool_quote_fare(self, flight_id: str, cabin_class: str) -> dict[str, Any]:
        """Compute fare for a selected flight and cabin class."""
        flight = self._require_flight(flight_id)
        multiplier = self._fare_multiplier(cabin_class)
        final_fare = int(round(flight.base_fare_usd * multiplier))

        return {
            "flight_id": flight.flight_id,
            "cabin_class": cabin_class,
            "base_fare_usd": flight.base_fare_usd,
            "final_fare_usd": final_fare,
        }

    def _tool_create_booking(
        self,
        flight_id: str,
        passenger_name: str,
        cabin_class: str,
    ) -> dict[str, Any]:
        """Create and store a booking record if seats are available."""
        flight = self._require_flight(flight_id)
        if flight.seats_left <= 0:
            return {"status": "failed", "reason": "No seats left for this flight."}

        fare_quote = self._tool_quote_fare(flight_id=flight.flight_id, cabin_class=cabin_class)
        booking_id = f"BK-{uuid.uuid4().hex[:8].upper()}"

        self._bookings[booking_id] = {
            "booking_id": booking_id,
            "status": "confirmed",
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "passenger_name": passenger_name.strip(),
            "flight_id": flight.flight_id,
            "airline": flight.airline,
            "origin": flight.origin,
            "destination": flight.destination,
            "date": flight.date,
            "depart_time": flight.depart_time,
            "arrive_time": flight.arrive_time,
            "cabin_class": cabin_class,
            "paid_fare_usd": fare_quote["final_fare_usd"],
        }

        self._flights[flight.flight_id] = Flight(
            flight_id=flight.flight_id,
            airline=flight.airline,
            origin=flight.origin,
            destination=flight.destination,
            date=flight.date,
            depart_time=flight.depart_time,
            arrive_time=flight.arrive_time,
            base_fare_usd=flight.base_fare_usd,
            seats_left=max(0, flight.seats_left - 1),
        )

        return self._bookings[booking_id]

    def _tool_get_booking(self, booking_id: str) -> dict[str, Any]:
        """Return booking details by booking ID."""
        booking_key = booking_id.strip().upper()
        booking = self._bookings.get(booking_key)
        if booking is None:
            return {
                "status": "not_found",
                "message": f"No booking found for ID {booking_key}.",
            }
        return booking

    def _tool_cancel_booking(self, booking_id: str) -> dict[str, Any]:
        """Cancel booking and release one seat back to the inventory."""
        booking_key = booking_id.strip().upper()
        booking = self._bookings.get(booking_key)
        if booking is None:
            return {
                "status": "not_found",
                "message": f"No booking found for ID {booking_key}.",
            }

        if booking["status"] == "cancelled":
            return booking

        booking["status"] = "cancelled"
        booking["cancelled_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        flight = self._flights.get(booking["flight_id"])
        if flight is not None:
            self._flights[flight.flight_id] = Flight(
                flight_id=flight.flight_id,
                airline=flight.airline,
                origin=flight.origin,
                destination=flight.destination,
                date=flight.date,
                depart_time=flight.depart_time,
                arrive_time=flight.arrive_time,
                base_fare_usd=flight.base_fare_usd,
                seats_left=flight.seats_left + 1,
            )

        return booking

    def _require_flight(self, flight_id: str) -> Flight:
        """Lookup a flight by ID or raise a validation error."""
        key = flight_id.strip().upper()
        flight = self._flights.get(key)
        if flight is None:
            raise ValueError(f"Unknown flight_id: {key}")
        return flight

    def _fare_multiplier(self, cabin_class: str) -> float:
        """Return fare multiplier for a supported cabin class."""
        normalized = cabin_class.strip().lower()
        multipliers = {
            "economy": 1.0,
            "premium_economy": 1.35,
            "business": 2.2,
        }
        if normalized not in multipliers:
            raise ValueError(
                "Invalid cabin_class. Expected one of: economy, premium_economy, business."
            )
        return multipliers[normalized]

    def _validate_date(self, date_value: str) -> None:
        """Validate travel date format (YYYY-MM-DD)."""
        try:
            datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must be in YYYY-MM-DD format.") from exc


def run_cli() -> None:
    """Run a simple terminal chat loop for the booking agent."""
    print("Flight Ticket Booking Agent (type 'exit' to quit)")
    agent = FlightTicketBookingAgent()

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break
        if not user_text:
            continue

        answer = agent.ask(user_text)
        print(f"Agent: {answer}")


if __name__ == "__main__":
    run_cli()
