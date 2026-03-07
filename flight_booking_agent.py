"""Flight ticket booking agent built on top of OllamaAICore with tool calling."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Any

from ai_config import AIConfig
from ollama_core import OllamaAICore

REASONING_MODEL_NAME: str = "lfm2.5-thinking:1.2b-q8_0"
ANSWER_MODEL_NAME: str = "granite4:3b"

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
- If uncertain, return FLIGHT_RELATED when the request explicitly mentions booking or flights.
- Prefer returning exactly one token, but if you return extra text include one clear label.
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
- For non-ISO date phrases (for example, "two days from today"), first call
    `get_current_system_date` if needed for reference, then call `resolve_travel_date`, and
    finally use the resolved YYYY-MM-DD value in booking tools.
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
        """Initialize reasoning and answer models with in-memory booking state.

        Parameters:
            config: Optional shared runtime configuration used by both cores.
        """
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
        self._booking_flow_active: bool = False
        self._register_tools()

    def _register_tools(self) -> None:
        """Register flight-booking tools exposed to the answer model."""
        self._answer_core.register_tool(
            name="get_current_system_date",
            description="Get the current system date in YYYY-MM-DD format.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_get_current_system_date,
        )

        self._answer_core.register_tool(
            name="resolve_travel_date",
            description="Resolve natural-language date text to YYYY-MM-DD.",
            parameters={
                "type": "object",
                "properties": {
                    "date_expression": {
                        "type": "string",
                        "description": "Examples: 'today', 'tomorrow', 'two days from today'.",
                    },
                },
                "required": ["date_expression"],
            },
            handler=self._tool_resolve_travel_date,
        )

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
        """Handle one user message with domain-guarded booking support.

        Parameters:
            user_input: Raw text entered by the user for the current turn.
        """
        is_flight_related = self._is_flight_related(user_input)
        if not is_flight_related and not self._booking_flow_active:
            self._debug_log("routing=NOT_FLIGHT_RELATED (blocked)")
            return (
                "I can only help with flight booking and flight-travel questions. "
                "Please share your route, travel date, and passenger details."
            )

        if is_flight_related:
            self._booking_flow_active = True
        else:
            self._debug_log("routing=FOLLOW_UP_IN_ACTIVE_BOOKING_FLOW (allowed)")

        self._debug_log("routing=FLIGHT_RELATED (allowed)")
        answer: str = self._answer_core.ask(user_input)
        self._debug_log(f"answer={self._shorten_for_debug(answer)}")
        return answer

    def _is_flight_related(self, text: str) -> bool:
        """Classify whether input is flight related using the reasoning model.

        Parameters:
            text: User message to classify for routing.
        """
        classification_prompt = (
            "Classify this user request. Return exactly FLIGHT_RELATED or "
            f"NOT_FLIGHT_RELATED.\n\nUser request:\n{text}"
        )
        if self._answer_core.config.debug_enabled and self._answer_core.config.debug_include_prompts:
            self._debug_log(f"reasoning.prompt={classification_prompt}")
        try:
            decision_raw = self._reasoning_core.ask(classification_prompt).strip()
            decision = decision_raw.upper()
            self._debug_log(f"reasoning.decision_raw={decision}")
        except Exception:
            self._debug_log("reasoning.error=classification call failed")
            return self._keyword_fallback_is_flight_related(text)

        if "NOT_FLIGHT_RELATED" in decision:
            if self._keyword_fallback_is_flight_related(text):
                self._debug_log("reasoning.overridden_by_keyword_fallback=true")
                return True
            return False

        if "FLIGHT_RELATED" in decision:
            return True

        return self._keyword_fallback_is_flight_related(text)

    def _keyword_fallback_is_flight_related(self, text: str) -> bool:
        """Fallback lexical intent check for flight-booking requests.

        Parameters:
            text: User message to evaluate with keyword-based heuristics.
        """
        lowered = text.lower()
        flight_keywords = {
            "flight",
            "flights",
            "ticket",
            "tickets",
            "book",
            "booking",
            "airline",
            "airport",
            "itinerary",
            "departure",
            "arrival",
            "fare",
            "baggage",
            "check-in",
            "boarding",
            "one-way",
            "round-trip",
            "round trip",
            "reschedule",
            "cancel booking",
        }
        if any(keyword in lowered for keyword in flight_keywords):
            return True

        # Accept common route-code style messages, e.g., "NYC to LON on 2026-03-20".
        tokens = [token.strip(".,!?()[]{}") for token in text.split()]
        iata_like_tokens = [token for token in tokens if len(token) == 3 and token.isalpha()]
        return len(iata_like_tokens) >= 2

    def _debug_log(self, message: str) -> None:
        """Print booking-agent debug logs when debug mode is enabled.

        Parameters:
            message: Debug message text to emit.
        """
        if not self._answer_core.config.debug_enabled:
            return
        print(f"[BookingAgent DEBUG] {message}")

    def _shorten_for_debug(self, text: str, max_len: int = 240) -> str:
        """Return shortened debug-safe text for logs.

        Parameters:
            text: Source text that may need truncation.
            max_len: Maximum number of characters to keep before ellipsis.
        """
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _tool_search_flights(self, origin: str, destination: str, date: str) -> dict[str, Any]:
        """Return available flights for route/date with remaining seats.

        Parameters:
            origin: Origin code/city input that is normalized to uppercase.
            destination: Destination code/city input normalized to uppercase.
            date: Travel date expression or YYYY-MM-DD value.
        """
        origin_code = self._normalize_location_code(origin)
        destination_code = self._normalize_location_code(destination)
        normalized_date = self._normalize_date_value(date)
        self._validate_date(normalized_date)

        matches = [
            flight
            for flight in self._flights.values()
            if flight.origin == origin_code
            and flight.destination == destination_code
            and flight.date == normalized_date
            and flight.seats_left > 0
        ]

        return {
            "origin": origin_code,
            "destination": destination_code,
            "date": normalized_date,
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

    def _tool_resolve_travel_date(self, date_expression: str) -> dict[str, Any]:
        """Resolve a human date phrase into a canonical YYYY-MM-DD string.

        Parameters:
            date_expression: Natural-language date input to normalize.
        """
        normalized_date = self._normalize_date_value(date_expression)
        self._validate_date(normalized_date)
        return {
            "input": date_expression,
            "resolved_date": normalized_date,
            "format": "YYYY-MM-DD",
        }

    def _tool_get_current_system_date(self) -> dict[str, str]:
        """Return the current UTC system date in canonical format."""
        current_date = datetime.utcnow().date().isoformat()
        return {
            "current_date": current_date,
            "format": "YYYY-MM-DD",
            "timezone": "UTC",
        }

    def _tool_quote_fare(self, flight_id: str, cabin_class: str) -> dict[str, Any]:
        """Compute fare for a selected flight and cabin class.

        Parameters:
            flight_id: Identifier of the selected flight.
            cabin_class: Requested cabin class (economy, premium_economy, business).
        """
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
        """Create and store a booking record if seats are available.

        Parameters:
                flight_id: Identifier of the flight to book.
                passenger_name: Passenger full name to store on the booking.
                cabin_class: Requested cabin class (economy, premium_economy, business).
        """
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
        """Return booking details by booking ID.

        Parameters:
            booking_id: Booking identifier provided by the user.
        """
        booking_key = booking_id.strip().upper()
        booking = self._bookings.get(booking_key)
        if booking is None:
            return {
                "status": "not_found",
                "message": f"No booking found for ID {booking_key}.",
            }
        return booking

    def _tool_cancel_booking(self, booking_id: str) -> dict[str, Any]:
        """Cancel booking and release one seat back to the inventory.

        Parameters:
            booking_id: Booking identifier to cancel.
        """
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
        """Lookup a flight by ID or raise a validation error.

        Parameters:
            flight_id: Candidate flight identifier to validate and retrieve.
        """
        key = flight_id.strip().upper()
        flight = self._flights.get(key)
        if flight is None:
            raise ValueError(f"Unknown flight_id: {key}")
        return flight

    def _fare_multiplier(self, cabin_class: str) -> float:
        """Return fare multiplier for a supported cabin class.

        Parameters:
            cabin_class: Cabin class name used to choose a fare multiplier.
        """
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
        """Validate travel date format (YYYY-MM-DD).

        Parameters:
            date_value: Date string expected in ISO format.
        """
        try:
            datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must be in YYYY-MM-DD format.") from exc

    def _normalize_date_value(self, raw_date: str) -> str:
        """Normalize date inputs like 'today', 'tomorrow', or 'N days from today'.

        Parameters:
            raw_date: Raw user-provided date phrase.
        """
        normalized = raw_date.strip().lower()
        today = datetime.utcnow().date()

        if normalized == "today":
            return today.isoformat()
        if normalized == "tomorrow":
            return (today + timedelta(days=1)).isoformat()

        # Accept forms like "18th of March" and "18th of March 2026".
        month_name_to_number = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        of_month_match = re.fullmatch(
            r"(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([a-z]+)(?:\s+(\d{4}))?",
            normalized,
        )
        if of_month_match is not None:
            day = int(of_month_match.group(1))
            month_name = of_month_match.group(2)
            year = int(of_month_match.group(3)) if of_month_match.group(3) else today.year
            if month_name in month_name_to_number:
                return datetime(year, month_name_to_number[month_name], day).date().isoformat()

        # Accept forms like "March 18" and "March 18 2026".
        month_day_match = re.fullmatch(
            r"([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?",
            normalized,
        )
        if month_day_match is not None:
            month_name = month_day_match.group(1)
            day = int(month_day_match.group(2))
            year = int(month_day_match.group(3)) if month_day_match.group(3) else today.year
            if month_name in month_name_to_number:
                return datetime(year, month_name_to_number[month_name], day).date().isoformat()

        word_to_number = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }

        relative_match = re.fullmatch(r"(\d+)\s+days?\s+from\s+today", normalized)
        if relative_match is not None:
            offset_days = int(relative_match.group(1))
            return (today + timedelta(days=offset_days)).isoformat()

        word_relative_match = re.fullmatch(
            r"([a-z]+)\s+days?\s+from\s+today", normalized
        )
        if word_relative_match is not None:
            offset_word = word_relative_match.group(1)
            if offset_word in word_to_number:
                return (today + timedelta(days=word_to_number[offset_word])).isoformat()

        return raw_date.strip()

    def _normalize_location_code(self, location_text: str) -> str:
        """Normalize location text into a route code used by the inventory.

        Parameters:
                location_text: User-entered city name or airport/metro code.
        """
        normalized = location_text.strip().lower()
        city_to_code = {
            "new york": "NYC",
            "nyc": "NYC",
            "london": "LON",
            "lon": "LON",
            "paris": "CDG",
            "cdg": "CDG",
            "los angeles": "LAX",
            "lax": "LAX",
            "tokyo": "NRT",
            "nrt": "NRT",
        }
        if normalized in city_to_code:
            return city_to_code[normalized]
        return location_text.strip().upper()


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
