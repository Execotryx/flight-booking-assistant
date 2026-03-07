"""Flight ticket booking agent built on top of OllamaAICore with tool calling."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
import re
from typing import Callable, Literal, TypedDict, cast

from ai_config import AIConfig
from ollama_core import OllamaAICore

REASONING_MODEL_NAME: str = "lfm2.5-thinking:1.2b-q8_0"
ANSWER_MODEL_NAME: str = "granite4:3b"

FLIGHT_KEYWORDS: frozenset[str] = frozenset(
    {
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
)

CITY_TO_CODE: dict[str, str] = {
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

MONTH_NAME_TO_NUMBER: dict[str, int] = {
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

WORD_TO_NUMBER: dict[str, int] = {
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

DAY_OF_MONTH_PATTERN = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([a-z]+)(?:\s+(\d{4}))?"
)
MONTH_DAY_PATTERN = re.compile(
    r"([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?"
)
RELATIVE_DAYS_PATTERN = re.compile(r"(\d+)\s+days?\s+from\s+today")
WORD_RELATIVE_DAYS_PATTERN = re.compile(r"([a-z]+)\s+days?\s+from\s+today")

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
- If `search_flights` returns no results, use `next_available_dates` from that tool or call
    `list_available_flights` to propose concrete alternatives instead of repeating the same failed
    route/date combination.
- If user asks "later date", "currently", "what flights are available", or similar, use
    `list_available_flights` with known constraints from context.
- Never invent available flights. Always use the provided tools to search, quote, create, cancel,
  or inspect bookings.
- For prices and IDs, trust and cite tool outputs.
- If a tool fails or returns no result, explain plainly and offer next booking-related step.
- Never claim a booking is confirmed unless the booking tool returns a booking_id.
""".strip()

class BookingRecord(TypedDict):
    """Stored booking payload for confirmed/cancelled reservations."""

    booking_id: str
    status: Literal["confirmed", "cancelled"]
    created_at: str
    cancelled_at: str | None
    passenger_name: str
    flight_id: str
    airline: str
    origin: str
    destination: str
    date: str
    depart_time: str
    arrive_time: str
    cabin_class: str
    paid_fare_usd: int


@dataclass(frozen=True, slots=True)
class ToolTemplate:
    """Immutable tool registration template without bound instance handler."""

    name: str
    description: str
    parameters: dict[str, object]
    handler_name: str


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Resolved tool registration payload with bound instance handler."""

    name: str
    description: str
    parameters: dict[str, object]
    handler: Callable[..., object]


class FlightToolResult(TypedDict):
    """Serialized flight row returned by search tool."""

    flight_id: str
    airline: str
    depart_time: str
    arrive_time: str
    base_fare_usd: int
    seats_left: int


class SearchFlightsToolResult(TypedDict):
    """Response payload for flight search tool."""

    origin: str
    destination: str
    date: str
    count: int
    flights: list[FlightToolResult]
    next_available_dates: list[str]


class ListAvailableFlightsToolResult(TypedDict):
    """Response payload for broad availability discovery tool."""

    origin: str | None
    destination: str | None
    earliest_date: str | None
    count: int
    flights: list[FlightToolResult]


class ResolveDateToolResult(TypedDict):
    """Response payload for natural-language date resolution tool."""

    input: str
    resolved_date: str
    format: str


class CurrentSystemDateToolResult(TypedDict):
    """Response payload for current system date tool."""

    current_date: str
    format: str
    timezone: str


class FareQuoteToolResult(TypedDict):
    """Response payload for fare quote tool."""

    flight_id: str
    cabin_class: str
    base_fare_usd: int
    final_fare_usd: int


class FailedBookingToolResult(TypedDict):
    """Failure payload when a booking cannot be created."""

    status: Literal["failed"]
    reason: str


class BookingNotFoundToolResult(TypedDict):
    """Failure payload when booking lookup misses."""

    status: Literal["not_found"]
    message: str


class SearchContext(TypedDict):
    """Track the most recent normalized search arguments for follow-up intents."""

    origin: str
    destination: str
    date: str


TOOL_TEMPLATES: tuple[ToolTemplate, ...] = (
    ToolTemplate(
        name="get_current_system_date",
        description="Get the current system date in YYYY-MM-DD format.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler_name="_tool_get_current_system_date",
    ),
    ToolTemplate(
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
        handler_name="_tool_resolve_travel_date",
    ),
    ToolTemplate(
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
        handler_name="_tool_search_flights",
    ),
    ToolTemplate(
        name="list_available_flights",
        description=(
            "List available flights with optional filters. Use this when user asks for "
            "later dates, currently available options, or alternatives after no results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "earliest_date": {
                    "type": "string",
                    "description": "Optional lower date bound in YYYY-MM-DD.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of flights to return (default 10, max 25).",
                },
            },
            "required": [],
        },
        handler_name="_tool_list_available_flights",
    ),
    ToolTemplate(
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
        handler_name="_tool_quote_fare",
    ),
    ToolTemplate(
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
        handler_name="_tool_create_booking",
    ),
    ToolTemplate(
        name="get_booking",
        description="Retrieve details for an existing booking.",
        parameters={
            "type": "object",
            "properties": {
                "booking_id": {"type": "string"},
            },
            "required": ["booking_id"],
        },
        handler_name="_tool_get_booking",
    ),
    ToolTemplate(
        name="cancel_booking",
        description="Cancel an existing booking by booking ID.",
        parameters={
            "type": "object",
            "properties": {
                "booking_id": {"type": "string"},
            },
            "required": ["booking_id"],
        },
        handler_name="_tool_cancel_booking",
    ),
)


@dataclass(frozen=True, slots=True)
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
        self._reasoning_core = self._create_reasoning_core(config)
        self._answer_core = self._create_answer_core(config)

        self._flights: dict[str, Flight] = self._build_initial_flights()
        self._flight_ids_by_route_date: dict[tuple[str, str, str], list[str]] = (
            self._build_flight_route_date_index(self._flights)
        )

        self._bookings: dict[str, BookingRecord] = {}
        self._booking_flow_active: bool = False
        self._last_search_context: SearchContext | None = None
        self._register_tools()

    def _build_flight_route_date_index(
        self,
        flights: dict[str, Flight],
    ) -> dict[tuple[str, str, str], list[str]]:
        """Build route/date index to accelerate flight searches.

        Parameters:
            flights: Inventory keyed by flight id.
        """
        index: dict[tuple[str, str, str], list[str]] = {}
        for flight in flights.values():
            route_key = (flight.origin, flight.destination, flight.date)
            index.setdefault(route_key, []).append(flight.flight_id)
        return index

    def _create_reasoning_core(self, config: AIConfig | None) -> OllamaAICore:
        """Create the low-cost classifier core used for routing decisions.

        Parameters:
            config: Optional shared runtime configuration.
        """
        return OllamaAICore(
            config=config,
            system_behavior=REASONING_SYSTEM_PROMPT,
            model_name=REASONING_MODEL_NAME,
            temperature=0.0,
            max_tokens=8,
        )

    def _create_answer_core(self, config: AIConfig | None) -> OllamaAICore:
        """Create the main answer core that performs booking interactions.

        Parameters:
            config: Optional shared runtime configuration.
        """
        return OllamaAICore(
            config=config,
            system_behavior=BOOKING_SYSTEM_PROMPT,
            model_name=ANSWER_MODEL_NAME,
        )

    def _build_initial_flights(self) -> dict[str, Flight]:
        """Build the seeded in-memory flight inventory."""
        seeded_flights: list[Flight] = [
            Flight(
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
            Flight(
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
            Flight(
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
            Flight(
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
        ]
        return {flight.flight_id: flight for flight in seeded_flights}

    def _register_tools(self) -> None:
        """Register flight-booking tools exposed to the answer model."""
        for tool_definition in self._tool_definitions():
            self._register_single_tool(tool_definition)

    def _tool_definitions(self) -> tuple[ToolDefinition, ...]:
        """Return resolved tool definitions with handlers bound for this instance."""
        resolved_definitions: list[ToolDefinition] = []
        for template in TOOL_TEMPLATES:
            handler = cast(Callable[..., object], getattr(self, template.handler_name))
            resolved_definitions.append(
                ToolDefinition(
                    name=template.name,
                    description=template.description,
                    parameters=template.parameters,
                    handler=handler,
                )
            )
        return tuple(resolved_definitions)

    def _register_single_tool(self, tool_definition: ToolDefinition) -> None:
        """Register one tool definition into the answer core.

        Parameters:
            tool_definition: Mapping with name, description, parameters, and handler.
        """
        self._answer_core.register_tool(
            name=tool_definition.name,
            description=tool_definition.description,
            parameters=tool_definition.parameters,
            handler=tool_definition.handler,
        )

    def ask(self, user_input: str) -> str:
        """Handle one user message with domain-guarded booking support.

        Parameters:
            user_input: Raw text entered by the user for the current turn.
        """
        is_flight_related = self._is_flight_related(user_input)
        if not self._should_allow_request(is_flight_related):
            self._debug_log("routing=NOT_FLIGHT_RELATED (blocked)")
            return (
                "I can only help with flight booking and flight-travel questions. "
                "Please share your route, travel date, and passenger details."
            )

        self._update_booking_flow_state(is_flight_related)
        if not is_flight_related:
            self._debug_log("routing=FOLLOW_UP_IN_ACTIVE_BOOKING_FLOW (allowed)")

        deterministic_follow_up = self._maybe_handle_availability_follow_up(user_input)
        if deterministic_follow_up is not None:
            self._debug_log("routing=DETERMINISTIC_FOLLOW_UP_HANDLER")
            return deterministic_follow_up

        self._debug_log("routing=FLIGHT_RELATED (allowed)")
        answer: str = self._answer_core.ask(user_input)
        self._debug_log(f"answer={self._shorten_for_debug(answer)}")
        return answer

    def _maybe_handle_availability_follow_up(self, user_input: str) -> str | None:
        """Handle common availability follow-ups without another model tool loop.

        Parameters:
            user_input: Raw user text for the current turn.
        """
        lowered = user_input.lower()
        if not self._booking_flow_active:
            return None

        if "later date" in lowered or "later" in lowered:
            return self._format_later_date_follow_up()

        if "available currently" in lowered or "what flights from" in lowered:
            origin = self._extract_origin_hint(lowered)
            return self._format_current_availability_follow_up(origin)

        return None

    def _format_later_date_follow_up(self) -> str:
        """Return concrete later-date options based on last search context."""
        if self._last_search_context is None:
            return (
                "Please share origin, destination, and date so I can suggest later options."
            )

        availability = self._tool_list_available_flights(
            origin=self._last_search_context["origin"],
            destination=self._last_search_context["destination"],
            earliest_date=self._last_search_context["date"],
            limit=5,
        )
        return self._format_availability_response(
            availability=availability,
            empty_message=(
                "I couldn't find later flights for that same route right now. "
                "Would you like a different route or a broader date range?"
            ),
        )

    def _format_current_availability_follow_up(self, origin: str | None) -> str:
        """Return currently available flights with an optional origin filter.

        Parameters:
            origin: Optional origin city/code hint extracted from user text.
        """
        availability = self._tool_list_available_flights(
            origin=origin,
            destination=None,
            earliest_date=None,
            limit=8,
        )
        if origin is None and self._last_search_context is not None:
            availability = self._tool_list_available_flights(
                origin=self._last_search_context["origin"],
                destination=None,
                earliest_date=None,
                limit=8,
            )

        return self._format_availability_response(
            availability=availability,
            empty_message=(
                "I couldn't find currently available flights for that origin. "
                "Try another origin or a specific route/date."
            ),
        )

    def _extract_origin_hint(self, lowered_text: str) -> str | None:
        """Extract likely origin hint from user follow-up message.

        Parameters:
            lowered_text: Lower-cased user message.
        """
        for city_name in CITY_TO_CODE:
            if city_name in lowered_text:
                return city_name
        return None

    def _format_availability_response(
        self,
        availability: ListAvailableFlightsToolResult,
        empty_message: str,
    ) -> str:
        """Format list-availability payload into concise user-facing text.

        Parameters:
            availability: Availability payload from list tool.
            empty_message: Message used when no flights are available.
        """
        flights = availability["flights"]
        if not flights:
            return empty_message

        lines = ["Here are available flights:"]
        for flight in flights:
            lines.append(
                (
                    f"- {flight['flight_id']} ({flight['airline']}): "
                    f"{flight['depart_time']}-{flight['arrive_time']}, "
                    f"fare ${flight['base_fare_usd']}, seats {flight['seats_left']}"
                )
            )
        return "\n".join(lines)

    def _should_allow_request(self, is_flight_related: bool) -> bool:
        """Return whether the current request should be accepted by routing guard.

        Parameters:
            is_flight_related: Classifier output for the user request.
        """
        return is_flight_related or self._booking_flow_active

    def _update_booking_flow_state(self, is_flight_related: bool) -> None:
        """Update booking-flow state after classifying the current request.

        Parameters:
            is_flight_related: Classifier output for the user request.
        """
        if is_flight_related:
            self._booking_flow_active = True

    def _is_flight_related(self, text: str) -> bool:
        """Classify whether input is flight related using the reasoning model.

        Parameters:
            text: User message to classify for routing.
        """
        classification_prompt = self._build_classification_prompt(text)
        if (
            self._answer_core.config.debug_enabled
            and self._answer_core.config.debug_include_prompts
        ):
            self._debug_log(f"reasoning.prompt={classification_prompt}")
        try:
            decision = self._get_reasoning_decision(classification_prompt)
        except Exception:
            self._debug_log("reasoning.error=classification call failed")
            return self._keyword_fallback_is_flight_related(text)

        return self._resolve_routing_decision(decision, text)

    def _build_classification_prompt(self, text: str) -> str:
        """Build a compact classifier prompt for flight-related routing.

        Parameters:
            text: User request to classify.
        """
        return (
            "Classify this user request. Return exactly FLIGHT_RELATED or "
            f"NOT_FLIGHT_RELATED.\n\nUser request:\n{text}"
        )

    def _get_reasoning_decision(self, classification_prompt: str) -> str:
        """Run the reasoning model classifier and normalize its decision token.

        Parameters:
            classification_prompt: Prompt text for the reasoning model.
        """
        decision_raw = self._reasoning_core.ask(classification_prompt).strip()
        decision = decision_raw.upper()
        self._debug_log(f"reasoning.decision_raw={decision}")
        return decision

    def _resolve_routing_decision(self, decision: str, text: str) -> bool:
        """Resolve final routing decision with lexical fallback safeguards.

        Parameters:
            decision: Uppercase classifier model output.
            text: Original user request.
        """
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
        if any(keyword in lowered for keyword in FLIGHT_KEYWORDS):
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

    def _tool_search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
    ) -> SearchFlightsToolResult:
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
        self._last_search_context = {
            "origin": origin_code,
            "destination": destination_code,
            "date": normalized_date,
        }

        matches = self._find_matching_flights(origin_code, destination_code, normalized_date)
        next_available_dates = self._find_next_available_dates(
            origin_code=origin_code,
            destination_code=destination_code,
            from_date=normalized_date,
            max_dates=5,
        )

        return {
            "origin": origin_code,
            "destination": destination_code,
            "date": normalized_date,
            "count": len(matches),
            "flights": [self._serialize_flight(flight) for flight in matches],
            "next_available_dates": next_available_dates,
        }

    def _tool_list_available_flights(
        self,
        origin: str | None = None,
        destination: str | None = None,
        earliest_date: str | None = None,
        limit: int = 10,
    ) -> ListAvailableFlightsToolResult:
        """List available flights with optional route/date filters.

        Parameters:
            origin: Optional origin city/code filter.
            destination: Optional destination city/code filter.
            earliest_date: Optional lower date bound (natural text or YYYY-MM-DD).
            limit: Maximum number of returned flights.
        """
        origin_code = self._normalize_location_code(origin) if origin else None
        destination_code = self._normalize_location_code(destination) if destination else None

        normalized_earliest_date: str | None = None
        if earliest_date:
            normalized_earliest_date = self._normalize_date_value(earliest_date)
            self._validate_date(normalized_earliest_date)

        normalized_limit = max(1, min(int(limit), 25))

        candidates = [
            flight
            for flight in self._flights.values()
            if flight.seats_left > 0
            and (origin_code is None or flight.origin == origin_code)
            and (destination_code is None or flight.destination == destination_code)
            and (
                normalized_earliest_date is None
                or flight.date >= normalized_earliest_date
            )
        ]
        candidates.sort(key=lambda flight: (flight.date, flight.depart_time, flight.flight_id))

        return {
            "origin": origin_code,
            "destination": destination_code,
            "earliest_date": normalized_earliest_date,
            "count": len(candidates),
            "flights": [
                self._serialize_flight(flight)
                for flight in candidates[:normalized_limit]
            ],
        }

    def _find_matching_flights(
        self,
        origin_code: str,
        destination_code: str,
        normalized_date: str,
    ) -> list[Flight]:
        """Return flights that match route/date and still have available seats.

        Parameters:
            origin_code: Canonical origin code.
            destination_code: Canonical destination code.
            normalized_date: Canonical travel date in YYYY-MM-DD format.
        """
        route_key = (origin_code, destination_code, normalized_date)
        candidate_flight_ids = self._flight_ids_by_route_date.get(route_key, [])
        return [
            self._flights[flight_id]
            for flight_id in candidate_flight_ids
            if self._flights[flight_id].seats_left > 0
        ]

    def _find_next_available_dates(
        self,
        origin_code: str,
        destination_code: str,
        from_date: str,
        max_dates: int,
    ) -> list[str]:
        """Return upcoming available travel dates for a route after a given date.

        Parameters:
            origin_code: Route origin code.
            destination_code: Route destination code.
            from_date: Inclusive lower date bound (YYYY-MM-DD).
            max_dates: Maximum number of date values to return.
        """
        dates = {
            flight.date
            for flight in self._flights.values()
            if flight.origin == origin_code
            and flight.destination == destination_code
            and flight.seats_left > 0
            and flight.date >= from_date
        }
        return sorted(dates)[:max_dates]

    def _serialize_flight(self, flight: Flight) -> FlightToolResult:
        """Convert a Flight object into a tool-friendly response payload.

        Parameters:
            flight: Flight inventory entity to serialize.
        """
        return {
            "flight_id": flight.flight_id,
            "airline": flight.airline,
            "depart_time": flight.depart_time,
            "arrive_time": flight.arrive_time,
            "base_fare_usd": flight.base_fare_usd,
            "seats_left": flight.seats_left,
        }

    def _tool_resolve_travel_date(self, date_expression: str) -> ResolveDateToolResult:
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

    def _tool_get_current_system_date(self) -> CurrentSystemDateToolResult:
        """Return the current UTC system date in canonical format."""
        current_date = datetime.utcnow().date().isoformat()
        return {
            "current_date": current_date,
            "format": "YYYY-MM-DD",
            "timezone": "UTC",
        }

    def _tool_quote_fare(self, flight_id: str, cabin_class: str) -> FareQuoteToolResult:
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
    ) -> BookingRecord | FailedBookingToolResult:
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

        self._bookings[booking_id] = self._build_booking_record(
            booking_id=booking_id,
            flight=flight,
            passenger_name=passenger_name,
            cabin_class=cabin_class,
            paid_fare_usd=int(fare_quote["final_fare_usd"]),
        )
        self._set_flight_seat_count(flight, max(0, flight.seats_left - 1))

        return self._bookings[booking_id]

    def _build_booking_record(
        self,
        booking_id: str,
        flight: Flight,
        passenger_name: str,
        cabin_class: str,
        paid_fare_usd: int,
    ) -> BookingRecord:
        """Build a normalized booking record payload for persistence.

        Parameters:
            booking_id: New booking identifier.
            flight: Flight selected by the user.
            passenger_name: Passenger full name to store.
            cabin_class: Requested cabin class for the booking.
            paid_fare_usd: Final fare paid in USD.
        """
        return {
            "booking_id": booking_id,
            "status": "confirmed",
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "cancelled_at": None,
            "passenger_name": passenger_name.strip(),
            "flight_id": flight.flight_id,
            "airline": flight.airline,
            "origin": flight.origin,
            "destination": flight.destination,
            "date": flight.date,
            "depart_time": flight.depart_time,
            "arrive_time": flight.arrive_time,
            "cabin_class": cabin_class,
            "paid_fare_usd": paid_fare_usd,
        }

    def _set_flight_seat_count(self, flight: Flight, seats_left: int) -> None:
        """Persist a Flight object update with a new seat count.

        Parameters:
            flight: Existing flight entry to update.
            seats_left: New remaining seat count.
        """
        self._flights[flight.flight_id] = replace(flight, seats_left=seats_left)

    def _tool_get_booking(
        self,
        booking_id: str,
    ) -> BookingRecord | BookingNotFoundToolResult:
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

    def _tool_cancel_booking(
        self,
        booking_id: str,
    ) -> BookingRecord | BookingNotFoundToolResult:
        """Cancel booking and release one seat back to the inventory.

        Parameters:
            booking_id: Booking identifier to cancel.
        """
        booking_key, booking = self._lookup_booking(booking_id)
        if booking is None:
            return self._booking_not_found_response(booking_key)

        if booking["status"] == "cancelled":
            return booking

        booking["status"] = "cancelled"
        booking["cancelled_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        self._restore_seat_for_booking(booking)

        return booking

    def _lookup_booking(self, booking_id: str) -> tuple[str, BookingRecord | None]:
        """Return normalized booking key and stored booking if found.

        Parameters:
            booking_id: Booking identifier from user/tool input.
        """
        booking_key = booking_id.strip().upper()
        return booking_key, self._bookings.get(booking_key)

    def _booking_not_found_response(self, booking_key: str) -> BookingNotFoundToolResult:
        """Build a standardized booking-not-found response payload.

        Parameters:
            booking_key: Normalized booking identifier.
        """
        return {
            "status": "not_found",
            "message": f"No booking found for ID {booking_key}.",
        }

    def _restore_seat_for_booking(self, booking: BookingRecord) -> None:
        """Increment seat count back to the source flight for a cancelled booking.

        Parameters:
            booking: Stored booking record being cancelled.
        """
        flight = self._flights.get(str(booking["flight_id"]))
        if flight is None:
            return
        self._set_flight_seat_count(flight, flight.seats_left + 1)

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

        for resolver in (
            self._resolve_today_or_tomorrow,
            self._resolve_day_of_month_phrase,
            self._resolve_month_day_phrase,
            self._resolve_relative_days_phrase,
        ):
            resolved_date = resolver(normalized, today)
            if resolved_date is not None:
                return resolved_date

        return raw_date.strip()

    def _resolve_today_or_tomorrow(
        self,
        normalized: str,
        today: date,
    ) -> str | None:
        """Resolve canonical relative day aliases.

        Parameters:
            normalized: Lower-cased and trimmed date expression.
            today: Reference date in UTC.
        """
        if normalized == "today":
            return today.isoformat()
        if normalized == "tomorrow":
            return (today + timedelta(days=1)).isoformat()
        return None

    def _resolve_day_of_month_phrase(
        self,
        normalized: str,
        today: date,
    ) -> str | None:
        """Resolve phrases like '18th of March' or '18th of March 2026'.

        Parameters:
            normalized: Lower-cased and trimmed date expression.
            today: Reference date in UTC.
        """
        of_month_match = DAY_OF_MONTH_PATTERN.fullmatch(normalized)
        if of_month_match is None:
            return None

        day = int(of_month_match.group(1))
        month_name = of_month_match.group(2)
        year = int(of_month_match.group(3)) if of_month_match.group(3) else today.year
        return self._compose_iso_date(month_name, day, year)

    def _resolve_month_day_phrase(
        self,
        normalized: str,
        today: date,
    ) -> str | None:
        """Resolve phrases like 'March 18' or 'March 18 2026'.

        Parameters:
            normalized: Lower-cased and trimmed date expression.
            today: Reference date in UTC.
        """
        month_day_match = MONTH_DAY_PATTERN.fullmatch(normalized)
        if month_day_match is None:
            return None

        month_name = month_day_match.group(1)
        day = int(month_day_match.group(2))
        year = int(month_day_match.group(3)) if month_day_match.group(3) else today.year
        return self._compose_iso_date(month_name, day, year)

    def _resolve_relative_days_phrase(
        self,
        normalized: str,
        today: date,
    ) -> str | None:
        """Resolve phrases like '2 days from today' or 'two days from today'.

        Parameters:
            normalized: Lower-cased and trimmed date expression.
            today: Reference date in UTC.
        """
        relative_match = RELATIVE_DAYS_PATTERN.fullmatch(normalized)
        if relative_match is not None:
            return (today + timedelta(days=int(relative_match.group(1)))).isoformat()

        word_relative_match = WORD_RELATIVE_DAYS_PATTERN.fullmatch(normalized)
        if word_relative_match is None:
            return None

        offset_days = WORD_TO_NUMBER.get(word_relative_match.group(1))
        if offset_days is None:
            return None
        return (today + timedelta(days=offset_days)).isoformat()

    def _compose_iso_date(self, month_name: str, day: int, year: int) -> str | None:
        """Compose YYYY-MM-DD from month/day/year parts when month is recognized.

        Parameters:
            month_name: Lower-case month name.
            day: Day of month.
            year: 4-digit year.
        """
        month_number = MONTH_NAME_TO_NUMBER.get(month_name)
        if month_number is None:
            return None
        return datetime(year, month_number, day).date().isoformat()

    def _normalize_location_code(self, location_text: str) -> str:
        """Normalize location text into a route code used by the inventory.

        Parameters:
                location_text: User-entered city name or airport/metro code.
        """
        normalized = location_text.strip().lower()
        if normalized in CITY_TO_CODE:
            return CITY_TO_CODE[normalized]
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
