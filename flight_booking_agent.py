"""Flight ticket booking agent built on top of OllamaAICore with tool calling."""

from __future__ import annotations

import json
import base64
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from datetime import date, datetime, timedelta
from typing import Callable, Literal, NotRequired, Protocol, TypedDict, cast

from dotenv import dotenv_values
from ai_config import AIConfig
from ollama_core import OllamaAICore
from postcard_generator import PostcardGenerationResult, PostcardGenerator

#region Constants and configuration

REASONING_MODEL_NAME: str = "lfm2.5-thinking:1.2b-q8_0"
ANSWER_MODEL_NAME: str = REASONING_MODEL_NAME

DEFAULT_SEEDING_ENV_FILE: str = ".seeding-env"
DEFAULT_SUPABASE_FLIGHTS_TABLE: str = "flights"
DEFAULT_SUPABASE_LOOKUP_TABLE: str = "city_code_lookup"
DEFAULT_SUPABASE_BOOKINGS_TABLE: str = "bookings"
DEFAULT_SUPABASE_RESULT_CACHE_SIZE: int = 5
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
IATA_CODE_PATTERN = re.compile(r"\b[A-Za-z]{3}\b")
FLIGHT_ID_PATTERN = re.compile(r"\bFL-\d+\b", flags=re.IGNORECASE)
ORIGIN_HINT_PATTERN = re.compile(r"\bfrom\s+([a-z]{3,}(?:\s+[a-z]{2,})?)\b")
TRAILING_ORIGIN_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "are",
        "is",
        "available",
        "currently",
        "now",
        "today",
        "tomorrow",
        "flights",
    }
)

#endregion

#region Prompts

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
- Ask only for missing details before booking; do not ask for fields that are already known.
- If a valid `flight_id` is already known (or user says "this flight" right after results), do not ask
    again for origin/destination/date. Use `get_flight_by_id` when needed to confirm flight details.
- For `create_booking`, the required inputs are `flight_id`, `passenger_name`, and `cabin_class`.
    Collect only those missing inputs.
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

#endregion

#region Type definitions for structured data used in tools and Supabase communication

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
    postcard: NotRequired["PostcardToolResult"]


PostcardToolResult = PostcardGenerationResult


class PostcardGeneratorProtocol(Protocol):
    """Interface for postcard generation dependency injection."""

    def generate_postcard(
        self,
        booking_id: str,
        destination_city: str,
    ) -> PostcardGenerationResult:
        """Generate postcard artifacts for a confirmed booking."""
        ...


class FlightRow(TypedDict):
    """Flight row returned by Supabase."""

    flight_id: str
    airline: str
    origin: str
    destination: str
    date: str
    depart_time: str
    arrive_time: str
    base_fare_usd: int
    seats_left: int


class ToolTemplate(TypedDict):
    """Tool registration template without bound handler instance."""

    name: str
    description: str
    parameters: dict[str, object]
    handler_name: str


class ToolDefinition(TypedDict):
    """Resolved tool registration payload with bound handler."""

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


class FlightDetailsToolResult(TypedDict):
    """Response payload for looking up a flight by ID."""

    flight_id: str
    airline: str
    origin: str
    destination: str
    date: str
    depart_time: str
    arrive_time: str
    base_fare_usd: int
    seats_left: int


class FailedBookingToolResult(TypedDict):
    """Failure payload when a booking cannot be created."""

    status: Literal["failed"]
    reason: str


class BookingNotFoundToolResult(TypedDict):
    """Failure payload when booking lookup misses."""

    status: Literal["not_found"]
    message: str


class FlightNotFoundToolResult(TypedDict):
    """Failure payload when flight lookup by ID misses."""

    status: Literal["not_found"]
    message: str


class SearchContext(TypedDict):
    """Track the most recent normalized search arguments for follow-up intents."""

    origin: str
    destination: str
    date: str


class SupabaseSettings(TypedDict):
    """Resolved Supabase runtime settings for booking data access."""

    base_url: str
    api_key: str
    flights_table: str
    lookup_table: str
    bookings_table: str
    result_cache_size: int

#endregion

class SupabaseClient:
    """Encapsulate Supabase REST communication and transport-level diagnostics."""

    def __init__(
        self,
        settings: SupabaseSettings,
        debug_log: Callable[[str], None],
        shorten_for_debug: Callable[[str, int], str],
    ) -> None:
        """Create a Supabase client wrapper for HTTP calls and debug logging.

        Parameters:
            settings: Resolved Supabase endpoint/auth/table settings.
            debug_log: Callback used to emit debug log lines.
            shorten_for_debug: Helper used to truncate long debug payloads.
        """
        self._settings: SupabaseSettings = settings
        self._debug_log: Callable[[str], None] = debug_log
        self._shorten_for_debug: Callable[[str, int], str] = shorten_for_debug
        self._validate_supabase_ref_alignment(
            base_url=settings["base_url"],
            api_key=settings["api_key"],
        )

    def get(self, table: str, query: dict[str, str]) -> list[dict[str, object]]:
        """Run a Supabase GET request and normalize output to row dictionaries.

        Parameters:
            table: Target Supabase table name.
            query: PostgREST query parameters.
        """
        response = self.request(method="GET", table=table, query=query, body=None)
        if isinstance(response, list):
            return [cast(dict[str, object], row) for row in response]
        return []

    def post(
        self,
        table: str,
        body: object,
        query: dict[str, str] | None,
    ) -> object:
        """Run a Supabase POST request.

        Parameters:
            table: Target Supabase table name.
            body: JSON-serializable payload for insert/upsert operations.
            query: Optional PostgREST query parameters.
        """
        return self.request(method="POST", table=table, query=query, body=body)

    def patch(self, table: str, query: dict[str, str], body: dict[str, object]) -> object:
        """Run a Supabase PATCH request.

        Parameters:
            table: Target Supabase table name.
            query: PostgREST filters selecting rows to update.
            body: JSON patch payload with fields to update.
        """
        return self.request(method="PATCH", table=table, query=query, body=body)

    def request(
        self,
        method: str,
        table: str,
        query: dict[str, str] | None,
        body: object | None,
    ) -> object:
        """Execute one low-level Supabase REST request.

        Parameters:
            method: HTTP method, for example GET/POST/PATCH.
            table: Target Supabase table name.
            query: Optional PostgREST query string parameters.
            body: Optional JSON-serializable request payload.
        """
        query_string = ""
        if query:
            query_string = "?" + urllib.parse.urlencode(query)

        url = (
            f"{self._settings['base_url']}/rest/v1/"
            f"{urllib.parse.quote(table)}{query_string}"
        )

        payload: bytes | None = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")

        request = urllib.request.Request(url=url, data=payload, method=method)
        request.add_header("apikey", self._settings["api_key"])
        request.add_header("Authorization", f"Bearer {self._settings['api_key']}")
        request.add_header("Content-Type", "application/json")
        request.add_header("Prefer", "return=representation")

        body_kind = "none" if body is None else type(body).__name__
        self._debug_log(
            f"supabase.request method={method} table={table} url={url} body={body_kind}"
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw_payload = response.read().decode("utf-8")
                if not raw_payload:
                    self._debug_log(
                        f"supabase.response method={method} table={table} status={response.status} rows=0"
                    )
                    return []
                decoded = json.loads(raw_payload)
                row_count = len(decoded) if isinstance(decoded, list) else 1
                self._debug_log(
                    f"supabase.response method={method} table={table} status={response.status} rows={row_count}"
                )
                return cast(object, decoded)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            self._debug_log(
                "supabase.http_error "
                f"method={method} table={table} status={exc.code} "
                f"body={self._shorten_for_debug(error_body, 240)}"
            )
            raise RuntimeError(
                f"Supabase {method} failed ({exc.code}) on table '{table}': {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.gaierror):
                parsed = urllib.parse.urlparse(url)
                host = parsed.hostname or "<unknown-host>"
                self._debug_log(
                    f"supabase.dns_error method={method} table={table} host={host} reason={reason}"
                )
                raise RuntimeError(
                    "Supabase hostname lookup failed. "
                    f"Could not resolve '{host}'. Check SUPABASE_URL and DNS connectivity."
                ) from exc
            self._debug_log(
                f"supabase.network_error method={method} table={table} reason={exc}"
            )
            raise RuntimeError(
                f"Supabase {method} network error on table '{table}': {exc}"
            ) from exc

    def _validate_supabase_ref_alignment(self, base_url: str, api_key: str) -> None:
        """Validate Supabase URL host matches the project ref from the JWT key.

        Parameters:
            base_url: Supabase project base URL.
            api_key: Supabase API key whose JWT payload may contain the `ref` claim.
        """
        parsed = urllib.parse.urlparse(base_url)
        host = parsed.hostname or ""
        expected_ref = self._extract_ref_from_jwt(api_key)
        if not expected_ref:
            return
        if host.startswith(f"{expected_ref}."):
            return
        raise ValueError(
            "SUPABASE_URL project ref does not match SUPABASE_KEY ref. "
            f"URL host='{host}', key ref='{expected_ref}'."
        )

    def _extract_ref_from_jwt(self, token: str) -> str | None:
        """Decode JWT payload and return the `ref` claim when available.

        Parameters:
            token: JWT token string (Supabase API key).
        """
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        try:
            decoded = base64.urlsafe_b64decode(payload)
            payload_obj = json.loads(decoded)
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
        ref_value = payload_obj.get("ref")
        if isinstance(ref_value, str) and ref_value:
            return ref_value
        return None


#region Tool templates with handler name references for dynamic binding during registration

TOOL_TEMPLATES: tuple[ToolTemplate, ...] = (
    {
        "name": "get_current_system_date",
        "description": "Get the current system date in YYYY-MM-DD format.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "handler_name": "_tool_get_current_system_date",
    },
    {
        "name": "resolve_travel_date",
        "description": "Resolve natural-language date text to YYYY-MM-DD.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_expression": {
                    "type": "string",
                    "description": "Examples: 'today', 'tomorrow', 'two days from today'.",
                }
            },
            "required": ["date_expression"],
        },
        "handler_name": "_tool_resolve_travel_date",
    },
    {
        "name": "search_flights",
        "description": "Search available flights by origin, destination, and date.",
        "parameters": {
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
        "handler_name": "_tool_search_flights",
    },
    {
        "name": "list_available_flights",
        "description": (
            "List available flights with optional filters. Use this when user asks for "
            "later dates, currently available options, or alternatives after no results."
        ),
        "parameters": {
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
        "handler_name": "_tool_list_available_flights",
    },
    {
        "name": "quote_fare",
        "description": "Get a fare quote for a flight and cabin class.",
        "parameters": {
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
        "handler_name": "_tool_quote_fare",
    },
    {
        "name": "get_flight_by_id",
        "description": "Retrieve a flight by flight_id only, including departure date/time.",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_id": {
                    "type": "string",
                    "description": "Flight identifier, for example FL-3001.",
                }
            },
            "required": ["flight_id"],
        },
        "handler_name": "_tool_get_flight_by_id",
    },
    {
        "name": "create_booking",
        "description": "Create a flight booking with passenger details.",
        "parameters": {
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
        "handler_name": "_tool_create_booking",
    },
    {
        "name": "generate_destination_postcard",
        "description": (
            "Generate a destination postcard image with DALL-E 3 only for an existing "
            "confirmed booking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "string",
                    "description": "Confirmed booking ID to validate generation eligibility.",
                },
                "destination_city": {
                    "type": "string",
                    "description": "Destination city name used to craft a postcard prompt.",
                },
            },
            "required": ["booking_id", "destination_city"],
        },
        "handler_name": "_tool_generate_destination_postcard",
    },
    {
        "name": "get_booking",
        "description": "Retrieve details for an existing booking.",
        "parameters": {
            "type": "object",
            "properties": {"booking_id": {"type": "string"}},
            "required": ["booking_id"],
        },
        "handler_name": "_tool_get_booking",
    },
    {
        "name": "cancel_booking",
        "description": "Cancel an existing booking by booking ID.",
        "parameters": {
            "type": "object",
            "properties": {"booking_id": {"type": "string"}},
            "required": ["booking_id"],
        },
        "handler_name": "_tool_cancel_booking",
    },
)

#endregion

class FlightTicketBookingAgent:
    """Domain-restricted booking assistant powered by OllamaAICore."""

    #region Lifecycle and tool registration methods

    def __init__(
        self,
        config: AIConfig | None = None,
        postcard_generator: PostcardGeneratorProtocol | None = None,
    ) -> None:
        """Initialize agent models, Supabase client, and in-memory state.

        Parameters:
            config: Optional runtime AI configuration used for both reasoning and
                answering cores. When omitted, default AIConfig behavior is used.
            postcard_generator: Optional postcard generator dependency. When omitted,
                a default `PostcardGenerator` is constructed.
        """
        self._reasoning_core = self._create_reasoning_core(config)
        self._answer_core = self._create_answer_core(config)

        self._supabase = self._load_supabase_settings()
        self._supabase_client = SupabaseClient(
            settings=self._supabase,
            debug_log=self._debug_log,
            shorten_for_debug=self._shorten_for_debug,
        )
        self._result_cache: OrderedDict[str, object] = OrderedDict()
        self._result_cache_max_size: int = self._supabase["result_cache_size"]

        self._booking_flow_active: bool = False
        self._last_search_context: SearchContext | None = None
        self._last_presented_flight_ids: list[str] = []
        self._latest_postcard: PostcardToolResult | None = None
        if postcard_generator is not None:
            self._postcard_generator = postcard_generator
        else:
            self._postcard_generator = PostcardGenerator(
                reasoning_model_name=REASONING_MODEL_NAME,
                reasoning_base_url=self._reasoning_core.config.base_url,
                reasoning_api_key=self._reasoning_core.config.openai_api_key,
                debug_log=self._debug_log,
            )
        self._register_tools()

    def _create_reasoning_core(self, config: AIConfig | None) -> OllamaAICore:
        """Create the low-temperature routing/classification model wrapper.

        Parameters:
            config: Optional AI runtime configuration passed into OllamaAICore.
        """
        return OllamaAICore(
            config=config,
            system_behavior=REASONING_SYSTEM_PROMPT,
            model_name=REASONING_MODEL_NAME,
            temperature=0.0,
            max_tokens=8,
        )

    def _create_answer_core(self, config: AIConfig | None) -> OllamaAICore:
        """Create the answer-generation model wrapper with booking system prompt.

        Parameters:
            config: Optional AI runtime configuration passed into OllamaAICore.
        """
        return OllamaAICore(
            config=config,
            system_behavior=BOOKING_SYSTEM_PROMPT,
            model_name=ANSWER_MODEL_NAME,
        )

    def _load_supabase_settings(self) -> SupabaseSettings:
        """Resolve Supabase settings from environment and validate required values.

        The method reads values first from process environment variables, then
        falls back to `.seeding-env` for local development defaults.
        """
        seeding_file_values = dotenv_values(DEFAULT_SEEDING_ENV_FILE)

        base_url = str(
            os.getenv("SUPABASE_URL")
            or seeding_file_values.get("SUPABASE_URL")
            or ""
        ).rstrip("/")
        api_key = str(
            os.getenv("SUPABASE_KEY")
            or seeding_file_values.get("SUPABASE_KEY")
            or ""
        ).strip()

        flights_table = str(
            os.getenv("SUPABASE_TABLE")
            or seeding_file_values.get("SUPABASE_TABLE")
            or DEFAULT_SUPABASE_FLIGHTS_TABLE
        ).strip()
        lookup_table = str(
            os.getenv("SUPABASE_LOOKUP_TABLE")
            or seeding_file_values.get("SUPABASE_LOOKUP_TABLE")
            or DEFAULT_SUPABASE_LOOKUP_TABLE
        ).strip()
        bookings_table = str(
            os.getenv("SUPABASE_BOOKINGS_TABLE")
            or seeding_file_values.get("SUPABASE_BOOKINGS_TABLE")
            or DEFAULT_SUPABASE_BOOKINGS_TABLE
        ).strip()
        result_cache_size = self._parse_positive_int(
            value=(
                os.getenv("SUPABASE_RESULT_CACHE_SIZE")
                or seeding_file_values.get("SUPABASE_RESULT_CACHE_SIZE")
                or DEFAULT_SUPABASE_RESULT_CACHE_SIZE
            ),
            fallback=DEFAULT_SUPABASE_RESULT_CACHE_SIZE,
        )

        if not base_url.startswith("http"):
            raise ValueError("SUPABASE_URL is required and must be a valid URL.")
        if not api_key:
            raise ValueError("SUPABASE_KEY is required for booking tools.")

        return {
            "base_url": base_url,
            "api_key": api_key,
            "flights_table": flights_table,
            "lookup_table": lookup_table,
            "bookings_table": bookings_table,
            "result_cache_size": result_cache_size,
        }

    def _parse_positive_int(self, value: object, fallback: int) -> int:
        """Parse a positive integer with fallback for invalid or non-positive input.

        Parameters:
            value: Raw value to parse as an integer.
            fallback: Value returned when parsing fails or result is <= 0.
        """
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            return fallback
        return parsed if parsed > 0 else fallback

    def _register_tools(self) -> None:
        """Register all booking tools on the answer core from templates."""
        for tool_definition in self._tool_definitions():
            self._answer_core.register_tool(
                name=tool_definition["name"],
                description=tool_definition["description"],
                parameters=tool_definition["parameters"],
                handler=tool_definition["handler"],
            )

    def _tool_definitions(self) -> tuple[ToolDefinition, ...]:
        """Resolve tool templates into concrete tool definitions with bound handlers."""
        resolved: list[ToolDefinition] = []
        for template in TOOL_TEMPLATES:
            resolved.append(
                {
                    "name": template["name"],
                    "description": template["description"],
                    "parameters": template["parameters"],
                    "handler": cast(
                        Callable[..., object],
                        getattr(self, template["handler_name"]),
                    ),
                }
            )
        return tuple(resolved)

    #endregion

    #region Public conversation entrypoint methods

    def ask(self, user_input: str) -> str:
        """Handle one user turn and return the final assistant response.

        This method applies routing checks, deterministic follow-up handlers for
        availability intents, and tool-enabled LLM response generation.

        Parameters:
            user_input: Raw text entered by the end user.
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
        enriched_input = self._maybe_enrich_booking_input(user_input)
        answer = self._sanitize_answer_output(self._answer_core.ask(enriched_input))
        self._debug_log(f"answer={self._shorten_for_debug(answer)}")
        return answer

    def get_latest_postcard_path(self) -> str | None:
        """Return local image path for the most recently generated destination postcard."""
        if self._latest_postcard is None:
            return None
        return self._latest_postcard.get("image_path")

    def get_latest_postcard(self) -> PostcardToolResult | None:
        """Return the most recent postcard generation payload for UI integration."""
        if self._latest_postcard is None:
            return None
        return cast(PostcardToolResult, json.loads(json.dumps(self._latest_postcard)))

    #endregion

    #region Request routing and domain classification methods

    def _should_allow_request(self, is_flight_related: bool) -> bool:
        """Decide whether a request is allowed within current conversation state.

        Parameters:
            is_flight_related: Whether the current request was classified as
                flight-related.
        """
        return is_flight_related or self._booking_flow_active

    def _update_booking_flow_state(self, is_flight_related: bool) -> None:
        """Update internal booking-flow state after routing classification.

        Parameters:
            is_flight_related: Whether the current request is in domain scope.
        """
        if is_flight_related:
            self._booking_flow_active = True

    def _is_flight_related(self, text: str) -> bool:
        """Classify user text as flight-related using reasoning core plus fallback.

        Parameters:
            text: User-provided message to classify.
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
        """Build the strict routing prompt sent to the reasoning model.

        Parameters:
            text: User request text inserted into classifier prompt.
        """
        return (
            "Classify this user request. Return exactly FLIGHT_RELATED or "
            f"NOT_FLIGHT_RELATED.\n\nUser request:\n{text}"
        )

    def _get_reasoning_decision(self, classification_prompt: str) -> str:
        """Execute the routing classifier and normalize its decision label.

        Parameters:
            classification_prompt: Fully composed prompt for routing model.
        """
        decision_raw = self._reasoning_core.ask(classification_prompt).strip()
        decision = decision_raw.upper()
        self._debug_log(f"reasoning.decision_raw={decision}")
        return decision

    def _resolve_routing_decision(self, decision: str, text: str) -> bool:
        """Resolve classifier output into final allow/deny routing decision.

        Parameters:
            decision: Upper-cased classifier output from reasoning model.
            text: Original user text used by keyword fallback logic.
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
        """Infer flight relevance from keywords and IATA-like token heuristics.

        Parameters:
            text: User request to evaluate when classifier output is uncertain.
        """
        lowered = text.lower()
        if any(keyword in lowered for keyword in FLIGHT_KEYWORDS):
            return True

        tokens = [token.strip(".,!?()[]{}") for token in text.split()]
        iata_like_tokens = [token for token in tokens if len(token) == 3 and token.isalpha()]
        return len(iata_like_tokens) >= 2

    #endregion

    #region Deterministic follow-up and response formatting methods

    def _maybe_handle_availability_follow_up(self, user_input: str) -> str | None:
        """Handle specific follow-up intents without another model call.

        Parameters:
            user_input: Latest user message used to detect availability intents.
        """
        lowered = user_input.lower()
        if not self._booking_flow_active:
            return None

        if "later date" in lowered or "later" in lowered:
            return self._format_later_date_follow_up()

        if "available currently" in lowered or "what flights from" in lowered:
            origin = self._extract_origin_hint(user_input)
            return self._format_current_availability_follow_up(origin)

        return None

    def _format_later_date_follow_up(self) -> str:
        """Build a response with later-date options from the last search context."""
        if self._last_search_context is None:
            return "Please share origin, destination, and date so I can suggest later options."

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
        """Build a response with currently available flights.

        Parameters:
            origin: Optional origin hint extracted from user follow-up text.
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

    def _extract_origin_hint(self, user_text: str) -> str | None:
        """Extract an origin hint from user text using phrase and IATA patterns.

        Parameters:
            user_text: Raw user follow-up text.
        """
        lowered = user_text.lower()
        match = ORIGIN_HINT_PATTERN.search(lowered)
        if match is not None:
            return self._clean_origin_hint(match.group(1))

        iata_match = IATA_CODE_PATTERN.search(user_text)
        if iata_match is not None:
            return iata_match.group(0)

        return None

    def _clean_origin_hint(self, hint: str) -> str | None:
        """Trim trailing filler words from an extracted origin phrase.

        Parameters:
            hint: Raw origin phrase captured from user text.
        """
        tokens = [token for token in hint.strip().split() if token]
        while tokens and tokens[-1] in TRAILING_ORIGIN_FILLER_WORDS:
            tokens.pop()

        if not tokens:
            return None
        return " ".join(tokens)

    def _format_availability_response(
        self,
        availability: ListAvailableFlightsToolResult,
        empty_message: str,
    ) -> str:
        """Format list-availability tool output into user-facing bullet lines.

        Parameters:
            availability: Structured availability data from tool calls.
            empty_message: Fallback message returned when no flights are present.
        """
        flights = availability["flights"]
        if not flights:
            self._last_presented_flight_ids = []
            return empty_message

        self._last_presented_flight_ids = [
            str(flight["flight_id"]).strip().upper()
            for flight in flights
            if str(flight.get("flight_id", "")).strip()
        ]

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

    def _maybe_enrich_booking_input(self, user_input: str) -> str:
        """Attach a known flight_id when user refers to "this flight" in booking intent.

        This reduces ambiguous follow-ups where the user asks to book the most recently
        shown single option without repeating its identifier.
        """
        text = user_input.strip()
        lowered = text.lower()

        has_booking_intent = any(
            token in lowered
            for token in ("book", "booking", "reserve", "ticket")
        )
        if not has_booking_intent:
            return text

        if FLIGHT_ID_PATTERN.search(text):
            return text

        refers_to_previous_option = any(
            phrase in lowered
            for phrase in (
                "this flight",
                "that flight",
                "book it",
                "this one",
                "that one",
                "same flight",
            )
        )
        if not refers_to_previous_option:
            return text

        if len(self._last_presented_flight_ids) != 1:
            return text

        inferred_flight_id = self._last_presented_flight_ids[0]
        enriched = (
            f"{text}\n\n"
            f"Known flight_id from immediate prior options: {inferred_flight_id}."
        )
        self._debug_log(f"booking.input_enriched_with_flight_id={inferred_flight_id}")
        return enriched

    def _sanitize_answer_output(self, answer: str) -> str:
        """Normalize model output and remove accidental wrapper tags.

        Parameters:
            answer: Raw answer text returned by the answering model.
        """
        cleaned = answer.strip()
        response_match = re.search(
            r"<response>\s*(.*?)\s*</response>",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if response_match is not None:
            wrapped_content = response_match.group(1).strip()
            if wrapped_content:
                return wrapped_content

            cleaned = re.sub(
                r"</?response>",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

        return cleaned

    #endregion

    #region Tool handler methods for flight discovery, dates, and pricing

    def _tool_search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
    ) -> SearchFlightsToolResult:
        """Search flights for a route/date and include nearby date alternatives.

        Parameters:
            origin: Origin city or IATA code.
            destination: Destination city or IATA code.
            date: Requested travel date or natural-language date phrase.
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

        cache_key = (
            f"search:{origin_code}:{destination_code}:{normalized_date}"
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cast(SearchFlightsToolResult, cached)

        matches = self._fetch_matching_flights(origin_code, destination_code, normalized_date)
        next_available_dates = self._find_next_available_dates(
            origin_code=origin_code,
            destination_code=destination_code,
            from_date=normalized_date,
            max_dates=5,
        )

        result: SearchFlightsToolResult = {
            "origin": origin_code,
            "destination": destination_code,
            "date": normalized_date,
            "count": len(matches),
            "flights": [self._serialize_flight(flight) for flight in matches],
            "next_available_dates": next_available_dates,
        }
        self._cache_set(cache_key, result)
        return result

    def _tool_list_available_flights(
        self,
        origin: str | None = None,
        destination: str | None = None,
        earliest_date: str | None = None,
        limit: int = 10,
    ) -> ListAvailableFlightsToolResult:
        """List available flights with optional route/date filters.

        Parameters:
            origin: Optional origin city or IATA code filter.
            destination: Optional destination city or IATA code filter.
            earliest_date: Optional lower-bound date (or resolvable phrase).
            limit: Maximum number of rows to return, clamped to [1, 25].
        """
        origin_code = self._normalize_location_code(origin) if origin else None
        destination_code = self._normalize_location_code(destination) if destination else None

        normalized_earliest_date: str | None = None
        if earliest_date:
            normalized_earliest_date = self._normalize_date_value(earliest_date)
            self._validate_date(normalized_earliest_date)

        normalized_limit = max(1, min(int(limit), 25))

        cache_key = (
            f"list:{origin_code or '*'}:{destination_code or '*'}:"
            f"{normalized_earliest_date or '*'}:{normalized_limit}"
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cast(ListAvailableFlightsToolResult, cached)

        flights = self._fetch_available_flights(
            origin_code=origin_code,
            destination_code=destination_code,
            earliest_date=normalized_earliest_date,
            limit=normalized_limit,
        )

        result: ListAvailableFlightsToolResult = {
            "origin": origin_code,
            "destination": destination_code,
            "earliest_date": normalized_earliest_date,
            "count": len(flights),
            "flights": [self._serialize_flight(flight) for flight in flights],
        }
        self._cache_set(cache_key, result)
        return result

    def _tool_resolve_travel_date(self, date_expression: str) -> ResolveDateToolResult:
        """Resolve and validate a natural-language date expression.

        Parameters:
            date_expression: Human-friendly date expression such as "today".
        """
        normalized_date = self._normalize_date_value(date_expression)
        self._validate_date(normalized_date)
        return {
            "input": date_expression,
            "resolved_date": normalized_date,
            "format": "YYYY-MM-DD",
        }

    def _tool_get_current_system_date(self) -> CurrentSystemDateToolResult:
        """Return the current UTC date in a structured response payload."""
        current_date = datetime.utcnow().date().isoformat()
        return {
            "current_date": current_date,
            "format": "YYYY-MM-DD",
            "timezone": "UTC",
        }

    def _tool_quote_fare(self, flight_id: str, cabin_class: str) -> FareQuoteToolResult:
        """Quote final fare for a given flight and cabin class.

        Parameters:
            flight_id: Flight identifier to quote.
            cabin_class: Cabin class key (`economy`, `premium_economy`, `business`).
        """
        flight = self._require_flight(flight_id)
        multiplier = self._fare_multiplier(cabin_class)
        final_fare = int(round(int(flight["base_fare_usd"]) * multiplier))

        return {
            "flight_id": flight["flight_id"],
            "cabin_class": cabin_class,
            "base_fare_usd": int(flight["base_fare_usd"]),
            "final_fare_usd": final_fare,
        }

    #endregion

    #region Tool handler methods for booking lifecycle

    def _tool_get_flight_by_id(
        self,
        flight_id: str,
    ) -> FlightDetailsToolResult | FlightNotFoundToolResult:
        """Return flight details for a flight ID, or a not-found payload.

        Parameters:
            flight_id: Flight identifier supplied by the user.
        """
        key = flight_id.strip().upper()
        if not key:
            return {
                "status": "not_found",
                "message": "No flight found for empty flight ID.",
            }

        try:
            flight = self._require_flight(key)
        except ValueError:
            return {
                "status": "not_found",
                "message": f"No flight found for ID {key}.",
            }

        return {
            "flight_id": flight["flight_id"],
            "airline": flight["airline"],
            "origin": flight["origin"],
            "destination": flight["destination"],
            "date": flight["date"],
            "depart_time": flight["depart_time"],
            "arrive_time": flight["arrive_time"],
            "base_fare_usd": int(flight["base_fare_usd"]),
            "seats_left": int(flight["seats_left"]),
        }

    def _tool_create_booking(
        self,
        flight_id: str,
        passenger_name: str,
        cabin_class: str,
    ) -> BookingRecord | FailedBookingToolResult:
        """Create a confirmed booking and decrement available seats.

        Parameters:
            flight_id: Flight identifier to book.
            passenger_name: Passenger name saved in booking record.
            cabin_class: Cabin class key used for fare quoting.
        """
        flight = self._require_flight(flight_id)
        if int(flight["seats_left"]) <= 0:
            return {"status": "failed", "reason": "No seats left for this flight."}

        fare_quote = self._tool_quote_fare(flight_id=flight["flight_id"], cabin_class=cabin_class)
        booking_id = f"BK-{uuid.uuid4().hex[:8].upper()}"

        booking_record: BookingRecord = {
            "booking_id": booking_id,
            "status": "confirmed",
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "cancelled_at": None,
            "passenger_name": passenger_name.strip(),
            "flight_id": flight["flight_id"],
            "airline": flight["airline"],
            "origin": flight["origin"],
            "destination": flight["destination"],
            "date": flight["date"],
            "depart_time": flight["depart_time"],
            "arrive_time": flight["arrive_time"],
            "cabin_class": cabin_class,
            "paid_fare_usd": int(fare_quote["final_fare_usd"]),
        }

        self._rest_post(
            table=self._supabase["bookings_table"],
            body=[booking_record],
            query=None,
        )

        # Invalidate small read cache after a write to avoid stale seat/search responses.
        self._cache_clear()
        self._update_flight_seats(flight_id=flight["flight_id"], seats_left=max(0, int(flight["seats_left"]) - 1))

        postcard_destination = self._city_name_from_iata(flight["destination"])
        postcard_result = self._tool_generate_destination_postcard(
            booking_id=booking_id,
            destination_city=postcard_destination,
        )
        booking_record["postcard"] = postcard_result

        return booking_record

    def _tool_generate_destination_postcard(
        self,
        booking_id: str,
        destination_city: str,
    ) -> PostcardToolResult:
        """Generate a destination postcard image for a confirmed booking only.

        Parameters:
            booking_id: Confirmed booking identifier used as generation gate.
            destination_city: City used for prompt construction.
        """
        booking_lookup = self._tool_get_booking(booking_id)
        if isinstance(booking_lookup, dict) and booking_lookup.get("status") == "not_found":
            failure = self._postcard_failure(
                booking_id=booking_id,
                destination_city=destination_city,
                prompt="",
                message="Postcard generation is allowed only for confirmed bookings.",
            )
            self._latest_postcard = failure
            return failure

        booking = cast(BookingRecord, booking_lookup)
        if booking["status"] != "confirmed":
            failure = self._postcard_failure(
                booking_id=booking_id,
                destination_city=destination_city,
                prompt="",
                message="Booking is not confirmed. Postcard generation was skipped.",
            )
            self._latest_postcard = failure
            return failure

        postcard_result = self._postcard_generator.generate_postcard(
            booking_id=booking_id,
            destination_city=destination_city,
        )
        self._latest_postcard = postcard_result
        return postcard_result

    def _tool_get_booking(
        self,
        booking_id: str,
    ) -> BookingRecord | BookingNotFoundToolResult:
        """Retrieve one booking by ID with cache acceleration.

        Parameters:
            booking_id: Booking identifier, case-insensitive.
        """
        booking_key = booking_id.strip().upper()
        cache_key = f"booking:{booking_key}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cast(BookingRecord, cached)

        rows = self._rest_get(
            table=self._supabase["bookings_table"],
            query={
                "select": (
                    "booking_id,status,created_at,cancelled_at,passenger_name,flight_id,"
                    "airline,origin,destination,date,depart_time,arrive_time,cabin_class,paid_fare_usd"
                ),
                "booking_id": f"eq.{booking_key}",
                "limit": "1",
            },
        )
        if not rows:
            return {
                "status": "not_found",
                "message": f"No booking found for ID {booking_key}.",
            }

        booking = self._row_to_booking(rows[0])
        self._cache_set(cache_key, booking)
        return booking

    def _tool_cancel_booking(
        self,
        booking_id: str,
    ) -> BookingRecord | BookingNotFoundToolResult:
        """Cancel a booking and return one seat to the associated flight.

        Parameters:
            booking_id: Identifier of the booking to cancel.
        """
        booking_lookup = self._tool_get_booking(booking_id)
        if isinstance(booking_lookup, dict) and booking_lookup.get("status") == "not_found":
            return booking_lookup

        booking = cast(BookingRecord, booking_lookup)
        if booking["status"] == "cancelled":
            return booking

        cancelled_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._rest_patch(
            table=self._supabase["bookings_table"],
            query={"booking_id": f"eq.{booking['booking_id']}"},
            body={"status": "cancelled", "cancelled_at": cancelled_at},
        )

        flight = self._require_flight(booking["flight_id"])
        self._update_flight_seats(
            flight_id=booking["flight_id"],
            seats_left=int(flight["seats_left"]) + 1,
        )

        booking["status"] = "cancelled"
        booking["cancelled_at"] = cancelled_at
        self._cache_clear()
        return booking

    #endregion

    #region Data access helper methods for flights and availability

    def _fetch_matching_flights(
        self,
        origin_code: str,
        destination_code: str,
        travel_date: str,
    ) -> list[FlightRow]:
        """Fetch flights matching exact route/date with seats available.

        Parameters:
            origin_code: Normalized IATA-like origin code.
            destination_code: Normalized IATA-like destination code.
            travel_date: Validated ISO travel date in YYYY-MM-DD format.
        """
        rows = self._rest_get(
            table=self._supabase["flights_table"],
            query={
                "select": (
                    "flight_id,airline,origin,destination,date,depart_time,"
                    "arrive_time,base_fare_usd,seats_left"
                ),
                "origin": f"eq.{origin_code}",
                "destination": f"eq.{destination_code}",
                "date": f"eq.{travel_date}",
                "seats_left": "gt.0",
                "order": "depart_time.asc,flight_id.asc",
            },
        )
        return [self._row_to_flight(row) for row in rows]

    def _fetch_available_flights(
        self,
        origin_code: str | None,
        destination_code: str | None,
        earliest_date: str | None,
        limit: int,
    ) -> list[FlightRow]:
        """Fetch flights with optional filters and deterministic ordering.

        Parameters:
            origin_code: Optional normalized origin filter.
            destination_code: Optional normalized destination filter.
            earliest_date: Optional lower-bound date filter.
            limit: Maximum number of records requested from Supabase.
        """
        query: dict[str, str] = {
            "select": (
                "flight_id,airline,origin,destination,date,depart_time,"
                "arrive_time,base_fare_usd,seats_left"
            ),
            "seats_left": "gt.0",
            "order": "date.asc,depart_time.asc,flight_id.asc",
            "limit": str(limit),
        }
        if origin_code is not None:
            query["origin"] = f"eq.{origin_code}"
        if destination_code is not None:
            query["destination"] = f"eq.{destination_code}"
        if earliest_date is not None:
            query["date"] = f"gte.{earliest_date}"

        rows = self._rest_get(table=self._supabase["flights_table"], query=query)
        return [self._row_to_flight(row) for row in rows]

    def _find_next_available_dates(
        self,
        origin_code: str,
        destination_code: str,
        from_date: str,
        max_dates: int,
    ) -> list[str]:
        """Find distinct upcoming dates with availability for a route.

        Parameters:
            origin_code: Normalized origin code.
            destination_code: Normalized destination code.
            from_date: Inclusive lower bound in YYYY-MM-DD format.
            max_dates: Maximum number of distinct dates to return.
        """
        cache_key = f"next-dates:{origin_code}:{destination_code}:{from_date}:{max_dates}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cast(list[str], cached)

        rows = self._rest_get(
            table=self._supabase["flights_table"],
            query={
                "select": "date",
                "origin": f"eq.{origin_code}",
                "destination": f"eq.{destination_code}",
                "date": f"gte.{from_date}",
                "seats_left": "gt.0",
                "order": "date.asc",
                "limit": "100",
            },
        )

        seen_dates: set[str] = set()
        next_dates: list[str] = []
        for row in rows:
            date_value = str(row.get("date", ""))
            if not date_value or date_value in seen_dates:
                continue
            seen_dates.add(date_value)
            next_dates.append(date_value)
            if len(next_dates) >= max_dates:
                break

        self._cache_set(cache_key, next_dates)
        return next_dates

    def _require_flight(self, flight_id: str) -> FlightRow:
        """Load a flight by ID or raise when no matching row exists.

        Parameters:
            flight_id: Flight identifier to look up.
        """
        key = flight_id.strip().upper()
        cache_key = f"flight:{key}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cast(FlightRow, cached)

        rows = self._rest_get(
            table=self._supabase["flights_table"],
            query={
                "select": (
                    "flight_id,airline,origin,destination,date,depart_time,"
                    "arrive_time,base_fare_usd,seats_left"
                ),
                "flight_id": f"eq.{key}",
                "limit": "1",
            },
        )
        if not rows:
            raise ValueError(f"Unknown flight_id: {key}")

        flight = self._row_to_flight(rows[0])
        self._cache_set(cache_key, flight)
        return flight

    def _update_flight_seats(self, flight_id: str, seats_left: int) -> None:
        """Persist a flight seat-count update with non-negative clamping.

        Parameters:
            flight_id: Flight identifier to update.
            seats_left: Desired seat count before clamping at zero.
        """
        self._rest_patch(
            table=self._supabase["flights_table"],
            query={"flight_id": f"eq.{flight_id}"},
            body={"seats_left": max(0, seats_left)},
        )

    def _fare_multiplier(self, cabin_class: str) -> float:
        """Resolve cabin class to fare multiplier used in price quoting.

        Parameters:
            cabin_class: Cabin class key expected by booking tools.
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

    #endregion

    #region Normalization and validation helper methods

    def _validate_date(self, date_value: str) -> None:
        """Validate that a date string conforms to YYYY-MM-DD format.

        Parameters:
            date_value: Date string to validate.
        """
        try:
            datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must be in YYYY-MM-DD format.") from exc

    def _normalize_date_value(self, raw_date: str) -> str:
        """Normalize date phrases to ISO format when a resolver matches.

        Parameters:
            raw_date: Raw date text provided by user or tool invocation.
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
        """Resolve exact keywords `today`/`tomorrow` into ISO date strings.

        Parameters:
            normalized: Lower-cased date phrase.
            today: Current date used as calculation base.
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
        """Resolve phrases like `15th of march [2026]` into ISO date.

        Parameters:
            normalized: Lower-cased date phrase.
            today: Current date used for default year when omitted.
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
        """Resolve phrases like `march 15 [2026]` into ISO date.

        Parameters:
            normalized: Lower-cased date phrase.
            today: Current date used for default year when omitted.
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
        """Resolve relative phrases such as `3 days from today`.

        Parameters:
            normalized: Lower-cased date phrase.
            today: Current date used as calculation base.
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
        """Compose an ISO date from month/day/year components.

        Parameters:
            month_name: Lower-cased month name looked up in month mapping.
            day: Day of month value.
            year: Four-digit year.
        """
        month_number = MONTH_NAME_TO_NUMBER.get(month_name)
        if month_number is None:
            return None
        return datetime(year, month_number, day).date().isoformat()

    def _normalize_location_code(self, location_text: str) -> str:
        """Normalize city text or IATA-like code to uppercase location code.

        Parameters:
            location_text: User-provided city name or code candidate.
        """
        normalized = location_text.strip().lower()
        if not normalized:
            raise ValueError("Location cannot be empty.")

        cache_key = f"location:{normalized}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cast(str, cached)

        rows = self._rest_get(
            table=self._supabase["lookup_table"],
            query={
                "select": "city_name,iata_code",
                "city_name": f"eq.{normalized}",
                "limit": "1",
            },
        )
        if rows:
            iata_code = str(rows[0].get("iata_code", "")).upper()
            if iata_code:
                self._cache_set(cache_key, iata_code)
                return iata_code

        if len(normalized) == 3 and normalized.isalpha():
            code = normalized.upper()
            self._cache_set(cache_key, code)
            return code

        fallback = location_text.strip().upper()
        self._cache_set(cache_key, fallback)
        return fallback

    def _city_name_from_iata(self, iata_code: str) -> str:
        """Resolve an IATA code to city name for postcard prompt quality."""
        normalized_code = iata_code.strip().upper()
        rows = self._rest_get(
            table=self._supabase["lookup_table"],
            query={
                "select": "city_name",
                "iata_code": f"eq.{normalized_code}",
                "limit": "1",
            },
        )
        if rows:
            city_name = str(rows[0].get("city_name", "")).strip()
            if city_name:
                return city_name.title()
        return normalized_code

    def _postcard_failure(
        self,
        booking_id: str,
        destination_city: str,
        prompt: str,
        message: str,
    ) -> PostcardToolResult:
        """Build a consistent postcard failure payload."""
        return {
            "status": "failed",
            "booking_id": booking_id,
            "destination_city": destination_city,
            "prompt": prompt,
            "image_path": None,
            "message": message,
        }

    #endregion

    #region Serialization, caching, and Supabase proxy methods

    def _serialize_flight(self, flight: FlightRow) -> FlightToolResult:
        """Project a full flight row into the public tool response shape.

        Parameters:
            flight: Internal flight row dictionary.
        """
        return {
            "flight_id": flight["flight_id"],
            "airline": flight["airline"],
            "depart_time": flight["depart_time"],
            "arrive_time": flight["arrive_time"],
            "base_fare_usd": int(flight["base_fare_usd"]),
            "seats_left": int(flight["seats_left"]),
        }

    def _row_to_flight(self, row: dict[str, object]) -> FlightRow:
        """Convert a generic Supabase row into a typed internal flight row.

        Parameters:
            row: Dictionary payload returned from Supabase.
        """
        return {
            "flight_id": str(row.get("flight_id", "")),
            "airline": str(row.get("airline", "")),
            "origin": str(row.get("origin", "")),
            "destination": str(row.get("destination", "")),
            "date": str(row.get("date", "")),
            "depart_time": str(row.get("depart_time", "")),
            "arrive_time": str(row.get("arrive_time", "")),
            "base_fare_usd": int(str(row.get("base_fare_usd", 0))),
            "seats_left": int(str(row.get("seats_left", 0))),
        }

    def _row_to_booking(self, row: dict[str, object]) -> BookingRecord:
        """Convert a generic Supabase row into a typed booking record.

        Parameters:
            row: Dictionary payload returned from Supabase.
        """
        return {
            "booking_id": str(row.get("booking_id", "")),
            "status": cast(Literal["confirmed", "cancelled"], str(row.get("status", "confirmed"))),
            "created_at": str(row.get("created_at", "")),
            "cancelled_at": cast(str | None, row.get("cancelled_at")),
            "passenger_name": str(row.get("passenger_name", "")),
            "flight_id": str(row.get("flight_id", "")),
            "airline": str(row.get("airline", "")),
            "origin": str(row.get("origin", "")),
            "destination": str(row.get("destination", "")),
            "date": str(row.get("date", "")),
            "depart_time": str(row.get("depart_time", "")),
            "arrive_time": str(row.get("arrive_time", "")),
            "cabin_class": str(row.get("cabin_class", "")),
            "paid_fare_usd": int(str(row.get("paid_fare_usd", 0))),
        }

    def _cache_get(self, key: str) -> object | None:
        """Return a deep-copied cached value and refresh its LRU position.

        Parameters:
            key: Cache entry key.
        """
        value = self._result_cache.get(key)
        if value is None:
            return None
        self._result_cache.move_to_end(key)
        return json.loads(json.dumps(value))

    def _cache_set(self, key: str, value: object) -> None:
        """Store a deep-copied value in LRU cache and enforce max size.

        Parameters:
            key: Cache entry key.
            value: JSON-serializable object to store.
        """
        self._result_cache[key] = json.loads(json.dumps(value))
        self._result_cache.move_to_end(key)
        if len(self._result_cache) > self._result_cache_max_size:
            self._result_cache.popitem(last=False)

    def _cache_clear(self) -> None:
        """Remove all in-memory cached tool/query results."""
        self._result_cache.clear()

    def _rest_get(self, table: str, query: dict[str, str]) -> list[dict[str, object]]:
        """Proxy GET request to Supabase client.

        Parameters:
            table: Target table name.
            query: PostgREST query parameter mapping.
        """
        return self._supabase_client.get(table=table, query=query)

    def _rest_post(
        self,
        table: str,
        body: object,
        query: dict[str, str] | None,
    ) -> object:
        """Proxy POST request to Supabase client.

        Parameters:
            table: Target table name.
            body: JSON-serializable request payload.
            query: Optional PostgREST query parameters.
        """
        return self._supabase_client.post(table=table, body=body, query=query)

    def _rest_patch(self, table: str, query: dict[str, str], body: dict[str, object]) -> object:
        """Proxy PATCH request to Supabase client.

        Parameters:
            table: Target table name.
            query: PostgREST filters selecting rows to patch.
            body: JSON patch payload.
        """
        return self._supabase_client.patch(table=table, query=query, body=body)

    #endregion

    #region Debug and diagnostics methods

    def _debug_log(self, message: str) -> None:
        """Emit a prefixed debug log line when debug mode is enabled.

        Parameters:
            message: Diagnostic message to print.
        """
        if not self._answer_core.config.debug_enabled:
            return
        print(f"[BookingAgent DEBUG] {message}")

    def _shorten_for_debug(self, text: str, max_len: int = 240) -> str:
        """Truncate long debug text payloads to keep logs readable.

        Parameters:
            text: Full text payload to potentially shorten.
            max_len: Maximum allowed length before ellipsis is appended.
        """
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    #endregion


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
