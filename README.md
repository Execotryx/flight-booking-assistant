# Flight Ticket Assistant

A flight-only booking assistant that uses an Ollama-compatible API and tool calling.

The assistant is domain-restricted to flight booking and travel tasks. It can search flights, quote fares, create/cancel bookings, and answer follow-up questions using live tool data from Supabase.

## What It Does

- Routes requests through a strict flight-domain classifier.
- Uses tool-calling for grounded answers (no invented availability, prices, or booking IDs).
- Supports deterministic handling for common follow-ups such as:
  - "what flights from London are available"
  - "same route, but later date"
- Provides both CLI and Gradio interfaces.

## Current Model Configuration

- Reasoning model: `lfm2.5-thinking:1.2b-q8_0`
- Answer model: `lfm2.5-thinking:1.2b-q8_0`

Note: `AIConfig.model_name` still exists as a base config field, but `FlightTicketBookingAgent` sets model names explicitly.

## Architecture Overview

- `flight_booking_agent.py`
  - Domain prompts and routing
  - Tool registration and handlers
  - Date/location normalization
  - Response cleanup (for example, removing stray `<response>...</response>` wrappers)
  - Supabase-backed booking/flight operations
- `SupabaseClient` (inside `flight_booking_agent.py`)
  - Encapsulates REST calls and network diagnostics
- `ai_core.py` and `ollama_core.py`
  - Tool-calling loop and Ollama-compatible API integration
- `main.py` and `gradio_app.py`
  - CLI and Gradio entry points

## Data Source

The assistant reads and writes flight data in Supabase tables:

- Flights table (default: `flights`)
- City lookup table (default: `city_code_lookup`)
- Bookings table (default: `bookings`)

It also keeps a small in-memory response cache for recent tool results (default size: `5`, configurable).

## Tools Available to the Model

- `get_current_system_date()`
- `resolve_travel_date(date_expression)`
- `search_flights(origin, destination, date)`
- `list_available_flights(origin?, destination?, earliest_date?, limit?)`
- `quote_fare(flight_id, cabin_class)`
- `get_flight_by_id(flight_id)`
- `create_booking(flight_id, passenger_name, cabin_class)`
- `get_booking(booking_id)`
- `cancel_booking(booking_id)`

## Requirements

- Python `>= 3.13`
- `uv`
- Ollama (or compatible OpenAI-style endpoint)
- Supabase project with seeded tables

Recommended model pull:

```bash
ollama pull lfm2.5-thinking:1.2b-q8_0
```

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Configure environment:

- Copy `.env.example` to `.env`
- Optionally copy `.seeding-env.example` to `.seeding-env` for seeding scripts

Minimum runtime variables:

```env
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:11434/v1

SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_KEY=<your-supabase-key>
```

Optional Supabase table and cache settings:

```env
SUPABASE_TABLE=flights
SUPABASE_LOOKUP_TABLE=city_code_lookup
SUPABASE_BOOKINGS_TABLE=bookings
SUPABASE_RESULT_CACHE_SIZE=5
```

Optional debug flags:

```env
AI_DEBUG_ENABLED=false
AI_DEBUG_INCLUDE_PROMPTS=false
```

## Seed Supabase (Optional but Recommended)

Use the included assets:

- `seed_supabase_flights.sql`
- `seed_supabase_flights.py`

If your tables are empty, seed before running so flight search and booking flows have data.

## Run

CLI:

```bash
uv run main.py --ui cli
```

Gradio:

```bash
uv run main.py --ui gradio
```

## Example Conversation

1. User: "What flights from London are available?"
2. Assistant: lists matching flights.
3. User: "At what date is FL-3001 departing?"
4. Assistant: resolves flight by ID via tool and returns departure date/time.

## Troubleshooting

- No model response:
  - Ensure Ollama/service is running and `OPENAI_BASE_URL` is reachable.
- Supabase errors:
  - Verify `SUPABASE_URL` and `SUPABASE_KEY`.
  - Ensure URL project ref matches key ref.
- Empty results for known cities:
  - Confirm `city_code_lookup` has expected city-to-IATA rows.
- Gradio launches but no debug text in UI:
  - Debug logs print to server stdout, not chat bubbles.

## Current Limitations

- No auth/payment integration.
- No external airline APIs.
- Availability depends on data in your Supabase tables.

