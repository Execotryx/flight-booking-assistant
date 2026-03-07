# Flight Ticket Assistant

Flight-only booking assistant built on an Ollama-compatible OpenAI API stack.

Current implementation includes:
- Dual-model architecture (reasoning + answering).
- Tool calling for flight search and booking operations.
- Deterministic fallback handling for common follow-up intents to avoid tool-loop dead ends.
- CLI and Gradio interfaces.

## Current Models

- Reasoning model: `lfm2.5-thinking:1.2b-q8_0`
- Answer model: `granite4:3b`

`AIConfig.model_name` still exists and defaults to `smollm2:latest`, but `FlightTicketBookingAgent` overrides model names per core.

## Project Layout

- `ai_config.py`: validated environment/override config, debug and tool-loop controls.
- `ai_core.py`: typed chat orchestration core, tool-call loop, history compaction/summarization.
- `ollama_core.py`: Ollama-specific `AICore` implementation.
- `flight_booking_agent.py`: booking domain logic, prompts, tools, and deterministic follow-up handlers.
- `gradio_app.py`: web UI wrapper.
- `main.py`: entrypoint with `--ui` selection.

## Features

- Strict domain scope: flight booking and flight travel only.
- Reasoning gate plus keyword fallback for routing.
- Tool-grounded responses for prices, IDs, and availability.
- In-memory flights and bookings with create/get/cancel lifecycle.
- Natural-language date normalization.
- Follow-up robustness for prompts like:
  - "same origin/destination, but a later date"
  - "what flights from London are available currently"
- Performance-oriented optimizations:
  - Precompiled regex and static lookup maps.
  - Route/date index for flight searches.
  - Slotted `Flight` dataclass.
  - Reduced duplicate tool-call extraction work in core.

## Tools Exposed to the Answer Model

- `get_current_system_date()`
- `resolve_travel_date(date_expression)`
- `search_flights(origin, destination, date)`
  - Returns `flights` plus `next_available_dates`.
- `list_available_flights(origin?, destination?, earliest_date?, limit?)`
  - Used for broader availability discovery and alternatives.
- `quote_fare(flight_id, cabin_class)`
- `create_booking(flight_id, passenger_name, cabin_class)`
- `get_booking(booking_id)`
- `cancel_booking(booking_id)`

## Requirements

- Python `>= 3.13`
- `uv`
- Ollama running at `http://localhost:11434/v1` (default fallback)
- Pulled models:
  - `lfm2.5-thinking:1.2b-q8_0`
  - `granite4:3b`

Example:

```bash
ollama pull lfm2.5-thinking:1.2b-q8_0
ollama pull granite4:3b
```

## Installation

```bash
uv sync
```

## Configuration

Use environment variables or `.env`:

```env
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:11434/v1

# Baseline config value (agent overrides per-core model names)
MODEL_NAME=smollm2:latest

# Optional diagnostics
AI_DEBUG_ENABLED=false
AI_DEBUG_INCLUDE_PROMPTS=false

# Optional history/tool settings
PAIR_COMPACTION_ENABLED=true
MAX_PAIRS_BEFORE_COMPACTION=12
PAIRS_TO_KEEP_RECENT=4
COMPACTION_MAX_RETRIES=1
MAX_TOOL_CALL_ROUNDS=8
```

## Run

CLI mode:

```bash
uv run main.py --ui cli
```

Gradio mode:

```bash
uv run main.py --ui gradio
```

## Agent Behavior Summary

1. User message is classified by reasoning model.
2. If in-domain, answer model handles request with tools.
3. Tool outputs are fed back through the loop until finalized.
4. For specific follow-up intents (later date/current availability), deterministic handlers may answer directly from current state and tool output to prevent repeated failed loops.

## Example Flow

- User: "I want to book a ticket from London"
- Assistant: asks for missing fields.
- User: "Paris, two days from today, V. Perepletkina, business"
- Assistant: resolves date, searches, quotes, creates booking.
- User: "same origin and destination, but a later date"
- Assistant: lists available alternatives instead of repeating failed query.

## Limitations

- Inventory is in memory and resets each run.
- No persistence/auth/payments/external airline APIs.
- Availability reflects only seeded sample flights.

## Troubleshooting

- Model errors: ensure both models are pulled in Ollama.
- Connection errors: verify Ollama is running and `OPENAI_BASE_URL` is correct.
- Frequent refusals: enable debug flags and verify reasoning model availability.
- Tool-round limit reached: review prompts/tool schemas and adjust `MAX_TOOL_CALL_ROUNDS` conservatively.

