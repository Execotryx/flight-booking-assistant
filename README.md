# Flight Ticket Assistant

Domain-focused flight ticket booking assistant powered by Ollama-compatible chat models, with tool calling, CLI mode, and Gradio UI.

## Highlights

- Strict domain scope: answers only flight-booking and flight-travel questions.
- Two-model routing:
	- Reasoning/classification model: `lfm2.5-thinking:1.2b-q8_0`
	- Answering/tool model: `smollm2:latest`
- Tool-calling flow for realistic booking actions:
	- search flights
	- quote fare
	- create booking
	- fetch booking
	- cancel booking
- Conversation history support with optional pair compaction/summarization.
- Two interfaces:
	- Terminal chat (`--ui cli`)
	- Web chat via Gradio (`--ui gradio`)

---

## Project Structure

- `ai_config.py`: environment-driven runtime configuration.
- `ai_core.py`: generic core for chat orchestration, history, compaction, and tool-calling loop.
- `ollama_core.py`: concrete `AICore` implementation for Ollama OpenAI-compatible endpoints.
- `flight_booking_agent.py`: domain agent, prompts, in-memory flight/booking data, and tool handlers.
- `gradio_app.py`: Gradio chat interface wrapper around the booking agent.
- `main.py`: single entrypoint with CLI flags to select UI mode.

---

## Requirements

- Python `>= 3.13`
- [uv](https://docs.astral.sh/uv/) for dependency management
- Ollama running locally (default endpoint `http://localhost:11434/v1`)
- Required Ollama models pulled:
	- `lfm2.5-thinking:1.2b-q8_0`
	- `smollm2:latest`

Example pulls:

```bash
ollama pull lfm2.5-thinking:1.2b-q8_0
ollama pull smollm2:latest
```

---

## Installation

```bash
uv sync
```

---

## Configuration

Set environment variables (optionally via `.env`):

```env
# Required by AIConfig
MODEL_NAME=smollm2:latest

# Optional for local Ollama (default fallback is this URL inside OllamaAICore)
OPENAI_BASE_URL=http://localhost:11434/v1

# Optional (dummy is fine for local Ollama)
OPENAI_API_KEY=dummy

# Optional history/tool behavior
PAIR_COMPACTION_ENABLED=true
MAX_PAIRS_BEFORE_COMPACTION=12
PAIRS_TO_KEEP_RECENT=4
COMPACTION_MAX_RETRIES=1
MAX_TOOL_CALL_ROUNDS=8
```

Notes:

- `MODEL_NAME` is still required by `AIConfig`, but per-core model overrides are used in the booking agent:
	- reasoning core uses `lfm2.5-thinking:1.2b-q8_0`
	- answer core uses `smollm2:latest`

---

## Running

Use `main.py` as the unified entrypoint.

### CLI mode (default)

```bash
uv run main.py
```

or explicitly:

```bash
uv run main.py --ui cli
```

### Gradio web UI

```bash
uv run main.py --ui gradio
```

Gradio will print a local URL (typically `http://127.0.0.1:7860`).

---

## How the Agent Works

1. User message arrives.
2. Reasoning model classifies message as either:
	 - `FLIGHT_RELATED`
	 - `NOT_FLIGHT_RELATED`
3. If not flight-related, agent refuses with a short domain-safe response.
4. If flight-related, answering model handles the request with tool calling.
5. Tool results are fed back into the model until a final response is produced.

The system prompt for the answering model explicitly restricts scope and requires tool-grounded claims for flights/prices/bookings.

---

## Available Tools

### `search_flights(origin, destination, date)`

- Inputs:
	- `origin` (IATA/city code style, normalized to uppercase)
	- `destination`
	- `date` (`YYYY-MM-DD`)
- Output: matching flights with fares and seats left.

### `quote_fare(flight_id, cabin_class)`

- Cabin classes:
	- `economy`
	- `premium_economy`
	- `business`
- Output: base fare and final fare.

### `create_booking(flight_id, passenger_name, cabin_class)`

- Creates a booking if seats are available.
- Returns booking record including `booking_id` and paid fare.

### `get_booking(booking_id)`

- Returns booking details or `not_found` response.

### `cancel_booking(booking_id)`

- Cancels existing booking and releases one seat back to inventory.

---

## Sample Questions

- "Find flights from NYC to LON on 2026-03-20"
- "Quote business fare for FL-1002"
- "Book FL-1002 for Alex Johnson in economy"
- "Get booking BK-1234ABCD"
- "Cancel booking BK-1234ABCD"

Off-topic example:

- "Explain Python decorators"

Expected behavior: refusal with redirection to flight-booking assistance.

---

## Troubleshooting

- **Model not found**
	- Pull required models in Ollama and retry.
- **Connection errors to model API**
	- Confirm Ollama server is running.
	- Verify `OPENAI_BASE_URL` points to your Ollama API.
- **No response / unexpected refusals**
	- Ensure reasoning model is available; if reasoning call fails, classifier defaults to non-flight.
- **Tool-call loops exceeded**
	- Increase `MAX_TOOL_CALL_ROUNDS` if needed, but keep it bounded.

---

## Development Notes

- The booking data is in-memory only and resets each run.
- The agent is intentionally strict and minimal by design.
- If you extend tools, keep system prompts and tool schemas aligned to preserve domain safety.

