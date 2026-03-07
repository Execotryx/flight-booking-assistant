"""Core abstractions for conversation history and AI completion orchestration."""

import json
import openai
from abc import ABC, abstractmethod
from ai_config import AIConfig
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Generic, TypedDict, TypeVar, cast
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

TAiResponse = TypeVar("TAiResponse", default=object)


class ToolFunctionSchema(TypedDict):
    """Function-specific schema payload for model tool registration."""

    name: str
    description: str
    parameters: dict[str, object]


class ToolSchema(TypedDict):
    """Tool schema payload accepted by chat completion tool-calling."""

    type: str
    function: ToolFunctionSchema


ToolHandler = Callable[..., object]


class OpenAIClientKwargs(TypedDict, total=False):
    """Supported keyword arguments for OpenAI client construction in this project."""

    api_key: str
    base_url: str


class CompletionCallConfiguration(TypedDict, total=False):
    """Subset of chat completion options used by this project."""

    model: str
    messages: list[ChatCompletionMessageParam]
    tools: list[ToolSchema]
    tool_choice: str
    temperature: float
    max_tokens: int

SUMMARY_PREFIX: str = "Conversation memory summary (auto-generated):"
SUMMARIZER_SYSTEM_PROMPT: str = (
    "You are a conversation summarizer. Summarize prior user-assistant exchanges "
    "accurately and concisely. Preserve facts, user preferences, constraints, "
    "decisions, commitments, and unresolved questions. Do not invent information."
)


class HistoryManager:
    """Manage chat history and system behavior."""

    @property
    def system_behavior(self) -> str:
        """System instruction for the conversation."""
        return self._system_behavior

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        """Return the full conversation history. Read-only conceptually."""
        return self._messages

    def complete_pair_indices(self) -> list[tuple[int, int]]:
        """Return (user_idx, assistant_idx) tuples for complete pairs."""
        pairs: list[tuple[int, int]] = []
        pending_user_idx: int | None = None
        for index in range(1, len(self._messages)):
            message_role = self._messages[index].get("role")
            if message_role == "user":
                pending_user_idx = index
            elif message_role == "assistant" and pending_user_idx is not None:
                pairs.append((pending_user_idx, index))
                pending_user_idx = None
        return pairs

    def complete_pair_count(self) -> int:
        """Return the number of complete user-assistant pairs."""
        return len(self.complete_pair_indices())

    def replace_messages(self, messages: list[ChatCompletionMessageParam]) -> None:
        """Replace full history with a new message list.

        Parameters:
                messages: New complete history to store.
        """
        self._messages = messages

    def add_message(self, message: ChatCompletionMessageParam) -> None:
        """Add a message to the conversation history.

        Parameters:
                message: Message object to append.
        """
        self._messages.append(message)

    def __init__(self, system_behavior: str) -> None:
        """Create a history manager with the given system instruction.

        Parameters:
                system_behavior: System-level behavior and instructions.
        """
        self._system_behavior: str = system_behavior
        self._messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_behavior)
        ]


class AICore(ABC, Generic[TAiResponse]):
    """Base class for AI calls and history handling."""

    @property
    def config(self) -> AIConfig:
        """Get the current AIConfig."""
        return self._config

    @property
    def history_manager(self) -> HistoryManager:
        """Return the HistoryManager instance."""
        return self._history_manager

    @property
    def _ai_api(self) -> openai.OpenAI:
        """Return the OpenAI API instance."""
        return self._ai_api_client

    def __init__(self, config: AIConfig | None, system_behavior: str) -> None:
        """Initialize with config and system behavior instructions.

        Parameters:
                config: Optional runtime configuration. Uses defaults when None.
                system_behavior: Base system instruction for the conversation.
        """
        if config is None:
            config = AIConfig()
        self._config: AIConfig = config
        self._history_manager: HistoryManager = HistoryManager(system_behavior)

        # Allow base_url override for local Ollama/vLLM endpoints
        client_kwargs: OpenAIClientKwargs = {"api_key": self.config.openai_api_key}
        if hasattr(self.config, "base_url") and self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        self._ai_api_client: openai.OpenAI = openai.OpenAI(**client_kwargs)
        self._tool_schemas: list[ToolSchema] = []
        self._tool_handlers: dict[str, ToolHandler] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, object],
        handler: ToolHandler,
    ) -> None:
        """Register a callable tool for model-driven function calling.

        Parameters:
                name: Tool/function name exposed to the model.
                description: Natural-language description for tool selection.
                parameters: JSON-schema object describing accepted arguments.
                handler: Python callable executed when the tool is invoked.
        """
        tool_schema: ToolSchema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

        for index, existing_tool in enumerate(self._tool_schemas):
            function_payload = existing_tool.get("function", {})
            if function_payload.get("name") == name:
                self._tool_schemas[index] = tool_schema
                self._tool_handlers[name] = handler
                return

        self._tool_schemas.append(tool_schema)
        self._tool_handlers[name] = handler

    def clear_tools(self) -> None:
        """Remove all registered tools and handlers."""
        self._tool_schemas.clear()
        self._tool_handlers.clear()

    def ask(self, request: str) -> TAiResponse:
        """Send a user request to the model and return the processed response.

        Parameters:
                request: User input text for the current turn.
        """
        if self.config.debug_enabled and getattr(self.config, "debug_include_prompts", False):
            self._debug_log(f"ask.request={request}")
        self._add_user_message(request)
        self._compact_history_if_needed(force=False)
        response: ChatCompletion = self._execute_with_compaction_retry(request)
        self._append_assistant_message_from_response(response)
        self._debug_log("ask.completed")
        return self._process_response(response)

    def ask_stream(self, request: str) -> Iterator[str]:
        """Stream assistant text deltas for one user request.

        Parameters:
                request: User input text for the current turn.
        """
        if self._tool_schemas:
            raise RuntimeError("ask_stream does not support tool calling. Use ask().")

        self._add_user_message(request)
        self._compact_history_if_needed(force=False)

        retry_count: int = 0
        while True:
            call_configuration: CompletionCallConfiguration = self._form_call_configuration(
                request
            )
            try:
                stream = self._ai_api.chat.completions.create(
                    **cast(dict[str, Any], call_configuration),
                    stream=True,
                )

                assistant_parts: list[str] = []
                for chunk in stream:
                    delta_content: str = self._extract_delta_content(chunk)
                    if not delta_content:
                        continue
                    assistant_parts.append(delta_content)
                    self._on_stream_delta(delta_content)
                    yield delta_content

                assistant_text: str = "".join(assistant_parts)
                response: ChatCompletion = self._build_chat_completion_from_text(
                    assistant_text
                )
                self._append_assistant_message_from_response(response)
                return
            except Exception as exc:
                if not self._should_retry_after_context_error(exc, retry_count):
                    raise
                if not self._compact_history_if_needed(force=True):
                    raise
                retry_count += 1

    def _add_user_message(self, request: str) -> None:
        """Append the current user request to conversation history.

        Parameters:
                request: User input text for the current turn.
        """
        self.history_manager.add_message(
            ChatCompletionUserMessageParam(role="user", content=request)
        )

    def _append_assistant_message_from_response(self, response: ChatCompletion) -> None:
        """Append assistant content from a model response to history.

        Parameters:
                response: Completed chat response from the model.
        """
        if not response.choices or not response.choices[0].message.content:
            return
        self.history_manager.add_message(
            ChatCompletionAssistantMessageParam(
                role="assistant", content=response.choices[0].message.content
            )
        )

    def _execute_with_compaction_retry(self, request: str) -> ChatCompletion:
        """Execute streamed completion with context-overflow compaction retries.

        Parameters:
                request: User input text for the current turn.
        """
        retry_count: int = 0
        while True:
            call_configuration: CompletionCallConfiguration = self._form_call_configuration(
                request
            )
            try:
                return self._execute_completion_with_tools(call_configuration)
            except Exception as exc:
                if not self._should_retry_after_context_error(exc, retry_count):
                    raise
                if not self._compact_history_if_needed(force=True):
                    raise
                retry_count += 1

    def _execute_completion_with_tools(
        self, initial_call_configuration: CompletionCallConfiguration
    ) -> ChatCompletion:
        """Execute one model turn including optional tool-call follow-up rounds.

        Parameters:
                initial_call_configuration: Initial completion call options.
        """
        response: ChatCompletion = self._complete_once(initial_call_configuration)
        self._debug_log("answer.round=0 completed")
        if not self._tool_schemas:
            return response

        tool_round_count: int = 0
        while True:
            response, tool_round_count, finalized = self._run_tool_call_round(
                response=response,
                tool_round_count=tool_round_count,
            )
            if finalized:
                return response

    def _run_tool_call_round(
        self,
        response: ChatCompletion,
        tool_round_count: int,
    ) -> tuple[ChatCompletion, int, bool]:
        """Execute one tool-call round and return next response state.

        Parameters:
                response: Current model response to inspect for tool calls.
                tool_round_count: Number of completed tool rounds.
        """
        tool_calls: list[object] = self._extract_tool_calls(response)
        if not tool_calls:
            self._debug_log(
                f"answer.finalized without tool calls after {tool_round_count} round(s)"
            )
            return response, tool_round_count, True

        self._debug_log(
            f"answer.tool_round={tool_round_count + 1} tool_call_count={len(tool_calls)}"
        )
        self._append_assistant_tool_call_message(response, tool_calls)
        self._append_tool_results_to_history(tool_calls)

        next_round_count = tool_round_count + 1
        self._raise_if_tool_round_limit_reached(next_round_count)

        follow_up_configuration: CompletionCallConfiguration = self._form_call_configuration(
            ""
        )
        next_response = self._complete_once(follow_up_configuration)
        self._debug_log(f"answer.round={next_round_count} completed")
        return next_response, next_round_count, False

    def _raise_if_tool_round_limit_reached(self, tool_round_count: int) -> None:
        """Raise when tool-calling exceeds configured safety round limit.

        Parameters:
                tool_round_count: Count after completing the current round.
        """
        if tool_round_count < self.config.max_tool_call_rounds:
            return
        raise RuntimeError(
            "Tool-calling exceeded maximum rounds "
            f"({self.config.max_tool_call_rounds})."
        )

    def _complete_once(self, call_configuration: CompletionCallConfiguration) -> ChatCompletion:
        """Run a single completion call.

        Parameters:
                call_configuration: Completion call options excluding stream mode.
        """
        if self._tool_schemas:
            return self._ai_api.chat.completions.create(
                **cast(dict[str, Any], call_configuration),
            )
        return self._stream_completion(call_configuration)

    def _extract_tool_calls(self, response: ChatCompletion) -> list[object]:
        """Extract tool calls from the first assistant choice in a response.

        Parameters:
                response: Completion response to inspect.
        """
        if not response.choices:
            return []
        message = response.choices[0].message
        raw_tool_calls: object = getattr(message, "tool_calls", None)
        if raw_tool_calls is None:
            return []
        if not isinstance(raw_tool_calls, Iterable):
            return []
        return list(raw_tool_calls)

    def _append_assistant_tool_call_message(
        self,
        response: ChatCompletion,
        tool_calls: list[object],
    ) -> None:
        """Persist assistant tool-call request message to history.

        Parameters:
                response: Completion response containing tool calls.
                tool_calls: Already extracted tool call objects for this response.
        """
        if not response.choices:
            return
        message = response.choices[0].message
        tool_calls_payload = self._build_tool_calls_payload(tool_calls)
        assistant_message = self._build_assistant_tool_call_history_message(
            message=message,
            tool_calls_payload=tool_calls_payload,
        )
        self.history_manager.add_message(assistant_message)

    def _build_tool_calls_payload(
        self,
        tool_calls: list[object],
    ) -> list[ChatCompletionMessageToolCallParam]:
        """Convert tool call objects from response into history payload format.

        Parameters:
                tool_calls: Tool call objects extracted from completion output.
        """
        payload: list[ChatCompletionMessageToolCallParam] = []
        for tool_call in tool_calls:
            payload.append(self._tool_call_to_payload(tool_call))
        return payload

    def _tool_call_to_payload(
        self,
        tool_call: object,
    ) -> ChatCompletionMessageToolCallParam:
        """Convert a single tool-call object into serializable typed payload.

        Parameters:
                tool_call: Provider tool call object from completion output.
        """
        function_payload: object = getattr(tool_call, "function", None)
        return {
            "id": str(getattr(tool_call, "id", "")),
            "type": "function",
            "function": {
                "name": str(getattr(function_payload, "name", "")),
                "arguments": str(getattr(function_payload, "arguments", "{}")),
            },
        }

    def _build_assistant_tool_call_history_message(
        self,
        message: object,
        tool_calls_payload: list[ChatCompletionMessageToolCallParam],
    ) -> ChatCompletionAssistantMessageParam:
        """Build assistant history message carrying tool call request payload.

        Parameters:
                message: Assistant message object from completion choice.
                tool_calls_payload: Converted tool call payload list for history.
        """
        assistant_content: object = getattr(message, "content", None)
        if assistant_content is None:
            return {
                "role": "assistant",
                "tool_calls": tool_calls_payload,
            }

        return {
            "role": "assistant",
            "content": self._content_to_text(assistant_content),
            "tool_calls": tool_calls_payload,
        }

    def _append_tool_results_to_history(self, tool_calls: list[object]) -> None:
        """Execute requested tools and append their outputs to conversation history.

        Parameters:
                tool_calls: Tool call objects requested by the model.
        """
        for tool_call in tool_calls:
            tool_name, raw_arguments = self._extract_tool_call_invocation(tool_call)
            self._debug_log(f"tool.request name={tool_name} args={raw_arguments}")
            tool_result = self._run_tool_call(tool_name, raw_arguments)
            tool_message: ChatCompletionToolMessageParam = self._build_tool_result_message(
                tool_call=tool_call,
                tool_result=tool_result,
            )
            self.history_manager.add_message(tool_message)
            self._debug_log(
                f"tool.response name={tool_name} content={self._shorten_for_debug(tool_result)}"
            )

    def _extract_tool_call_invocation(self, tool_call: object) -> tuple[str, str]:
        """Extract tool name and raw JSON argument string from tool call object.

        Parameters:
                tool_call: Provider tool call object from completion output.
        """
        function_payload: object = getattr(tool_call, "function", None)
        tool_name = str(getattr(function_payload, "name", ""))
        raw_arguments = str(getattr(function_payload, "arguments", "{}"))
        return tool_name, raw_arguments

    def _run_tool_call(self, tool_name: str, raw_arguments: str) -> str:
        """Parse tool arguments and execute the registered handler.

        Parameters:
                tool_name: Tool/function name from the model output.
                raw_arguments: JSON argument text from tool call.
        """
        parsed_arguments: object
        try:
            parsed_arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            return f"Invalid tool arguments JSON: {exc}"
        return self._execute_registered_tool(tool_name, parsed_arguments)

    def _build_tool_result_message(
        self,
        tool_call: object,
        tool_result: str,
    ) -> ChatCompletionToolMessageParam:
        """Build one tool-result history message for follow-up completion round.

        Parameters:
                tool_call: Provider tool call object that requested execution.
                tool_result: Serialized tool execution output text.
        """
        return {
            "role": "tool",
            "tool_call_id": str(getattr(tool_call, "id", "")),
            "content": tool_result,
        }

    def _debug_log(self, message: str) -> None:
        """Print a debug log line when debug mode is enabled.

        Parameters:
                message: Debug text payload to emit.
        """
        if not getattr(self.config, "debug_enabled", False):
            return
        print(f"[AICore DEBUG] {message}")

    def _shorten_for_debug(self, text: str, max_len: int = 240) -> str:
        """Return shortened debug-safe text for logs.

        Parameters:
                text: Source text to shorten for compact logging.
                max_len: Maximum output length before appending ellipsis.
        """
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _execute_registered_tool(self, tool_name: str, arguments: object) -> str:
        """Run one registered tool handler and return serialized text output.

        Parameters:
                tool_name: Registered tool/function name.
                arguments: Parsed JSON arguments object.
        """
        handler: ToolHandler | None = self._tool_handlers.get(tool_name)
        if handler is None:
            return f"Tool '{tool_name}' is not registered."

        try:
            if isinstance(arguments, dict):
                result: object = handler(**arguments)
            else:
                result = handler(arguments)
        except Exception as exc:
            return f"Tool '{tool_name}' execution failed: {exc}"

        if result is None:
            return ""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except TypeError:
            return str(result)

    def _should_retry_after_context_error(self, exc: Exception, retry_count: int) -> bool:
        """Return True when a failed request should trigger compaction retry.

        Parameters:
                exc: Raised exception from completion execution.
                retry_count: Number of retries already attempted.
        """
        return self._is_context_length_error(exc) and (
            retry_count < self.config.compaction_max_retries
        )

    def _stream_completion(
        self,
        call_configuration: CompletionCallConfiguration,
    ) -> ChatCompletion:
        """Execute a streamed completion call and rebuild it as ChatCompletion.

        Parameters:
                call_configuration: Completion call options excluding stream mode.
        """
        stream = self._ai_api.chat.completions.create(
            **cast(dict[str, Any], call_configuration),
            stream=True,
        )

        assistant_text: str = self._collect_streamed_assistant_text(stream)
        return self._build_chat_completion_from_text(assistant_text)

    def _collect_streamed_assistant_text(self, stream: Iterator[ChatCompletionChunk]) -> str:
        """Collect all assistant delta fragments from a streaming iterator.

        Parameters:
                stream: Streaming iterator produced by chat completions API.
        """
        assistant_parts: list[str] = []
        for chunk in stream:
            delta_content: str = self._extract_delta_content(chunk)
            if not delta_content:
                continue
            assistant_parts.append(delta_content)
            self._on_stream_delta(delta_content)
        return "".join(assistant_parts)

    def _extract_delta_content(self, chunk: ChatCompletionChunk) -> str:
        """Extract text content from a single streaming chunk.

        Parameters:
                chunk: Streaming chat completion chunk.
        """
        typed_chunk: ChatCompletionChunk = chunk
        if not typed_chunk.choices:
            return ""
        delta_content: str | None = typed_chunk.choices[0].delta.content
        if delta_content is None:
            return ""
        return delta_content

    def _build_chat_completion_from_text(self, assistant_text: str) -> ChatCompletion:
        """Build a synthetic ChatCompletion object from aggregated text.

        Parameters:
                assistant_text: Full assistant text assembled from stream deltas.
        """
        return ChatCompletion.model_validate(
            {
                "id": "streamed-completion",
                "object": "chat.completion",
                "created": 0,
                "model": self.config.model_name,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": None,
                        "message": {
                            "role": "assistant",
                            "content": assistant_text,
                        },
                    }
                ],
            }
        )

    def _on_stream_delta(self, delta: str) -> None:
        """Handle each streamed text delta from the model.

        Parameters:
                delta: Newly received assistant text fragment.
        """
        return

    def _compact_history_if_needed(self, force: bool) -> bool:
        """Compact old complete pairs into one summary while preserving system instruction.

        Parameters:
                force: Whether to compact even when under the normal pair threshold.
        """
        if not self.config.pair_compaction_enabled:
            return False

        pairs_to_compact: list[tuple[int, int]] = self._get_pairs_to_compact(force)
        if not pairs_to_compact:
            return False

        message_indices_to_compact: set[int] = self._collect_pair_message_indices(
            pairs_to_compact
        )
        summary_text: str = self._generate_summary_for_pairs(pairs_to_compact)
        if not summary_text:
            return False

        self._replace_compacted_pairs_with_summary(
            message_indices_to_compact=message_indices_to_compact,
            summary_text=summary_text,
        )
        return True

    def _get_pairs_to_compact(self, force: bool) -> list[tuple[int, int]]:
        """Get complete user-assistant pairs that should be compacted.

        Parameters:
                force: Whether to compact even when under the normal pair threshold.
        """
        pairs: list[tuple[int, int]] = self.history_manager.complete_pair_indices()
        pair_count: int = len(pairs)
        if not force and pair_count <= self.config.max_pairs_before_compaction:
            return []
        pairs_to_keep_recent: int = min(self.config.pairs_to_keep_recent, pair_count)
        return pairs[: max(0, pair_count - pairs_to_keep_recent)]

    def _collect_pair_message_indices(
        self, pairs_to_compact: list[tuple[int, int]]
    ) -> set[int]:
        """Collect history indices for all messages in selected pairs.

        Parameters:
                pairs_to_compact: Pair indices selected for summarization.
        """
        message_indices_to_compact: set[int] = set()
        for user_index, assistant_index in pairs_to_compact:
            message_indices_to_compact.add(user_index)
            message_indices_to_compact.add(assistant_index)
        return message_indices_to_compact

    def _generate_summary_for_pairs(
        self, pairs_to_compact: list[tuple[int, int]]
    ) -> str:
        """Generate compact summary text for selected history pairs.

        Parameters:
                pairs_to_compact: Pair indices selected for summarization.
        """
        existing_summary: str | None = self._get_existing_summary_message()
        summary_input: str = self._build_summary_input(
            pairs_to_compact, existing_summary
        )
        return self._summarize_pairs(summary_input)

    def _build_summary_input(
        self,
        pairs_to_compact: list[tuple[int, int]],
        existing_summary: str | None,
    ) -> str:
        """Build textual input for the summarizer request.

        Parameters:
                pairs_to_compact: Message index pairs selected for summarization.
                existing_summary: Previously generated summary text, if available.
        """
        sections: list[str] = []
        if existing_summary:
            sections.append(f"Existing memory summary:\n{existing_summary}")

        transcript_lines: list[str] = ["Transcript to summarize:"]
        for user_index, assistant_index in pairs_to_compact:
            user_message = self.history_manager.messages[user_index]
            assistant_message = self.history_manager.messages[assistant_index]
            user_text = self._content_to_text(user_message.get("content"))
            assistant_text = self._content_to_text(assistant_message.get("content"))
            transcript_lines.append(f"User: {user_text}")
            transcript_lines.append(f"Assistant: {assistant_text}")

        sections.append("\n".join(transcript_lines))
        sections.append(
            "Return only the summary text. Keep it concise and structured in short bullets."
        )
        return "\n\n".join(sections)

    def _summarize_pairs(self, summary_input: str) -> str:
        """Summarize old pairs using the same model without persisting summarizer exchanges.

        Parameters:
                summary_input: Prompt payload containing prior summary and transcript excerpt.
        """
        summary_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=SUMMARIZER_SYSTEM_PROMPT,
            ),
            ChatCompletionUserMessageParam(role="user", content=summary_input),
        ]

        summary_response: ChatCompletion = self._ai_api.chat.completions.create(
            model=self.config.model_name,
            messages=summary_messages,
        )
        if (
            not summary_response.choices
            or not summary_response.choices[0].message.content
        ):
            return ""
        return summary_response.choices[0].message.content.strip()

    def _replace_compacted_pairs_with_summary(
        self,
        message_indices_to_compact: set[int],
        summary_text: str,
    ) -> None:
        """Replace compacted old pairs with one summary system message.

        Parameters:
                message_indices_to_compact: History indices that should be removed.
                summary_text: Replacement summary text for compacted history.
        """
        current_messages: list[ChatCompletionMessageParam] = (
            self.history_manager.messages
        )
        if not current_messages:
            return

        new_messages: list[ChatCompletionMessageParam] = [current_messages[0]]
        new_messages.append(
            ChatCompletionSystemMessageParam(
                role="system",
                content=f"{SUMMARY_PREFIX}\n{summary_text}",
            )
        )

        for index in range(1, len(current_messages)):
            if index in message_indices_to_compact:
                continue
            message = current_messages[index]
            if self._is_summary_message(message):
                continue
            new_messages.append(message)

        self.history_manager.replace_messages(new_messages)

    def _get_existing_summary_message(self) -> str | None:
        """Return existing auto-generated summary text if present."""
        for message in self.history_manager.messages[1:]:
            if not self._is_summary_message(message):
                continue
            content = self._content_to_text(message.get("content")).strip()
            if content.startswith(SUMMARY_PREFIX):
                return content[len(SUMMARY_PREFIX) :].strip()
        return None

    def _is_summary_message(self, message: ChatCompletionMessageParam) -> bool:
        """Return True when a message is the auto-generated memory summary.

        Parameters:
                message: History message to check.
        """
        if message.get("role") != "system":
            return False
        content = self._content_to_text(message.get("content"))
        return content.startswith(SUMMARY_PREFIX)

    def _is_context_length_error(self, exc: Exception) -> bool:
        """Detect provider errors related to context length overflow.

        Parameters:
                exc: Raised exception from the API call.
        """
        if not isinstance(exc, openai.BadRequestError):
            return False
        exc_text: str = str(exc).lower()
        error_code: str = str(getattr(exc, "code", "")).lower()
        return (
            "context_length" in error_code
            or "context length" in exc_text
            or "maximum context" in exc_text
            or "too many tokens" in exc_text
        )

    def _content_to_text(self, content: object) -> str:
        """Convert OpenAI message content variants into plain text.

        Parameters:
                content: Message content in string, list, or provider-specific format.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
                    elif "content" in item and isinstance(item["content"], str):
                        parts.append(item["content"])
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    @abstractmethod
    def _form_call_configuration(self, request: str) -> CompletionCallConfiguration:
        """Build the API call configuration for the current request.

        Parameters:
                request: User input text for the current turn.
        """
        call_configuration: CompletionCallConfiguration = {
            "model": self.config.model_name,
            "messages": self.history_manager.messages,
        }
        if self._tool_schemas:
            call_configuration["tools"] = self._tool_schemas
            call_configuration["tool_choice"] = "auto"
        return call_configuration

    @abstractmethod
    def _process_response(self, response: ChatCompletion) -> TAiResponse:
        """Convert a raw model completion into the target response type.

        Parameters:
                response: Raw completion returned by the chat API.
        """
        pass
