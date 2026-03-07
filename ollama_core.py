"""Concrete AICore implementation for Ollama-compatible chat models.

Example:
        from ai_config import AIConfig
        from ollama_core import OllamaAICore

        def lookup_airport_city(iata: str) -> dict[str, str]:
            mapping = {
                "JFK": "New York",
                "LHR": "London",
                "NRT": "Tokyo",
            }
            return {"iata": iata.upper(), "city": mapping.get(iata.upper(), "Unknown")}

        core = OllamaAICore(config=AIConfig(), system_behavior="You are a flight assistant.")
        core.register_tool(
            name="lookup_airport_city",
            description="Lookup city by IATA airport code.",
            parameters={
                "type": "object",
                "properties": {"iata": {"type": "string"}},
                "required": ["iata"],
            },
            handler=lookup_airport_city,
        )

        answer = core.ask("What city is JFK in?")
        print(answer)
"""

from ai_config import AIConfig
from ai_core import AICore, CompletionCallConfiguration
from openai.types.chat.chat_completion import ChatCompletion


class OllamaAICore(AICore[str]):
    """Run chat completions against an Ollama OpenAI-compatible endpoint.

    Register tools with `register_tool(...)` before calling `ask(...)` to enable
    model-driven function calling.
    """

    def __init__(
        self,
        config: AIConfig | None,
        system_behavior: str,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize an Ollama-backed AI core.

        Parameters:
                config: Runtime configuration. Uses default config when None.
                system_behavior: Base system instructions for assistant behavior.
                model_name: Optional model override for this core instance only.
                temperature: Optional temperature override for response randomness.
                max_tokens: Optional completion length cap.
        """
        if config is None:
            resolved_config = AIConfig(
                overrides={"base_url": "http://localhost:11434/v1"}
            )
        elif config.base_url:
            resolved_config = config
        else:
            resolved_config = AIConfig(
                overrides={
                    "openai_api_key": config.openai_api_key,
                    "model_name": config.model_name,
                    "base_url": "http://localhost:11434/v1",
                    "pair_compaction_enabled": config.pair_compaction_enabled,
                    "max_pairs_before_compaction": config.max_pairs_before_compaction,
                    "pairs_to_keep_recent": config.pairs_to_keep_recent,
                    "compaction_max_retries": config.compaction_max_retries,
                    "max_tool_call_rounds": config.max_tool_call_rounds,
                }
            )

        self._model_name: str | None = model_name
        self._temperature: float | None = temperature
        self._max_tokens: int | None = max_tokens
        super().__init__(config=resolved_config, system_behavior=system_behavior)

    def _form_call_configuration(self, request: str) -> CompletionCallConfiguration:
        """Build the Ollama chat completion call configuration.

        Parameters:
                request: User input text for the current turn.
        """
        call_configuration: CompletionCallConfiguration = super()._form_call_configuration(
            request
        )
        if self._model_name is not None:
            call_configuration["model"] = self._model_name
        if self._temperature is not None:
            call_configuration["temperature"] = self._temperature
        if self._max_tokens is not None:
            call_configuration["max_tokens"] = self._max_tokens
        return call_configuration

    def _process_response(self, response: ChatCompletion) -> str:
        """Convert an Ollama chat completion response into plain text.

        Parameters:
                response: Raw completion returned by the chat API.
        """
        if not response.choices:
            return ""

        first_choice = response.choices[0]
        if first_choice.message.content is None:
            return ""
        return first_choice.message.content.strip()
