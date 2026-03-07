"""Configuration loading for model, API credentials, and compaction behavior."""

import os
from dotenv import load_dotenv


class AIConfig:
    """Load runtime configuration from environment variables."""

    @property
    def openai_api_key(self) -> str:
        """Get the OpenAI API key."""
        return self._openai_api_key

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def base_url(self) -> str | None:
        """Get the base URL for the API (e.g., for Ollama/vLLM)."""
        return self._base_url

    @property
    def pair_compaction_enabled(self) -> bool:
        """Whether pair-based history compaction is enabled."""
        return self._pair_compaction_enabled

    @property
    def max_pairs_before_compaction(self) -> int:
        """Maximum number of complete user-assistant pairs before compaction."""
        return self._max_pairs_before_compaction

    @property
    def pairs_to_keep_recent(self) -> int:
        """How many most-recent complete pairs stay unsummarized."""
        return self._pairs_to_keep_recent

    @property
    def compaction_max_retries(self) -> int:
        """Maximum number of additional compaction retries after an API failure."""
        return self._compaction_max_retries

    @property
    def max_tool_call_rounds(self) -> int:
        """Maximum number of model-tool turns allowed in one user request."""
        return self._max_tool_call_rounds

    def __init__(self) -> None:
        """Initialize configuration values from the process environment."""
        load_dotenv()
        self._openai_api_key: str = os.getenv("OPENAI_API_KEY", "dummy")
        self._model_name: str = os.environ["MODEL_NAME"]
        self._base_url: str | None = os.getenv("OPENAI_BASE_URL")
        self._pair_compaction_enabled: bool = os.getenv(
            "PAIR_COMPACTION_ENABLED", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._max_pairs_before_compaction: int = max(
            1,
            int(os.getenv("MAX_PAIRS_BEFORE_COMPACTION", "12")),
        )
        self._pairs_to_keep_recent: int = max(
            0,
            int(os.getenv("PAIRS_TO_KEEP_RECENT", "4")),
        )
        self._compaction_max_retries: int = max(
            0,
            int(os.getenv("COMPACTION_MAX_RETRIES", "1")),
        )
        self._max_tool_call_rounds: int = max(
            1,
            int(os.getenv("MAX_TOOL_CALL_ROUNDS", "8")),
        )
