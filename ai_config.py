"""Configuration loading for model, API credentials, and compaction behavior."""

import os
from typing import Any

from dotenv import load_dotenv


class AIConfig:
    """Load and validate runtime configuration with key whitelisting support.

    Config values can come from environment variables or explicit `overrides`.
    Only keys included in `permitted_keys` are read from these sources; other
    keys fall back to defaults.
    """

    ENV_NAME_BY_KEY: dict[str, str] = {
        "openai_api_key": "OPENAI_API_KEY",
        "model_name": "MODEL_NAME",
        "base_url": "OPENAI_BASE_URL",
        "debug_enabled": "AI_DEBUG_ENABLED",
        "debug_include_prompts": "AI_DEBUG_INCLUDE_PROMPTS",
        "pair_compaction_enabled": "PAIR_COMPACTION_ENABLED",
        "max_pairs_before_compaction": "MAX_PAIRS_BEFORE_COMPACTION",
        "pairs_to_keep_recent": "PAIRS_TO_KEEP_RECENT",
        "compaction_max_retries": "COMPACTION_MAX_RETRIES",
        "max_tool_call_rounds": "MAX_TOOL_CALL_ROUNDS",
    }
    DEFAULTS: dict[str, Any] = {
        "openai_api_key": "dummy",
        "model_name": "smollm2:latest",
        "base_url": None,
        "debug_enabled": False,
        "debug_include_prompts": False,
        "pair_compaction_enabled": True,
        "max_pairs_before_compaction": 12,
        "pairs_to_keep_recent": 4,
        "compaction_max_retries": 1,
        "max_tool_call_rounds": 8,
    }

    @property
    def openai_api_key(self) -> str:
        """Get the OpenAI API key."""
        return str(self._values["openai_api_key"])

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return str(self._values["model_name"])

    @property
    def base_url(self) -> str | None:
        """Get the base URL for the API (e.g., for Ollama/vLLM)."""
        value: Any = self._values["base_url"]
        if value is None:
            return None
        return str(value)

    @property
    def pair_compaction_enabled(self) -> bool:
        """Whether pair-based history compaction is enabled."""
        return bool(self._values["pair_compaction_enabled"])

    @property
    def debug_enabled(self) -> bool:
        """Whether runtime debug logs are enabled."""
        return bool(self._values["debug_enabled"])

    @property
    def debug_include_prompts(self) -> bool:
        """Whether debug logs should include full prompts and user inputs."""
        return bool(self._values["debug_include_prompts"])

    @property
    def max_pairs_before_compaction(self) -> int:
        """Maximum number of complete user-assistant pairs before compaction."""
        return int(self._values["max_pairs_before_compaction"])

    @property
    def pairs_to_keep_recent(self) -> int:
        """How many most-recent complete pairs stay unsummarized."""
        return int(self._values["pairs_to_keep_recent"])

    @property
    def compaction_max_retries(self) -> int:
        """Maximum number of additional compaction retries after an API failure."""
        return int(self._values["compaction_max_retries"])

    @property
    def max_tool_call_rounds(self) -> int:
        """Maximum number of model-tool turns allowed in one user request."""
        return int(self._values["max_tool_call_rounds"])

    def __init__(
        self,
        overrides: dict[str, Any] | None = None,
        permitted_keys: set[str] | None = None,
    ) -> None:
        """Initialize configuration with optional key filtering and overrides.

        Parameters:
                overrides: Explicit per-key values that take precedence over env.
                permitted_keys: Allowed keys that may be loaded from env/overrides.
                        Non-permitted keys are set to defaults.
        """
        load_dotenv()

        known_keys: set[str] = set(self.ENV_NAME_BY_KEY.keys())
        allowed_keys: set[str] = known_keys if permitted_keys is None else set(permitted_keys)
        unknown_allowed_keys: set[str] = allowed_keys - known_keys
        if unknown_allowed_keys:
            unknown_keys_text = ", ".join(sorted(unknown_allowed_keys))
            raise ValueError(f"Unknown permitted config keys: {unknown_keys_text}")

        resolved_overrides: dict[str, Any] = {} if overrides is None else dict(overrides)
        unknown_override_keys: set[str] = set(resolved_overrides.keys()) - known_keys
        if unknown_override_keys:
            unknown_override_text = ", ".join(sorted(unknown_override_keys))
            raise ValueError(f"Unknown override config keys: {unknown_override_text}")

        disallowed_override_keys: set[str] = set(resolved_overrides.keys()) - allowed_keys
        if disallowed_override_keys:
            disallowed_keys_text = ", ".join(sorted(disallowed_override_keys))
            raise ValueError(
                "Override keys not permitted by whitelist: "
                f"{disallowed_keys_text}"
            )

        self._values: dict[str, Any] = {}
        for key in known_keys:
            default_value: Any = self.DEFAULTS[key]
            if key not in allowed_keys:
                self._values[key] = default_value
                continue

            if key in resolved_overrides:
                raw_value: Any = resolved_overrides[key]
            else:
                env_name: str = self.ENV_NAME_BY_KEY[key]
                env_value: str | None = os.getenv(env_name)
                raw_value = default_value if env_value is None else env_value

            self._values[key] = self._normalize_value(key, raw_value)

    def _normalize_value(self, key: str, raw_value: Any) -> Any:
        """Normalize and validate one configuration value by key.

        Parameters:
            key: Logical configuration key to normalize.
            raw_value: Raw value from environment, overrides, or defaults.
        """
        if key == "openai_api_key":
            return str(raw_value)

        if key == "model_name":
            return str(raw_value)

        if key == "base_url":
            if raw_value is None:
                return None
            normalized_base_url = str(raw_value).strip()
            return normalized_base_url or None

        if key == "pair_compaction_enabled":
            return self._to_bool(raw_value)

        if key == "debug_enabled":
            return self._to_bool(raw_value)

        if key == "debug_include_prompts":
            return self._to_bool(raw_value)

        if key == "max_pairs_before_compaction":
            return max(1, self._to_int(raw_value, key))

        if key == "pairs_to_keep_recent":
            return max(0, self._to_int(raw_value, key))

        if key == "compaction_max_retries":
            return max(0, self._to_int(raw_value, key))

        if key == "max_tool_call_rounds":
            return max(1, self._to_int(raw_value, key))

        raise ValueError(f"Unsupported config key: {key}")

    def _to_bool(self, value: Any) -> bool:
        """Convert accepted bool-like values to bool.

        Parameters:
            value: Raw value to interpret as boolean.
        """
        if isinstance(value, bool):
            return value
        normalized_value = str(value).strip().lower()
        return normalized_value in {"1", "true", "yes", "on"}

    def _to_int(self, value: Any, key: str) -> int:
        """Convert value to int and raise a key-specific error on failure.

        Parameters:
                value: Raw value expected to be integer-compatible.
                key: Logical configuration key used in validation errors.
        """
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer value for '{key}': {value}") from exc
