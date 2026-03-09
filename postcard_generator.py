"""Postcard prompt and image generation service."""

from __future__ import annotations

import base64
import os
import re
from typing import Callable, Literal, TypedDict, cast

from openai import OpenAI

DEFAULT_DALLE_MODEL_NAME: str = "dall-e-3"
DEFAULT_DALLE_IMAGE_SIZE: str = "1024x1024"
DEFAULT_POSTCARD_OUTPUT_DIR: str = "generated_postcards"
DEFAULT_OLLAMA_BASE_URL: str = "http://localhost:11434/v1"

POSTCARD_PROMPT_SYSTEM_PROMPT: str = """
You are a travel postcard generator.

When given a city:

1. Identify ONE famous landmark, natural feature, or iconic element of that place.
2. Create a postcard prompt using that landmark.
3. Follow the template exactly.

Template:

A vintage travel postcard of {city}, featuring {landmark}.
The scene shows {environment description}.
Warm natural lighting, vibrant travel poster colors, cheerful holiday atmosphere.
Retro illustrated postcard style, wide scenic composition.

Return only the final prompt text.
""".strip()


class PostcardGenerationResult(TypedDict):
    """Result payload produced by postcard generation."""

    status: Literal["generated", "failed"]
    booking_id: str
    destination_city: str
    prompt: str
    image_path: str | None
    message: str


class PostcardGenerator:
    """Generate destination postcard prompts and images."""

    def __init__(
        self,
        reasoning_model_name: str,
        reasoning_base_url: str | None,
        reasoning_api_key: str | None,
        debug_log: Callable[[str], None] | None = None,
    ) -> None:
        self._reasoning_model_name = reasoning_model_name
        self._reasoning_base_url = reasoning_base_url or DEFAULT_OLLAMA_BASE_URL
        self._reasoning_api_key = (reasoning_api_key or "dummy").strip() or "dummy"
        self._debug_log = debug_log

    def generate_postcard(
        self,
        booking_id: str,
        destination_city: str,
    ) -> PostcardGenerationResult:
        """Create prompt and image for a destination postcard."""
        city = destination_city.strip() or "the destination city"
        self._log(f"postcard.generate.start booking_id={booking_id} city={city}")
        prompt = self._build_postcard_prompt(city)
        self._log(f"postcard.generate.prompt={self._shorten(prompt)}")
        try:
            image_path = self._generate_postcard_image(
                booking_id=booking_id,
                destination_city=city,
                prompt=prompt,
            )
        except Exception as exc:
            reason = str(exc)
            self._log(f"postcard.generate.failed booking_id={booking_id} reason={reason}")
            return {
                "status": "failed",
                "booking_id": booking_id,
                "destination_city": city,
                "prompt": prompt,
                "image_path": None,
                "message": f"Postcard generation failed: {reason}",
            }

        self._log(f"postcard.generate.success booking_id={booking_id} image_path={image_path}")
        return {
            "status": "generated",
            "booking_id": booking_id,
            "destination_city": city,
            "prompt": prompt,
            "image_path": image_path,
            "message": "Destination postcard generated successfully.",
        }

    def _build_postcard_prompt(self, destination_city: str) -> str:
        """Create a DALL-E prompt via isolated reasoning-model completion."""
        try:
            self._log(
                "postcard.prompt.request "
                f"model={self._reasoning_model_name} base_url={self._reasoning_base_url}"
            )
            prompt_client = OpenAI(
                api_key=self._reasoning_api_key,
                base_url=self._reasoning_base_url,
            )
            completion = prompt_client.chat.completions.create(
                model=self._reasoning_model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": POSTCARD_PROMPT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Destination city: {destination_city}"},
                ],
            )
            if completion.choices and completion.choices[0].message.content:
                generated_prompt = completion.choices[0].message.content.strip()
                if generated_prompt:
                    self._log("postcard.prompt.success")
                    return generated_prompt
        except Exception as exc:
            self._log(f"postcard.prompt.failed reason={exc}")

        self._log("postcard.prompt.fallback_used")
        return self._fallback_postcard_prompt(destination_city)

    def _fallback_postcard_prompt(self, destination_city: str) -> str:
        """Return a deterministic prompt when the isolated model call fails."""
        return (
            f"A vintage travel postcard of {destination_city}, featuring a recognizable landmark "
            "and a vibrant retro travel poster style. Warm cinematic lighting, rich atmosphere, "
            "and saturated colors with no modern text overlays."
        )

    def _generate_postcard_image(
        self,
        booking_id: str,
        destination_city: str,
        prompt: str,
    ) -> str:
        """Call DALL-E image generation API and persist the image to disk."""
        self._log("postcard.image.start")
        api_key = (
            os.getenv("DALLE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        if not api_key or api_key.lower() == "dummy":
            raise RuntimeError(
                "Missing valid OpenAI API key. Set DALLE_OPENAI_API_KEY for postcard generation."
            )

        base_url = (os.getenv("DALLE_BASE_URL") or "").strip()
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)

        configured_model_name = os.getenv("DALLE_MODEL_NAME") or DEFAULT_DALLE_MODEL_NAME
        model_candidates = self._build_model_candidates(configured_model_name)
        requested_size = (os.getenv("DALLE_IMAGE_SIZE") or DEFAULT_DALLE_IMAGE_SIZE).strip()
        allowed_sizes = {
            "auto",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "256x256",
            "512x512",
            "1792x1024",
            "1024x1792",
        }
        image_size = (
            cast(
                Literal[
                    "auto",
                    "1024x1024",
                    "1536x1024",
                    "1024x1536",
                    "256x256",
                    "512x512",
                    "1792x1024",
                    "1024x1792",
                ],
                requested_size,
            )
            if requested_size in allowed_sizes
            else cast(Literal["1024x1024"], DEFAULT_DALLE_IMAGE_SIZE)
        )

        response = None
        last_error: Exception | None = None
        for model_name in model_candidates:
            try:
                self._log(
                    "postcard.image.request "
                    f"model={model_name} size={image_size} base_url={base_url or 'api.openai.com'}"
                )
                response = client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    size=image_size,
                    n=1,
                    quality="standard",
                    response_format="b64_json",
                )
                self._log(
                    "postcard.image.generated "
                    f"model={model_name} size={image_size}"
                )
                break
            except Exception as exc:
                last_error = exc
                self._log(f"postcard.image.failed model={model_name} reason={exc}")

        if response is None:
            attempted_models = ", ".join(model_candidates)
            if last_error is None:
                raise RuntimeError(
                    f"Image generation failed without explicit error. Attempted models: {attempted_models}."
                )
            hint = self._build_model_access_hint(
                base_url=base_url,
                attempted_models=attempted_models,
                last_error=str(last_error),
            )
            raise RuntimeError(
                "Image generation failed after trying models "
                f"[{attempted_models}]. Last error: {last_error}. {hint}"
            )

        if not response.data:
            raise RuntimeError("DALL-E did not return image data.")

        image_b64 = response.data[0].b64_json
        if not image_b64:
            raise RuntimeError("DALL-E response did not include base64 image payload.")

        image_bytes = base64.b64decode(image_b64)
        output_dir = os.getenv("POSTCARD_OUTPUT_DIR") or DEFAULT_POSTCARD_OUTPUT_DIR
        absolute_output_dir = os.path.abspath(output_dir)
        os.makedirs(absolute_output_dir, exist_ok=True)

        city_slug = re.sub(r"[^a-zA-Z0-9]+", "_", destination_city.strip()).strip("_")
        if not city_slug:
            city_slug = "destination"

        file_name = f"{booking_id}_{city_slug}.png"
        output_path = os.path.join(absolute_output_dir, file_name)
        with open(output_path, "wb") as image_file:
            image_file.write(image_bytes)
        self._log(f"postcard.image.saved path={output_path}")
        return output_path

    def _build_model_candidates(self, configured_model_name: str) -> list[str]:
        """Return ordered unique image-model fallback candidates."""
        candidates = [configured_model_name.strip(), "gpt-image-1", "dall-e-3"]
        normalized: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in normalized:
                continue
            normalized.append(candidate)
        return normalized

    def _build_model_access_hint(
        self,
        base_url: str,
        attempted_models: str,
        last_error: str,
    ) -> str:
        """Return actionable guidance for model-not-found style failures."""
        normalized_base = (base_url or "").strip().lower()
        if "not found" not in last_error.lower() and "404" not in last_error:
            return "Verify API key validity, endpoint reachability, and image model permissions."

        if "azure" in normalized_base:
            return (
                "The endpoint appears to be Azure OpenAI. Use DALLE_MODEL_NAME as a deployment name "
                "(not a raw model id) and confirm the deployment supports image generation."
            )

        if normalized_base:
            return (
                "DALLE_BASE_URL is set to a custom endpoint. Ensure this provider exposes image models "
                f"and supports at least one of [{attempted_models}] for /images/generations."
            )

        return (
            "You are using api.openai.com. The current key/project likely lacks image-model access in this "
            "environment. Verify billing/entitlements, then try another DALLE_MODEL_NAME or use DALLE_BASE_URL "
            "for the provider where your image model is deployed."
        )

    def _log(self, message: str) -> None:
        """Emit a debug message when a logger callback is configured."""
        if self._debug_log is None:
            return
        self._debug_log(f"[PostcardGenerator] {message}")

    def _shorten(self, text: str, max_len: int = 240) -> str:
        """Shorten verbose text payloads for debug output."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."
