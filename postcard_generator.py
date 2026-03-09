"""Postcard prompt and image generation service."""

from __future__ import annotations

import base64
import os
import re
from typing import Literal, TypedDict, cast

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
    ) -> None:
        self._reasoning_model_name = reasoning_model_name
        self._reasoning_base_url = reasoning_base_url or DEFAULT_OLLAMA_BASE_URL
        self._reasoning_api_key = (reasoning_api_key or "dummy").strip() or "dummy"

    def generate_postcard(
        self,
        booking_id: str,
        destination_city: str,
    ) -> PostcardGenerationResult:
        """Create prompt and image for a destination postcard."""
        city = destination_city.strip() or "the destination city"
        prompt = self._build_postcard_prompt(city)
        try:
            image_path = self._generate_postcard_image(
                booking_id=booking_id,
                destination_city=city,
                prompt=prompt,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "booking_id": booking_id,
                "destination_city": city,
                "prompt": prompt,
                "image_path": None,
                "message": f"Postcard generation failed: {exc}",
            }

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
                    return generated_prompt
        except Exception:
            pass

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

        model_name = os.getenv("DALLE_MODEL_NAME") or DEFAULT_DALLE_MODEL_NAME
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

        response = client.images.generate(
            model=model_name,
            prompt=prompt,
            size=image_size,
            n=1,
            quality="standard",
            response_format="b64_json",
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
        return output_path
