from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

from PIL import Image

from paddleocrvl_api.PaddleOCRVLAPIConfiguration import (
    PaddleOCRVLAPIConfiguration,
)


class PaddleOCRVLVLLMClient:
    def __init__(self, configuration: PaddleOCRVLAPIConfiguration):
        self._configuration = configuration

    def parse_image(self, image: Image.Image, prompt: str) -> str:
        payload = self._build_payload(image, prompt)
        request = urllib.request.Request(
            url=f"{self._configuration.server_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._configuration.request_timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"PaddleOCR-VL vLLM request failed: HTTP {error.code}: {body}"
            ) from error

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"Unexpected vLLM response shape: {data}"
            ) from error

    def _build_payload(self, image: Image.Image, prompt: str) -> dict:
        image_url = self._image_to_data_url(image)
        return {
            "model": self._configuration.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self._configuration.max_tokens,
            "temperature": self._configuration.temperature,
        }

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
