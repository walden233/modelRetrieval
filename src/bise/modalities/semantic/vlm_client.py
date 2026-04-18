from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, request

import cv2
import numpy as np


class VLMClientError(RuntimeError):
    pass


class VLMResponseParseError(VLMClientError):
    pass


@dataclass
class VLMResponse:
    content: str
    metadata: Dict[str, Any]


class VLMClient:
    def __init__(self, provider_name: str = "stub", model_name: str = "stub-model"):
        self.provider_name = provider_name
        self.model_name = model_name

    def annotate_description(self, payload: Dict[str, Any]) -> VLMResponse:
        raise NotImplementedError

    def annotate_labels(self, payload: Dict[str, Any]) -> VLMResponse:
        raise NotImplementedError


class StubVLMClient(VLMClient):
    def __init__(self, description_response: Optional[Dict[str, Any]] = None, label_response: Optional[Dict[str, Any]] = None):
        super().__init__(provider_name="stub", model_name="stub-model")
        self._description_response = description_response or {"task_description": "the robot moves an object to a target"}
        self._label_response = label_response or {
            "capability_tags": ["transport"],
            "action_slots": {"object": "object", "target": "target", "verb": "move"},
        }

    def annotate_description(self, payload: Dict[str, Any]) -> VLMResponse:
        return VLMResponse(content=json.dumps(self._description_response, ensure_ascii=False), metadata={"provider": self.provider_name})

    def annotate_labels(self, payload: Dict[str, Any]) -> VLMResponse:
        return VLMResponse(content=json.dumps(self._label_response, ensure_ascii=False), metadata={"provider": self.provider_name})


class OpenAICompatibleVLMClient(VLMClient):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "",
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ):
        super().__init__(provider_name="openai_compatible", model_name=model_name)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def annotate_description(self, payload: Dict[str, Any]) -> VLMResponse:
        return self._chat_completion(payload)

    def annotate_labels(self, payload: Dict[str, Any]) -> VLMResponse:
        return self._chat_completion(payload)

    def _chat_completion(self, payload: Dict[str, Any]) -> VLMResponse:
        body = self._build_request_body(payload)
        last_error: Optional[Exception] = None
        for _ in range(max(1, self.max_retries)):
            try:
                return self._send_request(body)
            except (error.HTTPError, error.URLError, TimeoutError, VLMClientError) as exc:
                last_error = exc
        raise VLMClientError(f"VLM request failed after retries: {last_error}") from last_error

    def _send_request(self, body: Dict[str, Any]) -> VLMResponse:
        endpoint = f"{self.base_url}/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = request.Request(endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMResponseParseError(f"Unexpected VLM response payload: {payload}") from exc
        metadata = {"provider": self.provider_name, "model": self.model_name, "raw_response": payload}
        return VLMResponse(content=str(content), metadata=metadata)

    def _build_request_body(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(payload["prompt"])
        system_prompt = str(payload.get("system_prompt", "")).strip()
        frames = payload.get("frames", [])
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": _frame_to_data_url(frame)}})
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})
        body = {
            "model": payload.get("model", self.model_name),
            "messages": messages,
            "temperature": payload.get("temperature", 0.0),
        }
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return body


def _frame_to_data_url(frame: Any) -> str:
    if isinstance(frame, str) and frame.startswith("data:image/"):
        return frame
    matrix = np.asarray(frame)
    if matrix.ndim != 3:
        raise ValueError("Frame must be an HWC image array.")
    if matrix.dtype != np.uint8:
        matrix = matrix.astype("uint8")
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode frame as JPEG.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def build_vlm_client(config: Dict[str, Any]) -> VLMClient:
    provider_name = str(config.get("provider_name", "stub"))
    if provider_name == "stub":
        return StubVLMClient()
    if provider_name == "openai_compatible":
        api_key = config.get("api_key", "")
        return OpenAICompatibleVLMClient(
            base_url=str(config["base_url"]),
            model_name=str(config["model_name"]),
            api_key=str(api_key),
            timeout_seconds=int(config.get("timeout_seconds", 120)),
            max_retries=int(config.get("max_retries", 3)),
        )
    raise ValueError(f"Unsupported VLM provider: {provider_name}")
