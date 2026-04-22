from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from zai import ZhipuAiClient


class BatchClientError(RuntimeError):
    pass


@dataclass
class BatchFileResponse:
    id: str
    payload: Dict[str, Any]


@dataclass
class BatchJobResponse:
    id: str
    payload: Dict[str, Any]


class ZhipuBatchClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str | None = None,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = ZhipuAiClient(
            api_key=api_key or None,
            base_url=base_url or None,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def upload_batch_file(self, file_path: str | Path, purpose: str = "batch") -> BatchFileResponse:
        candidate = Path(file_path)
        try:
            with candidate.open("rb") as handle:
                response = self.client.files.create(file=handle, purpose=purpose)
        except Exception as exc:  # noqa: BLE001
            raise BatchClientError(f"Batch file upload failed: {exc}") from exc
        payload = _coerce_sdk_payload(response)
        return BatchFileResponse(id=str(payload["id"]), payload=payload)

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        auto_delete_input_file: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJobResponse:
        try:
            response = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint=endpoint,
                auto_delete_input_file=auto_delete_input_file,
                metadata=_stringify_metadata(metadata or {}),
            )
        except Exception as exc:  # noqa: BLE001
            raise BatchClientError(f"Batch job creation failed: {exc}") from exc
        payload = _coerce_sdk_payload(response)
        return BatchJobResponse(id=str(payload["id"]), payload=payload)

    def retrieve_batch(self, batch_id: str) -> BatchJobResponse:
        try:
            response = self.client.batches.retrieve(batch_id)
        except Exception as exc:  # noqa: BLE001
            raise BatchClientError(f"Batch job retrieval failed: {exc}") from exc
        payload = _coerce_sdk_payload(response)
        return BatchJobResponse(id=str(payload["id"]), payload=payload)

    def download_file(self, file_id: str) -> bytes:
        try:
            response = self.client.files.content(file_id)
        except Exception as exc:  # noqa: BLE001
            raise BatchClientError(f"Batch file download failed: {exc}") from exc
        if hasattr(response, "read"):
            data = response.read()
            if isinstance(data, bytes):
                return data
        data = getattr(response, "content", None)
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode("utf-8")
        raise BatchClientError(f"Unsupported batch file content type: {type(response)!r}")


def build_batch_client(config: Dict[str, Any]) -> ZhipuBatchClient:
    api_key = str(config.get("api_key", ""))
    api_key_env = str(config.get("api_key_env", ""))
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env, "")
    return ZhipuBatchClient(
        api_key=api_key,
        base_url=str(config.get("base_url", "")).strip() or None,
        timeout_seconds=int(config.get("timeout_seconds", 120)),
        max_retries=int(config.get("max_retries", 3)),
    )


def _coerce_sdk_payload(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return dict(response.model_dump())
    if hasattr(response, "to_dict"):
        return dict(response.to_dict())
    if isinstance(response, dict):
        return dict(response)
    raise BatchClientError(f"Unsupported SDK response type: {type(response)!r}")


def _stringify_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    return {str(key): str(value) for key, value in metadata.items()}
