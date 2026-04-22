import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic import batch_client as batch_client_module


class _FakeModel:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class _FakeContent:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class _FakeFiles:
    def __init__(self):
        self.created = []
        self.requested = []

    def create(self, *, file, purpose):
        self.created.append((file.name, purpose))
        return _FakeModel({"id": "file_123", "purpose": purpose})

    def content(self, file_id):
        self.requested.append(file_id)
        return _FakeContent(b"batch-result")


class _FakeBatches:
    def __init__(self):
        self.created = []
        self.retrieved = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return _FakeModel({"id": "batch_123", "status": "validating"})

    def retrieve(self, batch_id):
        self.retrieved.append(batch_id)
        return _FakeModel({"id": batch_id, "status": "completed", "output_file_id": "file_out"})


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.files = _FakeFiles()
        self.batches = _FakeBatches()


def test_zhipu_batch_client_uses_zai_sdk(monkeypatch, tmp_path: Path):
    captured = {}

    def _factory(**kwargs):
        client = _FakeClient(**kwargs)
        captured["client"] = client
        return client

    monkeypatch.setattr(batch_client_module, "ZhipuAiClient", _factory)
    client = batch_client_module.ZhipuBatchClient(
        api_key="test-key",
        base_url="https://example.com/v4",
        timeout_seconds=30,
        max_retries=5,
    )

    sample_file = tmp_path / "requests.jsonl"
    sample_file.write_text('{"ok": true}\n', encoding="utf-8")

    uploaded = client.upload_batch_file(sample_file)
    assert uploaded.id == "file_123"
    assert captured["client"].files.created == [(str(sample_file), "batch")]

    batch = client.create_batch(
        input_file_id="file_123",
        endpoint="/v4/chat/completions",
        auto_delete_input_file=True,
        metadata={"project": "semantic", "count": 2},
    )
    assert batch.id == "batch_123"
    assert captured["client"].batches.created[0]["metadata"] == {"project": "semantic", "count": "2"}

    status = client.retrieve_batch("batch_123")
    assert status.payload["status"] == "completed"
    assert captured["client"].batches.retrieved == ["batch_123"]

    content = client.download_file("file_out")
    assert content == b"batch-result"
    assert captured["client"].files.requested == ["file_out"]
