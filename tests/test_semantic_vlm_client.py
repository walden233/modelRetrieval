import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.vlm_client import OpenAICompatibleVLMClient


def test_openai_compatible_client_includes_thinking_type():
    client = OpenAICompatibleVLMClient(
        base_url="http://127.0.0.1:8000/v1",
        model_name="test-model",
        thinking_type="disabled",
    )
    body = client._build_request_body(
        {
            "prompt": "hello",
            "system_prompt": "system",
            "frames": [],
            "model": "test-model",
        }
    )
    assert body["thinking"] == {"type": "disabled"}
