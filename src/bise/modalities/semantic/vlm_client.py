from typing import Any, Dict


class VLMClient:
    def __init__(self, provider_name: str = "stub"):
        self.provider_name = provider_name

    def annotate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Integrate a concrete VLM provider before using semantic annotation.")
