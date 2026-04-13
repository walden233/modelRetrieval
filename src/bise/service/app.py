from flask import Flask, jsonify, request

from .query_pipeline import rank_results


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/query/multimodal")
    def query_multimodal():
        payload = request.get_json(force=True)
        weights = payload.get("weights", {})
        candidates = payload.get("candidates", [])
        results = rank_results(candidates, weights)
        return jsonify({"results": results[: payload.get("top_k", 10)]})

    @app.post("/index/rebuild")
    def rebuild_index():
        return jsonify({"status": "not_implemented"})

    return app
