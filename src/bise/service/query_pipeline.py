from typing import Dict, List

from bise.retrieval.fusion import weighted_score_fusion


def rank_results(candidate_scores: List[Dict[str, float]], weights: Dict[str, float]):
    ranked = []
    for candidate in candidate_scores:
        fused_score = weighted_score_fusion(candidate["scores"], weights)
        ranked.append({"sample_id": candidate["sample_id"], "score": fused_score})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)
