from typing import Dict, Iterable


def weighted_score_fusion(modality_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(modality_scores[name] * weights.get(name, 0.0) for name in modality_scores)


def grid_search_weights(score_names: Iterable[str], step: float = 0.1):
    names = list(score_names)
    if not names:
        return []
    if len(names) == 1:
        return [{names[0]: 1.0}]

    candidates = []
    steps = int(1.0 / step)
    for first in range(steps + 1):
        for second in range(steps + 1 - first):
            weights = [first * step, second * step]
            if len(names) == 2 and abs(sum(weights) - 1.0) < 1e-9:
                candidates.append(dict(zip(names, weights)))
    return candidates
