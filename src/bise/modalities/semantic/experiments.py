"""Controlled retrieval, cross-view diagnostics, and review data for chapter 3."""

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bise.modalities.semantic.encoding import load_embeddings
from bise.modalities.semantic.manifest import relative_path
from bise.modalities.semantic.representations import LABEL_FIELDS, REPRESENTATIONS, representation_texts
from bise.modalities.semantic.store import RequestStore, item_key, read_jsonl, utc_now, write_json
from bise.retrieval.metrics import calculate_label_retrieval_metrics


def select_available(records: list[dict], vectors: dict, directory: Path,
                     representations, allow_partial: bool) -> tuple[list[dict], list[str]]:
    positions = {sample: index for index, sample in enumerate(vectors["sample_ids"].tolist())}
    annotations = RequestStore(directory).annotations()
    selected, missing = [], []
    columns = [vectors["metadata"]["representations"].index(rep) for rep in representations]
    for row in records:
        index = positions.get(row["sample_id"])
        annotation = annotations.get(item_key(row))
        if (index is None or annotation is None or not vectors["ready"][index, columns].all()
                or str(vectors["source_keys"][index]) != item_key(row)
                or str(vectors["attempt_ids"][index]) != annotation["attempt_id"]):
            missing.append(row["sample_id"])
        else:
            selected.append(row)
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)}/{len(records)} evaluation videos lack matching annotations/"
                         f"embeddings; run annotate/encode. First IDs: {', '.join(missing[:3])}. "
                         "--allow-partial is only for explicitly labeled partial/smoke results.")
    return selected, missing


def evaluate(config: dict, directory: Path, embedding_file: Path, records: list[dict],
             output: Path, cfg: str, representations=None,
             allow_partial: bool = False, split: str = "test") -> dict:
    started = time.perf_counter()
    vectors = load_embeddings(embedding_file)
    representations = representations or vectors["metadata"]["representations"]
    rows, missing = select_available(records, vectors, directory, representations, allow_partial)
    human = [r for r in rows if r["role"] == "human"]
    robot = [r for r in rows if r["role"] == "robot"]
    if not human or not robot:
        raise ValueError(f"No complete human/robot evaluation set for {cfg}/{split}")
    positions = {sample: index for index, sample in enumerate(vectors["sample_ids"].tolist())}
    protocol = {"ids": [r["sample_id"] for r in rows],
                "attempts": [str(vectors["attempt_ids"][positions[r["sample_id"]]]) for r in rows],
                "embedding": vectors["metadata"]["config"], "job": directory.name,
                "split": split, "seed": config["evaluation"]["seed"]}
    signature = json.loads((directory / 'params.json').read_text())['signature']
    protocol['extraction_controls'] = {key: signature[key] for key in (
        'provider', 'model', 'base_url', 'request', 'frames', 'max_side', 'jpeg_quality')}
    protocol['joint_prompt'] = signature['prompt'] if signature['kind'] == 'video' else None
    previous_path = output / "metrics.json"
    old = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    compatible = old.get("protocol") == protocol
    metrics = old.get("metrics", {}) if compatible else {}
    rankings = [row for row in read_jsonl(output / "rankings.jsonl")
                if compatible and row["representation"] not in representations]
    matrices = {}
    for direction, queries, candidates in (
            ("human_to_robot", human, robot), ("robot_to_human", robot, human)):
        order = np.random.default_rng(config["evaluation"]["seed"]).permutation(len(candidates))
        candidates = [candidates[int(index)] for index in order]
        qi = [positions[r["sample_id"]] for r in queries]
        gi = [positions[r["sample_id"]] for r in candidates]
        for rep in representations:
            similarities = vectors[rep][qi] @ vectors[rep][gi].T
            matrices[f"{direction}_{rep}"] = similarities
            metrics.setdefault(direction, {})
            for level in ("task", "scene"):
                key = level + "_id"
                qlabels = [r[key] for r in queries]
                glabels = np.asarray([r[key] for r in candidates])
                result = calculate_label_retrieval_metrics(
                    similarities, qlabels, glabels, stable_sort=True)
                metrics[direction].setdefault(level, {})[rep] = result
                for row, scores in zip(queries, similarities):
                    indices = np.argsort(-scores, kind="stable")
                    positive = glabels[indices] == row[key]
                    ranks = (np.flatnonzero(positive) + 1).tolist()
                    top = indices[:config["evaluation"]["top_k"]]
                    rankings.append({
                        "cfg": cfg, "direction": direction, "level": level,
                        "representation": rep, "query_id": row["sample_id"],
                        "positive_ranks": ranks, "best_rank": min(ranks) if ranks else None,
                        "gallery_size": len(candidates),
                        "top1_ties": int(np.sum(scores == scores.max())),
                        "top_k": [{"sample_id": candidates[int(i)]["sample_id"],
                                   "score": float(scores[i]),
                                   "positive": bool(candidates[int(i)][key] == row[key])} for i in top],
                    })
    payload = {"schema_version": 3, "cfg": cfg, "split": split, "job": directory.name,
               "embedding_file": relative_path(embedding_file), "protocol": protocol,
               "coverage": {"expected": len(records), "present": len(rows), "missing_ids": missing},
               "partial": bool(missing), "metrics": metrics,
               "evaluated_utc": utc_now(), "seconds": time.perf_counter() - started,
               "stale_representations_removed": [] if compatible or not old else [
                   rep for rep in REPRESENTATIONS if rep not in representations]}
    write_json(previous_path, payload)
    temporary = output / "rankings.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rankings))
    temporary.replace(output / "rankings.jsonl")
    return payload


def analyze(config: dict, directory: Path, embedding_file: Path, records: list[dict],
            output: Path, split: str = "test", video_run: str | None = None,
            trajectory_run: str | None = None) -> dict:
    vectors = load_embeddings(embedding_file)
    representations = vectors["metadata"]["representations"]
    rows, missing = select_available(records, vectors, directory, representations, True)
    positions = {sample: index for index, sample in enumerate(vectors["sample_ids"].tolist())}
    annotations = RequestStore(directory).annotations()
    labels = [annotations[item_key(row)]["labels"] for row in rows
              if "labels" in annotations[item_key(row)]]
    label_counts = Counter(json.dumps(label, sort_keys=True) for label in labels)
    distributions = {field: dict(Counter(
        str(label[field]) for label in labels)) for field in LABEL_FIELDS}
    entropy = -sum((n / len(labels)) * math.log2(n / len(labels))
                   for n in label_counts.values()) if labels else None
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_id"]].append(row)
    view_rows = []
    rng = random.Random(config["evaluation"]["seed"])
    for scene, members in sorted(groups.items()):
        for left in members:
            for right in members:
                if left["role"] == "robot" and right["role"] == "human":
                    continue
                if left["role"] == right["role"] and left["sample_id"] >= right["sample_id"]:
                    continue
                if left["role"] == right["role"] and left["camera_id"] == right["camera_id"]:
                    continue
                kind = ("HH" if left["role"] == right["role"] == "human" else
                        "RR" if left["role"] == right["role"] == "robot" else
                        "HR_same_camera" if left["camera_id"] == right["camera_id"] else "HR_cross_camera")
                negatives = [row for row in rows if row["cfg"] == left["cfg"] and
                             row["role"] == right["role"] and row["task_id"] != left["task_id"]]
                negative = rng.choice(negatives) if negatives else None
                near = [row for row in rows if row["cfg"] == left["cfg"] and
                        row["role"] == right["role"] and row["task_id"] == left["task_id"] and
                        row["scene_id"] != scene]
                same_task_other_scene = rng.choice(near) if near else None
                a, b = positions[left["sample_id"]], positions[right["sample_id"]]
                for rep in vectors["metadata"]["representations"]:
                    score = float(vectors[rep][a] @ vectors[rep][b])
                    neg = float(vectors[rep][a] @ vectors[rep][positions[negative["sample_id"]]]) if negative else None
                    near_score = float(vectors[rep][a] @ vectors[rep][positions[same_task_other_scene["sample_id"]]]) if same_task_other_scene else None
                    view_rows.append({"cfg": left["cfg"], "representation": rep, "kind": kind,
                                      "left": left["sample_id"], "right": right["sample_id"],
                                      "positive": score, "negative_id": negative["sample_id"] if negative else None,
                                      "negative": neg, "margin": score - neg if neg is not None else None,
                                      "same_task_other_scene_id": same_task_other_scene["sample_id"] if same_task_other_scene else None,
                                      "same_task_other_scene": near_score})
    changes, failure_candidates = [], {}
    for path in (output / "evaluations").glob("*/rankings.jsonl"):
        indexed = {(row["direction"], row["level"], row["query_id"], row["representation"]): row
                   for row in read_jsonl(path)}
        for key, row in indexed.items():
            if row['best_rank'] is not None and row['best_rank'] > 1:
                failure_candidates.setdefault(row['query_id'], []).append({
                    'representation':row['representation'], 'level':row['level'],
                    'best_rank':row['best_rank'], 'top1':row['top_k'][0]['sample_id']})
            if key[-1] != "S2":
                continue
            baseline = indexed.get((*key[:-1], "S1"))
            if baseline:
                changes.append({"cfg": row["cfg"], "direction": row["direction"],
                                "level": row["level"], "query_id": row["query_id"],
                                "S1": baseline["best_rank"], "S2": row["best_rank"]})
    examples = {}
    for row in rows:
        key = f"{row['cfg']}_{row['role']}"
        annotation = annotations[item_key(row)]
        examples.setdefault(key, {
            "sample_id": row["sample_id"], "video_path": row["video_path"],
            "role": row["role"], "attempt_id": annotation["attempt_id"],
            "frames": annotation["frames"],
            "texts": representation_texts(annotation, config["taxonomy"]),
            "annotation_file": relative_path(directory / "annotations.jsonl"),
        })
    payload = {"schema_version": 3, "split": split, "job": directory.name,
               "videos": len(rows), "missing": missing, "views": view_rows,
               "labels": {"unique_combinations": len(label_counts), "entropy_bits": entropy,
                          "distributions": distributions,
                          "unknown_or_empty_values": sum(value in ([], "unknown", ["unknown"])
                              for label in labels for value in label.values())},
               "rank_changes": changes, "examples": examples,
               "failure_candidates": failure_candidates}
    if video_run or trajectory_run:
        payload['multimodal'] = multimodal_cases(config, vectors, rows, video_run, trajectory_run)
    review_path = output / "reviews.csv"
    existing = {}
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = {row["sample_id"]: row for row in csv.DictReader(handle)}
    selected_scenes = sorted(groups)
    if len(selected_scenes) > 12:
        by_cfg = defaultdict(list)
        for scene in selected_scenes:
            by_cfg[scene.split('/')[0]].append(scene)
        selected_scenes = []
        for scenes in by_cfg.values():
            selected_scenes.extend(rng.sample(scenes, min(6, len(scenes))))
    fields = ["sample_id", "kind", "actions_ok", "objects_ok", "result_supported", "role_focus_ok",
              "colors_s3_s4_ok", "error_type", "review_type", "reviewer", "notes"]
    for scene in selected_scenes:
        for row in groups[scene]:
            existing.setdefault(row["sample_id"], {"sample_id": row["sample_id"], "kind": "audit"})
    for sample in list(failure_candidates)[:20]:
        existing.setdefault(sample, {'sample_id':sample, 'kind':'failure_candidate'})
    by_id = {row['sample_id']:row for row in rows}
    for review in existing.values():
        if review.get('kind') not in ('case','failure_case') or review['sample_id'] not in by_id:
            continue
        row = by_id[review['sample_id']]
        annotation = annotations[item_key(row)]
        examples.setdefault('selected_' + str(len(examples)), {
            'sample_id':row['sample_id'], 'video_path':row['video_path'], 'role':row['role'],
            'attempt_id':annotation['attempt_id'], 'frames':annotation['frames'],
            'texts':representation_texts(annotation,config['taxonomy']),
            'annotation_file':relative_path(directory / 'annotations.jsonl')})
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing.values())
    write_json(output / "analysis.json", payload)
    return payload


def multimodal_cases(config: dict, vectors: dict, records: list[dict],
                     video_run: str | None, trajectory_run: str | None) -> dict:
    """Re-rank saved scores only on strictly common queries and candidates."""
    from bise.common.paths import resolve_path
    positions = {sample:i for i,sample in enumerate(vectors['sample_ids'].tolist())}
    result = {}
    for cfg in sorted({row['cfg'] for row in records}):
        rows = [row for row in records if row['cfg'] == cfg]
        human = {row['sample_id']:row for row in rows if row['role'] == 'human'}
        robot = {row['sample_id']:row for row in rows if row['role'] == 'robot'}
        saved = {}
        for name, run in [('video',video_run),('trajectory',trajectory_run)]:
            if not run:
                continue
            evaluation = resolve_path(run) / 'evaluations' / cfg
            meta = json.loads((evaluation / 'metadata.json').read_text())
            matrix = np.load(evaluation / 'similarity_matrix.npy', allow_pickle=False)
            ids = {}
            for role in ('human','robot'):
                scenes = meta.get(role + '_scene_ids', meta.get('scene_ids'))
                cameras = meta.get(role + '_camera_ids', meta.get('camera_ids'))
                ids[role] = [f"{cfg}/{scene}/{str(camera).removeprefix('cam_')}/{role}"
                             for scene,camera in zip(scenes,cameras)]
            if matrix.shape != (len(ids['human']),len(ids['robot'])):
                raise ValueError(f'Multimodal saved score/metadata shape mismatch: {evaluation}')
            if len(set(ids['human'])) != len(ids['human']) or len(set(ids['robot'])) != len(ids['robot']):
                raise ValueError(f'Ambiguous duplicated query/candidate identity: {evaluation}')
            saved[name] = {'matrix':matrix, 'queries':{x:i for i,x in enumerate(ids['human'])},
                           'candidates':{x:i for i,x in enumerate(ids['robot'])}}
        query_ids, candidate_ids = set(human), set(robot)
        for data in saved.values():
            query_ids &= data['queries'].keys()
            candidate_ids &= data['candidates'].keys()
        query_ids, candidate_ids = sorted(query_ids), sorted(candidate_ids)
        rng = np.random.default_rng(config['evaluation']['seed'])
        candidate_ids = [candidate_ids[int(i)] for i in rng.permutation(len(candidate_ids))]
        cfg_result = {'common_queries':len(query_ids), 'common_candidates':len(candidate_ids),
                      'semantic_queries':len(human), 'semantic_candidates':len(robot),
                      'scope':'explicit common subset; scores re-ranked, not copied top-k', 'rows':[]}
        if not query_ids or not candidate_ids:
            result[cfg] = cfg_result
            continue
        scores = {name:data['matrix'][np.ix_([data['queries'][x] for x in query_ids],
                                            [data['candidates'][x] for x in candidate_ids])]
                  for name,data in saved.items()}
        qi,gi = [positions[x] for x in query_ids],[positions[x] for x in candidate_ids]
        for rep in vectors["metadata"]["representations"]:
            scores[rep] = vectors[rep][qi] @ vectors[rep][gi].T
        for level in ('task','scene'):
            key = level + '_id'
            labels = np.asarray([robot[x][key] for x in candidate_ids])
            for index,query in enumerate(query_ids):
                positive = labels == human[query][key]
                if not positive.any():
                    continue
                ranks = {name:int(np.flatnonzero(positive[np.argsort(-matrix[index],kind='stable')])[0])+1
                         for name,matrix in scores.items()}
                for rep in vectors["metadata"]["representations"]:
                    cfg_result['rows'].append({'query_id':query,'representation':rep,'level':level,
                                              'semantic_rank':ranks[rep],
                                              'other_ranks':{name:ranks[name] for name in saved},
                                              'complementary_hit':ranks[rep] > 1 and any(ranks[name]==1 for name in saved),
                                              'all_fail':ranks[rep]>1 and all(ranks[name]>1 for name in saved)})
        result[cfg] = cfg_result
    return result


def scope_statistics(directory: Path, records: list[dict]) -> dict:
    store = RequestStore(directory)
    keys = {item_key(row) for row in records}
    events = read_jsonl(store.ledger)
    starts = {row["attempt_id"]: row for row in events
              if row["event"] == "started" and row["item_key"] in keys}
    all_responses = [row for row in events if row["event"] == "response" and row["attempt_id"] in starts]
    accepted = store.annotations()
    by_key = defaultdict(list)
    for row in starts.values():
        by_key[row['item_key']].append(row)
    active_attempts = set()
    for key, history in by_key.items():
        accepted_id = accepted.get(key, {}).get('attempt_id')
        end = next((index + 1 for index,row in enumerate(history) if row['attempt_id']==accepted_id), len(history))
        begin = max((index for index,row in enumerate(history[:end]) if row.get('new_episode')),default=0)
        active_attempts.update(row['attempt_id'] for row in history[begin:end])
    responses = [row for row in all_responses if row['attempt_id'] in active_attempts]
    latencies = [row["timings"]["request_seconds"] for row in responses]
    costs = [row["accounting"]["cost"] for row in responses if row["accounting"]["cost"] is not None]
    execution_keys = defaultdict(set)
    for row in starts.values():
        execution_keys[row["execution_id"]].add(row["item_key"])
    matched = [row for row in events if row["event"] == "execution_end" and
               row["new_items"] == len(keys) and execution_keys[row["execution_id"]] == keys]
    return {
        "selected": len(records), "attempted_items":len(by_key),
        "successful": sum(key in accepted for key in keys),
        "success_rate": sum(key in accepted for key in keys) / len(by_key) if by_key else None,
        "coverage_rate": sum(key in accepted for key in keys) / len(keys) if keys else None,
        "attempts": len(starts), "priced_attempts": len(costs),
        "known_cost": sum(costs), "currency": "CNY",
        "known_cost_per_video": sum(costs) / len(by_key) if by_key else None,
        "lifetime_known_cost":sum(row['accounting']['cost'] or 0 for row in all_responses),
        "unknown_cost_attempts": len(active_attempts) - len(costs),
        "comparison_cost_basis":"current accepted episode, including retries; forced repeats excluded from prior episode",
        "request_seconds_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "request_seconds_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "measured_items_per_second": matched[-1]["new_items_per_second"] if matched else None,
        "measurement_execution_id": matched[-1]["execution_id"] if matched else None,
        "latency_source": "original API response records; cache reads excluded",
    }
