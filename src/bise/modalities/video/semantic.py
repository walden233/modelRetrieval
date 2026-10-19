"""Frozen scene teachers for batch relations or full text-bank distillation."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bise.common.paths import resolve_path
from .sampling import subset_samples


def scene_key(cfg, scene):
    return f"{cfg}/{scene}"


class SemanticSupervisor(nn.Module):
    def __init__(self, ids, vectors, config):
        super().__init__()
        self.mode = config["mode"]
        if self.mode not in {"batch", "bank"}:
            raise ValueError("semantic.mode must be batch or bank")
        self.text_temperature = float(config.get("text_temperature", 0.1))
        self.video_temperature = float(config.get("video_temperature", 0.1))
        self.weight = float(config.get("weight", 0.1))
        self.warmup_epochs = float(config.get("warmup_epochs", 1.0))
        if (min(self.text_temperature, self.video_temperature) <= 0
                or self.weight < 0 or self.warmup_epochs < 0):
            raise ValueError("Invalid semantic temperature, weight or warmup")
        self.indices = {identity: index for index, identity in enumerate(ids)}
        self.register_buffer("vectors", torch.as_tensor(vectors, dtype=torch.float32),
                             persistent=False)

    def forward(self, human, robot, keys, projector=None):
        indices = torch.tensor([self.indices.get(key, -1) for key in keys],
                               device=human.device)
        valid = indices >= 0
        count = int(valid.sum())
        zero = (human.sum() + robot.sum()) * 0.0
        # One distinct teacher gives no useful within-batch relational supervision.
        if count == 0 or (self.mode == "batch" and indices[valid].unique().numel() < 2):
            return zero, count, 0.0, 0
        text = self.vectors[indices[valid]]
        if self.mode == "batch":
            target_logits = text @ text.T / self.text_temperature
            logits = human[valid].float() @ robot[valid].float().T
            logits = logits / self.video_temperature
            pairs = ((target_logits, logits), (target_logits.T, logits.T))
        else:
            if projector is None:
                raise ValueError("bank supervision requires the shared semantic projector")
            target_logits = text @ self.vectors.T / self.text_temperature
            h = F.normalize(projector(human[valid]).float(), dim=-1)
            r = F.normalize(projector(robot[valid]).float(), dim=-1)
            pairs = ((target_logits, h @ self.vectors.T / self.video_temperature),
                     (target_logits, r @ self.vectors.T / self.video_temperature))
        losses, entropies = [], []
        # Keep teacher and student distributions in float32 under AMP.
        for target_logits, logits in pairs:
            log_q = F.log_softmax(target_logits.float(), dim=-1)
            q = log_q.exp().detach()
            losses.append(F.kl_div(F.log_softmax(logits.float(), dim=-1), q,
                                   reduction="batchmean"))
            entropies.append(-(q * log_q).sum(-1).mean().detach())
        return sum(losses) / 2, count, float(sum(entropies) / 2), count

    def loss_weight(self, progress):
        ramp = min(1.0, progress / self.warmup_epochs) if self.warmup_epochs else 1.0
        return self.weight * ramp


def prepare_semantic_supervision(config, train_dataset, run_dir):
    """Snapshot only valid train teachers; never add val/test texts to the bank."""
    settings = config.get("loss", {}).get("semantic")
    if not settings:
        return None
    cfg = Path(config["dataset"]["root_dir"]).name.removeprefix("RH20T_")
    train_ids = {scene_key(cfg, sample.scene_id) for sample in subset_samples(train_dataset)}
    source = resolve_path(settings["teacher_path"])
    with np.load(source, allow_pickle=False) as data:
        ids = data["scene_ids"].tolist()
        vectors = data["vectors"].copy()
        texts = data["texts"].copy()
        usable = data["usable"] if "usable" in data else np.ones(len(ids), dtype=bool)
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    if (len(ids) != len(set(ids)) or vectors.ndim != 2
            or vectors.shape[0] != len(ids) or len(texts) != len(ids)
            or len(usable) != len(ids)):
        raise ValueError("Invalid scene teacher arrays or duplicate scene IDs")
    selected = sorted((index for index, identity in enumerate(ids)
                       if identity in train_ids and usable[index]), key=lambda i: ids[i])
    selected_ids = [ids[index] for index in selected]
    vectors = np.asarray(vectors[selected], dtype=np.float32)
    if (len(selected) < 2 or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) < 1e-8)):
        raise ValueError("Semantic training requires at least two finite, nonzero train teachers")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for text in texts[selected]:
        if not str(text).strip() or str(text).strip().lower() == "null":
            raise ValueError("Null or empty teacher text must be excluded before training")
    if settings["mode"] == "bank":
        if config["model"].get("semantic_dim") != vectors.shape[1]:
            raise ValueError("model.semantic_dim must equal the text teacher vector dimension")
    elif config["model"].get("semantic_dim", 0):
        raise ValueError("The batch method does not use a semantic projector")
    coverage = {
        "train_scenes": len(train_ids), "teacher_scenes": len(selected),
        "missing_scenes": sorted(train_ids - set(selected_ids)),
        "source_path": metadata.get("source_path", str(source)),
        "scene_job": metadata.get("scene_job"),
        "source_representation": metadata.get("source_representation"),
    }
    snapshot = Path(run_dir) / "semantic_teachers.npz"
    if snapshot.resolve() != source.resolve():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(snapshot, scene_ids=np.asarray(selected_ids),
                            vectors=vectors, texts=texts[selected],
                            metadata=np.asarray(json.dumps(coverage, ensure_ascii=False)))
    settings["teacher_path"] = str(snapshot.resolve())
    settings["coverage"] = coverage
    supervisor = SemanticSupervisor(selected_ids, vectors, settings)
    print(f"Semantic {settings['mode']}: {len(selected)}/{len(train_ids)} train scenes; "
          f"{len(coverage['missing_scenes'])} use video losses only", flush=True)
    return supervisor
