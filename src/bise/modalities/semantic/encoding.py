"""Incremental embeddings owned by each independent representation."""

import json
import time
from pathlib import Path

import numpy as np

from bise.modalities.semantic.embedder import build_text_embedder
from bise.modalities.semantic.representations import REPRESENTATIONS, representation_texts
from bise.modalities.semantic.store import RequestStore, file_lock, item_key, write_npz


def load_embeddings(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files if key != "metadata"}
        result["metadata"] = json.loads(str(data["metadata"]))
    return result


def encode(config: dict, directory: Path, records: list[dict],
           representations=None, embedder=None) -> tuple[Path, dict]:
    settings = config['embedding']
    annotations = RequestStore(directory).annotations()
    rows = {r['sample_id']: annotations[item_key(r)] for r in records if item_key(r) in annotations}
    if not rows:
        raise ValueError(f'No matching annotations in {directory}; run annotate first')
    available = representation_texts(next(iter(rows.values())), config['taxonomy'])
    names = [rep for rep in REPRESENTATIONS if rep in available]
    requested = representations or names
    if not set(requested) <= set(names):
        raise ValueError(f'{directory}: requested representations are not present')
    with file_lock(directory / '.lock'):
        files = sorted((directory / 'embeddings').glob('bge*.npz'),
                       key=lambda path: int(path.stem.removeprefix('bge')))
        previous, previous_path = None, None
        for path in reversed(files):
            candidate = load_embeddings(path)
            if (candidate['metadata']['config'] == settings and
                    candidate['metadata']['representations'] == names):
                previous, previous_path = candidate, path
                break
        ids = sorted(set(rows) | (set(previous['sample_ids'].tolist()) if previous else set()))
        positions = {sample: index for index, sample in enumerate(ids)}
        attempts, sources = [''] * len(ids), [''] * len(ids)
        ready = np.zeros((len(ids), len(names)), dtype=bool)
        lengths = np.zeros_like(ready, dtype=np.int32)
        matrices = {}
        if previous:
            matrices = {rep: np.zeros((len(ids), previous[rep].shape[1]), dtype=np.float32)
                        for rep in names}
            for old_index, sample in enumerate(previous['sample_ids'].tolist()):
                index = positions[sample]
                attempts[index] = str(previous['attempt_ids'][old_index])
                sources[index] = str(previous['source_keys'][old_index])
                ready[index] = previous['ready'][old_index]
                lengths[index] = previous['lengths'][old_index]
                for rep in names:
                    matrices[rep][index] = previous[rep][old_index]
        todo = []
        for sample, annotation in rows.items():
            index = positions[sample]
            if (attempts[index] != annotation['attempt_id'] or
                    sources[index] != item_key(annotation)):
                ready[index], lengths[index] = False, 0
            attempts[index], sources[index] = annotation['attempt_id'], item_key(annotation)
            texts = representation_texts(annotation, config['taxonomy'])
            for rep in requested:
                column = names.index(rep)
                if not ready[index, column]:
                    todo.append((index, column, rep, texts[rep]))
        started, load_seconds, peak_cuda_bytes = time.perf_counter(), 0.0, None
        if todo:
            load_start = time.perf_counter()
            embedder = embedder() if callable(embedder) else embedder
            embedder = embedder or build_text_embedder(settings)
            load_seconds = time.perf_counter() - load_start
            tokenizer = getattr(embedder, 'tokenizer', None)
            for index, column, rep, text in todo:
                length = len(tokenizer.encode(text)) if tokenizer is not None else len(text.split())
                lengths[index, column] = length
                if length > settings['max_length']:
                    raise ValueError(f'{ids[index]} {rep}: {length} tokens exceed max_length; '
                                     'increase embedding.max_length and re-encode')
            batch_size = settings['batch_size']
            for offset in range(0, len(todo), batch_size):
                batch = todo[offset:offset + batch_size]
                vectors = embedder.encode_texts([item[3] for item in batch])
                if not matrices:
                    matrices = {rep: np.zeros((len(ids), vectors.shape[1]), dtype=np.float32)
                                for rep in names}
                for (index, column, rep, _), vector in zip(batch, vectors):
                    matrices[rep][index], ready[index, column] = vector, True
                if offset == 0 or offset + batch_size >= len(todo) or offset % 160 == 0:
                    progress = min(offset + batch_size, len(todo))
                    print(f'Encode {directory.name}: {progress}/{len(todo)}', flush=True)
            torch = getattr(embedder, 'torch', None)
            if torch is not None and getattr(embedder, 'device', None).type == 'cuda':
                peak_cuda_bytes = torch.cuda.max_memory_allocated(embedder.device)
        stats = {'new_texts': len(todo), 'cache_texts': sum(
            bool(ready[positions[sample], names.index(rep)]) for sample in rows for rep in requested
        ) - len(todo), 'model_load_seconds': load_seconds,
            'encode_seconds': time.perf_counter() - started, 'peak_cuda_bytes': peak_cuda_bytes,
            'max_tokens': int(lengths.max()), 'truncated': 0}
        if not todo and previous is not None:
            return previous_path, stats
        revision = max([int(path.stem.removeprefix('bge')) for path in files], default=0) + 1
        path = directory / 'embeddings' / f'bge{revision}.npz'
        metadata = {'config': settings, 'revision': revision, 'stats': stats,
                    'representations': names}
        write_npz(path, sample_ids=np.asarray(ids, dtype=str),
                  attempt_ids=np.asarray(attempts, dtype=str),
                  source_keys=np.asarray(sources, dtype=str), ready=ready, lengths=lengths,
                  metadata=np.asarray(json.dumps(metadata)), **matrices)
        return path, stats


def matching_embeddings(config: dict, directory: Path) -> Path:
    files = sorted((directory / 'embeddings').glob('bge*.npz'),
                   key=lambda path: int(path.stem.removeprefix('bge')), reverse=True)
    for path in files:
        if load_embeddings(path)['metadata']['config'] == config['embedding']:
            return path
    raise FileNotFoundError(f'No compatible embeddings in {directory}; run --stage encode')
