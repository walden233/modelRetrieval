"""Offline scene-count comparisons with isolated annotation and vector snapshots."""

import copy
import csv
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from bise.common.paths import resolve_path
from bise.modalities.semantic.artifacts import aggregate, scope_records
from bise.modalities.semantic.encoding import encode, matching_embeddings
from bise.modalities.semantic.experiments import evaluate
from bise.modalities.semantic.figures import plot_all
from bise.modalities.semantic.store import RequestStore, item_key, utc_now, write_json
from bise.modalities.semantic.workflow import embedding_loader, preflight_embedding


def scene_subsets(records: list[dict], caps: list[str], seed: int) -> dict[str, list[dict]]:
    """Retain every task; smaller caps use nested prefixes of one scene ordering."""
    groups = defaultdict(set)
    for row in records:
        groups[row['task_id']].add(row['scene_id'])
    rng = random.Random(seed)
    ordered = {}
    for task, scenes in sorted(groups.items()):
        ordered[task] = sorted(scenes)
        rng.shuffle(ordered[task])
    subsets = {}
    for cap in caps:
        if cap != 'all' and (not cap.isdigit() or int(cap) < 1):
            raise ValueError('Scene caps must be all or positive integers')
        selected = {scene for scenes in ordered.values()
                    for scene in (scenes if cap == 'all' else scenes[:int(cap)])}
        subsets[cap] = [row for row in records if row['scene_id'] in selected]
    return subsets


def scope_counts(records: list[dict]) -> dict:
    groups = defaultdict(set)
    for row in records:
        groups[row['task_id']].add(row['scene_id'])
    return {'tasks': len(groups), 'scenes': sum(map(len, groups.values())),
            'videos': len(records), 'human': sum(r['role'] == 'human' for r in records),
            'robot': sum(r['role'] == 'robot' for r in records),
            'scenes_per_task': dict(sorted(Counter(map(len, groups.values())).items()))}


def evaluate_scenes(config: dict, output: Path, cfg: str = 'cfg2', split: str = 'all',
                    caps: list[str] | None = None, seed: int = 44,
                    representations: list[str] | None = None) -> dict:
    started = time.perf_counter()
    config = copy.deepcopy(config)
    config['evaluation']['seed'] = seed
    source_root, output = resolve_path(config['root']), resolve_path(output)
    if (output == source_root or source_root.is_relative_to(output) or
            (output.is_relative_to(source_root) and
             not output.is_relative_to(source_root / 'scene_eval'))):
        raise ValueError('Use a new output outside existing results or under root/scene_eval')
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new --output: {output}')
    if split not in ('all', 'train', 'val', 'test'):
        raise ValueError(f'Unknown split: {split}')
    if cfg not in config['cfgs']:
        raise ValueError(f'Unknown cfg: {cfg}')
    reps = list(dict.fromkeys(representations or config['representations']))
    if not reps or not set(reps) <= set(config['representations']):
        raise ValueError('Choose representations from config.representations')
    records, scope = scope_records(config, [cfg])
    records = [row for row in records if split == 'all' or row['split'] == split]
    if not records:
        raise ValueError(f'No records for {cfg}/{split}')
    subsets = scene_subsets(records, caps or ['all', '3', '2', '1'], seed)
    needed = {row['sample_id'] for rows in subsets.values() for row in rows}
    encoding_records = [row for row in records if row['sample_id'] in needed]
    snapshots, sources = {}, {}
    reference = None
    for rep in reps:
        directory = source_root / rep
        settings = json.loads((directory / 'params.json').read_text())
        controls = {key: value for key, value in settings['signature'].items()
                    if key != 'representation'}
        if reference is not None and controls != reference:
            raise ValueError(f'{rep}: extraction protocol differs across representations')
        reference = controls
        annotations = RequestStore(directory).annotations()
        missing = [row['sample_id'] for row in encoding_records if item_key(row) not in annotations]
        if missing:
            raise ValueError(f'{rep}: {len(missing)} videos lack current annotations; '
                             f'no API calls will be made. First IDs: {missing[:3]}')
        snapshots[rep] = (settings, [annotations[item_key(row)] for row in encoding_records])
        try:
            embedding = matching_embeddings(config, directory)
        except FileNotFoundError:
            embedding = None
        sources[rep] = {'directory': str(directory),
                        'embedding': str(embedding) if embedding else None,
                        'job': settings['job']}
    if (config['embedding']['provider_name'] == 'transformers' and
            not config['embedding'].get('local_files_only')):
        raise ValueError('Offline evaluation requires embedding.local_files_only=true')
    preflight_embedding(config)
    output.mkdir(parents=True)
    write_json(output / 'params.json', {
        'created_utc': utc_now(), 'cfg': cfg, 'split': split, 'seed': seed,
        'caps': list(subsets), 'representations': reps, 'sources': sources,
        'embedding': config['embedding'], 'scope_source': scope['sources'][cfg],
        'selection': 'at most N scenes per task; nested selection; retain all camera pairs',
        'baseline': scope_counts(records), 'api_calls': 0})
    load_embedder = embedding_loader(config)
    vectors, encoding = {}, {}
    for rep in reps:
        print(f'Scene evaluation: prepare isolated {rep} vectors', flush=True)
        cache = output / 'cache' / rep
        settings, annotations = snapshots[rep]
        write_json(cache / 'params.json', settings)
        (cache / 'annotations.jsonl').write_text(''.join(
            json.dumps(row, ensure_ascii=False) + '\n' for row in annotations))
        if sources[rep]['embedding']:
            original = Path(sources[rep]['embedding'])
            (cache / 'embeddings').mkdir()
            shutil.copy2(original, cache / 'embeddings' / original.name)
        vectors[rep], encoding[rep] = encode(config, cache, encoding_records, embedder=load_embedder)
    summary = {'cfg': cfg, 'split': split, 'seed': seed, 'representations': reps,
               'api_calls': 0, 'baseline': scope_counts(records), 'variants': [],
               'encoding': encoding}
    csv_rows = []
    for cap, selected in subsets.items():
        variant = output / ('all' if cap == 'all' else 's' + cap)
        counts = scope_counts(selected)
        print(f'Scene evaluation {cap}: {counts}', flush=True)
        write_json(variant / 'selection.json', {
            'cap': cap, 'seed': seed, **counts,
            'scene_ids': sorted({r['scene_id'] for r in selected})})
        for rep in reps:
            evaluate(config, output / 'cache' / rep, vectors[rep], selected,
                     variant / rep / 'evaluations' / cfg, cfg, [rep], split=split)
        payload = aggregate(variant, reps, [cfg])[cfg]
        if payload['excluded']:
            raise ValueError(f'Incompatible results: {payload["excluded"]}')
        summary['variants'].append({'cap': cap, **counts, 'metrics': payload['metrics'],
                                    'source': str((variant / 'evaluations' / cfg /
                                                   'metrics.json').relative_to(output))})
        for direction, levels in payload['metrics'].items():
            for level, values in levels.items():
                for rep, metrics in values.items():
                    for metric, value in metrics.items():
                        csv_rows.append({'cap': cap, 'cfg': cfg, 'split': split,
                                         'tasks': counts['tasks'], 'scenes': counts['scenes'],
                                         'videos': counts['videos'], 'direction': direction,
                                         'level': level, 'representation': rep,
                                         'metric': metric, 'value': value})
        plot_all(variant)
    summary['seconds'] = time.perf_counter() - started
    write_json(output / 'summary.json', summary)
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [f'# {cfg}/{split}: scene-count evaluation', '',
             'Each cap retains at most N scenes per task, including both camera pairs. '
             'Subsets are nested with a fixed seed. Results use existing annotations; '
             'missing vectors are encoded offline in this output only. '
             'MPR-all is in raw data and is omitted from figures.', '',
             '| cap | tasks | scenes | videos | representation | task R@1 | scene R@1 | '
             'task MRR | scene MRR |',
             '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for variant in summary['variants']:
        for rep in reps:
            values = variant['metrics']['human_to_robot']
            task, scene = values['task'][rep], values['scene'][rep]
            lines.append(f"| {variant['cap']} | {variant['tasks']} | {variant['scenes']} | "
                         f"{variant['videos']} | {rep} | {task['R@1']:.2%} | "
                         f"{scene['R@1']:.2%} | {task['MRR']:.4f} | {scene['MRR']:.4f} |")
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return summary
