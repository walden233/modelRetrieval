"""Centralized layout, frozen scope, and independent representation projections."""

import json
import shutil
from pathlib import Path

from bise.common.paths import resolve_path
from bise.modalities.semantic.manifest import expand_split, relative_path, video_record
from bise.modalities.semantic.representations import REPRESENTATIONS, representation_texts
from bise.modalities.semantic.store import RequestStore, item_key, read_jsonl, write_json


def migrate(root: Path, sources: dict) -> None:
    """Archive earlier protocol artifacts without converting them into new results."""
    if not root.is_dir():
        return
    moves = []
    for path in root.iterdir():
        params = path / 'params.json'
        if path.is_dir() and params.is_file():
            data = json.loads(params.read_text())
            if data.get('signature', {}).get('schema_version') == 2:
                moves.append((path, root / 'history/jobs' / path.name))
    batches = {resolve_path(path).parents[1].name for path in sources.values()}
    for batch in batches:
        old = resolve_path('artifacts/runs/retrain') / batch / 'semantic'
        if old.is_dir():
            moves.append((old, root / 'history' / batch))
    for old, new in moves:
        if new.exists():
            raise ValueError(f'Migration destination already exists: {new}; '
                             f'original retained: {old}')
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
    if not moves:
        return
    replacements = {relative_path(old): relative_path(new) for old, new in moves}
    # Only rewrite paths in archived generated JSON, never source manifests or datasets.
    for path in (root / 'history').rglob('*'):
        if path.suffix not in ('.json', '.jsonl', '.html', '.csv'):
            continue
        text = path.read_text(encoding='utf-8-sig')
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding='utf-8')
    for batch in batches:
        path = resolve_path('artifacts/runs/retrain') / batch / 'plots.json'
        if path.exists():
            plots = json.loads(path.read_text())
            if 'plots' not in plots:
                write_json(path, {key: value for key, value in plots.items()
                                  if not key.startswith('semantic/')})
    legacy = next((root / 'history' / batch / 'scope.json' for batch in sorted(batches)
                   if (root / 'history' / batch / 'scope.json').exists()), None)
    if legacy and not (root / 'scope.json').exists():
        data = json.loads(legacy.read_text())
        data.pop('batch', None)
        data['schema_version'] = 3
        write_json(root / 'scope.json', data)


def scope_records(config: dict, cfgs: list[str]) -> tuple[list[dict], dict]:
    root = resolve_path(config['root'])
    path = root / 'scope.json'
    scope = json.loads(path.read_text()) if path.exists() else {
        'schema_version': 3, 'sources': {}, 'extras': []}
    for cfg in config['cfgs']:
        if cfg not in scope['sources']:
            _, source = expand_split(config['sources'][cfg], cfg)
            scope['sources'][cfg] = source
    rows = []
    for cfg in cfgs:
        source = scope['sources'][cfg]
        seen, scene_splits = set(), {}
        for split in ('train', 'val', 'test'):
            for pair in source['splits'][split]:
                prefix, task, scene, camera = pair.split('::')
                if prefix != 'rh20t' or pair in seen:
                    raise ValueError(f'Invalid/duplicate frozen pair: {pair}')
                if (task, scene) in scene_splits and scene_splits[task, scene] != split:
                    raise ValueError(f'Scene crosses frozen splits: {task}/{scene}')
                seen.add(pair)
                scene_splits[task, scene] = split
                for role in ('human', 'robot'):
                    rows.append(video_record(cfg, resolve_path(source['data_root']),
                                             task, scene, camera, role, split))
    return rows, scope


def materialize(config: dict, job: Path, records: list[dict], representations: list[str],
                parent: Path | None = None) -> dict:
    """Each representation owns its task text and can outlive any sibling directory."""
    root = parent or resolve_path(config['root'])
    accepted = RequestStore(job).annotations()
    counts = {}
    for rep in representations:
        directory = root / rep
        directory.mkdir(parents=True, exist_ok=True)
        params = {'schema_version': 3, 'representation': rep, 'job': relative_path(job),
                  'signature': json.loads((job / 'params.json').read_text())['signature']}
        existing_params = directory / 'params.json'
        if existing_params.exists() and json.loads(existing_params.read_text()) != params:
            # A changed protocol must not leave an old representation's evaluation active.
            number = 1
            archive = root / 'history' / rep
            while archive.exists():
                number += 1
                archive = root / 'history' / f'{rep}-r{number}'
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(directory), str(archive))
            directory.mkdir()
            teachers = root / 'teachers' / rep
            if teachers.exists():
                teacher_archive = archive.with_name(archive.name + '-teachers')
                shutil.move(str(teachers), str(teacher_archive))
            if rep == 'S5':
                for legacy in (root / 'teachers').glob('*.npz'):
                    teacher_archive = archive.with_name(archive.name + '-teachers')
                    teacher_archive.mkdir(exist_ok=True)
                    shutil.move(str(legacy), str(teacher_archive / legacy.name))
        write_json(directory / 'params.json', params)
        store = RequestStore(directory)
        old = store.annotations()
        count = 0
        for record in records:
            annotation = accepted.get(item_key(record))
            if annotation is None:
                continue
            text = representation_texts(annotation, config['taxonomy'])[rep]
            data = {'representation': rep, 'text': text, 'job': relative_path(job)}
            if rep == 'S2':
                data['labels'] = annotation['labels']
            if rep == 'S5':
                data['structured'] = annotation['structured']
            previous = old.get(item_key(record))
            if previous is None or previous['attempt_id'] != annotation['attempt_id']:
                store.accept(record, data, annotation['attempt_id'], annotation.get('frames'))
            count += 1
        counts[rep] = count
    return counts


def active_representations(config: dict, root: Path, requested=None) -> list[str]:
    allowed = requested if requested is not None else config['representations']
    return [rep for rep in REPRESENTATIONS if rep in allowed and
            (root / rep / 'annotations.jsonl').exists()]


def aggregate(root: Path, representations: list[str], cfgs: list[str]) -> dict:
    """Merge compatible independent metrics; a removed sibling is never recovered."""
    params = root / 'params.json'
    if params.exists():
        settings = json.loads(params.read_text())
        settings['comparison_representations'] = representations
        write_json(params, settings)
    result = {}
    for cfg in cfgs:
        payloads = {}
        for rep in representations:
            path = root / rep / 'evaluations' / cfg / 'metrics.json'
            if path.exists():
                payloads[rep] = json.loads(path.read_text())
        output = root / 'evaluations' / cfg
        if not payloads:
            if output.exists():
                shutil.rmtree(output)
            continue
        reference = max(payloads.values(), key=lambda value: (
            len(value['protocol']['ids']), value.get('evaluated_utc', '')))
        def comparable(value):
            protocol = value['protocol']
            return {key: protocol[key] for key in (
                'ids', 'embedding', 'split', 'seed', 'extraction_controls')}
        mismatches = [rep for rep, value in payloads.items()
                      if comparable(value) != comparable(reference)]
        joint = [value for rep, value in payloads.items()
                 if rep not in mismatches and value['protocol']['joint_prompt'] is not None]
        if joint:
            latest_joint = max(joint, key=lambda value: value.get('evaluated_utc', ''))
            joint_prompt = latest_joint['protocol']['joint_prompt']
            mismatches.extend(rep for rep, value in payloads.items()
                              if rep not in mismatches and value['protocol']['joint_prompt'] is not None
                              and value['protocol']['joint_prompt'] != joint_prompt)
        excluded = {rep: 'different evaluation corpus or protocol' for rep in mismatches}
        payloads = {rep: value for rep, value in payloads.items() if rep not in mismatches}
        metrics, rankings = {}, []
        for rep, value in payloads.items():
            for direction, levels in value['metrics'].items():
                for level, values in levels.items():
                    metrics.setdefault(direction, {}).setdefault(level, {})[rep] = values[rep]
            rankings.extend(read_jsonl(root / rep / 'evaluations' / cfg / 'rankings.jsonl'))
        payload = {**reference, 'job': 'comparison', 'metrics': metrics,
                   'representations': list(payloads), 'excluded': excluded, 'sources': {
                       rep: relative_path(root / rep / 'evaluations' / cfg / 'metrics.json')
                       for rep in payloads}}
        write_json(output / 'metrics.json', payload)
        temporary = output / 'rankings.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(row) + '\n' for row in rankings))
        temporary.replace(output / 'rankings.jsonl')
        result[cfg] = payload
    return result
