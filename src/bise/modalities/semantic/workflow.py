"""Centralized semantic experiments with independent description and label encodings."""

import copy
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bise.common.paths import resolve_path
from bise.modalities.semantic.annotation import annotate, make_job, provider_key
from bise.modalities.semantic.artifacts import (
    active_representations, aggregate, materialize, migrate, scope_records,
)
from bise.modalities.semantic.embedder import build_text_embedder
from bise.modalities.semantic.encoding import encode, load_embeddings, matching_embeddings
from bise.modalities.semantic.experiments import (
    analyze, evaluate, scope_statistics, select_available,
)
from bise.modalities.semantic.figures import plot_all
from bise.modalities.semantic.manifest import (
    balanced_limit, expand_split, explicit_videos, relative_path, sample_cost,
)
from bise.modalities.semantic.report import export_report
from bise.modalities.semantic.representations import REPRESENTATIONS, validate_scene
from bise.modalities.semantic.store import (
    RequestStore, file_lock, item_key, read_jsonl, utc_now, write_json, write_npz,
)


def load_config(path: str | Path) -> dict:
    config = json.loads(resolve_path(path).read_text())
    if config.get('schema_version') != 3:
        raise ValueError('Semantic config requires schema_version=3')
    if not config['representations'] or not set(config['representations']) <= set(REPRESENTATIONS):
        raise ValueError('Configure representations using ' + ','.join(REPRESENTATIONS))
    for provider in config['providers'].values():
        if 'api_key' in provider:
            raise ValueError('Use api_key_env, not a literal API key')
    return config


def update_params(directory: Path, **values) -> None:
    path = directory / 'params.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(values)
    write_json(path, data)


def annotation_records(config: dict, records: list[dict]) -> list[dict]:
    splits = config['annotation']['splits']
    return [row for row in records if row['split'] in splits[row['cfg']]]


def prepare(config: dict, cfgs: list[str]) -> tuple[Path, list[dict], dict]:
    root = resolve_path(config['root'])
    migrate(root, config['sources'])
    path = root / 'params.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if any(rep not in REPRESENTATIONS for rep in previous.get('comparison_representations', [])):
        update_params(root, comparison_representations=config['representations'])
    records, scope = scope_records(config, config['cfgs'])
    cost_settings = {key: config['cost'][key] for key in ('cfg', 'split', 'tasks', 'seed')}
    frozen = scope.get('cost')
    previous_settings = ({key: frozen.get(key, len(frozen.get('scene_ids', []))
                                         if key == 'tasks' else None)
                          for key in cost_settings} if frozen else None)
    if previous_settings != cost_settings:
        cost = sample_cost(records, config)
        old_paths = [root / 'cost', root / 'charts/cost.json', root / 'charts/cost.png']
        if frozen and any(item.exists() for item in old_paths):
            archive = root / 'history' / f'cost-{time.time_ns()}'
            write_json(archive / 'scope.json', frozen)
            for item in old_paths:
                if item.exists():
                    destination = archive / item.relative_to(root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(destination))
            plots = root / 'plots.json'
            if plots.exists():
                index = json.loads(plots.read_text())
                index.pop('charts/cost.png', None)
                write_json(plots, index)
            print(f'Archived cost results for previous scope: {archive}', flush=True)
        scope['cost'] = {**cost_settings,
                         'scene_ids': list(dict.fromkeys(row['scene_id'] for row in cost)),
                         'sample_ids': [row['sample_id'] for row in cost]}
    write_json(root / 'scope.json', scope)
    with file_lock(root / '.inventory.lock'):
        inventory = {row['sample_id']: row for row in read_jsonl(root / 'videos.jsonl')}
        for row in records:
            inventory[row['sample_id']] = {
                key: value for key, value in row.items() if key != 'split'}
        temporary = root / 'videos.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(inventory[key]) + '\n'
                                     for key in sorted(inventory)))
        temporary.replace(root / 'videos.jsonl')
    update_params(root, schema_version=3, config=config, prepared_utc=utc_now(),
                  annotation_target_videos=len(annotation_records(config, records)),
                  counts={cfg: {split: sum(row['cfg'] == cfg and row['split'] == split
                                          for row in records) for split in ('train', 'val', 'test')}
                          for cfg in config['cfgs']})
    return root, [row for row in records if row['cfg'] in cfgs], scope


def preflight_embedding(config: dict) -> None:
    settings = config['embedding']
    if settings['provider_name'] != 'transformers' or not settings.get('local_files_only'):
        return
    from huggingface_hub import try_to_load_from_cache
    model, revision = settings['model_name'], settings.get('revision')
    for filename in ('config.json', 'tokenizer_config.json'):
        if not isinstance(try_to_load_from_cache(model, filename, revision=revision), str):
            raise FileNotFoundError(f'Offline embedding cache lacks {filename} for {model}')
    weights = ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json')
    if not any(isinstance(try_to_load_from_cache(model, filename, revision=revision), str)
               for filename in weights):
        raise FileNotFoundError(f'Offline embedding weights missing for {model}')


def embedding_loader(config: dict):
    """Share one model across groups; cached vectors need no model load."""
    embedder = None
    def load():
        nonlocal embedder
        if embedder is None:
            embedder = build_text_embedder(config['embedding'])
        return embedder
    return load


def evaluation_scope(config: dict, records: list[dict], cfgs: list[str],
                     override: str | None = None) -> tuple[list[dict], dict[str, str]]:
    setting = config['evaluation']['split']
    splits = {cfg: override or (setting.get(cfg, 'test') if isinstance(setting, dict) else setting)
              for cfg in cfgs}
    if any(split not in ('all', 'train', 'val', 'test') for split in splits.values()):
        raise ValueError('Evaluation split must be all, train, val, or test')
    return [row for row in records if row['cfg'] in splits and
            (splits[row['cfg']] == 'all' or row['split'] == splits[row['cfg']])], splits


def annotate_groups(config: dict, root: Path, selected: list[dict], records: list[dict],
                    representations: list[str], args) -> dict:
    provider, frames = config['annotation']['provider'], config['annotation']['frames']
    job, summary = annotate(config, selected, provider, frames,
                            force=args.force, retry_unknown=args.retry_unknown,
                            allow_failures=True, retry_failed=True)
    counts = materialize(config, job, records, representations, root)
    return {'video': {'job': relative_path(job), 'materialized': counts,
                      'selection': summary['selection'],
                      'engineering': scope_statistics(job, selected),
                      'last_execution': summary['executions'][-1]}}


def evaluate_groups(config: dict, root: Path, records: list[dict], representations: list[str],
                    cfgs: list[str], split: str, allow_partial: bool) -> dict:
    files, available = {}, []
    for rep in representations:
        directory = root / rep
        files[rep] = matching_embeddings(config, directory)
        rows, _ = select_available(records, load_embeddings(files[rep]), directory,
                                   [rep], allow_partial)
        available.append({row['sample_id'] for row in rows})
    common = set.intersection(*available) if available else set()
    selected = [row for row in records if row['sample_id'] in common]
    results = {}
    for rep in representations:
        for cfg in cfgs:
            target = [row for row in records if row['cfg'] == cfg]
            rows = [row for row in selected if row['cfg'] == cfg]
            output = root / rep / 'evaluations' / cfg
            payload = evaluate(config, root / rep, files[rep], rows, output, cfg,
                               representations=[rep], split=split)
            payload['coverage'] = {'expected': len(target), 'present': len(rows),
                                   'missing_ids': [row['sample_id'] for row in target
                                                   if row['sample_id'] not in common]}
            payload['partial'] = len(rows) != len(target)
            write_json(output / 'metrics.json', payload)
            if payload['partial']:
                print(f'{rep} {cfg}: partial evaluation {len(rows)}/{len(target)} videos; '
                      'all selected representations use the same available set.', flush=True)
            results.setdefault(rep, {})[cfg] = payload
    return results


def collect_analysis(config: dict, root: Path, representations: list[str]) -> dict:
    payload = {'schema_version': 3, 'representations': representations, 'views': [],
               'examples': {}, 'labels': {}, 'failure_candidates': {}, 'missing': []}
    review_path = root / 'reviews.csv'
    old_reviews = {}
    if review_path.exists():
        with review_path.open(encoding='utf-8-sig', newline='') as handle:
            old_reviews = {(row['representation'], row['sample_id']): row
                           for row in csv.DictReader(handle)}
    reviews = []
    for rep in representations:
        path = root / rep / 'analysis.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        payload.setdefault('splits', {}).update(data.get('splits', {}))
        payload['views'].extend(data['views'])
        payload['missing'].extend(data['missing'])
        if data['labels']['distributions'] and rep in ('S2c', 'S2w'):
            payload['labels'] = data['labels']
        for key, example in data['examples'].items():
            existing = payload['examples'].setdefault(key, {**example, 'texts': {}})
            if existing['sample_id'] == example['sample_id']:
                existing['texts'].update(example['texts'])
            else:
                payload['examples'][f'{key}_{rep}'] = example
        for query, cases in data['failure_candidates'].items():
            payload['failure_candidates'].setdefault(query, []).extend(cases)
        if 'multimodal' in data:
            payload.setdefault('multimodal', {})[rep] = data['multimodal']
        with (root / rep / 'reviews.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                key = (rep, row['sample_id'])
                reviews.append(old_reviews.get(key, {'representation': rep, **row}))
    indexed = {(row['direction'], row['level'], row['query_id'], row['representation']): row
               for path in (root / 'evaluations').glob('*/rankings.jsonl')
               for row in read_jsonl(path)}
    payload['rank_changes'] = []
    for key, row in indexed.items():
        baseline = indexed.get((*key[:-1], 'S1'))
        if key[-1] in ('S2c', 'S2w') and baseline:
            payload['rank_changes'].append({'query_id': row['query_id'], 'level': row['level'],
                                            'representation': key[-1],
                                            'S1': baseline['best_rank'], key[-1]: row['best_rank']})
    if reviews:
        with review_path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reviews[0]))
            writer.writeheader()
            writer.writerows(reviews)
    write_json(root / 'analysis.json', payload)
    return payload


def _repair_scene_cache(config: dict, root: Path, records: list[dict], extra: list[str]) -> list[dict]:
    job = make_job(config, config['scene']['provider'], 0, kind='scene')
    signature = json.loads((job / 'params.json').read_text())['signature']
    comparable = lambda sig: {k: v for k, v in sig.items()
                              if k not in ('prompt', 'validation_policy')}
    previous = []
    for path in sorted((root / 'jobs').glob('scene-*'),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if path == job or not (path / 'params.json').exists():
            continue
        old = json.loads((path / 'params.json').read_text())['signature']
        if comparable(old) == comparable(signature):
            previous.append((path, RequestStore(path).annotations()))
    unknown = set(extra) - {row['scene_id'] for row in records}
    if unknown:
        raise ValueError(f'Repair scene IDs outside selected scope: {sorted(unknown)}')
    pending, reused = [], 0
    store = RequestStore(job)
    with file_lock(job / '.lock'):
        current = store.annotations()
        for record in records:
            key = item_key(record)
            origin, row = job, current.get(key)
            if row is None:
                for path, annotations in previous:
                    if key in annotations:
                        origin, row = path, annotations[key]
                        break
            text = None
            if row is not None and row.get('observations') == record['observations']:
                try:
                    text = validate_scene({'description': row['description']},
                                          record['observations'])['description']
                except ValueError:
                    pass
            if not text or record['scene_id'] in extra:
                pending.append(record)
            elif key not in current:
                provenance = row.get('reused_from', {
                    'job': relative_path(origin), 'attempt_id': row['attempt_id']})
                store.accept(record, {'description': text, 'reused_from': provenance},
                             row['attempt_id'])
                store.event({'event': 'reused', 'item_key': key, **provenance})
                reused += 1
    print(f'Scene repair: {len(pending)} regenerate, {reused} reused, '
          f'{len(records) - len(pending) - reused} current cache.', flush=True)
    return pending


def export_scene(config: dict, root: Path, records: list[dict], scope: dict, args) -> dict:
    source_rep = args.source or config['scene']['source']
    if source_rep not in REPRESENTATIONS:
        raise ValueError('Scene source must be one of ' + ','.join(REPRESENTATIONS))
    config = copy.deepcopy(config)
    config['scene']['source'] = source_rep
    annotations = RequestStore(root / source_rep).annotations()
    selected = annotation_records(config, records)
    if args.run:
        mapping = json.loads((resolve_path(args.run) / 'split_manifest.json').read_text())
        pairs = set(mapping['train'])
        selected = [row for row in records
                    if row['cfg'] == 'cfg2' and row['source_pair_id'] in pairs]
        if {row['source_pair_id'] for row in selected} != pairs:
            raise ValueError('Run train split is outside the frozen semantic scope')
    groups = defaultdict(list)
    for row in selected:
        groups[row['scene_id']].append(row)
    scenes = sorted(groups)
    if args.limit is not None:
        scenes = scenes[:args.limit]
    scene_records = []
    for scene in scenes:
        members = sorted(groups[scene], key=lambda row: (row['role'], row['camera_id']))
        if len(members) != 4 or {row['role'] for row in members} != {'human', 'robot'}:
            raise ValueError(f'Scene teacher needs exactly two frozen camera pairs: {scene}')
        observations, sources = {}, []
        for row in members:
            annotation = annotations.get(item_key(row))
            if annotation is None:
                raise ValueError(f'Missing {source_rep} task text: {row["sample_id"]}; '
                                 f'run --stage annotate --cfg {row["cfg"]} '
                                 f'--split {row["split"]} --only {source_rep}')
            letter = 'H' if row['role'] == 'human' else 'R'
            number = sum(value['role'] == row['role'] for value in observations.values()) + 1
            local_id = f'{letter}_{number}'
            observations[local_id] = {'role': row['role'], 'text': annotation['text']}
            sources.append({'sample_id': row['sample_id'], 'attempt_id': annotation['attempt_id'],
                            'source': row['source']})
        scene_records.append({'sample_id': scene, 'scene_id': scene,
                              'source': {'representation': source_rep, 'observations': sources},
                              'observations': observations})
    repair = getattr(args, 'repair', False)
    pending = (_repair_scene_cache(config, root, scene_records, getattr(args, 'scene_ids', None) or [])
               if repair else scene_records)
    job, summary = annotate(config, pending, config['scene']['provider'], 0, kind='scene',
                            force=args.force or repair, retry_unknown=args.retry_unknown,
                            retry_failed=True)
    accepted = RequestStore(job).annotations()
    texts = []
    for row in scene_records:
        try:
            text = validate_scene({'description': accepted[item_key(row)]['description']},
                                  row['observations'])['description']
        except (KeyError, ValueError):
            text = None
        texts.append(text)
    usable = [bool(text) for text in texts]
    nonempty = [text for text in texts if text]
    preflight_embedding(config)
    embedder = build_text_embedder(config['embedding'])
    if hasattr(embedder, 'tokenizer') and any(
            len(embedder.tokenizer.encode(text)) > config['embedding']['max_length']
            for text in nonempty):
        raise ValueError('Scene text exceeds embedding.max_length; increase it and rerun scene')
    batches = [embedder.encode_texts(nonempty[i:i + config['embedding']['batch_size']])
               for i in range(0, len(nonempty), config['embedding']['batch_size'])]
    encoded = np.concatenate(batches) if batches else np.zeros((0, 0), dtype=np.float32)
    vectors = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
    if usable:
        vectors[np.asarray(usable)] = encoded
    files = []
    for cfg in sorted({scene.split('/')[0] for scene in scenes}):
        cfg_indices = [i for i, scene in enumerate(scenes) if scene.startswith(cfg + '/')]
        indices = [i for i in cfg_indices if usable[i]]
        name = cfg
        if args.run:
            run = resolve_path(args.run)
            name = f'{run.parents[1].name}-{run.parent.name}-{run.name}'
        path = root / 'teachers' / source_rep / f'{name}.npz'
        metadata = {'scene_job': relative_path(job), 'run': args.run, 'cfg': cfg,
                    'split': 'train' if args.run else 'all', 'source_representation': source_rep,
                    'partial': len(scenes) != len(groups), 'target_scenes': len(groups),
                    'usable_scenes': len(indices),
                    'excluded_scenes': [scenes[i] for i in cfg_indices if not usable[i]],
                    'embedding': config['embedding'], 'scope': scope['sources'][cfg]}
        write_npz(path, scene_ids=np.asarray([scenes[i] for i in indices]),
                  texts=np.asarray([texts[i] or '' for i in indices]), vectors=vectors[indices],
                  usable=np.asarray([usable[i] for i in indices], dtype=bool),
                  sources=np.asarray([json.dumps(scene_records[i]['source']) for i in indices]),
                  metadata=np.asarray(json.dumps(metadata)))
        files.append(relative_path(path))
    return {'files': files, 'scenes': len(scenes), 'usable': sum(usable), 'engineering': summary}


def run_cost(config: dict, root: Path, records: list[dict], scope: dict,
             representations: list[str], args) -> dict:
    providers = args.providers or config['cost']['providers']
    main = config['annotation']['provider']
    enabled = set(providers) if args.providers is not None else {main, *providers}
    if not enabled <= config['providers'].keys():
        raise ValueError('Unknown cost provider selection')
    comparison_providers = list(dict.fromkeys([*config['cost']['providers'], *providers]))
    plans = list(dict.fromkeys(
        [(main, frames) for frames in config['cost']['frames']] +
        [(provider, config['cost']['provider_frames']) for provider in comparison_providers]))
    for provider in enabled:
        provider_key(config['providers'][provider])
    preflight_embedding(config)
    indexed = {row['sample_id']: row for row in records}
    if not set(scope['cost']['sample_ids']) <= indexed.keys():
        raise ValueError('Cost stage requires the complete frozen cfg2 scope')
    target = [indexed[sample] for sample in scope['cost']['sample_ids']]
    selected = balanced_limit(target, args.limit)
    variants, failures, skipped = [], [], []
    load_embedder = embedding_loader(config)
    # Use the same six-output extraction protocol for every cost variant.
    reps = representations
    for provider, frames in plans:
        job = make_job(config, provider, frames)
        output = root / 'cost' / job.name
        if provider not in enabled:
            accepted = RequestStore(job).annotations()
            missing = sum(item_key(row) not in accepted for row in target)
            if missing:
                skipped.append({'job': job.name, 'missing': missing,
                                'reason': 'provider not selected; incomplete cache'})
                print(f'Skip {job.name}: API disabled, {missing} videos missing from cache.',
                      flush=True)
                continue
            print(f'Use {job.name} from cache; API calls disabled.', flush=True)
        try:
            if provider in enabled:
                annotate(config, selected, provider, frames,
                         force=args.force or args.fresh_benchmark, retry_unknown=args.retry_unknown,
                         allow_failures=True, retry_failed=True)
            materialize(config, job, target, reps, output)
            for rep in reps:
                encode(config, output / rep, target, embedder=load_embedder)
            variants.append((provider, frames, job, output))
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            failures.append({'job': job.name, 'error': str(exc)})
    if failures:
        write_json(root / 'cost/failures.json', failures)
        raise RuntimeError('Cost variants incomplete; successful requests are saved. Fix/retry: ' +
                           '; '.join(row['job'] for row in failures))
    available = []
    for _, _, _, output in variants:
        for rep in reps:
            vectors = load_embeddings(matching_embeddings(config, output / rep))
            rows, _ = select_available(target, vectors, output / rep, [rep], True)
            available.append({row['sample_id'] for row in rows})
    common = set.intersection(*available)
    if len(common) != len(target):
        print(f'Cost comparison: using {len(common)}/{len(target)} common videos; '
              'missing annotations remain available for retry.', flush=True)
    rows = [row for row in target if row['sample_id'] in common]
    summary = {'cfg': config['cost']['cfg'], 'split': config['cost']['split'],
               'tasks': config['cost']['tasks'], 'target_videos': len(target),
               'common_videos': len(rows), 'sample_ids': [row['sample_id'] for row in rows],
               'representations': reps, 'task_scene_positives_identical': True,
               'annotation_providers': sorted(enabled), 'skipped_variants': skipped,
               'variants': []}
    for provider, frames, job, output in variants:
        evaluated = evaluate_groups(config, output, rows, reps, [config['cost']['cfg']],
                                    config['cost']['split'], False)
        for rep, cfg_payloads in evaluated.items():
            value = cfg_payloads[config['cost']['cfg']]
            value['partial'] = len(rows) != len(target)
            value['coverage'] = {'expected': len(target), 'present': len(rows),
                                 'missing_ids': [row['sample_id'] for row in target
                                                 if row['sample_id'] not in common]}
            write_json(output / rep / 'evaluations' / config['cost']['cfg'] / 'metrics.json', value)
        payload = aggregate(output, reps, [config['cost']['cfg']])[config['cost']['cfg']]
        summary['variants'].append({'name': job.name, 'provider': provider, 'frames': frames,
                                    'annotation_mode': 'resume' if provider in enabled else 'cache_only',
                                    'model': config['providers'][provider]['model'],
                                    'partial': len(rows) != len(target),
                                    'metrics': payload['metrics'],
                                    'engineering': scope_statistics(job, selected)})
    write_json(root / 'cost/summary.json', summary)
    return summary


def _run(config: dict, args) -> dict:
    cfgs = args.cfg or config['cfgs']
    if not set(cfgs) <= set(config['cfgs']):
        raise ValueError('Unknown cfg selection')
    if args.provider:
        config['annotation']['provider'] = args.provider
    if args.frames is not None:
        config['annotation']['frames'] = args.frames
    if args.concurrency is not None:
        config['annotation']['concurrency'] = args.concurrency
        for provider in config['providers'].values():
            provider['concurrency'] = args.concurrency
    if config['annotation']['frames'] <= 0 or config['annotation']['concurrency'] <= 0:
        raise ValueError('frames and concurrency must be positive')
    root, records, scope = prepare(config, cfgs)
    params = json.loads((root / 'params.json').read_text())
    disabled = set(params.get('disabled', []))
    requested = args.only or [rep for rep in config['representations'] if rep not in disabled]
    extras = []
    if args.videos:
        extras.extend(explicit_videos(args.videos, scope['sources']))
    for path in args.add_manifest or []:
        extras.extend(expand_split(path)[0])
    if extras:
        scope['extras'] = list({row['sample_id']: row for row in scope['extras'] + extras}.values())
        write_json(root / 'scope.json', scope)
        inventory = {row['sample_id']: row for row in read_jsonl(root / 'videos.jsonl')}
        for row in extras:
            inventory[row['sample_id']] = {
                key: value for key, value in row.items() if key != 'split'}
        with file_lock(root / '.inventory.lock'):
            temporary = root / 'videos.jsonl.tmp'
            temporary.write_text(''.join(json.dumps(inventory[key]) + '\n'
                                         for key in sorted(inventory)))
            temporary.replace(root / 'videos.jsonl')
    all_records = list({row['sample_id']: row for row in records + scope['extras']}.values())
    selected = extras if (args.videos or args.add_manifest) else annotation_records(config, records)
    if args.split and args.stage == 'annotate':
        pool = extras if (args.videos or args.add_manifest) else records
        selected = [row for row in pool if args.split == 'all' or row['split'] == args.split]
    selected = balanced_limit(selected, args.limit)
    preview = {'root': relative_path(root), 'scope_videos': len(records),
               'selected_videos': len(selected), 'representations': requested,
               'cost_scope_videos': len(scope['cost']['sample_ids'])}
    print(json.dumps(preview, indent=2), flush=True)
    if args.dry_run or args.stage == 'prepare':
        return preview
    stages = (['annotate', 'encode', 'eval', 'analyze', 'plot', 'report']
              if args.stage == 'all' else [args.stage])
    if 'annotate' in stages:
        provider_key(config['providers'][config['annotation']['provider']])
    if 'cost' in stages:
        for provider in args.providers or config['cost']['providers']:
            provider_key(config['providers'][provider])
    if ('annotate' in stages and 'encode' in stages) or 'cost' in stages:
        preflight_embedding(config)
    results = {}
    for stage in stages:
        started = time.perf_counter()
        started_utc = utc_now()
        print(f'Semantic stage: {stage}', flush=True)
        try:
            if stage == 'annotate':
                results[stage] = annotate_groups(config, root, selected, all_records,
                                                 requested, args)
                update_params(root, disabled=sorted(disabled - set(requested)),
                              annotation_engineering=results[stage])
            elif stage == 'encode':
                load_embedder = embedding_loader(config)
                results[stage] = {rep: encode(config, root / rep, all_records,
                                             embedder=load_embedder)[1] for rep in requested}
                update_params(root, encoding_engineering=results[stage])
            elif stage == 'eval':
                rows, splits = evaluation_scope(config, records, cfgs, args.split)
                results[stage] = {}
                for cfg, split in splits.items():
                    values = evaluate_groups(config, root, [row for row in rows if row['cfg'] == cfg],
                                             requested, [cfg], split, True)
                    for rep, payloads in values.items():
                        results[stage].setdefault(rep, {}).update(payloads)
                active = active_representations(config, root)
                aggregate(root, active, cfgs)
            elif stage == 'analyze':
                rows, splits = evaluation_scope(config, records, cfgs, args.split)
                split = next(iter(splits.values())) if len(set(splits.values())) == 1 else 'mixed'
                active = active_representations(config, root, args.only)
                aggregate(root, active, cfgs)
                for rep in active:
                    payload = analyze(config, root / rep, matching_embeddings(config, root / rep),
                                      rows, root / rep, split=split, video_run=args.video_run,
                                      trajectory_run=args.trajectory_run)
                    payload['splits'] = splits
                    write_json(root / rep / 'analysis.json', payload)
                results[stage] = collect_analysis(config, root, active)
            elif stage == 'scene':
                results[stage] = export_scene(config, root, records, scope, args)
            elif stage == 'cost':
                results[stage] = run_cost(config, root, records, scope, requested, args)
                if args.stage == 'cost':
                    plot_all(root)
                    export_report(root)
            elif stage == 'remove':
                for rep in args.only:
                    if (root / rep).exists():
                        shutil.rmtree(root / rep)
                    for variant in (root / 'cost').glob('*'):
                        if variant.is_dir() and (variant / rep).exists():
                            shutil.rmtree(variant / rep)
                disabled.update(args.only)
                update_params(root, disabled=sorted(disabled))
                active = active_representations(config, root)
                aggregate(root, active, config['cfgs'])
                collect_analysis(config, root, active)
                cost_path = root / 'cost/summary.json'
                if cost_path.exists():
                    cost = json.loads(cost_path.read_text())
                    cost['representations'] = [rep for rep in cost['representations']
                                               if rep in active]
                    for variant in cost['variants']:
                        for levels in variant['metrics'].values():
                            for level, values in levels.items():
                                levels[level] = {rep: value for rep, value in values.items()
                                                 if rep in active}
                    write_json(cost_path, cost)
                results[stage] = {'removed': args.only}
                plot_all(root)
                export_report(root)
            elif stage == 'plot':
                active = active_representations(config, root, args.only)
                aggregate(root, active, cfgs)
                collect_analysis(config, root, active)
                results[stage] = plot_all(root)
            elif stage == 'report':
                results[stage] = str(export_report(root))
        finally:
            params = json.loads((root / 'params.json').read_text())
            params.setdefault('executions', []).append({
                'stage': stage, 'started_utc': started_utc, 'finished_utc': utc_now(),
                'seconds': time.perf_counter() - started,
                'success': stage in results, 'representations': requested, 'limit': args.limit,
                'allow_partial': args.allow_partial})
            write_json(root / 'params.json', params)
    return {'root': relative_path(root), 'stages': list(results), 'representations': requested}


def run(config: dict, args) -> dict:
    root = resolve_path(config['root'])
    with file_lock(root / '.workflow.lock'):
        return _run(copy.deepcopy(config), args)
