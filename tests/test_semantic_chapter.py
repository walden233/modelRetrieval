import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from bise.modalities.semantic import annotation as annotation_module
from bise.modalities.semantic.annotation import annotate, make_job
from bise.modalities.semantic.artifacts import aggregate, materialize, scope_records
from bise.modalities.semantic.encoding import encode, load_embeddings
from bise.modalities.semantic.experiments import evaluate, multimodal_cases, scope_statistics
from bise.modalities.semantic.figures import prepare_figures
from bise.modalities.semantic.manifest import balanced_limit, sample_cost
from bise.modalities.semantic.representations import (
    representation_texts, validate_scene, validate_video, video_prompt,
)
from bise.modalities.semantic.store import RequestStore, account_usage, item_key, read_jsonl, write_json
from bise.modalities.semantic.workflow import (
    annotation_records, evaluation_scope, export_scene, load_config, prepare, run_cost,
)


def sample_response():
    return {
        'description': 'The demonstrator moves a red block onto a support.',
        'labels': {'capability_tags': ['transport', 'place'], 'task_complexity': 'low',
                   'environment_tags': ['unknown'], 'scene_category': 'unknown'},
        'object_action_text': 'Objects: a block and a support. Operations: move the block onto the support.',
        'action_focus_text': 'Objects: a movable object and its support. Operations: transfer and place.',
        'structured': {'objects': [{'id': 'o1', 'name': 'block', 'role': 'manipulated object'}],
                       'operations': [{'action': 'move', 'object': 'o1', 'actor': 'primary',
                                       'target': {'location': 'on the support'}}], 'result': None},
    }


def config_and_records(tmp_path):
    config = load_config(ROOT / 'configs/semantic/ch3.json')
    config['root'] = str(tmp_path / 'semantic/v2')
    config['sources'] = {}
    config['annotation'].update(concurrency=2, max_attempts=2, backoff_seconds=0,
                                splits={'cfg2': ['test'], 'cfg3': ['train', 'val', 'test']})
    config['evaluation']['split'] = 'test'
    config['embedding'] = {'provider_name': 'hash', 'dimension': 16, 'batch_size': 8, 'max_length': 2048}
    config['providers']['glm'] = {'model': 'stub-model', 'stub': True,
                                'stub_response': sample_response(),
                                'price': {'currency': 'CNY', 'input': 1, 'output': 2}}
    config['cost'].update(tasks=2, split='test', providers=['glm'])
    for cfg in ['cfg2', 'cfg3']:
        root = tmp_path / 'data' / cfg
        mapping = {'train': [], 'val': [], 'test': []}
        for task in ['task_0001', 'task_0002']:
            for scene in ['scene_1', 'scene_2']:
                split = 'train' if scene == 'scene_1' else 'test'
                for camera in ['a', 'b']:
                    mapping[split].append(f'rh20t::{task}::{scene}::{camera}')
                    for role in ['human', 'robot']:
                        p = root / task / scene / f'cam_{camera}_{role}.mp4'
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(b'not decoded during prepare')
        source = tmp_path / 'manifests/test' / cfg
        config['sources'][cfg] = str(source / 'split_manifest_video.json')
        write_json(source / 'split_manifest_video.json', mapping)
        write_json(source / 'metadata.json', {'data_root': str(root)})
    records, _ = scope_records(config, config['cfgs'])
    return config, records


def fake_frames(path, count, max_side, jpeg_quality):
    return ['data:image/jpeg;base64,AA=='] * count, {
        'actual': count, 'requested': count, 'indices': list(range(count)),
        'timestamps_seconds': list(range(count)), 'decode_seconds': 0.01,
        'image_encode_seconds': 0.01,
    }


def test_prepare_frozen_pairs_cross_cfg_identity_and_balanced_smoke(tmp_path, monkeypatch):
    config, records = config_and_records(tmp_path)
    monkeypatch.setattr(annotation_module, 'sample_frames', lambda *a: pytest.fail('prepare decoded video'))
    output, rows, scope = prepare(config, config['cfgs'])
    assert len(rows) == 32
    assert len({row['sample_id'] for row in rows}) == 32
    assert len(balanced_limit(rows, 16)) == 16
    assert {row['cfg'] for row in balanced_limit(rows, 8)} == {'cfg2', 'cfg3'}
    assert all(row['split'] == 'test' for row in balanced_limit(rows, 8))
    assert len(sample_cost(rows, config)) == 8
    assert len(scope['cost']['scene_ids']) == 2
    assert 'sha256' not in (output / 'scope.json').read_text()
    assert all('split' not in row for row in read_jsonl(Path(config['root']) / 'videos.jsonl'))


def test_cfg_specific_evaluation_scopes_and_explicit_override(tmp_path):
    config, records = config_and_records(tmp_path)
    config['evaluation']['split'] = {'cfg2': 'all', 'cfg3': 'test'}
    rows, splits = evaluation_scope(config, records, config['cfgs'])
    assert splits == {'cfg2': 'all', 'cfg3': 'test'}
    assert sum(r['cfg'] == 'cfg2' for r in rows) == 16
    assert sum(r['cfg'] == 'cfg3' for r in rows) == 8
    rows, splits = evaluation_scope(config, records, config['cfgs'], 'test')
    assert len(rows) == 16 and all(r['split'] == 'test' for r in rows)
    rows, _ = evaluation_scope(config, records, config['cfgs'], 'all')
    assert len(rows) == 32


def test_changed_cost_scope_archives_results_and_keeps_request_cache(tmp_path):
    config, records = config_and_records(tmp_path)
    config['cost'].update(split='all', tasks=1)
    root, _, scope = prepare(config, config['cfgs'])
    previous = copy.deepcopy(scope['cost'])
    assert len(previous['sample_ids']) == 4
    write_json(root / 'cost/summary.json', {'target_videos': 4})
    write_json(root / 'charts/cost.json', {'old': True})
    (root / 'charts/cost.png').write_bytes(b'old figure')
    write_json(root / 'plots.json', {'charts/cost.png': {}, 'charts/cfg3.png': {}})
    (root / 'jobs').mkdir()
    (root / 'jobs/keep.txt').write_text('accepted requests')
    config['cost']['tasks'] = 2
    _, _, updated = prepare(config, config['cfgs'])
    assert len(updated['cost']['sample_ids']) == 8
    assert updated['cost']['tasks'] == 2 and updated['cost']['split'] == 'all'
    assert len(updated['cost']['scene_ids']) == 2
    assert (root / 'jobs/keep.txt').read_text() == 'accepted requests'
    archive, = (root / 'history').glob('cost-*')
    assert json.loads((archive / 'scope.json').read_text()) == previous
    assert (archive / 'cost/summary.json').is_file()
    assert (archive / 'charts/cost.png').read_bytes() == b'old figure'
    assert not (root / 'cost').exists() and not (root / 'charts/cost.json').exists()
    assert json.loads((root / 'plots.json').read_text()) == {'charts/cfg3.png': {}}
    config['cost']['frames'] = [16]
    _, _, unchanged = prepare(config, config['cfgs'])
    assert unchanged['cost'] == updated['cost']
    assert len(list((root / 'history').glob('cost-*'))) == 1


def test_main_eval_and_plot_replace_cfg2_test_with_all_and_keep_cfg3_test(tmp_path):
    from bise.modalities.semantic.workflow import run

    config, records = config_and_records(tmp_path)
    job, _ = annotate(config, records, 'glm', 16, frame_loader=fake_frames)
    materialize(config, job, records, config['representations'])
    root = Path(config['root'])
    for rep in config['representations']:
        encode(config, root / rep, records)
    run(config, arguments(stage='eval'))
    run(config, arguments(stage='plot'))
    old_cfg3 = (root / 'evaluations/cfg3/metrics.json').read_bytes()
    config['evaluation']['split'] = {'cfg2': 'all', 'cfg3': 'test'}
    run(config, arguments(stage='eval', cfg=['cfg2']))
    run(config, arguments(stage='plot'))
    payload = json.loads((root / 'evaluations/cfg2/metrics.json').read_text())
    recipe = json.loads((root / 'charts/cfg2.json').read_text())['data']
    assert payload['split'] == recipe['split'] == 'all'
    assert payload['coverage']['expected'] == payload['coverage']['present'] == 16
    assert recipe['protocol']['ids'] == payload['protocol']['ids']
    assert (root / 'evaluations/cfg3/metrics.json').read_bytes() == old_cfg3


def test_role_prompts_original_labels_and_representation_isolation(tmp_path):
    config, _ = config_and_records(tmp_path)
    human, robot = video_prompt(config, 'human'), video_prompt(config, 'robot')
    assert 'primary human demonstrator' in human and 'Focus on the robot arm' not in human
    assert 'Focus on the robot arm' in robot and 'Ignore unrelated human presence' in robot
    assert 'task_0001' not in human + robot
    data = validate_video(sample_response(), config['taxonomy'])
    texts = representation_texts(data, config['taxonomy'])
    assert texts['S2c'].startswith(texts['S1'] + '\nLabels: ')
    assert json.loads(texts['S5']) == data['structured']
    assert not {'quality', 'evidence', 'labels'} & json.loads(texts['S5']).keys()
    bad = copy.deepcopy(data)
    bad['structured']['operations'][0]['object'] = 'nonexistent'
    with pytest.raises(ValueError, match='missing object'):
        validate_video(bad, config['taxonomy'])
    bad = copy.deepcopy(data)
    bad['labels']['capability_tags'] = []
    validate_video(bad, config['taxonomy'])


def test_content_guards_and_recorded_label_normalization(tmp_path):
    config, _ = config_and_records(tmp_path)
    bad = sample_response()
    bad['object_action_text'] += ' The outcome is unclear.'
    with pytest.raises(ValueError, match='task content only'):
        validate_video(bad, config['taxonomy'])
    bad = sample_response()
    bad['description'] = 'A robot moves an object. A person types in the background.'
    with pytest.raises(ValueError, match='Remove unrelated people'):
        validate_video(bad, config['taxonomy'], role='robot')
    bad['structured']['operations'][0]['actor'] = 'joint'
    validate_video(bad, config['taxonomy'], role='robot')
    data = sample_response()
    data['labels']['capability_tags'] += ['push']
    normalized = validate_video(data, config['taxonomy'])
    assert normalized['labels']['capability_tags'] == ['transport', 'place']
    assert normalized['normalization_notes'] == [{'field':'capability_tags','unsupported':['push']}]


def test_request_resume_raw_recovery_and_frame_protocol_invalidation(tmp_path):
    config, records = config_and_records(tmp_path)
    calls = []
    def sender(provider, body, timeout):
        calls.append(body)
        return annotation_module.send_request(provider, body, timeout)
    job, summary = annotate(config, records[:2], 'glm', 20, frame_loader=fake_frames, sender=sender)
    assert len(calls) == 2 and summary['known_cost'] == pytest.approx(0.0006)
    assert all('split' not in row for row in read_jsonl(job / 'annotations.jsonl'))
    annotate(config, records[:2], 'glm', 20, frame_loader=lambda *a: pytest.fail('cache decoded'), sender=sender)
    assert len(calls) == 2
    (job / 'annotations.jsonl').unlink()
    annotate(config, records[:2], 'glm', 20, frame_loader=lambda *a: pytest.fail('recovery decoded'), sender=sender)
    assert len(calls) == 2 and len(RequestStore(job).annotations()) == 2
    other = make_job(config, 'glm', 16)
    assert other != job
    changed = copy.deepcopy(config)
    changed['providers']['glm']['request'] = {'max_tokens': 200}
    assert make_job(changed, 'glm', 20) != job
    assert summary['executions'][-1]['new_items'] == 2


def test_running_job_freezes_prompt_and_raw_recovery_uses_latest_attempt(tmp_path):
    config, records = config_and_records(tmp_path)
    prompt = tmp_path / 'video.md'
    original = Path(config['video_prompt']).read_text()
    prompt.write_text(original)
    config['video_prompt'] = str(prompt)
    config['annotation']['concurrency'] = 1
    bodies = []
    def sender(provider, body, timeout):
        bodies.append(body)
        prompt.write_text('changed while requests are in flight')
        return annotation_module.send_request(provider, body, timeout)
    job,_ = annotate(config,records[:2],'glm',20,frame_loader=fake_frames,sender=sender)
    assert all('Human focus' not in body['messages'][0]['content'][0]['text'] for body in bodies)
    assert all('Return only the requested JSON' in body['messages'][0]['content'][0]['text'] for body in bodies)
    prompt.write_text(original)
    config['providers']['glm']['stub_response']['description'] = 'The object moves onto a support.'
    annotate(config,records[:1],'glm',20,force=True,frame_loader=fake_frames)
    (job/'annotations.jsonl').unlink()
    annotate(config,records[:2],'glm',20,frame_loader=lambda *a:pytest.fail('recovery decoded'))
    assert RequestStore(job).annotations()[item_key(records[0])]['description'] == 'The object moves onto a support.'
    accounting=scope_statistics(job,records[:1])
    assert accounting['known_cost']==pytest.approx(.0003)
    assert accounting['lifetime_known_cost']==pytest.approx(.0006)


def test_validation_repairs_keep_prior_output_and_all_feedback(tmp_path):
    config, records = config_and_records(tmp_path)
    config['annotation']['max_attempts'] = 3
    bodies = []
    def sender(provider,body,timeout):
        bodies.append(copy.deepcopy(body))
        data = sample_response()
        if len(bodies) == 1:
            data['object_action_text'] = 'Move the object.'
        elif len(bodies) == 2:
            data['action_focus_text'] = 'Transfer the object.'
        raw = annotation_module.send_request(provider,body,timeout)
        raw['choices'][0]['message']['content'] = json.dumps(data)
        return raw
    job,summary = annotate(config,records[:1],'glm',20,frame_loader=fake_frames,sender=sender)
    assert len(bodies)==3 and summary['accepted_videos_or_scenes']==1
    assert bodies[1]['messages'][-2]['role']=='assistant'
    assert 'object_action_text needs Objects:' in bodies[2]['messages'][-1]['content']
    assert 'action_focus_text needs Objects:' in bodies[2]['messages'][-1]['content']


def test_retry_cost_accounts_invalid_output_and_reasoning_once(tmp_path):
    config, records = config_and_records(tmp_path)
    counter = [0]
    def sender(provider, body, timeout):
        counter[0] += 1
        raw = annotation_module.send_request(provider, body, timeout)
        if counter[0] == 1:
            raw['choices'][0]['message']['content'] = 'invalid json'
        raw['usage']['completion_tokens_details'] = {'reasoning_tokens': 40}
        return raw
    job, summary = annotate(config, records[:1], 'glm', 20, frame_loader=fake_frames, sender=sender)
    assert counter[0] == 2
    assert summary['known_cost'] == pytest.approx(0.0006)
    assert summary['schema_valid_rate'] == 0.5 and summary['first_attempt_success_rate'] == 0
    assert account_usage({}, config['providers']['glm']['price'])['cost'] is None
    assert account_usage({'usage': {'prompt_tokens': 100, 'completion_tokens': 100,
                                   'completion_tokens_details': {'reasoning_tokens': 40}}},
                         config['providers']['glm']['price'])['cost'] == pytest.approx(.0003)
    _, cached_summary = annotate(config, records[:1], 'glm', 20, frame_loader=fake_frames, sender=sender)
    execution = cached_summary['executions'][-1]
    assert execution['cache_hits'] == 1 and execution['new_known_cost'] == 0
    assert execution['new_items_per_second'] is None and counter[0] == 2


def test_missing_usage_is_null_and_uncached_estimate_is_explicit(tmp_path):
    price = {'input': 1, 'output': 2, 'cached_input': 0.5}
    missing = account_usage({}, price)
    assert all(missing[key] is None for key in (
        'input_tokens', 'output_tokens', 'cached_tokens', 'reasoning_tokens', 'cost'))
    partial = account_usage({'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}, price)
    assert partial['cached_tokens'] is None and partial['reasoning_tokens'] is None
    assert partial['cost_kind'] == 'usage_priced_assuming_uncached'
    assert partial['cost'] == pytest.approx(0.0003)
    summary = RequestStore(tmp_path).summarize()
    assert all(value is None for value in summary['usage'].values())


def test_scene_returns_task_description_or_null_without_explanations():
    observations = {'H_1': {'role': 'human'}, 'R_1': {'role': 'robot'}}
    assert validate_scene({'description': 'Press the button.'}, observations)['description']
    assert validate_scene({'description': None}, observations)['description'] is None
    for text in ['null', 'None', 'In the human observation a box is opened.',
                 'The robot side only grasps the box.', 'The box is possibly moved.',
                 'A person and a robot each move a block.',
                 'The human slides a block, while the robot lifts it.']:
        with pytest.raises(ValueError):
            validate_scene({'description': text}, observations)
    with pytest.raises(ValueError, match='description only'):
        validate_scene({'description': None, 'reason': 'Occluded'}, observations)
    with pytest.raises(ValueError, match='task content only'):
        validate_scene({'description': 'The final result is unclear.'}, observations)


def test_interrupted_request_is_not_automatically_resubmitted(tmp_path):
    config, records = config_and_records(tmp_path)
    job = make_job(config, 'glm', 20)
    RequestStore(job).event({'event': 'started', 'attempt_id': 'lost', 'attempt': 1,
                             'item_key': item_key(records[0]), 'sample_id': records[0]['sample_id']})
    with pytest.raises(RuntimeError, match='unknown outcome'):
        annotate(config, records[:1], 'glm', 20, frame_loader=fake_frames)
    annotate(config, records[:1], 'glm', 20, retry_unknown=True, frame_loader=fake_frames)


def test_evaluate_missing_artifacts_selected_refresh_and_positive_definitions(tmp_path):
    config, records = config_and_records(tmp_path)
    rows = [row for row in records if row['cfg'] == 'cfg2' and row['split'] == 'test']
    job, _ = annotate(config, rows, 'glm', 20, frame_loader=fake_frames)
    embedding_file, _ = encode(config, job, rows)
    output = tmp_path / 'results'
    result = evaluate(config, job, embedding_file, rows, output, 'cfg2')
    assert result['metrics']['human_to_robot']['scene']['S1']['gallery_size'] == 4
    rank_rows = read_jsonl(output / 'rankings.jsonl')
    assert all(len(row['positive_ranks']) == 2 for row in rank_rows)
    assert 'MPR-all' in result['metrics']['human_to_robot']['scene']['S1']
    before = result['metrics']['human_to_robot']['task']['S1']
    refreshed = evaluate(config, job, embedding_file, rows, output, 'cfg2', representations=['S2c'])
    assert refreshed['metrics']['human_to_robot']['task']['S1'] == before
    assert len(read_jsonl(output / 'rankings.jsonl')) == len(rank_rows)
    assert not (output / 'run_info.json').exists()
    with pytest.raises(ValueError, match='lack matching'):
        evaluate(config, job, embedding_file, records, output, 'cfg2')
    same, stats = encode(config, job, rows)
    assert same == embedding_file and stats['new_texts'] == 0
    assert np.all(load_embeddings(same)['ready'])


def test_partial_evaluation_uses_common_five_representation_corpus(tmp_path):
    config, records = config_and_records(tmp_path)
    rows = [r for r in records if r['cfg'] == 'cfg2' and r['split'] == 'test']
    job, _ = annotate(config, rows[:4], 'glm', 20, frame_loader=fake_frames)
    file, _ = encode(config, job, rows)
    result = evaluate(config, job, file, rows, tmp_path / 'eval', 'cfg2', allow_partial=True)
    assert result['partial'] and result['coverage']['present'] == 4
    assert result['coverage']['expected'] == 8


@pytest.mark.parametrize('source', ['S1', 'S2c', 'S2w', 'S3', 'S4', 'S5'])
def test_scene_teacher_uses_only_selected_group_and_honors_train_split(tmp_path, source):
    config, records = config_and_records(tmp_path)
    output, records, scope = prepare(config, ['cfg2', 'cfg3'])
    selected = [r for r in records if r['cfg'] == 'cfg2' and r['split'] == 'train']
    job, _ = annotate(config, selected, 'glm', 20, frame_loader=fake_frames)
    materialize(config, job, selected, [source])
    run = tmp_path / 'runs/s44/video/E3'
    write_json(run / 'split_manifest.json', {'train': list(dict.fromkeys(
        r['source_pair_id'] for r in selected))})
    scene_config = copy.deepcopy(config)
    scene_config['providers']['glm']['stub_response'] = {'description': 'Move the object onto its support.'}
    args = SimpleNamespace(run=str(run), source=source, limit=None, force=False, retry_unknown=False)
    result = export_scene(scene_config, output, records, scope, args)
    with np.load(result['files'][0], allow_pickle=False) as data:
        assert len(data['scene_ids']) == 2
        assert all('scene_1' in x for x in data['scene_ids'])
        assert data['usable'].all()
        assert json.loads(str(data['metadata']))['split'] == 'train'
        assert json.loads(str(data['metadata']))['source_representation'] == source
    assert Path(result['files'][0]).parent == output / 'teachers' / source
    teacher_job = output / 'jobs' / ('scene-' + source.lower())
    scene_row = read_jsonl(teacher_job / 'annotations.jsonl')[0]
    assert scene_row['source']['representation'] == source
    assert len(scene_row['observations']) == 4
    assert {row['text'] for row in scene_row['observations'].values()} == {
        row['text'] for row in read_jsonl(output / source / 'annotations.jsonl')}
    assert all(not (output / rep).exists()
               for rep in ['S1', 'S2c', 'S2w', 'S3', 'S4', 'S5'] if rep != source)


def test_scene_repair_reuses_good_old_text_and_exports_only_nonnull_teachers(tmp_path, monkeypatch):
    config, records = config_and_records(tmp_path)
    config['annotation'].update(concurrency=1, splits={'cfg2': ['train', 'test'], 'cfg3': ['test']})
    prompt = tmp_path / 'scene.md'
    prompt.write_text(Path(config['scene_prompt']).read_text())
    config['scene_prompt'] = str(prompt)
    root, records, scope = prepare(config, ['cfg2'])
    job, _ = annotate(config, records, 'glm', 16, frame_loader=fake_frames)
    materialize(config, job, records, ['S3'])
    config['providers']['glm']['stub_response'] = {'description': 'Lift the block.'}
    args = SimpleNamespace(run=None, source='S3', limit=None, force=False, retry_unknown=False)
    result = export_scene(config, root, records, scope, args)
    old = RequestStore(root / 'jobs/scene-s3')
    old_rows = list(old.annotations().values())
    for index, value in enumerate([None, 'null', 'In the human observation a block is lifted.'], 1):
        old.accept(old_rows[index], {'description': value}, f'legacy{index}')
    before = {name: (old.directory / name).read_bytes()
              for name in ('requests.jsonl', 'annotations.jsonl')}
    prompt.write_text(prompt.read_text() + '\nStronger shared-operation constraint.\n')
    outputs = iter(['Place the block.', None, 'Open the box.'])
    calls = []

    def sender(provider, body, timeout):
        calls.append(body)
        return {'choices': [{'message': {'content': json.dumps({'description': next(outputs)})}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}

    original = annotate
    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate',
                        lambda *a, **kw: original(*a, **kw, sender=sender))
    args.repair, args.scene_ids = True, None
    result = export_scene(config, root, records, scope, args)
    assert len(calls) == 3
    assert all((old.directory / name).read_bytes() == value for name, value in before.items())
    new_rows = RequestStore(root / 'jobs/scene-s3-r2').annotations()
    carried = next(row for row in new_rows.values() if row['description'] == 'Lift the block.')
    assert carried['reused_from']['job'].endswith('jobs/scene-s3')
    with np.load(result['files'][0], allow_pickle=False) as data:
        assert len(data['scene_ids']) == 3 and data['usable'].all()
        assert set(data['texts'].tolist()) == {'Lift the block.', 'Place the block.', 'Open the box.'}
        metadata = json.loads(str(data['metadata']))
        assert metadata['usable_scenes'] == 3 and len(metadata['excluded_scenes']) == 1
        assert metadata['target_scenes'] == 4
        assert np.isfinite(data['vectors']).all()


def test_cost_frames_same_subset_and_stage_restore(tmp_path, monkeypatch):
    config, records = config_and_records(tmp_path)
    output, records, scope = prepare(config, config['cfgs'])
    original = annotate
    def patched(config, rows, provider, frames, **kwargs):
        return original(config, rows, provider, frames, frame_loader=fake_frames, **kwargs)
    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate', patched)
    args = SimpleNamespace(providers=None, limit=None, force=False, fresh_benchmark=False,
                           retry_unknown=False, allow_partial=False)
    summary = run_cost(config, output, records, scope, ['S1','S2c','S2w','S3','S4','S5'], args)
    assert [row['frames'] for row in summary['variants']] == [32, 16, 8]
    assert summary['common_videos'] == 8 and summary['task_scene_positives_identical']
    assert all(row['engineering']['attempts'] == 8 for row in summary['variants'])
    assert all('S1' in row['metrics']['human_to_robot']['task'] for row in summary['variants'])
    recipes = prepare_figures(output)
    assert any(path.name == 'cost.json' for path, _ in recipes)
    (output / 'charts/cost.json').unlink()
    assert any(path.name == 'cost.json' for path, _ in prepare_figures(output))


def test_three_provider_cost_pipeline_uses_five_variants_offline(tmp_path, monkeypatch):
    from bise.modalities.semantic.workflow import run

    config, records = config_and_records(tmp_path)
    config['cost']['providers'] = ['glm', 'qwen', 'deepseek']
    for name in ['qwen', 'deepseek']:
        config['providers'][name].update(stub=True, stub_response=sample_response(),
                                         concurrency=2, requests_per_second=0)
    monkeypatch.setattr(annotation_module.request, 'urlopen',
                        lambda *a, **k: pytest.fail('Offline test made an API call'))
    output = Path(config['root'])
    original = annotate

    def patched(config, rows, provider, frames, **kwargs):
        return original(config, rows, provider, frames, frame_loader=fake_frames, **kwargs)

    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate', patched)
    result = run(config, arguments(stage='cost'))
    assert result['stages'] == ['cost']
    summary = json.loads((output / 'cost/summary.json').read_text())
    assert [(v['provider'], v['frames']) for v in summary['variants']] == [
        ('glm', 32), ('glm', 16), ('glm', 8), ('qwen', 16), ('deepseek', 16)]
    assert summary['common_videos'] == summary['target_videos'] == 8
    assert all(v['engineering']['successful'] == 8 for v in summary['variants'])
    assert any(path.name == 'cost.json' for path, _ in prepare_figures(output))
    for artifact in ['charts/cost.png', 'charts/cost.json', 'plots.json',
                     'results.csv', 'report.html']:
        assert (output / artifact).is_file()


@pytest.mark.parametrize('timestamp,period,input_rate', [
    ('2026-10-19T01:00:00+00:00', 'peak', 2),
    ('2026-10-19T04:00:00+00:00', 'off_peak', 1),
    ('2026-10-19T06:00:00+00:00', 'peak', 2),
    ('2026-10-19T10:00:00+00:00', 'off_peak', 1),
    ('2026-10-18T02:00:00+00:00', 'off_peak', 1),
    ('2026-10-01T02:00:00+00:00', 'off_peak', 1),
    ('2027-10-01T02:00:00+00:00', 'peak', 2),
])
def test_deepseek_accounting_uses_china_time_cache_and_holiday_tariffs(
        timestamp, period, input_rate):
    import datetime as dt

    price = load_config('configs/semantic/ch3.json')['providers']['deepseek']['price']
    stamp = dt.datetime.fromisoformat(timestamp).timestamp()
    raw = {'created': stamp, 'usage': {'prompt_tokens': 1000, 'completion_tokens': 100,
                                     'prompt_cache_hit_tokens': 200}}
    result = account_usage(raw, price)
    assert result['cached_tokens'] == 200
    assert result['price']['period'] == period
    assert result['price']['time_basis'] == 'response_created'
    assert result['cost'] == pytest.approx(input_rate * (800 + 200 * .02 + 100 * 4) / 1e6)
    raw.pop('created')
    fallback = account_usage(raw, price, stamp)
    assert fallback['cost'] == result['cost']
    assert fallback['price']['time_basis'] == 'request_started'


def test_shared_redraw_indexes_semantic_without_changing_existing_modalities(tmp_path, monkeypatch):
    from bise.common.plots import index_plots, redraw_plots
    root = tmp_path / 'runs/retrain/test'
    metric = {'schema_version': 2, 'cfg':'cfg2','split':'test','partial':False,
              'metrics': {'human_to_robot': {level: {'S1': {'R@1':.5,'R@5':1.,'R@10':1.,'MRR':.7,'NDCG@10':.8}}
                                              for level in ['task','scene']}}}
    write_json(root / 'semantic/evaluations/cfg2/metrics.json', metric)
    recipes = index_plots(root)
    assert recipes['semantic/charts/cfg2.png']['kind'] == 'semantic'
    seen = []
    monkeypatch.setattr('bise.modalities.semantic.figures.render_figure', lambda source, path: seen.append(path))
    redraw_plots(root)
    assert len(seen) == 1
    assert 'sha256' not in (root / 'plots.json').read_text()


def test_complementarity_re_ranks_only_the_identical_candidate_subset(tmp_path):
    config,records = config_and_records(tmp_path)
    rows=[r for r in records if r['cfg']=='cfg2' and r['split']=='test']
    job,_=annotate(config,rows,'glm',20,frame_loader=fake_frames)
    file,_=encode(config,job,rows)
    vectors=load_embeddings(file)
    humans=[r for r in rows if r['role']=='human']
    evaluation=tmp_path/'video/evaluations/cfg2'
    scenes=['/'.join(r['scene_id'].split('/')[1:]) for r in humans]+['task_extra/scene_extra']
    cameras=[r['camera_id'] for r in humans]+['a']
    write_json(evaluation/'metadata.json',{'scene_ids':scenes,'camera_ids':cameras})
    matrix=np.eye(5,dtype=np.float32)
    matrix[:,-1]=100
    np.save(evaluation/'similarity_matrix.npy',matrix)
    result=multimodal_cases(config,vectors,rows,str(tmp_path/'video'),None)['cfg2']
    assert result['common_queries']==result['common_candidates']==4
    assert all(row['other_ranks']['video']==1 for row in result['rows'])
    assert any(row['complementary_hit'] for row in result['rows'])


def arguments(**values):
    defaults = dict(cfg=None, provider=None, frames=None, concurrency=None, only=None,
                    videos=None, add_manifest=None, split=None, limit=None, dry_run=False,
                    stage='all', source=None, providers=None, force=False, retry_unknown=False,
                    fresh_benchmark=False, allow_partial=False, run=None,
                    video_run=None, trajectory_run=None)
    return SimpleNamespace(**{**defaults, **values})


def test_failed_annotations_do_not_block_main_pipeline_and_resume_only_missing(tmp_path, monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError
    from bise.modalities.semantic.workflow import run

    config, records = config_and_records(tmp_path)
    failures = {next(row['video_path'] for row in records
                     if row['cfg'] == cfg and row['split'] == 'test')
                for cfg in config['cfgs']}
    missing_ids = {row['sample_id'] for row in records if row['video_path'] in failures}
    calls = []
    original = annotate

    def frames(path, *settings):
        urls, metadata = fake_frames(path, *settings)
        urls[0] = str(path)
        return urls, metadata

    def sender(provider, body, timeout):
        path = body['messages'][0]['content'][2]['image_url']['url']
        calls.append(path)
        if path in failures:
            raise HTTPError('https://stub.invalid', 400, 'Bad Request', {}, BytesIO(b'{}'))
        return annotation_module.send_request(provider, body, timeout)

    def patched(cfg, rows, provider, count, **kwargs):
        return original(cfg, rows, provider, count, frame_loader=frames, sender=sender, **kwargs)

    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate', patched)
    monkeypatch.setattr('bise.modalities.semantic.workflow.plot_all',
                        lambda root, overwrite=True: prepare_figures(root, overwrite))
    result = run(config, arguments())
    assert result['stages'] == ['annotate', 'encode', 'eval', 'analyze', 'plot', 'report']
    root = Path(config['root'])
    job = make_job(config, 'glm', 16)
    summary = json.loads((job / 'summary.json').read_text())
    assert set(summary['selection']['missing']) == missing_ids
    assert summary['selection']['accepted'] == len(annotation_records(config, records)) - len(missing_ids)
    assert all(calls.count(row['video_path']) == (2 if row['video_path'] in failures else 1)
               for row in annotation_records(config, records))
    params = json.loads((root / 'params.json').read_text())
    assert params['annotation_engineering']['video']['selection']['partial']
    assert (root / 'report.html').exists()
    for cfg in config['cfgs']:
        common = None
        for rep in config['representations']:
            payload = json.loads((root / rep / 'evaluations' / cfg / 'metrics.json').read_text())
            assert payload['partial'] and payload['coverage']['present'] == 7
            assert payload['coverage']['expected'] == 8
            assert set(payload['coverage']['missing_ids']) == {
                sample for sample in missing_ids if sample.startswith(cfg + '/')}
            if common is not None:
                assert common == payload['protocol']['ids']
            common = payload['protocol']['ids']
        assert json.loads((root / 'charts' / (cfg + '.json')).read_text())['data']['partial']
    attempts = {key: row['attempt_id'] for key, row in RequestStore(job).annotations().items()}
    before = len(calls)
    failures.clear()
    run(config, arguments())
    assert len(calls) - before == len(missing_ids)
    current = RequestStore(job).annotations()
    assert all(current[key]['attempt_id'] == attempt for key, attempt in attempts.items())
    assert not json.loads((job / 'summary.json').read_text())['selection']['partial']
    for cfg in config['cfgs']:
        payload = json.loads((root / 'evaluations' / cfg / 'metrics.json').read_text())
        assert not payload['partial'] and payload['coverage']['present'] == 8
        assert not json.loads((root / 'charts' / (cfg + '.json')).read_text())['data']['partial']


@pytest.mark.parametrize('failure', ['http', 'schema', 'frames'])
@pytest.mark.parametrize('fail_all', [False, True])
def test_failed_round_retries_once_after_first_pass_and_reuses_successes(
        tmp_path, failure, fail_all):
    from collections import Counter
    from io import BytesIO
    from urllib.error import HTTPError

    config, records = config_and_records(tmp_path)
    config['annotation'].update(concurrency=1, max_attempts=1)
    rows = records[:2]
    failed_paths = {row['video_path'] for row in (rows if fail_all else rows[:1])}
    frame_calls, api_calls = Counter(), Counter()

    def frames(path, *settings):
        path = str(path)
        frame_calls[path] += 1
        if failure == 'frames' and path in failed_paths and frame_calls[path] == 1:
            raise ValueError('Temporary frame preparation failure')
        urls, metadata = fake_frames(path, *settings)
        urls[0] = path
        return urls, metadata

    def sender(provider, body, timeout):
        path = body['messages'][0]['content'][2]['image_url']['url']
        api_calls[path] += 1
        first_failure = path in failed_paths and api_calls[path] == 1
        if failure == 'http' and first_failure:
            raise HTTPError('https://stub.invalid', 400, 'Bad Request', {}, BytesIO(b'{}'))
        raw = annotation_module.send_request(provider, body, timeout)
        if failure == 'schema' and first_failure:
            raw['choices'][0]['message']['content'] = '{"invalid": true}'
        return raw

    job, summary = annotate(config, rows, 'glm', 16, frame_loader=frames, sender=sender,
                            retry_failed=True)
    assert summary['selection'] == {'count': 2, 'accepted': 2, 'missing': [], 'partial': False}
    assert len(summary['executions']) == 2
    assert summary['executions'][0]['new_items'] == 2
    assert summary['executions'][1]['new_items'] == len(failed_paths)
    assert summary['executions'][1]['cache_hits'] == 2 - len(failed_paths)
    for row in rows:
        path = row['video_path']
        expected = 2 if path in failed_paths else 1
        assert frame_calls[path] == expected
        assert api_calls[path] == (1 if failure == 'frames' else expected)
    events = read_jsonl(job / 'requests.jsonl')
    first_end = next(i for i, event in enumerate(events) if event['event'] == 'execution_end')
    second_start = [i for i, event in enumerate(events)
                    if event['event'] == 'execution_start'][1]
    assert first_end < second_start


def test_annotation_can_continue_after_validation_failures_but_not_total_failure(tmp_path):
    config, records = config_and_records(tmp_path)
    config['annotation']['concurrency'] = 1
    count = 0

    def sender(provider, body, timeout):
        nonlocal count
        count += 1
        raw = annotation_module.send_request(provider, body, timeout)
        if count > 1:
            data = sample_response()
            data['baseline_description'] = 'Move the red block.'
            raw['choices'][0]['message']['content'] = json.dumps(data)
        return raw

    _, summary = annotate(config, records[:2], 'glm', 20, allow_failures=True,
                          frame_loader=fake_frames, sender=sender)
    assert summary['selection']['missing'] == [records[1]['sample_id']]
    with pytest.raises(RuntimeError, match='annotations failed'):
        annotate(config, records[2:3], 'glm', 20, allow_failures=True,
                 frame_loader=fake_frames, sender=sender)


def test_cost_failures_use_shared_subset_across_all_variants(tmp_path, monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    config, records = config_and_records(tmp_path)
    root, records, scope = prepare(config, config['cfgs'])
    indexed = {row['sample_id']: row for row in records}
    target = [indexed[sample] for sample in scope['cost']['sample_ids']]
    failed = {32: target[0]['video_path'], 16: target[1]['video_path']}
    original = annotate

    def frames(path, *settings):
        urls, metadata = fake_frames(path, *settings)
        urls[0] = str(path)
        return urls, metadata

    def sender(provider, body, timeout):
        images = [part['image_url']['url'] for part in body['messages'][0]['content']
                  if part['type'] == 'image_url']
        if failed.get(len(images)) == images[0]:
            raise HTTPError('https://stub.invalid', 400, 'Bad Request', {}, BytesIO(b'{}'))
        return annotation_module.send_request(provider, body, timeout)

    def patched(cfg, rows, provider, count, **kwargs):
        return original(cfg, rows, provider, count, frame_loader=frames, sender=sender, **kwargs)

    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate', patched)
    summary = run_cost(config, root, records, scope, config['representations'], arguments())
    assert summary['common_videos'] == 6 and summary['target_videos'] == 8
    assert set(summary['sample_ids']) == {
        row['sample_id'] for row in target if row['video_path'] not in failed.values()}
    for variant in summary['variants']:
        assert variant['partial']
        for rep in config['representations']:
            path = root / 'cost' / variant['name'] / rep / 'evaluations/cfg2/metrics.json'
            payload = json.loads(path.read_text())
            assert payload['partial'] and payload['coverage']['present'] == 6
            assert set(payload['protocol']['ids']) == set(summary['sample_ids'])


def test_six_representations_share_one_request_and_keep_independent_projections(tmp_path):
    config, records = config_and_records(tmp_path)
    calls = []
    def sender(provider, body, timeout):
        calls.append(body)
        return annotation_module.send_request(provider, body, timeout)
    job, _ = annotate(config, records[:2], 'glm', 20,
                      frame_loader=fake_frames, sender=sender)
    reps = ['S1', 'S2c', 'S2w', 'S3', 'S4', 'S5']
    materialize(config, job, records[:2], reps)
    assert len(calls) == 2 and job.name == 'glm20'
    prompt = calls[0]['messages'][0]['content'][0]['text']
    assert 'baseline_description' not in prompt
    for rep in reps:
        directory = Path(config['root']) / rep
        row = read_jsonl(directory / 'annotations.jsonl')[0]
        assert row['attempt_id'] == read_jsonl(job / 'annotations.jsonl')[0]['attempt_id']
        file, _ = encode(config, directory, records[:2])
        assert load_embeddings(file)['metadata']['representations'] == [rep]
    data = sample_response()
    data['description'] = 'A person moves the red block beside the blue block.'
    assert validate_video(data, config['taxonomy'])
    data['baseline_description'] = 'Old separate baseline.'
    with pytest.raises(ValueError, match='requires exactly'):
        validate_video(data, config['taxonomy'])


def test_six_output_protocol_preserves_but_does_not_reuse_old_joint_cache(tmp_path):
    config, records = config_and_records(tmp_path)
    old_job = make_job(config, 'glm', 20)
    params = json.loads((old_job / 'params.json').read_text())
    params['signature']['validation_policy'] = 'task_content_p3'
    write_json(old_job / 'params.json', params)
    old_data = sample_response()
    old_data['baseline_description'] = 'Old separate baseline.'
    RequestStore(old_job).accept(records[0], old_data, 'old-attempt')
    saved = (old_job / 'annotations.jsonl').read_bytes()
    new_job, summary = annotate(config, records[:1], 'glm', 20, frame_loader=fake_frames)
    assert new_job != old_job and summary['executions'][-1]['new_items'] == 1
    assert (old_job / 'annotations.jsonl').read_bytes() == saved
    assert 'baseline_description' not in read_jsonl(new_job / 'annotations.jsonl')[0]


def test_teacher_sources_have_separate_caches_and_only_changed_source_is_archived(tmp_path):
    config, records = config_and_records(tmp_path)
    root, records, scope = prepare(config, config['cfgs'])
    selected = [r for r in records if r['cfg'] == 'cfg2' and r['split'] == 'test']
    job, _ = annotate(config, selected, 'glm', 20, frame_loader=fake_frames)
    materialize(config, job, selected, ['S3', 'S4'])
    scene_config = copy.deepcopy(config)
    scene_config['providers']['glm']['stub_response'] = {'description': 'Move the object.'}
    for rep in ('S3', 'S4'):
        export_scene(scene_config, root, records, scope,
                     arguments(source=rep, limit=1))
    saved = (root / 'teachers/S4/cfg2.npz').read_bytes()
    for rep in ('S3', 'S4'):
        params = json.loads((root / 'jobs' / ('scene-' + rep.lower()) / 'params.json').read_text())
        assert params['signature']['source_representation'] == rep
    changed = copy.deepcopy(config)
    changed['providers']['glm']['request'] = {'temperature': 0.5}
    new_job, _ = annotate(changed, selected, 'glm', 20, frame_loader=fake_frames)
    materialize(changed, new_job, selected, ['S3'])
    assert not (root / 'teachers/S3').exists()
    assert (root / 'history/S3-teachers/cfg2.npz').exists()
    assert (root / 'teachers/S4/cfg2.npz').read_bytes() == saved


def test_frozen_semantic_scope_outlives_source_manifests(tmp_path):
    config, records = config_and_records(tmp_path)
    root, records, _ = prepare(config, config['cfgs'])
    for source in config['sources'].values():
        Path(source).unlink()
        (Path(source).parent / 'metadata.json').unlink()
    _, restored, _ = prepare(config, config['cfgs'])
    assert [row['sample_id'] for row in restored] == [row['sample_id'] for row in records]
    assert 'batch' not in json.loads((root / 'params.json').read_text())


def test_independent_groups_survive_s1_removal_and_resume(tmp_path, monkeypatch):
    from bise.modalities.semantic.workflow import run
    config, records = config_and_records(tmp_path)
    original = annotate
    calls = []
    def sender(provider, body, timeout):
        calls.append(body)
        raw = annotation_module.send_request(provider, body, timeout)
        prompt = body['messages'][0]['content']
        if isinstance(prompt, str):
            data = {'description': 'Move the object onto its support.'}
        else:
            data = sample_response()
        raw['choices'][0]['message']['content'] = json.dumps(data)
        return raw
    def patched(cfg, rows, provider, frames, **kwargs):
        return original(cfg, rows, provider, frames, frame_loader=fake_frames, sender=sender, **kwargs)
    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate', patched)
    monkeypatch.setattr('bise.modalities.semantic.workflow.plot_all',
                        lambda root, overwrite=True: prepare_figures(root, overwrite))
    root = Path(config['root'])
    run(config, arguments())
    assert len(calls) == len(annotation_records(config, records))
    assert not (root / 'cost').exists() and not (root / 'teachers').exists()
    run(config, arguments(stage='scene', limit=1))
    teacher = (root / 'teachers/S5/cfg2.npz').read_bytes()
    before_calls = len(calls)
    saved = (root / 'S2c/annotations.jsonl').read_bytes()
    metrics_before = json.loads((root / 'S2c/evaluations/cfg2/metrics.json').read_text())['metrics']
    run(config, arguments(stage='remove', only=['S1']))
    assert not (root / 'S1').exists()
    assert (root / 'teachers/S5/cfg2.npz').read_bytes() == teacher
    assert (root / 'S2c/annotations.jsonl').read_bytes() == saved
    assert 'S1' not in json.loads((root / 'evaluations/cfg2/metrics.json').read_text())['metrics']['human_to_robot']['task']
    assert 'S1' not in json.loads((root / 'charts/cfg2.json').read_text())['data']['metrics']['human_to_robot']['task']
    assert 'S1' not in {row['representation'] for row in read_jsonl(root / 'evaluations/cfg2/rankings.jsonl')}
    run(config, arguments(stage='eval', only=['S2c']))
    assert json.loads((root / 'S2c/evaluations/cfg2/metrics.json').read_text())['metrics'] == metrics_before
    run(config, arguments(stage='annotate'))
    assert len(calls) == before_calls and not (root / 'S1').exists()
    run(config, arguments(stage='scene', limit=1))
    assert (root / 'teachers/S5/cfg2.npz').exists()
    assert not (root / 'S1').exists()


def test_partial_comparison_intersects_only_selected_groups(tmp_path):
    from bise.modalities.semantic.workflow import evaluate_groups
    config, records = config_and_records(tmp_path)
    rows = [r for r in records if r['cfg'] == 'cfg2' and r['split'] == 'test']
    job, _ = annotate(config, rows, 'glm', 20, frame_loader=fake_frames)
    materialize(config, job, rows, ['S2c', 'S3'])
    root = Path(config['root'])
    encode(config, root / 'S2c', rows)
    encode(config, root / 'S3', rows[:4])
    result = evaluate_groups(config, root, rows, ['S2c', 'S3'], ['cfg2'], 'test', True)
    assert all(result[rep]['cfg2']['coverage']['present'] == 4 for rep in ['S2c', 'S3'])
    assert not (root / 'S1').exists() and not (root / 'S1').exists()
    merged = aggregate(root, ['S2c', 'S3'], ['cfg2'])['cfg2']
    assert merged['partial']
    result = evaluate_groups(config, root, rows, ['S2c'], ['cfg2'], 'test', False)
    assert result['S2c']['cfg2']['coverage']['present'] == 8
    merged = aggregate(root, ['S2c', 'S3'], ['cfg2'])['cfg2']
    assert list(merged['metrics']['human_to_robot']['task']) == ['S2c']
    assert merged['excluded'] == {'S3': 'different evaluation corpus or protocol'}


def test_protocol_change_archives_only_affected_groups(tmp_path):
    config, records = config_and_records(tmp_path)
    job, _ = annotate(config, records[:2], 'glm', 20, frame_loader=fake_frames)
    materialize(config, job, records[:2], ['S2c', 'S3'])
    root = Path(config['root'])
    for rep in ('S2c', 'S3'):
        file, _ = encode(config, root / rep, records[:2])
        evaluate(config, root / rep, file, records[:2], root / rep / 'evaluations/cfg2', 'cfg2')
    s3 = (root / 'S3/annotations.jsonl').read_bytes()
    changed = copy.deepcopy(config)
    changed['providers']['glm']['request'] = {'temperature': 0.5}
    new_job, _ = annotate(changed, records[:2], 'glm', 20, frame_loader=fake_frames)
    materialize(changed, new_job, records[:2], ['S2c'])
    assert (root / 'history/S2c/annotations.jsonl').exists()
    assert (root / 'S3/annotations.jsonl').read_bytes() == s3
    assert json.loads((root / 'S2c/params.json').read_text())['job'] != json.loads((root / 'S3/params.json').read_text())['job']
    file, _ = encode(changed, root / 'S2c', records[:2])
    evaluate(changed, root / 'S2c', file, records[:2], root / 'S2c/evaluations/cfg2', 'cfg2')
    merged = aggregate(root, ['S2c', 'S3'], ['cfg2'])['cfg2']
    assert list(merged['metrics']['human_to_robot']['task']) == ['S2c']
    assert 'S3' in merged['excluded']


def test_old_artifacts_are_archived_once_and_not_reused(tmp_path):
    config, _ = config_and_records(tmp_path)
    root = Path(config['root'])
    old = root / 'glm20'
    write_json(old / 'params.json', {'signature': {'schema_version': 2}})
    (old / 'annotations.jsonl').write_text('{"description":"old format"}\n')
    prepare(config, config['cfgs'])
    assert not old.exists()
    archive = root / 'history/jobs/glm20'
    assert (archive / 'annotations.jsonl').read_text() == '{"description":"old format"}\n'
    prepare(config, config['cfgs'])
    assert make_job(config, 'glm', 20) == root / 'jobs/glm20'
    assert not (root / 'S1').exists()


def test_central_redraw_recovers_missing_aggregate_and_preserves_saved_data(tmp_path, monkeypatch):
    import shutil
    from bise.common.plots import index_plots, redraw_plots
    from bise.modalities.semantic.workflow import evaluate_groups
    config, records = config_and_records(tmp_path)
    root, records, _ = prepare(config, config['cfgs'])
    rows = [row for row in records if row['cfg'] == 'cfg2' and row['split'] == 'test']
    job, _ = annotate(config, rows, 'glm', 20, frame_loader=fake_frames)
    materialize(config, job, rows, ['S2c'])
    encode(config, root / 'S2c', rows)
    evaluate_groups(config, root, rows, ['S2c'], ['cfg2'], 'test', False)
    aggregate(root, ['S2c'], ['cfg2'])
    prepare_figures(root)
    source = root / 'charts/cfg2.json'
    saved = json.loads(source.read_text())
    saved['data']['metrics']['human_to_robot']['task']['S2c']['MRR'] = 0.123
    write_json(source, saved)
    shutil.rmtree(root / 'evaluations')
    recipes = index_plots(root)
    assert 'charts/cfg2.png' in recipes
    assert (root / 'evaluations/cfg2/metrics.json').exists()
    assert json.loads(source.read_text())['data']['metrics']['human_to_robot']['task']['S2c']['MRR'] == 0.123
    source.unlink()
    index_plots(root)
    assert json.loads(source.read_text())['data']['metrics']['human_to_robot']['task']['S2c']['MRR'] != 0.123
    seen = []
    monkeypatch.setattr('bise.modalities.semantic.figures.render_figure',
                        lambda source, target: seen.append(target))
    redraw_plots(root)
    assert seen == [root / 'charts/cfg2.png']
    assert 'sha256' not in (root / 'plots.json').read_text()


def test_concat_and_weighted_label_encoding_are_separate_and_cache_recipe(tmp_path):
    from bise.modalities.semantic.encoding import matching_embeddings
    from bise.modalities.semantic.representations import label_text
    config, records = config_and_records(tmp_path)
    job, _ = annotate(config, records[:1], 'glm', 16, frame_loader=fake_frames)
    materialize(config, job, records[:1], ['S2c', 'S2w'])
    root = Path(config['root'])
    desc = sample_response()['description']
    labels = label_text(sample_response()['labels'], config['taxonomy'])
    calls = []

    class Encoder:
        def encode_texts(self, texts):
            calls.extend(texts)
            return np.asarray([[3., 0.] if x == desc else [0., 2.] if x == labels
                               else [2 ** -.5, 2 ** -.5] for x in texts], dtype=np.float32)

    concat_file, concat_stats = encode(config, root / 'S2c', records[:1], embedder=Encoder())
    weighted_file, weighted_stats = encode(config, root / 'S2w', records[:1], embedder=Encoder())
    concat, weighted = load_embeddings(concat_file), load_embeddings(weighted_file)
    expected = np.array([.7, .3]) / np.linalg.norm([.7, .3])
    assert concat_stats['new_texts'] == 1 and weighted_stats['new_texts'] == 2
    assert calls == [desc + '\nLabels: ' + labels, desc, labels]
    assert np.allclose(weighted['S2w'][0], expected)
    assert np.allclose(weighted['S2w_description'][0], [1, 0])
    assert np.allclose(weighted['S2w_labels'][0], [0, 1])
    assert not np.allclose(concat['S2c'], weighted['S2w'])
    assert not (root / 'S1').exists()
    same, stats = encode(config, root / 'S2w', records[:1],
                         embedder=lambda: pytest.fail('Cached weighted vector loaded model'))
    assert same == weighted_file and stats['cache_texts'] == 2
    changed = copy.deepcopy(config)
    changed['label_fusion'] = {'description_weight': .6, 'label_weight': .4}
    with pytest.raises(FileNotFoundError):
        matching_embeddings(changed, root / 'S2w')
    new_file, _ = encode(changed, root / 'S2w', records[:1], embedder=Encoder())
    assert new_file != weighted_file
    assert np.allclose(load_embeddings(new_file)['S2w'][0], np.array([.6, .4]) / np.linalg.norm([.6, .4]))
    assert matching_embeddings(changed, root / 'S2c') == concat_file
    assert make_job(changed, 'glm', 16) == job
    annotate(config, records[:2], 'glm', 16, frame_loader=fake_frames)
    materialize(config, job, records[:2], ['S2w'])
    extended_file, stats = encode(changed, root / 'S2w', records[:2], embedder=Encoder())
    extended = load_embeddings(extended_file)
    assert stats['new_texts'] == 2 and stats['cache_texts'] == 2
    assert extended['S2w'].shape == (2, 2)
    for key in ('S2w', 'S2w_description', 'S2w_labels'):
        assert np.allclose(extended[key][0], load_embeddings(new_file)[key][0])
        assert np.allclose(extended[key][1], extended[key][0])


def test_default_annotation_scope_and_explicit_training_increment(tmp_path, monkeypatch):
    from bise.modalities.semantic.workflow import run
    config, records = config_and_records(tmp_path)
    selected = annotation_records(config, records)
    assert len(selected) == 24
    assert all(row['split'] == 'test' for row in selected if row['cfg'] == 'cfg2')
    assert {row['split'] for row in selected if row['cfg'] == 'cfg3'} == {'train', 'test'}
    original = annotate

    def patched(cfg, rows, provider, frames, **kwargs):
        assert frames == 16
        return original(cfg, rows, provider, frames, frame_loader=fake_frames, **kwargs)

    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate', patched)
    run(config, arguments(stage='annotate'))
    root = Path(config['root'])
    job = make_job(config, 'glm', 16)
    assert {row['sample_id'] for row in RequestStore(job).annotations().values()} == {
        row['sample_id'] for row in selected}
    before = (root / 'S3/annotations.jsonl').read_bytes()
    run(config, arguments(stage='annotate', cfg=['cfg2'], split='train', only=['S1']))
    assert len(RequestStore(job).annotations()) == 32
    assert (root / 'S3/annotations.jsonl').read_bytes() == before
    assert len(RequestStore(root / 'S1').annotations()) == 32
    assert json.loads((root / 'params.json').read_text())['annotation_target_videos'] == 24


def test_retired_groups_are_archived_and_description_keeps_labels_separate(tmp_path):
    config, _ = config_and_records(tmp_path)
    root = Path(config['root'])
    for rep in ('S0', 'S2'):
        write_json(root / rep / 'params.json', {'signature': {'schema_version': 3}})
        (root / rep / 'annotations.jsonl').write_text('old product\n')
    prepare(config, config['cfgs'])
    for rep in ('S0', 'S2'):
        assert not (root / rep).exists()
        assert (root / 'history' / rep / 'annotations.jsonl').read_text() == 'old product\n'
    data = sample_response()
    data['description'] += ' Labels: capability_tags: transport'
    with pytest.raises(ValueError, match='outside description'):
        validate_video(data, config['taxonomy'])
