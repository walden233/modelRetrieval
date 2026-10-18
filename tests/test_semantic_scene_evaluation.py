import copy
from collections import Counter
from pathlib import Path

import pytest

from bise.modalities.semantic import annotation as annotation_module
from bise.modalities.semantic.annotation import annotate
from bise.modalities.semantic.artifacts import materialize
from bise.modalities.semantic.encoding import encode
from bise.modalities.semantic.scene_evaluation import evaluate_scenes, scene_subsets
from test_semantic_chapter import config_and_records, fake_frames


def test_scene_caps_are_nested_keep_all_tasks_and_camera_pairs():
    records = []
    for task, count in [('a', 4), ('b', 3), ('c', 2)]:
        for scene in range(count):
            for camera in ['left', 'right']:
                for role in ['human', 'robot']:
                    records.append({'task_id': task, 'scene_id': f'{task}/{scene}',
                                    'camera_id': camera, 'role': role})
    result = scene_subsets(records, ['all', '3', '2', '1'], 44)
    assert [len(result[cap]) for cap in ['all', '3', '2', '1']] == [36, 32, 24, 12]
    sets = [{r['scene_id'] for r in result[cap]} for cap in ['all', '3', '2', '1']]
    assert sets[3] < sets[2] < sets[1] < sets[0]
    assert all({r['task_id'] for r in rows} == {'a', 'b', 'c'} for rows in result.values())
    assert all(all(n == 4 for n in Counter(r['scene_id'] for r in rows).values())
               for rows in result.values())
    reversed_result = scene_subsets(list(reversed(records)), ['all', '3', '2', '1'], 44)
    assert all({r['scene_id'] for r in rows} ==
               {r['scene_id'] for r in reversed_result[cap]} for cap, rows in result.items())
    with pytest.raises(ValueError, match='positive integers'):
        scene_subsets(records, ['0'], 44)


def prepare_source(tmp_path):
    config, records = config_and_records(tmp_path)
    rows = [r for r in records if r['cfg'] == 'cfg2']
    job, _ = annotate(config, rows, 'glm', 16, frame_loader=fake_frames)
    materialize(config, job, rows, config['representations'])
    for rep in config['representations']:
        encode(config, Path(config['root']) / rep, [r for r in rows if r['split'] == 'test'])
    return config


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def test_offline_scene_evaluation_encodes_missing_vectors_without_touching_source(
        tmp_path, monkeypatch):
    config = prepare_source(tmp_path)
    source = Path(config['root'])
    before = snapshot(source)
    original_config = copy.deepcopy(config)
    monkeypatch.setattr(annotation_module.request, 'urlopen',
                        lambda *a, **k: pytest.fail('Evaluation made an API call'))
    output = tmp_path / 'isolated'
    summary = evaluate_scenes(config, output)
    assert config == original_config
    assert snapshot(source) == before
    assert summary['api_calls'] == 0
    assert [v['videos'] for v in summary['variants']] == [16, 16, 16, 8]
    assert all(v['tasks'] == 2 for v in summary['variants'])
    assert all(v['new_vectors'] == 8 for v in summary['encoding'].values())
    one = summary['variants'][-1]['metrics']['human_to_robot']
    assert one['task'] == one['scene']
    for cap in ['all', 's3', 's2', 's1']:
        assert (output / cap / 'charts/cfg2.png').is_file()
        assert (output / cap / 'evaluations/cfg2/metrics.json').is_file()
    assert (output / 'summary.csv').is_file() and (output / 'README.md').is_file()
    saved = snapshot(output)
    with pytest.raises(FileExistsError):
        evaluate_scenes(config, output)
    assert snapshot(output) == saved
    with pytest.raises(ValueError, match='new output'):
        evaluate_scenes(config, source)


def test_missing_annotations_stop_before_creating_outputs(tmp_path):
    config = prepare_source(tmp_path)
    path = Path(config['root']) / 'S1/annotations.jsonl'
    path.write_text(''.join(path.read_text().splitlines(keepends=True)[1:]))
    output = tmp_path / 'missing'
    with pytest.raises(ValueError, match='lack current annotations'):
        evaluate_scenes(config, output)
    assert not output.exists()
