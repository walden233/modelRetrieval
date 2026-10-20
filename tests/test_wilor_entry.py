import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np


ENTRY = Path(__file__).resolve().parents[1] / 'runs/trajectory/hand_wilor.py'


def test_list_only_discovers_cfgs_without_writing_or_loading_models(tmp_path):
    dataset = tmp_path/'dataset'
    for cfg in ('cfg2', 'cfg3'):
        scene = dataset/f'RH20T_{cfg}/task_0001/scene_1'
        scene.mkdir(parents=True)
        (scene/'cam_123_human.mp4').touch()
    output = tmp_path/'output'
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'dataset_root': str(dataset), 'output_dir': str(output),
                                 'frame_stride': 1, 'batch_size': 1, 'scenes': []}))
    result = subprocess.run([sys.executable, str(ENTRY), '--config', str(config),
                             '--stage', 'extract', '--all-scenes', '--cfg', 'cfg2',
                             '--archive', '--list-only'], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['selection'] == {'cfg2': {'scenes': 1, 'videos': 1}}
    assert not output.exists()
    assert not output.with_name('output.tar.gz').exists()


def test_cached_extraction_packages_outputs_without_running_inference(tmp_path):
    source = tmp_path/'dataset/RH20T_cfg2/task_0001/scene_1'
    source.mkdir(parents=True)
    video = source/'cam_123_human.mp4'
    video.write_bytes(b'cached-source-not-to-be-decoded')
    repo = tmp_path/'repo'
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=test', '-c', 'user.email=test@example.invalid',
                    'commit', '-q', '--allow-empty', '-m', 'test'], check=True)
    output = tmp_path/'output'
    saved = output/'cfg2/task_0001/scene_1'
    saved.mkdir(parents=True)
    np.savez(saved/'123.npz', complete=True)
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'dataset_root': str(tmp_path/'dataset'), 'output_dir': str(output),
                                 'repo_dir': str(repo), 'frame_stride': 1, 'batch_size': 1,
                                 'scenes': ['cfg2/task_0001/scene_1']}))
    before = video.read_bytes(), video.stat().st_mtime_ns
    # Batch changes are execution settings and must reuse complete camera results.
    old = json.loads(config.read_text())
    old.update(repo_revision=subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(), max_frames=None)
    (output/'params.json').write_text(json.dumps(old))
    updated = json.loads(config.read_text())
    updated.update(batch_size=4, frame_batch_size=4)
    config.write_text(json.dumps(updated))
    subprocess.run([sys.executable, str(ENTRY), '--config', str(config), '--stage', 'extract',
                    '--archive'], check=True, capture_output=True, text=True)
    with tarfile.open(tmp_path/'output.tar.gz') as archive:
        assert 'output/cfg2/task_0001/scene_1/123.npz' in archive.getnames()
        assert 'output/params.json' in archive.getnames()
    assert before == (video.read_bytes(), video.stat().st_mtime_ns)
    assert not (tmp_path/'output.tar.gz.part').exists()
    assert json.loads((output/'params.json').read_text())['frame_batch_size'] == 4
