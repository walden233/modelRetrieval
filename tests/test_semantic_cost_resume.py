import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from bise.modalities.semantic import annotation as annotation_module
from bise.modalities.semantic.annotation import annotate, make_job
from bise.modalities.semantic.store import RequestStore, read_jsonl
from bise.modalities.semantic.workflow import prepare, run
from test_semantic_chapter import arguments, config_and_records, fake_frames, sample_response


def test_explicit_cost_providers_resume_qwen_and_keep_glm_api_disabled(tmp_path, monkeypatch):
    config, records = config_and_records(tmp_path)
    config['cost']['providers'] = ['glm', 'qwen', 'deepseek']
    for name in ['qwen', 'deepseek']:
        config['providers'][name].update(stub=True, stub_response=sample_response(),
                                         concurrency=2, requests_per_second=0)
    root, records, scope = prepare(config, config['cfgs'])
    indexed = {row['sample_id']: row for row in records}
    target = [indexed[sample] for sample in scope['cost']['sample_ids']]
    original = {}
    for frames in config['cost']['frames']:
        job, _ = annotate(config, target, 'glm', frames, frame_loader=fake_frames)
        for filename in ['requests.jsonl', 'annotations.jsonl']:
            path = job / filename
            original[path] = path.read_bytes()
    qwen, _ = annotate(config, target[:4], 'qwen', 16, frame_loader=fake_frames)
    accepted = RequestStore(qwen).annotations()
    calls = []

    def selected(cfg, rows, provider, frames, **kwargs):
        assert provider in ('qwen', 'deepseek'), 'GLM must not enter annotation'
        calls.append(provider)
        return annotate(cfg, rows, provider, frames, frame_loader=fake_frames, **kwargs)

    def key(provider):
        assert provider is not None and provider['model'] != config['providers']['glm']['model']
        return ''

    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate', selected)
    monkeypatch.setattr('bise.modalities.semantic.workflow.provider_key', key)
    monkeypatch.setattr(annotation_module.request, 'urlopen',
                        lambda *a, **k: pytest.fail('Stub test made a network call'))
    run(config, arguments(stage='cost', providers=['qwen', 'deepseek']))
    summary = json.loads((root / 'cost/summary.json').read_text())
    assert calls == ['qwen', 'deepseek']
    assert len(summary['variants']) == 5
    assert [v['annotation_mode'] for v in summary['variants']] == [
        'cache_only', 'cache_only', 'cache_only', 'resume', 'resume']
    assert summary['common_videos'] == len(target)
    assert all(path.read_bytes() == data for path, data in original.items())
    current = RequestStore(qwen).annotations()
    assert len(current) == len(target)
    assert all(current[key]['attempt_id'] == row['attempt_id'] for key, row in accepted.items())
    execution = RequestStore(qwen).summarize()['executions'][-1]
    assert execution['new_items'] == execution['cache_hits'] == 4


def test_provider_concurrency_and_bounded_decoding_do_not_change_job_signature(tmp_path):
    config, records = config_and_records(tmp_path)
    config['annotation'].update(concurrency=1, decode_concurrency=1)
    original = make_job(config, 'glm', 16)
    config['providers']['glm']['concurrency'] = 4
    assert make_job(config, 'glm', 16) == original
    barrier, lock = threading.Barrier(4), threading.Lock()
    decoding, peak = 0, 0

    def frames(*args):
        nonlocal decoding, peak
        with lock:
            decoding += 1
            peak = max(peak, decoding)
        time.sleep(.005)
        result = fake_frames(*args)
        with lock:
            decoding -= 1
        return result

    def sender(provider, body, timeout):
        barrier.wait(timeout=10)
        return annotation_module.send_request(provider, body, timeout)

    _, summary = annotate(config, records[:4], 'glm', 16, frame_loader=frames, sender=sender)
    assert peak == 1
    assert summary['executions'][-1]['concurrency'] == 4
    assert summary['executions'][-1]['decode_concurrency'] == 1


def test_pacing_is_not_included_in_recorded_api_latency(tmp_path, monkeypatch):
    config, records = config_and_records(tmp_path)
    config['providers']['glm'].update(concurrency=1, requests_per_second=2)
    clock = [0.]
    fake_time = SimpleNamespace(perf_counter=lambda: clock[0], time=lambda: 1792300000.)

    def sleep(seconds):
        clock[0] += seconds

    fake_time.sleep = sleep
    monkeypatch.setattr(annotation_module, 'time', fake_time)

    def sender(provider, body, timeout):
        clock[0] += .02
        return annotation_module.send_request(provider, body, timeout)

    job, summary = annotate(config, records[:3], 'glm', 16,
                             frame_loader=fake_frames, sender=sender)
    responses = [e for e in read_jsonl(job / 'requests.jsonl') if e['event'] == 'response']
    assert [e['timings']['request_seconds'] for e in responses] == pytest.approx([.02] * 3)
    assert [e['timings']['rate_wait_seconds'] for e in responses] == pytest.approx([0, .48, .48])
    assert summary['executions'][-1]['requests_per_second'] == 2


def test_incomplete_unselected_models_are_skipped_without_api_calls(tmp_path, monkeypatch):
    config, records = config_and_records(tmp_path)
    config['cost']['providers'] = ['glm', 'qwen', 'deepseek']
    config['providers']['qwen'].update(stub=True, stub_response=sample_response(),
                                      concurrency=2, requests_per_second=0)
    calls = []

    def selected(cfg, rows, provider, frames, **kwargs):
        calls.append(provider)
        assert provider == 'qwen'
        return annotate(cfg, rows, provider, frames, frame_loader=fake_frames, **kwargs)

    monkeypatch.setattr('bise.modalities.semantic.workflow.annotate', selected)
    result = run(config, arguments(stage='cost', providers=['qwen']))
    summary = json.loads((Path(config['root']) / 'cost/summary.json').read_text())
    assert result['stages'] == ['cost'] and calls == ['qwen']
    assert [v['provider'] for v in summary['variants']] == ['qwen']
    assert len(summary['skipped_variants']) == 4
