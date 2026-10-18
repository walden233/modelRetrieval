import io
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from urllib.error import HTTPError

import pytest

from bise.modalities.semantic import annotation
from bise.modalities.semantic.credentials import (
    NoActiveKeys, active_key, add_keys, mark_overdue, provider_key,
)


def pool(tmp_path):
    path = tmp_path / 'keys.local.json'
    add_keys(path, ['old-test-key', 'new-test-key'])
    return path, {'base_url': 'https://stub.invalid/v4', 'key_pool': str(path),
                  'api_key_env': ['TEST_GLM_KEY']}


def response(payload):
    return io.BytesIO(json.dumps(payload).encode())


def test_active_pool_orders_keys_and_never_falls_back_to_environment(tmp_path, monkeypatch):
    path, provider = pool(tmp_path)
    monkeypatch.setenv('TEST_GLM_KEY', 'old-test-key')
    assert provider_key(provider) == 'old-test-key'
    assert mark_overdue(path, 'old-test-key')
    assert not mark_overdue(path, 'old-test-key')
    add_keys(path, ['old-test-key', 'new-test-key'])
    assert provider_key(provider) == 'new-test-key'
    mark_overdue(path, 'new-test-key')
    with pytest.raises(NoActiveKeys, match='No usable API keys'):
        provider_key(provider)
    add_keys(path, ['third-test-key'])
    assert active_key(path) == 'third-test-key'
    # An accidental duplicate in active must not override permanent quarantine.
    data = json.loads(path.read_text())
    data['active'].insert(0, {'id': 'duplicate', 'key': 'old-test-key'})
    path.write_text(json.dumps(data))
    assert active_key(path) == 'third-test-key'
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('status', [200, 429])
def test_balance_error_switches_key_in_same_request_and_persists(tmp_path, monkeypatch, status):
    path, provider = pool(tmp_path)
    sent = []

    def urlopen(req, timeout):
        auth = req.get_header('Authorization')
        sent.append(auth)
        if auth == 'Bearer old-test-key':
            payload = {'error': {'code': '1113', 'message': 'Balance exhausted'}}
            if status == 429:
                raise HTTPError(req.full_url, status, 'Limited', {}, response(payload))
            return response(payload)
        return response({'choices': [{'message': {'content': 'ok'}}]})

    monkeypatch.setattr(annotation.request, 'urlopen', urlopen)
    raw = annotation.send_request(provider, {'model': 'test'}, 10)
    assert raw['choices'][0]['message']['content'] == 'ok'
    assert sent == ['Bearer old-test-key', 'Bearer new-test-key']
    data = json.loads(path.read_text())
    assert [r['key'] for r in data['active']] == ['new-test-key']
    assert data['overdue'][0]['code'] == '1113' and data['overdue'][0]['since']
    assert active_key(path) == 'new-test-key'


@pytest.mark.parametrize('code', ['1302', '1305', '1316', '1002'])
def test_other_errors_do_not_rotate_and_redact_all_credentials(tmp_path, monkeypatch, code):
    path, provider = pool(tmp_path)

    def urlopen(req, timeout):
        payload = {'error': {'code': code, 'message': 'old-test-key new-test-key'}}
        raise HTTPError(req.full_url, 429, 'old-test-key', {}, response(payload))

    monkeypatch.setattr(annotation.request, 'urlopen', urlopen)
    with pytest.raises(HTTPError) as caught:
        annotation.send_request(provider, {'model': 'test'}, 10)
    text = str(caught.value) + caught.value.read().decode()
    assert 'old-test-key' not in text and 'new-test-key' not in text
    assert '[redacted]' in text
    assert len(json.loads(path.read_text())['active']) == 2
    assert not json.loads(path.read_text())['overdue']


def test_parallel_balance_failures_quarantine_once(tmp_path, monkeypatch):
    path, provider = pool(tmp_path)
    barrier = threading.Barrier(8)

    def urlopen(req, timeout):
        if req.get_header('Authorization') == 'Bearer old-test-key':
            barrier.wait(timeout=10)
            raise HTTPError(req.full_url, 429, 'Limited', {},
                            response({'error': {'code': 1113}}))
        return response({'ok': True})

    monkeypatch.setattr(annotation.request, 'urlopen', urlopen)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda _: annotation.send_request(provider, {'model': 'test'}, 10), range(8)))
    assert all(row['ok'] for row in results)
    data = json.loads(path.read_text())
    assert len(data['active']) == len(data['overdue']) == 1


def test_all_keys_overdue_stops_without_retrying_retired_keys(tmp_path, monkeypatch):
    path, provider = pool(tmp_path)
    sent = []

    def urlopen(req, timeout):
        sent.append(req.get_header('Authorization'))
        raise HTTPError(req.full_url, 429, 'Limited', {},
                        response({'error': {'code': '1113'}}))

    monkeypatch.setattr(annotation.request, 'urlopen', urlopen)
    with pytest.raises(NoActiveKeys):
        annotation.send_request(provider, {'model': 'test'}, 10)
    assert len(sent) == 2
    with pytest.raises(NoActiveKeys):
        annotation.send_request(provider, {'model': 'test'}, 10)
    assert len(sent) == 2 and len(json.loads(path.read_text())['overdue']) == 2


def test_environment_only_provider_remains_supported(monkeypatch):
    monkeypatch.setenv('TEST_GLM_KEY', 'environment-test-key')
    assert provider_key({'api_key_env': 'TEST_GLM_KEY'}) == 'environment-test-key'
    assert provider_key({'stub': True}) == ''


@pytest.mark.parametrize('status,code,settings,retired', [
    (400, 'Arrearage', {'overdue_codes': ['Arrearage']}, True),
    (429, 'PostpaidBillOverdue', {'overdue_codes': ['PostpaidBillOverdue']}, True),
    (402, 'insufficient_balance', {'overdue_codes': [], 'overdue_statuses': [402]}, True),
    (401, 'invalid_api_key', {'overdue_codes': [], 'overdue_statuses': [402]}, False),
    (429, 'insufficient_quota', {'overdue_codes': ['Arrearage']}, False),
])
def test_provider_specific_balance_errors_only_retire_exhausted_keys(
        tmp_path, monkeypatch, status, code, settings, retired):
    path, provider = pool(tmp_path)
    provider.update(settings)
    sent = []

    def urlopen(req, timeout):
        sent.append(req.get_header('Authorization'))
        if sent[-1] == 'Bearer old-test-key':
            raise HTTPError(req.full_url, status, 'Error', {},
                            response({'error': {'code': code}}))
        return response({'ok': True})

    monkeypatch.setattr(annotation.request, 'urlopen', urlopen)
    if retired:
        assert annotation.send_request(provider, {'model': 'test'}, 10)['ok']
        data = json.loads(path.read_text())
        assert data['overdue'][0]['code'] == (str(status) if status == 402 else code)
        assert len(sent) == 2 and len(data['active']) == 1
    else:
        with pytest.raises(HTTPError):
            annotation.send_request(provider, {'model': 'test'}, 10)
        assert len(sent) == 1 and not json.loads(path.read_text())['overdue']


def test_qwen_custom_endpoint_and_deepseek_payload_are_openai_compatible(tmp_path, monkeypatch):
    from bise.modalities.semantic.workflow import load_config

    config = load_config('configs/semantic/ch3.json')
    path, _ = pool(tmp_path)
    sent = []

    def urlopen(req, timeout):
        sent.append((req.full_url, json.loads(req.data)))
        return response({'choices': [{'message': {'content': '{}'}}]})

    monkeypatch.setattr(annotation.request, 'urlopen', urlopen)
    for name in ['qwen', 'deepseek']:
        provider = {**config['providers'][name], 'key_pool': str(path)}
        body = annotation.video_body(config, provider, {'role': 'robot'},
                                     ['data:image/jpeg;base64,AA=='] * 16,
                                     {'timestamps_seconds': list(range(16))})
        annotation.send_request(provider, body, 10)
    assert sent[0][0] == config['providers']['qwen']['base_url'] + '/chat/completions'
    assert 'llm-rgdqstg3ghvg6ayi.cn-beijing.maas.aliyuncs.com' in sent[0][0]
    assert sent[1][0] == 'https://api.deepseek.com/chat/completions'
    assert sent[0][1]['enable_thinking'] is False
    assert sent[1][1]['thinking'] == {'type': 'disabled'}
    assert all(len([x for x in body['messages'][0]['content'] if x['type'] == 'image_url']) == 16
               and body['response_format'] == {'type': 'json_object'} for _, body in sent)


def test_all_keys_overdue_preserves_checkpoint_and_can_resume(tmp_path, monkeypatch):
    from bise.modalities.semantic.store import RequestStore, item_key
    from bise.modalities.semantic.workflow import load_config

    path, provider = pool(tmp_path)
    config = load_config('configs/semantic/ch3.json')
    config['root'] = str(tmp_path / 'artifacts')
    config['providers']['glm'].update(provider)
    config['annotation'].update(concurrency=1, backoff_seconds=0)
    records = [{'sample_id': f'sample{i}', 'source': {'path': f'video{i}'},
                'video_path': 'unused.mp4', 'role': 'human'} for i in range(2)]
    job = annotation.make_job(config, 'glm', 16)
    store = RequestStore(job)
    store.accept(records[0], {'description': 'Existing accepted result.'}, 'existing')

    def urlopen(req, timeout):
        raise HTTPError(req.full_url, 429, 'Limited', {},
                        response({'error': {'code': '1113', 'message': 'old-test-key'}}))

    monkeypatch.setattr(annotation.request, 'urlopen', urlopen)
    frames = lambda *args: (['data:image/jpeg;base64,AA=='], {'timestamps_seconds': [0.]})
    with pytest.raises(NoActiveKeys):
        annotation.annotate(config, records, 'glm', 16, frame_loader=frames)
    assert not store.unfinished()
    assert store.annotations()[item_key(records[0])]['attempt_id'] == 'existing'
    assert 'old-test-key' not in store.ledger.read_text()
    assert 'new-test-key' not in store.ledger.read_text()
    add_keys(path, ['third-test-key'])
    accepted = {'description': 'Move an object.',
                'labels': {'capability_tags': [], 'task_complexity': 'unknown',
                           'environment_tags': ['unknown'], 'scene_category': 'unknown'},
                'object_action_text': 'Objects: an object. Operations: move the object.',
                'action_focus_text': 'Objects: an object. Operations: move the object.',
                'structured': {'objects': [], 'operations': [], 'result': None}}
    monkeypatch.setattr(annotation.request, 'urlopen', lambda *args, **kwargs: response({
        'choices': [{'message': {'content': json.dumps(accepted)}}]}))
    _, summary = annotation.annotate(config, records, 'glm', 16, frame_loader=frames)
    assert summary['selection']['accepted'] == 2
    assert store.annotations()[item_key(records[0])]['attempt_id'] == 'existing'


def test_pool_settings_do_not_change_annotation_cache_signature(tmp_path):
    from bise.modalities.semantic.workflow import load_config

    config = load_config('configs/semantic/ch3.json')
    config['root'] = str(tmp_path / 'artifacts')
    config['providers']['glm'].pop('key_pool')
    job = annotation.make_job(config, 'glm', 16)
    config['providers']['glm']['key_pool'] = str(tmp_path / 'keys.local.json')
    assert annotation.make_job(config, 'glm', 16) == job
    assert 'key_pool' not in (job / 'params.json').read_text()
