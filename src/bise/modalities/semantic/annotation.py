"""Offline frame sampling and bounded, journaled VLM/scene requests."""

import base64
import copy
import io
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request

import numpy as np

from bise.common.paths import resolve_path
from bise.modalities.semantic.credentials import NoActiveKeys, mark_overdue, provider_key, redact
from bise.modalities.semantic.representations import (
    validate_scene, validate_video, video_prompt,
)
from bise.modalities.semantic.store import (
    RequestStore, account_usage, file_lock, item_key, read_jsonl, resolve_job, write_json,
)


def make_job(config: dict, provider_name: str, frames: int, kind: str = "video") -> Path:
    if kind not in ('video', 'scene'):
        raise ValueError('Annotation kind must be video or scene')
    provider = config["providers"][provider_name]
    prompt_path = config[f"{kind}_prompt"]
    signature = {
        "schema_version": 3, "kind": kind, "provider": provider_name,
        "model": provider["model"], "base_url": provider.get("base_url", "stub"),
        "request": provider.get("request", {}), "stub": provider.get("stub", False),
        "prompt": resolve_path(prompt_path).read_text(),
        "validation_policy": "description_labels_p1" if kind == 'video' else 'task_content_p4',
    }
    if kind == "video":
        signature.update({"frames": frames, "taxonomy": config["taxonomy"],
                          "max_side": config["annotation"]["max_side"],
                          "jpeg_quality": config["annotation"]["jpeg_quality"]})
    name = "scene" if kind == "scene" else provider_name + str(frames)
    if kind == 'scene':
        signature['source_representation'] = config['scene']['source']
        name += '-' + config['scene']['source'].lower()
    return resolve_job(resolve_path(config["root"]) / 'jobs', name, signature)


def sample_frames(path: Path, count: int, max_side: int, jpeg_quality: int) -> tuple[list, dict]:
    import cv2
    from decord import VideoReader, cpu

    started = time.perf_counter()
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    total = len(reader)
    if total == 0:
        raise ValueError(f"Empty video: {path}")
    indices = np.linspace(0, total - 1, min(total, count), dtype=int)
    frames = reader.get_batch(indices).asnumpy()
    timestamps = reader.get_frame_timestamp(indices)[:, 0].tolist()
    decoded = time.perf_counter() - started
    height, width = frames[0].shape[:2]
    urls, sent_sizes = [], []
    encode_start = time.perf_counter()
    for frame in frames:
        scale = min(1.0, max_side / max(frame.shape[:2]))
        if scale < 1:
            frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
        success, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not success:
            raise ValueError(f"Could not encode a sampled frame: {path}")
        urls.append("data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
        sent_sizes.append([int(frame.shape[1]), int(frame.shape[0])])
    metadata = {
        "requested": count, "actual": len(urls), "total_frames": total,
        "sampling": "uniform_frame_index",
        "fps": float(reader.get_avg_fps()), "indices": indices.tolist(),
        "timestamps_seconds": timestamps, "original_size": [width, height],
        "sent_sizes": sent_sizes, "jpeg_quality": jpeg_quality,
        "decode_seconds": decoded, "image_encode_seconds": time.perf_counter() - encode_start,
    }
    return urls, metadata


def video_body(config: dict, provider: dict, record: dict, urls: list, frames: dict) -> dict:
    content = [{"type": "text", "text": video_prompt(config, record["role"])}]
    for index, (url, timestamp) in enumerate(zip(urls, frames["timestamps_seconds"])):
        content.extend([
            {"type": "text", "text": f"Sampled frame {index}; video time {timestamp:.3f} seconds."},
            {"type": "image_url", "image_url": {"url": url}},
        ])
    return {"model": provider["model"], "messages": [{"role": "user", "content": content}],
            "stream": False, **provider.get("request", {})}


def response_content(raw: dict) -> str:
    content = raw["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Response has no textual JSON content")
    return content


def send_request(provider: dict, body: dict, timeout: float) -> dict:
    if provider.get("stub"):
        content = provider.get("stub_response")
        if content is None:
            raise ValueError("Stub provider requires stub_response")
        return {"model": provider["model"], "choices": [{"message": {
            "content": json.dumps(content)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100}}
    while True:
        key = provider_key(provider)
        req = request.Request(provider["base_url"].rstrip("/") + "/chat/completions",
                              data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + key)
        http_error = None
        try:
            with request.urlopen(req, timeout=timeout) as response:
                text = response.read().decode()
        except error.HTTPError as exc:
            http_error = exc
            text = exc.read().decode(errors='replace')
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        api_error = payload.get('error', {}) if isinstance(payload, dict) else {}
        code = str(api_error.get('code')) if isinstance(api_error, dict) else ''
        if code in provider.get('overdue_codes', ['1113']):
            balance_code = code
        elif http_error is not None and http_error.code in provider.get('overdue_statuses', []):
            balance_code = str(http_error.code)
        else:
            balance_code = None
        if balance_code is not None and provider.get('key_pool'):
            if mark_overdue(provider['key_pool'], key, balance_code):
                print(f'API balance exhausted ({balance_code}): key moved to overdue; '
                      'trying the next active key.', flush=True)
            continue
        safe_text = redact(provider, text)
        if http_error is not None:
            raise error.HTTPError(http_error.url, http_error.code,
                                  redact(provider, str(http_error.reason)), http_error.headers,
                                  io.BytesIO(safe_text.encode())) from None
        return json.loads(safe_text)


def annotate(config: dict, records: list[dict], provider_name: str, frames: int,
             force: bool = False, retry_unknown: bool = False, kind: str = "video",
             frame_loader=sample_frames, sender=send_request,
             allow_failures: bool = False, retry_failed: bool = False) -> tuple[Path, dict]:
    provider = config["providers"][provider_name]
    provider_key(provider)
    directory = make_job(config, provider_name, frames, kind)
    config = copy.deepcopy(config)
    signature = json.loads((directory / 'params.json').read_text())['signature']
    config[f'_{kind}_prompt_text'] = signature['prompt']
    store = RequestStore(directory)
    settings = config["annotation"]
    settings['concurrency'] = int(provider.get('concurrency', settings['concurrency']))
    decode_concurrency = int(settings.get('decode_concurrency', settings['concurrency']))
    requests_per_second = float(provider.get('requests_per_second', 0))
    if settings['concurrency'] < 1 or decode_concurrency < 1 or requests_per_second < 0:
        raise ValueError('Concurrency must be positive and request rate nonnegative')
    decode_slots = threading.BoundedSemaphore(decode_concurrency)
    pace_lock, next_request = threading.Lock(), 0.0

    def pace_request():
        nonlocal next_request
        if not requests_per_second:
            return
        with pace_lock:
            delay = max(0, next_request - time.perf_counter())
            if delay:
                time.sleep(delay)
            next_request = time.perf_counter() + 1 / requests_per_second

    execution_id = uuid.uuid4().hex

    def validate(raw, record, frame_info):
        content = response_content(raw)
        if kind == "video":
            return validate_video(content, config["taxonomy"], role=record['role'])
        return validate_scene(content, record["observations"])

    with file_lock(directory / ".lock"):
        recovered = store.recover(validate)
        existing = store.annotations()
        pending = [r for r in records if force or item_key(r) not in existing]
        previous_attempts, previous_responses, prior_errors = {}, {}, {}
        history = read_jsonl(store.ledger)
        start_keys = {event['attempt_id']: event['item_key'] for event in history
                      if event['event'] == 'started'}
        for event in history:
            if event['event'] == 'started':
                previous_attempts[event['item_key']] = previous_attempts.get(event['item_key'], 0) + 1
            elif event['event'] == 'response':
                previous_responses[item_key(event['record'])] = event
            elif event['event'] == 'validation' and not event['valid']:
                key = start_keys.get(event['attempt_id'])
                prior_errors.setdefault(key, [])
                if event['error'] not in prior_errors[key]:
                    prior_errors[key].append(event['error'])
        unknown = store.unfinished() & {item_key(r) for r in pending}
        if unknown and not retry_unknown:
            raise RuntimeError(f"{len(unknown)} interrupted requests have unknown outcome; "
                               "inspect requests.jsonl or use --retry-unknown")
        queued_at = time.perf_counter()
        store.event({"event": "execution_start", "execution_id": execution_id,
                     "selected": len(records), "pending": len(pending), "recovered": recovered,
                     "concurrency": settings["concurrency"],
                     "decode_concurrency": decode_concurrency,
                     "requests_per_second": requests_per_second})

        def process(record):
            item_started = time.perf_counter()
            frame_info = {}
            try:
                if kind == "video":
                    with decode_slots:
                        urls, frame_info = frame_loader(
                            resolve_path(record["video_path"]), frames,
                            settings["max_side"], settings["jpeg_quality"])
                    body = video_body(config, provider, record, urls, frame_info)
                else:
                    body = {"model": provider["model"], "stream": False,
                            **provider.get("request", {}), "messages": [{"role": "user", "content":
                            config['_scene_prompt_text'] + "\nObservations:\n" +
                            json.dumps(record["observations"], ensure_ascii=False)}]}
            except Exception as exc:
                store.event({"event": "preparation_error", "execution_id": execution_id,
                             "sample_id": record["sample_id"], "error": str(exc)})
                return False
            feedback = list(prior_errors.get(item_key(record), [])) if not force else []
            previous_content = None
            previous_attempt_id = None
            if feedback and item_key(record) in previous_responses:
                previous = previous_responses[item_key(record)]
                previous_content = response_content(previous['raw_response'])
                previous_attempt_id = previous['attempt_id']
            for local_attempt in range(1, settings["max_attempts"] + 1):
                attempt = previous_attempts.get(item_key(record), 0) + local_attempt
                attempt_id = uuid.uuid4().hex
                store.event({"event": "started", "attempt_id": attempt_id,
                             "execution_id": execution_id, "attempt": attempt,
                             "new_episode": bool(force and local_attempt == 1),
                             "item_key": item_key(record), "sample_id": record["sample_id"],
                             "queue_seconds": item_started - queued_at})
                request_started = time.perf_counter()
                request_timestamp = time.time()
                rate_wait_seconds = 0.0
                retry_after = None
                try:
                    request_body = copy.deepcopy(body)
                    if feedback:
                        if previous_content:
                            request_body['messages'].append({'role':'assistant', 'content':previous_content})
                        request_body['messages'].append({'role': 'user', 'content':
                            'Correct the previous JSON. All validation issues observed:\n- ' +
                            '\n- '.join(feedback) +
                            '\nReturn the corrected COMPLETE JSON. Preserve unaffected facts and fields. '
                            'Output task content only. Omit unsupported facts and observation commentary. '
                            'Do not add explanations, reasons, uncertainty notes, or extra fields.'})
                    if provider.get('send_request_id'):
                        request_body['request_id'] = attempt_id
                    pace_started = time.perf_counter()
                    pace_request()
                    rate_wait_seconds = time.perf_counter() - pace_started
                    request_started = time.perf_counter()
                    request_timestamp = time.time()
                    raw = sender(provider, request_body, settings["timeout_seconds"])
                    request_seconds = time.perf_counter() - request_started
                    store.event({"event": "response", "execution_id": execution_id,
                                 "attempt_id": attempt_id, "record": record, "frames": frame_info,
                                 "validation_feedback": list(feedback),
                                 "repair_previous_attempt_id": previous_attempt_id,
                                 "raw_response": raw, "accounting": account_usage(
                                     raw, provider.get("price", {}), request_timestamp),
                                 "timings": {"request_seconds": request_seconds,
                                             "rate_wait_seconds": rate_wait_seconds,
                                             "item_elapsed_seconds": time.perf_counter() - item_started}})
                    parse_started = time.perf_counter()
                    try:
                        data = validate(raw, record, frame_info)
                    except (ValueError, KeyError, TypeError) as exc:
                        if str(exc) not in feedback:
                            feedback.append(str(exc))
                        previous_content = raw.get('choices', [{}])[0].get('message', {}).get('content')
                        previous_attempt_id = attempt_id
                        store.event({"event": "validation", "execution_id": execution_id,
                                     "attempt_id": attempt_id, "valid": False, "error": str(exc),
                                     "parse_seconds": time.perf_counter() - parse_started})
                    else:
                        store.accept(record, data, attempt_id,
                                     frame_info if kind == "video" else None)
                        store.event({"event": "validation", "execution_id": execution_id,
                                     "attempt_id": attempt_id, "valid": True,
                                     "raw_schema_valid": not data.get('normalization_notes'),
                                     "normalization_notes": data.get('normalization_notes', []),
                                     "parse_seconds": time.perf_counter() - parse_started})
                        return True
                except Exception as exc:
                    status = exc.code if isinstance(exc, error.HTTPError) else None
                    detail = None
                    if isinstance(exc, error.HTTPError):
                        retry_after = exc.headers.get("Retry-After")
                        detail = exc.read().decode(errors='replace')[:2000]
                        detail = redact(provider, detail)
                    store.event({"event": "error", "execution_id": execution_id,
                                 "attempt_id": attempt_id, "sample_id": record["sample_id"],
                                 "error_type": type(exc).__name__,
                                 "error": redact(provider, str(exc)),
                                 "error_detail": detail,
                                 "http_status": status,
                                 "timings": {"request_seconds": time.perf_counter() - request_started,
                                             "rate_wait_seconds": rate_wait_seconds}})
                    if isinstance(exc, NoActiveKeys):
                        raise
                    if status is not None and status not in (408, 429) and status < 500:
                        return False
                if local_attempt < settings["max_attempts"]:
                    delay = min(60, settings["backoff_seconds"] * 2 ** (local_attempt - 1))
                    if retry_after and retry_after.isdigit():
                        delay = min(60, max(delay, int(retry_after)))
                    store.event({"event": "backoff", "execution_id": execution_id,
                                 "sample_id": record["sample_id"], "seconds": delay})
                    time.sleep(delay)
            return False

        successes = 0
        with ThreadPoolExecutor(max_workers=min(settings["concurrency"], max(1, len(pending)))) as pool:
            futures = {pool.submit(process, record): record for record in pending}
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    success = future.result()
                except NoActiveKeys:
                    for queued in futures:
                        queued.cancel()
                    raise
                successes += success
                elapsed = time.perf_counter() - queued_at
                eta = elapsed / index * (len(pending)-index)
                print(f"{directory.name}: {index}/{len(pending)} "
                      f"{'OK' if success else 'FAILED'} {futures[future]['sample_id']} "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s new_cost={store.new_known_cost:.4f} CNY", flush=True)
        elapsed = time.perf_counter() - queued_at
        store.event({"event": "execution_end", "execution_id": execution_id,
                     "selected": len(records), "new_items": len(pending), "new_successes": successes,
                     "cache_hits": len(records) - len(pending), "recovered": recovered,
                     "active_seconds": elapsed,
                     "new_known_cost": store.new_known_cost,
                     "new_items_per_second": len(pending) / elapsed if pending else None,
                     "concurrency": settings["concurrency"],
                     "decode_concurrency": decode_concurrency,
                     "requests_per_second": requests_per_second})
        summary = store.summarize()
        current = store.annotations()
        failed = [r["sample_id"] for r in records if item_key(r) not in current]
        summary["selection"] = {"count": len(records), "accepted": len(records) - len(failed),
                                "missing": failed, "partial": bool(failed)}
        write_json(directory / 'summary.json', summary)
        if failed and not retry_failed:
            message = (f"{len(failed)} annotations failed in {directory}; "
                       "see requests.jsonl, then rerun annotate. First IDs: " +
                       ", ".join(failed[:3]))
            if not allow_failures or len(failed) == len(records):
                raise RuntimeError(message)
            print(f'WARNING: {message}. Continuing with '
                  f'{summary["selection"]["accepted"]}/{len(records)} accepted annotations.',
                  flush=True)
    if failed and retry_failed:
        print(f'{directory.name}: automatically retrying {len(failed)} failed annotations '
              'once before continuing.', flush=True)
        # Release the job lock first; accepted items are cached and never resubmitted.
        return annotate(config, records, provider_name, frames, force=False,
                        retry_unknown=retry_unknown, kind=kind, frame_loader=frame_loader,
                        sender=sender, allow_failures=allow_failures, retry_failed=False)
    return directory, summary
