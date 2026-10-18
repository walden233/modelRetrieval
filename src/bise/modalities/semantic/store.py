"""Append-only request accounting and recoverable annotation projections."""

import contextlib
import datetime as dt
import fcntl
import json
import math
import os
import threading
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith("\n"):
                raise
    return rows


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Repair only an interrupted trailing write, never silently drop complete lines.
    if path.exists():
        with path.open("rb+") as handle:
            handle.seek(0, 2)
            length = handle.tell()
            if length:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.seek(0)
                    data = handle.read()
                    end = data.rfind(b"\n") + 1
                    tail = data[end:]
                    try:
                        json.loads(tail)
                        handle.seek(0, 2)
                        handle.write(b"\n")
                    except json.JSONDecodeError:
                        handle.truncate(end)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using this semantic job: {path.parent}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_job(root: Path, name: str, signature: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".jobs.lock"):
        revision = 1
        while True:
            directory = root / (name if revision == 1 else f"{name}-r{revision}")
            params = directory / "params.json"
            if not params.exists():
                write_json(params, {"schema_version": 2, "created_utc": utc_now(),
                                    "signature": signature})
                return directory
            if json.loads(params.read_text())["signature"] == signature:
                return directory
            revision += 1


def item_key(record: dict) -> str:
    return json.dumps({"id": record["sample_id"], "source": record["source"]}, sort_keys=True)


class RequestStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.ledger = directory / "requests.jsonl"
        self.annotations_path = directory / "annotations.jsonl"
        self.mutex = threading.RLock()
        self.new_known_cost = 0.0

    def event(self, value: dict) -> None:
        with self.mutex:
            append_jsonl(self.ledger, {"time_utc": utc_now(), **value})
            if value['event']=='response' and value['accounting']['cost'] is not None:
                self.new_known_cost += value['accounting']['cost']

    def accept(self, record: dict, data: dict, attempt_id: str, frames: dict | None = None) -> dict:
        annotation = {**{k: v for k, v in record.items() if k != "split"}, **data, "attempt_id": attempt_id,
                      "job": self.directory.name, "annotated_utc": utc_now()}
        if frames is not None:
            annotation["frames"] = frames
        with self.mutex:
            append_jsonl(self.annotations_path, annotation)
        return annotation

    def annotations(self) -> dict[str, dict]:
        return {item_key(row): row for row in read_jsonl(self.annotations_path)}

    def recover(self, validator) -> int:
        accepted = self.annotations()
        responses = [row for row in read_jsonl(self.ledger) if row["event"] == "response"]
        recovered = 0
        for row in reversed(responses):
            if item_key(row["record"]) in accepted:
                continue
            try:
                data = validator(row["raw_response"], row["record"], row.get("frames", {}))
            except (ValueError, KeyError, TypeError):
                continue
            accepted[item_key(row["record"])] = self.accept(
                row["record"], data, row["attempt_id"], row.get("frames"))
            self.event({"event": "validation", "attempt_id": row["attempt_id"],
                        "valid": True, "raw_schema_valid": not data.get('normalization_notes'),
                        "recovered": True, "parse_seconds": 0})
            recovered += 1
        return recovered

    def unfinished(self) -> set[str]:
        events = read_jsonl(self.ledger)
        starts = {row["attempt_id"]: row["item_key"] for row in events
                  if row["event"] == "started"}
        finished = {row["attempt_id"] for row in events
                    if row["event"] in ("response", "error")}
        return {key for attempt, key in starts.items() if attempt not in finished}

    def summarize(self) -> dict:
        events = read_jsonl(self.ledger)
        starts = [row for row in events if row["event"] == "started"]
        responses = [row for row in events if row["event"] == "response"]
        errors = [row for row in events if row["event"] == "error"]
        validations = [row for row in events if row["event"] == "validation"]
        valid_ids = {row["attempt_id"] for row in validations if row["valid"]}
        raw_valid_ids = {row['attempt_id'] for row in validations
                         if row['valid'] and row.get('raw_schema_valid', True)}
        latencies = [row["timings"]["request_seconds"] for row in responses + errors
                     if "request_seconds" in row.get("timings", {})]
        costs = [row["accounting"]["cost"] for row in responses
                 if row["accounting"]["cost"] is not None]
        executions = [row for row in events if row["event"] == "execution_end"]
        known = {row["attempt_id"] for row in responses}
        first = {row["attempt_id"] for row in starts if row["attempt"] == 1}
        token_fields = ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens")
        usage_values = {key:[row['accounting'][key] for row in responses
                             if row['accounting'].get(key) is not None] for key in token_fields}
        usage = {key:sum(values) if values else None for key,values in usage_values.items()}
        summary = {
            "job": self.directory.name, "updated_utc": utc_now(),
            "attempts": len(starts), "responses": len(responses), "errors": len(errors),
            "valid_responses": len(valid_ids), "accepted_videos_or_scenes": len(self.annotations()),
            "first_attempt_success_rate": len(first & valid_ids) / len(first) if first else None,
            "schema_valid_rate": len(raw_valid_ids) / len(responses) if responses else None,
            "normalized_response_count": len(valid_ids - raw_valid_ids),
            "retry_attempts": sum(row["attempt"] > 1 for row in starts),
            "retry_rate": sum(row["attempt"] > 1 for row in starts) / len(starts) if starts else None,
            "known_cost": sum(costs), "currency": "CNY",
            "unknown_cost_attempts": len(starts) - len(costs),
            "unresolved_attempts": len({row["attempt_id"] for row in starts} - known -
                                       {row["attempt_id"] for row in errors}),
            "usage": usage, "usage_available_responses": {
                key:len(values) for key,values in usage_values.items()},
            "latency_seconds": {"mean": float(np.mean(latencies)),
                                "p50": float(np.percentile(latencies, 50)),
                                "p95": float(np.percentile(latencies, 95))} if latencies else {},
            "executions": executions,
        }
        write_json(self.directory / "summary.json", summary)
        return summary


def account_usage(raw: dict, price: dict, timestamp: float | None = None) -> dict:
    schedule = price.get('schedule')
    if schedule:
        # Preserve the effective tariff with each response, including the time basis.
        moment = raw.get('created', timestamp)
        moment = dt.datetime.fromtimestamp(moment, dt.timezone.utc) if isinstance(
            moment, (int, float)) else dt.datetime.now(dt.timezone.utc)
        local = moment.astimezone(ZoneInfo(schedule['timezone']))
        day = local.date().isoformat()
        known_calendar = local.year in schedule['calendar_years']
        holiday = any(start <= day <= end for start, end in schedule['holidays'])
        clock = local.strftime('%H:%M')
        peak = (not known_calendar or (local.weekday() < 5 and not holiday and any(
            start <= clock < end for start, end in schedule['peak_hours'])))
        period = 'peak' if peak else 'off_peak'
        price = {**{key: value for key, value in price.items() if key != 'schedule'},
                 **schedule[period], 'period': period, 'priced_at': moment.isoformat(),
                 'time_basis': 'response_created' if isinstance(raw.get('created'), (int, float))
                 else 'request_started' if timestamp is not None else 'accounting_time',
                 'calendar_known': known_calendar}
    usage = raw.get("usage") or {}
    inputs = usage.get("prompt_tokens", usage.get("input_tokens"))
    outputs = usage.get("completion_tokens", usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
    out_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    reasoning = out_details.get("reasoning_tokens", usage.get("reasoning_tokens"))
    cost = None
    if inputs is not None and outputs is not None and price:
        billed_cached = cached or 0
        cost = (max(0, inputs - billed_cached) * price["input"] +
                billed_cached * price.get("cached_input", price["input"]) +
                outputs * price["output"]) / 1_000_000
    # Reasoning is a completion subset for this protocol; never add it a second time.
    return {"input_tokens": inputs, "output_tokens": outputs, "cached_tokens": cached,
            "reasoning_tokens": reasoning, "cost": cost, "price": price,
            "cost_kind": ("unknown" if cost is None else
                          "usage_priced_assuming_uncached" if cached is None else "usage_priced")}
