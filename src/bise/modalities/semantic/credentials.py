"""Local API-key pools with persistent balance-exhaustion quarantine."""

import contextlib
import datetime as dt
import fcntl
import json
import os
import tempfile
import threading
from pathlib import Path

from bise.common.paths import resolve_path


_LOCK = threading.RLock()


class NoActiveKeys(RuntimeError):
    pass


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        descriptor = os.open(str(path) + '.lock', os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    if not path.exists():
        raise NoActiveKeys(f'Create the local API-key pool: {path}')
    data = json.loads(path.read_text())
    for zone in ('active', 'overdue'):
        if not isinstance(data.get(zone), list) or any(
                not isinstance(row, dict) or not isinstance(row.get('key'), str)
                or not row['key'].strip() for row in data[zone]):
            raise ValueError(f'Invalid {zone} list in API-key pool: {path}')
    return data


def _write(path: Path, data: dict) -> None:
    # NamedTemporaryFile uses mode 0600; replacement never exposes a public file.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def add_keys(path: str | Path, keys: list[str]) -> None:
    """Append new credentials without reactivating duplicates or overdue keys."""
    path = resolve_path(path)
    with _locked(path):
        data = _read(path) if path.exists() else {'active': [], 'overdue': []}
        known = {row['key'] for zone in ('active', 'overdue') for row in data[zone]}
        for key in keys:
            key = key.strip()
            if key and key not in known:
                data['active'].append({'id': f'key{len(known) + 1}', 'key': key})
                known.add(key)
        _write(path, data)


def active_key(path: str | Path) -> str:
    path = resolve_path(path)
    with _locked(path):
        data = _read(path)
        overdue = {row['key'] for row in data['overdue']}
        for row in data['active']:
            if row['key'] not in overdue:
                return row['key']
    raise NoActiveKeys(f'No usable API keys remain; add a new key to active in {path}')


def mark_overdue(path: str | Path, key: str, code: str = '1113') -> bool:
    """Move a key exactly once, including concurrent failures of the same key."""
    path = resolve_path(path)
    with _locked(path):
        data = _read(path)
        if any(row['key'] == key for row in data['overdue']):
            return False
        entry = next((row for row in data['active'] if row['key'] == key), None)
        if entry is None:
            return False
        data['active'] = [row for row in data['active'] if row['key'] != key]
        data['overdue'].append({**entry, 'code': str(code),
                                'since': dt.datetime.now(dt.timezone.utc).isoformat()})
        _write(path, data)
        return True


def provider_key(provider: dict) -> str:
    if provider.get('stub'):
        return ''
    if provider.get('key_pool'):
        # A configured pool is authoritative; environment keys cannot bypass quarantine.
        return active_key(provider['key_pool'])
    names = provider.get('api_key_env', [])
    names = [names] if isinstance(names, str) else names
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    raise ValueError(f"Set API key environment variable: {' or '.join(names)}")


def redact(provider: dict, value: str) -> str:
    names = provider.get('api_key_env', [])
    names = [names] if isinstance(names, str) else names
    secrets = {os.environ[name] for name in names if os.environ.get(name)}
    if provider.get('key_pool'):
        path = resolve_path(provider['key_pool'])
        if path.exists():
            with _locked(path):
                data = _read(path)
                secrets.update(row['key'] for zone in ('active', 'overdue') for row in data[zone])
    for secret in sorted(secrets, key=len, reverse=True):
        value = value.replace(secret, '[redacted]')
    return value
