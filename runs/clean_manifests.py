"""Remove SHA-256 fields from existing data manifests without changing their splits."""

import argparse
import json
from _bootstrap import bootstrap
bootstrap()

from bise.common.fingerprints import without_sha256, write_json
from bise.common.paths import resolve_path
from bise.data.rh20t.splits import protocol_hash, protocol_metadata_path


def clean_manifests(root, runs_root=None, check=False):
    root = resolve_path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Manifest directory missing: {root}")
    changes = {}
    for path in sorted(root.rglob("*.json")):
        original = json.loads(path.read_text())
        cleaned = without_sha256(original)
        if cleaned != original:
            changes[path] = cleaned

    bindings, old_hashes = {}, {}
    if runs_root is not None:
        for path in sorted(resolve_path(runs_root).rglob("protocols.json")):
            records = json.loads(path.read_text())
            affected = {}
            for cfg, record in records.items():
                source = protocol_metadata_path(record["path"])
                if source not in changes:
                    continue
                if source not in old_hashes:
                    old_hashes[source] = protocol_hash(source)
                # Only migrate a binding that matched the exact pre-cleanup protocol.
                if record.get("sha256") == old_hashes[source]:
                    affected[cfg] = source
            if affected:
                bindings[path] = (records, affected)

    if not check:
        for path, cleaned in changes.items():
            write_json(path, cleaned)
        new_hashes = {source: protocol_hash(source) for source in old_hashes}
        for path, (records, affected) in bindings.items():
            for cfg, source in affected.items():
                records[cfg]["sha256"] = new_hashes[source]
            write_json(path, records)
    return {"manifests": len(changes), "protocol_bindings": len(bindings), "check": check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/manifests")
    parser.add_argument("--runs-root", help="Update matching protocol bindings in existing runs")
    parser.add_argument("--check", action="store_true", help="Report changes without writing files")
    args = parser.parse_args()
    print(json.dumps(clean_manifests(args.root, args.runs_root, args.check), ensure_ascii=False))


if __name__ == "__main__":
    main()
