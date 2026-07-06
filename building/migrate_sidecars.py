"""One-shot: move scale runs to the geo-task layout and rewrite model identities.

    layout:  models/<scale>/<model>/size_x<m>/k<k>/sample_<i>/  ->  models/<scale>/k<k>/sample_<i>/<model>/size_x<m>/
    identity: {name, multiplier}  ->  {base_model, multi}

Idempotent: run once, then delete.

    python building/migrate_sidecars.py [root]   # root defaults to models/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def rewrite_identity(obj: object) -> tuple[object, int]:
    if isinstance(obj, dict):
        if set(obj) == {"name", "multiplier"}:
            return {"base_model": obj["name"], "multi": obj["multiplier"]}, 1
        out, n = {}, 0
        for k, v in obj.items():
            out[k], c = rewrite_identity(v)
            n += c
        return out, n
    if isinstance(obj, list):
        out, n = [], 0
        for v in obj:
            mv, c = rewrite_identity(v)
            out.append(mv)
            n += c
        return out, n
    return obj, 0


def relayout(root: Path) -> tuple[int, int]:
    moved, touched = 0, set()
    for src in sorted(root.glob("*/*/size_x*/k*/sample_*")):
        if not src.is_dir():
            continue
        collection, model, size_x, k, sample = src.relative_to(root).parts
        dst = root / collection / k / sample / model / size_x
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        touched.add(root / collection)
        moved += 1
    removed = 0
    for collection in touched:
        for d in sorted(collection.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                removed += 1
    return moved, removed


def main(root: Path) -> None:
    moved, removed = relayout(root)
    rewritten = 0
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text())
        out, n = rewrite_identity(data)
        if n:
            path.write_text(json.dumps(out, indent=2) + "\n")
            rewritten += n
    print(f"moved {moved} run dir(s) | removed {removed} empty dir(s) | rewrote {rewritten} identities")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models"))
