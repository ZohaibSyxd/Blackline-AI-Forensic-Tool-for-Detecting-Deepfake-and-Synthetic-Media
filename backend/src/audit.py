"""
Lightweight append-only auditing utilities.

Provides:
- append_audit_row(audit_path, rec): append a JSON object to a JSONL file
- audit_step(event, ..., inputs={}, params={}) context manager that writes
  START and END rows to backend/data/audit/pipeline_log.jsonl

All records contain exactly the specified keys for this project; any None
values and unknown keys are omitted before writing.
"""
from __future__ import annotations

import contextlib
import getpass
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterator, Optional


DEFAULT_AUDIT_DIR = Path("backend/data/audit")
DEFAULT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INGEST_LOG = DEFAULT_AUDIT_DIR / "ingest_log.jsonl"
DEFAULT_PIPELINE_LOG = DEFAULT_AUDIT_DIR / "pipeline_log.jsonl"


def _iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _filter_fields(rec: Dict, allowed_keys: set[str]) -> Dict:
    return {k: v for k, v in rec.items() if k in allowed_keys and v is not None}


def append_audit_row(audit_path: str | Path, rec: Dict) -> None:
    """Append a single JSON object as a line to a JSONL file.

    Ensures parent directory exists. Does not modify the input record except
    for ensuring it's JSON-serializable. Caller is responsible for passing only
    allowed fields for the target log.
    """
    p = Path(audit_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as w:
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _path_size_bytes(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except Exception:
        return None


def _count_rows_for_path(path: Path) -> Optional[int]:
    try:
        ext = path.suffix.lower()
        if ext in (".jsonl", ".logl"):
            # Count non-empty lines
            with path.open("r", encoding="utf-8", errors="ignore") as r:
                return sum(1 for line in r if line.strip())
        if ext == ".json":
            with path.open("r", encoding="utf-8", errors="ignore") as r:
                try:
                    data = json.load(r)
                    if isinstance(data, list):
                        return len(data)
                    return 1
                except Exception:
                    return None
        if ext == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore") as r:
                # Count lines; subtract one for header if present
                lines = [ln for ln in r if ln.strip()]
                if not lines:
                    return 0
                # Assume first line is header if it contains commas and any alpha
                first = lines[0]
                headerish = ("," in first) and any(c.isalpha() for c in first)
                return max(0, len(lines) - (1 if headerish else 0))
    except Exception:
        return None
    return None


@contextlib.contextmanager
def audit_step(
    event: str,
    *,
    audit_path: str | Path = DEFAULT_PIPELINE_LOG,
    user: Optional[str] = None,
    asset_id: Optional[str] = None,
    sha256: Optional[str] = None,
    params: Optional[Dict] = None,
    inputs: Optional[Dict[str, str]] = None,
) -> Iterator[Dict[str, Dict]]:
    """Context manager that logs START and END rows for a pipeline step.

    Usage:
        with audit_step("probe", params=vars(args), inputs={"audit":"path"}) as out:
            ... do work ...
            out["probe"] = {"path": args.out, "rows": N}

    On successful exit, writes END with status=ok, duration_ms, and outputs
    where each entry is of the form {"path","rows","size_bytes"} (rows and
    size_bytes are auto-computed if missing and the path exists). On error,
    writes END with status=error and error=string.
    """
    start_ts = _iso_utc()
    who = user or getpass.getuser()
    start = time.time()
    # START row
    start_row = {
        "event": event,
        "phase": "START",
        "ts": start_ts,
        "user": who,
        "asset_id": asset_id,
        "sha256": sha256,
        "params": dict(params) if params else {},
        "inputs": dict(inputs) if inputs else {},
    }
    # Only allowed keys for pipeline rows
    allowed_start = {"event","phase","ts","user","asset_id","sha256","params","inputs"}
    append_audit_row(audit_path, _filter_fields(start_row, allowed_start))

    outputs: Dict[str, Dict] = {}
    error_msg: Optional[str] = None
    try:
        yield outputs
        status = "ok"
    except Exception as e:  # noqa: PIE786 - re-raise after logging
        status = "error"
        error_msg = str(e)
        raise
    finally:
        end_ts = _iso_utc()
        dur_ms = int(round((time.time() - start) * 1000.0))
        out_clean: Dict[str, Dict] = {}
        if status == "ok":
            for name, meta in (outputs or {}).items():
                # normalize meta
                if isinstance(meta, (str, os.PathLike)):
                    meta = {"path": str(meta)}
                meta = dict(meta or {})
                p = Path(meta.get("path", ""))
                if p and p.exists():
                    meta.setdefault("rows", _count_rows_for_path(p))
                    meta.setdefault("size_bytes", _path_size_bytes(p))
                # retain only the required keys
                meta = {k: v for k, v in meta.items() if k in {"path","rows","size_bytes"} and v is not None}
                out_clean[name] = meta

        end_row = {
            "event": event,
            "phase": "END",
            "ts": end_ts,
            "user": who,
            "asset_id": asset_id,
            "sha256": sha256,
            "status": status,
            "duration_ms": dur_ms,
        }
        if status == "ok" and out_clean:
            end_row["outputs"] = out_clean
        if status == "error" and error_msg:
            end_row["error"] = error_msg

        allowed_end = {"event","phase","ts","user","asset_id","sha256","status","duration_ms","outputs","error"}
        append_audit_row(audit_path, _filter_fields(end_row, allowed_end))
