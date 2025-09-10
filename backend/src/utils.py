"""
Shared utilities for Blackline forensic tool scripts.

Includes:
- JSONL asset reader (deduplicated on stored_path)
- Safe subprocess runner
- ffprobe/exiftool wrappers
- ffprobe summary helpers
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


def read_unique_assets(audit_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Yield one record per asset from the audit log (deduplicated).

    Backwards/forwards compatibility:
    - If records already contain `stored_path` (and optional `store_root`/`mime`),
      pass them through as-is.
    - If simplified records only contain `sha256` (no `stored_path`), reconstruct
      `stored_path` by inspecting the content-addressed store under
      `<data_root>/raw/<sha256>/<filename>`, where `data_root` is inferred from the
      audit path (".../data/audit/ingest_log.jsonl" → data root ".../data").
    """
    seen: set[str] = set()
    data_root = audit_path.parent.parent  # .../data
    default_store_root = str(data_root / "raw")
    with open(audit_path, encoding="utf-8") as r:
        for line in r:
            if not line.strip():
                continue
            try:
                rec: Dict[str, Any] = json.loads(line)
            except Exception:
                continue

            stored_path = rec.get("stored_path")
            store_root = rec.get("store_root")

            if not stored_path:
                sha = rec.get("sha256")
                if not sha:
                    continue
                # Infer store root from audit path if not present in record
                store_root = store_root or default_store_root
                sha_dir = Path(store_root) / sha
                filename = None
                try:
                    for child in sha_dir.iterdir():
                        if child.is_file():
                            filename = child.name
                            break
                except Exception:
                    filename = None
                if filename is None:
                    # Fall back to a placeholder to keep the record traceable; downstream
                    # validators will mark it as missing if it doesn't exist.
                    filename = sha
                stored_path = f"{sha}/{filename}"
                rec["stored_path"] = stored_path
                rec["store_root"] = store_root

            # Use stored_path if present; otherwise dedupe by sha256
            dedupe_key = stored_path or rec.get("sha256")
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            yield rec


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run a subprocess command returning the CompletedProcess (never raises)."""
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return e


def ffprobe_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed ffprobe JSON or None if ffprobe missing/failed."""
    if shutil.which("ffprobe") is None:
        return None
    p = run_command(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except Exception:
        return None


def exiftool_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed ExifTool JSON (single-object) or None if missing/failed."""
    if shutil.which("exiftool") is None:
        return None
    p = run_command(["exiftool", "-json", "-n", str(path)])
    if p.returncode != 0:
        return None
    try:
        data = json.loads(p.stdout)
        return data[0] if isinstance(data, list) and data else data
    except Exception:
        return None


def parse_rate(value: Optional[str]) -> Optional[float]:
    """Parse rates like '30000/1001' or numeric strings to float."""
    if not value:
        return None
    try:
        if "/" in value:
            num_s, den_s = value.split("/")
            num, den = int(num_s), int(den_s)
            return (num / den) if den else float(num)
        return float(value)
    except Exception:
        return None


def summarize_ffprobe(probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract quick fields for convenience (width/height/fps/codec/duration)."""
    if not probe:
        return {}
    streams = probe.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        return {}
    fmt = probe.get("format") or {}
    return {
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate")),
        "codec": v.get("codec_name"),
        "duration_s": float(fmt.get("duration")) if fmt.get("duration") else None,
        "nb_streams": fmt.get("nb_streams"),
    }


def ffmpeg_decode_ok(path: Path) -> Optional[bool]:
    """Return True if a decode dry-run succeeds, False if errors, None if ffmpeg missing."""
    if shutil.which("ffmpeg") is None:
        return None
    p = run_command(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(path), "-f", "null", "-"])
    return p.returncode == 0


