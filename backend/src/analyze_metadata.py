"""
Analyze metadata from probe.jsonl and exif to emit structured PASS/WARN/FAIL checks
and a final suspicion score per asset (13-step plan).
"""

import argparse, json, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import parse_rate


def parse_ratio(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        s = str(value)
        if ":" in s:
            a, b = s.split(":", 1)
            return (float(a) / float(b)) if float(b) else float(a)
        if "/" in s:
            a, b = s.split("/", 1)
            return (float(a) / float(b)) if float(b) else float(a)
        return float(s)
    except Exception:
        return None


def pick_primary_video(streams: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if not videos:
        return None
    for s in videos:
        disp = s.get("disposition") or {}
        if disp.get("default") == 1:
            return s
    return videos[0]


def get_container_fields(fmt: Dict[str, Any]) -> Dict[str, Any]:
    tags = fmt.get("tags") or {}
    return {
        "format_name": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
        "nb_streams": fmt.get("nb_streams"),
        "duration": fmt.get("duration"),
        "size": fmt.get("size"),
        "bit_rate": fmt.get("bit_rate"),
        "probe_score": fmt.get("probe_score"),
        "tags": {
            "major_brand": tags.get("major_brand"),
            "minor_version": tags.get("minor_version"),
            "compatible_brands": tags.get("compatible_brands"),
            "encoder": tags.get("encoder"),
        },
    }


def expected_mime_for_container(container: Dict[str, Any]) -> Optional[str]:
    fmt_name = (container.get("format_name") or "").lower()
    tags = container.get("tags") or {}
    major_brand = (tags.get("major_brand") or "").lower()
    if "webm" in fmt_name or major_brand == "webm":
        return "video/webm"
    if "matroska" in fmt_name or "mkv" in fmt_name:
        return "video/x-matroska"
    if "mp4" in fmt_name or major_brand in {"isom", "mp42", "mp41"}:
        return "video/mp4"
    if "mov" in fmt_name or "quicktime" in fmt_name:
        return "video/quicktime"
    return None


def add_check(checks: List[Dict[str, str]], code: str, status: str, reason: str) -> None:
    checks.append({"code": code, "status": status, "reason": reason})


def rel_diff(a: float, b: float) -> float:
    denom = max(1e-9, max(abs(a), abs(b)))
    return abs(a - b) / denom


def to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def analyze_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    probe = rec.get("probe") or {}
    exif = rec.get("exif") or {}
    fmt = probe.get("format") or {}
    streams = probe.get("streams") or []
    checks: List[Dict[str, str]] = []

    # 1) Identity
    identity = {
        "asset_id": rec.get("asset_id"),
        "sha256": rec.get("sha256"),
        "stored_path": rec.get("stored_path"),
        "store_root": rec.get("store_root"),
        "mime": rec.get("mime"),
        "when": rec.get("when"),
        "tool_versions": {"ingest": (rec.get("tool_versions") or {}).get("ingest")},
    }

    # 2) Container sanity
    container = get_container_fields(fmt)
    nb_streams = container.get("nb_streams")
    if isinstance(nb_streams, str):
        try:
            nb_streams = int(nb_streams)
        except Exception:
            nb_streams = None
    if not nb_streams or nb_streams <= 0:
        add_check(checks, "FAIL_NOSTREAMS", "FAIL", "nb_streams <= 0")
    ps = to_float(container.get("probe_score"))
    if ps is not None and ps < 50:
        add_check(checks, "WARN_PROBE_SCORE", "WARN", f"low probe_score={ps}")
    exp_mime = expected_mime_for_container(container)
    if exp_mime and identity.get("mime") and exp_mime != identity.get("mime"):
        add_check(checks, "WARN_CONTAINER_MIME", "WARN", f"mime={identity.get('mime')} vs container={exp_mime}")

    # 3) Primary video stream
    v = pick_primary_video(streams)
    video = {}
    if v:
        video = {
            "index": v.get("index"),
            "codec_name": v.get("codec_name"),
            "codec_long_name": v.get("codec_long_name"),
            "profile": v.get("profile"),
            "level": v.get("level"),
            "width": v.get("width"),
            "height": v.get("height"),
            "coded_width": v.get("coded_width"),
            "coded_height": v.get("coded_height"),
            "pix_fmt": v.get("pix_fmt"),
            "field_order": v.get("field_order"),
            "chroma_location": v.get("chroma_location"),
            "has_b_frames": v.get("has_b_frames"),
            "refs": v.get("refs"),
            "is_avc": v.get("is_avc"),
            "nal_length_size": v.get("nal_length_size"),
            "extradata_size": v.get("extradata_size"),
            "avg_frame_rate": v.get("avg_frame_rate"),
            "r_frame_rate": v.get("r_frame_rate"),
            "nb_frames": v.get("nb_frames"),
            "start_time": v.get("start_time"),
            "time_base": v.get("time_base"),
            "sample_aspect_ratio": v.get("sample_aspect_ratio"),
            "display_aspect_ratio": v.get("display_aspect_ratio"),
            "disposition": v.get("disposition"),
        }

    # 4) Timing and frame rate consistency
    if v:
        fps = parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate"))
        duration = to_float(fmt.get("duration"))
        nb_frames = None
        try:
            nb_frames = int(v.get("nb_frames")) if v.get("nb_frames") is not None else None
        except Exception:
            nb_frames = None
        if fps and duration and nb_frames is not None:
            expected_frames = fps * duration
            if rel_diff(nb_frames, expected_frames) > 0.02:
                add_check(checks, "WARN_TIMING", "WARN", "frame count vs duration mismatch")
        afr = parse_rate(v.get("avg_frame_rate"))
        rfr = parse_rate(v.get("r_frame_rate"))
        if afr and rfr and rel_diff(afr, rfr) > 0.01:
            add_check(checks, "WARN_VFR", "WARN", "avg_frame_rate differs from r_frame_rate")
        st = to_float(v.get("start_time"))
        if st is not None and abs(st) > 0.5:
            add_check(checks, "WARN_START_TIME", "WARN", f"start_time={st}")
        tb = v.get("time_base")
        tb_val = parse_ratio(tb) if isinstance(tb, str) else None
        if tb_val is not None and (tb_val < 1e-6 or tb_val > 1.0):
            add_check(checks, "WARN_TIMEBASE", "WARN", f"time_base={tb}")

    # 5) Geometry & aspect ratio coherence
    if v:
        w, h = to_float(v.get("width")), to_float(v.get("height"))
        sar = parse_ratio(v.get("sample_aspect_ratio")) or 1.0
        dar_expected = (w / h) * sar if (w and h) else None
        dar_tag = parse_ratio(v.get("display_aspect_ratio"))
        if dar_expected and dar_tag and rel_diff(dar_expected, dar_tag) > 0.02:
            add_check(checks, "WARN_DAR", "WARN", "display_aspect_ratio mismatch")
        cw, ch = to_float(v.get("coded_width")), to_float(v.get("coded_height"))
        if (cw and w and cw < w) or (ch and h and ch < h):
            add_check(checks, "WARN_CROPPING", "WARN", "coded_* smaller than displayed dims")

    # 6) Codec/GOP plausibility (H.264 sample rules)
    if v and (str(v.get("codec_name") or "").lower() in {"h264", "avc1"}):
        profile = str(v.get("profile") or "")
        has_b = v.get("has_b_frames")
        if profile.lower() == "high" and has_b == 0:
            add_check(checks, "WARN_GOP", "WARN", "High profile without B-frames")
        refs = v.get("refs")
        try:
            if refs is not None and int(refs) > 16:
                add_check(checks, "WARN_GOP_REFS", "WARN", f"refs={refs}")
        except Exception:
            pass
        is_avc = v.get("is_avc")
        nls = v.get("nal_length_size")
        try:
            nls_i = int(nls) if nls is not None else None
        except Exception:
            nls_i = None
        if is_avc == 1 and (nls_i not in {1, 2, 4}):
            add_check(checks, "WARN_BITSTREAM", "WARN", f"is_avc=1 but nal_length_size={nls}")
        if is_avc in {0, None} and nls_i is not None:
            add_check(checks, "WARN_BITSTREAM", "WARN", "nal_length_size present but is_avc != 1")
        pix = str(v.get("pix_fmt") or "").lower()
        if pix and pix not in {"yuv420p", "yuvj420p", "nv12", "yuv420p10le"}:
            add_check(checks, "WARN_PIXFMT", "WARN", f"pix_fmt={pix}")

    # 7) Bitrate realism
    if v:
        w, h = to_float(v.get("width")), to_float(v.get("height"))
        fps = parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate"))
        vbr = to_float(v.get("bit_rate"))
        fbr = to_float(fmt.get("bit_rate"))
        br = vbr or fbr
        if w and h and fps and br:
            bpp = br / (w * h * fps)
            if bpp < 0.03:
                add_check(checks, "WARN_BITRATE_LOW", "WARN", f"bpp={bpp:.4f}")
            elif bpp > 0.25:
                add_check(checks, "WARN_BITRATE_HIGH", "WARN", f"bpp={bpp:.4f}")

    # 8) Audio streams quick sanity
    audio_out: List[Dict[str, Any]] = []
    for a in streams:
        if a.get("codec_type") != "audio":
            continue
        row = {
            "codec_type": a.get("codec_type"),
            "sample_rate": a.get("sample_rate"),
            "channels": a.get("channels"),
            "channel_layout": a.get("channel_layout"),
            "bits_per_sample": a.get("bits_per_sample"),
        }
        audio_out.append(row)
        try:
            sr_ok = int(a.get("sample_rate")) in {44100, 48000}
        except Exception:
            sr_ok = False
        ch_ok = a.get("channels") in {1, 2}
        if not (sr_ok and ch_ok):
            add_check(checks, "WARN_AUDIO", "WARN", "audio sample_rate/channels unusual")

    # 9) Provenance & re-encode fingerprints
    prov = {
        "format_encoder": (fmt.get("tags") or {}).get("encoder"),
        "stream_tags": [],
        "exif_encoder": exif.get("Encoder"),
        "exif_handler_desc": exif.get("HandlerDescription"),
        "exif_handler_vendor": exif.get("HandlerVendorID"),
    }
    for s in streams:
        t = s.get("tags") or {}
        if t:
            prov["stream_tags"].append({
                "handler_name": t.get("handler_name"),
                "vendor_id": t.get("vendor_id"),
                "language": t.get("language"),
            })
    enc_strs = [x for x in [prov.get("format_encoder"), exif.get("Encoder")] if x]
    transcode_hint = any(
        any(token in str(es) for token in ("Lavf", "HandBrake", "ffmpeg", "libavformat"))
        for es in enc_strs
    )
    device_original_claim = any(
        bool(exif.get(k)) for k in (
            "Make",
            "Model",
            "CameraModelName",
            "DeviceModelName",
            "LensModel",
            "LensMake",
        )
    )
    if transcode_hint and device_original_claim:
        add_check(checks, "WARN_TRANSCODE", "WARN", "encoder hints at transcoding vs device-original tags")
    vend = prov.get("exif_handler_vendor")
    if isinstance(vend, str) and vend and len(vend) != 4:
        add_check(checks, "WARN_HANDLER", "WARN", f"vendor_id length={len(vend)}")

    # 10) Timestamp monotonicity
    def parse_dt(s: Any) -> Optional[float]:
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s)
        try:
            txt = str(s)
            for fmt_str in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
                try:
                    return float(time.mktime(time.strptime(txt, fmt_str)))
                except Exception:
                    continue
            return None
        except Exception:
            return None

    fcd = parse_dt(exif.get("FileCreateDate"))
    mcd = parse_dt(exif.get("MediaCreateDate") or exif.get("CreateDate"))
    mod = parse_dt(exif.get("FileModifyDate") or exif.get("ModifyDate"))
    now = time.time()
    if fcd and mcd and fcd > mcd:
        add_check(checks, "WARN_DATES", "WARN", "FileCreateDate > MediaCreateDate")
    if mcd and mod and mcd > mod:
        add_check(checks, "WARN_DATES", "WARN", "MediaCreateDate > ModifyDate")
    if mod and mod > now + 60:
        add_check(checks, "WARN_DATES", "WARN", "ModifyDate in the future")
    exif_dur = to_float(exif.get("Duration"))
    fmt_dur = to_float(fmt.get("duration"))
    if exif_dur and fmt_dur and rel_diff(exif_dur, fmt_dur) > 0.05:
        add_check(checks, "WARN_DURATION", "WARN", "exif.Duration vs format.duration")

    # 11) One cross-check bundle
    if v:
        vw, vh = to_float(v.get("width")), to_float(v.get("height"))
        ew, eh = to_float(exif.get("ImageWidth")), to_float(exif.get("ImageHeight"))
        rot = int(exif.get("Rotation")) if str(exif.get("Rotation") or "").isdigit() else None
        if ew and eh and vw and vh:
            if rot in {90, 270}:
                ew, eh = eh, ew
            if abs(vw - ew) > 2 or abs(vh - eh) > 2:
                add_check(checks, "WARN_DIMENSIONS", "WARN", "EXIF vs stream dimensions")
        xfps = to_float(exif.get("VideoFrameRate"))
        fps = parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate")) if v else None
        if xfps and fps and rel_diff(xfps, fps) > 0.02:
            add_check(checks, "WARN_FPS_TAG", "WARN", "EXIF frame rate vs computed fps")
        xbr = to_float(exif.get("AvgBitrate"))
        fbr = to_float(fmt.get("bit_rate"))
        if xbr and fbr and rel_diff(xbr, fbr) > 0.15:
            add_check(checks, "WARN_AVG_BITRATE", "WARN", "EXIF AvgBitrate vs format.bit_rate")

    # 12) Dispositions & flags
    if v:
        disp = (v.get("disposition") or {})
        unexpected = []
        for k in ("attached_pic", "forced", "timed_thumbnails", "captions"):
            if disp.get(k) in {1, True}:
                unexpected.append(k)
        if unexpected:
            add_check(checks, "WARN_DISPOSITION", "WARN", ",".join(unexpected))

    # 13) Summarize & score
    fps = parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate")) if v else None
    summary = {
        "width": v.get("width") if v else None,
        "height": v.get("height") if v else None,
        "fps": fps,
        "codec": v.get("codec_name") if v else None,
        "duration_s": to_float(fmt.get("duration")),
        "nb_streams": container.get("nb_streams"),
    }

    score = 0
    for c in checks:
        if c.get("status") == "FAIL":
            score += 2
        elif c.get("status") == "WARN":
            score += 1

    out = {
        "identity": identity,
        "container": container,
        "video": video,
        "audio": audio_out,
        "provenance": {
            "encoder": (fmt.get("tags") or {}).get("encoder"),
            "exif": {
                "Encoder": exif.get("Encoder"),
                "HandlerDescription": exif.get("HandlerDescription"),
                "HandlerVendorID": exif.get("HandlerVendorID"),
            },
        },
        "checks": checks,
        "score": score,
        "summary": summary,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze metadata from probe.jsonl and emit checks + score")
    ap.add_argument("--probe", default="backend/data/derived/probe.jsonl", help="input probe JSONL")
    ap.add_argument("--out", default="backend/data/derived/metadata.jsonl", help="output JSONL path")
    ap.add_argument("--limit", type=int, default=None, help="analyze at most N assets")
    args = ap.parse_args()

    in_path = Path(args.probe)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(in_path, encoding="utf-8") as r, open(out_path, "w", encoding="utf-8") as w:
        for line in r:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            result = analyze_record(rec)
            w.write(json.dumps(result) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} rows -> {out_path}")


if __name__ == "__main__":
    main()


