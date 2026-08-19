#!/usr/bin/env python3
"""
Build fd/manifest.jsonl — a slim index of all FD stereo conversations for
the explorer Space. One JSONL row per (conversation, orientation).

Reads: dir of <conv>__oriA/oriB.wav+.json files
Writes: manifest.jsonl with only the fields the explorer needs (no words, no
        per-turn full text — those are loaded on-demand from the paired JSON).

Usage:
    python scripts/build_fd_manifest.py \\
        --stereo-dir data/full_duplex_stereo_v3 \\
        --out data/full_duplex_stereo_v3/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _preview(turns: list[dict], max_chars: int = 240) -> str:
    """Concat first few turns' text, capped to max_chars."""
    parts = []
    total = 0
    for t in turns:
        text = (t.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"{t.get('speaker', '?').split('_')[-1]}: {text}")
        total += len(text)
        if total >= max_chars:
            break
    joined = " · ".join(parts)
    if len(joined) > max_chars + 40:
        joined = joined[:max_chars + 40].rsplit(" ", 1)[0] + "…"
    return joined


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stereo-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    jsons = sorted(args.stereo_dir.glob("*.json"))
    if not jsons:
        raise SystemExit(f"no .json files under {args.stereo_dir}")

    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for jp in jsons:
            d = json.loads(jp.read_text(encoding="utf-8"))
            row = {
                "conversation_id":  d["conversation_id"],
                "source_clip":      d["source_clip"],
                "source_url":       d.get("source_url"),
                "source_channel":   d.get("source_channel"),
                "source_genre":     d.get("source_genre"),
                "duration_s":       d["duration_s"],
                "n_turns":          d["n_turns"],
                "unique_speakers":  d["unique_speakers"],
                "channels":         d["channels"],
                "orientation":      "A" if d["conversation_id"].endswith("__oriA") else "B",
                "wav_path":         f"fd/{'oriA' if '__oriA' in jp.name else 'oriB'}/{jp.stem}.wav",
                "json_path":        f"fd/{'oriA' if '__oriA' in jp.name else 'oriB'}/{jp.name}",
                "preview":          _preview(d["turns"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[done] wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
