#!/usr/bin/env python3
"""Per-segment background-music detection using PANNS CNN14.

The pipeline already has PANNS wired up but only runs it when --demucs True
(see main_original_ASR_MoE.py:824 preprocess_segments_with_demucs). We run
with --demucs False to save compute, so BGM info isn't captured. This
script runs PANNS independently as a post-process step, writing a sidecar
JSON the normalizer can use to drop BGM-contaminated segments.

PANNS CNN14: trained on AudioSet (527 sound classes). We threshold on the
"Music" class probability (idx 137 in the AudioSet ontology). A segment is
flagged has_bgm=True if music_prob > BGM_THRESHOLD.

Usage:
  python scripts/check_bgm.py --run-dir <clip-out-dir>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("check_bgm")

BGM_THRESHOLD = 0.3  # PANNS music-prob threshold; matches the pipeline default.


def get_music_class_indices() -> list[int]:
    """AudioSet 'Music' superclass and its prominent children. CNN14's class
    labels are at https://github.com/qiuqiangkong/audioset_tagging_cnn .
    We check the parent 'Music' class (137) plus 'Background music' (216)."""
    return [137, 216]


def load_panns(device: torch.device):
    """Load PANNS CNN14 via panns_inference (already a project dep)."""
    from panns_inference import AudioTagging
    log.info("Loading PANNS CNN14 (AudioTagging)...")
    return AudioTagging(checkpoint_path=None, device=str(device))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=BGM_THRESHOLD,
                   help="music-class probability threshold (default 0.3)")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    clip = run_dir.name
    json_path = run_dir / f"{clip}.json"
    if not json_path.exists():
        raise SystemExit(f"no pipeline JSON at {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        raise SystemExit("no segments")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panns = load_panns(device)
    music_idxs = get_music_class_indices()

    out_segments = []
    n_bgm = 0
    for i, seg in enumerate(segments):
        spk = seg.get("speaker", "?")
        spk_for_path = seg.get("original_speaker", spk)
        audio_file = run_dir / clip / f"{i:05d}_{spk_for_path}.mp3"
        if not audio_file.exists():
            out_segments.append({"index": i, "skipped": "audio_missing"})
            continue
        try:
            audio, _ = librosa.load(str(audio_file), sr=32000, mono=True)  # PANNS wants 32kHz
        except Exception as e:
            out_segments.append({"index": i, "skipped": f"load_failed: {e}"})
            continue
        if len(audio) < 32000 * 0.1:  # too short for tagging
            out_segments.append({"index": i, "skipped": "too_short"})
            continue

        # panns_inference expects (B, T) numpy
        clip_audio = audio[np.newaxis, :].astype(np.float32)
        try:
            clipwise_output, _ = panns.inference(clip_audio)  # (B, 527)
        except Exception as e:
            out_segments.append({"index": i, "skipped": f"panns_failed: {e}"})
            continue

        scores = clipwise_output[0]  # (527,)
        music_score = float(max(scores[idx] for idx in music_idxs))
        has_bgm = music_score > args.threshold

        if has_bgm:
            n_bgm += 1

        out_segments.append({
            "index": i,
            "has_bgm": bool(has_bgm),
            "music_score": music_score,
        })

    log.info(f"Flagged {n_bgm}/{len(segments)} segments as has_bgm "
             f"(threshold={args.threshold})")
    out_path = run_dir / "bgm_check.json"
    out_path.write_text(json.dumps({
        "clip": clip,
        "n_segments": len(segments),
        "n_bgm": n_bgm,
        "threshold": args.threshold,
        "segments": out_segments,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
