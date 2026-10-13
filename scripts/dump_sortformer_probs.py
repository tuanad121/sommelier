#!/usr/bin/env python3
"""Dump Sortformer's raw per-frame speaker activity probabilities for 5 clips.

For each clip:
  - Run nvidia/diar_sortformer_4spk-v1 with include_tensor_outputs=True
  - Save: segments (start, end, speaker), raw probs tensor (T_frames × 4 speakers),
    frame stride, and per-frame timestamps
  - Output goes to <clip-dir>/sortformer_probs.json + sortformer_probs.npy

Designed as a one-shot diagnostic. Not meant to replace the pipeline's
existing diarization.

Usage (on cluster):
  python scripts/dump_sortformer_probs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import nemo.collections.asr as nemo_asr

RAW_DIR = Path("/home/tuanda67/projects/sommelier/data/raw")
OUT_DIR = Path("/home/tuanda67/projects/sommelier/data/raw/_final/-sepreformer-True-demucs-True-vad-True-diaModel-dia3-initPrompt-True-merge_gap-2.0-seg_th-0.15-cl_min-10-cl-th-0.5-LLM-case_2")


def discover_clips() -> list[str]:
    """Auto-discover all clips under OUT_DIR (skip _excluded). Idempotent:
    clips with existing sortformer_probs.json are skipped."""
    clips = []
    for d in sorted(OUT_DIR.iterdir()):
        if not d.is_dir() or d.name == "_excluded":
            continue
        if (d / "sortformer_probs.json").exists():
            continue
        if not (RAW_DIR / f"{d.name}.wav").exists():
            continue
        clips.append(d.name)
    return clips


def main() -> None:
    print("Loading Sortformer 4spk-v1...", file=sys.stderr)
    model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_sortformer_4spk-v1"
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    clips_to_run = discover_clips()
    if not clips_to_run:
        print("nothing to do — all clips already have sortformer_probs.json", file=sys.stderr)
        return
    wavs = [(c, str(RAW_DIR / f"{c}.wav")) for c in clips_to_run]
    print(f"running on {len(wavs)} clips (skipping any already done)", file=sys.stderr)

    # Diarize all 5 with tensor outputs
    audio_paths = [w for _, w in wavs]
    segments_list, probs_list = model.diarize(
        audio=audio_paths, batch_size=1, include_tensor_outputs=True, verbose=False
    )

    # Write per-clip outputs
    for (clip, wpath), segs, probs in zip(wavs, segments_list, probs_list):
        # segs is a list of strings "begin end speaker_index"
        # probs is a tensor shape (T_frames, n_speakers) — raw probabilities
        clip_dir = OUT_DIR / clip
        clip_dir.mkdir(exist_ok=True)

        probs_np = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
        # Squeeze leading batch dim if present (returned shape may be (1, T, n_spks))
        while probs_np.ndim > 2:
            probs_np = probs_np.squeeze(0)
        if probs_np.ndim != 2:
            print(f"  ! unexpected probs shape {probs_np.shape}", file=sys.stderr)
            continue

        # Sortformer uses frame stride = window_stride * subsampling_factor
        # default = 0.01 * 8 = 0.08s per frame
        frame_stride_s = 0.08

        parsed_segs = []
        for s in segs:
            parts = s.strip().split()
            if len(parts) >= 3:
                spk = parts[2]
                # API returns "speaker_0" format
                if spk.startswith("speaker_"):
                    spk_idx = int(spk[len("speaker_"):])
                else:
                    try:
                        spk_idx = int(spk)
                    except ValueError:
                        spk_idx = -1
                parsed_segs.append({
                    "start": float(parts[0]),
                    "end": float(parts[1]),
                    "speaker_idx": spk_idx,
                    "speaker_label": spk,
                })

        # Save raw probs as .npy (small enough — 240s × 12.5fps × 4 speakers × 4 bytes ≈ 48KB)
        np.save(clip_dir / "sortformer_probs.npy", probs_np.astype(np.float32))

        # Compute per-frame summary stats for quick consumption
        n_frames, n_spks = probs_np.shape
        active_counts = (probs_np > 0.5).sum(axis=1)
        overlap_frames = (active_counts >= 2).sum()
        single_frames  = (active_counts == 1).sum()
        silent_frames  = (active_counts == 0).sum()

        meta = {
            "clip": clip,
            "audio_path": wpath,
            "frame_stride_s": frame_stride_s,
            "n_frames": n_frames,
            "n_speakers": n_spks,
            "segments_parsed": parsed_segs,
            "stats": {
                "duration_s": round(n_frames * frame_stride_s, 2),
                "silent_frames": int(silent_frames),
                "single_speaker_frames": int(single_frames),
                "overlap_frames": int(overlap_frames),
                "overlap_pct": round(overlap_frames / max(n_frames, 1) * 100, 2),
                "silent_pct": round(silent_frames / max(n_frames, 1) * 100, 2),
            },
        }
        (clip_dir / "sortformer_probs.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
        print(
            f"  {clip}: {n_frames} frames @ {frame_stride_s}s, "
            f"{meta['stats']['overlap_pct']:.1f}% co-speech, "
            f"{meta['stats']['silent_pct']:.1f}% silent",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
