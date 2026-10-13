#!/usr/bin/env python3
"""Per-segment speaker-attribution verification.

For each segment in a pipeline output dir, embed the segment's MP3 audio with
pyannote/embedding and compare cosine similarity to every speaker's reference
embedding (built from longest non-overlap segments — same logic as the Phase A
clean-reference builder). If the best-matching speaker is NOT the diarization-
assigned one AND the margin exceeds SPEAKER_ASSIGN_MARGIN, the segment is
marked `is_misattributed` in the sidecar JSON.

Catches sub-1s overlap leaks where the separator was skipped (Phase A rule)
and Sortformer assigned the segment to whichever speaker was supposed to be
talking, but the actual audio is mostly the OTHER speaker bleeding through.

Usage:
  python scripts/verify_speaker_attribution.py \\
    --run-dir data/raw/_final/-sepreformer-True-.../vi_smoke_havesip_khanhvy_ep118

Produces: <run-dir>/attribution_check.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

# Match Phase A constants
REF_PER_SEG_MIN = 1.5
REF_TARGET_DURATION = 5.0
MIN_EMBED_DURATION = 0.15  # ECAPA-TDNN can produce useful embeddings down
                            # to ~150ms; below that the encoder pads or fails.
                            # This is the sub-1s mis-attribution territory we
                            # specifically want to catch.
SPEAKER_ASSIGN_MARGIN = 0.05

log = logging.getLogger("verify_attr")


def load_embedding_model(hf_token: str | None = None):
    """ECAPA-TDNN from speechbrain — more robust to short audio than
    pyannote/embedding (which gives degenerate sims on sub-1s clips)."""
    from speechbrain.inference.speaker import EncoderClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    return classifier, device


def embed_audio(model, device, audio: np.ndarray, sr: int) -> torch.Tensor | None:
    """Return (1, D) embedding or None if audio too short / fails.
    ECAPA-TDNN expects (B, T) mono audio at 16kHz."""
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio) < int(MIN_EMBED_DURATION * 16000):
        return None
    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    try:
        with torch.inference_mode():
            emb = model.encode_batch(tensor)  # (1, 1, D)
        return emb.squeeze(1)  # (1, D)
    except Exception as e:
        log.warning(f"Embedding failed: {e}")
        return None


def load_segment_audio(audio_file: Path) -> tuple[np.ndarray, int] | None:
    """Load an MP3 segment file and return (audio, sample_rate)."""
    if not audio_file.exists():
        return None
    try:
        audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
        return audio.astype(np.float32), sr
    except Exception as e:
        log.warning(f"Failed to load {audio_file}: {e}")
        return None


def build_speaker_references(
    segments: list[dict],
    audio_root: Path,
    model,
    device,
) -> dict[str, torch.Tensor]:
    """Same logic as main_original_ASR_MoE.py's _build_clean_reference_embeddings:
    pick longest non-overlap segments per speaker, embed each, average."""
    # Identify segments that participate in overlaps
    sorted_by_start = sorted(enumerate(segments), key=lambda kv: kv[1].get("start", 0))
    overlap_indices = set()
    for i in range(len(sorted_by_start)):
        idx_i, seg_i = sorted_by_start[i]
        for j in range(i + 1, len(sorted_by_start)):
            idx_j, seg_j = sorted_by_start[j]
            if seg_j.get("start", 0) >= seg_i.get("end", 0):
                break
            overlap = min(seg_i["end"], seg_j["end"]) - max(seg_i["start"], seg_j["start"])
            if overlap >= 0.2:  # same threshold as overlap detector
                overlap_indices.add(idx_i)
                overlap_indices.add(idx_j)

    all_speakers = sorted({s.get("speaker", "?") for s in segments if s.get("speaker", "?") != "?"})
    refs: dict[str, torch.Tensor] = {}

    REF_WINDOW_SEC = 5.0
    REF_MAX_WINDOWS = 6  # cap total accumulated audio at ~30s/speaker

    def _segment_audio_path(idx: int, seg: dict) -> Path:
        """Pipeline JSON usually has audio_file=None. MP3s on disk are named
        `<clip_name>/NNNNN_<speaker>.mp3`. Use the diarization-assigned
        speaker for the filename (NOT post-merge canonical)."""
        audio_file = seg.get("audio_file") or ""
        if not audio_file:
            spk = seg.get("original_speaker", seg.get("speaker", "SPEAKER_00"))
            audio_file = f"{audio_root.name}/{idx:05d}_{spk}.mp3"
        elif not audio_file.startswith(audio_root.name + "/"):
            audio_file = f"{audio_root.name}/{Path(audio_file).name}"
        return audio_root / audio_file

    for speaker in all_speakers:
        # Collect non-overlap segments; sort longest-first so we get the
        # cleanest material first.
        clean_segs = [
            (i, s) for i, s in enumerate(segments)
            if s.get("speaker") == speaker and i not in overlap_indices
        ]
        clean_segs.sort(key=lambda kv: kv[1].get("end", 0) - kv[1].get("start", 0), reverse=True)

        # Concatenate clean audio per speaker (16kHz mono).
        target_samples = int(REF_WINDOW_SEC * REF_MAX_WINDOWS * 16000)
        concat_buf: list[np.ndarray] = []
        accumulated_dur = 0.0
        for i, seg in clean_segs:
            audio_path = _segment_audio_path(i, seg)
            loaded = load_segment_audio(audio_path)
            if not loaded:
                continue
            audio, sr = loaded
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            concat_buf.append(audio.astype(np.float32))
            accumulated_dur += len(audio) / 16000
            if sum(len(a) for a in concat_buf) >= target_samples:
                break

        if not concat_buf:
            log.warning(f"No clean audio for {speaker}; verification will skip its segments")
            continue

        concat_audio = np.concatenate(concat_buf)
        n_samples = len(concat_audio)
        ref_tensor = torch.from_numpy(concat_audio).float().unsqueeze(0).to(device)  # (1, T)
        try:
            with torch.inference_mode():
                emb = model.encode_batch(ref_tensor)  # (1, 1, D)
            refs[speaker] = emb.squeeze(1)  # (1, D)
            log.info(
                f"Reference for {speaker}: {accumulated_dur:.2f}s of clean audio "
                f"({n_samples} samples), encoded with ECAPA-TDNN"
            )
        except Exception as e:
            log.warning(f"Failed to embed reference for {speaker}: {e}")

    return refs


def verify_attribution(run_dir: Path, hf_token: str | None) -> dict:
    json_path = run_dir / f"{run_dir.name}.json"
    if not json_path.exists():
        raise SystemExit(f"no pipeline JSON at {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        raise SystemExit("no segments")

    audio_root = run_dir  # MP3 paths in segments are relative to the clip dir

    log.info(f"Loading pyannote/embedding...")
    model, device = load_embedding_model(hf_token)

    log.info(f"Building reference embeddings for {len({s.get('speaker', '?') for s in segments})} speakers...")
    refs = build_speaker_references(segments, audio_root, model, device)

    if not refs:
        raise SystemExit("could not build any reference embedding")

    def _segment_path(idx: int, seg: dict) -> Path:
        audio_file = seg.get("audio_file") or ""
        if not audio_file:
            spk = seg.get("original_speaker", seg.get("speaker", "SPEAKER_00"))
            audio_file = f"{run_dir.name}/{idx:05d}_{spk}.mp3"
        elif not audio_file.startswith(run_dir.name + "/"):
            audio_file = f"{run_dir.name}/{Path(audio_file).name}"
        return audio_root / audio_file

    log.info(f"Verifying attribution for {len(segments)} segments...")
    results = []
    n_misattr = 0
    for i, seg in enumerate(segments):
        assigned = seg.get("speaker", "?")
        audio_path = _segment_path(i, seg)
        loaded = load_segment_audio(audio_path)
        if not loaded:
            results.append({"index": i, "skipped": "audio_load_failed"})
            continue
        audio, sr = loaded
        emb = embed_audio(model, device, audio, sr)
        if emb is None:
            results.append({"index": i, "skipped": "too_short"})
            continue

        sims = {}
        for spk, ref in refs.items():
            sim = torch.nn.functional.cosine_similarity(emb, ref, dim=1).item()
            sims[spk] = sim

        best_speaker = max(sims, key=sims.get)
        best_sim = sims[best_speaker]
        assigned_sim = sims.get(assigned, None)

        is_misattr = False
        if assigned_sim is not None and assigned != best_speaker:
            margin = best_sim - assigned_sim
            if margin > SPEAKER_ASSIGN_MARGIN:
                is_misattr = True
                n_misattr += 1

        results.append({
            "index": i,
            "assigned_speaker": assigned,
            "best_match_speaker": best_speaker,
            "sim_to_assigned": assigned_sim,
            "sim_to_best": best_sim,
            "is_misattributed": is_misattr,
            "all_sims": sims,
        })

    log.info(f"Flagged {n_misattr}/{len(segments)} segments as misattributed")

    out = {
        "clip": run_dir.name,
        "n_segments": len(segments),
        "n_misattributed": n_misattr,
        "references_per_speaker": {spk: ref.shape for spk, ref in refs.items()},
        "segments": results,
    }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="per-clip output dir with <clip>.json + segment MP3s")
    p.add_argument("--out", type=Path, default=None, help="output sidecar path (default: <run-dir>/attribution_check.json)")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    result = verify_attribution(args.run_dir.resolve(), hf_token)

    # Strip non-serializable bits before writing
    result["references_per_speaker"] = {
        spk: list(shape) for spk, shape in result["references_per_speaker"].items()
    }

    out_path = args.out or (args.run_dir / "attribution_check.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
