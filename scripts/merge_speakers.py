#!/usr/bin/env python3
"""
Post-process speaker merge using pyannote/embedding.

Sortformer + cross-chunk linker over-cluster on 4+ min clips (we see 5-7
SPEAKER_NN ids for what should be 2-3 real speakers).  This script computes
a centroid embedding per detected speaker, then agglomeratively merges pairs
with cosine similarity > threshold until either:
  - all remaining pairs are below threshold, OR
  - the number of merged speakers equals --max-speakers.

Writes a side-car  <run_dir>/merged_speaker_map.json  with
  {threshold, max_speakers, map: {original_id: merged_id}}

Usage:
  python scripts/merge_speakers.py --run-dir <run_dir> [--threshold 0.75] [--max-speakers 2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_embedder(hf_token: str | None = None):
    """Return a callable that takes an MP3 path and returns a numpy embedding.

    Swapped from pyannote/embedding to speechbrain ECAPA-TDNN because pyannote's
    embedding gives degenerate cosine sims on the short per-segment MP3s in our
    pipeline output (most segments are 0.5-2s). ECAPA-TDNN handles short audio
    robustly (we already proved this for the attribution-check fix)."""
    from speechbrain.inference.speaker import EncoderClassifier
    import librosa
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    def embed(path: str) -> np.ndarray:
        audio, sr = librosa.load(path, sr=16000, mono=True)
        if len(audio) < int(0.15 * 16000):
            raise ValueError("audio too short for ECAPA-TDNN")
        tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.inference_mode():
            emb = classifier.encode_batch(tensor)  # (1, 1, D)
        return emb.squeeze().cpu().numpy()

    return embed


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _speaker_sort_key(sid: str) -> tuple:
    """Sort key so SPEAKER_00 < SPEAKER_01 < SPEAKER_02 etc.  Falls back to
    string compare when the suffix isn't an integer."""
    try:
        return (0, int(sid.split("_")[-1]))
    except (ValueError, IndexError):
        return (1, sid)


def agglomerative_merge(
    centroids: dict[str, np.ndarray],
    counts: dict[str, int],
    threshold: float,
    expected_speakers: int | None,
    safety_floor: float = 0.4,
) -> dict[str, str]:
    """Two-phase greedy agglomerative merge by cosine similarity.

    Phase 1 — confidence merge: collapse any pair with cosine >= `threshold`.
        Greedy, lowest-ID-wins as canonical.

    Phase 2 — targeted to expected count: if `expected_speakers` is set AND we
        still have more clusters than expected, keep merging the highest-
        remaining-similarity pair UNTIL we hit `expected_speakers` clusters,
        BUT never merge a pair whose cosine is below `safety_floor` (defaults
        to 0.4 — well below same-speaker but above unrelated voices).  This
        catches borderline cases like 0.5-0.6 fragments while refusing to
        merge clearly different speakers even if the metadata says we should.

    Returns {original_id: merged_id} where merged_id is the lowest-numbered
    original_id in the merged group (Sortformer assigns lower IDs to earlier-
    appearing speakers; fragments tend to get later IDs).
    """
    merge_map: dict[str, str] = {sid: sid for sid in centroids}
    cur_centroids = {sid: vec.copy() for sid, vec in centroids.items()}
    cur_counts = dict(counts)

    def do_merge(a: str, b: str) -> None:
        keep, drop = sorted([a, b], key=_speaker_sort_key)
        wa, wb = cur_counts[keep], cur_counts[drop]
        cur_centroids[keep] = (cur_centroids[keep] * wa + cur_centroids[drop] * wb) / (wa + wb)
        cur_counts[keep] += cur_counts[drop]
        del cur_centroids[drop]
        del cur_counts[drop]
        for orig, target in list(merge_map.items()):
            if target == drop:
                merge_map[orig] = keep

    def best_pair() -> tuple[tuple[str, str] | None, float]:
        ids = list(cur_centroids.keys())
        bp = None
        bs = -1.0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                s = cosine(cur_centroids[ids[i]], cur_centroids[ids[j]])
                if s > bs:
                    bs = s
                    bp = (ids[i], ids[j])
        return bp, bs

    # Phase 1: confidence merge
    while len(cur_centroids) > 1:
        pair, sim = best_pair()
        if pair is None or sim < threshold:
            break
        do_merge(*pair)

    # Phase 2: targeted to expected count, with safety floor
    if expected_speakers is not None and expected_speakers > 0:
        while len(cur_centroids) > expected_speakers:
            pair, sim = best_pair()
            if pair is None or sim < safety_floor:
                # Refuse to merge below safety floor even if metadata expects fewer
                break
            do_merge(*pair)

    return merge_map


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="pipeline output dir for one clip")
    p.add_argument("--threshold", type=float, default=0.85, help="cosine similarity above which to confidence-merge (default 0.85; was 0.6 but too loose on similar VN male voices)")
    p.add_argument("--expected-speakers", type=int, default=None, help="target speaker count; phase-2 merges down to this with safety floor 0.4")
    p.add_argument("--safety-floor", type=float, default=0.4, help="never merge below this cosine even to reach expected-speakers (default 0.4)")
    p.add_argument("--out", type=Path, default=None, help="output map path (default: <run-dir>/merged_speaker_map.json)")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    # find the transcript JSON (smoke layout: <run_dir>/<name>.json + audio in <run_dir>/<name>/)
    json_candidates = list(run_dir.glob("*.json"))
    json_candidates = [j for j in json_candidates if j.stem == run_dir.name]
    if not json_candidates:
        sys.exit(f"no <{run_dir.name}>.json found under {run_dir}")
    tr_path = json_candidates[0]
    data = json.loads(tr_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        sys.exit(f"no segments in {tr_path}")

    audio_root = run_dir / run_dir.name
    if not audio_root.exists():
        sys.exit(f"audio dir not found: {audio_root}")

    # Load contamination sidecars so we can exclude unreliable references.
    # Without this, an overlap/cross-talk segment that Sortformer assigned to
    # SPEAKER_01 but actually contains SPEAKER_00 audio will pull SPEAKER_01's
    # centroid toward SPEAKER_00, causing a false-positive merge downstream.
    word_attr_path = run_dir / "word_attribution.json"
    multi_speaker_idxs: set[int] = set()
    if word_attr_path.exists():
        try:
            wa = json.loads(word_attr_path.read_text(encoding="utf-8"))
            for s in wa.get("segments", []):
                if s.get("multi_speaker"):
                    multi_speaker_idxs.add(s["index"])
        except Exception as e:
            print(f"  [warn] could not read {word_attr_path}: {e}")

    print(f"Loading ECAPA-TDNN ...", flush=True)
    embed = load_embedder()

    # Bucket segments per speaker into "clean" (no overlap, no multi-speaker
    # attribution) and "all". The reference centroid is built from CLEAN if
    # that speaker has ≥1 clean segment; otherwise we fall back to ALL so the
    # speaker still gets a centroid (round-table clips where every segment
    # has co-speech would otherwise lose all speakers).
    per_speaker_clean: dict[str, list[np.ndarray]] = {}
    per_speaker_all:   dict[str, list[np.ndarray]] = {}
    n_skipped_contam = 0
    for i, seg in enumerate(segments):
        spk = seg.get("speaker", "?")
        idx = seg.get("index", f"{i:05d}")
        seg_idx = seg.get("index", i)
        is_contam = (seg.get("is_separated") or seg.get("is_overlap")
                     or seg_idx in multi_speaker_idxs)
        mp3 = audio_root / f"{idx}_{spk}.mp3"
        if not mp3.exists():
            print(f"  [skip] missing audio: {mp3}")
            continue
        try:
            vec = np.asarray(embed(str(mp3))).reshape(-1)
        except Exception as e:
            print(f"  [skip] embed failed {mp3}: {e}")
            continue
        per_speaker_all.setdefault(spk, []).append(vec)
        if not is_contam:
            per_speaker_clean.setdefault(spk, []).append(vec)
        else:
            n_skipped_contam += 1

    # Prefer clean per speaker, fall back to all when none clean
    per_speaker_vecs: dict[str, list[np.ndarray]] = {}
    for spk in per_speaker_all:
        if per_speaker_clean.get(spk):
            per_speaker_vecs[spk] = per_speaker_clean[spk]
        else:
            per_speaker_vecs[spk] = per_speaker_all[spk]
            print(f"  [fallback] {spk}: no clean segments, using all", flush=True)

    if n_skipped_contam:
        print(f"  built clean references (skipping {n_skipped_contam} overlap/multi-speaker segs across the clip)", flush=True)

    if not per_speaker_vecs:
        sys.exit("no embeddings — nothing to merge")

    centroids = {sid: np.mean(vecs, axis=0) for sid, vecs in per_speaker_vecs.items()}
    counts = {sid: len(vecs) for sid, vecs in per_speaker_vecs.items()}

    print(f"  detected {len(centroids)} speakers, segments per spk: {counts}", flush=True)

    # Diagnostic: pairwise centroid similarity, helps tune threshold
    sids = sorted(centroids.keys(), key=_speaker_sort_key)
    print("  pairwise cosine similarity:", flush=True)
    print("       " + "".join(f"{s[-3:]:>6}" for s in sids))
    for s1 in sids:
        row = f"  {s1[-3:]:>4} "
        for s2 in sids:
            if s1 == s2:
                row += f"{'   -':>6}"
            else:
                row += f"{cosine(centroids[s1], centroids[s2]):>6.2f}"
        print(row, flush=True)

    merge_map = agglomerative_merge(
        centroids, counts, args.threshold,
        expected_speakers=args.expected_speakers,
        safety_floor=args.safety_floor,
    )
    canonical = sorted(set(merge_map.values()))
    print(f"  merged → {len(canonical)} speakers: {canonical}", flush=True)
    for orig, mapped in sorted(merge_map.items()):
        if orig != mapped:
            print(f"    {orig:<14} → {mapped}")

    out_path = args.out or (run_dir / "merged_speaker_map.json")
    out_path.write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "expected_speakers": args.expected_speakers,
                "safety_floor": args.safety_floor,
                "detected_count": len(centroids),
                "merged_count": len(canonical),
                "map": merge_map,
                "original_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
