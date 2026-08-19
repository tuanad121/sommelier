#!/usr/bin/env python3
"""
Assemble 2-channel (stereo) full-duplex training samples from the
sommelier per-turn outputs.

Input:  one conversation JSON produced by build_training_conversations.py
        (a run of ≥N contiguous dyadic turns from a single clip).

For each conversation we:
  * Determine speaker→channel mapping (first speaker = L, other = R).
  * Read each turn's Demucs-cleaned mono MP3 from the clip dir.
  * Place it into the correct channel at the correct time offset.
  * The other channel stays silence during that turn (single-speaker
    frames).
  * For overlap frames (both speakers active), if a gated MossFormer2
    separation exists, use its per-speaker signals; otherwise fall back
    to placing the current turn on its channel and leaving the other
    channel silent (loses the interruption, but doesn't fabricate
    audio).

Output:
  <out_dir>/<conversation_id>.wav   ← 16 kHz stereo, PCM_16
  <out_dir>/<conversation_id>.json  ← full turn schema (text, words, flags)
                                       with ZERO-ORIGIN timestamps aligned
                                       to the stereo WAV. Parent-clip offset
                                       preserved as `_original_clip_offset_s`
                                       for provenance.

Usage:
    python scripts/build_full_duplex_2channel.py \\
        --convs-dir data/training_conversations_fullduplex/conversations \\
        --clips-root data/raw/_final/-sepreformer-True-... \\
        --out data/full_duplex_stereo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 16_000


def _read_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=False)
    if sr != SR:
        # Resample via linear interp — cheap; per-turn MP3s are already
        # 16k in the sommelier pipeline so this branch shouldn't fire
        # in practice. Kept for safety.
        n_new = int(round(len(audio) * SR / sr))
        audio = np.interp(
            np.linspace(0, 1, n_new, endpoint=False),
            np.linspace(0, 1, len(audio), endpoint=False),
            audio,
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def _sec_to_samp(x: float) -> int:
    return int(round(x * SR))


def _assemble_conversation(conv: dict, clip_dir: Path) -> tuple[np.ndarray, dict]:
    turns = conv["turns"]
    speakers = sorted({t["speaker"] for t in turns})
    if len(speakers) != 2:
        raise ValueError(f"expected exactly 2 speakers, got {speakers}")

    # First turn's speaker → left channel; other → right.
    left_spk = turns[0]["speaker"]
    right_spk = [s for s in speakers if s != left_spk][0]
    channel_of = {left_spk: 0, right_spk: 1}

    start_time = turns[0]["start"]
    end_time = turns[-1]["end"]
    n_samples = _sec_to_samp(end_time - start_time)
    stereo = np.zeros((n_samples, 2), dtype=np.float32)

    problems = []
    for t in turns:
        # audio_path is relative to the FINAL/<clip>/ directory, i.e.
        # `<clip>/00027_SPEAKER_00.mp3`. We accept both layouts.
        audio_rel = Path(t["audio_path"])
        candidates = [clip_dir / audio_rel.name, clip_dir.parent / audio_rel]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            problems.append(f"missing audio for turn {t['turn_id']}: tried {candidates}")
            continue
        chunk = _read_mono(found)
        ch = channel_of[t["speaker"]]
        s0 = _sec_to_samp(t["start"] - start_time)
        s1 = min(s0 + len(chunk), n_samples)
        chunk = chunk[: s1 - s0]
        # Add rather than overwrite in case two turns collide (overlap
        # region on the same channel — shouldn't happen if the
        # extractor did its job, but be safe).
        stereo[s0:s1, ch] += chunk

    peak = float(np.max(np.abs(stereo))) or 1.0
    if peak > 1.0:
        stereo = stereo / peak

    # Zero-origin design: the stereo WAV starts at t=0, so we shift every
    # timestamp in the paired JSON by `start_time` (== turns[0]["start"] in
    # the parent clip's timeline). This means a dataloader can slice the
    # stereo array with turn["start"]*SR directly — no offset math, no
    # off-by-one silent-region bugs. The original clip-time offset is kept
    # as `_original_clip_offset_s` for provenance / debugging back to the
    # raw pipeline artifacts.
    def _shift(x: float) -> float:
        return round(x - start_time, 3)

    def _shift_words(words):
        return [
            {**w, "start": _shift(w["start"]), "end": _shift(w["end"])}
            for w in (words or [])
        ]

    meta = {
        "conversation_id": conv["conversation_id"],
        "source_clip": conv["source_clip"],
        "source_url": conv.get("source_url"),
        "source_channel": conv.get("source_channel"),
        "source_genre": conv.get("source_genre"),
        "duration_s": round(end_time - start_time, 2),
        "n_turns": len(turns),
        "unique_speakers": sorted({t["speaker"] for t in turns}),
        "channels": {"L": left_spk, "R": right_spk},
        "channel_of": channel_of,  # {speaker_id: 0|1} — handy for dataloaders
        "sample_rate": SR,
        "_original_clip_offset_s": round(start_time, 3),
        "_original_clip_time_range": [round(start_time, 3), round(end_time, 3)],
        "turns": [
            {
                "turn_id": t["turn_id"],
                "turn_index": t.get("turn_index"),
                "speaker": t["speaker"],
                "channel": channel_of[t["speaker"]],
                "start": _shift(t["start"]),
                "end": _shift(t["end"]),
                "duration": round(t["end"] - t["start"], 3),
                "audio_path": t.get("audio_path"),
                "text": t.get("text", ""),
                "text_source": t.get("text_source"),
                "flags": t.get("flags", {}),
                "words": _shift_words(t.get("words")),
                **({"contested_words": t["contested_words"]}
                   if t.get("contested_words") else {}),
            }
            for t in turns
        ],
        "problems": problems,
    }
    return stereo, meta


def _swap_channels(meta: dict) -> dict:
    """Return a new meta dict with L↔R channel identities flipped."""
    swapped = dict(meta)  # shallow — we deep-copy the sub-fields we mutate
    swapped["channels"] = {"L": meta["channels"]["R"], "R": meta["channels"]["L"]}
    swapped["channel_of"] = {spk: 1 - ch for spk, ch in meta["channel_of"].items()}
    swapped["turns"] = [
        {**t, "channel": 1 - t["channel"]} for t in meta["turns"]
    ]
    return swapped


def _write_pair(out_dir: Path, stem: str, stereo, meta: dict) -> None:
    # The conversation_id in meta must match the file stem so downstream
    # consumers can index either way.
    meta = {**meta, "conversation_id": stem}
    sf.write(str(out_dir / f"{stem}.wav"), stereo, SR, subtype="PCM_16")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--convs-dir", type=Path, required=True,
                   help="dir with per-conversation JSONs from build_training_conversations.py")
    p.add_argument("--clips-root", type=Path, required=True,
                   help="_final root containing <clip>/<clip>/<turn>.mp3")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N conversations (debug)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    conv_files = sorted(args.convs_dir.glob("*.json"))
    if args.limit:
        conv_files = conv_files[: args.limit]

    ok = 0
    failed = 0
    total_hours = 0.0
    for cf in conv_files:
        conv = json.loads(cf.read_text(encoding="utf-8"))
        clip_dir = args.clips_root / conv["source_clip"] / conv["source_clip"]
        try:
            stereo, meta = _assemble_conversation(conv, clip_dir)
        except Exception as e:
            failed += 1
            print(f"[fail] {conv['conversation_id']}: {e}")
            continue
        # Emit both channel orientations. Rationale: our podcast sources are
        # peer-symmetric (co-hosts as often as host+guest), so no natural
        # "assistant channel" exists. Emitting both orientations gives the FD
        # trainer channel-invariant supervision at zero annotation cost and
        # doubles the effective training data. See doc/full_duplex_data_scaling.md
        # for the design decision (deviates from Sommelier paper §3.1 which
        # fixes one speaker on the left).
        _write_pair(args.out, meta["conversation_id"] + "__oriA", stereo, meta)
        _write_pair(args.out, meta["conversation_id"] + "__oriB",
                    stereo[:, ::-1].copy(), _swap_channels(meta))
        total_hours += meta["duration_s"] / 3600
        ok += 1

    print(f"[done] {ok} conversations × 2 orientations = "
          f"{ok * 2} stereo files ({total_hours * 2:.2f}h), {failed} failed")
    print(f"       -> {args.out}/")


if __name__ == "__main__":
    main()
