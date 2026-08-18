#!/usr/bin/env python3
"""
Minimal reader / worked example for the vi-Sommelier full-duplex stereo
format. Shipped inside the HF dataset alongside the WAV/JSON pairs so
downstream consumers have a canonical decoder.

Format contract (see also: doc/README_fullduplex.md in the HF repo):

  <conversation_id>.wav   — 16 kHz, 2 channels, PCM_16.
                            L holds `channels["L"]`'s audio (silence
                            elsewhere); R holds `channels["R"]`'s audio.
                            Duration = json["duration_s"] (= last_turn.end).

  <conversation_id>.json  — companion metadata + turn transcripts.
                            Every timestamp in `turns[*].start/end` and
                            `turns[*].words[*].start/end` is stereo-WAV-
                            relative (starts at 0). The parent clip's
                            original offset is preserved in
                            `_original_clip_offset_s` for provenance.

Orientations: each conversation is emitted twice — `__oriA` (natural
first-turn speaker on L) and `__oriB` (channels swapped). Consumers
should treat oriA and oriB as independent training samples; the swap is
free data augmentation for channel-invariance.

Usage
-----
    # print all turns of one conversation, aligned to the stereo timeline
    python scripts/read_full_duplex.py <conversation_id>.json

    # slice one turn's mono audio out of the correct channel
    python scripts/read_full_duplex.py <conversation_id>.json --extract-turn 3 \
        --out turn3.wav
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load_conversation(json_path: Path) -> tuple[np.ndarray, int, dict]:
    """Load a (stereo, sr, meta) triple from a paired .wav/.json.

    The .wav is expected next to the .json with the same stem.
    """
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    wav_path = json_path.with_suffix(".wav")
    stereo, sr = sf.read(str(wav_path), always_2d=True)
    assert stereo.shape[1] == 2, f"expected stereo, got shape {stereo.shape}"
    assert sr == meta["sample_rate"], (
        f"sr mismatch: wav={sr} meta={meta['sample_rate']}"
    )
    return stereo, sr, meta


def slice_turn(stereo: np.ndarray, sr: int, turn: dict) -> np.ndarray:
    """Return the mono audio for a single turn from the correct channel.

    Timestamps in `turn` are already stereo-WAV-relative (zero-origin), so
    no offset math is needed — this is the whole point of the format.
    """
    s = int(round(turn["start"] * sr))
    e = int(round(turn["end"]   * sr))
    return stereo[s:e, turn["channel"]]


def slice_word(stereo: np.ndarray, sr: int, turn: dict, word: dict) -> np.ndarray:
    """Return the mono audio for a single word from the correct channel."""
    s = int(round(word["start"] * sr))
    e = int(round(word["end"]   * sr))
    return stereo[s:e, turn["channel"]]


def print_conversation(meta: dict) -> None:
    print(f"# {meta['conversation_id']}")
    print(f"  source_clip:     {meta['source_clip']}")
    print(f"  source_channel:  {meta.get('source_channel')}")
    print(f"  duration_s:      {meta['duration_s']}")
    print(f"  n_turns:         {meta['n_turns']}")
    print(f"  channels:        L={meta['channels']['L']}  R={meta['channels']['R']}")
    print(f"  _original_clip_offset_s: {meta['_original_clip_offset_s']}  "
          f"(add to a turn's start/end to recover clip-time)")
    print()
    for t in meta["turns"]:
        ch_tag = "L" if t["channel"] == 0 else "R"
        print(f"  [{t['start']:6.2f}-{t['end']:6.2f}s] "
              f"{ch_tag} ({t['speaker']}): {t['text']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path", type=Path,
                   help="path to <conversation_id>.json (the .wav is loaded from next to it)")
    p.add_argument("--extract-turn", type=int, default=None, metavar="IDX",
                   help="also write turns[IDX] as a mono WAV via --out")
    p.add_argument("--out", type=Path, default=None,
                   help="output path for --extract-turn (default: <conv>__turn<IDX>.wav)")
    args = p.parse_args()

    stereo, sr, meta = load_conversation(args.json_path)
    print_conversation(meta)

    if args.extract_turn is not None:
        turn = meta["turns"][args.extract_turn]
        mono = slice_turn(stereo, sr, turn)
        out = args.out or args.json_path.with_name(
            f"{meta['conversation_id']}__turn{args.extract_turn}.wav"
        )
        sf.write(str(out), mono, sr, subtype="PCM_16")
        ch_tag = "L" if turn["channel"] == 0 else "R"
        print()
        print(f"[extracted] turn {args.extract_turn} ({ch_tag}, {turn['speaker']}, "
              f"{len(mono)/sr:.2f}s) -> {out}")


if __name__ == "__main__":
    main()
