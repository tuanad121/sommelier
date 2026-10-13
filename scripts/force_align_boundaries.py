#!/usr/bin/env python3
"""Force-align ASR text to segment audio using wav2vec2-VN.

Produces per-segment refined boundaries (first-word-start, last-word-end)
and per-word timestamps. Writes <run-dir>/force_alignment.json sidecar
that the normalizer can consume to:

  - replace Sortformer's ±100ms segment start/end with wav2vec2's ~20ms
    word-level boundaries → recovers boundary-dropped content (seg 1 case)
  - expose per-word (word, t_start, t_end) tuples for future SA-ASR training

Uses torchaudio.functional.forced_align with wav2vec2-base-vietnamese-250h
CTC posteriors.

Usage:
  python scripts/force_align_boundaries.py --run-dir <clip-out-dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("force_align")

MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
MIN_SEG_DUR_FOR_ALIGN = 0.3  # below this, wav2vec2 alignment is unreliable


def normalize_text_for_align(text: str) -> str:
    """wav2vec2 VN tokenizer is character-level. Normalize text for alignment:
       NFC normalize, lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[.,!?;:\"'\(\)\[\]\{\}\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model(device: torch.device):
    log.info(f"Loading {MODEL_NAME}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device).eval()
    return model, processor


def force_align_one(
    audio_16k: np.ndarray,
    text: str,
    model,
    processor,
    device: torch.device,
) -> dict | None:
    """Force-align a single segment.

    Returns dict with:
      refined_start_offset_s: float — sec from segment start where first word begins
      refined_end_offset_s:   float — sec from segment start where last word ends
      words: list of {word: str, start_offset_s: float, end_offset_s: float}
    Or None if alignment fails.
    """
    norm = normalize_text_for_align(text)
    if not norm:
        return None
    words = norm.split()
    if not words:
        return None

    # Build target token sequence (characters of the normalized text + spaces).
    # The tokenizer has space '|' for word boundary in wav2vec2 CTC models.
    vocab = processor.tokenizer.get_vocab()
    word_delim = "|" if "|" in vocab else " "
    # Sequence: chars + word_delim between words (no leading/trailing delim).
    char_seq = []
    word_spans = []  # (start_char_idx, end_char_idx) for each word
    for w_i, w in enumerate(words):
        start_i = len(char_seq)
        for ch in w:
            char_seq.append(ch)
        end_i = len(char_seq)
        word_spans.append((start_i, end_i))
        if w_i < len(words) - 1:
            char_seq.append(word_delim)

    # Map characters to vocab IDs (skip OOV chars by stripping them).
    token_ids = []
    char_to_token_pos = []  # char_idx → position in token_ids (None if skipped)
    for ci, ch in enumerate(char_seq):
        if ch in vocab:
            char_to_token_pos.append(len(token_ids))
            token_ids.append(vocab[ch])
        else:
            char_to_token_pos.append(None)

    if not token_ids:
        return None

    # Run wav2vec2 to get log-probabilities
    inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.inference_mode():
        logits = model(input_values).logits  # (1, T_frames, V)
    log_probs = torch.log_softmax(logits, dim=-1).cpu()  # (1, T_frames, V)

    # Force-align. torchaudio.functional.forced_align returns frame-level
    # TOKEN IDs (vocab values), NOT target positions. To get per-target-position
    # frame spans we collapse consecutive same-token runs via merge_tokens —
    # for a successful alignment, that yields one TokenSpan per target position
    # in order.
    targets = torch.tensor([token_ids], dtype=torch.int32)
    blank_id = processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = vocab.get("<pad>", vocab.get("[PAD]", 0))
    try:
        aligned_tokens, alignment_scores = F.forced_align(log_probs, targets, blank=blank_id)
    except Exception as e:
        log.debug(f"forced_align failed: {e}")
        return None

    try:
        spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0], blank=blank_id)
    except Exception as e:
        log.debug(f"merge_tokens failed: {e}")
        return None

    if len(spans) != len(token_ids):
        log.debug(
            f"span count {len(spans)} != target count {len(token_ids)} — "
            "alignment likely degraded"
        )

    # Map target position i → (first_frame, last_frame) for the i-th token.
    pos_to_range: dict[int, tuple[int, int]] = {}
    for i, span in enumerate(spans):
        if i >= len(token_ids):
            break
        pos_to_range[i] = (span.start, span.end)

    if not pos_to_range:
        return None

    # Convert frame indices to seconds. wav2vec2 frame stride is 20ms (320 samples @ 16kHz).
    frames_per_sec = 16000 / 320  # = 50 Hz

    # Per-word start/end (in seconds from segment start).
    word_results = []
    for w, (start_cidx, end_cidx) in zip(words, word_spans):
        # Target positions for this word's characters
        positions = []
        for ci in range(start_cidx, end_cidx):
            pos = char_to_token_pos[ci]
            if pos is not None and pos in pos_to_range:
                positions.append(pos)
        if not positions:
            continue
        starts = [pos_to_range[p][0] for p in positions]
        ends = [pos_to_range[p][1] for p in positions]
        f_start, f_end = min(starts), max(ends)
        word_results.append({
            "word": w,
            "start_offset_s": f_start / frames_per_sec,
            "end_offset_s": (f_end + 1) / frames_per_sec,
        })

    if not word_results:
        return None

    return {
        "refined_start_offset_s": word_results[0]["start_offset_s"],
        "refined_end_offset_s": word_results[-1]["end_offset_s"],
        "words": word_results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
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
    model, processor = load_model(device)

    out_segments = []
    n_aligned = 0
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        dur = end - start
        spk = seg.get("speaker", "?")
        text = seg.get("text", "") or ""

        result = {
            "index": i,
            "orig_start": start,
            "orig_end": end,
            "text": text,
            "aligned": False,
        }

        if dur < MIN_SEG_DUR_FOR_ALIGN or not text.strip():
            result["skipped"] = "too_short_or_empty"
            out_segments.append(result)
            continue

        # Load segment audio
        spk_for_path = seg.get("original_speaker", spk)
        audio_file = run_dir / clip / f"{i:05d}_{spk_for_path}.mp3"
        if not audio_file.exists():
            result["skipped"] = "audio_missing"
            out_segments.append(result)
            continue

        try:
            audio, sr = librosa.load(str(audio_file), sr=16000, mono=True)
        except Exception as e:
            result["skipped"] = f"load_failed: {e}"
            out_segments.append(result)
            continue

        try:
            align = force_align_one(audio.astype(np.float32), text, model, processor, device)
        except Exception as e:
            log.debug(f"seg {i}: alignment exception: {e}")
            align = None

        if align is None:
            result["skipped"] = "alignment_failed"
            out_segments.append(result)
            continue

        result["aligned"] = True
        result["refined_start"] = start + align["refined_start_offset_s"]
        result["refined_end"] = start + align["refined_end_offset_s"]
        result["word_count"] = len(align["words"])
        result["words"] = [
            {
                "word": w["word"],
                "start": start + w["start_offset_s"],
                "end": start + w["end_offset_s"],
            }
            for w in align["words"]
        ]
        out_segments.append(result)
        n_aligned += 1

    log.info(f"Aligned {n_aligned}/{len(segments)} segments")
    out_path = run_dir / "force_alignment.json"
    out_path.write_text(json.dumps({
        "clip": clip,
        "n_segments": len(segments),
        "n_aligned": n_aligned,
        "segments": out_segments,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
