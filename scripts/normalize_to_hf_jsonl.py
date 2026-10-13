#!/usr/bin/env python3
"""
Normalize a sommelier pipeline output JSON into the HF dialogue JSONL schema.

Input:  <run_dir>/<audio_name>.json  (the file written by main_original_ASR_MoE.py)
Output: <run_dir>/<audio_name>.jsonl (one row per turn, HF-ready)

Schema (one JSON object per line):
{
  "conversation_id": str,
  "source_url": str | null,
  "source_channel": str | null,
  "source_genre": str | null,
  "audio_sample_rate": int,
  "turn_id": str, "turn_index": int,
  "speaker": str, "start": float, "end": float, "duration": float,
  "audio_path": str,                       # relative to the JSONL file
  "asr": {"rover": str, "whisper": str, "phowhisper": str, "chunkformer": str},
  "gold": null | {"text": str, "note": str, "annotator": str, "annotated_at": str},
  "flags": {
    "is_backchannel": bool,                # ROVER (or gold) text matches VN BC vocab
    "is_overlap": bool,                    # pipeline marked is_separated
    "is_short": bool,                      # duration < 1.0s
    "diar_orphan_candidate": bool          # same-speaker-consecutive + gap > 1.5s
  },
  "context": {
    "prev_speaker": str | null, "prev_end": float | null,
    "gap_to_prev_s": float | null,
    "next_speaker": str | null, "next_start": float | null,
    "is_same_speaker_consecutive": bool
  }
}
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import unicodedata

# Port of RoverEnsembler.align_and_vote — N-way word-position ROVER vote
# (no GPU). Use to do 5-way ROVER (Whisper + PhoWhisper + ChunkFormer +
# Qwen3-ASR + Parakeet-VN) without re-running the ASR ensemble step.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from rover_voter import align_and_vote as rover_vote_n_way
from pathlib import Path

VN_BACKCHANNELS = {
    "ừ", "dạ", "vâng", "à", "ờ", "ừm",
    "đúng rồi", "vậy hả", "thế à", "ok",
    "ừ ha", "ờ ha", "à ha", "ừ nhỉ",
    "uhm", "hử", "hả", "đúng", "rồi",
}

ORPHAN_GAP_THRESHOLD_S = 1.5
# Source language hint — informs language-flip detection.  Hard-coded for VN
# work; should be plumbed through from the CLI source-language arg eventually.
SOURCE_LANGUAGE_HINT = "vi"
# max_volume below this is treated as silent / phantom diar segment.  Real
# VN backchannels measured ~-25 to -40 dB max; phantom slices show ~-55 to -75 dB.
SILENCE_MAX_DBFS = -45.0
_VOL_RE = re.compile(r"(?:mean_volume|max_volume):\s*(-?\d+(?:\.\d+)?)\s*dB")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", (text or "").lower().strip())
    t = re.sub(r"[\.,!?;:\-\"'…]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


_VN_DIACRITIC_RE = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
)
_COMMON_ENGLISH_TOKENS = {
    "i", "i'm", "im", "you", "we", "they", "he", "she", "it",
    "the", "and", "or", "but", "in", "on", "at", "to", "of", "is", "are",
    "was", "were", "do", "don't", "dont", "doesn't", "won't", "wont",
    "can", "can't", "cant", "well", "right", "okay", "ok", "yeah", "yes",
    "no", "not", "hi", "hey", "hello", "thank", "thanks", "know", "think",
    "really", "now", "actually", "just", "good", "bad", "what", "why",
    "how", "when", "where", "who", "this", "that", "these", "those",
}


def _text_looks_english(text: str) -> bool:
    """Heuristic: Whisper sometimes claims `language=vi` while emitting
    English text.  Detect by absence of VN diacritics + presence of common
    English tokens.  Conservative — only fires when both signals agree."""
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    if _VN_DIACRITIC_RE.search(t):
        return False  # has VN diacritic, definitely not English flip
    # All ASCII + alphabetic — could be English flip OR a single word.
    # Require at least one common English token to fire.
    tokens = [tok.lower().strip(".,!?;:'\"") for tok in t.split()]
    if not any(tok in _COMMON_ENGLISH_TOKENS for tok in tokens):
        return False
    # Most tokens should be either common English or empty
    alpha_tokens = [tok for tok in tokens if tok and tok.isascii() and tok.replace("'", "").isalpha()]
    return len(alpha_tokens) >= 1


def measure_loudness(audio_path: Path) -> tuple[float | None, float | None]:
    """Return (mean_dbfs, max_dbfs) via ffmpeg's volumedetect filter.  Returns
    (None, None) when ffmpeg is unavailable or the file can't be read."""
    if not audio_path.exists() or shutil.which("ffmpeg") is None:
        return (None, None)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(audio_path),
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=10,
        )
        out = proc.stderr
    except (subprocess.TimeoutExpired, OSError):
        return (None, None)
    mean = max_ = None
    for line in out.splitlines():
        m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", line)
        if m:
            mean = float(m.group(1))
        m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", line)
        if m:
            max_ = float(m.group(1))
    return (mean, max_)


def is_backchannel(text: str) -> bool:
    n = normalize(text)
    if not n:
        return False
    if n in VN_BACKCHANNELS:
        return True
    toks = n.split()
    return len(toks) <= 2 and toks[0] in VN_BACKCHANNELS


_DECODE_RE = re.compile(r"'decode'\s*:\s*'((?:[^'\\]|\\.)*)'", re.DOTALL)
_DECODE_TRAILING_RE = re.compile(
    r"'decode'\s*:\s*'(.*?)(?:'\s*,\s*'(?:start|end)'|$)", re.DOTALL
)
# When WhisperX word-timestamp windows overlap, the decode strings can include
# the same trailing token twice (e.g. "...thiệt. thiệt.").  Dedupe trailing
# duplicate tokens up to 2 levels deep — only at the END of the string to
# avoid clobbering legitimate VN backchannel repetition like "rồi rồi".
_TAIL_DUP_RE = re.compile(r"(\b\S+[.,!?])(\s+\1)+\s*$")


def clean_asr_text(raw: str) -> str:
    """The pipeline writes ASR text in a few shapes:
      - plain strings: "Đấy là em bắt đầu đi làm"
      - WhisperX wrapper as Python-repr list-of-dicts:
        "[{'decode': ' foo ', 'start': '00:00:00:000', 'end': '00:00:03:200'}]"
      - **TRUNCATED / malformed** Python-repr where ast.literal_eval fails,
        e.g. "[{'decode': ' ...rồi. '00:00:00:000', 'end': rồi."
        (export bug; missing comma + unquoted trailing value)
    Salvage `decode` values via regex when ast can't parse."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s == "[]":
        return ""
    result = s
    if s.startswith("[") and "decode" in s:
        # Happy path: well-formed Python literal
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                texts = [p.get("decode", "").strip() for p in parsed if isinstance(p, dict)]
                result = " ".join(t for t in texts if t).strip()
                return _dedupe_tail(result)
        except (ValueError, SyntaxError):
            pass
        # Fallback 1: pull all properly-quoted 'decode': '<text>' chunks
        chunks = [m.strip() for m in _DECODE_RE.findall(s)]
        if chunks:
            return _dedupe_tail(" ".join(chunks).strip())
        # Fallback 2: salvage one decode value even when trailing is truncated
        m = _DECODE_TRAILING_RE.search(s)
        if m:
            return _dedupe_tail(m.group(1).strip())
    return _dedupe_tail(result)


def _dedupe_tail(text: str) -> str:
    """Collapse trailing duplicate tokens introduced by WhisperX overlap
    stitching, e.g. 'X. X.' -> 'X.'  Only acts on the END so backchannel
    repetition like 'rồi rồi' mid-string is preserved."""
    return _TAIL_DUP_RE.sub(r"\1", text).strip()


def apply_speaker_merge(segments: list[dict], merge_map: dict[str, str]) -> list[dict]:
    """Rewrite each segment's `speaker` per the merge map; stash the original
    in `original_speaker`.  Mutates a copy and returns it."""
    out = []
    for s in segments:
        s2 = dict(s)
        orig = s.get("speaker", "?")
        merged = merge_map.get(orig, orig)
        s2["original_speaker"] = orig
        s2["speaker"] = merged
        out.append(s2)
    return out


def normalize_segments(segments: list[dict]) -> list[dict]:
    """Walk segments and compute context fields once.  Operates on the
    (possibly merge-mapped) speaker field, so adjacency/orphan logic uses
    canonical merged identities."""
    enriched = []
    for i, seg in enumerate(segments):
        prev = segments[i - 1] if i > 0 else None
        nxt = segments[i + 1] if i < len(segments) - 1 else None

        gap_to_prev = (seg.get("start", 0) - prev["end"]) if prev else None
        same_spk = prev is not None and prev.get("speaker") == seg.get("speaker")
        diar_orphan = bool(same_spk and gap_to_prev is not None and gap_to_prev > ORPHAN_GAP_THRESHOLD_S)

        enriched.append({
            "seg": seg,
            "prev": prev,
            "next": nxt,
            "gap_to_prev": gap_to_prev,
            "same_speaker_consecutive": same_spk,
            "diar_orphan": diar_orphan,
        })
    return enriched


def _token_wer(a: str, b: str) -> float:
    """Token-level normalized Levenshtein: 0 = identical, 1 = disjoint."""
    ta = normalize(a).split()
    tb = normalize(b).split()
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 1.0
    n, m = len(ta), len(tb)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ta[i - 1] == tb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / max(n, m)


def _find_contested_words(words: list[dict], duration: float,
                          cospeech_frac_fn=None) -> list[dict]:
    """Detect VN backchannel tokens embedded mid-utterance — the "ừ, ừm"
    contamination pattern where PhoWhisper transcribes a quiet backchannel
    from the OTHER speaker into a long turn.

    Heuristic:
      - Segment duration >= 2.0s (short segments can legitimately BE a backchannel)
      - Token normalises to a VN backchannel (`ừ`, `ừm`, `vâng`, `à`, `ạ`, …)
      - Token position >= 3 (skip start-of-segment fillers by the same speaker)
      - Not the last token (trailing "à" is often the same speaker)

    Adds cospeech-verification when available: if Sortformer says >=2 speakers
    were active during the word's ~200ms window, marks reason="cospeech_verified".
    """
    if duration < 2.0 or not words:
        return []
    contested = []
    for i, w in enumerate(words):
        if i < 3 or i == len(words) - 1:
            continue
        token = (w.get("word", "") or "").lower().strip()
        token_clean = re.sub(r"[^\w]", "", token)
        if token_clean not in VN_BACKCHANNELS:
            continue
        reasons = ["mid_utterance_backchannel_token"]
        # Optional: verify against Sortformer if we can query cospeech at
        # this word's local time range (padded ±150 ms).
        if cospeech_frac_fn is not None:
            w_start = float(w.get("start", 0))
            w_end = float(w.get("end", w_start + 0.05))
            local_cs = cospeech_frac_fn(max(0.0, w_start - 0.15), w_end + 0.15)
            if local_cs is not None and local_cs > 0.3:
                reasons.append("cospeech_verified")
        contested.append({
            "word_index": i,
            "word": w.get("word", ""),
            "start": float(w.get("start", 0)),
            "end": float(w.get("end", 0)),
            "reasons": reasons,
        })
    return contested


def _overlap_confidence(seg: dict, rover: str, wh: str, pw: str, ck: str,
                        duration: float, cospeech_frac: float | None = None) -> float:
    """Score in [0, 1] that this segment is a genuine cross-speaker overlap.

    Signal sources (independent evidence):
      cospeech_frac    Fraction of Sortformer 0.08s frames in this segment
                       where >=2 speakers exceed prob 0.5.  Direct + reliable.
      duration < 1.0s  Short segments containing multi-speaker text are more
                       likely mis-attributions.
      ASR disagreement Token-WER between rover_final and each of whisper /
                       phowhisper / chunkformer.  High disagreement often
                       means different speakers dominate different portions
                       of the audio.

    The pipeline's `is_separated` flag is NOT used — it's a SepReformer
    artifact that fires on all segments.
    """
    score = 0.0
    if cospeech_frac is not None and cospeech_frac > 0.05:
        # Sortformer says at least 5 % of this segment has >=2 active speakers
        score += min(0.45, cospeech_frac * 1.5)
    if duration < 1.0:
        score += 0.15
    disagreements = [_token_wer(rover, other) for other in (wh, pw, ck) if other]
    if disagreements:
        avg_wer = sum(disagreements) / len(disagreements)
        score += min(0.30, avg_wer * 0.4)
    return round(min(score, 1.0), 3)


def make_record(
    conv_id: str,
    src_url: str | None,
    src_channel: str | None,
    src_genre: str | None,
    sample_rate: int,
    audio_subdir: str,
    e: dict,
    idx: int,
    audio_root_abs: Path | None = None,
    qwen3_text: str = "",
    parakeet_text: str = "",
    cospeech_frac: float | None = None,
) -> dict:
    seg = e["seg"]
    rover = clean_asr_text(seg.get("text", ""))
    wh = clean_asr_text(seg.get("text_whisper", ""))
    pw = clean_asr_text(seg.get("text_phowhisper", ""))
    ck = clean_asr_text(seg.get("text_chunkformer", ""))
    qw = clean_asr_text(qwen3_text or "")
    pa = clean_asr_text(parakeet_text or "")

    duration = float(seg.get("duration", seg.get("end", 0) - seg.get("start", 0)))
    audio_file = seg.get("audio_file", "")
    if audio_file:
        # Normalize to <audio_subdir>/<filename>
        if not audio_file.startswith(audio_subdir):
            audio_file = f"{audio_subdir}/{Path(audio_file).name}"
    else:
        # Pipeline output sometimes omits audio_file even though the file
        # is exported.  Synthesize from the index + ORIGINAL speaker — the
        # MP3 files on disk were written with the pre-merge speaker label,
        # not the post-merge canonical label.
        turn_id = seg.get("index", f"{idx:05d}")
        speaker = seg.get("original_speaker", seg.get("speaker", "SPEAKER_00"))
        audio_file = f"{audio_subdir}/{turn_id}_{speaker}.mp3"

    # Measure audio loudness for phantom-segment detection
    mean_dbfs = max_dbfs = None
    if audio_root_abs is not None and audio_file:
        abs_path = audio_root_abs.parent / audio_file
        mean_dbfs, max_dbfs = measure_loudness(abs_path)

    is_silent = max_dbfs is not None and max_dbfs < SILENCE_MAX_DBFS

    # Detect Whisper language flip — two complementary checks:
    #   (a) Whisper's self-reported `language` disagrees with source language
    #   (b) Whisper's TEXT itself looks like English / no-VN-diacritics with
    #       common English tokens (catches cases where Whisper claims `vi`
    #       but emits "I don't know." — observed on klt_hue_banhkhoai).
    whisper_language = seg.get("language") or None
    flipped_by_lang = bool(
        whisper_language
        and SOURCE_LANGUAGE_HINT
        and whisper_language.lower() != SOURCE_LANGUAGE_HINT.lower()
    )
    flipped_by_text = _text_looks_english(wh) if SOURCE_LANGUAGE_HINT == "vi" else False
    language_flipped = flipped_by_lang or flipped_by_text

    # Canonical ROVER selection. The user-preferred form for a podcast dataset
    # is WRITTEN ("Intelligence", "the way of working") not SPOKEN ("intel li
    # chừn", "quê ấp quốc khi"). The pipeline ROVER (spoken form) is preserved
    # in rover_original.
    #
    # Priority (highest first):
    #   1. Whisper       — best written-form bias (capitalization, English
    #                      entities, numerals), AS LONG AS its length is
    #                      reasonable vs the others. Reject when Whisper
    #                      output is much shorter than VN ASRs — that's
    #                      almost always a hallucination ("Ừ." for 4s of
    #                      real speech, "Thank you" for silence, etc.)
    #   2. PhoWhisper    — also written-form-preserving (often keeps English
    #                      code-switching that ChunkFormer phoneticizes),
    #                      same length sanity-check.
    #   3. Pipeline ROVER — spoken form fallback.
    # Lang-flip override chain (PhoWhisper → Qwen3 → ChunkFormer → Parakeet)
    # still wins when Whisper has fully lang-flipped to English.
    rover_final = rover
    rover_source = "pipeline_rover"

    wh_len = len(wh) if wh and wh.strip() else 0
    pw_len = len(pw) if pw and pw.strip() else 0
    ck_len = len(ck) if ck and ck.strip() else 0
    qw_len = len(qw) if qw and qw.strip() else 0
    pa_len = len(pa) if pa and pa.strip() else 0

    # Whisper-disagreement detection (three zones based on word-set Jaccard
    # vs PhoWhisper, after punctuation-stripped lowercase tokenization):
    #   <25%   = hallucination (YouTube-prior "Hãy subscribe...", etc.) → drop WH
    #   25-95% = word-level substitution error ("ra gián" vs "ra dáng") → use
    #            pipeline ROVER instead of WH; the pipeline's word-level vote
    #            across all 3 ASRs is more reliable than WH alone here
    #   >95%   = cosmetic differences only (casing/punctuation) → WH wins for
    #            written-form bias
    def _norm_toks(s: str) -> set[str]:
        # Strip punctuation and lowercase before tokenizing so "minh."=="minh".
        return set(re.sub(r"[^\w\s]", " ", s.lower()).split())

    whisper_hallucinated = False
    whisper_word_diff = False
    if wh_len > 0 and pw_len > 0:
        wh_toks = _norm_toks(wh)
        pw_toks = _norm_toks(pw)
        if wh_toks and pw_toks:
            jaccard = len(wh_toks & pw_toks) / len(wh_toks | pw_toks)
            if jaccard < 0.25:
                whisper_hallucinated = True
                wh = ""
                wh_len = 0
            elif jaccard < 0.95:
                whisper_word_diff = True

    if language_flipped:
        for candidate_text, candidate_label in (
            (pw, "phowhisper"),
            (None, None),   # Qwen3/Parakeet inserted below from observer fields
            (None, None),
            (ck, "chunkformer"),
            (wh, "whisper"),
        ):
            if candidate_text and candidate_label and candidate_text.strip():
                rover_final = candidate_text
                rover_source = f"lang_flip_override_{candidate_label}"
                break
    elif whisper_word_diff and rover and rover.strip():
        # Whisper has a word-level substitution error vs PhoWhisper (Jaccard
        # 25-95%, e.g. "ra gián" vs "ra dáng"). The pipeline's word-vote
        # ROVER across all 3 ASRs picks the majority — more reliable than
        # WH alone for fine-grained errors.
        rover_final = rover
        rover_source = "pipeline_rover_word_vote"
    else:
        # Top-tier selection: pick among the canonical 3 (Whisper/PhoWhisper/
        # ChunkFormer) by length-tier, with Whisper > PhoWhisper > ChunkFormer
        # for tie-breaks.
        max_len = max(wh_len, pw_len, ck_len)
        if max_len > 0:
            TOP_TIER_FLOOR = 0.95
            threshold = max_len * TOP_TIER_FLOOR
            if wh_len >= threshold:
                rover_final, rover_source = wh, "whisper_written_form"
            elif pw_len >= threshold:
                rover_final, rover_source = pw, "phowhisper_written_form"
            elif ck_len >= threshold:
                rover_final, rover_source = ck, "chunkformer_most_complete"

    # 5-way ROVER attempted 2026-06-21 and REVERTED. Adding Qwen3 + Parakeet
    # as 4th/5th voters in the pipeline's native align_and_vote (ported in
    # rover_voter.py) produced net -4 words across 19 user-corrected segments
    # in havesip_rhymastic_95. Root cause: Qwen3 + Parakeet vote with the
    # attention-model crowd (WH, PW), diluting ChunkFormer's boundary-recall
    # advantage. CK alone catches words like "à", "còn", "thôi" at segment
    # edges; with 4 attention-style voters vs 1 CTC, CK gets out-voted
    # consistently. A real win would require CTC-aware weighting (CK 2x at
    # sentence edges) or per-position confidence voting — not vanilla
    # majority. See [[project-5way-rover-reverted]].

    # Silent override — final word.  On silent audio, no ASR output is
    # trustworthy (everything is a hallucination on noise floor); blank
    # the ROVER text but keep per-model outputs visible in the viewer for
    # transparency.
    if is_silent:
        rover_final = ""
        rover_source = "silent_suppressed"

    return {
        "conversation_id": conv_id,
        "source_url": src_url,
        "source_channel": src_channel,
        "source_genre": src_genre,
        "audio_sample_rate": sample_rate,

        "turn_id": seg.get("index", f"{idx:05d}"),
        "turn_index": idx,
        "speaker": seg.get("speaker", "?"),
        "original_speaker": seg.get("original_speaker", seg.get("speaker", "?")),
        "start": float(seg.get("start", 0)),
        "end": float(seg.get("end", 0)),
        "duration": duration,
        "audio_path": audio_file,
        "audio_mean_dbfs": mean_dbfs,
        "audio_max_dbfs": max_dbfs,
        "whisper_language": whisper_language,
        "rover_source": rover_source,

        # Dual-form transcript: `rover` = spoken-form (majority of VN-tuned ASRs
        # collapse entities to phonetic VN like `xi i ô`); `text_written` = the
        # written form preserved by Whisper (e.g. `CEO`).  When Whisper has
        # fully language-flipped, text_written is None — that output isn't a
        # useful written form, it's English-language hallucination.
        "text_written": (wh if (wh and wh.strip() and not language_flipped) else None),

        "asr": {
            "rover": rover_final,
            "rover_original": rover,
            "whisper": wh,
            "phowhisper": pw,
            "chunkformer": ck,
        },
        "gold": None,

        "flags": {
            # backchannel only fires when the segment actually has audio.
            # Check BOTH the written-form rover_final (Whisper-preferred) AND
            # the spoken-form rover (pipeline ROVER) — VN tokens like `ừ` may
            # only survive in the spoken form.
            "is_backchannel": (is_backchannel(rover_final) or is_backchannel(rover)) and not is_silent,
            # is_overlap fires when Sortformer detected co-speech during this
            # segment (>=5 % of frames with 2+ active speakers).  Falls back to
            # the pipeline's is_separated flag ONLY if no Sortformer probs are
            # available — that flag is unreliable (fires on every segment).
            "is_overlap": (
                (cospeech_frac is not None and cospeech_frac > 0.05)
                if cospeech_frac is not None
                else bool(seg.get("is_separated", False))
            ),
            "is_short": duration < 1.0,
            "is_silent": is_silent,
            "diar_orphan_candidate": e["diar_orphan"],
            "language_flipped": language_flipped,
            # overlap_confidence signals how likely this segment holds two
            # speakers in <1s (SPEAKER_00 says "chưa", SPEAKER_01 says "em cảm
            # giác là nó" → pipeline mis-attributes to one speaker).
            # Downstream half-duplex training should NOT drop these — they are
            # the exact turn-transition cases the model needs. Instead, use the
            # score to weight the training loss or route to a human-review
            # queue.
            #
            # Score derivation:
            #   is_overlap flag             +0.35
            #   duration < 1.0s             +0.20
            #   ASR text disagreement       +0.15..0.35 (WER between rover
            #                                            and each of whisper /
            #                                            phowhisper / chunkformer)
            "cospeech_frac": (round(cospeech_frac, 3) if cospeech_frac is not None else None),
            "overlap_confidence": _overlap_confidence(seg, rover, wh, pw, ck, duration, cospeech_frac),
        },
        # `contested_words` are filled AFTER force-alignment attaches word
        # timestamps to `rec["words"]` — see the loop below.
        "contested_words": [],

        "context": {
            "prev_speaker": e["prev"]["speaker"] if e["prev"] else None,
            "prev_end": float(e["prev"]["end"]) if e["prev"] else None,
            "gap_to_prev_s": e["gap_to_prev"],
            "next_speaker": e["next"]["speaker"] if e["next"] else None,
            "next_start": float(e["next"]["start"]) if e["next"] else None,
            "is_same_speaker_consecutive": e["same_speaker_consecutive"],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="pipeline output JSON")
    p.add_argument("--out", type=Path, default=None, help="output JSONL path (default: alongside input)")
    p.add_argument("--source-url", default=None)
    p.add_argument("--source-channel", default=None)
    p.add_argument("--source-genre", default=None, help="podcast_interview|news|street_interview|talkshow|livestream|vlog")
    p.add_argument("--conversation-id", default=None, help="default: input filename stem")
    p.add_argument(
        "--observers-dir", type=Path, default=None,
        help="dir with transcript_<slug>.json side-cars (default: <input.parent>/04_asr/)",
    )
    args = p.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        raise SystemExit(f"no segments in {args.input}")

    # Apply optional speaker-merge map produced by scripts/merge_speakers.py
    merge_map_path = args.input.parent / "merged_speaker_map.json"
    if merge_map_path.exists():
        try:
            mm = json.loads(merge_map_path.read_text(encoding="utf-8"))
            merge_map = mm.get("map", {})
            if merge_map:
                segments = apply_speaker_merge(segments, merge_map)
                print(f"  + applied speaker merge: {mm.get('detected_count', '?')} → "
                      f"{mm.get('merged_count', '?')} speakers")
        except (json.JSONDecodeError, OSError):
            print(f"  ! could not read {merge_map_path}, skipping merge")

    conv_id = args.conversation_id or args.input.stem
    sample_rate = int(data.get("metadata", {}).get("sample_rate") or 16000)

    # Audio files in the pipeline output live in <input.stem>/NNNNN_<spk>.mp3.
    audio_subdir = args.input.stem

    # Load any observer side-car transcripts (transcript_<slug>.json) and key
    # them by position (smoke pipeline output doesn't carry stable indices).
    observers_dir = args.observers_dir or (args.input.parent / "04_asr")
    observer_texts: dict[str, list[str]] = {}  # slug -> list[text] by position
    if observers_dir.exists():
        for p in sorted(observers_dir.glob("transcript_*.json")):
            slug = p.stem.removeprefix("transcript_")
            obs_data = json.loads(p.read_text(encoding="utf-8"))
            obs_segs = obs_data.get("segments", [])
            if len(obs_segs) == len(segments):
                observer_texts[slug] = [s.get("text", "") for s in obs_segs]
                print(f"  + observer: {slug} ({len(obs_segs)} segments)")
            else:
                print(f"  - skip {slug}: {len(obs_segs)} obs segs != {len(segments)} main segs")

    # Resolve absolute audio root so loudness can probe each segment file
    audio_root_abs = (args.input.parent / audio_subdir).resolve()

    # Optional: per-segment speaker-attribution verification sidecar (built
    # by scripts/verify_speaker_attribution.py). When present, segments whose
    # audio matches a different speaker better than the assigned one (margin
    # > SPEAKER_ASSIGN_MARGIN) are dropped. This catches sub-1s overlap leaks
    # where the separator was skipped and the segment audio is mostly the
    # other speaker bleeding through.
    attr_path = args.input.parent / "attribution_check.json"
    attribution: dict[int, dict] = {}
    if attr_path.exists():
        try:
            attr_data = json.loads(attr_path.read_text(encoding="utf-8"))
            for entry in attr_data.get("segments", []):
                attribution[entry["index"]] = entry
            print(f"  + attribution_check: {attr_data.get('n_misattributed', '?')}/"
                  f"{attr_data.get('n_segments', '?')} flagged as misattributed")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! could not read {attr_path}: {e}")

    # Optional: per-segment BGM sidecar (built by scripts/check_bgm.py).
    # Segments with detected background music are flagged has_bgm=True and
    # dropped from the export — BGM contamination makes them unusable for
    # speech-model training (user directive 2026-06-14).
    bgm_path = args.input.parent / "bgm_check.json"
    bgm: dict[int, dict] = {}
    if bgm_path.exists():
        try:
            bgm_data = json.loads(bgm_path.read_text(encoding="utf-8"))
            for entry in bgm_data.get("segments", []):
                bgm[entry["index"]] = entry
            print(f"  + bgm_check: {bgm_data.get('n_bgm', '?')}/"
                  f"{bgm_data.get('n_segments', '?')} flagged as has_bgm")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! could not read {bgm_path}: {e}")

    # Optional: per-segment force-alignment sidecar (built by
    # scripts/force_align_boundaries.py). When present, segment start/end
    # are replaced with wav2vec2's refined boundaries (≈20ms precision vs
    # Sortformer's ±100ms). Per-word timestamps are also stored.
    align_path = args.input.parent / "force_alignment.json"
    alignment: dict[int, dict] = {}
    if align_path.exists():
        try:
            align_data = json.loads(align_path.read_text(encoding="utf-8"))
            for entry in align_data.get("segments", []):
                alignment[entry["index"]] = entry
            print(f"  + force_alignment: {align_data.get('n_aligned', '?')}/"
                  f"{align_data.get('n_segments', '?')} aligned with wav2vec2-VN")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! could not read {align_path}: {e}")

    # Optional: force-aligned bucketed ROVER sidecar (built by
    # scripts/force_aligned_rover.py). When present, adds a NEW field
    # `rover_force_aligned` to each record alongside the existing `rover` —
    # doesn't replace anything, so deleting the sidecar reverts cleanly.
    fa_rover_path = args.input.parent / "force_aligned_rover.json"
    fa_rover: dict[int, dict] = {}
    if fa_rover_path.exists():
        try:
            fa_data = json.loads(fa_rover_path.read_text(encoding="utf-8"))
            for entry in fa_data.get("segments", []):
                fa_rover[entry["index"]] = entry
            print(f"  + force_aligned_rover: {fa_data.get('n_voted', '?')}/"
                  f"{fa_data.get('n_segments', '?')} segments voted")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! could not read {fa_rover_path}: {e}")

    # Optional: per-word speaker classification sidecar (built by
    # scripts/verify_word_attribution.py). Segments flagged multi_speaker=True
    # contain content from >1 speaker (Sortformer merged turns) and are
    # dropped — they can't be cleanly attributed without splitting.
    word_attr_path = args.input.parent / "word_attribution.json"
    word_attribution: dict[int, dict] = {}
    if word_attr_path.exists():
        try:
            wa_data = json.loads(word_attr_path.read_text(encoding="utf-8"))
            for entry in wa_data.get("segments", []):
                word_attribution[entry["index"]] = entry
            print(f"  + word_attribution: {wa_data.get('n_multi_speaker', '?')}/"
                  f"{wa_data.get('n_segments', '?')} flagged as multi-speaker")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! could not read {word_attr_path}: {e}")

    # Optional: Sortformer per-frame probability tensor (built by
    # scripts/dump_sortformer_probs.py). Used to compute per-segment
    # cospeech_frac — a direct signal that >=2 speakers overlap in this
    # segment. Falls through to `is_separated` flag when the sidecar is
    # missing.
    sortformer_probs = None
    sortformer_stride = 0.08
    sf_probs_path = args.input.parent / "sortformer_probs.npy"
    sf_meta_path = args.input.parent / "sortformer_probs.json"
    if sf_probs_path.exists() and sf_meta_path.exists():
        try:
            import numpy as np
            sortformer_probs = np.load(sf_probs_path)
            sf_meta = json.loads(sf_meta_path.read_text(encoding="utf-8"))
            sortformer_stride = float(sf_meta.get("frame_stride_s", 0.08))
            print(f"  + sortformer_probs: {sortformer_probs.shape[0]} frames "
                  f"@ {sortformer_stride}s")
        except Exception as ex:
            print(f"  ! could not read sortformer_probs: {ex}")
            sortformer_probs = None

    def _cospeech_frac(start: float, end: float) -> float | None:
        if sortformer_probs is None:
            return None
        f0 = max(0, int(start / sortformer_stride))
        f1 = min(sortformer_probs.shape[0], int(end / sortformer_stride) + 1)
        if f1 <= f0:
            return 0.0
        window = sortformer_probs[f0:f1]
        active = window > 0.5
        cospeech = (active.sum(axis=1) >= 2).sum()
        return float(cospeech) / (f1 - f0)

    enriched = normalize_segments(segments)
    records = []
    dropped_phantom = 0
    dropped_misattributed = 0
    dropped_bgm = 0
    dropped_multi_speaker = 0
    refined_boundaries = 0
    qwen3_list = observer_texts.get("qwen3-asr", [])
    parakeet_list = observer_texts.get("parakeet-vn", [])
    for i, e in enumerate(enriched):
        qw_t = qwen3_list[i] if i < len(qwen3_list) else ""
        pa_t = parakeet_list[i] if i < len(parakeet_list) else ""
        seg = e["seg"]
        cs_frac = _cospeech_frac(float(seg.get("start", 0)), float(seg.get("end", 0)))
        rec = make_record(
            conv_id, args.source_url, args.source_channel, args.source_genre,
            sample_rate, audio_subdir, e, i, audio_root_abs=audio_root_abs,
            qwen3_text=qw_t, parakeet_text=pa_t,
            cospeech_frac=cs_frac,
        )
        for slug, texts in observer_texts.items():
            rec["asr"][slug] = texts[i] if i < len(texts) else ""

        # Post-merge: if language flipped AND phowhisper was empty, fall back to
        # an observer (qwen3-asr / parakeet-vn) before settling on chunkformer.
        if rec["flags"]["language_flipped"] and rec["rover_source"].startswith("lang_flip_override_chunkformer"):
            for slug in ("qwen3-asr", "parakeet-vn"):
                obs = rec["asr"].get(slug, "")
                if obs and obs.strip():
                    rec["asr"]["rover"] = obs
                    rec["rover_source"] = f"lang_flip_override_{slug}"
                    rec["flags"]["is_backchannel"] = is_backchannel(obs) and not rec["flags"]["is_silent"]
                    break

        # Phantom-segment filter: ChunkFormer is the only non-hallucinating
        # member of the ensemble (CTC, no LM). When it emits nothing on a
        # sub-1s segment, the segment is almost always silence/bleed inside
        # an overlap region — Whisper/PhoWhisper hallucinate, ROVER picks
        # the hallucination, and the segment pollutes the dataset.
        if rec["duration"] < 1.0 and not rec["asr"]["chunkformer"].strip():
            dropped_phantom += 1
            continue

        # Misattribution filter: when the attribution_check sidecar flags
        # this segment, the audio matches a different speaker better than
        # the assigned one (Sortformer mis-assigned a sub-1s overlap, etc).
        if attribution.get(i, {}).get("is_misattributed", False):
            dropped_misattributed += 1
            continue

        # BGM filter: drop segments with background music (user directive
        # 2026-06-14 — unusable for speech-model training).
        if bgm.get(i, {}).get("has_bgm", False):
            dropped_bgm += 1
            continue

        # Multi-speaker filter: DISABLED 2026-06-18 after spot-check showed
        # 12% TP rate (5 TP / 38 FP across 43 user votes). ECAPA sims on
        # short word-windows are too noisy at the current SPEAKER_MARGIN=0.05
        # threshold. word_attribution.json is still produced (for future
        # tuning) but no records are dropped based on it.
        # if word_attribution.get(i, {}).get("multi_speaker", False):
        #     dropped_multi_speaker += 1
        #     continue

        # Force-alignment refinement: replace Sortformer's ±100ms boundaries
        # with wav2vec2's word-level start/end (~20ms precision). Recovers
        # boundary-dropped content like seg 1's leading "thì nhận ra là".
        align_entry = alignment.get(i, {})
        if align_entry.get("aligned"):
            rec["start"] = align_entry["refined_start"]
            rec["end"] = align_entry["refined_end"]
            rec["duration"] = rec["end"] - rec["start"]
            rec["words"] = align_entry.get("words", [])
            refined_boundaries += 1

        # Contested-word detection: catches the "ừ, ừm" backchannel-contamination
        # pattern where PhoWhisper/ChunkFormer transcribed a quiet backchannel
        # from the OTHER speaker into a long turn. Runs after FA so we have
        # word-level timestamps to verify against Sortformer.
        if rec.get("words"):
            rec["contested_words"] = _find_contested_words(
                rec["words"], rec["duration"], cospeech_frac_fn=_cospeech_frac
            )

        # Force-aligned bucketed ROVER (additive — doesn't replace `rover`).
        # If present, expose as `rover_force_aligned` so the viewer can render
        # both side-by-side and we have a clean rollback path.
        fa_entry = fa_rover.get(i, {})
        fa_result = fa_entry.get("fa_rover") if isinstance(fa_entry, dict) else None
        if fa_result and fa_result.get("text"):
            rec["rover_force_aligned"] = fa_result["text"]

        records.append(rec)

    # Auto-apply human corrections if a sidecar `corrections.json` is present.
    # This preserves gold text and speaker remaps across normalize re-runs.
    corr_path = args.input.parent / "corrections.json"
    corr_applied_text = corr_applied_remap = 0
    if corr_path.exists():
        try:
            corr = json.loads(corr_path.read_text(encoding="utf-8"))
            text_corr = corr.get("text_corrections", {}) or {}
            spk_remap = corr.get("speaker_remap", {}) or {}
            now = _dt.now().strftime("%Y-%m-%d") if False else __import__("datetime").datetime.now().strftime("%Y-%m-%d")
            for r in records:
                tid = str(r.get("turn_id", ""))
                if tid in text_corr:
                    r["gold"] = {
                        "text": text_corr[tid],
                        "note": "",
                        "annotator": "manual_review",
                        "annotated_at": now,
                    }
                    corr_applied_text += 1
                if spk_remap and r.get("speaker") in spk_remap:
                    r["speaker"] = spk_remap[r["speaker"]]
                    corr_applied_remap += 1
            if corr_applied_text or corr_applied_remap:
                print(f"  + applied corrections.json: "
                      f"{corr_applied_text} gold texts, "
                      f"{corr_applied_remap} speaker remaps")
        except (json.JSONDecodeError, OSError) as ex:
            print(f"  ! could not apply corrections.json: {ex}")

    out_path = args.out or args.input.with_suffix(".jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary stats
    n = len(records)
    bc = sum(r["flags"]["is_backchannel"] for r in records)
    short = sum(r["flags"]["is_short"] for r in records)
    orphan = sum(r["flags"]["diar_orphan_candidate"] for r in records)
    overlap = sum(r["flags"]["is_overlap"] for r in records)
    print(f"Wrote {out_path}")
    print(
        f"  {n} turns · backchannels={bc} · short(<1s)={short} · "
        f"orphan_candidate={orphan} · overlap={overlap} · "
        f"dropped_phantom={dropped_phantom} · "
        f"dropped_misattributed={dropped_misattributed} · "
        f"dropped_bgm={dropped_bgm} · "
        f"dropped_multi_speaker={dropped_multi_speaker} · "
        f"refined_boundaries={refined_boundaries}"
    )


if __name__ == "__main__":
    main()
