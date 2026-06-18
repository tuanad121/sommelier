# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT

"""
Speaker diarization utilities for podcast pipeline.
Includes pyannote-based diarization, segment processing, and Sortformer integration.
"""

import datetime
import pandas as pd
import torch
from utils.logger import time_logger

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


@time_logger
def speaker_diarization(audio, dia_pipeline, device):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        dia_pipeline: Pyannote diarization pipeline.
        device: torch device (cuda/cpu).

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """
    logger.debug(f"Start speaker diarization")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        },
        max_speakers=4
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df


@time_logger
def cut_by_speaker_label(vad_list, args):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.
        args: Arguments object containing merge_gap parameter.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = args.merge_gap  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            logger.warning(
                f"cut_by_speaker_label > segment longer than 30s, force trimming to 30s smaller segments"
            )
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH  # update end time
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start  # update start time
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    logger.debug(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list


@time_logger
def detect_overlapping_segments(segment_list, overlap_threshold=0.2):
    """
    Detect segments that overlap for more than overlap_threshold seconds.

    Args:
        segment_list (list): List of segments with 'start', 'end', and 'speaker' keys
        overlap_threshold (float): Minimum overlap duration in seconds to be considered

    Returns:
        list: List of overlapping segment pairs with overlap info
            [{'seg1': segment1, 'seg2': segment2, 'overlap_start': float, 'overlap_end': float, 'overlap_duration': float}]
    """
    overlapping_pairs = []

    # Sort segments by start time
    sorted_segments = sorted(segment_list, key=lambda x: x['start'])

    for i in range(len(sorted_segments)):
        for j in range(i + 1, len(sorted_segments)):
            seg1 = sorted_segments[i]
            seg2 = sorted_segments[j]

            # If seg2 starts after seg1 ends, no more overlaps possible for seg1
            if seg2['start'] >= seg1['end']:
                break

            # Calculate overlap
            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])
            overlap_duration = overlap_end - overlap_start

            # Check if overlap exceeds threshold
            if overlap_duration >= overlap_threshold:
                overlapping_pairs.append({
                    'seg1': seg1,
                    'seg2': seg2,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration
                })
                logger.info(f"Overlap detected: {overlap_duration:.2f}s between "
                           f"[{seg1['start']:.2f}-{seg1['end']:.2f}] and "
                           f"[{seg2['start']:.2f}-{seg2['end']:.2f}]")

    return overlapping_pairs


def sortformer_dia(predicted_segments):
    """
    Convert Sortformer output to pandas DataFrame.

    Args:
        predicted_segments: Sortformer model output segments.

    Returns:
        pd.DataFrame: DataFrame with columns ['segment', 'label', 'speaker', 'start', 'end']
    """
    lists = [x for x in predicted_segments if isinstance(x, (list, tuple))]
    if not lists:
        lists = predicted_segments
    segs = [s for sub in lists for s in sub]

    rows = []
    for idx, seg in enumerate(segs):
        start_s, end_s, sp = seg.split()
        start, end = float(start_s), float(end_s)
        # Convert SPEAKER format
        num = int(sp.split('_')[1])
        speaker = f"SPEAKER_{num:02d}"
        # Label (A, B, C, ...)
        label = chr(ord('A') + idx)
        # Time formatting function
        def fmt(sec):
            td = datetime.timedelta(seconds=sec)
            hrs = td.seconds // 3600 + td.days * 24
            mins = (td.seconds // 60) % 60
            secs = td.seconds % 60
            ms = int(td.microseconds / 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
        segment_str = f"[ {fmt(start)} --> {fmt(end)}]"
        rows.append({
            'segment': segment_str,
            'label': label,
            'speaker': speaker,
            'start': start,
            'end': end
        })

    df = pd.DataFrame(rows, columns=['segment','label','speaker','start','end'])
    df = df.sort_values(by='start').reset_index(drop=True)
    return df


def df_to_list(df: pd.DataFrame) -> list[dict]:
    """
    Convert each row of the DataFrame into a dict and return as a list.
      - index: zero-padded 5-digit string
      - start: float
      - end: float
      - speaker: str
    """
    records = []
    for i, row in df.iterrows():
        records.append({
            'index': f"{i:05d}",
            'start': float(row['start']),
            'end': float(row['end']),
            'speaker': row['speaker']
        })
    return records


def build_micro_overlap_candidates(
    raw_segments: list[dict],
    main_segments: list[dict],
    *,
    main_min_duration: float,
    min_overlap_duration: float = 0.0,
) -> list[dict]:
    """
    Preserve short non-main speaker regions that overlap kept speech segments.

    These candidates are intentionally not promoted to normal transcript/ASR
    segments. Stage 03 can use them as cleanup hints for the longer target
    speaker segment.
    """
    candidates = []
    main_ids = {id(segment) for segment in main_segments}
    min_duration = float(main_min_duration)
    min_overlap = float(min_overlap_duration)

    for raw_idx, raw in enumerate(raw_segments):
        raw_start = float(raw["start"])
        raw_end = float(raw["end"])
        raw_duration = max(0.0, raw_end - raw_start)
        if raw_duration <= 0.0 or raw_duration >= min_duration or id(raw) in main_ids:
            continue

        for target in main_segments:
            if raw.get("speaker") == target.get("speaker"):
                continue
            overlap_start = max(raw_start, float(target["start"]))
            overlap_end = min(raw_end, float(target["end"]))
            overlap_duration = overlap_end - overlap_start
            if overlap_duration <= min_overlap:
                continue

            candidates.append(
                {
                    "index": raw.get("index", f"raw_{raw_idx:05d}"),
                    "speaker": raw.get("speaker"),
                    "start": round(raw_start, 3),
                    "end": round(raw_end, 3),
                    "duration": round(raw_duration, 3),
                    "is_speech_segment": False,
                    "is_overlap_candidate": True,
                    "target_index": target.get("index"),
                    "target_speaker": target.get("speaker"),
                    "target_start": round(float(target["start"]), 3),
                    "target_end": round(float(target["end"]), 3),
                    "overlap_start": round(overlap_start, 3),
                    "overlap_end": round(overlap_end, 3),
                    "overlap_duration": round(overlap_duration, 3),
                }
            )

    return candidates


def split_long_segments(segment_list, max_duration=30.0):
    """
    Split segments longer than max_duration based on time.
    (Does not use VAD)

    Args:
        segment_list (list): List of segment dictionaries to split.
        max_duration (float): Maximum allowed segment length in seconds.

    Returns:
        list: New list of segments after splitting.
    """
    new_segments = []
    new_index = 0

    for segment in segment_list:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        duration = end_time - start_time

        # If segment duration is less than or equal to max duration, add as is
        if duration <= max_duration:
            segment['index'] = str(new_index).zfill(5)
            new_segments.append(segment)
            new_index += 1
        # If segment duration exceeds max duration, split it
        else:
            current_start = start_time
            # Repeat while current start time is less than the original segment's end time
            while current_start < end_time:
                # Calculate next split point (add max duration, but do not exceed original end time)
                chunk_end = min(current_start + max_duration, end_time)

                new_segments.append({
                    'index': str(new_index).zfill(5),
                    'start': round(current_start, 3),
                    'end': round(chunk_end, 3),
                    'speaker': speaker
                })
                new_index += 1
                # Update the next chunk's start time to the current chunk's end time
                current_start = chunk_end

    return new_segments
