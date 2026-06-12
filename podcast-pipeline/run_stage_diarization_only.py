from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import librosa
import lightning_fabric.utilities.cloud_io as cloud_io
import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
from pyannote.audio import Inference
from nemo.collections.asr.models import SortformerEncLabelModel

from models import silero_vad
from utils.audio_preprocessing import (
    prepare_diarization_chunks,
    set_logger as set_audio_preprocessing_logger,
    standardization,
)
from utils.diarization import (
    df_to_list,
    set_logger as set_diarization_logger,
    sortformer_dia,
    split_long_segments,
)
from utils.logger import Logger
from utils.stage_diarization import (
    apply_sortformer_streaming_config,
    build_run_dir,
    collect_audio_paths,
    refine_speaker_boundaries,
    resolve_config_path,
    resolve_sortformer_postprocessing_yaml,
    write_boundary_refinement_report,
    write_input_artifacts,
)
from utils.tool import check_env, detect_gpu, load_cfg
from utils.trace_artifacts import TraceRunWriter


_original_load = cloud_io._load


def _patched_load(path_or_url, map_location=None):
    if not isinstance(path_or_url, (str, Path)):
        return torch.load(path_or_url, map_location=map_location, weights_only=False)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)
    from lightning_fabric.utilities.cloud_io import get_filesystem

    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as fh:
        return torch.load(fh, map_location=map_location, weights_only=False)


cloud_io._load = _patched_load


def _device_name_from_index(device_index: int) -> str:
    if not torch.cuda.is_available() or device_index < 0:
        return "cpu"
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device_index} is not available. Visible CUDA device count: {torch.cuda.device_count()}."
        )
    return f"cuda:{device_index}"


def _torch_device_from_index(device_index: int) -> torch.device:
    return torch.device(_device_name_from_index(device_index))


def _apply_sortformer_segment_padding_from_args(
    df: pd.DataFrame, args, audio_duration: float | None = None
) -> pd.DataFrame:
    if df is None or df.empty or not getattr(args, "sortformer_param", False):
        return df
    pad_onset = float(getattr(args, "sortformer_pad_onset", 0.0))
    pad_offset = float(getattr(args, "sortformer_pad_offset", 0.0))
    if pad_onset == 0.0 and pad_offset == 0.0:
        return df
    df = df.copy()
    df["start"] = (df["start"].astype(float) + pad_onset).clip(lower=0.0)
    df["end"] = df["end"].astype(float) + pad_offset
    if audio_duration is not None and audio_duration > 0:
        df["end"] = df["end"].clip(lower=0.0, upper=float(audio_duration))
    else:
        df["end"] = df["end"].clip(lower=0.0)
    df["end"] = df[["start", "end"]].max(axis=1)
    return df


def convert_opus_to_wav_if_needed(audio_path: str, target_sr: int, logger) -> tuple[str, str | None]:
    lower = audio_path.lower()
    if not (lower.endswith(".opus") or lower.endswith(".ogg")):
        return audio_path, None

    temp_dir = tempfile.mkdtemp(prefix="opus2wav_")
    out_wav = os.path.join(temp_dir, "converted.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-sample_fmt",
        "s16",
        out_wav,
    ]

    try:
        logger.info(f"[OPUS] Converting to wav via ffmpeg: {audio_path} -> {out_wav}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_wav):
            logger.error(f"[OPUS] ffmpeg conversion failed.\nSTDERR:\n{proc.stderr}")
            raise RuntimeError("ffmpeg opus->wav conversion failed")
        return out_wav, temp_dir
    except Exception as exc:
        logger.warning(f"[OPUS] ffmpeg path failed, trying pydub fallback: {exc}")
        try:
            seg = AudioSegment.from_file(audio_path)
            seg = seg.set_channels(1).set_frame_rate(int(target_sr)).set_sample_width(2)
            seg.export(out_wav, format="wav")
            if not os.path.exists(out_wav):
                raise RuntimeError("pydub export failed")
            return out_wav, temp_dir
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise


def _extract_speaker_embedding(
    audio_info,
    start: float,
    end: float,
    embedder: Inference | None,
    sample_window: float = 2.0,
    min_duration: float = 0.5,
):
    if embedder is None:
        return None
    waveform = audio_info.get("waveform")
    sample_rate = audio_info.get("sample_rate")
    if waveform is None or sample_rate is None:
        return None
    total_duration = len(waveform) / sample_rate
    start = max(0.0, min(start, total_duration))
    end = max(start, min(end, total_duration))
    duration = end - start
    if duration < min_duration:
        return None
    if duration > sample_window:
        center = (start + end) / 2.0
        start = center - sample_window / 2.0
        end = center + sample_window / 2.0
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment = waveform[start_idx:end_idx]
    if segment.size == 0:
        return None
    target_sr = getattr(embedder, "sample_rate", 16000)
    try:
        if sample_rate != target_sr:
            segment = librosa.resample(segment, orig_sr=sample_rate, target_sr=target_sr)
    except Exception:
        target_sr = sample_rate
    try:
        torch_seg = torch.as_tensor(segment, dtype=torch.float32).unsqueeze(0)
        emb = embedder({"waveform": torch_seg, "sample_rate": target_sr})
    except Exception:
        return None
    if emb is None:
        return None
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    if isinstance(emb, np.ndarray) and emb.ndim > 1:
        emb = emb.mean(axis=0)
    return emb


def _cosine_similarity(vec_a, vec_b) -> float:
    if vec_a is None or vec_b is None:
        return -1.0
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return -1.0
    return float(np.dot(vec_a, vec_b) / denom)


def _compute_chunk_speaker_centroids(chunk_df: pd.DataFrame, audio_info, embedder: Inference | None):
    if embedder is None or chunk_df is None or chunk_df.empty:
        return {}
    centroids: dict[str, np.ndarray] = {}
    for speaker, rows in chunk_df.groupby("speaker"):
        embeddings = []
        for _, row in rows.sort_values("start").iterrows():
            emb = _extract_speaker_embedding(audio_info, row["start"], row["end"], embedder=embedder)
            if emb is not None:
                embeddings.append(emb)
            if len(embeddings) >= 3:
                break
        if embeddings:
            centroids[speaker] = np.mean(embeddings, axis=0)
    return centroids


def align_speakers_across_chunks(
    chunk_frames: list[pd.DataFrame],
    audio_info,
    embedder: Inference | None,
    similarity_threshold: float = 0.75,
    logger=None,
):
    if embedder is None or not chunk_frames:
        if logger is not None:
            logger.warning("Speaker embedder unavailable; skipping cross-chunk speaker linking.")
        return chunk_frames

    global_centroids: dict[str, np.ndarray | None] = {}
    global_counts: dict[str, int] = {}
    next_global_idx = 0
    aligned_frames: list[pd.DataFrame] = []

    for df in chunk_frames:
        if df is None or df.empty:
            aligned_frames.append(df)
            continue
        local_centroids = _compute_chunk_speaker_centroids(df, audio_info, embedder)
        mapping: dict[str, str] = {}
        used_global_ids_in_chunk: set[str] = set()

        for local_speaker in df["speaker"].unique():
            emb = local_centroids.get(local_speaker)
            best_id = None
            best_sim = -1.0
            if emb is not None:
                for gid, centroid in global_centroids.items():
                    if centroid is None or gid in used_global_ids_in_chunk:
                        continue
                    sim = _cosine_similarity(emb, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = gid
            if best_sim >= similarity_threshold and best_id is not None:
                mapping[local_speaker] = best_id
                used_global_ids_in_chunk.add(best_id)
                count = global_counts.get(best_id, 0)
                global_centroids[best_id] = (global_centroids[best_id] * count + emb) / (count + 1)
                global_counts[best_id] = count + 1
            else:
                global_id = f"SPEAKER_{next_global_idx:02d}"
                next_global_idx += 1
                mapping[local_speaker] = global_id
                used_global_ids_in_chunk.add(global_id)
                global_centroids[global_id] = emb
                global_counts[global_id] = 1 if emb is not None else 0

        remapped_df = df.copy()
        remapped_df["speaker"] = remapped_df["speaker"].map(mapping)
        aligned_frames.append(remapped_df)

    return aligned_frames


def process_audio(
    audio_path: str,
    args,
    cfg: dict[str, Any],
    logger,
    vad_model,
    diar_model,
    speaker_embedder,
) -> Path:
    target_sr = int(cfg["entrypoint"]["SAMPLE_RATE"])
    proc_audio_path, opus_temp_dir = convert_opus_to_wav_if_needed(audio_path, target_sr=target_sr, logger=logger)
    temp_chunk_dir = None
    try:
        audio = standardization(proc_audio_path, cfg)
        audio_duration = len(audio["waveform"]) / audio["sample_rate"]
        diar_chunks, temp_chunk_dir = prepare_diarization_chunks(
            proc_audio_path,
            audio,
            vad_model if args.vad else None,
            silero_vad,
            max_duration=float(args.max_dia_chunk_duration),
            min_silence=float(args.min_split_silence),
        )

        run_dir = build_run_dir(Path(args.output_root), audio_path)
        trace_writer = TraceRunWriter(run_dir, source_audio_path=audio_path, logger=logger)
        write_input_artifacts(run_dir, audio, diar_chunks)
        sortformer_postprocessing_yaml_path = resolve_sortformer_postprocessing_yaml(args, run_dir)
        sortformer_postprocessing_yaml = (
            str(sortformer_postprocessing_yaml_path) if sortformer_postprocessing_yaml_path else None
        )
        if sortformer_postprocessing_yaml:
            logger.info(f"Using Sortformer postprocessing YAML: {sortformer_postprocessing_yaml}")

        logger.info(f"Stage 01 diarization only: {audio_path}")
        dia_start = time.time()
        diarization_frames: list[pd.DataFrame] = []
        try:
            for chunk in diar_chunks:
                predicted_segments, _ = diar_model.diarize(
                    audio=chunk["path"],
                    batch_size=int(args.sortformer_batch_size),
                    include_tensor_outputs=True,
                    postprocessing_yaml=sortformer_postprocessing_yaml,
                    num_workers=int(args.sortformer_num_workers),
                )
                chunk_df = sortformer_dia(predicted_segments)
                if not chunk_df.empty:
                    chunk_df["start"] += chunk["offset"]
                    chunk_df["end"] += chunk["offset"]
                    chunk_df = _apply_sortformer_segment_padding_from_args(
                        chunk_df,
                        args=args,
                        audio_duration=audio_duration,
                    )
                diarization_frames.append(chunk_df)
        finally:
            if temp_chunk_dir:
                shutil.rmtree(temp_chunk_dir, ignore_errors=True)

        diarization_frames = align_speakers_across_chunks(
            diarization_frames,
            audio_info=audio,
            embedder=speaker_embedder,
            similarity_threshold=float(args.speaker_link_threshold),
            logger=logger,
        )
        if diarization_frames:
            speakerdia = pd.concat(diarization_frames, ignore_index=True)
        else:
            speakerdia = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])

        boundary_refine_parameters = {
            "enabled": bool(args.speaker_boundary_refinement),
            "max_shift": float(args.boundary_refine_max_shift),
            "step": float(args.boundary_refine_step),
            "embedding_window": float(args.boundary_refine_embed_window),
            "reference_min_segment": float(args.boundary_refine_reference_min_segment),
            "nested_max_segment": float(args.boundary_refine_nested_max_segment),
            "min_segment": float(args.boundary_refine_min_segment),
            "max_gap": float(args.boundary_refine_max_gap),
            "min_improvement": float(args.boundary_refine_min_improvement),
            "speaker_embedder_loaded": speaker_embedder is not None,
        }
        boundary_refinements: list[dict[str, Any]] = []
        boundary_refinement_report = None
        if bool(args.speaker_boundary_refinement):
            if speaker_embedder is None:
                logger.warning("Speaker boundary refinement enabled but speaker embedder is unavailable; skipping.")
            else:
                def boundary_embedding_fn(start: float, end: float):
                    duration = max(0.0, float(end) - float(start))
                    min_duration = max(
                        0.05,
                        min(float(args.boundary_refine_embed_window) * 0.75, duration),
                    )
                    return _extract_speaker_embedding(
                        audio,
                        start,
                        end,
                        embedder=speaker_embedder,
                        sample_window=max(duration, 0.05),
                        min_duration=min_duration,
                    )

                speakerdia, boundary_refinements = refine_speaker_boundaries(
                    speakerdia,
                    embedding_fn=boundary_embedding_fn,
                    max_shift=float(args.boundary_refine_max_shift),
                    step=float(args.boundary_refine_step),
                    embedding_window=float(args.boundary_refine_embed_window),
                    min_segment=float(args.boundary_refine_min_segment),
                    reference_min_segment=float(args.boundary_refine_reference_min_segment),
                    nested_max_segment=float(args.boundary_refine_nested_max_segment),
                    max_gap=float(args.boundary_refine_max_gap),
                    min_improvement=float(args.boundary_refine_min_improvement),
                    logger=logger,
                )
            boundary_refinement_report = write_boundary_refinement_report(
                run_dir,
                boundary_refinements,
                boundary_refine_parameters,
            )

        segment_list = split_long_segments(df_to_list(speakerdia))
        dia_end = time.time()
        rt = (dia_end - dia_start) / audio_duration if audio_duration > 0 else 0.0
        trace_writer.write_diarization(
            segment_list,
            metadata={
                "audio_path": audio_path,
                "audio_duration_seconds": audio_duration,
                "sample_rate": audio["sample_rate"],
                "audio_gain_clamp_db": float(args.audio_gain_clamp_db),
                "speaker_boundary_refinement_enabled": bool(args.speaker_boundary_refinement),
                "speaker_boundary_refinement_count": len(boundary_refinements),
                "speaker_boundary_refinement_report": str(boundary_refinement_report) if boundary_refinement_report else None,
                "boundary_refine_max_shift": float(args.boundary_refine_max_shift),
                "boundary_refine_step": float(args.boundary_refine_step),
                "boundary_refine_embed_window": float(args.boundary_refine_embed_window),
                "boundary_refine_reference_min_segment": float(args.boundary_refine_reference_min_segment),
                "boundary_refine_nested_max_segment": float(args.boundary_refine_nested_max_segment),
                "boundary_refine_min_segment": float(args.boundary_refine_min_segment),
                "boundary_refine_max_gap": float(args.boundary_refine_max_gap),
                "boundary_refine_min_improvement": float(args.boundary_refine_min_improvement),
                "processing_time_seconds": dia_end - dia_start,
                "rt_factor": rt,
                "speaker_link_threshold": float(args.speaker_link_threshold),
                "vad_enabled_for_chunking": bool(args.vad),
                "sortformer_model_name": args.sortformer_model_name,
                "sortformer_streaming_config_enabled": bool(args.sortformer_streaming_config),
                "sortformer_chunk_len": int(args.sortformer_chunk_len),
                "sortformer_chunk_left_context": int(args.sortformer_chunk_left_context),
                "sortformer_chunk_right_context": int(args.sortformer_chunk_right_context),
                "sortformer_fifo_len": int(args.sortformer_fifo_len),
                "sortformer_spkcache_update_period": int(args.sortformer_spkcache_update_period),
                "sortformer_spkcache_len": int(args.sortformer_spkcache_len),
                "sortformer_postprocessing_enabled": bool(sortformer_postprocessing_yaml),
                "sortformer_postprocessing_yaml": sortformer_postprocessing_yaml,
                "sortformer_pp_onset": float(args.sortformer_pp_onset),
                "sortformer_pp_offset": float(args.sortformer_pp_offset),
                "sortformer_pp_pad_onset": float(args.sortformer_pp_pad_onset),
                "sortformer_pp_pad_offset": float(args.sortformer_pp_pad_offset),
                "sortformer_pp_min_duration_on": float(args.sortformer_pp_min_duration_on),
                "sortformer_pp_min_duration_off": float(args.sortformer_pp_min_duration_off),
                "sortformer_batch_size": int(args.sortformer_batch_size),
                "sortformer_num_workers": int(args.sortformer_num_workers),
                "sortformer_pad_onset": float(args.sortformer_pad_onset),
                "sortformer_pad_offset": float(args.sortformer_pad_offset),
            },
        )
        logger.info(f"Saved diarization-only artifacts to {run_dir}")
        return run_dir
    finally:
        if opus_temp_dir:
            shutil.rmtree(opus_temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the speaker diarization stage of Sommelier.")
    parser.add_argument("--input_audio_path", type=str, default="", help="Single audio file to process.")
    parser.add_argument("--input_folder_path", type=str, default="", help="Folder of audio files to process.")
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to config JSON.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="sommelier_batch_outputs/diarization_only_runs",
        help="Directory where diarization-only run folders will be created.",
    )
    parser.add_argument("--vad", action=argparse.BooleanOptionalAction, default=True, help="Use Silero VAD to help pre-diarization chunk splitting.")
    parser.add_argument("--audio-gain-clamp-db", type=float, default=6.0, help="Maximum absolute gain in dB applied during input audio normalization.")
    parser.add_argument("--speaker-boundary-refinement", action=argparse.BooleanOptionalAction, default=False, help="Use speaker embeddings to refine close speaker-change boundaries after diarization.")
    parser.add_argument("--boundary-refine-max-shift", type=float, default=0.4, help="Maximum seconds a speaker boundary may move during refinement.")
    parser.add_argument("--boundary-refine-step", type=float, default=0.05, help="Seconds between candidate boundary positions during refinement.")
    parser.add_argument("--boundary-refine-embed-window", type=float, default=0.4, help="Seconds of audio on each side of a candidate boundary for embedding scoring.")
    parser.add_argument("--boundary-refine-reference-min-segment", type=float, default=2.0, help="Minimum segment duration used when building speaker reference embeddings.")
    parser.add_argument("--boundary-refine-nested-max-segment", type=float, default=1.0, help="Maximum duration of a nested short segment eligible for boundary trimming.")
    parser.add_argument("--boundary-refine-min-segment", type=float, default=0.3, help="Minimum segment duration preserved after boundary refinement.")
    parser.add_argument("--boundary-refine-max-gap", type=float, default=0.35, help="Only refine adjacent speaker turns whose gap or overlap is within this many seconds.")
    parser.add_argument("--boundary-refine-min-improvement", type=float, default=0.05, help="Minimum embedding-score improvement required to accept a boundary shift.")
    parser.add_argument("--speaker-link-threshold", type=float, default=0.75, help="Cosine similarity threshold for linking speakers across chunks.")
    parser.add_argument("--diar_device_index", type=int, default=0, help="CUDA device index for VAD and speaker embedding. Use -1 for CPU.")
    parser.add_argument("--sortformer_device_index", type=int, default=0, help="CUDA device index for Sortformer. Use -1 for CPU.")
    parser.add_argument("--sortformer_model_name", type=str, default="nvidia/diar_streaming_sortformer_4spk-v2.1", help="Hugging Face model id for Sortformer.")
    parser.add_argument("--sortformer_batch_size", type=int, default=1, help="Batch size passed to Sortformer diarize(). NVIDIA recommends 1 for best accuracy.")
    parser.add_argument("--sortformer_num_workers", type=int, default=0, help="DataLoader worker count passed to Sortformer diarize().")
    parser.add_argument("--sortformer-streaming-config", action=argparse.BooleanOptionalAction, default=True, help="Apply streaming Sortformer cache/chunk parameters after model load.")
    parser.add_argument("--sortformer_chunk_len", type=int, default=340, help="Streaming Sortformer chunk size in 80 ms frames.")
    parser.add_argument("--sortformer_chunk_left_context", type=int, default=1, help="Streaming Sortformer left context frames.")
    parser.add_argument("--sortformer_chunk_right_context", type=int, default=40, help="Streaming Sortformer right context frames.")
    parser.add_argument("--sortformer_fifo_len", type=int, default=40, help="Streaming Sortformer FIFO queue size in frames.")
    parser.add_argument("--sortformer_spkcache_update_period", type=int, default=300, help="Streaming Sortformer speaker cache update period in frames.")
    parser.add_argument("--sortformer_spkcache_len", type=int, default=200, help="Streaming Sortformer speaker cache size in frames.")
    parser.add_argument("--sortformer-postprocessing", action=argparse.BooleanOptionalAction, default=True, help="Enable NVIDIA NeMo Sortformer postprocessing YAML.")
    parser.add_argument("--sortformer-postprocessing-yaml", type=str, default="", help="Optional existing NeMo postprocessing YAML path. Overrides generated values.")
    parser.add_argument("--sortformer-pp-onset", type=float, default=0.3, help="NeMo postprocessing onset threshold for speech segment start.")
    parser.add_argument("--sortformer-pp-offset", type=float, default=0.33, help="NeMo postprocessing offset threshold for speech segment end.")
    parser.add_argument("--sortformer-pp-pad-onset", type=float, default=0.015, help="NeMo postprocessing seconds added before segment start.")
    parser.add_argument("--sortformer-pp-pad-offset", type=float, default=0.015, help="NeMo postprocessing seconds added after segment end.")
    parser.add_argument("--sortformer-pp-min-duration-on", type=float, default=0.35, help="NeMo postprocessing minimum speech segment duration.")
    parser.add_argument("--sortformer-pp-min-duration-off", type=float, default=0.35, help="NeMo postprocessing minimum non-speech duration before keeping a split.")
    parser.add_argument("--sortformer-param", dest="sortformer_param", action=argparse.BooleanOptionalAction, default=False, help="Enable post-hoc boundary padding for Sortformer output.")
    parser.add_argument("--sortformer-pad-offset", type=float, default=-0.24, help="Seconds added to segment end.")
    parser.add_argument("--sortformer-pad-onset", type=float, default=0.0, help="Seconds added to segment start.")
    parser.add_argument("--min_split_silence", type=float, default=1.0, help="Minimum silence duration used by VAD chunk splitting.")
    parser.add_argument("--max_dia_chunk_duration", type=float, default=900.0, help="Maximum seconds per diarization chunk before splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(str(resolve_config_path(args.config_path, script_dir=Path(__file__).resolve().parent)))
    args.audio_gain_clamp_db = abs(float(args.audio_gain_clamp_db))
    cfg.setdefault("entrypoint", {})["AUDIO_GAIN_CLAMP_DB"] = float(args.audio_gain_clamp_db)

    logger = Logger.init_logger("stage_diarization_only")
    set_audio_preprocessing_logger(logger)
    set_diarization_logger(logger)

    if detect_gpu():
        logger.info(f"Using GPU. Visible CUDA device count: {torch.cuda.device_count()}")
    else:
        logger.info("Using CPU")
    check_env(logger)

    diar_device = _torch_device_from_index(args.diar_device_index)
    sortformer_device = _torch_device_from_index(args.sortformer_device_index)
    logger.info(f"Device map: diar/vad/speaker-link={diar_device}, sortformer={sortformer_device}")

    logger.info("Loading stage-01 models")
    vad_model = silero_vad.SileroVAD(device=diar_device) if args.vad else None
    speaker_embedder = None
    try:
        speaker_embedder = Inference(
            "pyannote/embedding",
            device=diar_device,
            use_auth_token=cfg["huggingface_token"],
            window="whole",
        )
        logger.info("Speaker embedding model loaded for cross-chunk linking")
    except Exception as exc:
        logger.warning(f"Failed to load speaker embedding model; continuing without cross-chunk speaker linking: {exc}")

    diar_model = SortformerEncLabelModel.from_pretrained(args.sortformer_model_name)
    diar_model = diar_model.to(sortformer_device)
    diar_model.eval()
    apply_sortformer_streaming_config(diar_model, args, logger=logger)
    logger.info(f"Sortformer loaded on {sortformer_device}")

    audio_paths = collect_audio_paths(args, cfg)
    if not audio_paths:
        raise FileNotFoundError("No audio files found to process.")

    output_runs = []
    for audio_path in audio_paths:
        output_runs.append(str(process_audio(audio_path, args, cfg, logger, vad_model, diar_model, speaker_embedder)))

    logger.info("Finished diarization-only stage")
    for run_dir in output_runs:
        logger.info(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
