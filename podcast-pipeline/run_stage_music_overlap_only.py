from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_config_path(config_path: str, cwd: Path | None = None) -> Path:
    cwd = Path.cwd() if cwd is None else Path(cwd)
    requested = Path(config_path)
    candidates = [requested] if requested.is_absolute() else [
        cwd / requested,
        SCRIPT_DIR / requested,
        SCRIPT_DIR / requested.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"config_path not found: {config_path}. Searched: {searched}")


def load_cfg(config_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _device_name_from_index(index: int | None) -> str:
    import torch

    if index is None or int(index) < 0 or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{int(index)}"


def _torch_device_from_index(index: int | None):
    import torch

    return torch.device(_device_name_from_index(index))


def resolve_stage1_artifacts(
    *,
    input_run_dir: str | Path | None,
    audio_path: str | Path | None,
    diarization_json: str | Path | None,
) -> tuple[Path, Path]:
    run_dir = Path(input_run_dir) if input_run_dir else None
    resolved_audio = Path(audio_path) if audio_path else (run_dir / "00_input" / "full.wav" if run_dir else None)
    resolved_diarization = (
        Path(diarization_json)
        if diarization_json
        else (run_dir / "01_diarization" / "diarization.json" if run_dir else None)
    )

    missing: list[str] = []
    if resolved_audio is None:
        missing.append("--audio_path or --input_run_dir/00_input/full.wav")
    elif not resolved_audio.exists():
        missing.append(str(resolved_audio))
    if resolved_diarization is None:
        missing.append("--diarization_json or --input_run_dir/01_diarization/diarization.json")
    elif not resolved_diarization.exists():
        missing.append(str(resolved_diarization))
    if missing:
        raise FileNotFoundError("Missing stage 1 artifact(s): " + ", ".join(missing))

    return resolved_audio, resolved_diarization


def _normalize_stage1_segment(segment: dict[str, Any], idx: int) -> dict[str, Any]:
    item = dict(segment)
    item.setdefault("index", f"{idx:05d}")
    item["start"] = float(item["start"])
    item["end"] = float(item["end"])
    item["speaker"] = str(item.get("speaker", "UNKNOWN"))
    return item


def load_stage1_diarization(diarization_json: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(Path(diarization_json).read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"Invalid diarization JSON: missing list field 'segments' in {diarization_json}")
    micro_overlap_candidates = payload.get("micro_overlap_candidates", [])
    if micro_overlap_candidates is None:
        micro_overlap_candidates = []
    if not isinstance(micro_overlap_candidates, list):
        raise ValueError(
            f"Invalid diarization JSON: field 'micro_overlap_candidates' must be a list in {diarization_json}"
        )

    normalized_segments = [_normalize_stage1_segment(segment, idx) for idx, segment in enumerate(segments)]
    normalized_candidates = [
        _normalize_stage1_segment(candidate, idx)
        for idx, candidate in enumerate(micro_overlap_candidates)
    ]
    for candidate in normalized_candidates:
        for key in ("target_start", "target_end", "overlap_start", "overlap_end", "overlap_duration"):
            if key in candidate:
                candidate[key] = float(candidate[key])
        if "target_index" in candidate:
            candidate["target_index"] = str(candidate["target_index"])
        if "target_speaker" in candidate:
            candidate["target_speaker"] = str(candidate["target_speaker"])
        candidate["is_speech_segment"] = bool(candidate.get("is_speech_segment", False))
        candidate["is_overlap_candidate"] = bool(candidate.get("is_overlap_candidate", True))
    return normalized_segments, normalized_candidates


def load_stage1_segments(diarization_json: str | Path) -> list[dict[str, Any]]:
    segments, _micro_overlap_candidates = load_stage1_diarization(diarization_json)
    return segments


def _load_panns_model(args, logger):
    try:
        from panns_inference import AudioTagging
    except ModuleNotFoundError as exc:
        if exc.name != "panns_inference":
            raise
        raise ModuleNotFoundError(
            "Missing panns_inference. Install it with `python -m pip install panns-inference` "
            "or rerun the Kaggle notebook dependency cell before running stage 02."
        ) from exc

    panns_data_dir = Path(args.panns_data_dir) if args.panns_data_dir else PROJECT_ROOT / "panns_data"
    checkpoint_path = panns_data_dir / "Cnn14_mAP=0.431.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"PANNs checkpoint not found: {checkpoint_path}. "
            "Run the notebook download cell first."
        )
    os.environ["PANNS_DATA"] = str(panns_data_dir)
    device_name = _device_name_from_index(args.panns_device_index)
    logger.info(f"Loading PANNs model on {device_name}: {checkpoint_path}")
    return AudioTagging(checkpoint_path=str(checkpoint_path), device=device_name)


def _load_demucs_model(args, logger):
    from utils.music_processing import DemucsModel

    device_name = _device_name_from_index(args.demucs_device_index)
    logger.info(f"Loading Demucs model on {device_name}")
    return DemucsModel(model_name=args.demucs_model_name, device=device_name)


def _load_srcorrnet_separator(args, logger):
    from utils.separation import SRCorrNetSeparator

    device = _torch_device_from_index(args.srcorrnet_device_index)
    logger.info(f"Loading SR-CorrNet-SS separator on {device}")
    return SRCorrNetSeparator(device=device)


def process_stage_music_overlap(args) -> Path:
    from utils.audio_preprocessing import set_logger as set_audio_logger
    from utils.audio_preprocessing import standardization
    from utils.diarization import detect_overlapping_segments
    from utils.diarization import set_logger as set_diarization_logger
    from utils.logger import Logger
    from utils.music_processing import preprocess_segments_with_demucs
    from utils.music_processing import set_logger as set_music_logger
    from utils.separation import process_overlapping_segments_with_separation
    from utils.separation import set_logger as set_separation_logger
    from utils.trace_artifacts import TraceRunWriter

    logger = Logger.init_logger("stage_music_overlap_only")
    for setter in (set_audio_logger, set_diarization_logger, set_music_logger, set_separation_logger):
        setter(logger)

    config_path = resolve_config_path(args.config_path)
    cfg = load_cfg(config_path)
    audio_path, diarization_path = resolve_stage1_artifacts(
        input_run_dir=args.input_run_dir,
        audio_path=args.audio_path,
        diarization_json=args.diarization_json,
    )
    run_dir = Path(args.output_run_dir or args.input_run_dir or audio_path.parent.parent).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Stage 02+03 input_run_dir: {args.input_run_dir}")
    logger.info(f"Stage 02+03 audio_path: {audio_path}")
    logger.info(f"Stage 02+03 diarization_json: {diarization_path}")
    logger.info(f"Stage 02+03 output_run_dir: {run_dir}")
    logger.info(
        "Device map: "
        f"panns={_device_name_from_index(args.panns_device_index)}, "
        f"demucs={_device_name_from_index(args.demucs_device_index)}, "
        f"srcorrnet={_device_name_from_index(args.srcorrnet_device_index)}"
    )

    audio = standardization(str(audio_path), cfg)
    audio_duration = len(audio["waveform"]) / audio["sample_rate"] if audio["sample_rate"] else 0.0
    segment_list, micro_overlap_candidates = load_stage1_diarization(diarization_path)
    writer = TraceRunWriter(run_dir, source_audio_path=str(audio_path), logger=logger)

    logger.info(f"Loaded {len(segment_list)} diarization segments from stage 01")
    logger.info(f"Loaded {len(micro_overlap_candidates)} micro overlap candidates from stage 01")
    logger.info("Stage 02: Background music detection/removal")
    music_start = time.time()
    panns_model = _load_panns_model(args, logger) if args.demucs else None
    demucs_model = _load_demucs_model(args, logger) if args.demucs else None
    audio, segment_demucs_flags = preprocess_segments_with_demucs(
        segment_list,
        audio,
        panns_model=panns_model,
        use_demucs=bool(args.demucs),
        demucs_model=demucs_model,
        padding=float(args.demucs_padding),
    )
    music_time = time.time() - music_start
    writer.write_music_clean(
        segment_list,
        segment_demucs_flags,
        audio,
        metadata={
            "enabled": bool(args.demucs),
            "demucs_applied_count": int(sum(bool(flag) for flag in segment_demucs_flags)),
            "total_segments": len(segment_list),
            "processing_time_seconds": music_time,
            "rt_factor": music_time / audio_duration if audio_duration > 0 else 0.0,
            "demucs_padding_seconds": float(args.demucs_padding),
            "demucs_model_name": args.demucs_model_name,
        },
    )

    logger.info("Stage 03: Overlap separation")
    separation_start = time.time()
    separator = _load_srcorrnet_separator(args, logger) if args.srcorrnet else None
    embedding_model = _load_embedding_model(args, cfg, logger) if args.srcorrnet else None
    overlap_pairs = detect_overlapping_segments(segment_list, float(args.overlap_threshold))
    if args.srcorrnet and separator is not None and embedding_model is not None:
        audio, segment_list = process_overlapping_segments_with_separation(
            segment_list,
            audio,
            overlap_threshold=float(args.overlap_threshold),
            separator=separator,
            embedding_model=embedding_model,
            device=_torch_device_from_index(args.srcorrnet_device_index),
        )
    else:
        logger.info("SR-CorrNet-SS overlap separation skipped")
    separation_time = time.time() - separation_start
    writer.write_overlap(
        segment_list,
        audio,
        metadata={
            "enabled": bool(args.srcorrnet),
            "separator_available": separator is not None,
            "embedding_model_available": embedding_model is not None,
            "processing_time_seconds": separation_time,
            "rt_factor": separation_time / audio_duration if audio_duration > 0 else 0.0,
            "overlap_threshold_seconds": float(args.overlap_threshold),
            "overlap_pair_count": len(overlap_pairs),
            "separated_segments": int(sum(bool(seg.get("is_separated")) or bool(seg.get("srcorrnet")) for seg in segment_list)),
        },
    )

    print("Stage 02+03 complete")
    print("run_dir =", run_dir)
    print("music_json =", run_dir / "02_music_clean" / "segment_flags.json")
    print("cleaned_audio =", run_dir / "02_music_clean" / "cleaned_audio.wav")
    print("overlap_json =", run_dir / "03_overlap" / "segments.json")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only Sommelier stage 02 music clean and stage 03 overlap separation.")
    parser.add_argument("--input_run_dir", type=str, default="", help="Existing run_full_* folder from stage 01.")
    parser.add_argument("--audio_path", type=str, default="", help="Override stage 01 audio path. Defaults to input_run_dir/00_input/full.wav.")
    parser.add_argument("--diarization_json", type=str, default="", help="Override stage 01 diarization JSON. Defaults to input_run_dir/01_diarization/diarization.json.")
    parser.add_argument("--output_run_dir", type=str, default="", help="Where to write 02_music_clean and 03_overlap. Defaults to input_run_dir.")
    parser.add_argument("--config_path", type=str, default="config.json", help="Pipeline config.json path.")
    parser.add_argument("--demucs", action=argparse.BooleanOptionalAction, default=True, help="Enable PANNs music detection and Demucs cleaning.")
    parser.add_argument("--srcorrnet", action=argparse.BooleanOptionalAction, default=True, help="Enable SR-CorrNet-SS overlap separation.")
    parser.add_argument("--overlap_threshold", type=float, default=1.0, help="Minimum overlap seconds to run SR-CorrNet-SS on a pair.")
    parser.add_argument("--demucs_padding", type=float, default=0.5, help="Seconds of context around each segment for music detection/cleaning.")
    parser.add_argument("--demucs_model_name", type=str, default="htdemucs", help="Demucs model name.")
    parser.add_argument("--panns_data_dir", type=str, default="", help="Folder containing Cnn14_mAP=0.431.pth.")
    parser.add_argument("--srcorrnet_path", type=str, default="", help="SR-CorrNet-SS repo folder.")
    parser.add_argument("--panns_device_index", type=int, default=0, help="Visible CUDA device index for PANNs. Use -1 for CPU.")
    parser.add_argument("--demucs_device_index", type=int, default=0, help="Visible CUDA device index for Demucs. Use -1 for CPU.")
    parser.add_argument("--srcorrnet_device_index", type=int, default=0, help="Visible CUDA device index for SR-CorrNet-SS/embedding. Use -1 for CPU.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_stage_music_overlap(args)


if __name__ == "__main__":
    main()
