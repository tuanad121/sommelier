from __future__ import annotations

import json
import re
from pathlib import Path


SOURCE_PATH = Path("kaggle_notebooks/stage1_updated.ipynb")
OUT_PATH = Path("kaggle_notebooks/07_stage_diarization_only.ipynb")


SYNC_ASSIGNMENT_NAMES = (
    "DIAR_DEVICE_INDEX",
    "SORTFORMER_DEVICE_INDEX",
    "SPEAKER_LINK_THRESHOLD",
    "MIN_SPLIT_SILENCE",
    "MAX_DIA_CHUNK_DURATION",
    "AUDIO_GAIN_CLAMP_DB",
    "SORTFORMER_MODEL_NAME",
    "SORTFORMER_BATCH_SIZE",
    "SORTFORMER_NUM_WORKERS",
    "SORTFORMER_STREAMING_CONFIG",
    "SORTFORMER_CHUNK_LEN",
    "SORTFORMER_CHUNK_LEFT_CONTEXT",
    "SORTFORMER_CHUNK_RIGHT_CONTEXT",
    "SORTFORMER_FIFO_LEN",
    "SORTFORMER_SPKCACHE_UPDATE_PERIOD",
    "SORTFORMER_SPKCACHE_LEN",
    "SORTFORMER_POSTPROCESSING",
    "SORTFORMER_POSTPROCESSING_YAML",
    "SORTFORMER_PP_ONSET",
    "SORTFORMER_PP_OFFSET",
    "SORTFORMER_PP_PAD_ONSET",
    "SORTFORMER_PP_PAD_OFFSET",
    "SORTFORMER_PP_MIN_DURATION_ON",
    "SORTFORMER_PP_MIN_DURATION_OFF",
    "SORTFORMER_PRESERVE_MICRO_OVERLAPS",
    "SORTFORMER_MICRO_OVERLAP_MIN_DURATION_ON",
    "MICRO_OVERLAP_MIN_DURATION",
    "TORCH_PACKAGE",
    "TORCHAUDIO_PACKAGE",
    "TORCHVISION_PACKAGE",
    "PYTORCH_WHEEL_EXTRA_INDEX_URL",
    "LIGHTNING_PACKAGE",
    "PYTORCH_LIGHTNING_PACKAGE",
    "NEMO_PACKAGE",
    "NUMPY_PACKAGE",
    "NUMBA_PACKAGE",
    "LLVMLITE_PACKAGE",
    "TORCHMETRICS_PACKAGE",
    "PILLOW_PACKAGE",
    "CTRANSLATE2_PACKAGE",
    "SPEAKER_EMBEDDING_MODEL_NAME",
    "SILERO_VAD_REPO",
)

DEFAULT_ASSIGNMENTS = {
    "DIAR_DEVICE_INDEX": "0",
    "SORTFORMER_DEVICE_INDEX": "0",
    "SPEAKER_LINK_THRESHOLD": "0.75",
    "MIN_SPLIT_SILENCE": "1.0",
    "MAX_DIA_CHUNK_DURATION": "900.0",
    "AUDIO_GAIN_CLAMP_DB": "6.0",
    "SORTFORMER_MODEL_NAME": '"nvidia/diar_streaming_sortformer_4spk-v2.1"',
    "SORTFORMER_BATCH_SIZE": "1",
    "SORTFORMER_NUM_WORKERS": "0",
    "SORTFORMER_STREAMING_CONFIG": "True",
    "SORTFORMER_CHUNK_LEN": "340",
    "SORTFORMER_CHUNK_LEFT_CONTEXT": "1",
    "SORTFORMER_CHUNK_RIGHT_CONTEXT": "40",
    "SORTFORMER_FIFO_LEN": "40",
    "SORTFORMER_SPKCACHE_UPDATE_PERIOD": "300",
    "SORTFORMER_SPKCACHE_LEN": "200",
    "SORTFORMER_POSTPROCESSING": "True",
    "SORTFORMER_POSTPROCESSING_YAML": "OUTPUT_ROOT / 'sortformer_postprocessing.yaml'",
    "SORTFORMER_PP_ONSET": "0.3",
    "SORTFORMER_PP_OFFSET": "0.33",
    "SORTFORMER_PP_PAD_ONSET": "0.02",
    "SORTFORMER_PP_PAD_OFFSET": "-0.13",
    "SORTFORMER_PP_MIN_DURATION_ON": "0.28",
    "SORTFORMER_PP_MIN_DURATION_OFF": "0.4",
    "SORTFORMER_PRESERVE_MICRO_OVERLAPS": "True",
    "SORTFORMER_MICRO_OVERLAP_MIN_DURATION_ON": "0.0",
    "MICRO_OVERLAP_MIN_DURATION": "0.0",
}

ASSIGNMENT_RE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>.+?)\s*$"
)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def extract_assignments(notebook_path: Path) -> dict[str, str]:
    if not notebook_path.exists():
        return {}

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assignments: dict[str, str] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in _cell_source(cell).splitlines():
            match = ASSIGNMENT_RE.match(line)
            if not match:
                continue
            name = match.group("name")
            if name in SYNC_ASSIGNMENT_NAMES:
                assignments[name] = match.group("value").strip()
    return assignments


def sync_assignments(notebook: dict, assignments: dict[str, str]) -> int:
    changed = 0
    names = set(SYNC_ASSIGNMENT_NAMES)
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        if "SORTFORMER_MODEL_NAME" not in source:
            continue

        new_lines: list[str] = []
        for line in source.splitlines():
            match = ASSIGNMENT_RE.match(line)
            if match and match.group("name") in names and match.group("name") in assignments:
                name = match.group("name")
                replacement = f"{match.group('indent')}{name} = {assignments[name]}"
                if replacement != line:
                    changed += 1
                new_lines.append(replacement + "\n")
            else:
                new_lines.append(line + "\n")
        cell["source"] = new_lines
        break
    return changed


def clear_outputs(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def main() -> None:
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"Missing notebook template: {OUT_PATH}")

    notebook = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    assignments = dict(DEFAULT_ASSIGNMENTS)
    assignments.update(extract_assignments(SOURCE_PATH))
    changed = sync_assignments(notebook, assignments)
    clear_outputs(notebook)

    OUT_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_PATH} ({changed} assignment lines updated)")


if __name__ == "__main__":
    main()
