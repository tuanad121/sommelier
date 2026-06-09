from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_GOLDEN_CSV = Path("datasets/diarization_eval/diarization_eval_dataset.csv")
DEFAULT_MANIFEST = Path("datasets/diarization_eval/golden_manifest.json")


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_manifest_entry(run_name: str, manifest_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = run_name.replace("run_full_", "", 1)
    exact = [row for row in manifest_rows if row["audio_stem"] == target or row["run_name"] == run_name]
    if exact:
        return exact[0]
    contains = [row for row in manifest_rows if target in row["audio_stem"] or row["audio_stem"] in target]
    return contains[0] if contains else None


def load_run_segments(run_dir: Path) -> list[dict[str, Any]]:
    diar_path = run_dir / "01_diarization" / "diarization.json"
    payload = load_json(diar_path)
    return payload.get("segments", [])


def compare_segments(run_segments: list[dict[str, Any]], golden_segments: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    matched_count = min(len(run_segments), len(golden_segments))
    start_matches = 0
    end_matches = 0
    speaker_matches = 0
    exact_triplet_matches = 0
    mismatches: list[dict[str, Any]] = []

    for idx in range(matched_count):
        pred = run_segments[idx]
        gold = golden_segments[idx]
        start_ok = abs(float(pred.get("start", 0.0)) - float(gold.get("start", 0.0))) <= tolerance
        end_ok = abs(float(pred.get("end", 0.0)) - float(gold.get("end", 0.0))) <= tolerance
        speaker_ok = str(pred.get("speaker", "")) == str(gold.get("speaker", ""))
        start_matches += int(start_ok)
        end_matches += int(end_ok)
        speaker_matches += int(speaker_ok)
        exact_triplet_matches += int(start_ok and end_ok and speaker_ok)
        if not (start_ok and end_ok and speaker_ok) and len(mismatches) < 20:
            mismatches.append(
                {
                    "index": idx,
                    "pred": {
                        "start": pred.get("start"),
                        "end": pred.get("end"),
                        "speaker": pred.get("speaker"),
                    },
                    "gold": {
                        "start": gold.get("start"),
                        "end": gold.get("end"),
                        "speaker": gold.get("speaker"),
                    },
                }
            )

    return {
        "pred_segment_count": len(run_segments),
        "gold_segment_count": len(golden_segments),
        "matched_count": matched_count,
        "start_match_rate": start_matches / matched_count if matched_count else 0.0,
        "end_match_rate": end_matches / matched_count if matched_count else 0.0,
        "speaker_match_rate": speaker_matches / matched_count if matched_count else 0.0,
        "exact_triplet_match_rate": exact_triplet_matches / matched_count if matched_count else 0.0,
        "mismatch_examples": mismatches,
    }


def render_markdown(run_name: str, entry: dict[str, Any], result: dict[str, Any], tolerance: float) -> str:
    lines = [
        f"# Diarization vs Golden Baseline",
        "",
        f"- Run: `{run_name}`",
        f"- Golden source: `{entry['run_name']}`",
        f"- Tolerance: **{tolerance:.2f}s**",
        f"- Pred segments: **{result['pred_segment_count']}**",
        f"- Gold segments: **{result['gold_segment_count']}**",
        f"- Compared rows: **{result['matched_count']}**",
        "",
        "## Match rates",
        "",
        f"- Start: **{result['start_match_rate']:.2%}**",
        f"- End: **{result['end_match_rate']:.2%}**",
        f"- Speaker: **{result['speaker_match_rate']:.2%}**",
        f"- Exact triplet (`start+end+speaker`): **{result['exact_triplet_match_rate']:.2%}**",
    ]
    if result["mismatch_examples"]:
        lines.extend(["", "## Mismatch examples", ""])
        for item in result["mismatch_examples"]:
            lines.append(
                f"- idx {item['index']}: pred=({item['pred']['start']}, {item['pred']['end']}, {item['pred']['speaker']}) "
                f"gold=({item['gold']['start']}, {item['gold']['end']}, {item['gold']['speaker']})"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a diarization-only run against the podcast golden baseline.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to one diarization-only run_full directory.")
    parser.add_argument("--golden-csv", type=Path, default=DEFAULT_GOLDEN_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tolerance", type=float, default=0.2, help="Allowed absolute timing delta in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = load_json(args.manifest)
    golden_rows = load_csv(args.golden_csv)

    entry = find_manifest_entry(args.run_dir.name, manifest_rows)
    if entry is None:
        raise SystemExit(f"No golden manifest entry found for run dir: {args.run_dir.name}")

    start_idx = int(entry["row_start_index"])
    end_idx = int(entry["row_end_index"])
    golden_slice = golden_rows[start_idx : end_idx + 1]
    run_segments = load_run_segments(args.run_dir)
    result = compare_segments(run_segments, golden_slice, args.tolerance)
    result["run_name"] = args.run_dir.name
    result["golden_source_run"] = entry["run_name"]
    result["tolerance_seconds"] = args.tolerance

    out_json = args.run_dir / "01_diarization" / "golden_compare.json"
    out_md = args.run_dir / "01_diarization" / "golden_compare.md"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(args.run_dir.name, entry, result, args.tolerance), encoding="utf-8")

    print(json.dumps({
        "run_name": result["run_name"],
        "golden_source_run": result["golden_source_run"],
        "pred_segment_count": result["pred_segment_count"],
        "gold_segment_count": result["gold_segment_count"],
        "exact_triplet_match_rate": result["exact_triplet_match_rate"],
        "speaker_match_rate": result["speaker_match_rate"],
        "report_json": str(out_json),
        "report_md": str(out_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
