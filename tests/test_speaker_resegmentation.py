import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"

import sys

sys.path.insert(0, str(PIPELINE_DIR))

from utils.speaker_resegmentation import (  # noqa: E402
    AuditConfig,
    LocalActivity,
    SlidingWindowAuditConfig,
    apply_resegmentation_audit,
    apply_sliding_window_audit,
    build_global_references,
    decode_frame_states,
    map_local_activities_to_global,
    write_resegmentation_report,
)


def _df(rows):
    return pd.DataFrame(
        [
            {
                "segment": f"seg{i}",
                "label": chr(ord("A") + i),
                "speaker": speaker,
                "start": start,
                "end": end,
            }
            for i, (speaker, start, end) in enumerate(rows)
        ]
    )


def _refs():
    return {
        "SPEAKER_00": np.asarray([1.0, 0.0], dtype=np.float32),
        "SPEAKER_01": np.asarray([0.0, 1.0], dtype=np.float32),
    }


class SpeakerResegmentationTests(unittest.TestCase):
    def test_build_global_references_uses_multiple_clean_segments_per_speaker(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 1.0),
            ("SPEAKER_00", 2.0, 5.0),
            ("SPEAKER_00", 6.0, 9.0),
            ("SPEAKER_01", 10.0, 13.0),
        ])

        def embedding_fn(start, end):
            if start < 10.0:
                return np.asarray([1.0, 0.0])
            return np.asarray([0.0, 1.0])

        references, report = build_global_references(
            segments,
            embedding_fn=embedding_fn,
            min_segment_duration=2.0,
            max_segments_per_speaker=4,
        )

        self.assertEqual(set(references), {"SPEAKER_00", "SPEAKER_01"})
        self.assertEqual(report["SPEAKER_00"]["segment_count"], 2)
        self.assertTrue(np.allclose(references["SPEAKER_00"], np.asarray([1.0, 0.0])))

    def test_decode_frame_states_marks_none_single_overlap_other_and_uncertain(self):
        mapped = [
            LocalActivity("local_a", 0.2, 0.5, np.asarray([1.0, 0.0])),
            LocalActivity("local_b", 0.4, 0.7, np.asarray([0.0, 1.0])),
            LocalActivity("local_c", 0.8, 0.9, np.asarray([-1.0, 0.0])),
        ]
        activities = map_local_activities_to_global(
            mapped,
            {
                "SPEAKER_00": np.asarray([1.0, 0.0]),
                "SPEAKER_01": np.asarray([0.0, 1.0]),
                "SPEAKER_02": np.asarray([-1.0, 0.0]),
            },
            AuditConfig(min_mapping_score=0.7),
        )
        states = decode_frame_states(
            0.0,
            1.0,
            activities,
            primary_speakers=("SPEAKER_00", "SPEAKER_01"),
            frame_step=0.1,
        )

        self.assertIn("NONE", states)
        self.assertIn("A_ONLY", states)
        self.assertIn("A_AND_B", states)
        self.assertIn("B_ONLY", states)
        self.assertIn("OTHER", states)

    def test_boundary_audit_shifts_boundary_when_local_activity_is_confident(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])

        def local_activity(region):
            return [
                LocalActivity("local_a", 0.0, 4.6, np.asarray([1.0, 0.0])),
                LocalActivity("local_b", 4.6, 10.0, np.asarray([0.0, 1.0])),
            ]

        refined, report = apply_resegmentation_audit(
            segments,
            references=_refs(),
            local_activity_provider=local_activity,
            config=AuditConfig(boundary_window=1.0, max_shift=0.8, min_mapping_score=0.7),
        )

        self.assertAlmostEqual(float(refined.loc[0, "end"]), 4.6)
        self.assertAlmostEqual(float(refined.loc[1, "start"]), 4.6)
        self.assertEqual(report["adjustments"][0]["action"], "shift")
        self.assertEqual(report["adjustments"][0]["confidence"], "high")

    def test_interior_audit_splits_turn_swallowed_inside_long_segment(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 8.0),
        ])

        def local_activity(region):
            return [
                LocalActivity("local_a", 0.0, 3.2, np.asarray([1.0, 0.0])),
                LocalActivity("local_b", 3.2, 5.0, np.asarray([0.0, 1.0])),
                LocalActivity("local_a", 5.0, 8.0, np.asarray([1.0, 0.0])),
            ]

        refined, report = apply_resegmentation_audit(
            segments,
            references=_refs(),
            local_activity_provider=local_activity,
            config=AuditConfig(interior_min_duration=6.0, min_mapping_score=0.7),
        )

        self.assertEqual(list(refined["speaker"]), ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
        self.assertAlmostEqual(float(refined.loc[1, "start"]), 3.2)
        self.assertAlmostEqual(float(refined.loc[1, "end"]), 5.0)
        self.assertEqual(report["adjustments"][0]["action"], "split")

    def test_overlap_activity_extends_adjacent_segments_to_mark_overlap(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])

        def local_activity(region):
            return [
                LocalActivity("local_a", 0.0, 5.5, np.asarray([1.0, 0.0])),
                LocalActivity("local_b", 4.5, 10.0, np.asarray([0.0, 1.0])),
            ]

        refined, report = apply_resegmentation_audit(
            segments,
            references=_refs(),
            local_activity_provider=local_activity,
            config=AuditConfig(boundary_window=1.0, max_extend=0.8, min_mapping_score=0.7),
        )

        self.assertAlmostEqual(float(refined.loc[0, "end"]), 5.5)
        self.assertAlmostEqual(float(refined.loc[1, "start"]), 4.5)
        self.assertEqual(report["adjustments"][0]["action"], "mark_overlap")
        self.assertEqual(report["adjustments"][0]["speakers"], ["SPEAKER_00", "SPEAKER_01"])

    def test_low_margin_local_activity_keeps_original_segments(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])
        ambiguous_refs = {
            "SPEAKER_00": np.asarray([1.0, 0.0], dtype=np.float32),
            "SPEAKER_01": np.asarray([0.99, 0.01], dtype=np.float32),
        }

        def local_activity(region):
            return [
                LocalActivity("local_a", 0.0, 4.4, np.asarray([1.0, 0.0])),
                LocalActivity("local_b", 4.4, 10.0, np.asarray([0.99, 0.01])),
            ]

        refined, report = apply_resegmentation_audit(
            segments,
            references=ambiguous_refs,
            local_activity_provider=local_activity,
            config=AuditConfig(boundary_window=1.0, min_mapping_score=0.7, min_mapping_margin=0.2),
        )

        self.assertEqual(refined[["speaker", "start", "end"]].to_dict("records"), segments[["speaker", "start", "end"]].to_dict("records"))
        self.assertEqual(report["adjustments"], [])
        self.assertGreaterEqual(len(report["skipped_regions"]), 1)
        self.assertEqual(report["skipped_regions"][0]["reason"], "uncertain_local_mapping")

    def test_sliding_window_audit_shifts_adjacent_boundary_from_embeddings(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])

        def embedding_fn(start, end):
            center = (float(start) + float(end)) / 2.0
            if center < 4.6:
                return np.asarray([1.0, 0.0])
            return np.asarray([0.0, 1.0])

        refined, report = apply_sliding_window_audit(
            segments,
            config=SlidingWindowAuditConfig(
                window_size=0.2,
                step_size=0.1,
                threshold_high=0.75,
                threshold_low=0.55,
                max_shift=0.8,
                min_duration=0.3,
            ),
            embedding_fn=embedding_fn,
            references=_refs(),
            min_segment_duration=1.0,
            max_segments_per_speaker=2,
        )

        self.assertAlmostEqual(float(refined.loc[0, "end"]), 4.6)
        self.assertAlmostEqual(float(refined.loc[1, "start"]), 4.6)
        self.assertEqual(report["adjustments"][0]["action"], "shift")
        self.assertEqual(report["adjustments"][0]["method"], "sliding_window")
        self.assertEqual(report["metadata"]["adjustment_count"], 1)

    def test_sliding_window_audit_pulls_head_of_right_segment_into_left_speaker(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])

        def embedding_fn(start, end):
            center = (float(start) + float(end)) / 2.0
            if center < 5.4:
                return np.asarray([1.0, 0.0])
            return np.asarray([0.0, 1.0])

        refined, report = apply_sliding_window_audit(
            segments,
            config=SlidingWindowAuditConfig(
                window_size=0.2,
                step_size=0.1,
                threshold_high=0.75,
                max_shift=0.8,
                min_duration=0.3,
            ),
            embedding_fn=embedding_fn,
            references=_refs(),
            min_segment_duration=1.0,
            max_segments_per_speaker=2,
        )

        self.assertAlmostEqual(float(refined.loc[0, "end"]), 5.4)
        self.assertAlmostEqual(float(refined.loc[1, "start"]), 5.4)
        self.assertEqual(report["adjustments"][0]["action"], "shift")
        self.assertEqual(report["adjustments"][0]["direction"], "right")

    def test_sliding_window_audit_marks_overlap_when_right_head_contains_both_speakers(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])

        def embedding_fn(start, end):
            center = (float(start) + float(end)) / 2.0
            if center < 5.0:
                return np.asarray([1.0, 0.0])
            if center < 5.4:
                return np.asarray([1.0, 1.0])
            return np.asarray([0.0, 1.0])

        refined, report = apply_sliding_window_audit(
            segments,
            config=SlidingWindowAuditConfig(
                window_size=0.2,
                step_size=0.1,
                threshold_high=0.65,
                max_shift=0.8,
                min_duration=0.3,
            ),
            embedding_fn=embedding_fn,
            min_segment_duration=1.0,
            max_segments_per_speaker=2,
        )

        self.assertAlmostEqual(float(refined.loc[0, "end"]), 5.3)
        self.assertAlmostEqual(float(refined.loc[1, "start"]), 5.0)
        self.assertEqual(report["adjustments"][0]["action"], "mark_overlap")
        self.assertEqual(report["adjustments"][0]["direction"], "right_overlap")

    def test_sliding_window_audit_marks_overlap_when_left_tail_contains_both_speakers(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])

        def embedding_fn(start, end):
            center = (float(start) + float(end)) / 2.0
            if center < 4.6:
                return np.asarray([1.0, 0.0])
            if center <= 5.0:
                return np.asarray([1.0, 1.0])
            return np.asarray([0.0, 1.0])

        refined, report = apply_sliding_window_audit(
            segments,
            config=SlidingWindowAuditConfig(
                window_size=0.2,
                step_size=0.1,
                threshold_high=0.65,
                max_shift=0.8,
                max_extend=0.8,
                min_duration=0.3,
            ),
            embedding_fn=embedding_fn,
            min_segment_duration=1.0,
            max_segments_per_speaker=2,
        )

        self.assertAlmostEqual(float(refined.loc[0, "end"]), 5.0)
        self.assertAlmostEqual(float(refined.loc[1, "start"]), 4.6)
        self.assertEqual(report["adjustments"][0]["action"], "mark_overlap")
        self.assertEqual(report["adjustments"][0]["direction"], "left_overlap")

    def test_sliding_window_audit_skips_conflicting_swapped_boundary_evidence(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])

        def embedding_fn(start, end):
            center = (float(start) + float(end)) / 2.0
            if center < 5.0:
                return np.asarray([0.0, 1.0])
            return np.asarray([1.0, 0.0])

        refined, report = apply_sliding_window_audit(
            segments,
            config=SlidingWindowAuditConfig(
                window_size=0.2,
                step_size=0.1,
                threshold_high=0.75,
                max_shift=0.8,
                min_duration=0.3,
            ),
            embedding_fn=embedding_fn,
            references=_refs(),
            min_segment_duration=1.0,
            max_segments_per_speaker=2,
        )

        self.assertEqual(refined[["speaker", "start", "end"]].to_dict("records"), segments[["speaker", "start", "end"]].to_dict("records"))
        self.assertEqual(report["adjustments"], [])
        self.assertEqual(report["skipped_regions"][0]["reason"], "conflicting_boundary_evidence")

    def test_sliding_window_audit_skips_when_windows_are_ambiguous(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ])

        def embedding_fn(start, end):
            return np.asarray([0.6, 0.6])

        refined, report = apply_sliding_window_audit(
            segments,
            config=SlidingWindowAuditConfig(
                window_size=0.2,
                step_size=0.1,
                threshold_high=0.95,
                threshold_low=0.55,
                max_shift=0.8,
                min_duration=0.3,
            ),
            embedding_fn=embedding_fn,
            min_segment_duration=1.0,
            max_segments_per_speaker=2,
        )

        self.assertEqual(refined[["speaker", "start", "end"]].to_dict("records"), segments[["speaker", "start", "end"]].to_dict("records"))
        self.assertEqual(report["adjustments"], [])
        self.assertEqual(report["skipped_regions"][0]["reason"], "no_confident_switch")

    def test_write_resegmentation_report_serializes_adjustments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_resegmentation_report(
                Path(tmp),
                {
                    "metadata": {"enabled": True},
                    "adjustments": [{"action": "shift", "new_boundary": 4.6}],
                    "skipped_regions": [],
                },
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "speaker_resegmentation_audit.json")
        self.assertEqual(payload["metadata"]["stage"], "speaker_resegmentation_audit")
        self.assertEqual(payload["adjustments"][0]["action"], "shift")


if __name__ == "__main__":
    unittest.main()
