import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"

sys.path.insert(0, str(PIPELINE_DIR))

from utils.speaker_linking import (  # noqa: E402
    build_weighted_reference_embedding,
    select_speaker_reference_candidates,
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


class SpeakerLinkingTests(unittest.TestCase):
    def test_select_speaker_reference_candidates_orders_quality_before_legacy_fallback(self):
        segments = _df([
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 1.0, 4.0),
            ("SPEAKER_00", 6.0, 9.0),
            ("SPEAKER_00", 10.0, 10.8),
            ("SPEAKER_00", 12.0, 15.0),
            ("SPEAKER_01", 13.0, 14.0),
        ])

        candidates = select_speaker_reference_candidates(
            segments,
            "SPEAKER_00",
            min_segment_duration=2.0,
        )

        self.assertEqual(candidates[0]["type"], "naturally_clean_long")
        self.assertEqual((candidates[0]["start"], candidates[0]["end"]), (6.0, 9.0))
        self.assertEqual(candidates[1]["type"], "naturally_clean_short")
        self.assertIn("pure_chunk_short", [item["type"] for item in candidates])
        overlap_candidates = [item for item in candidates if item["type"] == "original_overlap"]
        self.assertEqual(
            [(item["start"], item["end"]) for item in overlap_candidates],
            [(12.0, 15.0), (0.0, 5.0)],
        )
        self.assertLess(overlap_candidates[0]["overlap_ratio"], overlap_candidates[1]["overlap_ratio"])

    def test_build_weighted_reference_embedding_keeps_low_quality_samples_low_influence(self):
        candidates = [
            {"start": 0.0, "end": 3.0, "duration": 3.0, "type": "naturally_clean_long", "weight": 1.0},
            {"start": 4.0, "end": 5.0, "duration": 1.0, "type": "original_longest", "weight": 0.3},
        ]

        def embedding_fn(start, end):
            if start == 0.0:
                return np.asarray([1.0, 0.0])
            return np.asarray([0.0, 1.0])

        reference, report = build_weighted_reference_embedding(
            candidates,
            embedding_fn=embedding_fn,
            max_segments_per_speaker=2,
        )

        self.assertEqual(report["segment_count"], 2)
        self.assertAlmostEqual(report["quality_weight"], 0.65)
        self.assertGreater(float(reference[0]), float(reference[1]))
        self.assertEqual(report["segments"][1]["type"], "original_longest")


if __name__ == "__main__":
    unittest.main()
