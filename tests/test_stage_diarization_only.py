import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"

import sys

sys.path.insert(0, str(PIPELINE_DIR))


class StageDiarizationOnlyTests(unittest.TestCase):
    def test_collect_audio_paths_prefers_single_audio_and_filters_folder(self):
        from utils.stage_diarization import collect_audio_paths

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            keep_wav = folder / "a.wav"
            keep_mp3 = folder / "b.mp3"
            skip_txt = folder / "notes.txt"
            keep_wav.write_bytes(b"wav")
            keep_mp3.write_bytes(b"mp3")
            skip_txt.write_text("not audio", encoding="utf-8")

            args = Namespace(input_audio_path=str(keep_wav), input_folder_path=str(folder))
            self.assertEqual(collect_audio_paths(args, {"entrypoint": {}}), [str(keep_wav)])

            args = Namespace(input_audio_path="", input_folder_path=str(folder))
            self.assertEqual(
                collect_audio_paths(args, {"entrypoint": {}}),
                [str(keep_wav), str(keep_mp3)],
            )

    def test_resolve_config_path_works_from_repo_root_or_pipeline_dir(self):
        from utils.stage_diarization import resolve_config_path

        repo_root = Path(tempfile.mkdtemp())
        script_dir = repo_root / "podcast-pipeline"
        script_dir.mkdir()
        config = script_dir / "config.json"
        config.write_text("{}", encoding="utf-8")

        self.assertEqual(
            resolve_config_path("podcast-pipeline/config.json", cwd=repo_root, script_dir=script_dir),
            config,
        )
        self.assertEqual(
            resolve_config_path("config.json", cwd=script_dir, script_dir=script_dir),
            config,
        )

    def test_write_input_artifacts_records_stage_00_inputs_for_diarization(self):
        from utils.stage_diarization import write_input_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_full_clip"
            audio = {"waveform": [0.0, 0.1], "sample_rate": 16000}
            chunks = [{"path": "/tmp/chunk.wav", "offset": 0.0, "duration": 1.0}]
            written_audio_paths = []

            def fake_write_audio(path, payload):
                written_audio_paths.append((path, payload))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"wav")

            write_input_artifacts(run_dir, audio, chunks, write_audio=fake_write_audio)

            self.assertEqual(written_audio_paths[0][0], run_dir / "00_input" / "full.wav")
            payload = json.loads((run_dir / "01_diarization" / "vad_chunks.json").read_text())
            self.assertEqual(payload["audio_path"], "00_input/full.wav")
            self.assertEqual(payload["chunks"], chunks)
            self.assertEqual(payload["metadata"]["stage"], "speaker_diarization")

    def test_write_sortformer_postprocessing_yaml_uses_official_nemo_keys(self):
        from utils.stage_diarization import write_sortformer_postprocessing_yaml

        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                sortformer_pp_onset=0.64,
                sortformer_pp_offset=0.74,
                sortformer_pp_pad_onset=0.06,
                sortformer_pp_pad_offset=0.0,
                sortformer_pp_min_duration_on=0.1,
                sortformer_pp_min_duration_off=0.15,
            )

            yaml_path = write_sortformer_postprocessing_yaml(Path(tmp) / "sortformer_pp.yaml", args)
            text = yaml_path.read_text(encoding="utf-8")

            self.assertIn("parameters:", text)
            self.assertIn("  onset: 0.64", text)
            self.assertIn("  offset: 0.74", text)
            self.assertIn("  pad_onset: 0.06", text)
            self.assertIn("  pad_offset: 0.0", text)
            self.assertIn("  min_duration_on: 0.1", text)
            self.assertIn("  min_duration_off: 0.15", text)

    def test_refine_speaker_boundaries_moves_boundary_toward_speaker_change(self):
        import numpy as np
        import pandas as pd

        from utils.stage_diarization import refine_speaker_boundaries

        diarization = pd.DataFrame(
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.2},
                {"speaker": "SPEAKER_01", "start": 1.2, "end": 2.2},
            ]
        )

        def fake_embedding(start: float, end: float):
            if end <= 1.0:
                return np.array([1.0, 0.0])
            if start >= 1.0:
                return np.array([0.0, 1.0])
            return None

        refined, adjustments = refine_speaker_boundaries(
            diarization,
            embedding_fn=fake_embedding,
            max_shift=0.4,
            step=0.1,
            embedding_window=0.2,
            min_segment=0.4,
            max_gap=0.5,
            min_improvement=0.1,
        )

        self.assertEqual(round(float(refined.loc[0, "end"]), 3), 1.0)
        self.assertEqual(round(float(refined.loc[1, "start"]), 3), 1.0)
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0]["left_speaker"], "SPEAKER_00")
        self.assertEqual(adjustments[0]["right_speaker"], "SPEAKER_01")

    def test_refine_speaker_boundaries_uses_separate_reference_min_segment(self):
        import numpy as np
        import pandas as pd

        from utils.stage_diarization import refine_speaker_boundaries

        diarization = pd.DataFrame(
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
                {"speaker": "SPEAKER_01", "start": 4.0, "end": 7.0},
                {"speaker": "SPEAKER_00", "start": 10.0, "end": 11.2},
                {"speaker": "SPEAKER_01", "start": 11.2, "end": 11.7},
            ]
        )

        def fake_embedding(start: float, end: float):
            if 0.0 <= start and end <= 3.0:
                return np.array([1.0, 0.0])
            if 4.0 <= start and end <= 7.0:
                return np.array([0.0, 1.0])
            if end <= 11.0:
                return np.array([1.0, 0.0])
            if start >= 11.0:
                return np.array([0.0, 1.0])
            return None

        refined, adjustments = refine_speaker_boundaries(
            diarization,
            embedding_fn=fake_embedding,
            max_shift=0.3,
            step=0.1,
            embedding_window=0.2,
            min_segment=0.25,
            reference_min_segment=2.0,
            max_gap=0.5,
            min_improvement=0.1,
        )

        self.assertEqual(round(float(refined.loc[2, "end"]), 3), 11.0)
        self.assertEqual(round(float(refined.loc[3, "start"]), 3), 11.0)
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0]["old_boundary"], 11.2)
        self.assertEqual(adjustments[0]["new_boundary"], 11.0)

    def test_refine_speaker_boundaries_trims_nested_short_segment_without_splitting_container(self):
        import numpy as np
        import pandas as pd

        from utils.stage_diarization import refine_speaker_boundaries

        diarization = pd.DataFrame(
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
                {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0},
                {"speaker": "SPEAKER_00", "start": 12.0, "end": 12.6},
            ]
        )

        def fake_embedding(start: float, end: float):
            if 0.0 <= start and end <= 3.0:
                return np.array([1.0, 0.0])
            if 12.0 <= start and end <= 12.4:
                return np.array([1.0, 0.0])
            if 10.0 <= start and end <= 20.0 and (end <= 12.0 or start >= 12.4):
                return np.array([0.0, 1.0])
            return None

        refined, adjustments = refine_speaker_boundaries(
            diarization,
            embedding_fn=fake_embedding,
            max_shift=0.3,
            step=0.1,
            embedding_window=0.2,
            min_segment=0.3,
            reference_min_segment=2.0,
            nested_max_segment=1.0,
            max_gap=0.5,
            min_improvement=0.1,
        )

        container = refined[(refined["speaker"] == "SPEAKER_01")].iloc[0]
        short = refined[(refined["speaker"] == "SPEAKER_00") & (refined["start"] > 10.0)].iloc[0]

        self.assertEqual(round(float(container["start"]), 3), 10.0)
        self.assertEqual(round(float(container["end"]), 3), 20.0)
        self.assertEqual(round(float(short["start"]), 3), 12.0)
        self.assertEqual(round(float(short["end"]), 3), 12.4)
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0]["type"], "nested_short_segment_trim")
        self.assertEqual(adjustments[0]["short_speaker"], "SPEAKER_00")
        self.assertEqual(adjustments[0]["container_speaker"], "SPEAKER_01")

    def test_stage_script_exposes_official_sortformer_postprocessing_flags(self):
        text = (PIPELINE_DIR / "run_stage_diarization_only.py").read_text(encoding="utf-8")

        for flag in [
            "--audio-gain-clamp-db",
            "--speaker-boundary-refinement",
            "--boundary-refine-max-shift",
            "--boundary-refine-step",
            "--boundary-refine-embed-window",
            "--boundary-refine-reference-min-segment",
            "--boundary-refine-nested-max-segment",
            "--boundary-refine-min-segment",
            "--boundary-refine-max-gap",
            "--boundary-refine-min-improvement",
            "--sortformer-postprocessing",
            "--sortformer-postprocessing-yaml",
            "--sortformer-pp-onset",
            "--sortformer-pp-offset",
            "--sortformer-pp-pad-onset",
            "--sortformer-pp-pad-offset",
            "--sortformer-pp-min-duration-on",
            "--sortformer-pp-min-duration-off",
            "--sortformer_batch_size",
            "--sortformer_num_workers",
        ]:
            self.assertIn(flag, text)

        self.assertIn("postprocessing_yaml=sortformer_postprocessing_yaml", text)
        self.assertIn("num_workers=int(args.sortformer_num_workers)", text)
        self.assertIn("batch_size=int(args.sortformer_batch_size)", text)
        self.assertIn('parser.add_argument("--audio-gain-clamp-db", type=float, default=6.0', text)
        self.assertIn("refine_speaker_boundaries(", text)
        self.assertIn("write_boundary_refinement_report(", text)
        self.assertIn('"speaker_boundary_refinement_enabled": bool(args.speaker_boundary_refinement)', text)
        self.assertIn('"speaker_boundary_refinement_count": len(boundary_refinements)', text)
        self.assertIn('parser.add_argument("--speaker-boundary-refinement", action=argparse.BooleanOptionalAction, default=False', text)
        self.assertIn('parser.add_argument("--boundary-refine-reference-min-segment", type=float, default=2.0', text)
        self.assertIn('parser.add_argument("--boundary-refine-nested-max-segment", type=float, default=1.0', text)
        self.assertIn('parser.add_argument("--boundary-refine-min-segment", type=float, default=0.3', text)
        self.assertIn('cfg.setdefault("entrypoint", {})["AUDIO_GAIN_CLAMP_DB"] = float(args.audio_gain_clamp_db)', text)
        self.assertIn('"audio_gain_clamp_db": float(args.audio_gain_clamp_db)', text)
        self.assertIn('parser.add_argument("--sortformer-postprocessing", action=argparse.BooleanOptionalAction, default=True', text)
        self.assertIn('parser.add_argument("--sortformer-pp-onset", type=float, default=0.3', text)
        self.assertIn('parser.add_argument("--sortformer-pp-offset", type=float, default=0.33', text)
        self.assertIn('parser.add_argument("--sortformer-pp-pad-onset", type=float, default=0.015', text)
        self.assertIn('parser.add_argument("--sortformer-pp-pad-offset", type=float, default=0.015', text)
        self.assertIn('parser.add_argument("--sortformer-pp-min-duration-on", type=float, default=0.35', text)
        self.assertIn('parser.add_argument("--sortformer-pp-min-duration-off", type=float, default=0.35', text)

    def test_apply_sortformer_streaming_config_sets_v21_cache_parameters(self):
        from utils.stage_diarization import apply_sortformer_streaming_config

        class DummyModules:
            pass

        class DummyModel:
            def __init__(self):
                self.sortformer_modules = DummyModules()

        args = Namespace(
            sortformer_streaming_config=True,
            sortformer_chunk_len=340,
            sortformer_chunk_left_context=1,
            sortformer_chunk_right_context=40,
            sortformer_fifo_len=40,
            sortformer_spkcache_update_period=300,
            sortformer_spkcache_len=200,
        )

        model = DummyModel()
        applied = apply_sortformer_streaming_config(model, args)

        self.assertEqual(
            applied,
            {
                "chunk_len": 340,
                "chunk_left_context": 1,
                "chunk_right_context": 40,
                "fifo_len": 40,
                "spkcache_update_period": 300,
                "spkcache_len": 200,
            },
        )
        self.assertEqual(model.sortformer_modules.chunk_len, 340)
        self.assertEqual(model.sortformer_modules.chunk_right_context, 40)
        self.assertEqual(model.sortformer_modules.spkcache_update_period, 300)

    def test_stage_script_exposes_streaming_sortformer_v21_flags(self):
        text = (PIPELINE_DIR / "run_stage_diarization_only.py").read_text(encoding="utf-8")

        for flag in [
            "--sortformer_model_name",
            "--sortformer-streaming-config",
            "--sortformer_chunk_len",
            "--sortformer_chunk_left_context",
            "--sortformer_chunk_right_context",
            "--sortformer_fifo_len",
            "--sortformer_spkcache_update_period",
            "--sortformer_spkcache_len",
        ]:
            self.assertIn(flag, text)

        self.assertIn("SortformerEncLabelModel.from_pretrained(args.sortformer_model_name)", text)
        self.assertIn("apply_sortformer_streaming_config(diar_model, args, logger=logger)", text)
        self.assertIn('default="nvidia/diar_streaming_sortformer_4spk-v2.1"', text)
        self.assertIn('parser.add_argument("--sortformer-streaming-config", action=argparse.BooleanOptionalAction, default=True', text)
        self.assertIn('parser.add_argument("--sortformer_spkcache_len", type=int, default=200', text)
        self.assertIn('parser.add_argument("--max_dia_chunk_duration", type=float, default=900.0', text)


if __name__ == "__main__":
    unittest.main()
