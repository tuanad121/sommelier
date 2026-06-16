import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "podcast-pipeline" / "run_stage_music_overlap_only.py"


def load_stage_music_overlap_module():
    spec = importlib.util.spec_from_file_location("run_stage_music_overlap_only", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StageMusicOverlapOnlyTests(unittest.TestCase):
    def test_resolve_stage1_artifacts_defaults_to_run_dir_outputs(self):
        module = load_stage_music_overlap_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_full_audio"
            audio_path = run_dir / "00_input" / "full.wav"
            diar_path = run_dir / "01_diarization" / "diarization.json"
            audio_path.parent.mkdir(parents=True)
            diar_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"RIFF")
            diar_path.write_text(json.dumps({"segments": []}), encoding="utf-8")

            resolved_audio, resolved_diar = module.resolve_stage1_artifacts(
                input_run_dir=run_dir,
                audio_path=None,
                diarization_json=None,
            )

        self.assertEqual(resolved_audio, audio_path)
        self.assertEqual(resolved_diar, diar_path)

    def test_script_exposes_stage_02_03_controls(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("--input_run_dir", script)
        self.assertIn("--diarization_json", script)
        self.assertIn("--audio_path", script)
        self.assertIn("--demucs", script)
        self.assertIn("--clearvoice_tse", script)
        self.assertIn("--clearvoice_device_index", script)
        self.assertIn("ClearVoiceTSESeparator", script)
        self.assertNotIn("--sepreformer", script)
        self.assertNotIn("sepreformer_device_index", script)
        self.assertIn("--overlap_threshold", script)
        self.assertIn("preprocess_segments_with_demucs", script)
        self.assertIn("process_overlapping_segments_with_separation", script)
        self.assertIn("TraceRunWriter", script)


if __name__ == "__main__":
    unittest.main()
