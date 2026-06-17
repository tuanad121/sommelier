import importlib.util
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
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
        self.assertIn("--metis_tse", script)
        self.assertIn("--metis_device_index", script)
        self.assertIn("--metis_repo_dir", script)
        self.assertIn("MetisTSESeparator", script)
        self.assertNotIn("ClearVoiceTSESeparator", script)
        self.assertNotIn("damo/speech_mossformer2_tse_16k", script)
        self.assertNotIn("--sepreformer", script)
        self.assertNotIn("sepreformer_device_index", script)
        self.assertIn("--overlap_threshold", script)
        self.assertIn("preprocess_segments_with_demucs", script)
        self.assertIn("process_overlapping_segments_with_separation", script)
        self.assertIn("TraceRunWriter", script)

    def test_missing_panns_import_error_names_install_package(self):
        module = load_stage_music_overlap_module()

        class DummyLogger:
            def info(self, *_args, **_kwargs):
                pass

        args = types.SimpleNamespace(panns_data_dir="", panns_device_index=-1)
        with self.assertRaises(ModuleNotFoundError) as raised:
            module._load_panns_model(args, DummyLogger())

        self.assertIn("panns_inference", str(raised.exception))
        self.assertIn("panns-inference", str(raised.exception))

    def test_tse_overlap_processing_does_not_require_embedding_model(self):
        sys.path.insert(0, str(PIPELINE_DIR))
        sys.modules.setdefault(
            "librosa",
            types.SimpleNamespace(resample=lambda audio, orig_sr, target_sr: np.asarray(audio, dtype=np.float32)),
        )
        from utils import diarization as diarization_utils
        from utils import separation as separation_utils

        class DummyLogger:
            def debug(self, *_args, **_kwargs):
                pass

            def info(self, *_args, **_kwargs):
                pass

            def warning(self, *_args, **_kwargs):
                pass

            def error(self, *_args, **_kwargs):
                pass

        class DummyTSESeparator:
            is_tse = True

            def __init__(self):
                self.calls = []

            def separate_target(self, mixed_audio, reference_audio, sample_rate):
                self.calls.append((len(mixed_audio), len(reference_audio), sample_rate))
                return np.asarray(mixed_audio, dtype=np.float32) * 0.5

        dummy_logger = DummyLogger()
        separation_utils.set_logger(dummy_logger)
        diarization_utils.set_logger(dummy_logger)

        sample_rate = 10
        waveform = np.ones(100, dtype=np.float32)
        audio = {"waveform": waveform, "sample_rate": sample_rate}
        segments = [
            {"index": "00000", "speaker": "SPEAKER_00", "start": 0.0, "end": 2.2},
            {"index": "00001", "speaker": "SPEAKER_01", "start": 2.4, "end": 4.7},
            {"index": "00002", "speaker": "SPEAKER_00", "start": 5.0, "end": 7.2},
            {"index": "00003", "speaker": "SPEAKER_01", "start": 6.0, "end": 8.0},
        ]
        separator = DummyTSESeparator()

        _audio, updated_segments = separation_utils.process_overlapping_segments_with_separation(
            segments,
            audio,
            overlap_threshold=0.5,
            separator=separator,
            embedding_model=None,
            device="cpu",
        )

        self.assertEqual(len(separator.calls), 2)
        self.assertTrue(updated_segments[2]["sepreformer"])
        self.assertTrue(updated_segments[3]["sepreformer"])
        self.assertEqual(len(updated_segments[2]["enhanced_audio"]), 22)
        self.assertEqual(len(updated_segments[3]["enhanced_audio"]), 20)

    def test_metis_import_installs_python312_pkgutil_compatibility(self):
        sys.path.insert(0, str(PIPELINE_DIR))
        sys.modules.setdefault(
            "librosa",
            types.SimpleNamespace(resample=lambda audio, orig_sr, target_sr: np.asarray(audio, dtype=np.float32)),
        )
        from utils import separation as separation_utils
        import importlib.machinery
        import pkgutil

        had_imp_importer = hasattr(pkgutil, "ImpImporter")
        original_imp_importer = getattr(pkgutil, "ImpImporter", None)
        if had_imp_importer:
            delattr(pkgutil, "ImpImporter")
        try:
            self.assertFalse(hasattr(pkgutil, "ImpImporter"))
            separation_utils._ensure_pkgutil_impimporter_compat()
            self.assertIs(pkgutil.ImpImporter, importlib.machinery.FileFinder)
            source = inspect.getsource(separation_utils.MetisTSESeparator.__init__)
            self.assertIn("_ensure_pkgutil_impimporter_compat()", source)
        finally:
            if had_imp_importer:
                pkgutil.ImpImporter = original_imp_importer
            elif hasattr(pkgutil, "ImpImporter"):
                delattr(pkgutil, "ImpImporter")

    def test_metis_repo_dir_must_be_full_amphion_checkout(self):
        sys.path.insert(0, str(PIPELINE_DIR))
        sys.modules.setdefault(
            "librosa",
            types.SimpleNamespace(resample=lambda audio, orig_sr, target_sr: np.asarray(audio, dtype=np.float32)),
        )
        from utils import separation as separation_utils

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "Amphion"
            (repo_dir / "models" / "tts" / "metis" / "ckpt").mkdir(parents=True)

            with self.assertRaises(FileNotFoundError) as raised:
                separation_utils._ensure_metis_repo_layout(repo_dir)

        message = str(raised.exception)
        self.assertIn("models/tts/metis/metis.py", message)
        self.assertIn("METIS_FORCE_RECLONE=True", message)

    def test_metis_import_path_is_forced_to_front(self):
        sys.path.insert(0, str(PIPELINE_DIR))
        sys.modules.setdefault(
            "librosa",
            types.SimpleNamespace(resample=lambda audio, orig_sr, target_sr: np.asarray(audio, dtype=np.float32)),
        )
        from utils import separation as separation_utils

        original_sys_path = sys.path.copy()
        try:
            sys.path[:] = ["/tmp/other_models", "/tmp/Amphion", "/tmp/another"]
            separation_utils._prepend_import_path(Path("/tmp/Amphion"))

            expected = str(Path("/tmp/Amphion").resolve())
            self.assertEqual(sys.path[0], expected)
            self.assertEqual(sys.path.count(expected), 1)
        finally:
            sys.path[:] = original_sys_path


if __name__ == "__main__":
    unittest.main()
