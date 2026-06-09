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


if __name__ == "__main__":
    unittest.main()
