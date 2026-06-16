import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "kaggle_notebooks" / "08_stage_music_overlap_only.ipynb"
GENERATOR_PATH = ROOT / "tools" / "build_kaggle_stage_music_overlap_notebook.py"


class KaggleStageMusicOverlapNotebookTests(unittest.TestCase):
    def test_notebook_runs_only_stage_02_and_03_from_stage_01_artifacts(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn('REPO_URL = "https://github.com/tuanad121/sommelier.git"', joined)
        self.assertIn('BRANCH = "kaggle-gpu"', joined)
        self.assertIn("DIARIZATION_JSON_PATH =", joined)
        self.assertIn("AUDIO_WAV_PATH =", joined)
        self.assertIn("DIARIZATION_JSON =", joined)
        self.assertIn("FULL_AUDIO_PATH =", joined)
        self.assertIn("RUN_DIR = FULL_AUDIO_PATH.parents[1]", joined)
        self.assertIn("DIARIZATION_JSON.parents[1] != RUN_DIR", joined)
        self.assertIn("RUN_DEMUCS = True", joined)
        self.assertIn("RUN_CLEARVOICE_TSE = True", joined)
        self.assertIn("OVERLAP_THRESHOLD = 1.0", joined)
        self.assertIn('NUMPY_PACKAGE = "numpy==1.26.4"', joined)
        self.assertIn('"librosa==0.10.2.post1"', joined)
        self.assertIn('"soundfile==0.12.1"', joined)
        self.assertIn('"clearvoice==0.1.2"', joined)
        self.assertNotIn("librosa==0.11.0", joined)
        self.assertNotIn("soundfile==0.13.1", joined)
        self.assertNotIn("numpy==2.2.6", joined)
        self.assertIn("PANNS_DEVICE_INDEX = 0", joined)
        self.assertIn("DEMUCS_DEVICE_INDEX = 0", joined)
        self.assertIn("CLEARVOICE_DEVICE_INDEX = 0", joined)
        self.assertIn('"clearvoice==0.1.2"', joined)
        self.assertIn('"modelscope"', joined)
        self.assertNotIn('"pyannote.audio==3.3.2"', joined)
        self.assertNotIn('"lightning==2.4.0"', joined)
        self.assertNotIn('"pytorch-lightning==2.5.2"', joined)
        self.assertIn("hf_hub_download", joined)
        self.assertIn("run_stage_music_overlap_only.py", joined)
        self.assertIn("--audio_path", joined)
        self.assertIn("--diarization_json", joined)
        self.assertIn("--output_run_dir", joined)
        self.assertIn("--overlap_threshold", joined)
        self.assertIn("--demucs", joined)
        self.assertIn("--clearvoice_tse", joined)
        self.assertIn("--clearvoice_device_index", joined)
        self.assertNotIn("--sepreformer", joined)
        self.assertNotIn("SEPREFORMER", joined)
        self.assertNotIn("SepReformer", joined)
        self.assertIn("02_music_clean/segment_flags.json", joined)
        self.assertIn("03_overlap/segments.json", joined)
        self.assertIn("cleaned_audio = Audio(", joined)
        self.assertIn("segment_audio_df", joined)
        self.assertNotIn("run_stage_diarization_only.py", joined)
        self.assertNotIn("main_original_ASR_MoE.py", joined)
        self.assertNotIn("build_trace_html.py", joined)
        self.assertNotIn("compare_diarization_run_to_golden.py", joined)

        for idx, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"08_stage_music_overlap_only.ipynb:cell{idx}")

    def test_generator_builds_stage_music_overlap_notebook(self):
        generator = GENERATOR_PATH.read_text(encoding="utf-8")

        self.assertIn('OUT_PATH = Path("kaggle_notebooks/08_stage_music_overlap_only.ipynb")', generator)
        self.assertIn("run_stage_music_overlap_only.py", generator)
        self.assertIn("DIARIZATION_JSON_PATH", generator)
        self.assertIn("RUN_DIR = FULL_AUDIO_PATH.parents[1]", generator)
        self.assertIn("RUN_CLEARVOICE_TSE", generator)
        self.assertIn('"librosa==0.10.2.post1"', generator)
        self.assertIn('"soundfile==0.12.1"', generator)
        self.assertIn('"clearvoice==0.1.2"', generator)
        self.assertIn('NUMPY_PACKAGE = "numpy==1.26.4"', generator)
        self.assertNotIn("librosa==0.11.0", generator)
        self.assertNotIn("numpy==2.2.6", generator)
        self.assertNotIn('"pyannote.audio==3.3.2"', generator)
        self.assertNotIn('"lightning==2.4.0"', generator)
        self.assertIn("--clearvoice_tse", generator)
        self.assertNotIn("--sepreformer", generator)


if __name__ == "__main__":
    unittest.main()
