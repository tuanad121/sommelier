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
        self.assertIn("AUDIO_INPUT_PATH =", joined)
        self.assertIn("STAGE1_RUN_DIR =", joined)
        self.assertIn("DIARIZATION_JSON =", joined)
        self.assertIn("FULL_AUDIO_PATH =", joined)
        self.assertIn("RUN_DEMUCS = True", joined)
        self.assertIn("RUN_SEPREFORMER = True", joined)
        self.assertIn("OVERLAP_THRESHOLD = 1.0", joined)
        self.assertIn("PANNS_DEVICE_INDEX = 0", joined)
        self.assertIn("DEMUCS_DEVICE_INDEX = 0", joined)
        self.assertIn("SEPREFORMER_DEVICE_INDEX = 0", joined)
        self.assertIn("run_stage_music_overlap_only.py", joined)
        self.assertIn("--input_run_dir", joined)
        self.assertIn("--overlap_threshold", joined)
        self.assertIn("--demucs", joined)
        self.assertIn("--sepreformer", joined)
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
        self.assertIn("STAGE1_RUN_DIR", generator)
        self.assertIn("RUN_SEPREFORMER", generator)


if __name__ == "__main__":
    unittest.main()
