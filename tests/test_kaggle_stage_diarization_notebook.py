import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KaggleStageDiarizationNotebookTests(unittest.TestCase):
    def test_notebook_runs_one_manual_audio_path_and_prints_results(self):
        notebook = json.loads((ROOT / "kaggle_notebooks" / "07_stage_diarization_only.ipynb").read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("AUDIO_INPUT_PATH =", joined)
        self.assertIn("--input_audio_path", joined)
        self.assertNotIn("--input_folder_path", joined)
        self.assertIn("diarization.json", joined)
        self.assertIn("pd.DataFrame", joined)
        self.assertIn("display(", joined)
        self.assertIn("speaker_summary", joined)
        self.assertNotIn("build_trace_html.py", joined)
        self.assertNotIn("compare_diarization_run_to_golden.py", joined)


if __name__ == "__main__":
    unittest.main()
