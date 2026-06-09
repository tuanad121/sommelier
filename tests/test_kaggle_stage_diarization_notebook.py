import json
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KaggleStageDiarizationNotebookTests(unittest.TestCase):
    def test_notebook_runs_one_manual_audio_path_and_prints_results(self):
        notebook = json.loads((ROOT / "kaggle_notebooks" / "07_stage_diarization_only.ipynb").read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("AUDIO_INPUT_PATH =", joined)
        self.assertIn('REPO_URL = "https://github.com/tuanad121/sommelier.git"', joined)
        self.assertIn('BRANCH = "kaggle-gpu"', joined)
        self.assertIn("git", joined)
        self.assertIn("clone", joined)
        self.assertIn("TORCH_PACKAGE = \"torch==2.7.1\"", joined)
        self.assertIn("TORCHAUDIO_PACKAGE = \"torchaudio==2.7.1\"", joined)
        self.assertIn("TORCHVISION_PACKAGE = \"torchvision==0.22.1\"", joined)
        self.assertIn("NEMO_PACKAGE = \"nemo-toolkit[asr]==2.4.0\"", joined)
        self.assertIn("PYTORCH_WHEEL_EXTRA_INDEX_URL", joined)
        self.assertIn("requirements-kaggle-diarization.txt", joined)
        self.assertIn("HF_SECRET_NAME = \"HF_TOKEN\"", joined)
        self.assertIn("UserSecretsClient", joined)
        self.assertIn("huggingface_token", joined)
        self.assertIn("nvidia/diar_sortformer_4spk-v1", joined)
        self.assertIn("pyannote/embedding", joined)
        self.assertIn("--input_audio_path", joined)
        self.assertNotIn("--input_folder_path", joined)
        self.assertIn("diarization.json", joined)
        self.assertIn("pd.DataFrame", joined)
        self.assertIn("display(", joined)
        self.assertIn("speaker_summary", joined)
        self.assertNotIn("build_trace_html.py", joined)
        self.assertNotIn("compare_diarization_run_to_golden.py", joined)

        for idx, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"07_stage_diarization_only.ipynb:cell{idx}")


if __name__ == "__main__":
    unittest.main()
