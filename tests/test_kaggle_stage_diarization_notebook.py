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
        self.assertIn("REQUIREMENTS_PATH = PIPELINE_DIR / 'requirements-kaggle-diarization.txt'", joined)
        self.assertIn("str(REQUIREMENTS_PATH)", joined)
        self.assertIn("HF_SECRET_NAME = \"HF_TOKEN\"", joined)
        self.assertIn("UserSecretsClient", joined)
        self.assertIn("huggingface_token", joined)
        self.assertIn("nvidia/diar_sortformer_4spk-v1", joined)
        self.assertIn("pyannote/embedding", joined)
        self.assertIn("SORTFORMER_POSTPROCESSING = True", joined)
        self.assertIn("SORTFORMER_POSTPROCESSING_YAML = OUTPUT_ROOT / 'sortformer_postprocessing.yaml'", joined)
        self.assertIn("SORTFORMER_PP_ONSET = 0.64", joined)
        self.assertIn("SORTFORMER_PP_OFFSET = 0.74", joined)
        self.assertIn("SORTFORMER_PP_PAD_ONSET = 0.06", joined)
        self.assertIn("SORTFORMER_PP_PAD_OFFSET = 0.0", joined)
        self.assertIn("SORTFORMER_PP_MIN_DURATION_ON = 0.1", joined)
        self.assertIn("SORTFORMER_PP_MIN_DURATION_OFF = 0.15", joined)
        self.assertIn("SORTFORMER_BATCH_SIZE = 1", joined)
        self.assertIn("SORTFORMER_NUM_WORKERS = 0", joined)
        self.assertIn("sortformer_pp = {", joined)
        self.assertIn("'parameters': sortformer_pp", joined)
        self.assertIn("--sortformer-postprocessing", joined)
        self.assertIn("--sortformer-postprocessing-yaml", joined)
        self.assertIn("--sortformer-pp-onset", joined)
        self.assertIn("--sortformer-pp-offset", joined)
        self.assertIn("--sortformer-pp-pad-onset", joined)
        self.assertIn("--sortformer-pp-pad-offset", joined)
        self.assertIn("--sortformer-pp-min-duration-on", joined)
        self.assertIn("--sortformer-pp-min-duration-off", joined)
        self.assertIn("--sortformer_batch_size", joined)
        self.assertIn("--sortformer_num_workers", joined)
        self.assertIn("--input_audio_path", joined)
        self.assertNotIn("--input_folder_path", joined)
        self.assertIn("diarization.json", joined)
        self.assertIn("pd.DataFrame", joined)
        self.assertIn("display(", joined)
        self.assertIn("speaker_summary", joined)
        self.assertIn("IPython.display", joined)
        self.assertIn("Audio(", joined)
        self.assertIn("FULL_AUDIO_PATH = RUN_DIR / '00_input' / 'full.wav'", joined)
        self.assertIn("SEGMENT_AUDIO_DIR = RUN_DIR / '01_diarization' / 'segment_audio_preview'", joined)
        self.assertIn("full_audio = Audio(str(FULL_AUDIO_PATH))", joined)
        self.assertIn("segment_audio", joined)
        self.assertIn("display(segment_audio_df", joined)
        self.assertNotIn("build_trace_html.py", joined)
        self.assertNotIn("compare_diarization_run_to_golden.py", joined)

        for idx, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"07_stage_diarization_only.ipynb:cell{idx}")


if __name__ == "__main__":
    unittest.main()
