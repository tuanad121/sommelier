from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT_PATH = Path("kaggle_notebooks/08_stage_music_overlap_only.ipynb")


def _source(text: str) -> list[str]:
    return [line + "\n" for line in textwrap.dedent(text).strip("\n").splitlines()]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": _source(text),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        md(
            """
            # Stage 02 + 03 Only: Music Clean + Overlap Separation

            Notebook này dùng output từ stage 01 diarization đã chạy trước đó, rồi chỉ chạy:
            - Stage 02: phát hiện/loại background music bằng PANNs + Demucs.
            - Stage 03: tách overlap bằng Metis-TSE Target Speaker Extraction.

            Không chạy lại speaker diarization, không chạy ASR, không tạo HTML.
            """
        ),
        md(
            """
            ## 0. Cấu hình input stage 01

            Điền đường dẫn trực tiếp tới file `diarization.json` và file âm thanh gốc `full.wav` để chạy tiếp Stage 2 & 3.
            """
        ),
        code(
            """
            from pathlib import Path

            # =========================
            # 0. GitHub branch controls
            # =========================
            REPO_URL = "https://github.com/tuanad121/sommelier.git"
            BRANCH = "kaggle-gpu"
            FORCE_RECLONE = True

            # =========================
            # 1. Stage 01 input files
            # =========================
            DIARIZATION_JSON_PATH = ""
            AUDIO_WAV_PATH = ""

            # Optional. Để "" thì notebook tự chọn:
            # - Nếu input là run_full_* writable trong /kaggle/working: ghi tiếp vào run đó.
            # - Nếu input nằm trong /kaggle/input hoặc folder upload read-only: ghi vào OUTPUT_ROOT/stage23_<audio>.
            OUTPUT_RUN_DIR = ""

            # =========================
            # 2. Kaggle paths
            # =========================
            PROJECT_ROOT = Path("/kaggle/working/sommelier")
            PIPELINE_DIR = PROJECT_ROOT / "podcast-pipeline"
            CONFIG_PATH = PIPELINE_DIR / "config.json"
            BATCH_ROOT = Path("/kaggle/working/sommelier_batch_outputs")
            OUTPUT_ROOT = BATCH_ROOT / "diarization_only_runs"
            SETUP_LOG_DIR = BATCH_ROOT / "setup_logs_stage23"

            # =========================
            # 3. Runtime / GPU controls
            # =========================
            CUDA_VISIBLE_DEVICES = "0"
            PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
            REQUIRE_GPU = True
            PRINT_NVIDIA_SMI = True

            RUN_DEMUCS = True
            RUN_METIS_TSE = True
            OVERLAP_THRESHOLD = 1.0
            DEMUCS_PADDING = 0.5
            DEMUCS_MODEL_NAME = "htdemucs"
            METIS_REPO_URL = "https://github.com/open-mmlab/Amphion.git"
            METIS_REPO_DIR = Path("/kaggle/working/Amphion")
            METIS_FORCE_RECLONE = False
            METIS_CKPT_DIR = METIS_REPO_DIR / "models/tts/metis/ckpt"
            METIS_N_TIMESTEPS = 10
            METIS_GUIDANCE_CFG = 0.0

            # Kaggle 1x GPU: dùng 0. Dùng -1 để ép CPU.
            PANNS_DEVICE_INDEX = 0
            DEMUCS_DEVICE_INDEX = 0
            METIS_DEVICE_INDEX = 0

            # =========================
            # 4. Pinned package versions
            # =========================
            INSTALL_DEPENDENCIES = True
            TORCH_PACKAGE = "torch==2.7.1"
            TORCHAUDIO_PACKAGE = "torchaudio==2.7.1"
            TORCHVISION_PACKAGE = "torchvision==0.22.1"
            PYTORCH_WHEEL_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
            SETUPTOOLS_PACKAGE = "setuptools>=70.0.0"
            PANNS_PACKAGE = "panns-inference"
            PILLOW_PACKAGE = "pillow==11.3.0"
            TRANSFORMERS_PACKAGE = "transformers==4.41.2"
            TOKENIZERS_PACKAGE = "tokenizers>=0.19,<0.20"
            ACCELERATE_PACKAGE = "accelerate==0.24.1"
            PEFT_PACKAGE = "peft==0.13.2"
            NUMPY_PACKAGE = "numpy==1.26.4"
            NUMBA_PACKAGE = "numba==0.61.2"
            LLVMLITE_PACKAGE = "llvmlite==0.44.0"

            STAGE23_PACKAGES = [
                "pydub==0.25.1",
                "numpy==1.26.4",
                "librosa==0.10.2.post1",
                "soundfile==0.12.1",
                "pandas==2.3.1",
                "huggingface-hub==0.33.4",
                "PyYAML==6.0.2",
                "demucs==4.0.1",
                PANNS_PACKAGE,
                TRANSFORMERS_PACKAGE,
                TOKENIZERS_PACKAGE,
                ACCELERATE_PACKAGE,
                "safetensors",
                PEFT_PACKAGE,
                "scipy==1.12.0",
                "json5",
                "ruamel.yaml",
                "langid",
                "unidecode",
                "encodec",
                "onnxruntime",
                "phonemizer",
                "g2p_en",
                "jieba",
                "cn2an",
                "pypinyin",
                "LangSegment",
                "pyopenjtalk",
                "pykakasi",
            ]

            # =========================
            # 5. Secrets
            # =========================
            HF_SECRET_NAME = "HF_TOKEN"

            DIARIZATION_JSON = Path(DIARIZATION_JSON_PATH)
            FULL_AUDIO_PATH = Path(AUDIO_WAV_PATH)
            if not DIARIZATION_JSON_PATH or "YOUR_AUDIO" in DIARIZATION_JSON_PATH:
                raise FileNotFoundError(f"Hãy sửa DIARIZATION_JSON_PATH thành đường dẫn đúng: {DIARIZATION_JSON_PATH}")
            if not AUDIO_WAV_PATH or "YOUR_AUDIO" in AUDIO_WAV_PATH:
                raise FileNotFoundError(f"Hãy sửa AUDIO_WAV_PATH thành đường dẫn đúng: {AUDIO_WAV_PATH}")

            import re

            def safe_stem(path):
                stem = Path(path).stem
                stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
                return (stem or "audio")[:80]

            def is_writable_stage1_run_dir(audio_path, diarization_path):
                audio_path = Path(audio_path)
                diarization_path = Path(diarization_path)
                if audio_path.parent.name != "00_input" or diarization_path.parent.name != "01_diarization":
                    return False
                if audio_path.parents[1] != diarization_path.parents[1]:
                    return False
                candidate = audio_path.parents[1]
                if str(candidate).startswith("/kaggle/input"):
                    return False
                return True

            if OUTPUT_RUN_DIR:
                RUN_DIR = Path(OUTPUT_RUN_DIR)
            elif is_writable_stage1_run_dir(FULL_AUDIO_PATH, DIARIZATION_JSON):
                RUN_DIR = FULL_AUDIO_PATH.parents[1]
            else:
                RUN_DIR = OUTPUT_ROOT / f"stage23_{safe_stem(FULL_AUDIO_PATH)}"
            MUSIC_JSON = RUN_DIR / "02_music_clean" / "segment_flags.json"
            CLEANED_AUDIO_PATH = RUN_DIR / "02_music_clean" / "cleaned_audio.wav"
            OVERLAP_JSON = RUN_DIR / "03_overlap" / "segments.json"

            print("REPO_URL =", REPO_URL)
            print("BRANCH =", BRANCH)
            print("RUN_DIR =", RUN_DIR)
            print("DIARIZATION_JSON =", DIARIZATION_JSON)
            print("FULL_AUDIO_PATH =", FULL_AUDIO_PATH)
            print("RUN_DEMUCS =", RUN_DEMUCS)
            print("RUN_METIS_TSE =", RUN_METIS_TSE)
            print("OVERLAP_THRESHOLD =", OVERLAP_THRESHOLD)
            print("METIS_REPO_DIR =", METIS_REPO_DIR)
            print("METIS_CKPT_DIR =", METIS_CKPT_DIR)
            print("Expected outputs: 02_music_clean/segment_flags.json, 02_music_clean/cleaned_audio.wav, 03_overlap/segments.json")
            print("Device indices:")
            print("  panns =", PANNS_DEVICE_INDEX)
            print("  demucs =", DEMUCS_DEVICE_INDEX)
            print("  metis =", METIS_DEVICE_INDEX)
            """
        ),
        md("## 1. Helper chạy lệnh và thiết lập GPU"),
        code(
            """
            import os
            import shlex
            import subprocess
            import time
            from pathlib import Path

            for _dir in [BATCH_ROOT, OUTPUT_ROOT, SETUP_LOG_DIR]:
                Path(_dir).mkdir(parents=True, exist_ok=True)

            if CUDA_VISIBLE_DEVICES is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
            if PYTORCH_CUDA_ALLOC_CONF:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(PYTORCH_CUDA_ALLOC_CONF)

            print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
            print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<not set>"))

            def _format_cmd(cmd):
                return " ".join(shlex.quote(str(part)) for part in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)

            def tail_file(path, n=30):
                path = Path(path)
                if not path.exists():
                    return ""
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                return "\\n".join(lines[-n:])

            def run_logged(cmd, log_name, cwd="/kaggle/working", env=None, shell=False, tail=20):
                log_path = SETUP_LOG_DIR / log_name
                log_path.parent.mkdir(parents=True, exist_ok=True)
                print("Running:", _format_cmd(cmd))
                print("Log:", log_path)
                start = time.time()
                with open(log_path, "w", encoding="utf-8", errors="replace") as log:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(cwd),
                        env=env,
                        shell=shell,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                print("Exit code:", proc.returncode, "| seconds:", round(time.time() - start, 2))
                log_tail = tail_file(log_path, n=tail)
                if log_tail:
                    print(f"--- last {tail} log lines ---")
                    print(log_tail)
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
                return log_path
            """
        ),
        md("## 2. Clone đúng nhánh GitHub"),
        code(
            """
            import os
            import shutil
            import subprocess
            from pathlib import Path

            os.chdir("/kaggle/working")
            if PROJECT_ROOT.exists() and FORCE_RECLONE:
                shutil.rmtree(PROJECT_ROOT)

            if not PROJECT_ROOT.exists():
                run_logged(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(PROJECT_ROOT)], "01_clone_repo.log", tail=40)
            else:
                print("Repo already exists:", PROJECT_ROOT)
                run_logged(["git", "fetch", "origin", BRANCH], "01_fetch_repo.log", cwd=PROJECT_ROOT, tail=30)
                run_logged(["git", "checkout", BRANCH], "01_checkout_branch.log", cwd=PROJECT_ROOT, tail=30)

            os.chdir(PIPELINE_DIR)
            print("cwd:", os.getcwd())
            print("branch:", subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip())
            print("commit:", subprocess.check_output(["git", "log", "-1", "--oneline"], text=True).strip())
            """
        ),
        md("## 3. Cài dependencies cho stage 02 + 03"),
        code(
            """
            import importlib
            import os
            import shutil
            import sys
            from pathlib import Path

            os.chdir(PIPELINE_DIR)
            print("Python executable =", sys.executable)

            if INSTALL_DEPENDENCIES:
                run_logged(["apt-get", "update", "-y"], "02_apt_update.log", tail=10)
                run_logged(["apt-get", "install", "-y", "ffmpeg", "git", "git-lfs", "libaio-dev", "espeak-ng", "libespeak-ng1"], "03_apt_install.log", tail=10)
                run_logged([sys.executable, "-m", "pip", "install", "-U", "pip", SETUPTOOLS_PACKAGE, "wheel", "packaging", "ninja"], "04_pip_base.log", tail=12)

                torch_stack_cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall"]
                if PYTORCH_WHEEL_EXTRA_INDEX_URL:
                    torch_stack_cmd.extend(["--extra-index-url", PYTORCH_WHEEL_EXTRA_INDEX_URL])
                torch_stack_cmd.extend([TORCH_PACKAGE, TORCHAUDIO_PACKAGE, TORCHVISION_PACKAGE])
                run_logged(torch_stack_cmd, "05_pip_torch_stack.log", tail=30)

                run_logged([sys.executable, "-m", "pip", "install", *STAGE23_PACKAGES], "06_pip_stage23_packages.log", tail=30)
                run_logged([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "--no-deps", PANNS_PACKAGE], "07_pip_panns_inference.log", tail=20)
                run_logged([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", NUMPY_PACKAGE, NUMBA_PACKAGE, LLVMLITE_PACKAGE], "08_pip_numpy_numba.log", tail=16)
                run_logged([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", SETUPTOOLS_PACKAGE], "09_pip_setuptools_py312.log", tail=12)
                run_logged([sys.executable, "-m", "pip", "uninstall", "-y", "Pillow", "pillow"], "10_pip_pillow_uninstall.log", tail=12)
                run_logged([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", PILLOW_PACKAGE], "11_pip_pillow.log", tail=12)
            else:
                print("INSTALL_DEPENDENCIES=False, bỏ qua cài dependencies.")

            def required_metis_paths(repo_dir):
                repo_dir = Path(repo_dir)
                return [
                    repo_dir / "models/tts/metis/metis.py",
                    repo_dir / "models/tts/metis/audio_tokenizer.py",
                    repo_dir / "models/tts/metis/config/tse.json",
                    repo_dir / "models/tts/maskgct/maskgct_utils.py",
                    repo_dir / "models/tts/maskgct/g2p/g2p_generation.py",
                ]

            def missing_metis_paths(repo_dir):
                return [path for path in required_metis_paths(repo_dir) if not path.exists()]

            def is_valid_metis_repo(repo_dir):
                return not missing_metis_paths(repo_dir)

            if RUN_METIS_TSE:
                metis_entrypoint = METIS_REPO_DIR / "models/tts/metis/metis.py"
                missing_paths = missing_metis_paths(METIS_REPO_DIR) if METIS_REPO_DIR.exists() else []
                if METIS_REPO_DIR.exists() and (METIS_FORCE_RECLONE or not is_valid_metis_repo(METIS_REPO_DIR)):
                    print("Removing stale/invalid Metis repo:", METIS_REPO_DIR)
                    print("Expected Metis entrypoint:", metis_entrypoint)
                    if missing_paths:
                        print("Missing Metis repo files:")
                        for path in missing_paths:
                            print(" -", path)
                    shutil.rmtree(METIS_REPO_DIR)
                if not METIS_REPO_DIR.exists():
                    run_logged(["git", "clone", "--depth", "1", METIS_REPO_URL, str(METIS_REPO_DIR)], "10_clone_amphion_metis.log", tail=40)
                else:
                    print("Metis repo already exists:", METIS_REPO_DIR)
                if not is_valid_metis_repo(METIS_REPO_DIR):
                    missing_paths = missing_metis_paths(METIS_REPO_DIR)
                    missing_text = "\\n".join(f" - {path}" for path in missing_paths)
                    raise FileNotFoundError(f"Missing Metis repo files after clone:\\n{missing_text}")
            else:
                print("RUN_METIS_TSE=False, bỏ qua clone Amphion.")

            def configure_cuda_library_paths():
                module_names = [
                    "nvidia.cublas.lib",
                    "nvidia.cuda_runtime.lib",
                    "nvidia.cuda_nvrtc.lib",
                    "nvidia.cudnn.lib",
                    "nvidia.cufft.lib",
                    "nvidia.curand.lib",
                    "nvidia.cusolver.lib",
                    "nvidia.cusparse.lib",
                    "nvidia.nccl.lib",
                ]
                paths = []
                for module_name in module_names:
                    try:
                        module = importlib.import_module(module_name)
                        paths.append(Path(module.__file__).parent)
                    except Exception as exc:
                        print(f"Skip CUDA lib path {module_name}: {exc}")
                paths.extend([Path("/usr/local/nvidia/lib64"), Path("/usr/local/cuda/lib64")])
                seen = set()
                result = []
                for path in paths:
                    path = str(path)
                    if path and path not in seen and Path(path).exists():
                        seen.add(path)
                        result.append(path)
                os.environ["LD_LIBRARY_PATH"] = ":".join(result + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(":")
                os.environ.pop("LD_PRELOAD", None)
                print("LD_LIBRARY_PATH =", os.environ["LD_LIBRARY_PATH"])
                print("LD_PRELOAD cleared")
                return result

            configure_cuda_library_paths()
            """
        ),
        md("## 4. Kiểm tra runtime"),
        code(
            """
            import importlib
            import importlib.metadata as importlib_metadata
            import subprocess
            import sys
            import torch
            import torchaudio
            from PIL import Image, ImageDraw, ImageFont
            print("Pillow image modules import OK")
            import torchvision

            def clear_import_prefixes(*prefixes):
                for module_name in list(sys.modules):
                    if module_name in prefixes or any(module_name.startswith(prefix + ".") for prefix in prefixes):
                        del sys.modules[module_name]

            def install_python_packages(packages, log_name, install_args=None, tail=30):
                cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall"]
                if install_args:
                    cmd.extend(install_args)
                cmd.extend(packages)
                run_logged(cmd, log_name, tail=tail)
                importlib.invalidate_caches()

            def ensure_python_module(module_name, package_name, log_name, install_args=None):
                try:
                    return importlib.import_module(module_name)
                except ModuleNotFoundError as exc:
                    if exc.name != module_name:
                        raise
                    print(f"{module_name} missing, reinstalling {package_name}")
                    install_python_packages([package_name], log_name, install_args=install_args)
                    clear_import_prefixes(module_name)
                    return importlib.import_module(module_name)

            def ensure_peft_stack():
                try:
                    importlib.import_module("transformers")
                    print("transformers import OK")
                    importlib.import_module("peft")
                    print("peft import OK")
                    from peft import LoraConfig, LoraModel
                    print("peft LoraModel import OK")
                    return
                except (ImportError, ModuleNotFoundError) as exc:
                    print(f"transformers/peft import failed: {exc}")
                    print("Reinstalling Amphion/Metis-compatible transformers/tokenizers/accelerate/peft stack")
                    install_python_packages([TRANSFORMERS_PACKAGE, TOKENIZERS_PACKAGE, ACCELERATE_PACKAGE, PEFT_PACKAGE], "13_pip_metis_dependency_stack_runtime.log", install_args=["--no-deps"], tail=40)
                    clear_import_prefixes("transformers", "tokenizers", "accelerate", "peft")
                    importlib.import_module("transformers")
                    print("transformers import OK")
                    importlib.import_module("peft")
                    print("peft import OK")
                    from peft import LoraConfig, LoraModel
                    print("peft LoraModel import OK")

            if RUN_DEMUCS:
                ensure_python_module("panns_inference", PANNS_PACKAGE, "12_pip_panns_inference_runtime.log", install_args=["--no-deps"])
                print("panns_inference import OK")

            for pkg in [
                "torch",
                "torchaudio",
                "torchvision",
                "demucs",
                "panns-inference",
                "librosa",
                "soundfile",
                "pandas",
                "numpy",
                "numba",
                "safetensors",
                "peft",
                "accelerate",
                "langid",
                "json5",
            ]:
                try:
                    if pkg == "torch":
                        version = torch.__version__
                    elif pkg == "torchaudio":
                        version = torchaudio.__version__
                    elif pkg == "torchvision":
                        version = torchvision.__version__
                    else:
                        version = importlib_metadata.version(pkg)
                    print(pkg + ":", version)
                except Exception as exc:
                    print(pkg + ":", "missing", exc)

            print("CUDA:", torch.cuda.is_available())
            visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print("Visible CUDA device count:", visible_gpu_count)
            if torch.cuda.is_available():
                for i in range(visible_gpu_count):
                    print(f"GPU {i}:", torch.cuda.get_device_name(i))
                invalid = {
                    name: idx for name, idx in {
                        "panns": PANNS_DEVICE_INDEX,
                        "demucs": DEMUCS_DEVICE_INDEX,
                        "metis": METIS_DEVICE_INDEX,
                    }.items()
                    if idx is not None and idx >= visible_gpu_count
                }
                if invalid:
                    raise ValueError(f"GPU index không hợp lệ. Kaggle chỉ thấy {visible_gpu_count} GPU: {invalid}")
            if REQUIRE_GPU and not torch.cuda.is_available():
                raise RuntimeError("REQUIRE_GPU=True nhưng torch.cuda.is_available() = False. Hãy bật Kaggle GPU hoặc đặt REQUIRE_GPU=False.")
            if PRINT_NVIDIA_SMI:
                subprocess.run(["nvidia-smi"], check=False)

            if RUN_METIS_TSE:
                ensure_peft_stack()
                from phonemizer.backend import EspeakBackend
                if not EspeakBackend.is_available():
                    raise RuntimeError("espeak-ng is not available for phonemizer. Re-run the dependency cell that installs espeak-ng and libespeak-ng1.")
                print("espeak-ng available for phonemizer")
            """
        ),
        md("## 5. Gắn Hugging Face token vào config"),
        code(
            """
            import json
            from kaggle_secrets import UserSecretsClient
            from huggingface_hub import login, whoami

            try:
                token = UserSecretsClient().get_secret(HF_SECRET_NAME)
            except Exception as exc:
                raise RuntimeError(f"Không đọc được Kaggle Secret {HF_SECRET_NAME}. Hãy tạo secret HF_TOKEN trước khi chạy.") from exc

            if not token:
                raise RuntimeError(f"Kaggle Secret {HF_SECRET_NAME} đang trống.")

            login(token=token, add_to_git_credential=False)
            print("HF token:", token[:8] + "...")
            try:
                print(whoami(token=token))
            except Exception as exc:
                print("HF whoami failed, nhưng token vẫn được ghi vào config:", exc)

            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg["huggingface_token"] = token
            cfg.setdefault("entrypoint", {})["SAMPLE_RATE"] = 16000
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            print("config.json updated:", CONFIG_PATH)
            """
        ),
        md("## 6. Tải PANNs và chuẩn bị Metis-TSE"),
        code(
            """
            import os
            from pathlib import Path
            from huggingface_hub import hf_hub_download, snapshot_download

            if RUN_DEMUCS:
                panns_path = hf_hub_download(
                    repo_id="thelou1s/panns-inference",
                    filename="Cnn14_mAP=0.431.pth",
                    local_dir=str(PROJECT_ROOT / "panns_data"),
                )
                print("PANNs checkpoint:", panns_path)
            else:
                print("RUN_DEMUCS=False, bỏ qua tải PANNs")

            if RUN_METIS_TSE:
                METIS_CKPT_DIR.mkdir(parents=True, exist_ok=True)
                metis_dir = snapshot_download(
                    "amphion/metis",
                    repo_type="model",
                    local_dir=str(METIS_CKPT_DIR),
                    allow_patterns=[
                        "metis_base/model.safetensors",
                        "metis_tse/metis_tse_lora_32.safetensors",
                        "metis_tse/metis_tse_lora_32_adapter.safetensors",
                    ],
                )
                maskgct_dir = snapshot_download(
                    "amphion/MaskGCT",
                    repo_type="model",
                    local_dir=str(METIS_CKPT_DIR),
                    allow_patterns=[
                        "s2a_model/s2a_model_1layer/model.safetensors",
                        "s2a_model/s2a_model_full/model.safetensors",
                    ],
                )
                print("Metis checkpoint dir:", metis_dir)
                print("MaskGCT checkpoint dir:", maskgct_dir)
            else:
                print("RUN_METIS_TSE=False, bỏ qua Metis-TSE")

            os.chdir(PIPELINE_DIR)
            """
        ),
        md("## 7. Kiểm tra stage 01 artifacts"),
        code(
            """
            for required in [DIARIZATION_JSON, FULL_AUDIO_PATH]:
                if not required.exists():
                    raise FileNotFoundError(
                        f"Không tìm thấy stage 01 artifact: {required}\\n"
                        "Vui lòng kiểm tra lại đường dẫn DIARIZATION_JSON_PATH hoặc AUDIO_WAV_PATH."
                    )
            print("Stage 01 artifacts OK")
            print("DIARIZATION_JSON =", DIARIZATION_JSON)
            print("FULL_AUDIO_PATH =", FULL_AUDIO_PATH)
            """
        ),
        md("## 8. Chạy stage 02 + 03"),
        code(
            """
            import sys

            cmd = [
                sys.executable, str(PIPELINE_DIR / "run_stage_music_overlap_only.py"),
                "--audio_path", str(FULL_AUDIO_PATH),
                "--diarization_json", str(DIARIZATION_JSON),
                "--output_run_dir", str(RUN_DIR),
                "--config_path", str(CONFIG_PATH),
                "--overlap_threshold", str(OVERLAP_THRESHOLD),
                "--demucs_padding", str(DEMUCS_PADDING),
                "--demucs_model_name", DEMUCS_MODEL_NAME,
                "--panns_data_dir", str(PROJECT_ROOT / "panns_data"),
                "--panns_device_index", str(PANNS_DEVICE_INDEX),
                "--demucs_device_index", str(DEMUCS_DEVICE_INDEX),
                "--metis_repo_dir", str(METIS_REPO_DIR),
                "--metis_ckpt_dir", str(METIS_CKPT_DIR),
                "--metis_n_timesteps", str(METIS_N_TIMESTEPS),
                "--metis_guidance_cfg", str(METIS_GUIDANCE_CFG),
                "--metis_device_index", str(METIS_DEVICE_INDEX),
                "--demucs" if RUN_DEMUCS else "--no-demucs",
                "--metis_tse" if RUN_METIS_TSE else "--no-metis_tse",
            ]
            run_logged(cmd, f"20_stage_02_03_{FULL_AUDIO_PATH.stem}.log", cwd=PIPELINE_DIR, env=os.environ.copy(), tail=80)
            """
        ),
        md("## 9. In kết quả stage 02 + 03"),
        code(
            """
            import json
            import pandas as pd

            if not MUSIC_JSON.exists():
                raise FileNotFoundError(f"Không tìm thấy music output: {MUSIC_JSON}")
            if not OVERLAP_JSON.exists():
                raise FileNotFoundError(f"Không tìm thấy overlap output: {OVERLAP_JSON}")

            music_payload = json.loads(MUSIC_JSON.read_text(encoding="utf-8"))
            overlap_payload = json.loads(OVERLAP_JSON.read_text(encoding="utf-8"))
            segments = overlap_payload.get("segments", [])

            print("music_json =", MUSIC_JSON)
            print("cleaned_audio =", CLEANED_AUDIO_PATH)
            print("overlap_json =", OVERLAP_JSON)
            print("music metadata =", music_payload.get("metadata", {}))
            print("overlap metadata =", overlap_payload.get("metadata", {}))
            print("demucs applied =", sum(bool(x) for x in music_payload.get("segment_demucs_flags", [])))
            print("segments =", len(segments))

            df = pd.DataFrame(segments)
            if df.empty:
                print("Không có segment overlap output.")
            else:
                df["duration"] = (df["end"].astype(float) - df["start"].astype(float)).round(3)
                for col in ["sepreformer", "is_separated"]:
                    if col not in df.columns:
                        df[col] = False
                keep = ["index", "start", "end", "duration", "speaker", "sepreformer", "is_separated"]
                if "enhanced_audio_path" in df.columns:
                    keep.append("enhanced_audio_path")
                display(df[keep])

                speaker_summary = (
                    df.groupby("speaker", dropna=False)
                    .agg(segments=("speaker", "size"), total_seconds=("duration", "sum"), separated=("sepreformer", "sum"))
                    .reset_index()
                    .sort_values("speaker")
                )
                speaker_summary["total_seconds"] = speaker_summary["total_seconds"].round(3)
                display(speaker_summary)
            """
        ),
        md("## 10. Nghe cleaned audio và segment sau overlap"),
        code(
            """
            from IPython.display import Audio, HTML, display
            from pydub import AudioSegment

            if not CLEANED_AUDIO_PATH.exists():
                raise FileNotFoundError(f"Không tìm thấy cleaned audio: {CLEANED_AUDIO_PATH}")

            display(HTML("<h3>Cleaned full audio</h3>"))
            cleaned_audio = Audio(str(CLEANED_AUDIO_PATH))
            display(cleaned_audio)

            SEGMENT_AUDIO_DIR = RUN_DIR / "03_overlap" / "segment_audio_preview"
            SEGMENT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

            if df.empty:
                print("Không có segment để tạo audio player.")
            else:
                full_audio_segment = AudioSegment.from_file(CLEANED_AUDIO_PATH)
                segment_audio_rows = []
                for _, row in df.iterrows():
                    index = str(row["index"])
                    speaker = str(row["speaker"])
                    start = float(row["start"])
                    end = float(row["end"])
                    duration = float(row["duration"])
                    enhanced_relative = row.get("enhanced_audio_path") if "enhanced_audio_path" in row else None
                    if isinstance(enhanced_relative, str) and enhanced_relative:
                        segment_path = RUN_DIR / enhanced_relative
                    else:
                        start_ms = max(0, int(round(start * 1000)))
                        end_ms = max(start_ms, int(round(end * 1000)))
                        segment_path = SEGMENT_AUDIO_DIR / f"{index}_{speaker}.wav"
                        full_audio_segment[start_ms:end_ms].export(segment_path, format="wav")
                    segment_audio_rows.append({
                        "index": index,
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "speaker": speaker,
                        "sepreformer": bool(row.get("sepreformer", False)),
                        "segment_audio": str(segment_path),
                    })

                segment_audio_df = pd.DataFrame(segment_audio_rows)
                display(HTML("<h3>Stage 03 segment audio players</h3>"))
                display(segment_audio_df)

                for item in segment_audio_rows:
                    label = (
                        f"<b>{item['index']}</b> | {item['speaker']} | "
                        f"{item['start']:.3f}s - {item['end']:.3f}s | "
                        f"duration {item['duration']:.3f}s | "
                        f"sepreformer={item['sepreformer']}"
                    )
                    display(HTML(label))
                    display(Audio(item["segment_audio"]))
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
