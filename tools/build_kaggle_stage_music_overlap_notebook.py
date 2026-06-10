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
            - Stage 03: tách overlap bằng SepReformer.

            Không chạy lại speaker diarization, không chạy ASR, không tạo HTML.
            """
        ),
        md(
            """
            ## 0. Cấu hình input stage 01

            Nếu bạn vừa chạy notebook `07_stage_diarization_only.ipynb` trong cùng Kaggle session, chỉ cần sửa `AUDIO_INPUT_PATH` giống file audio đã chạy ở stage 01 và để `STAGE1_RUN_DIR = ""`.

            Nếu bạn đã copy/download output stage 01 từ nơi khác, điền thẳng `STAGE1_RUN_DIR`.
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
            # 1. Stage 01 input controls
            # =========================
            # Phải trùng với audio đã chạy ở notebook stage 01 nếu STAGE1_RUN_DIR để trống.
            AUDIO_INPUT_PATH = "/kaggle/input/YOUR_DATASET/YOUR_AUDIO.wav"
            AUDIO_INPUT_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus")

            # Để "" thì notebook tự dùng:
            # /kaggle/working/sommelier_batch_outputs/diarization_only_runs/run_full_<AUDIO_STEM>
            STAGE1_RUN_DIR = ""

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
            RUN_SEPREFORMER = True
            OVERLAP_THRESHOLD = 1.0
            DEMUCS_PADDING = 0.5
            DEMUCS_MODEL_NAME = "htdemucs"

            # Kaggle 1x GPU: dùng 0. Dùng -1 để ép CPU.
            PANNS_DEVICE_INDEX = 0
            DEMUCS_DEVICE_INDEX = 0
            SEPREFORMER_DEVICE_INDEX = 0

            # =========================
            # 4. Pinned package versions
            # =========================
            INSTALL_DEPENDENCIES = True
            TORCH_PACKAGE = "torch==2.7.1"
            TORCHAUDIO_PACKAGE = "torchaudio==2.7.1"
            TORCHVISION_PACKAGE = "torchvision==0.22.1"
            PYTORCH_WHEEL_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
            NUMPY_PACKAGE = "numpy==2.2.6"
            NUMBA_PACKAGE = "numba==0.61.2"
            LLVMLITE_PACKAGE = "llvmlite==0.44.0"

            STAGE23_PACKAGES = [
                "pydub==0.25.1",
                "librosa==0.11.0",
                "soundfile==0.13.1",
                "pandas==2.3.1",
                "huggingface-hub==0.33.4",
                "PyYAML==6.0.2",
                "lightning==2.4.0",
                "pytorch-lightning==2.5.2",
                "torchmetrics==1.7.4",
                "pyannote.audio==3.3.2",
                "demucs==4.0.1",
                "panns-inference",
            ]
            SEPREFORMER_EXTRA_PACKAGES = [
                "mir-eval==0.7",
                "ptflops==0.7.4",
                "thop==0.1.1.post2209072238",
                "torchinfo==1.8.0",
            ]

            # =========================
            # 5. Secrets
            # =========================
            HF_SECRET_NAME = "HF_TOKEN"

            AUDIO_PATH = Path(AUDIO_INPUT_PATH)
            if not AUDIO_INPUT_PATH or "YOUR_AUDIO" in AUDIO_INPUT_PATH:
                raise FileNotFoundError(f"Hãy sửa AUDIO_INPUT_PATH thành đường dẫn audio đã chạy stage 01: {AUDIO_INPUT_PATH}")
            if AUDIO_PATH.suffix.lower() not in AUDIO_INPUT_EXTENSIONS:
                raise ValueError(f"Định dạng audio chưa hỗ trợ: {AUDIO_PATH.suffix}. Hỗ trợ: {AUDIO_INPUT_EXTENSIONS}")

            RUN_DIR = Path(STAGE1_RUN_DIR) if STAGE1_RUN_DIR else OUTPUT_ROOT / f"run_full_{AUDIO_PATH.stem}"
            DIARIZATION_JSON = RUN_DIR / "01_diarization" / "diarization.json"
            FULL_AUDIO_PATH = RUN_DIR / "00_input" / "full.wav"
            MUSIC_JSON = RUN_DIR / "02_music_clean" / "segment_flags.json"
            CLEANED_AUDIO_PATH = RUN_DIR / "02_music_clean" / "cleaned_audio.wav"
            OVERLAP_JSON = RUN_DIR / "03_overlap" / "segments.json"

            print("REPO_URL =", REPO_URL)
            print("BRANCH =", BRANCH)
            print("AUDIO_INPUT_PATH =", AUDIO_PATH)
            print("RUN_DIR =", RUN_DIR)
            print("DIARIZATION_JSON =", DIARIZATION_JSON)
            print("FULL_AUDIO_PATH =", FULL_AUDIO_PATH)
            print("RUN_DEMUCS =", RUN_DEMUCS)
            print("RUN_SEPREFORMER =", RUN_SEPREFORMER)
            print("OVERLAP_THRESHOLD =", OVERLAP_THRESHOLD)
            print("Expected outputs: 02_music_clean/segment_flags.json, 02_music_clean/cleaned_audio.wav, 03_overlap/segments.json")
            print("Device indices:")
            print("  panns =", PANNS_DEVICE_INDEX)
            print("  demucs =", DEMUCS_DEVICE_INDEX)
            print("  sepreformer =", SEPREFORMER_DEVICE_INDEX)
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
            from pathlib import Path

            os.chdir(PIPELINE_DIR)

            if INSTALL_DEPENDENCIES:
                run_logged(["apt-get", "update", "-y"], "02_apt_update.log", tail=10)
                run_logged(["apt-get", "install", "-y", "ffmpeg", "git", "git-lfs", "libaio-dev"], "03_apt_install.log", tail=10)
                run_logged(["python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "packaging", "ninja"], "04_pip_base.log", tail=12)

                torch_stack_cmd = ["python", "-m", "pip", "install", "--no-cache-dir", "--force-reinstall"]
                if PYTORCH_WHEEL_EXTRA_INDEX_URL:
                    torch_stack_cmd.extend(["--extra-index-url", PYTORCH_WHEEL_EXTRA_INDEX_URL])
                torch_stack_cmd.extend([TORCH_PACKAGE, TORCHAUDIO_PACKAGE, TORCHVISION_PACKAGE])
                run_logged(torch_stack_cmd, "05_pip_torch_stack.log", tail=30)

                run_logged(["python", "-m", "pip", "install", *STAGE23_PACKAGES], "06_pip_stage23_packages.log", tail=30)
                run_logged(["python", "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", NUMPY_PACKAGE, NUMBA_PACKAGE, LLVMLITE_PACKAGE], "07_pip_numpy_numba.log", tail=16)
            else:
                print("INSTALL_DEPENDENCIES=False, bỏ qua cài dependencies.")

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
            import importlib.metadata as importlib_metadata
            import subprocess
            import torch
            import torchaudio
            import torchvision

            for pkg in [
                "torch",
                "torchaudio",
                "torchvision",
                "pyannote.audio",
                "demucs",
                "panns-inference",
                "librosa",
                "soundfile",
                "pandas",
                "numpy",
                "numba",
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
                        "sepreformer": SEPREFORMER_DEVICE_INDEX,
                    }.items()
                    if idx is not None and idx >= visible_gpu_count
                }
                if invalid:
                    raise ValueError(f"GPU index không hợp lệ. Kaggle chỉ thấy {visible_gpu_count} GPU: {invalid}")
            if REQUIRE_GPU and not torch.cuda.is_available():
                raise RuntimeError("REQUIRE_GPU=True nhưng torch.cuda.is_available() = False. Hãy bật Kaggle GPU hoặc đặt REQUIRE_GPU=False.")
            if PRINT_NVIDIA_SMI:
                subprocess.run(["nvidia-smi"], check=False)
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
        md("## 6. Tải PANNs và SepReformer"),
        code(
            """
            import os
            from pathlib import Path
            from huggingface_hub import hf_hub_download

            if RUN_DEMUCS:
                panns_path = hf_hub_download(
                    repo_id="thelou1s/panns-inference",
                    filename="Cnn14_mAP=0.431.pth",
                    local_dir=str(PROJECT_ROOT / "panns_data"),
                )
                print("PANNs checkpoint:", panns_path)
            else:
                print("RUN_DEMUCS=False, bỏ qua tải PANNs")

            if RUN_SEPREFORMER:
                run_logged(["git", "lfs", "install"], "08_git_lfs_install.log", cwd=PROJECT_ROOT, tail=10)
                os.chdir(PROJECT_ROOT)
                if not Path("SepReformer").exists():
                    run_logged(["git", "clone", "https://github.com/dmlguq456/SepReformer.git", "SepReformer"], "09_clone_sepreformer.log", cwd=PROJECT_ROOT, tail=20)
                run_logged(["git", "lfs", "pull"], "10_sepreformer_lfs_pull.log", cwd=PROJECT_ROOT / "SepReformer", tail=20)
                run_logged(["python", "-m", "pip", "install", "--no-deps", *SEPREFORMER_EXTRA_PACKAGES], "11_sepreformer_extra_deps.log", cwd=PROJECT_ROOT / "SepReformer", tail=12)

                log = PROJECT_ROOT / "SepReformer" / "models" / "SepReformer_Base_WSJ0" / "log"
                src = log / "scratch_weight"
                dst = log / "scratch_weights"
                if src.exists() and not dst.exists():
                    os.symlink(src, dst)

                ckpts = list(log.rglob("*.pt")) + list(log.rglob("*.pth"))
                print("SepReformer checkpoints:", len(ckpts))
                for p in ckpts[:10]:
                    print(p)
            else:
                print("RUN_SEPREFORMER=False, bỏ qua SepReformer")

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
                        "Hãy chạy notebook 07 trước, hoặc điền đúng STAGE1_RUN_DIR."
                    )
            print("Stage 01 artifacts OK")
            print("DIARIZATION_JSON =", DIARIZATION_JSON)
            print("FULL_AUDIO_PATH =", FULL_AUDIO_PATH)
            """
        ),
        md("## 8. Chạy stage 02 + 03"),
        code(
            """
            cmd = [
                "python", str(PIPELINE_DIR / "run_stage_music_overlap_only.py"),
                "--input_run_dir", str(RUN_DIR),
                "--config_path", str(CONFIG_PATH),
                "--overlap_threshold", str(OVERLAP_THRESHOLD),
                "--demucs_padding", str(DEMUCS_PADDING),
                "--demucs_model_name", DEMUCS_MODEL_NAME,
                "--panns_data_dir", str(PROJECT_ROOT / "panns_data"),
                "--sepreformer_path", str(PROJECT_ROOT / "SepReformer"),
                "--panns_device_index", str(PANNS_DEVICE_INDEX),
                "--demucs_device_index", str(DEMUCS_DEVICE_INDEX),
                "--sepreformer_device_index", str(SEPREFORMER_DEVICE_INDEX),
                "--demucs" if RUN_DEMUCS else "--no-demucs",
                "--sepreformer" if RUN_SEPREFORMER else "--no-sepreformer",
            ]
            run_logged(cmd, f"20_stage_02_03_{AUDIO_PATH.stem}.log", cwd=PIPELINE_DIR, env=os.environ.copy(), tail=80)
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
