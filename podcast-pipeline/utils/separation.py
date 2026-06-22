# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT
"""
Audio source separation utilities for podcast pipeline.
Includes SepReformer-based speaker separation and overlapping segment processing.
"""

import os
import sys
from pathlib import Path
import numpy as np
import librosa
import torch
from utils.logger import time_logger
from utils.diarization import detect_overlapping_segments
from utils.speaker_linking import build_weighted_reference_embedding, select_speaker_reference_candidates

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


def _ensure_pkgutil_impimporter_compat():
    """Keep legacy pkg_resources imports working on Python 3.12+."""
    import importlib.machinery
    import pkgutil

    if not hasattr(importlib.machinery.FileFinder, "find_module"):
        def find_module(self, fullname, path=None):
            spec = self.find_spec(fullname)
            return spec.loader if spec else None

        importlib.machinery.FileFinder.find_module = find_module

    if not hasattr(pkgutil, "ImpImporter"):
        pkgutil.ImpImporter = importlib.machinery.FileFinder


def _alias_langsegment_filters(module):
    if hasattr(module, "setfilters") and not hasattr(module, "setLangfilters"):
        module.setLangfilters = module.setfilters
    if hasattr(module, "getfilters") and not hasattr(module, "getLangfilters"):
        module.getLangfilters = module.getfilters


def _ensure_langsegment_compat():
    """Handle LangSegment packages that export old alias names from __init__."""
    import importlib
    import importlib.machinery
    import importlib.util
    import types

    try:
        module = importlib.import_module("LangSegment")
    except ImportError as exc:
        message = str(exc)
        if "setLangfilters" not in message and "getLangfilters" not in message:
            raise

        package_spec = importlib.machinery.PathFinder.find_spec("LangSegment", sys.path)
        if package_spec is None or not package_spec.submodule_search_locations:
            raise

        package_dir = Path(next(iter(package_spec.submodule_search_locations)))
        impl_path = package_dir / "LangSegment.py"
        if not impl_path.exists():
            raise

        sys.modules.pop("LangSegment", None)
        sys.modules.pop("LangSegment.LangSegment", None)

        package_module = types.ModuleType("LangSegment")
        package_module.__file__ = str(package_dir / "__init__.py")
        package_module.__path__ = [str(package_dir)]
        package_module.__package__ = "LangSegment"
        sys.modules["LangSegment"] = package_module

        impl_spec = importlib.util.spec_from_file_location("LangSegment.LangSegment", impl_path)
        if impl_spec is None or impl_spec.loader is None:
            sys.modules.pop("LangSegment", None)
            raise

        impl_module = importlib.util.module_from_spec(impl_spec)
        sys.modules["LangSegment.LangSegment"] = impl_module
        try:
            impl_spec.loader.exec_module(impl_module)
        except Exception:
            sys.modules.pop("LangSegment", None)
            sys.modules.pop("LangSegment.LangSegment", None)
            raise

        for name in (
            "LangSegment",
            "getTexts",
            "classify",
            "getCounts",
            "printList",
            "setfilters",
            "getfilters",
        ):
            if hasattr(impl_module, name):
                setattr(package_module, name, getattr(impl_module, name))

        _alias_langsegment_filters(impl_module)
        _alias_langsegment_filters(package_module)
        return package_module

    _alias_langsegment_filters(module)
    submodule = sys.modules.get("LangSegment.LangSegment")
    if submodule is not None:
        _alias_langsegment_filters(submodule)
    return module


def _prepend_import_path(path):
    path = str(Path(path).resolve())
    sys.path[:] = [item for item in sys.path if str(Path(item).resolve()) != path]
    sys.path.insert(0, path)


def _ensure_metis_repo_layout(repo_dir):
    entrypoint = Path(repo_dir) / "models" / "tts" / "metis" / "metis.py"
    if not entrypoint.exists():
        raise FileNotFoundError(
            "Metis repo_dir is not a full Amphion checkout. "
            f"Missing {entrypoint.relative_to(repo_dir)} under {repo_dir}. "
            "If this directory was created by checkpoint download or a partial clone, "
            "set METIS_FORCE_RECLONE=True or delete it, then rerun the Kaggle clone/dependency cell."
        )


class SepReformerSeparator:
    """
    Class that loads the SepReformer model once and can perform inference multiple times.
    """
    def __init__(self, sepreformer_path, device):
        """
        Initialize and load the SepReformer model.

        Args:
            sepreformer_path: Path to the SepReformer model directory
            device: torch device (cuda/cpu)
        """
        import yaml

        self.sepreformer_path = sepreformer_path
        self.device = device

        print(f"[SepReformer] Initializing on device: {self.device}")

        # Store original sys.path to restore later
        original_sys_path = sys.path.copy()

        try:
            # Save the current 'models' and 'utils' modules if they exist
            original_models = sys.modules.get('models', None)
            original_utils = sys.modules.get('utils', None)

            # Remove podcast-pipeline from sys.path temporarily
            podcast_pipeline_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            paths_to_remove = [p for p in sys.path if podcast_pipeline_path in p]
            for path in paths_to_remove:
                sys.path.remove(path)

            # Add SepReformer to path
            if sepreformer_path not in sys.path:
                sys.path.insert(0, sepreformer_path)

            # Clear conflicting modules
            modules_to_clear = [key for key in sys.modules.keys()
                              if key.startswith('models.') or key.startswith('utils.') or key in ['models', 'utils']]
            cleared_modules = {}
            for module_name in modules_to_clear:
                cleared_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

            # Import SepReformer's model
            from models.SepReformer_Base_WSJ0.model import Model

            # Restore the original modules
            for module_name, module_obj in cleared_modules.items():
                sys.modules[module_name] = module_obj

            # Load SepReformer config
            config_path = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/configs.yaml")
            with open(config_path, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            self.config = yaml_dict["config"]

            # Load model
            print("[SepReformer] Loading model...")
            self.model = Model(**self.config["model"])

            # Load checkpoint
            checkpoint_dir = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/log/pretrain_weights")
            if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
                checkpoint_dir = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/log/scratch_weights")

            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(device)
            self.model.eval()

            print("[SepReformer] Model initialization complete!")

        finally:
            # Restore original sys.path
            sys.path = original_sys_path

    def separate(self, audio_segment, sample_rate):
        """
        Perform audio source separation.

        Args:
            audio_segment (np.ndarray): Audio segment to separate
            sample_rate (int): Audio sample rate

        Returns:
            tuple: (separated_audio_1, separated_audio_2) as numpy arrays
        """
        try:
            # Resample to 8kHz if needed
            if sample_rate != 8000:
                audio_8k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=8000)
            else:
                audio_8k = audio_segment

            # Prepare tensor
            mixture_tensor = torch.tensor(audio_8k, dtype=torch.float32).unsqueeze(0)

            # Padding
            stride = self.config["model"]["module_audio_enc"]["stride"]
            remains = mixture_tensor.shape[-1] % stride
            if remains != 0:
                padding = stride - remains
                mixture_padded = torch.nn.functional.pad(mixture_tensor, (0, padding), "constant", 0)
            else:
                mixture_padded = mixture_tensor

            # Inference
            with torch.inference_mode():
                nnet_input = mixture_padded.to(self.device)
                estim_src, _ = self.model(nnet_input)

                # Extract separated sources
                src1 = estim_src[0][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()
                src2 = estim_src[1][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()

            # Resample back to original sample rate if needed
            if sample_rate != 8000:
                src1 = librosa.resample(src1, orig_sr=8000, target_sr=sample_rate)
                src2 = librosa.resample(src2, orig_sr=8000, target_sr=sample_rate)

                # Match length exactly to original after resampling (rounding error correction)
                target_length = len(audio_segment)
                if len(src1) != target_length:
                    if len(src1) > target_length:
                        src1 = src1[:target_length]
                    else:
                        src1 = np.pad(src1, (0, target_length - len(src1)), mode='constant')

                if len(src2) != target_length:
                    if len(src2) > target_length:
                        src2 = src2[:target_length]
                    else:
                        src2 = np.pad(src2, (0, target_length - len(src2)), mode='constant')

            return src1, src2

        except Exception as e:
            logger.error(f"SepReformer separation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return audio_segment, audio_segment


class MetisTSESeparator:
    """
    Wrapper for Amphion Metis Target Speaker Extraction.
    """
    is_tse = True

    def __init__(
        self,
        repo_dir,
        ckpt_dir=None,
        device="cuda:0",
        n_timesteps=10,
        guidance_cfg=0.0,
    ):
        from huggingface_hub import snapshot_download

        repo_dir = Path(repo_dir).expanduser().resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"Metis repo_dir not found: {repo_dir}. Clone open-mmlab/Amphion first."
            )
        _ensure_metis_repo_layout(repo_dir)

        ckpt_dir = Path(ckpt_dir).expanduser().resolve() if ckpt_dir else repo_dir / "models" / "tts" / "metis" / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.repo_dir = repo_dir
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.n_timesteps = int(n_timesteps)
        self.guidance_cfg = float(guidance_cfg)
        self.sample_rate = 24000

        print(f"[Metis] Initializing Target Speaker Extraction model on {device}")
        original_cwd = os.getcwd()
        original_sys_path = sys.path.copy()
        cleared_modules = {}
        original_work_dir = os.environ.get("WORK_DIR")
        try:
            _prepend_import_path(repo_dir)
            for module_name in list(sys.modules.keys()):
                if module_name == "models" or module_name.startswith("models.") or module_name == "utils" or module_name.startswith("utils."):
                    cleared_modules[module_name] = sys.modules[module_name]
                    del sys.modules[module_name]

            os.chdir(repo_dir)
            os.environ["WORK_DIR"] = str(repo_dir)
            _ensure_pkgutil_impimporter_compat()
            _ensure_langsegment_compat()
            from models.tts.metis.metis import Metis
            from utils.util import load_config

            metis_cfg = load_config(str(repo_dir / "models" / "tts" / "metis" / "config" / "tse.json"))
            base_ckpt_dir = snapshot_download(
                "amphion/metis",
                repo_type="model",
                local_dir=str(ckpt_dir),
                allow_patterns=["metis_base/model.safetensors"],
            )
            lora_ckpt_dir = snapshot_download(
                "amphion/metis",
                repo_type="model",
                local_dir=str(ckpt_dir),
                allow_patterns=["metis_tse/metis_tse_lora_32.safetensors"],
            )
            adapter_ckpt_dir = snapshot_download(
                "amphion/metis",
                repo_type="model",
                local_dir=str(ckpt_dir),
                allow_patterns=["metis_tse/metis_tse_lora_32_adapter.safetensors"],
            )

            self.model = Metis(
                base_ckpt_path=str(Path(base_ckpt_dir) / "metis_base" / "model.safetensors"),
                lora_ckpt_path=str(Path(lora_ckpt_dir) / "metis_tse" / "metis_tse_lora_32.safetensors"),
                adapter_ckpt_path=str(Path(adapter_ckpt_dir) / "metis_tse" / "metis_tse_lora_32_adapter.safetensors"),
                cfg=metis_cfg,
                device=device,
                model_type="tse",
            )
            print("[Metis] Target Speaker Extraction model initialization complete!")
        except Exception as e:
            if logger:
                logger.error(f"Failed to initialize Metis TSE: {e}")
            raise
        finally:
            os.chdir(original_cwd)
            if original_work_dir is None:
                os.environ.pop("WORK_DIR", None)
            else:
                os.environ["WORK_DIR"] = original_work_dir
            sys.path = original_sys_path
            for module_name in list(sys.modules.keys()):
                if module_name == "models" or module_name.startswith("models.") or module_name == "utils" or module_name.startswith("utils."):
                    del sys.modules[module_name]
            for module_name, module_obj in cleared_modules.items():
                sys.modules[module_name] = module_obj

    def _match_length(self, audio_segment, target_length):
        audio_segment = np.asarray(audio_segment, dtype=np.float32).reshape(-1)
        if len(audio_segment) > target_length:
            return audio_segment[:target_length]
        if len(audio_segment) < target_length:
            return np.pad(audio_segment, (0, target_length - len(audio_segment)), mode="constant")
        return audio_segment

    def separate_target(self, mixed_audio, reference_audio, sample_rate):
        """
        Extract the target speaker using a clean reference segment as prompt audio.
        """
        import tempfile
        import soundfile as sf

        mix_path = None
        ref_path = None
        try:
            target_length = len(mixed_audio)
            if sample_rate != self.sample_rate:
                mixed_model_sr = librosa.resample(mixed_audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                ref_model_sr = librosa.resample(reference_audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            else:
                mixed_model_sr = mixed_audio
                ref_model_sr = reference_audio

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as mix_file, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
                mix_path = mix_file.name
                ref_path = ref_file.name

            sf.write(mix_path, mixed_model_sr, self.sample_rate)
            sf.write(ref_path, ref_model_sr, self.sample_rate)

            original_cwd = os.getcwd()
            try:
                os.chdir(self.repo_dir)
                extracted_audio = self.model(
                    prompt_speech_path=ref_path,
                    source_speech_path=mix_path,
                    cfg=self.guidance_cfg,
                    n_timesteps=self.n_timesteps,
                    model_type="tse",
                )
            finally:
                os.chdir(original_cwd)

            if sample_rate != self.sample_rate:
                extracted_audio = librosa.resample(extracted_audio, orig_sr=self.sample_rate, target_sr=sample_rate)
            return self._match_length(extracted_audio, target_length)
        except Exception as e:
            if logger:
                logger.error(f"Metis TSE separation failed: {e}")
            return mixed_audio
        finally:
            for path in (mix_path, ref_path):
                if path and os.path.exists(path):
                    os.remove(path)


class SRCorrNetSeparator:
    """
    Class that loads the SR-CorrNet-SS model and can perform inference.
    """
    is_tse = False

    def __init__(self, model_id="shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk", device="cuda:0"):
        """
        Initialize and load the SR-CorrNet-SS model from Hugging Face Hub.

        Args:
            model_id: Hugging Face model id
            device: torch device (cuda/cpu)
        """
        self.device = device
        self.sample_rate = 8000 # Default sample rate for WSJ0 models

        if logger:
            logger.info(f"[SR-CorrNet-SS] Initializing on device: {self.device}")
        else:
            print(f"[SR-CorrNet-SS] Initializing on device: {self.device}")

        try:
            from sr_corrnet import SSInference
            self.model = SSInference.from_pretrained(model_id, device=device)
            if logger:
                logger.info("[SR-CorrNet-SS] Model initialization complete!")
            else:
                print("[SR-CorrNet-SS] Model initialization complete!")
        except ImportError as e:
            msg = "Missing sr_corrnet library. Please install via: pip install -e '.[hub]' in the SR_CorrNet_SS repo"
            if logger: logger.error(msg)
            else: print(msg)
            raise e
        except Exception as e:
            if logger: logger.error(f"Failed to initialize SR-CorrNet-SS: {e}")
            else: print(f"Failed to initialize SR-CorrNet-SS: {e}")
            raise

    def separate(self, audio_segment, sample_rate):
        """
        Perform audio source separation using SR-CorrNet-SS.

        Args:
            audio_segment (np.ndarray): Audio segment to separate
            sample_rate (int): Audio sample rate

        Returns:
            tuple: (separated_audio_1, separated_audio_2) as numpy arrays
        """
        try:
            target_length = len(audio_segment)

            # Resample to model sample rate if needed
            if sample_rate != self.sample_rate:
                audio_model_sr = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=self.sample_rate)
            else:
                audio_model_sr = audio_segment

            # Prepare tensor: model expects (channels, samples). We use 1 channel.
            waveform_tensor = torch.tensor(audio_model_sr, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Inference
            with torch.inference_mode():
                result = self.model.process_waveform(waveform_tensor, n_spks=torch.tensor(2))
            
            srcs = result["waveforms"]
            src1 = srcs[0].cpu().numpy()
            src2 = srcs[1].cpu().numpy()

            # Resample back to original sample rate if needed
            if sample_rate != self.sample_rate:
                src1 = librosa.resample(src1, orig_sr=self.sample_rate, target_sr=sample_rate)
                src2 = librosa.resample(src2, orig_sr=self.sample_rate, target_sr=sample_rate)

            # Match length exactly to original
            def match_len(src, target):
                if len(src) > target: return src[:target]
                elif len(src) < target: return np.pad(src, (0, target - len(src)), mode='constant')
                return src

            src1 = match_len(src1, target_length)
            src2 = match_len(src2, target_length)

            return src1, src2

        except Exception as e:
            if logger:
                logger.error(f"SR-CorrNet-SS separation failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            return audio_segment, audio_segment


@time_logger
def identify_speaker_with_embedding(audio_segment, sample_rate, reference_embeddings, speaker_labels, embedding_model, device):
    """
    Identify which speaker an audio segment belongs to using speaker embeddings.

    Args:
        audio_segment (np.ndarray): Audio segment to identify
        sample_rate (int): Sample rate of the audio
        reference_embeddings (dict): Dictionary of {speaker_label: embedding_tensor}
        speaker_labels (list): List of possible speaker labels
        embedding_model: Pre-loaded pyannote embedding model
        device: torch device (cuda/cpu)

    Returns:
        tuple: (best_speaker_label or None, best_similarity_score)
    """

    embedding = _extract_embedding_from_audio(audio_segment, sample_rate, embedding_model, device)
    if embedding is None:
        logger.debug("Speaker identification: no embedding extracted")
        return None, -1.0

    # Compare with reference embeddings using cosine similarity
    best_speaker = None
    best_similarity = -1.0

    for speaker_label in speaker_labels:
        if speaker_label in reference_embeddings:
            ref_embedding = _as_embedding_tensor(reference_embeddings[speaker_label], device)
            if ref_embedding is None:
                continue
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embedding,
                ref_embedding,
                dim=0
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_label

    logger.debug(f"Speaker identification: {best_speaker} (similarity: {best_similarity:.3f})")
    return best_speaker, best_similarity


def _as_embedding_tensor(value, device):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return None
    if tensor.ndim > 1:
        tensor = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)
    else:
        tensor = tensor.reshape(-1)
    norm = torch.linalg.vector_norm(tensor)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        return None
    return tensor / norm


def _extract_embedding_from_audio(audio_segment, sample_rate, embedding_model, device):
    if embedding_model is None:
        return None
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio_segment
    if len(audio_16k) == 0:
        return None

    audio_tensor = torch.as_tensor(audio_16k, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        embedding = embedding_model(audio_tensor)
    return _as_embedding_tensor(embedding, device)


def _build_reference_speaker_assets(
    segment_list,
    audio,
    embedding_model=None,
    device="cuda",
    extra_overlap_segments=None,
):
    import pandas as pd

    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    reference_embeddings = {}
    reference_audios = {}
    reference_reports = {}
    speaker_order = list(dict.fromkeys(str(seg["speaker"]) for seg in segment_list))

    frame_rows = [
        {
            "speaker": str(seg["speaker"]),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
        }
        for seg in segment_list
    ]
    for seg in extra_overlap_segments or []:
        frame_rows.append(
            {
                "speaker": str(seg.get("speaker", "UNKNOWN_MICRO_OVERLAP")),
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            }
        )
    segments_df = pd.DataFrame(frame_rows)

    for speaker in speaker_order:
        candidates = select_speaker_reference_candidates(
            segments_df,
            speaker,
            min_segment_duration=2.0,
        )
        if not candidates:
            reference_reports[speaker] = {"segment_count": 0, "quality_weight": 0.0, "segments": []}
            continue
        reference_reports[speaker] = {
            "segment_count": 1,
            "quality_weight": round(float(candidates[0].get("weight", 0.0)), 6),
            "segments": [
                {
                    "start": round(float(candidates[0]["start"]), 3),
                    "end": round(float(candidates[0]["end"]), 3),
                    "duration": round(float(candidates[0].get("duration", 0.0)), 3),
                    "type": str(candidates[0].get("type", "unknown")),
                    "weight": round(float(candidates[0].get("weight", 0.0)), 3),
                    "overlap_ratio": round(float(candidates[0].get("overlap_ratio", 0.0)), 3),
                }
            ],
        }

        reference_candidate = candidates[0]
        ref_start = int(float(reference_candidate["start"]) * sample_rate)
        ref_end = int(float(reference_candidate["end"]) * sample_rate)
        reference_audios[speaker] = waveform[ref_start:ref_end]

        if embedding_model is None:
            continue

        def embedding_fn(start, end):
            start_frame = int(float(start) * sample_rate)
            end_frame = int(float(end) * sample_rate)
            seg_audio = waveform[start_frame:end_frame]
            return _extract_embedding_from_audio(seg_audio, sample_rate, embedding_model, device)

        embedding, report = build_weighted_reference_embedding(
            candidates,
            embedding_fn=embedding_fn,
            max_segments_per_speaker=3,
        )
        reference_reports[speaker] = report
        embedding_tensor = _as_embedding_tensor(embedding, device)
        if embedding_tensor is not None:
            reference_embeddings[speaker] = embedding_tensor

    return reference_embeddings, reference_audios, reference_reports


@time_logger
def process_overlapping_segments_with_separation(segment_list, audio, overlap_threshold=1.0,
                                                 separator=None, embedding_model=None, device="cuda",
                                                 micro_overlap_candidates=None,
                                                 micro_overlap_padding=0.35):
    """
    Process overlapping segments by separating them with blind separation or TSE.
    [Updated] Matches the volume of separated audio to the original overlap audio to prevent volume jumps.

    Args:
        segment_list: List of segments
        audio: Audio dictionary
        overlap_threshold: Overlap threshold
        separator: Pre-loaded separator object
        embedding_model: Pre-loaded pyannote embedding model for blind separation
        device: torch device (cuda/cpu)
    """
    if separator is None:
        logger.warning("Separator not provided, skipping separation")
        return audio, segment_list

    is_tse_separator = getattr(separator, 'is_tse', False)
    if embedding_model is None and not is_tse_separator:
        logger.warning("Embedding model not provided for blind separation, skipping separation")
        return audio, segment_list

    if is_tse_separator:
        separator_name = "TSE"
    elif type(separator).__name__ == "SRCorrNetSeparator":
        separator_name = "SR-CorrNet-SS"
    else:
        separator_name = "SepReformer"
    micro_overlap_candidates = list(micro_overlap_candidates or [])
    logger.info(
        f"Processing overlapping segments with {separator_name} "
        f"(threshold: {overlap_threshold}s, micro_candidates: {len(micro_overlap_candidates)})"
    )

    # -------------------------------------------------------------------------
    # [Added] Volume matching helper functions
    # -------------------------------------------------------------------------
    def get_non_overlap_rms(segment, waveform, sample_rate, overlapping_pairs):
        """
        Calculate the RMS energy from the non-overlap regions of a segment.

        Args:
            segment: Segment dictionary
            waveform: Full audio waveform
            sample_rate: Sample rate
            overlapping_pairs: List of overlapping pairs

        Returns:
            float: RMS energy of the non-overlap region (None if cannot be calculated)
        """
        seg_start = segment['start']
        seg_end = segment['end']

        # Find all overlapping regions for this segment
        overlap_regions = []
        for pair in overlapping_pairs:
            if pair['seg1'] == segment or pair['seg2'] == segment:
                overlap_regions.append((pair['overlap_start'], pair['overlap_end']))

        if not overlap_regions:
            # If no overlap, use the entire segment
            start_frame = int(seg_start * sample_rate)
            end_frame = int(seg_end * sample_rate)
            seg_audio = waveform[start_frame:end_frame]
        else:
            # Extract only non-overlap regions
            non_overlap_parts = []
            overlap_regions.sort()

            # From segment start to first overlap
            if overlap_regions[0][0] > seg_start:
                start_frame = int(seg_start * sample_rate)
                end_frame = int(overlap_regions[0][0] * sample_rate)
                non_overlap_parts.append(waveform[start_frame:end_frame])

            # Regions between overlaps
            for i in range(len(overlap_regions) - 1):
                start_frame = int(overlap_regions[i][1] * sample_rate)
                end_frame = int(overlap_regions[i+1][0] * sample_rate)
                if end_frame > start_frame:
                    non_overlap_parts.append(waveform[start_frame:end_frame])

            # From last overlap to segment end
            if overlap_regions[-1][1] < seg_end:
                start_frame = int(overlap_regions[-1][1] * sample_rate)
                end_frame = int(seg_end * sample_rate)
                non_overlap_parts.append(waveform[start_frame:end_frame])

            if not non_overlap_parts:
                return None

            seg_audio = np.concatenate(non_overlap_parts)

        if len(seg_audio) == 0:
            return None

        rms = np.sqrt(np.mean(seg_audio**2))
        return rms if rms > 1e-10 else None

    def match_target_amplitude(source_wav, target_rms):
        """
        Match the volume (RMS) of source_wav to the target_rms energy level.

        Args:
            source_wav: Audio waveform to adjust
            target_rms: Target RMS energy value
        """
        # Epsilon to prevent division by zero
        epsilon = 1e-10

        # Calculate RMS (Root Mean Square) energy
        src_rms = np.sqrt(np.mean(source_wav**2))

        if src_rms < epsilon or target_rms is None or target_rms < epsilon:
            return source_wav

        # Calculate ratio (how much larger/smaller the target is compared to the source)
        gain = target_rms / (src_rms + epsilon)

        # Apply gain
        adjusted_wav = source_wav * gain

        # Prevent clipping (-1.0 to 1.0)
        return np.clip(adjusted_wav, -1.0, 1.0)

    def calculate_energy(audio_segment):
        """
        Calculate the energy of an audio segment.
        """
        return np.sum(audio_segment**2)
    # -------------------------------------------------------------------------

    # 1. Initialization: Only set the 'sepreformer' flag for all segments
    #    enhanced_audio is created later when needed (non-overlapping segments use original)
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    for seg in segment_list:
        if 'sepreformer' not in seg:
            seg['sepreformer'] = False

    # Detect overlapping segments
    overlapping_pairs = detect_overlapping_segments(segment_list, overlap_threshold)

    if not overlapping_pairs and not micro_overlap_candidates:
        logger.info("No overlapping segments found")
        return audio, segment_list

    logger.info(f"Found {len(overlapping_pairs)} overlapping segment pairs")

    reference_embeddings, reference_audios, reference_reports = _build_reference_speaker_assets(
        segment_list,
        audio,
        embedding_model=embedding_model,
        device=device,
        extra_overlap_segments=micro_overlap_candidates,
    )
    for speaker, report in reference_reports.items():
        logger.debug(f"Reference embedding candidates for {speaker}: {report}")

    # 2. Process overlap pairs
    for pair_idx, pair in enumerate(overlapping_pairs):
        overlap_start = pair['overlap_start']
        overlap_end = pair['overlap_end']
        seg1 = pair['seg1']
        seg2 = pair['seg2']

        seg1_speaker = seg1['speaker']
        seg2_speaker = seg2['speaker']

        # Extract overlapping audio (Original Mixture)
        start_frame = int(overlap_start * sample_rate)
        end_frame = int(overlap_end * sample_rate)
        overlap_audio = waveform[start_frame:end_frame]

        if is_tse_separator:
            # TARGET SPEAKER EXTRACTION
            ref_audio1 = reference_audios.get(seg1_speaker)
            ref_audio2 = reference_audios.get(seg2_speaker)
            
            if ref_audio1 is None or ref_audio2 is None:
                logger.warning(f"  Missing reference audio for {seg1_speaker} or {seg2_speaker}, skipping TSE.")
                seg1_part = overlap_audio
                seg2_part = overlap_audio
                assignment_method = "skipped"
            else:
                logger.info(f"  Extracting {seg1_speaker} from mixture using TSE...")
                seg1_part = separator.separate_target(overlap_audio, ref_audio1, sample_rate)
                
                logger.info(f"  Extracting {seg2_speaker} from mixture using TSE...")
                seg2_part = separator.separate_target(overlap_audio, ref_audio2, sample_rate)
                
                assignment_method = "tse_direct"
                speaker1_identity = seg1_speaker
                speaker2_identity = seg2_speaker
        else:
            # BLIND SOURCE SEPARATION (SepReformer)
            separated_src1, separated_src2 = separator.separate(
                overlap_audio, sample_rate
            )

            # Identify speakers with embedding matching
            speaker1_identity, similarity1 = identify_speaker_with_embedding(
                separated_src1, sample_rate, reference_embeddings, [seg1_speaker, seg2_speaker], embedding_model, device
            )
            speaker2_identity, similarity2 = identify_speaker_with_embedding(
                separated_src2, sample_rate, reference_embeddings, [seg1_speaker, seg2_speaker], embedding_model, device
            )

            # ---------------------------------------------------------------------
            # Fallback handling when embedding matching fails
            # ---------------------------------------------------------------------
            assignment_method = "embedding"

            # Case 1: Embedding matching succeeded and the two sources matched to different speakers
            if (speaker1_identity is not None and speaker2_identity is not None and
                speaker1_identity != speaker2_identity):
                if speaker1_identity == seg1_speaker:
                    seg1_part = separated_src1
                    seg2_part = separated_src2
                else:
                    seg1_part = separated_src2
                    seg2_part = separated_src1
                logger.info(f"  Speaker assignment by embedding: src1={speaker1_identity} ({similarity1:.3f}), src2={speaker2_identity} ({similarity2:.3f})")

            # Case 2: Embedding matching failed or both sources matched to the same speaker -> energy-based fallback
            else:
                assignment_method = "energy_fallback"
                logger.warning(f"  Embedding matching failed or ambiguous (src1={speaker1_identity}, src2={speaker2_identity})")
                logger.info(f"  Using energy-based fallback for speaker assignment")

                # Assign the higher-energy source to the longer segment based on segment duration
                seg1_duration = seg1['end'] - seg1['start']
                seg2_duration = seg2['end'] - seg2['start']

                energy1 = calculate_energy(separated_src1)
                energy2 = calculate_energy(separated_src2)

                # Assign the higher-energy source to the longer segment
                if seg1_duration >= seg2_duration:
                    if energy1 >= energy2:
                        seg1_part = separated_src1
                        seg2_part = separated_src2
                    else:
                        seg1_part = separated_src2
                        seg2_part = separated_src1
                else:
                    if energy2 >= energy1:
                        seg1_part = separated_src2
                        seg2_part = separated_src1
                    else:
                        seg1_part = separated_src1
                        seg2_part = separated_src2

                logger.info(f"  Energy-based assignment: seg1_dur={seg1_duration:.2f}s, seg2_dur={seg2_duration:.2f}s, "
                           f"energy1={energy1:.2e}, energy2={energy2:.2e}")
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # [Modified] Apply volume correction (match to the non-overlap region RMS of each segment)
        # ---------------------------------------------------------------------
        seg1_target_rms = get_non_overlap_rms(seg1, waveform, sample_rate, overlapping_pairs)
        seg2_target_rms = get_non_overlap_rms(seg2, waveform, sample_rate, overlapping_pairs)

        # Fallback: if non-overlap RMS cannot be calculated, use half of the overlap region's RMS
        overlap_rms = np.sqrt(np.mean(overlap_audio**2))
        if seg1_target_rms is None:
            seg1_target_rms = overlap_rms * 0.7  # Slightly conservative
            logger.debug(f"   No non-overlap region for seg1, using fallback RMS")
        if seg2_target_rms is None:
            seg2_target_rms = overlap_rms * 0.7
            logger.debug(f"   No non-overlap region for seg2, using fallback RMS")

        logger.debug(f"   Adjusting volume for overlap {pair_idx+1} (method: {assignment_method})...")
        logger.debug(f"   seg1 target RMS: {seg1_target_rms:.6f}, seg2 target RMS: {seg2_target_rms:.6f}")

        seg1_part = match_target_amplitude(seg1_part, seg1_target_rms)
        seg2_part = match_target_amplitude(seg2_part, seg2_target_rms)
        # ---------------------------------------------------------------------

        # Temporarily store separated audio (used later for full reconstruction)
        if 'separated_regions' not in seg1:
            seg1['separated_regions'] = []
        if 'separated_regions' not in seg2:
            seg2['separated_regions'] = []

        seg1['separated_regions'].append({
            'start': overlap_start,
            'end': overlap_end,
            'audio': seg1_part
        })
        seg2['separated_regions'].append({
            'start': overlap_start,
            'end': overlap_end,
            'audio': seg2_part
        })
        seg1['sepreformer'] = True
        seg2['sepreformer'] = True
        seg1['is_separated'] = True
        seg2['is_separated'] = True

        logger.info(f"  ✓ Stored separated audio for Seg1 and Seg2 (overlap: {overlap_start:.2f}-{overlap_end:.2f})")

    # 2b. Suppress short overlap-only candidates inside the main speaker segment.
    if micro_overlap_candidates and not is_tse_separator:
        logger.warning("Micro overlap cleanup requires a TSE separator; skipping micro overlap candidates.")

    if micro_overlap_candidates and is_tse_separator:
        segment_by_index = {str(seg.get("index")): seg for seg in segment_list if seg.get("index") is not None}
        padding = max(0.0, float(micro_overlap_padding))
        audio_start = 0.0
        audio_end = len(waveform) / sample_rate if sample_rate else 0.0

        def find_target_segment(candidate):
            target_index = candidate.get("target_index")
            if target_index is not None and str(target_index) in segment_by_index:
                return segment_by_index[str(target_index)]

            target_speaker = candidate.get("target_speaker")
            cand_start = float(candidate.get("overlap_start", candidate.get("start", 0.0)))
            cand_end = float(candidate.get("overlap_end", candidate.get("end", cand_start)))
            for segment in segment_list:
                if target_speaker and str(segment.get("speaker")) != str(target_speaker):
                    continue
                if float(segment["start"]) <= cand_start and cand_end <= float(segment["end"]):
                    return segment
            return None

        for candidate_idx, candidate in enumerate(micro_overlap_candidates):
            if not bool(candidate.get("is_overlap_candidate", True)):
                continue
            target_seg = find_target_segment(candidate)
            if target_seg is None:
                logger.warning(f"  Missing target segment for micro overlap candidate {candidate_idx}, skipping.")
                continue

            target_speaker = str(candidate.get("target_speaker") or target_seg.get("speaker"))
            ref_audio = reference_audios.get(target_speaker)
            if ref_audio is None:
                logger.warning(f"  Missing reference audio for {target_speaker}, skipping micro overlap cleanup.")
                continue

            base_start = float(candidate.get("overlap_start", candidate.get("start", target_seg["start"])))
            base_end = float(candidate.get("overlap_end", candidate.get("end", base_start)))
            region_start = max(float(target_seg["start"]), audio_start, base_start - padding)
            region_end = min(float(target_seg["end"]), audio_end, base_end + padding)
            if region_end <= region_start:
                continue

            start_frame = int(region_start * sample_rate)
            end_frame = int(region_end * sample_rate)
            mixed_region = waveform[start_frame:end_frame]
            if len(mixed_region) == 0:
                continue

            logger.info(
                f"  Cleaning micro overlap for {target_speaker}: "
                f"{region_start:.2f}-{region_end:.2f} "
                f"(interferer={candidate.get('speaker')})"
            )
            target_part = separator.separate_target(mixed_region, ref_audio, sample_rate)
            target_part = match_target_amplitude(
                target_part,
                get_non_overlap_rms(target_seg, waveform, sample_rate, overlapping_pairs) or np.sqrt(np.mean(mixed_region**2)),
            )

            if 'separated_regions' not in target_seg:
                target_seg['separated_regions'] = []
            target_seg['separated_regions'].append({
                'start': region_start,
                'end': region_end,
                'audio': target_part
            })
            target_seg['sepreformer'] = True
            target_seg['is_separated'] = True
            target_seg['micro_overlap_suppressed'] = True
            target_seg.setdefault('micro_overlap_regions', []).append({
                'start': round(region_start, 3),
                'end': round(region_end, 3),
                'interferer_speaker': candidate.get('speaker'),
                'candidate_start': round(float(candidate.get('start', base_start)), 3),
                'candidate_end': round(float(candidate.get('end', base_end)), 3),
            })

    # -------------------------------------------------------------------------
    # 3. After processing all overlaps, reconstruct enhanced_audio for each segment
    # -------------------------------------------------------------------------
    logger.info("Reconstructing enhanced_audio for all segments...")

    for seg in segment_list:
        seg_start = seg['start']
        seg_end = seg['end']
        seg_start_frame = int(seg_start * sample_rate)
        seg_end_frame = int(seg_end * sample_rate)

        if 'separated_regions' in seg and seg['separated_regions']:
            # This segment had overlap -> reconstruction needed
            separated_regions = sorted(seg['separated_regions'], key=lambda x: x['start'])

            # Split the segment into parts in chronological order
            parts = []
            current_time = seg_start

            for region in separated_regions:
                region_start = region['start']
                region_end = region['end']
                region_audio = region['audio']

                # 1) Original audio between current_time and region_start (non-overlapping part)
                if current_time < region_start:
                    start_f = int(current_time * sample_rate)
                    end_f = int(region_start * sample_rate)
                    original_part = waveform[start_f:end_f]
                    parts.append(original_part)
                    logger.debug(f"  Seg [{seg_start:.2f}-{seg_end:.2f}]: Added original audio [{current_time:.2f}-{region_start:.2f}] ({len(original_part)} samples)")

                # 2) Separated audio from region_start to region_end
                expected_region_length = int((region_end - region_start) * sample_rate)
                actual_region_length = len(region_audio)
                parts.append(region_audio)
                logger.debug(f"  Seg [{seg_start:.2f}-{seg_end:.2f}]: Added separated audio [{region_start:.2f}-{region_end:.2f}] ({actual_region_length} samples, expected: {expected_region_length})")

                current_time = region_end

            # 3) Original audio from after the last region to seg_end
            if current_time < seg_end:
                start_f = int(current_time * sample_rate)
                end_f = seg_end_frame
                original_part = waveform[start_f:end_f]
                parts.append(original_part)
                logger.debug(f"  Seg [{seg_start:.2f}-{seg_end:.2f}]: Added original audio [{current_time:.2f}-{seg_end:.2f}] ({len(original_part)} samples)")

            # Concatenate all parts
            seg['enhanced_audio'] = np.concatenate(parts) if parts else waveform[seg_start_frame:seg_end_frame]

            # Length verification
            expected_length = seg_end_frame - seg_start_frame
            actual_length = len(seg['enhanced_audio'])
            if abs(expected_length - actual_length) > 1:
                logger.warning(f"  ⚠️ Seg [{seg_start:.2f}-{seg_end:.2f}]: Length mismatch! Expected {expected_length}, got {actual_length} (diff: {actual_length - expected_length})")
            else:
                logger.debug(f"  ✓ Seg [{seg_start:.2f}-{seg_end:.2f}]: Length verified ({actual_length} samples)")

            # Clean up temporary data
            del seg['separated_regions']

        else:
            # Non-overlapping segment -> use original as is
            seg['enhanced_audio'] = waveform[seg_start_frame:seg_end_frame].copy()

    logger.info("Enhanced audio reconstruction complete!")

    return audio, segment_list
