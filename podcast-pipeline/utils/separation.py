# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT
"""
Audio source separation utilities for podcast pipeline.
Includes SepReformer-based speaker separation and overlapping segment processing.
"""

import os
import sys
import numpy as np
import librosa
import torch
from utils.logger import time_logger
from utils.diarization import detect_overlapping_segments

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


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


class ClearVoiceTSESeparator:
    """
    Wrapper for ModelScope ClearerVoice (MossFormer2) Target Speaker Extraction.
    """
    def __init__(self, model_name='damo/speech_mossformer2_tse_16k', device="cuda"):
        from clearvoice import ClearVoice
        import os
        
        self.device = device
        self.is_tse = True # Flag to indicate this is a Target Speaker Extraction model
        
        # ModelScope requires some ENV vars to use custom cache dirs if needed, but defaults are usually fine
        print(f"[ClearVoice] Initializing Target Speaker Extraction model on {device}: {model_name}")
        
        # In Kaggle/Linux, the clearvoice pipeline uses 'task' and 'model_names'
        try:
            self.myClearVoice = ClearVoice(task='target_speaker_extraction', model_names=[model_name])
            print("[ClearVoice] Model initialization complete!")
        except Exception as e:
            logger.error(f"Failed to initialize ClearVoice TSE: {e}")
            raise e

    def separate_target(self, mixed_audio, reference_audio, sample_rate):
        """
        Perform Target Speaker Extraction.
        """
        import tempfile
        import soundfile as sf
        import librosa
        
        try:
            target_sr = 16000
            if sample_rate != target_sr:
                mixed_16k = librosa.resample(mixed_audio, orig_sr=sample_rate, target_sr=target_sr)
                ref_16k = librosa.resample(reference_audio, orig_sr=sample_rate, target_sr=target_sr)
            else:
                mixed_16k = mixed_audio
                ref_16k = reference_audio

            # Write to temp files since clearvoice pipeline expects paths
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_mix, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_ref, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
                mix_path = f_mix.name
                ref_path = f_ref.name
                out_path = f_out.name
                
            try:
                sf.write(mix_path, mixed_16k, target_sr)
                sf.write(ref_path, ref_16k, target_sr)
                
                # Run MossFormer2 TSE
                output_wav_dict = self.myClearVoice(input_path=mix_path, online_write=False, reference_path=ref_path)
                
                # output_wav_dict is usually a dict { 'mixture_name': audio_array }
                # But myClearVoice.write handles it
                self.myClearVoice.write(output_wav_dict, output_path=out_path)
                
                # Read back
                extracted_audio, _ = librosa.load(out_path, sr=sample_rate)
                
                # Length matching due to resample/saving
                target_length = len(mixed_audio)
                if len(extracted_audio) != target_length:
                    extracted_audio = np.pad(extracted_audio, (0, max(0, target_length - len(extracted_audio))))[:target_length]
                    
                return extracted_audio
            finally:
                if os.path.exists(mix_path): os.remove(mix_path)
                if os.path.exists(ref_path): os.remove(ref_path)
                if os.path.exists(out_path): os.remove(out_path)

        except Exception as e:
            if logger:
                logger.error(f"ClearVoice TSE separation failed: {e}")
            return mixed_audio


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

    # Extract embedding from audio segment
    # Resample to 16kHz if needed (pyannote expects 16kHz)
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio_segment

    # Convert to tensor
    audio_tensor = torch.tensor(audio_16k, dtype=torch.float32).unsqueeze(0).to(device)

    # Extract embedding
    with torch.inference_mode():
        embedding = embedding_model(audio_tensor)

    # Compare with reference embeddings using cosine similarity
    best_speaker = None
    best_similarity = -1.0

    for speaker_label in speaker_labels:
        if speaker_label in reference_embeddings:
            ref_embedding = reference_embeddings[speaker_label]
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embedding.mean(dim=1),
                ref_embedding.mean(dim=1),
                dim=0
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_label

    logger.debug(f"Speaker identification: {best_speaker} (similarity: {best_similarity:.3f})")
    return best_speaker, best_similarity


@time_logger
def process_overlapping_segments_with_separation(segment_list, audio, overlap_threshold=1.0,
                                                 separator=None, embedding_model=None, device="cuda"):
    """
    Process overlapping segments by separating them with SepReformer.
    [Updated] Matches the volume of separated audio to the original overlap audio to prevent volume jumps.

    Args:
        segment_list: List of segments
        audio: Audio dictionary
        overlap_threshold: Overlap threshold
        separator: Pre-loaded SepReformerSeparator object
        embedding_model: Pre-loaded pyannote embedding model
        device: torch device (cuda/cpu)
    """
    if separator is None:
        logger.warning("SepReformer separator not provided, skipping separation")
        return audio, segment_list

    if embedding_model is None:
        logger.warning("Embedding model not provided, skipping separation")
        return audio, segment_list

    logger.info(f"Processing overlapping segments with SepReformer (threshold: {overlap_threshold}s)")

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

    if not overlapping_pairs:
        logger.info("No overlapping segments found")
        return audio, segment_list

    logger.info(f"Found {len(overlapping_pairs)} overlapping segment pairs")

    # (Reference Embeddings extraction logic - kept as is)
    reference_embeddings = {}
    reference_audios = {}
    all_speakers = list(set([seg['speaker'] for seg in segment_list]))

    # ... (Reference Embedding extraction - kept as is) ...
    for speaker in all_speakers:
        speaker_segments = [seg for seg in segment_list if seg['speaker'] == speaker]
        for seg in speaker_segments:
            is_overlapping = any(pair['seg1'] == seg or pair['seg2'] == seg for pair in overlapping_pairs)
            if not is_overlapping and (seg['end'] - seg['start']) >= 2.0:
                start_frame = int(seg['start'] * sample_rate)
                end_frame = int(seg['end'] * sample_rate)
                seg_audio = waveform[start_frame:end_frame]
                if sample_rate != 16000:
                    seg_audio_16k = librosa.resample(seg_audio, orig_sr=sample_rate, target_sr=16000)
                else:
                    seg_audio_16k = seg_audio
                
                # STORE REFERENCE AUDIO FOR TSE
                reference_audios[speaker] = seg_audio_16k
                
                if embedding_model is not None:
                    seg_tensor = torch.tensor(seg_audio_16k, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        embedding = embedding_model(seg_tensor)
                    reference_embeddings[speaker] = embedding
                break

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

        if getattr(separator, 'is_tse', False):
            # TARGET SPEAKER EXTRACTION (ClearVoice)
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

        logger.info(f"  ✓ Stored separated audio for Seg1 and Seg2 (overlap: {overlap_start:.2f}-{overlap_end:.2f})")

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
