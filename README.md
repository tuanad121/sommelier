# vi-Sommelier

A Vietnamese fork of NAVER Cloud's [Sommelier](https://github.com/naver-ai/sommelier) audio-processing pipeline. Takes YouTube URLs of Vietnamese podcasts and produces diarized ASR output, per-turn audio, and quality sidecars, ready to be shipped to HuggingFace as a training dataset.

**Upstream**: `naver-ai/sommelier` (ASR MoE + Sortformer diarization + Demucs/PANNs + WhisperX, originally tuned for Korean and English)
**This fork**: Vietnamese ASR ensemble + VN whisper initial-prompt + whole-episode source mode + a batch of post-pipeline quality sidecars.

---

## What's different from upstream

- **`--lang vi` ASR ensemble** — Whisper + [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large) + [ChunkFormer-VN](https://huggingface.co/khanhld/chunkformer-ctc-large-vie), voted by ROVER (see `podcast-pipeline/utils/asr_ensemble.py`). Parakeet / Canary are swapped out for VN runs.
- **VN Whisper initprompt** — Vietnamese backchannel tokens injected to stabilize decoding of conversational hesitation sounds.
- **Operator tooling** — TSV-driven source management, idempotent batch pulls, HF push automation. See `doc/handoff_pipeline_operator.md`.
- **Sidecar stages** — four post-pipeline passes that add quality signals to each clip: BGM detection (PANNs), speaker attribution (ECAPA), force-alignment (wav2vec2-vi-250h), and gated MossFormer2 overlap separation.
- **Downstream extractor** — `scripts/build_training_conversations.py` turns per-clip output into contiguous dyadic SFT conversations.

---

## Pipeline shape

```
TSV of YouTube URLs
  └─▶ pick_clips.py            (yt-dlp pull → data/raw/<label>.wav + .meta.json)
        └─▶ podcast-pipeline/main_original_ASR_MoE.py --lang vi
              │  (VAD → Sortformer diarization → ASR MoE → Demucs → WhisperX)
              └─▶ data/raw/_final/<config>/<label>/
                    ├── <label>.json            (segments + 3-way ASR + diarization)
                    └── NNNNN_SPEAKER_XX.mp3   (per-turn audio, Demucs-cleaned)
                      └─▶ 4 sidecar stages (sortformer probs, bgm/attribution,
                      │   force-align, gated MossFormer2)
                      └─▶ normalize_to_hf_jsonl.py
                            │  (drops: phantom / misattributed / has_bgm / multi_speaker)
                            └─▶ <label>.jsonl                     (per-kept-turn)
                                  └─▶ upload_to_hf.py             (push to your HF dataset)
                                        └─▶ build_training_conversations.py
                                              └─▶ dyadic SFT conversation JSONs
```

---

## Quickstart

### Environment

```bash
conda create -n sommelier python=3.10 -y
conda activate sommelier
conda install -c conda-forge ffmpeg -y

cd podcast-pipeline
pip install torch==2.7.1 torchaudio==2.7.1
pip install -r requirements.txt
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1
```

Or with `uv` (preferred on cluster — see handoff doc §3).

### Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python main_original_ASR_MoE.py --help
```

### Smoke test (one Vietnamese clip)

```bash
# Pull a short clip (edit the row with your target URL + 0-120s window)
echo -e "smoke_test\thttps://youtube.com/watch?v=XXXXXXXXXXX\t0\t120\tsmoke\tpodcast_interview" \
  > data/clips_to_pull.tsv
python scripts/pick_clips.py --tsv data/clips_to_pull.tsv --out data/raw --workers 1

# Run the pipeline (requires a GPU)
cd podcast-pipeline
export LD_LIBRARY_PATH=$HOME/.local/cudnn8/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python main_original_ASR_MoE.py \
  --input_folder_path ../data/raw \
  --lang vi --vad --dia3 --ASRMoE --whisperx_word_timestamps --initprompt \
  --demucs --sepreformer --no-qwen3omni --merge_gap 2.0
```

Expected output: `../data/raw/_final/-sepreformer-True-demucs-True-vad-True-diaModel-dia3-initPrompt-True-merge_gap-2.0-seg_th-0.15-cl_min-10-cl-th-0.5-LLM-case_2/smoke_test/smoke_test.json` + per-turn mp3s.

### Running real batches

**Do not improvise the operator workflow.** The TSV format, sidecar launch order, dedup cache, HF staging, and storage discipline rules all live in **`doc/handoff_pipeline_operator.md`**. Read it before adding a batch.

---

## Repo layout

| Path | What's there |
|---|---|
| `podcast-pipeline/` | Core pipeline (ASR MoE, diarization, Demucs, WhisperX). Entry: `main_original_ASR_MoE.py`. |
| `podcast-pipeline/utils/asr_ensemble.py` | ROVER voting (`RoverEnsembler`) over Whisper + PhoWhisper + ChunkFormer-VN. |
| `scripts/pick_clips.py` | yt-dlp batch puller. TSV in, WAV + meta.json out. |
| `scripts/build_training_conversations.py` | Per-clip output → dyadic SFT conversation JSONs. |
| `scripts/normalize_to_hf_jsonl.py` | Applies drop filters (phantom / misattributed / BGM / multi-speaker). |
| `scripts/launch_*.sh` | Sidecar stage orchestrators (sortformer, BGM+attribution, force-align, gated MF2, normalize). |
| `scripts/upload_to_hf.py` | HF dataset push (edit `REPO_ID` before use). |
| `doc/handoff_pipeline_operator.md` | **The operator manual. Read first.** |
| `data/clips_to_pull.tsv` | Source-of-truth list of what gets pulled. |
| `data/processed_urls.tsv` | Dedup cache. |

---

## Where to go next

- **Running the pipeline on real batches** → `doc/handoff_pipeline_operator.md`
- **Original pipeline architecture / Korean+English config** → upstream [`naver-ai/sommelier`](https://github.com/naver-ai/sommelier)
- **Design decisions behind the Vietnamese ASR ensemble choice** → ask Tuan

---

## License

MIT (same as upstream). See `LICENSE` and `NOTICE` for the Microsoft dnsmos CC-BY-4.0 exception that comes from upstream.
