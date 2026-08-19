# vi-Sommelier full-duplex stereo — format reference

A Vietnamese full-duplex conversational corpus derived from the vi-Sommelier pipeline.
Follows the Sommelier data-processing philosophy (Jung et al., 2026) — Sortformer
diarization + Case-4 overlap separation + 3-way ROVER ASR + PANNs-gated Demucs BGM
removal — and is packaged for Moshi-style full-duplex training.

Ships alongside the upstream per-turn tabular artifacts (`<clip>.jsonl` on
`tuanamz/vi-sommelier-v0`); this doc covers only the FD stereo view.

## File layout

Each conversation is written twice, once per channel orientation. Orientations live in separate subdirectories because a single git tree cannot exceed 10000 files:

```
fd/
├── README.md
├── read_full_duplex.py                (canonical decoder)
├── build_full_duplex_2channel.py      (reproducer)
├── manifest.jsonl                     (slim index for the explorer)
├── oriA/                              # 2559 conversations × 2 files
│   ├── <conversation_id>__oriA.wav
│   └── <conversation_id>__oriA.json
└── oriB/                              # same, L/R swapped
    ├── <conversation_id>__oriB.wav
    └── <conversation_id>__oriB.json
```

If you only need one orientation (e.g. matching the Sommelier paper's fixed-channel convention), `snapshot_download(..., allow_patterns=["fd/oriA/*"])` gives you a self-contained subset.

The oriA/oriB pair covers the same underlying dialogue — B is the free
augmentation for channel-invariance (see *Design decisions* below). Treat
them as independent training samples.

## Audio contract

- **Sample rate:** 16 000 Hz
- **Channels:** 2 (stereo, interleaved)
- **Encoding:** signed 16-bit PCM
- **L channel:** carries `channels["L"]`'s audio during that speaker's turns, silence elsewhere
- **R channel:** carries `channels["R"]`'s audio during that speaker's turns, silence elsewhere
- **Duration:** exactly `duration_s` seconds; equals `turns[-1]["end"] - turns[0]["start"]` from the parent clip
- **Overlap:** genuine simultaneous speech (both speakers talking) is preserved — both channels have non-zero signal at the same sample. No cross-talk between channels.

## JSON schema

```jsonc
{
  "conversation_id":              "<clip>__runNN__oriA",  // matches the file stem
  "source_clip":                  "<clip>",              // parent clip on vi-sommelier-v0
  "source_url":                   "https://youtube.com/...",
  "source_channel":               "Have A Sip",
  "source_genre":                 "podcast",
  "duration_s":                   34.9,
  "n_turns":                      13,
  "unique_speakers":              ["SPEAKER_00", "SPEAKER_01"],
  "channels":                     { "L": "SPEAKER_01", "R": "SPEAKER_00" },
  "channel_of":                   { "SPEAKER_01": 0, "SPEAKER_00": 1 },
  "sample_rate":                  16000,
  "_original_clip_offset_s":      121.024,               // add to any turn start/end
                                                         // to recover parent-clip time
  "_original_clip_time_range":    [121.024, 155.924],

  "turns": [
    {
      "turn_id":       "00028",
      "turn_index":    28,                 // index in the parent <clip>.jsonl
      "speaker":       "SPEAKER_02",
      "channel":       0,                  // 0 = L, 1 = R
      "start":         0.000,              // seconds, stereo-WAV-relative
      "end":           2.300,
      "duration":      2.300,
      "audio_path":    "<clip>/00028_SPEAKER_02.mp3",  // per-turn mono in parent
      "text":          "Nếu vậy thì ba chất bốn cẳng cũng là từ động vật à.",
      "text_source":   "gold",             // gold | rover | whisper
      "flags":         { "is_overlap": true, ... },
      "words": [
        { "word": "nếu", "start": 0.000, "end": 0.100 },
        { "word": "vậy", "start": 0.100, "end": 0.240 },
        ...
      ]
    },
    ...
  ]
}
```

### Timestamp convention

**All timestamps in `turns[*].start/end` and `turns[*].words[*].start/end`
are zero-origin — they align directly to the sample index in the paired
stereo WAV.** No offset math in your dataloader.

To recover clip-time (i.e. seconds into the parent `<source_clip>`), add
`_original_clip_offset_s`:

```python
clip_time = stereo_time + meta["_original_clip_offset_s"]
```

This is only needed when cross-referencing against the raw pipeline
artifacts (`<clip>.jsonl`) — not for training.

## Reader — minimal usage

Both scripts referenced below ship inside this dataset repo alongside the audio (`fd/read_full_duplex.py`, `fd/build_full_duplex_2channel.py`):

```bash
# print all turns of one conversation, aligned to the stereo timeline
python read_full_duplex.py <conversation_id>__oriA.json

# also slice turn 3's mono audio out of the correct channel
python read_full_duplex.py <conversation_id>__oriA.json \
    --extract-turn 3 --out turn3.wav
```

Programmatic use (the two hot loops a trainer needs):

```python
import json, soundfile as sf

meta = json.load(open("conv__oriA.json"))
stereo, sr = sf.read("conv__oriA.wav", always_2d=True)   # shape (N, 2)

# 1. Iterate turns as (mono_audio, text, speaker, channel) tuples
for t in meta["turns"]:
    s = int(round(t["start"] * sr))
    e = int(round(t["end"]   * sr))
    mono = stereo[s:e, t["channel"]]     # only the speaking channel
    yield mono, t["text"], t["speaker"], t["channel"]

# 2. Feed both channels as-is to a full-duplex model
model.forward(stereo)                     # (N, 2) → whatever FD arch expects
```

## Filter profile

Conversations are extracted with the FD-loose profile (see
`doc/full_duplex_data_scaling.md` for the full analysis):

| Filter                     | Value | Rationale                                           |
| -------------------------- | ----- | --------------------------------------------------- |
| `--min-turns`              | 6     | enough for meaningful turn-taking dynamics          |
| `--max-gap-s`              | 1.0   | drop runs with long dead air                        |
| `--min-duration-s`         | 30    | practical minimum for FD training samples           |
| `--max-turn-duration-s`    | 10    | Sommelier paper §3.1 — long monologues destabilize Moshi |
| `--only-two-speaker`       | on    | non-negotiable for stereo (one speaker per channel) |
| `--exclude-bgm`            | on    | drop runs where any turn has background music       |

A run is dropped entirely if any single filter fails. Over-length turns
(`> --max-turn-duration-s`) act as **run splitters** — they're excluded
and the surrounding turns are treated as two separate runs, mirroring the
paper's "truncate the region if an utterance exceeded 10 seconds appeared."

## How to (re)generate

The two scripts that produce this format ship in `scripts/`:

```bash
# 1. Extract per-conversation transcript JSONs from a pipeline `_final/` root
python scripts/build_training_conversations.py \
    --root <pipeline _final root>/ \
    --out  work/conversations_fd/ \
    --min-turns 6 --max-gap-s 1.0 --max-turn-duration-s 10.0 \
    --min-duration-s 30 --only-two-speaker --exclude-bgm

# 2. Assemble stereo WAV + companion JSON (both orientations)
python scripts/build_full_duplex_2channel.py \
    --convs-dir  work/conversations_fd/conversations \
    --clips-root <pipeline _final root>/ \
    --out        work/full_duplex_stereo/
```

## Design decisions

### Why zero-origin timestamps?

The per-conversation JSON is a **training-time artifact**. Zero-origin
timestamps mean a dataloader can slice the stereo array with
`turn["start"] * sr` directly — no offset bookkeeping, no silent
off-by-one bugs. The parent-clip offset is preserved as
`_original_clip_offset_s` for provenance and for cross-referencing
against the upstream per-turn tabular data on `vi-sommelier-v0`.

### Why emit both channel orientations?

Two reasons specific to our data:

1. **Podcast sources are peer-symmetric.** Some clips have a clear
   host+guest (Have A Sip, Chuyện Trò), others are co-host dialogues
   (Tự Tình Lúc 0h). Forcing "host on L" requires per-clip host
   annotation for ~1,000 episodes, which isn't well-defined for the
   symmetric ones.
2. **Channel-invariant supervision + free 2× augmentation.** Emitting
   both orientations doubles training data at zero annotation cost and
   removes channel-identity leakage. A model that trains on this can't
   memorize "L = specific person."

Sommelier paper §3.1 chose the opposite (fix one speaker on L) — that's
the theoretically cleaner match to Moshi's pretraining convention when
you have industrial dialogue with clear host/guest roles + full
fine-tuning budget. For our peer-symmetric data + LoRA fine-tuning, both
orientations is the better trade. If a downstream user prefers the
paper's design, they can filter to `__oriA` only.

## Cross-references

- `doc/full_duplex_data_scaling.md` — filter-profile analysis + kill criteria
- `doc/handoff_pipeline_operator.md` — end-to-end pipeline operator guide
- `doc/sommelier.pdf` — Jung et al., 2026 — upstream design paper
- Upstream per-turn artifacts: `tuanamz/vi-sommelier-v0` on HuggingFace
