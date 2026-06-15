# Speaker Resegmentation Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative post-SortFormer speaker resegmentation audit that uses global speaker references, local activity evidence, frame-state decoding, rules, and validation to fix only high-confidence diarization errors.

**Architecture:** Add a focused `utils/speaker_resegmentation.py` module with pure data structures and testable functions. `run_stage_diarization_only.py` will call it after SortFormer speaker linking only when the new CLI flag is enabled, then write a JSON report next to `diarization.json`.

**Tech Stack:** Python, pandas, numpy, pyannote embeddings, optional local pyannote activity provider, unittest.

---

### Task 1: Pure Audit Engine

**Files:**
- Create: `podcast-pipeline/utils/speaker_resegmentation.py`
- Test: `tests/test_speaker_resegmentation.py`

- [x] **Step 1: Write failing tests**

Cover candidate detection for suspicious boundaries and long segments, local-to-global mapping with top-1/top-2 margin, frame-state decoding, split/trim/overlap decisions, and rollback when confidence is low.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_speaker_resegmentation`
Expected: import failure before implementation.

- [x] **Step 3: Implement minimal engine**

Implement `SpeakerReference`, `LocalActivity`, `AuditConfig`, `build_global_references`, `find_audit_regions`, `map_local_activities_to_global`, `decode_frame_states`, `apply_resegmentation_audit`, and `write_resegmentation_report`.

- [x] **Step 4: Run tests to verify pass**

Run: `python -m unittest tests.test_speaker_resegmentation`
Expected: OK.

### Task 2: Stage 01 Integration

**Files:**
- Modify: `podcast-pipeline/run_stage_diarization_only.py`
- Modify: `tests/test_stage_diarization_only.py`

- [x] **Step 1: Write failing tests**

Assert the Stage 01 script exposes the new `--speaker-resegmentation-audit` flags and no legacy boundary-refine flags.

- [x] **Step 2: Run tests to verify fail**

Run: `python -m unittest tests.test_stage_diarization_only`
Expected: missing new flag assertions.

- [x] **Step 3: Implement integration**

Load optional audit config, build local activity provider, call `apply_resegmentation_audit`, write report, and include audit metadata.

- [x] **Step 4: Run tests to verify pass**

Run: `python -m unittest tests.test_stage_diarization_only tests.test_speaker_resegmentation`
Expected: OK.

### Task 3: Kaggle Notebook Wiring

**Files:**
- Modify: `kaggle_notebooks/07_stage_diarization_only.ipynb`
- Modify: `kaggle_notebooks/stage1_updated.ipynb`
- Modify: `tools/build_kaggle_stage_diarization_notebook.py`
- Modify: `tests/test_kaggle_stage_diarization_notebook.py`

- [x] **Step 1: Write failing tests**

Assert notebook and builder expose new resegmentation controls, and still do not contain legacy boundary-refine controls.

- [x] **Step 2: Run tests to verify fail**

Run: `python -m unittest tests.test_kaggle_stage_diarization_notebook`
Expected: missing new notebook controls.

- [x] **Step 3: Wire notebook and builder**

Add conservative default controls and pass CLI flags only when enabled.

- [x] **Step 4: Run tests to verify pass**

Run: `python -m unittest tests.test_kaggle_stage_diarization_notebook`
Expected: OK.

### Task 4: Verification

**Files:**
- All changed files.

- [x] **Step 1: Run targeted tests**

Run: `python -m unittest tests.test_speaker_resegmentation tests.test_stage_diarization_only tests.test_kaggle_stage_diarization_notebook`
Expected: OK.

- [x] **Step 2: Compile Python files**

Run: `python -m py_compile podcast-pipeline/run_stage_diarization_only.py podcast-pipeline/utils/speaker_resegmentation.py tools/build_kaggle_stage_diarization_notebook.py`
Expected: exit 0.

- [x] **Step 3: Check removed legacy strings**

Run: `rg "BOUNDARY_REFINE|boundary-refine|speaker-boundary-refinement" podcast-pipeline tools kaggle_notebooks`
Expected: no matches.
