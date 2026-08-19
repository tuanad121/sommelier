#!/usr/bin/env python3
"""
Push the full-duplex stereo subset to `tuanamz/vi-sommelier-v0` under `fd/`.

Bundles alongside the WAV/JSON pairs:
  - fd/README.md                     (rename of doc/README_fullduplex.md)
  - fd/read_full_duplex.py           (canonical decoder)
  - fd/build_full_duplex_2channel.py (reproducer)

Existing Layer-1 per-turn tabular artifacts at the root of the repo are
untouched — this is a purely additive push.

Reads HF write token from the HF_WRITE_TOKEN env var. Never persists it.

Usage:
    HF_WRITE_TOKEN=hf_... python scripts/push_fd_to_hf.py \\
        --stereo-dir data/full_duplex_stereo_v2 \\
        --readme     doc/README_fullduplex.md \\
        --scripts    scripts/read_full_duplex.py scripts/build_full_duplex_2channel.py \\
        --commit-msg "fd: initial stereo subset (v3-fd, oriA+oriB, max-turn=Ns)"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "tuanamz/vi-sommelier-v0"
REPO_SUBDIR = "fd"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stereo-dir", type=Path, required=True,
                   help="dir with <conv>__oriA/oriB.wav+.json from build_full_duplex_2channel.py")
    p.add_argument("--readme", type=Path, required=True,
                   help="README to publish as fd/README.md (typically doc/README_fullduplex.md)")
    p.add_argument("--scripts", type=Path, nargs="+", required=True,
                   help="scripts to bundle at fd/<name>.py (reader + reproducer)")
    p.add_argument("--commit-msg", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="stage everything, print what would upload, but do not push")
    args = p.parse_args()

    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        sys.exit("HF_WRITE_TOKEN env var required")

    if not args.stereo_dir.is_dir():
        sys.exit(f"stereo dir not found: {args.stereo_dir}")
    wavs = sorted(args.stereo_dir.glob("*.wav"))
    jsons = sorted(args.stereo_dir.glob("*.json"))
    if not wavs:
        sys.exit(f"no .wav files under {args.stereo_dir}")
    if len(wavs) != len(jsons):
        sys.exit(f"wav/json count mismatch: {len(wavs)} vs {len(jsons)}")

    ori_a = [w for w in wavs if "__oriA" in w.name]
    ori_b = [w for w in wavs if "__oriB" in w.name]
    print(f"[stage] {len(wavs)} wavs ({len(ori_a)} oriA + {len(ori_b)} oriB), "
          f"{len(jsons)} jsons")
    print(f"[stage] +README {args.readme}")
    print(f"[stage] +{len(args.scripts)} scripts: {[s.name for s in args.scripts]}")

    # Build a temporary staging dir mirroring the fd/ layout so upload_folder
    # publishes to the right subdir.
    with tempfile.TemporaryDirectory(prefix="fd_stage_") as tmp:
        stage = Path(tmp) / REPO_SUBDIR
        stage.mkdir()
        # Symlink audio + jsons (avoid duplicating GB-sized data). Fall back to
        # copy if symlinks aren't supported (Windows / some FUSE mounts).
        for src in wavs + jsons:
            dst = stage / src.name
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        # README goes in as fd/README.md
        shutil.copy2(args.readme, stage / "README.md")
        # Scripts go in verbatim
        for s in args.scripts:
            shutil.copy2(s, stage / s.name)

        print(f"[stage] {stage} contents:")
        for p in sorted(stage.iterdir())[:10]:
            print(f"        {p.name}")
        if len(list(stage.iterdir())) > 10:
            print(f"        ... and {len(list(stage.iterdir())) - 10} more")

        if args.dry_run:
            print("[dry-run] not pushing.")
            return

        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(Path(tmp)),
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=args.commit_msg,
        )
        print(f"[done] pushed to https://huggingface.co/datasets/{REPO_ID}/tree/main/{REPO_SUBDIR}")


if __name__ == "__main__":
    main()
