#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
EPOCHS = 150
TRAIN_SCRIPT = ROOT / "train.py"

sys.path.insert(0, str(ROOT))

from datasets.batching import BATCHING_STRATEGIES


def archive_run(strategy: str) -> None:
    dest = CHECKPOINTS_DIR / strategy
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "history.json"):
        src = CHECKPOINTS_DIR / name
        if src.exists():
            shutil.copy2(src, dest / name)


def main() -> int:
    for index, strategy in enumerate(BATCHING_STRATEGIES, start=1):
        print(f"\n{'=' * 60}")
        print(f"Run {index}/{len(BATCHING_STRATEGIES)}: {strategy} ({EPOCHS} epochs)")
        print("=" * 60)

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--batching-strategy",
            strategy,
            "--epochs",
            str(EPOCHS),
            "--run-name",
            strategy,
        ]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"Training failed for strategy '{strategy}' (exit {result.returncode})")
            return result.returncode

        archive_run(strategy)
        print(f"Archived checkpoints to {CHECKPOINTS_DIR / strategy}")

    print(f"\nAll {len(BATCHING_STRATEGIES)} training runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
