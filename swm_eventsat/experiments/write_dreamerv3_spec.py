#!/usr/bin/env python3
"""Write a DreamerV3 training spec for the AUTOPS EventSat baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swm_eventsat.dreamerv3_adapter import DreamerV3TrainingSpec, write_training_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/eventsat_dreamerv3_training_spec.json")
    parser.add_argument("--training-steps", type=int, default=1_000_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--cluster", default="LRZ AI Systems")
    args = parser.parse_args()
    spec = DreamerV3TrainingSpec(
        training_steps=args.training_steps,
        evaluation_episodes=args.evaluation_episodes,
        cluster=args.cluster,
    )
    write_training_spec(args.out, spec)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
