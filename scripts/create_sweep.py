"""Create a W&B sweep from a YAML file, so its runs use THIS Python interpreter.

Usage (from the repository root, with the venv activated):
    python scripts/create_sweep.py configs/sweep_mlp.yaml
    wandb agent <entity>/<project>/<sweep_id>      # printed by this script

Why not just `wandb sweep configs/sweep_mlp.yaml`?
    `wandb agent` starts every run with the bare command `python`. On Windows, a venv's
    python.exe is only a redirector to the global Python installation, and Windows looks
    for `python` in that installation's folder first. The runs then start the global
    Python, which lacks our packages ("No module named 'lightning'"). Writing the full
    path of the current interpreter (sys.executable) into the sweep's `command` avoids
    this. On Linux/Colab plain `wandb sweep` also works.
"""

from __future__ import annotations

import argparse
import os
import sys

import wandb
import yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="sweep YAML, e.g. configs/sweep_mlp.yaml")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--project", default=None, help="default: the YAML's `project`")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)
    # Same as wandb's default command, except "python" -> full path of this interpreter.
    sweep_config["command"] = [sys.executable, "${program}", "${args}"]
    project = args.project or sweep_config.get("project", "nlp-hs26-project1")

    sweep_id = wandb.sweep(sweep_config, entity=args.entity, project=project)
    entity = args.entity or wandb.Api().default_entity
    print(f"\nInterpreter for the runs: {sys.executable}")
    print(f"Start the agent with:\n    wandb agent {entity}/{project}/{sweep_id}")


if __name__ == "__main__":
    main()
