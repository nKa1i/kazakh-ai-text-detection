# -*- coding: utf-8 -*-
"""
Kaggle Kernel Packaging and Push Automation Tool.
Packages the embedded Kazakh knowledge corpus into the Kaggle runner script,
configures kernel metadata for Dual Tesla T4 GPUs, and deploys via the Kaggle CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict

# Ensure project root is in sys.path when invoked directly as a script
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from scripts.package_kaggle_kernel import (
        compress_corpus_to_base64,
        package_kernel_script,
    )
except ImportError:
    from package_kaggle_kernel import (
        compress_corpus_to_base64,
        package_kernel_script,
    )



def _resolve_path(path: str) -> str:
    """
    Resolves a file path checking the current directory, then repo root.
    """
    if os.path.exists(path):
        return os.path.abspath(path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alt_path = os.path.join(repo_root, path)
    if os.path.exists(alt_path):
        return os.path.abspath(alt_path)
    return os.path.abspath(path)


def check_kernel_status(kernel_id: str = "dauletanekesh/kazakh-gpu-runner-nb") -> str:
    """
    Queries Kaggle CLI for the current execution status of a kernel.

    Args:
        kernel_id: Kaggle kernel identifier in the form username/kernel-name.

    Returns:
        String containing the status reported by Kaggle CLI.
    """
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        status_text = res.stdout.strip()
        print(f"Kernel status for '{kernel_id}': {status_text}")
        return status_text
    except subprocess.CalledProcessError as e:
        print(f"Failed to check kernel status for '{kernel_id}': {e.stderr}")
        return ""
    except FileNotFoundError as e:
        print(f"Kaggle CLI not found: {e}")
        return ""


def prepare_and_push(
    dry_run: bool = False,
    kernel_dir: str = "kaggle_runner",
    corpus_file: str = "kazakh_knowledge_corpus.jsonl",
    script_file: str = "generate_fever_and_social_kernel.py",
) -> bool:
    """
    Packages the kernel script with the embedded knowledge corpus,
    updates kernel-metadata.json, and pushes the kernel to Kaggle.

    Args:
        dry_run: If True, packages script and updates metadata but does not push.
        kernel_dir: Directory containing the kernel script and metadata.
        corpus_file: Filename of the JSONL corpus inside kernel_dir.
        script_file: Filename of the target kernel script inside kernel_dir.

    Returns:
        True if all operations succeeded, False otherwise.
    """
    print(f"Preparing Kaggle Kernel Package in '{kernel_dir}'...")
    resolved_dir = _resolve_path(kernel_dir)
    os.makedirs(resolved_dir, exist_ok=True)
    os.environ["PYTHONUTF8"] = "1"

    # 1. Package embedded corpus into kernel script
    corpus_path = os.path.join(resolved_dir, corpus_file)
    target_script = os.path.join(resolved_dir, script_file)

    if not os.path.exists(corpus_path):
        corpus_path = _resolve_path(os.path.join("kaggle_runner", corpus_file))

    if not os.path.exists(target_script):
        target_script = _resolve_path(os.path.join("kaggle_runner", script_file))

    print(f"Compressing knowledge corpus from: {corpus_path}")
    corpus_b64 = compress_corpus_to_base64(corpus_path)
    print(f"Embedding compressed corpus into: {target_script}")
    package_kernel_script(target_script, target_script, corpus_b64)

    # 2. Update kernel-metadata.json
    metadata_path = os.path.join(resolved_dir, "kernel-metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata: Dict[str, Any] = json.load(f)
    else:
        metadata = {
            "id": "dauletanekesh/kazakh-gpu-runner-nb",
            "title": "kazakh-gpu-runner-nb",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }

    metadata["code_file"] = script_file
    metadata["enable_gpu"] = True
    metadata["enable_tpu"] = False
    metadata["enable_internet"] = True
    metadata["machine_shape"] = "NvidiaTeslaT4"
    metadata["accelerator"] = "gpu_t4_x2"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated metadata at: {metadata_path}")

    # 3. Copy human seed data if available
    seed_source = _resolve_path("data/seed_human_reviews_1k.json")
    seed_target = os.path.join(resolved_dir, "seed_human_reviews_1k.json")
    if os.path.exists(seed_source) and os.path.abspath(seed_source) != os.path.abspath(seed_target):
        shutil.copy(seed_source, seed_target)
        print(f"Copied seed reviews from {seed_source} to {seed_target}")

    # 4. Handle dry run
    if dry_run:
        print("Dry-run mode enabled: skipping Kaggle CLI push.")
        return True

    # 5. Push to Kaggle
    push_target = kernel_dir if os.path.exists(kernel_dir) else resolved_dir
    print(f"\nPushing kernel to Kaggle via CLI: 'kaggle kernels push -p {push_target}'")
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", push_target],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        print("Kaggle CLI Output:\n", res.stdout)
        print("Kaggle push SUCCESSFUL!")
        return True
    except subprocess.CalledProcessError as e:
        print("Kaggle push failed:", e.stderr)
        print("Standard output:", e.stdout)
        return False
    except FileNotFoundError as e:
        print("Kaggle CLI command not found:", e)
        return False


def main() -> None:
    """
    Main command-line entry point.
    """
    parser = argparse.ArgumentParser(
        description="Package and push Kaggle GPU batch generation kernel."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Package kernel script and update metadata without executing Kaggle CLI push.",
    )
    parser.add_argument(
        "--kernel-dir",
        default="kaggle_runner",
        help="Directory containing kernel artifacts (default: kaggle_runner).",
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Check remote kernel status after push.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check remote kernel status without packaging or pushing.",
    )
    parser.add_argument(
        "--kernel-id",
        default="dauletanekesh/kazakh-gpu-runner-nb",
        help="Kaggle kernel identifier (default: dauletanekesh/kazakh-gpu-runner-nb).",
    )
    args = parser.parse_args()

    if args.status_only:
        check_kernel_status(args.kernel_id)
        return

    success = prepare_and_push(dry_run=args.dry_run, kernel_dir=args.kernel_dir)
    if success and not args.dry_run and args.check_status:
        check_kernel_status(args.kernel_id)

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
