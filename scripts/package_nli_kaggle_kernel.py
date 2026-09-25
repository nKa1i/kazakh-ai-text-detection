# -*- coding: utf-8 -*-
"""
scripts/package_nli_kaggle_kernel.py: Autonomous Kaggle GPU Training Packager.

Compresses and embeds the Kazakh-FEVER training, validation, and test datasets
along with the knowledge corpus into a self-contained single-file GPU training
script ready for execution on Kaggle Dual Tesla T4 / A100 GPUs or local simulation.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional


def _resolve_path(path: str) -> str:
    """
    Resolves a file path checking both current working directory and repository root.
    """
    if os.path.exists(path):
        return os.path.abspath(path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alt_path = os.path.join(repo_root, path)
    if os.path.exists(alt_path):
        return os.path.abspath(alt_path)
    return os.path.abspath(path)


def compress_dataset_to_base64(file_path: str) -> str:
    """
    Reads a UTF-8 JSONL dataset file, compresses its content bytes using gzip (level 9),
    and returns an ASCII base64-encoded string.

    Args:
        file_path: Path to the JSONL dataset file.

    Returns:
        Base64-encoded ASCII string of the compressed content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    resolved_path = _resolve_path(file_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path} (resolved: {resolved_path})")

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    raw_bytes = raw_text.encode("utf-8")
    compressed_bytes = gzip.compress(raw_bytes, compresslevel=9)
    payload_b64 = base64.b64encode(compressed_bytes).decode("ascii")
    return payload_b64


def decompress_dataset_from_base64(b64_str: str) -> List[Dict[str, Any]]:
    """
    Decodes an ASCII base64 string, decompresses it using gzip,
    decodes the UTF-8 text, and parses each line as a JSON record.

    Args:
        b64_str: Base64-encoded gzip string.

    Returns:
        List of dictionaries parsed from each JSON line in the dataset.
    """
    clean_b64 = b64_str.strip() if b64_str else ""
    if not clean_b64:
        return []

    compressed_bytes = base64.b64decode(clean_b64.encode("ascii"))
    decompressed_bytes = gzip.decompress(compressed_bytes)
    decompressed_str = decompressed_bytes.decode("utf-8")

    records: List[Dict[str, Any]] = []
    for line in decompressed_str.splitlines():
        line_clean = line.strip()
        if line_clean:
            records.append(json.loads(line_clean))
    return records


def _replace_embedded_constant(content: str, const_name: str, placeholder: str, payload_b64: str) -> str:
    """
    Replaces both template token placeholders and existing assignment statements for a given constant.
    """
    if placeholder in content:
        content = content.replace(placeholder, payload_b64)

    pattern = rf'{const_name}\s*=\s*["\'][^"\']*["\']'
    replacement = f'{const_name} = "{payload_b64}"'

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
    else:
        docstring_match = re.match(r'^(?:#.*?[\r\n]+)*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')[\r\n]+', content)
        if docstring_match:
            end_pos = docstring_match.end()
            content = content[:end_pos] + f"\n{replacement}\n" + content[end_pos:]
        else:
            content = f"{replacement}\n\n" + content

    return content


def package_nli_kernel(
    output_path: str = "kaggle_runner/train_nli_kernel.py",
    train_path: str = "data/kazakh_fever_train.jsonl",
    dev_path: str = "data/kazakh_fever_dev.jsonl",
    test_path: str = "data/kazakh_fever_test.jsonl",
    corpus_path: str = "data/kazakh_knowledge_corpus.jsonl",
    dry_run: bool = False,
    push: bool = False,
    template_path: Optional[str] = None,
) -> str:
    """
    Packages the autonomous GPU training script with embedded base64 datasets.

    Args:
        output_path: Destination path for the packaged kernel script.
        train_path: Path to the Kazakh-FEVER training JSONL dataset.
        dev_path: Path to the Kazakh-FEVER validation JSONL dataset.
        test_path: Path to the Kazakh-FEVER test JSONL dataset.
        corpus_path: Path to the knowledge corpus JSONL file.
        dry_run: If True, packages script without pushing to Kaggle.
        push: If True and not dry_run, pushes packaged kernel to Kaggle via CLI.
        template_path: Optional explicit template script path.

    Returns:
        The output path of the packaged script.

    Raises:
        FileNotFoundError: If the source template script is not found.
    """
    if template_path:
        resolved_template = _resolve_path(template_path)
    elif os.path.exists(output_path):
        resolved_template = os.path.abspath(output_path)
    else:
        resolved_template = _resolve_path("kaggle_runner/train_nli_kernel.py")

    if not os.path.exists(resolved_template):
        raise FileNotFoundError(
            f"Kernel template script not found: {template_path or output_path} "
            f"(resolved: {resolved_template})"
        )

    with open(resolved_template, "r", encoding="utf-8") as f:
        content = f.read()

    if not dry_run:
        train_b64 = compress_dataset_to_base64(train_path)
        dev_b64 = compress_dataset_to_base64(dev_path)
        test_b64 = compress_dataset_to_base64(test_path)
        corpus_b64 = compress_dataset_to_base64(corpus_path)

        content = _replace_embedded_constant(content, "EMBEDDED_TRAIN_B64", "__TRAIN_DATA_B64__", train_b64)
        content = _replace_embedded_constant(content, "EMBEDDED_DEV_B64", "__DEV_DATA_B64__", dev_b64)
        content = _replace_embedded_constant(content, "EMBEDDED_TEST_B64", "__TEST_DATA_B64__", test_b64)
        content = _replace_embedded_constant(content, "EMBEDDED_CORPUS_B64", "__CORPUS_DATA_B64__", corpus_b64)

    resolved_output = os.path.abspath(output_path)
    parent_dir = os.path.dirname(resolved_output)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(resolved_output, "w", encoding="utf-8") as f:
        f.write(content)

    if push and not dry_run:
        kernel_dir = parent_dir or os.path.abspath(".")
        metadata_path = os.path.join(kernel_dir, "kernel-metadata.json")
        code_file = os.path.basename(resolved_output)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata: Dict[str, Any] = json.load(f)
        else:
            metadata = {
                "id": "dauletanekesh/kazakh-nli-gpu-runner",
                "title": "kazakh-nli-gpu-runner",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }

        metadata["code_file"] = code_file
        metadata["enable_gpu"] = True
        metadata["enable_tpu"] = False
        metadata["enable_internet"] = True
        metadata["machine_shape"] = "NvidiaTeslaT4"
        metadata["accelerator"] = "gpu_t4_x2"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        try:
            subprocess.run(
                ["kaggle", "kernels", "push", "-p", kernel_dir],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Kaggle push invocation failed: {e}")

    return resolved_output


def main() -> None:
    """
    Main CLI entry point for packaging the NLI GPU training kernel.
    """
    parser = argparse.ArgumentParser(
        description="Package Kaggle NLI GPU training kernel with embedded datasets."
    )
    parser.add_argument(
        "--output",
        default="kaggle_runner/train_nli_kernel.py",
        help="Destination path for packaged kernel script",
    )
    parser.add_argument(
        "--train_data",
        default="data/kazakh_fever_train.jsonl",
        help="Path to Kazakh-FEVER training JSONL dataset",
    )
    parser.add_argument(
        "--dev_data",
        default="data/kazakh_fever_dev.jsonl",
        help="Path to Kazakh-FEVER validation JSONL dataset",
    )
    parser.add_argument(
        "--test_data",
        default="data/kazakh_fever_test.jsonl",
        help="Path to Kazakh-FEVER test JSONL dataset",
    )
    parser.add_argument(
        "--corpus",
        default="data/kazakh_knowledge_corpus.jsonl",
        help="Path to knowledge corpus JSONL file",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Package script without pushing to Kaggle",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Deploy packaged kernel to Kaggle via CLI",
    )
    args = parser.parse_args()

    out = package_nli_kernel(
        output_path=args.output,
        train_path=args.train_data,
        dev_path=args.dev_data,
        test_path=args.test_data,
        corpus_path=args.corpus,
        dry_run=args.dry_run,
        push=args.push,
    )
    print(f"Packaged kernel written to: {out}")


if __name__ == "__main__":
    main()
