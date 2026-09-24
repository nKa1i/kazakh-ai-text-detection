# -*- coding: utf-8 -*-
"""
Autonomous Self-Contained Kernel Packaging Tool.
Compresses and embeds the Kazakh knowledge corpus into the Kaggle kernel script
using gzip compression and base64 encoding.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
from typing import Any, Dict, List


def _resolve_path(path: str) -> str:
    """
    Resolves a file path, checking both current working directory and repository root.
    """
    if os.path.exists(path):
        return os.path.abspath(path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alt_path = os.path.join(repo_root, path)
    if os.path.exists(alt_path):
        return os.path.abspath(alt_path)
    return os.path.abspath(path)


def compress_corpus_to_base64(
    corpus_path: str = "kaggle_runner/kazakh_knowledge_corpus.jsonl",
) -> str:
    """
    Reads a JSONL corpus file, compresses the UTF-8 content bytes using gzip,
    and returns an ASCII base64-encoded string.

    Args:
        corpus_path: Path to the JSONL corpus file.

    Returns:
        Base64-encoded ASCII string of the compressed corpus.

    Raises:
        FileNotFoundError: If the specified corpus file does not exist.
    """
    resolved_path = _resolve_path(corpus_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path} (resolved: {resolved_path})")

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    raw_bytes = raw_text.encode("utf-8")
    compressed_bytes = gzip.compress(raw_bytes, compresslevel=9)
    payload_b64 = base64.b64encode(compressed_bytes).decode("ascii")
    return payload_b64


def decompress_corpus_from_base64(payload_b64: str) -> List[Dict[str, Any]]:
    """
    Decodes an ASCII base64 string, decompresses it using gzip,
    decodes the UTF-8 string, and parses each line as JSON.

    Args:
        payload_b64: Base64-encoded gzip string.

    Returns:
        List of dictionaries parsed from each JSON line in the corpus.
    """
    clean_b64 = payload_b64.strip()
    if not clean_b64:
        return []

    compressed_bytes = base64.b64decode(clean_b64.encode("ascii"))
    decompressed_bytes = gzip.decompress(compressed_bytes)
    decompressed_str = decompressed_bytes.decode("utf-8")

    records: List[Dict[str, Any]] = []
    for line in decompressed_str.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def package_kernel_script(
    template_path: str,
    output_path: str,
    corpus_b64: str,
) -> None:
    """
    Injects the embedded base64 corpus payload into the kernel script,
    ensuring EMBEDDED_KNOWLEDGE_CORPUS_B64 = "<payload>" is populated in the output.

    Args:
        template_path: Path to the template kernel script.
        output_path: Path to write the packaged kernel script.
        corpus_b64: The base64-encoded compressed corpus payload string.

    Raises:
        FileNotFoundError: If template_path does not exist.
    """
    resolved_template = _resolve_path(template_path)
    if not os.path.exists(resolved_template):
        raise FileNotFoundError(f"Template script not found: {template_path} (resolved: {resolved_template})")

    with open(resolved_template, "r", encoding="utf-8") as f:
        content = f.read()

    target_pattern = r'EMBEDDED_KNOWLEDGE_CORPUS_B64\s*=\s*["\'][^"\']*["\']'
    replacement = f'EMBEDDED_KNOWLEDGE_CORPUS_B64 = "{corpus_b64}"'

    if re.search(target_pattern, content):
        new_content = re.sub(target_pattern, replacement, content)
    else:
        docstring_match = re.match(r'^(?:#.*?[\r\n]+)*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')[\r\n]+', content)
        if docstring_match:
            end_pos = docstring_match.end()
            new_content = content[:end_pos] + f"\n{replacement}\n" + content[end_pos:]
        else:
            new_content = f"{replacement}\n\n" + content

    resolved_output = os.path.abspath(output_path)
    parent_dir = os.path.dirname(resolved_output)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(resolved_output, "w", encoding="utf-8") as f:
        f.write(new_content)


def main() -> None:
    """
    Command-line entry point for packaging the Kaggle GPU kernel.
    """
    parser = argparse.ArgumentParser(
        description="Package Kaggle GPU kernel with embedded knowledge corpus."
    )
    parser.add_argument(
        "--corpus",
        default="kaggle_runner/kazakh_knowledge_corpus.jsonl",
        help="Path to JSONL knowledge corpus file",
    )
    parser.add_argument(
        "--template",
        default="kaggle_runner/generate_fever_and_social_kernel.py",
        help="Path to source template kernel script",
    )
    parser.add_argument(
        "--output",
        default="kaggle_runner/generate_fever_and_social_kernel.py",
        help="Destination path for packaged kernel script",
    )
    args = parser.parse_args()

    print("Compressing knowledge corpus...")
    corpus_b64 = compress_corpus_to_base64(args.corpus)
    records = decompress_corpus_from_base64(corpus_b64)
    print(f"Loaded and verified {len(records)} articles in compressed payload ({len(corpus_b64)} b64 chars).")

    print(f"Packaging kernel: {args.template} -> {args.output}...")
    package_kernel_script(args.template, args.output, corpus_b64)
    print("Kernel packaging complete.")


if __name__ == "__main__":
    main()
