# -*- coding: utf-8 -*-
"""
scripts/push_to_hf.py: Automated 1-Command Deployment to Hugging Face Spaces.

Usage:
    python scripts/push_to_hf.py --repo-id <username/space-name> [--token <hf_token>] [--private] [--bundle-dir <dir>]
"""

import os
import sys
import argparse
from typing import Optional, List

try:
    from huggingface_hub import HfApi, get_token
    _HF_AVAILABLE = True
except ImportError:
    HfApi = None
    get_token = lambda: None
    _HF_AVAILABLE = False

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BUNDLE_DIR = os.path.join(_PROJECT_ROOT, "hf_space")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for Hugging Face Spaces deployment."""
    parser = argparse.ArgumentParser(
        description="Deploy standalone Kazakh AI Text Detector bundle to Hugging Face Spaces."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face Space repository ID (e.g., 'username/kazakh-ai-text-detector').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API write token. If omitted, uses HF_TOKEN env var or cached CLI token.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the Hugging Face Space private if creating a new repository.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default=_DEFAULT_BUNDLE_DIR,
        help=f"Directory containing the self-contained deployment bundle (default: {_DEFAULT_BUNDLE_DIR}).",
    )
    return parser.parse_args(args)


def resolve_token(token: Optional[str] = None) -> Optional[str]:
    """Resolves Hugging Face access token with hierarchical fallback."""
    if token and token.strip():
        return token.strip()

    env_token = os.environ.get("HF_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    if _HF_AVAILABLE and callable(get_token):
        try:
            cached = get_token()
            if cached and str(cached).strip():
                return str(cached).strip()
        except Exception:
            pass

    return None


def validate_bundle_dir(bundle_dir: str) -> None:
    """Validates that the bundle directory exists and contains all required artifacts."""
    if not os.path.exists(bundle_dir):
        raise FileNotFoundError(f"Deployment bundle directory not found: '{bundle_dir}'")
    if not os.path.isdir(bundle_dir):
        raise NotADirectoryError(f"Specified bundle path is not a directory: '{bundle_dir}'")

    required_files = ["app.py", "README.md"]
    for req in required_files:
        fpath = os.path.join(bundle_dir, req)
        if not os.path.isfile(fpath):
            raise ValueError(f"Bundle directory '{bundle_dir}' is missing required file '{req}'")


def push_to_hf(
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    bundle_dir: Optional[str] = None,
) -> str:
    """
    Creates/verifies a Hugging Face Space repository and uploads the bundle directory.

    Returns:
        Clickable web URL to the deployed Hugging Face Space.
    """
    if bundle_dir is None:
        bundle_dir = _DEFAULT_BUNDLE_DIR

    validate_bundle_dir(bundle_dir)

    if not _HF_AVAILABLE or HfApi is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Please install it using: pip install huggingface_hub"
        )

    resolved_token = resolve_token(token)
    api = HfApi(token=resolved_token)

    # 1. Verify or create Space repository
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"[*] Found existing Space repository: {repo_id}")
    except Exception:
        print(f"[*] Space repository '{repo_id}' not found. Creating new Gradio Space...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
        )
        print(f"[+] Successfully created Space: {repo_id} (private={private})")

    # 2. Upload bundle folder
    print(f"[*] Uploading deployment bundle from '{bundle_dir}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=bundle_dir,
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"],
    )
    print(f"[+] Upload complete!")

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"[+] Successfully deployed to Hugging Face Spaces!")
    print(f"[*] Web URL: {space_url}")
    return space_url


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        push_to_hf(
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            bundle_dir=args.bundle_dir,
        )
    except Exception as e:
        print(f"[ERROR] Deployment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
