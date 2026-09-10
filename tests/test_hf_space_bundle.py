# -*- coding: utf-8 -*-
"""
tests/test_hf_space_bundle.py: Comprehensive test suite for Hugging Face Spaces
cloud deployment bundle and automated upload script.
"""

import os
import sys
import unittest
import py_compile
import tempfile
import yaml
from unittest.mock import patch, MagicMock

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HF_SPACE_DIR = os.path.join(PROJECT_ROOT, "hf_space")
PUSH_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "push_to_hf.py")


class TestHfSpaceBundleStructure(unittest.TestCase):
    """Verifies that the hf_space/ bundle contains all required files and modules."""

    def test_bundle_directory_exists(self):
        self.assertTrue(
            os.path.isdir(HF_SPACE_DIR),
            f"Expected directory {HF_SPACE_DIR} does not exist"
        )

    def test_bundle_required_files_exist(self):
        expected_files = [
            "app.py",
            "README.md",
            "requirements.txt",
            "fst_analyzer.py",
        ]
        for fname in expected_files:
            fpath = os.path.join(HF_SPACE_DIR, fname)
            self.assertTrue(
                os.path.isfile(fpath),
                f"Missing bundle file: {fpath}"
            )

    def test_bundle_kaz_mage_package_exists(self):
        kaz_mage_dir = os.path.join(HF_SPACE_DIR, "kaz_mage")
        self.assertTrue(os.path.isdir(kaz_mage_dir), "hf_space/kaz_mage/ must exist")
        expected_modules = ["__init__.py", "chunker.py", "aggregator.py", "document.py", "data.py"]
        for mod in expected_modules:
            fpath = os.path.join(kaz_mage_dir, mod)
            self.assertTrue(os.path.isfile(fpath), f"Missing kaz_mage module: {fpath}")

    def test_bundle_ui_package_exists(self):
        ui_dir = os.path.join(HF_SPACE_DIR, "ui")
        self.assertTrue(os.path.isdir(ui_dir), "hf_space/ui/ must exist")
        expected_modules = [
            "__init__.py",
            "app.py",
            "highlighting.py",
            "presets.py",
            "file_loader.py",
            "sentence_analyzer.py",
            "linguistic_explainer.py",
        ]
        for mod in expected_modules:
            fpath = os.path.join(ui_dir, mod)
            self.assertTrue(os.path.isfile(fpath), f"Missing ui module: {fpath}")


class TestHfSpaceMetadata(unittest.TestCase):
    """Validates the YAML frontmatter header and academic documentation in hf_space/README.md."""

    def setUp(self):
        self.readme_path = os.path.join(HF_SPACE_DIR, "README.md")

    def test_readme_yaml_frontmatter_schema(self):
        self.assertTrue(os.path.isfile(self.readme_path), "README.md must exist")
        with open(self.readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertTrue(
            content.startswith("---"),
            "README.md must start with YAML frontmatter delimiter '---'"
        )

        parts = content.split("---", 2)
        self.assertGreaterEqual(
            len(parts), 3,
            "README.md must contain closing '---' frontmatter delimiter"
        )

        yaml_text = parts[1]
        metadata = yaml.safe_load(yaml_text)
        self.assertIsInstance(metadata, dict, "YAML metadata must parse to a dictionary")

        self.assertEqual(metadata.get("title"), "Kazakh AI Text Detector")
        self.assertEqual(metadata.get("emoji"), "🔍")
        self.assertEqual(metadata.get("colorFrom"), "indigo")
        self.assertEqual(metadata.get("colorTo"), "gray")
        self.assertEqual(metadata.get("sdk"), "gradio")
        self.assertEqual(str(metadata.get("sdk_version")), "6.26.0")
        self.assertEqual(metadata.get("app_file"), "app.py")
        self.assertFalse(metadata.get("pinned"))

    def test_readme_academic_content(self):
        with open(self.readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Academic sections
        self.assertIn("Dual-Stream", content)
        self.assertIn("Cross-Attention", content)
        self.assertIn("Kazakh Morphological FST", content)
        self.assertIn("Kaz-MAGE", content)
        self.assertIn("Q1", content)
        self.assertIn("Q4", content)

        # Research team attribution
        self.assertIn("Da Lei", content)
        self.assertIn("Guo", content)
        self.assertIn("Wang Na", content)

    def test_requirements_txt(self):
        req_path = os.path.join(HF_SPACE_DIR, "requirements.txt")
        self.assertTrue(os.path.isfile(req_path), "requirements.txt must exist")
        with open(req_path, "r", encoding="utf-8") as f:
            reqs = f.read()

        self.assertIn("gradio", reqs.lower())
        # Ensure heavy ML frameworks are not mandatory in requirements.txt
        self.assertNotIn("torch", reqs.lower())
        self.assertNotIn("transformers", reqs.lower())


class TestHfSpaceAppExecution(unittest.TestCase):
    """Tests syntax and cold-start instantiation of hf_space/app.py."""

    def test_py_compile_app(self):
        app_path = os.path.join(HF_SPACE_DIR, "app.py")
        self.assertTrue(os.path.isfile(app_path), "hf_space/app.py must exist")
        try:
            py_compile.compile(app_path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"py_compile failed on hf_space/app.py: {e}")

    def test_cold_start_app_instance(self):
        """Verifies that hf_space/app.py instantiates create_app(load_model=False) cleanly."""
        import gradio as gr
        import importlib.util

        app_path = os.path.join(HF_SPACE_DIR, "app.py")
        self.assertTrue(os.path.isfile(app_path), "hf_space/app.py must exist")

        spec = importlib.util.spec_from_file_location("hf_space_app", app_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)

        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec.loader.exec_module(mod)
            self.assertTrue(hasattr(mod, "demo"), "hf_space/app.py must define 'demo'")
            self.assertIsInstance(mod.demo, gr.Blocks, "'demo' must be a gradio.Blocks instance")
        finally:
            sys.path = old_sys_path


class TestPushToHfScript(unittest.TestCase):
    """Tests scripts/push_to_hf.py CLI argument parsing and validation logic."""

    def setUp(self):
        self.script_path = PUSH_SCRIPT

    def test_script_exists_and_compiles(self):
        self.assertTrue(os.path.isfile(self.script_path), "scripts/push_to_hf.py must exist")
        try:
            py_compile.compile(self.script_path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"py_compile failed on scripts/push_to_hf.py: {e}")

    def test_parse_args_defaults(self):
        from scripts.push_to_hf import parse_args
        args = parse_args(["--repo-id", "username/test-space"])
        self.assertEqual(args.repo_id, "username/test-space")
        self.assertIsNone(args.token)
        self.assertFalse(args.private)
        self.assertTrue(os.path.samefile(args.bundle_dir, HF_SPACE_DIR) or args.bundle_dir.endswith("hf_space"))

    def test_parse_args_custom(self):
        from scripts.push_to_hf import parse_args
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args([
                "--repo-id", "org/custom-space",
                "--token", "hf_test_token_123",
                "--private",
                "--bundle-dir", tmpdir,
            ])
            self.assertEqual(args.repo_id, "org/custom-space")
            self.assertEqual(args.token, "hf_test_token_123")
            self.assertTrue(args.private)
            self.assertEqual(args.bundle_dir, tmpdir)

    def test_validate_bundle_missing_dir(self):
        from scripts.push_to_hf import validate_bundle_dir
        with self.assertRaises(FileNotFoundError):
            validate_bundle_dir("nonexistent_directory_xyz_123")

    def test_validate_bundle_missing_required_files(self):
        from scripts.push_to_hf import validate_bundle_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                validate_bundle_dir(tmpdir)

    def test_validate_bundle_valid(self):
        from scripts.push_to_hf import validate_bundle_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "app.py"), "w", encoding="utf-8") as f:
                f.write("# app")
            with open(os.path.join(tmpdir, "README.md"), "w", encoding="utf-8") as f:
                f.write("# readme")
            validate_bundle_dir(tmpdir)

    def test_resolve_token(self):
        from scripts.push_to_hf import resolve_token
        self.assertEqual(resolve_token("explicit_token"), "explicit_token")

        with patch.dict(os.environ, {"HF_TOKEN": "env_token_abc"}):
            self.assertEqual(resolve_token(None), "env_token_abc")

    @patch("scripts.push_to_hf.HfApi")
    def test_push_to_hf_execution(self, mock_hf_api_cls):
        from scripts.push_to_hf import push_to_hf

        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "app.py"), "w", encoding="utf-8") as f:
                f.write("# app")
            with open(os.path.join(tmpdir, "README.md"), "w", encoding="utf-8") as f:
                f.write("# readme")

            url = push_to_hf(
                repo_id="testuser/testspace",
                token="hf_test_123",
                private=False,
                bundle_dir=tmpdir,
            )

            self.assertEqual(url, "https://huggingface.co/spaces/testuser/testspace")
            mock_hf_api_cls.assert_called_once_with(token="hf_test_123")
            mock_api.upload_folder.assert_called_once_with(
                folder_path=tmpdir,
                repo_id="testuser/testspace",
                repo_type="space",
                ignore_patterns=["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"],
            )


if __name__ == "__main__":
    unittest.main()
