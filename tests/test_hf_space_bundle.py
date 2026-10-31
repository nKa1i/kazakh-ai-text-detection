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

    def test_bundle_models_package_exists(self):
        models_dir = os.path.join(HF_SPACE_DIR, "models")
        self.assertTrue(os.path.isdir(models_dir), "hf_space/models/ must exist")
        expected_modules = [
            "__init__.py",
            "document_detector.py",
            "domain_calibrator.py",
            "generator_profiler.py",
            "generator_viz.py",
            "heuristic_detector.py",
            "losses.py",
            "morpho_contrastive_detector.py",
            "morpho_nli_verifier.py",
            "pattern_prober.py",
        ]
        for mod in expected_modules:
            fpath = os.path.join(models_dir, mod)
            self.assertTrue(os.path.isfile(fpath), f"Missing models module: {fpath}")

    def test_bundle_verification_package_exists(self):
        verif_dir = os.path.join(HF_SPACE_DIR, "verification")
        self.assertTrue(os.path.isdir(verif_dir), "hf_space/verification/ must exist")
        expected_modules = [
            "__init__.py",
            "claim_extractor.py",
            "evaluator.py",
            "evidence.py",
            "fever_generator.py",
            "knowledge_store.py",
            "nli_verifier.py",
            "retriever.py",
            "trust_scorer.py",
            "verifier.py",
        ]
        for mod in expected_modules:
            fpath = os.path.join(verif_dir, mod)
            self.assertTrue(os.path.isfile(fpath), f"Missing verification module: {fpath}")


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
        self.assertIn("onnxruntime", reqs.lower())
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


class TestHfSpaceSynchronizedComponents(unittest.TestCase):
    """Verifies that synchronized components in hf_space export expected classes and functions."""

    def test_morpho_nli_verifier_exports(self):
        """Assert hf_space/models/morpho_nli_verifier.py exists and exports MorphologicalAffixExtractor and MorphoNLIVerifier."""
        import importlib.util
        target_path = os.path.join(HF_SPACE_DIR, "models", "morpho_nli_verifier.py")
        self.assertTrue(os.path.isfile(target_path), f"File {target_path} must exist in hf_space bundle")

        spec = importlib.util.spec_from_file_location("hf_space_models_morpho_nli_verifier", target_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.assertTrue(hasattr(mod, "MorphologicalAffixExtractor"), "Must export MorphologicalAffixExtractor")
        self.assertTrue(hasattr(mod, "MorphoNLIVerifier"), "Must export MorphoNLIVerifier")
        self.assertTrue(hasattr(mod, "OfflineHeuristicNLIVerifier"), "Must export OfflineHeuristicNLIVerifier")

        extractor = mod.MorphologicalAffixExtractor()
        feats = extractor.extract_features("Бұл мәтін факт емес.", "Бұл мәтін шындық.")
        self.assertEqual(len(feats), 16, "Must extract 16-dimensional morphological feature vector")

    def test_render_2d_trust_matrix_plane_in_highlighting(self):
        """Assert hf_space/ui/highlighting.py defines render_2d_trust_matrix_plane."""
        import importlib.util
        target_path = os.path.join(HF_SPACE_DIR, "ui", "highlighting.py")
        self.assertTrue(os.path.isfile(target_path), f"File {target_path} must exist")

        spec = importlib.util.spec_from_file_location("hf_space_ui_highlighting", target_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.assertTrue(hasattr(mod, "render_2d_trust_matrix_plane"), "Must define render_2d_trust_matrix_plane")
        svg_html = mod.render_2d_trust_matrix_plane(t_fact=0.65, t_gen=0.85, quadrant="Q1", lang="en")
        self.assertIn("<svg", svg_html)
        self.assertIn("Q1", svg_html)
        self.assertIn("Q2", svg_html)
        self.assertIn("Q3", svg_html)
        self.assertIn("Q4", svg_html)

    def test_trustworthy_verifier_dual_mode_support(self):
        """Assert hf_space/verification/verifier.py supports verifier_mode ('morpho' vs 'baseline')."""
        import importlib.util
        target_path = os.path.join(HF_SPACE_DIR, "verification", "verifier.py")
        self.assertTrue(os.path.isfile(target_path), f"File {target_path} must exist")

        spec = importlib.util.spec_from_file_location("hf_space_verification_verifier", target_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec.loader.exec_module(mod)
            self.assertTrue(hasattr(mod, "TrustworthyDocumentVerifier"))
            verifier = mod.TrustworthyDocumentVerifier()
            self.assertEqual(verifier.verifier_mode, "morpho")
            res = verifier.verify("")
            self.assertTrue(hasattr(res, "quadrant_code"))
            self.assertTrue(hasattr(res, "composite_trust"))
            self.assertTrue(hasattr(res, "t_fact"))
            self.assertTrue(hasattr(res, "t_gen"))
        finally:
            sys.path = old_sys_path

    def test_heuristic_detector_autofst_v2_and_dynamic_gate_in_hf_space(self):
        """Assert hf_space/models/heuristic_detector.py exports OfflineHeuristicDetector with compute_dynamic_gate."""
        import importlib.util
        target_path = os.path.join(HF_SPACE_DIR, "models", "heuristic_detector.py")
        self.assertTrue(os.path.isfile(target_path), f"File {target_path} must exist in hf_space bundle")

        spec = importlib.util.spec_from_file_location("hf_space_models_heuristic_detector", target_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec.loader.exec_module(mod)
            self.assertTrue(hasattr(mod, "OfflineHeuristicDetector"))
            det = mod.OfflineHeuristicDetector()
            self.assertTrue(hasattr(det, "compute_dynamic_gate"))
            gate = det.compute_dynamic_gate("Бұл қазақша сынақ мәтіні.")
            self.assertGreaterEqual(gate, 0.22)
            self.assertLessEqual(gate, 0.74)
        finally:
            sys.path = old_sys_path

    def test_morpheme_tokenizer_autofst_v2_in_hf_space(self):
        """Assert hf_space/morpheme_tokenizer.py exports MorphemeTokenizer using DynamicFSTOverlay."""
        import importlib.util
        target_path = os.path.join(HF_SPACE_DIR, "morpheme_tokenizer.py")
        self.assertTrue(os.path.isfile(target_path), f"File {target_path} must exist in hf_space bundle")

        spec = importlib.util.spec_from_file_location("hf_space_morpheme_tokenizer", target_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec.loader.exec_module(mod)
            self.assertTrue(hasattr(mod, "MorphemeTokenizer"))
            tok = mod.MorphemeTokenizer()
            self.assertTrue(hasattr(tok.fst, "normalizer"))
            tokens = tok.tokenize_text("Бугин каспиден заказ бердим")
            self.assertIn("-ден", tokens)
        finally:
            sys.path = old_sys_path

    def test_auto_fst_v2_package_exports_in_hf_space(self):
        """Assert hf_space/auto_fst exports DynamicFSTOverlay and RussianLayoutNormalizer."""
        import importlib.util
        norm_path = os.path.join(HF_SPACE_DIR, "auto_fst", "normalizer.py")
        over_path = os.path.join(HF_SPACE_DIR, "auto_fst", "overlay.py")
        self.assertTrue(os.path.isfile(norm_path))
        self.assertTrue(os.path.isfile(over_path))

        spec_norm = importlib.util.spec_from_file_location("hf_space_auto_fst_normalizer", norm_path)
        mod_norm = importlib.util.module_from_spec(spec_norm)
        spec_norm.loader.exec_module(mod_norm)
        self.assertTrue(hasattr(mod_norm, "RussianLayoutNormalizer"))

        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec_over = importlib.util.spec_from_file_location("hf_space_auto_fst_overlay", over_path)
            mod_over = importlib.util.module_from_spec(spec_over)
            spec_over.loader.exec_module(mod_over)
            self.assertTrue(hasattr(mod_over, "DynamicFSTOverlay"))
        finally:
            sys.path = old_sys_path


    def test_generator_profiler_and_phase2_parity(self):
        """Verifies hf_space/models/generator_profiler.py exists, exports symbols, and maintains root byte parity."""
        import importlib.util
        target_path = os.path.join(HF_SPACE_DIR, "models", "generator_profiler.py")
        root_path = os.path.join(PROJECT_ROOT, "models", "generator_profiler.py")
        self.assertTrue(os.path.isfile(target_path), f"File {target_path} must exist in hf_space bundle")

        # Parity check
        with open(root_path, "rb") as f_root, open(target_path, "rb") as f_hf:
            self.assertEqual(f_root.read(), f_hf.read(), "models/generator_profiler.py and hf_space/models/generator_profiler.py must be byte-identical")

        # Symbol export and execution check
        mod_name = "hf_space_models_generator_profiler"
        spec = importlib.util.spec_from_file_location(mod_name, target_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        old_sys_path = list(sys.path)
        sys.modules[mod_name] = mod
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec.loader.exec_module(mod)
            self.assertTrue(hasattr(mod, "MultiModelGeneratorProfiler"))
            self.assertTrue(hasattr(mod, "GeneratorProfileResult"))
            self.assertTrue(hasattr(mod, "KNOWN_GENERATOR_FAMILIES"))
            profiler = mod.MultiModelGeneratorProfiler()
            res = profiler.profile_document("Қазақстанның экологиялық қауіпсіздігі маңызды.")
            self.assertTrue(hasattr(res, "predicted_family"))
            self.assertTrue(hasattr(res, "is_unseen_generator"))
            self.assertTrue(hasattr(res, "ood_distance"))
        finally:
            sys.path = old_sys_path
            sys.modules.pop(mod_name, None)

    def test_onnx_dual_stream_detector_in_hf_space(self):
        """Verifies hf_space/models/document_detector.py exports ONNXDualStreamDetector with fallback."""
        import importlib.util
        target_path = os.path.join(HF_SPACE_DIR, "models", "document_detector.py")
        self.assertTrue(os.path.isfile(target_path), f"File {target_path} must exist in hf_space bundle")

        mod_name = "hf_space_models_document_detector"
        spec = importlib.util.spec_from_file_location(mod_name, target_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        old_sys_path = list(sys.path)
        sys.modules[mod_name] = mod
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec.loader.exec_module(mod)
            self.assertTrue(hasattr(mod, "ONNXDualStreamDetector"))
            det = mod.ONNXDualStreamDetector(model_path="non_existent.onnx")
            self.assertFalse(det.is_onnx_active)
            res = det.predict_document("Бұл қазақша сынақ мәтіні.")
            self.assertIsNotNone(res)
            self.assertTrue(hasattr(res, "document_ai_probability"))
        finally:
            sys.path = old_sys_path
            sys.modules.pop(mod_name, None)

    def test_phase3_pattern_prober_and_viz_parity(self):
        """Verifies Phase 3 pattern_prober.py, generator_viz.py, and ui/app.py maintain byte parity with hf_space/."""
        import importlib.util

        rel_files = [
            os.path.join("models", "pattern_prober.py"),
            os.path.join("models", "generator_viz.py"),
            os.path.join("ui", "app.py"),
        ]
        for rel in rel_files:
            root_f = os.path.join(PROJECT_ROOT, rel)
            hf_f = os.path.join(HF_SPACE_DIR, rel)
            self.assertTrue(os.path.isfile(hf_f), f"Missing hf_space file: {hf_f}")
            with open(root_f, "rb") as f1, open(hf_f, "rb") as f2:
                self.assertEqual(
                    f1.read(),
                    f2.read(),
                    f"{rel} must be byte-identical between root and hf_space/",
                )

        # Standalone load & execution of hf_space/models/pattern_prober.py and generator_viz.py
        prober_path = os.path.join(HF_SPACE_DIR, "models", "pattern_prober.py")
        viz_path = os.path.join(HF_SPACE_DIR, "models", "generator_viz.py")
        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec_p = importlib.util.spec_from_file_location("hf_space_pattern_prober", prober_path)
            mod_p = importlib.util.module_from_spec(spec_p)
            sys.modules["hf_space_pattern_prober"] = mod_p
            spec_p.loader.exec_module(mod_p)

            spec_v = importlib.util.spec_from_file_location("hf_space_generator_viz", viz_path)
            mod_v = importlib.util.module_from_spec(spec_v)
            sys.modules["hf_space_generator_viz"] = mod_v
            spec_v.loader.exec_module(mod_v)

            prober = mod_p.GeneratorPatternProber()
            viz = mod_v.GeneratorFingerprintVisualizer()
            sample = "Табиғи ресурстарды тиімді пайдалану мәселелері маңызды бағыттардың бірі болып табылады."
            probe_res = prober.probe(sample)
            self.assertGreater(probe_res.copula_ratio, 0.0)
            svg = viz.render_radar_chart_svg(
                {"Target": prober.normalize_for_radar(probe_res)},
                labels=list(mod_p.RADAR_AXIS_KEYS),
            )
            self.assertIn("<polygon", svg)
        finally:
            sys.path = old_sys_path
            sys.modules.pop("hf_space_pattern_prober", None)
            sys.modules.pop("hf_space_generator_viz", None)

    def test_phase4_domain_calibrator_and_presets_parity(self):
        """Verifies Phase 4 domain_calibrator.py, ui/presets.py, and ui/app.py maintain byte parity with hf_space/."""
        import importlib.util

        rel_files = [
            os.path.join("models", "domain_calibrator.py"),
            os.path.join("ui", "presets.py"),
            os.path.join("ui", "app.py"),
        ]
        for rel in rel_files:
            root_f = os.path.join(PROJECT_ROOT, rel)
            hf_f = os.path.join(HF_SPACE_DIR, rel)
            self.assertTrue(os.path.isfile(hf_f), f"Missing hf_space file: {hf_f}")
            with open(root_f, "rb") as f1, open(hf_f, "rb") as f2:
                self.assertEqual(
                    f1.read(),
                    f2.read(),
                    f"{rel} must be byte-identical between root and hf_space/",
                )

        cal_path = os.path.join(HF_SPACE_DIR, "models", "domain_calibrator.py")
        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, HF_SPACE_DIR)
            spec_c = importlib.util.spec_from_file_location("hf_space_domain_calibrator", cal_path)
            mod_c = importlib.util.module_from_spec(spec_c)
            sys.modules["hf_space_domain_calibrator"] = mod_c
            spec_c.loader.exec_module(mod_c)

            calibrator = mod_c.DomainCalibrator()
            badge = calibrator.get_domain_robustness_badge("news", lang="en")
            self.assertIn("FPR<=5% Verified", badge["badge_text"])
        finally:
            sys.path = old_sys_path
            sys.modules.pop("hf_space_domain_calibrator", None)


if __name__ == "__main__":
    unittest.main()
