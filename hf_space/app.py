# -*- coding: utf-8 -*-
"""
hf_space/app.py: Hugging Face Spaces Entrypoint for Kazakh AI Text Detector.

Provides 1-click cloud deployment with sub-second cold start on Hugging Face Spaces
Free CPU tier (2 vCPU, 16GB RAM) using the deterministic OfflineHeuristicDetector
and pure-Python AdvancedKazakhFSTAnalyzer.
"""

import os
import sys

# Ensure hf_space root and project root are in sys.path for local module resolution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from ui.app import create_app

# Instantiate Gradio application in lightweight CPU/cold-start mode
demo = create_app(detector=None, load_model=False)

if __name__ == "__main__":
    demo.launch()
