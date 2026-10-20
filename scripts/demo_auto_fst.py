"""Auto-FST Demonstration Console: Automated Benchmark Suite & Interactive REPL.

Zero-GPU, ultra-lightweight adaptive morphological parser for code-switched and
Russian-layout informal Kazakh text.
"""

import sys
import os
import time

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fst_analyzer import fst_analyzer as baseline_fst
from src.auto_fst.normalizer import RussianLayoutNormalizer
from src.auto_fst.inducer import AutoFSTInducer
from src.auto_fst.guard import RegressionGuard
from src.auto_fst.overlay import DynamicFSTOverlay

BENCHMARK_SAMPLES = [
    {
        "category": "Russian Keyboard Diacritic Flattening",
        "text": "салем калайсын, жумысын калай болып жатыр?",
        "note": "Replaces standard Russian layout letters with authentic Kazakh diacritics"
    },
    {
        "category": "Russian Keyboard Particle Agglutination",
        "text": "Ол али жумыста турсынгой, кешке келедигой.",
        "note": "Resolves flattened -сын and -гой modal particles without suffix collision"
    },
    {
        "category": "Emerging English/Russian Slang Verbalization",
        "text": "Биз кеше достармен катты вайбтадык жане чиллдедик.",
        "note": "Auto-induces loan stems 'вайб' and 'чилл' and decomposes verbalizer chains"
    },
    {
        "category": "Youth Consumer & Gaming Slang",
        "text": "Ойынга коп донаттау жасады, енди озинин крашына жазады.",
        "note": "Auto-induces 'донат' (voiceless coda) and 'краш' (noun possessive case)"
    },
    {
        "category": "Compound Disambiguation (Longest-Match First)",
        "text": "Чатботтын жауабын каспиредпен толедик.",
        "note": "Prevents 'чат' from shadowing 'чатбот', isolates 'каспиред'"
    },
    {
        "category": "Noise & Typo Rejection Guard",
        "text": "Бббазар123 жоккк соз емес.",
        "note": "Rejects malformed tokens without corrupting morphological lexicon"
    }
]


def run_benchmark():
    print("=" * 80)
    print("AUTO-FST AUTOMATED BENCHMARK SUITE")
    print("Adaptive Neuro-Symbolic Morphology for Informal & Code-Switched Kazakh")
    print("Hardware: Standard CPU (0 MB GPU VRAM required) | Platform: Lightweight")
    print("=" * 80)

    normalizer = RussianLayoutNormalizer()
    inducer = AutoFSTInducer()
    guard = RegressionGuard()
    overlay = DynamicFSTOverlay(normalizer=normalizer)
    lex_path = os.path.join("data", "custom_slang_lexicon.json")
    if os.path.exists(lex_path):
        loaded = overlay.load_from_json(lex_path)
        if loaded:
            print(f"Loaded {loaded} existing custom slang roots from {lex_path}\n")

    for i, item in enumerate(BENCHMARK_SAMPLES, 1):
        print(f"[{i}/6] Category: {item['category']}")
        print(f"Note : {item['note']}")
        print(f"Input: {item['text']}")

        # 1. Baseline Parse
        t0 = time.perf_counter()
        baseline_parse = baseline_fst.analyze_and_segment(item['text'])
        baseline_ms = (time.perf_counter() - t0) * 1000

        # 2. Check for candidate slang words and induce
        words = item['text'].split()
        induced_stems = []
        for w in words:
            cand = inducer.induce_stem(w)
            if cand:
                passed, reason = guard.validate_candidate(cand.stem)
                if passed:
                    overlay.register_root(cand.stem, {"harmony": cand.harmony, "coda": cand.coda})
                    induced_stems.append(f"{cand.stem} ({cand.harmony}, {cand.coda})")

        # 3. Auto-FST Parse
        t1 = time.perf_counter()
        auto_parse = overlay.analyze_and_segment(item['text'])
        auto_ms = (time.perf_counter() - t1) * 1000

        print(f"Before (Baseline FST) : {baseline_parse} ({baseline_ms:.2f} ms)")
        print(f"After  (Auto-FST)     : {auto_parse} ({auto_ms:.2f} ms)")
        if induced_stems:
            print(f"Auto-Inductions       : {', '.join(induced_stems)}")
        else:
            print(f"Auto-Inductions       : [None - Pure Orthographic / Existing Rule]")
        print("-" * 80)

    overlay.save_to_json(lex_path)
    print(f"Benchmark completed successfully. Updated lexicon saved to {lex_path}.\n")


def run_repl():
    print("=" * 80)
    print("AUTO-FST INTERACTIVE CONSOLE")
    print("Type any informal or code-switched Kazakh text. Type 'exit' to return.")
    print("=" * 80)

    normalizer = RussianLayoutNormalizer()
    inducer = AutoFSTInducer()
    guard = RegressionGuard()
    overlay = DynamicFSTOverlay(normalizer=normalizer)
    lex_path = os.path.join("data", "custom_slang_lexicon.json")
    if os.path.exists(lex_path):
        overlay.load_from_json(lex_path)

    while True:
        try:
            user_input = input("Auto-FST > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            t0 = time.perf_counter()
            # Analyze words
            words = user_input.split()
            for w in words:
                cand = inducer.induce_stem(w)
                if cand:
                    passed, reason = guard.validate_candidate(cand.stem)
                    if passed:
                        overlay.register_root(cand.stem, {"harmony": cand.harmony, "coda": cand.coda})
                        print(f"  [Auto-Induction] Root '{cand.stem}' -> Suffixes: {cand.suffixes} | Harmony={cand.harmony}, Coda={cand.coda}")
                    else:
                        print(f"  [Guard Rejected] Candidate '{cand.stem}': {reason}")

            result = overlay.analyze_and_segment(user_input)
            latency = (time.perf_counter() - t0) * 1000
            print(f"  Result : {result}")
            print(f"  Latency: {latency:.2f} ms\n")
        except (KeyboardInterrupt, EOFError):
            break

    overlay.save_to_json(lex_path)
    print("\nSession ended.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "--repl":
        run_repl()
    else:
        run_benchmark()
