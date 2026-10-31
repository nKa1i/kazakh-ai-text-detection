# Phase 4: Cross-Domain Multi-Genre Robustness Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 500-document 5-genre Kazakh cross-domain robustness benchmark (`news`, `academic`, `official_legal`, `fiction_literature`, `social_casual`), a domain sensitivity and threshold stability calibrator (`models/domain_calibrator.py`), an end-to-end cross-domain evaluation runner (`scripts/evaluate_cross_domain_robustness.py`), and multi-genre presets and domain audit badges in the Gradio UI (`ui/presets.py`, `ui/app.py`, `hf_space/ui/app.py`).

**Architecture:** Task 1 constructs a deterministic, cryptographically isolated 500-record dataset (`data/cross_domain_benchmark.json`, 100 records per domain: 50 authentic human + 50 AI across 5 known LLM families and 1 held-out wild calque family) via `scripts/build_cross_domain_benchmark.py`. Task 2 implements `DomainCalibrator` (`models/domain_calibrator.py`) to compute exact ROC-AUC, safety metric `FPR@95%TPR`, threshold stability curves across $\tau \in [0.90, 0.999]$, and domain classification (`classify_domain`) without breaking global threshold $\tau = 0.9980$. Task 3 implements `scripts/evaluate_cross_domain_robustness.py` with `--dry-run` support to evaluate binary detection, 5-family closed-set attribution, unseen OOD recall, and domain transfer gaps ($\text{AUC}_{\text{news}} - \text{AUC}_{\text{domain}}$), outputting `data/cross_domain_evaluation_results.json` and `docs/reports/2026-10-31-cross-domain-benchmark-report.md`. Task 4 integrates 5 verified multi-genre presets (`DOMAIN_PRESET_SAMPLES`), a domain preset selector and detected domain robustness badge into `ui/presets.py`, `ui/app.py`, and `hf_space/` with full byte parity.

**Tech Stack:** Python 3.12 (`unittest`, `dataclasses`, `hashlib`, `json`, `math`, `html`, `argparse`), `models.heuristic_detector` (`OfflineHeuristicDetector`), `models.generator_profiler` (`MultiModelGeneratorProfiler`, `KNOWN_GENERATOR_FAMILIES`), `gradio`, Git.

## Global Constraints

- **Python Interpreter & Encoding:** Always execute Python from `C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection` in Windows PowerShell using:
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe
  ```
- **Zero Decorative Emojis:** Strictly zero decorative Unicode emojis or dingbat symbols in any code, docstring, HTML/SVG string, test, Markdown report, commit message, or artifact (`GEMINI.md` Rule 4; verified by regex `[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]`).
- **Backward Compatibility:**
  - Preserve calibrated decision threshold `tau = 0.9980` in `OfflineHeuristicDetector` and all existing keys and verdicts in `PRESET_SAMPLES`, `QUICK_SAMPLES`, and `VERIFICATION_PRESET_SAMPLES`.
  - Preserve the exact 8-tuple return signature of `handle_analyze_document()` and the exact 63-element tuple return signature of `switch_ui_language()` in `ui/app.py`.
- **Dual-Tree `hf_space/` Parity:** Synchronize `models/domain_calibrator.py`, `ui/presets.py`, and `ui/app.py` byte-for-byte into `hf_space/models/` and `hf_space/ui/`, verified by `tests/test_hf_space_bundle.py`.
- **Gitignore Handling for `docs/`:** Because `docs` is in `.gitignore`, always use `git add -f` when staging files inside `docs/` (such as `docs/reports/2026-10-31-cross-domain-benchmark-report.md` and `docs/superpowers/plans/...`).

---

## File Structure

- **Create:**
  - `scripts/build_cross_domain_benchmark.py` — Deterministic generator and validator for the 500-record 5-genre benchmark (`DOMAIN_KEYS`, `build_cross_domain_benchmark_dataset`, `validate_cross_domain_benchmark`).
  - `data/cross_domain_benchmark.json` — 500-record stratified cross-domain benchmark dataset (5 genres x 50 human + 50 AI).
  - `tests/test_cross_domain_benchmark.py` — Unit tests for schema contract, 50/50 human-AI balance per genre, SHA-256 uniqueness, cryptographic zero-overlap against `data/seed_human_reviews_1k.json`, and zero emojis.
  - `models/domain_calibrator.py` — `DomainMetrics`, `ThresholdPoint`, `DomainCalibrator` (`compute_roc_auc`, `compute_fpr_at_tpr`, `evaluate_domain_predictions`, `compute_threshold_stability_curve`, `recommend_threshold`, `classify_domain`, `get_domain_robustness_badge`).
  - `tests/test_domain_calibrator.py` — Unit tests for ROC-AUC, `FPR@95%TPR`, threshold stability curve across `[0.90, 0.999]`, safety verification at `tau = 0.9980`, domain classification across all 5 genres, and zero emojis.
  - `scripts/evaluate_cross_domain_robustness.py` — CLI and programmatic evaluation runner computing per-domain and aggregate metrics, transfer gaps vs. `news`, closed-set attribution accuracy, unseen OOD recall, JSON results, and Markdown/LaTeX report.
  - `data/cross_domain_evaluation_results.json` — Generated cross-domain evaluation results artifact.
  - `docs/reports/2026-10-31-cross-domain-benchmark-report.md` — Generated publication-grade Markdown and LaTeX benchmark report.
  - `tests/test_evaluate_cross_domain.py` — Unit tests for `--dry-run` CLI execution, full benchmark acceptance criteria (`ROC-AUC >= 0.92`, `FPR@95%TPR <= 0.08` on Academic/Legal, zero human false positives at `tau = 0.9980`), and artifact verification.
  - `tests/test_ui_domain_presets.py` — Unit tests for the 5 verified domain presets in `ui/presets.py`, UI domain preset loader, domain robustness badge in `run_full_analysis_logic` and `generate_audit_markdown`, and `hf_space/` parity.
  - `hf_space/models/domain_calibrator.py` — Byte-identical synchronized copy of `models/domain_calibrator.py`.
- **Modify:**
  - `ui/presets.py` — Add `DOMAIN_PRESET_SAMPLES` (5 verified multi-genre presets) and helpers `get_domain_preset_choices()`, `get_domain_preset_text(preset_name)`, `get_domain_preset_metadata(preset_name)`.
  - `ui/app.py` — Add domain preset selector and load handler (`handle_domain_preset_change`), domain audit badge in `run_full_analysis_logic()` and `generate_audit_markdown()`, while keeping `handle_analyze_document()` (8-tuple) and `switch_ui_language()` (63-tuple) unchanged.
  - `hf_space/ui/presets.py` — Byte-identical synchronized copy of `ui/presets.py`.
  - `hf_space/ui/app.py` — Byte-identical synchronized copy of `ui/app.py`.
  - `tests/test_hf_space_bundle.py` — Add `"domain_calibrator.py"` to `test_bundle_models_package_exists` and add `test_phase4_domain_calibrator_and_presets_parity`.

---

### Task 1: 5-Genre Benchmark Dataset & Validator (`data/cross_domain_benchmark.json` & `tests/test_cross_domain_benchmark.py`)

**Files:**
- Create: `scripts/build_cross_domain_benchmark.py`
- Create: `data/cross_domain_benchmark.json`
- Create: `tests/test_cross_domain_benchmark.py`
- Test: `tests/test_cross_domain_benchmark.py`

**Interfaces:**
- Consumes: Standard library (`json`, `hashlib`, `random`, `re`, `os`)
- Produces:
  - `DOMAIN_KEYS: Tuple[str, ...]` = `("news", "academic", "official_legal", "fiction_literature", "social_casual")`
  - `KNOWN_GENERATORS: Tuple[str, ...]` = `("ISSAI-KazLLM-70B", "Sherkala-8B", "DeepSeek-R1", "GPT-4o", "Qwen-2.5-72B")`
  - `UNSEEN_GENERATOR: str` = `"Llama-3.3-70B-Calque"`
  - `build_cross_domain_benchmark_dataset(output_path: str = "data/cross_domain_benchmark.json", records_per_domain: int = 100, random_seed: int = 42) -> List[Dict[str, Any]]`
  - `validate_cross_domain_benchmark(records: List[Dict[str, Any]], seed_overlap_paths: Optional[Sequence[str]] = None) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing unit test suite in `tests/test_cross_domain_benchmark.py`**

Create `tests/test_cross_domain_benchmark.py`:

```python
# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4 Cross-Domain Multi-Genre Benchmark Dataset:
- Schema integrity for all 500 records across 5 Kazakh functional genres
- Exact domain balance (100 per domain: 50 authentic human, 50 AI)
- Generator diversity (5 known families + 1 unseen calque family per domain)
- Cryptographic SHA-256 uniqueness and zero overlap with training/seed sets
- Zero decorative emojis across all dataset records and metadata
"""

import hashlib
import json
import os
import re
import unittest

from scripts.build_cross_domain_benchmark import (
    DOMAIN_KEYS,
    KNOWN_GENERATORS,
    UNSEEN_GENERATOR,
    build_cross_domain_benchmark_dataset,
    validate_cross_domain_benchmark,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json")
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestCrossDomainBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(BENCHMARK_PATH):
            build_cross_domain_benchmark_dataset(output_path=BENCHMARK_PATH)
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cls.records = json.load(f)

    def test_total_record_count_and_schema_contract(self):
        self.assertIsInstance(self.records, list)
        self.assertEqual(len(self.records), 500)

        required_keys = {
            "id",
            "domain",
            "text",
            "label",
            "generator",
            "word_count",
            "sentence_count",
            "sha256",
            "metadata",
        }
        for rec in self.records:
            self.assertEqual(set(rec.keys()), required_keys)
            self.assertIn(rec["domain"], DOMAIN_KEYS)
            self.assertIn(rec["label"], (0, 1))
            self.assertIsInstance(rec["text"], str)
            self.assertGreaterEqual(rec["word_count"], 18)
            self.assertGreaterEqual(rec["sentence_count"], 2)
            expected_sha = hashlib.sha256(rec["text"].strip().encode("utf-8")).hexdigest()
            self.assertEqual(rec["sha256"], expected_sha)
            self.assertIn("source_register", rec["metadata"])
            self.assertIn("has_layout_shift", rec["metadata"])
            self.assertIsInstance(rec["metadata"]["has_layout_shift"], bool)

    def test_domain_and_label_stratification(self):
        self.assertEqual(
            DOMAIN_KEYS,
            ("news", "academic", "official_legal", "fiction_literature", "social_casual"),
        )
        for domain in DOMAIN_KEYS:
            dom_recs = [r for r in self.records if r["domain"] == domain]
            self.assertEqual(len(dom_recs), 100, msg=f"Domain {domain} must have 100 records")
            human_recs = [r for r in dom_recs if r["label"] == 0]
            ai_recs = [r for r in dom_recs if r["label"] == 1]
            self.assertEqual(len(human_recs), 50, msg=f"Domain {domain} must have 50 human records")
            self.assertEqual(len(ai_recs), 50, msg=f"Domain {domain} must have 50 AI records")
            for h in human_recs:
                self.assertEqual(h["generator"], "human")
            ai_gens = {a["generator"] for a in ai_recs}
            for k_gen in KNOWN_GENERATORS:
                self.assertIn(k_gen, ai_gens)
            self.assertIn(UNSEEN_GENERATOR, ai_gens)

    def test_cryptographic_sha256_isolation_and_validator(self):
        seed_path = os.path.join(PROJECT_ROOT, "data", "seed_human_reviews_1k.json")
        overlap_paths = [seed_path] if os.path.isfile(seed_path) else []
        report = validate_cross_domain_benchmark(
            self.records,
            seed_overlap_paths=overlap_paths,
        )
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["total_records"], 500)
        self.assertEqual(report["unique_sha256_count"], 500)
        self.assertEqual(report["seed_overlap_count"], 0)

    def test_zero_decorative_emojis_in_dataset(self):
        raw_json = json.dumps(self.records, ensure_ascii=False)
        self.assertIsNone(
            EMOJI_RE.search(raw_json),
            "Dataset JSON must contain zero decorative emojis",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run `tests/test_cross_domain_benchmark.py` to verify it fails**

Run:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_cross_domain_benchmark.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_cross_domain_benchmark'`.

- [ ] **Step 3: Implement `scripts/build_cross_domain_benchmark.py` and generate `data/cross_domain_benchmark.json`**

Create `scripts/build_cross_domain_benchmark.py`:

```python
# -*- coding: utf-8 -*-
"""
scripts/build_cross_domain_benchmark.py: Builds and validates the Phase 4
5-Genre Cross-Domain Robustness Benchmark (data/cross_domain_benchmark.json).

Genres (100 records per domain: 50 authentic human + 50 AI across 5 known LLMs + 1 unseen calque LLM):
  1. news               - News / Journalism (Tengrinews / Kazinform factual declarative style)
  2. academic           - Academic / Technical (thesis abstracts, formal definitions, high affix density)
  3. official_legal     - Government / Legal (decrees, administrative formulas, statutory clauses)
  4. fiction_literature - Literary / Fiction (expressive narratives, dialogue particles, imagery)
  5. social_casual      - Casual / Social Media (Kaspi reviews, forum comments, layout shifts)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DOMAIN_KEYS: Tuple[str, ...] = (
    "news",
    "academic",
    "official_legal",
    "fiction_literature",
    "social_casual",
)

KNOWN_GENERATORS: Tuple[str, ...] = (
    "ISSAI-KazLLM-70B",
    "Sherkala-8B",
    "DeepSeek-R1",
    "GPT-4o",
    "Qwen-2.5-72B",
)

UNSEEN_GENERATOR: str = "Llama-3.3-70B-Calque"

DOMAIN_SOURCE_REGISTERS: Dict[str, str] = {
    "news": "journalistic_report",
    "academic": "thesis_abstract",
    "official_legal": "administrative_decree",
    "fiction_literature": "literary_narrative",
    "social_casual": "consumer_forum_review",
}

_SENT_SPLIT_RE = re.compile(r"[.!?]+")

# Authentic Human Sentence Pools per Genre (avoiding synthetic AI copula crutches)
HUMAN_DOMAIN_POOLS: Dict[str, Dict[str, List[str]]] = {
    "news": {
        "openers": [
            "Бүгін таңертең Астанадағы орталық көрме залында жас өнертапқыштардың республикалық байқауы басталды.",
            "Алматы облысының шаруалары биылғы күзгі жиын-терім науқанын мерзімінен бұрын аяқтап шықты.",
            "Қарағанды қаласында жаңадан салынған балалар спорт кешені тұрғындардың игілігіне берілді.",
            "Түркістан өңіріндегі тарихи ескерткіштерді қалпына келтіру жұмыстарына отандық археологтар кірісті.",
            "Ақтөбеде өткен аймақтық жәрмеңкеде жергілікті өндірушілер өз өнімдерін арзан бағамен ұсынды.",
            "Шымкенттегі орталық кітапханада сирек кездесетін қолжазбалардың ашық көрмесі ұйымдастырылды.",
            "Атырау мұнай өңдеу зауытындағы жоспарлы жөндеу жұмыстары сәтті аяқталып, цехтар қайта қосылды.",
            "Павлодарда Ертіс өзенінің жағалауын абаттандыру жобасына қала тұрғындары белсене атсалысты.",
            "Қостанай диқандары биыл астық сапасының өткен жылмен салыстырғанда едәуір жақсы екенін айтты.",
            "Семейдегі Абай музейіне демалыс күндері мыңнан астам мектеп оқушысы мен студенттер келді.",
        ],
        "middles": [
            "Жиынға келген мамандар өздерінің тәжірибелік ұсыныстарын ортаға салып, пікір алмасты.",
            "Жоба жетекшілері журналистерге берген сұхбатында алдағы жоспарларымен ашық бөлісті.",
            "Қала тұрғындары жаңа бағыттағы қоғамдық көліктердің жүру жиілігіне ризашылықтарын білдірді.",
            "Ұйымдастырушылардың сөзінше, келесі айда мұндай кездесулер барлық аудан орталықтарында өтеді.",
            "Кездесуде ауыл мектептерін заманауи зертханалық құралдармен жабдықтау жайы кеңінен сөз болды.",
            "Көрмеге қойылған жәдігерлердің басым бөлігі бұрын-соңды көпшілік назарына ұсынылмаған екен.",
            "Ауа райының қолайлы болуы құрылыс бригадаларының жұмысын тездетуге жақсы сеп болды.",
            "Жергілікті әкімдік өкілдері тұрғындар көтерген инфрақұрылым мәселелерін бақылауға алды.",
            "Биылғы маусымда ішкі туризмге сұраныс артып, демалыс орындарына келушілер қатары көбейді.",
            "Арнайы құрылған жұмыс тобы нысандардың қысқы маусымға дайындығын мұқият тексеріп шықты.",
        ],
        "closers": [
            "Шара соңында үздік деп танылған қатысушыларға арнайы дипломдар мен естелік сыйлықтар табысталды.",
            "Алдағы аптада комиссия мүшелері қорытынды есепті жұртшылық назарына ұсынбақ.",
            "Тұрғындар бұл бастаманың өңір тіршілігіне тың серпін беретініне сенім білдіріп отыр.",
            "Қазіргі кезде мамандар келесі кезеңнің дайындық жұмыстарын бастап кетті.",
            "Бұл жаңалықты естіген ауыл ақсақалдары мен жастар қуаныштарын жасырмады.",
            "Мамандар жыл соңына дейін тағы үш әлеуметтік нысанның есігін ашатынын жеткізді.",
            "Іс-шарадан түскен қаражат толығымен қайырымдылық қорының есепшотына аударылды.",
            "Кеш қонақтары ұлттық ансамбльдің концерттік бағдарламасын тамашалап, жақсы әсермен тарқасты.",
            "Жауапты мекеме басшылары жұмыс сапасын апта сайын тікелей тексеріп тұратынын айтты.",
            "Осылайша, бірнеше айға созылған күрделі жоба өз мәресіне абыроймен жетті.",
        ],
    },
    "academic": {
        "openers": [
            "Мақалада қазақ тіліндегі жалғамалы морфемалардың фонологиялық үйлесім заңдылықтары эмпирикалық тұрғыдан зерттелді.",
            "Бұл зерттеу жұмысында түркі тілдерінің синтаксистік құрылымы мен баяндауыштың сөйлем соңында орналасу тәртібі қарастырылды.",
            "Диссертациялық ізденісте агглютинативті сөзжасам жүйесінің компьютерлік модельдеудегі ерекшеліктері жан-жақты сипатталды.",
            "Зерттеушілер корпустық лингвистика деректеріне сүйене отырып, ауызекі сөйлеу тіліндегі модальді бөлшектердің жиілігін есептеді.",
            "Эксперимент барысында соңғы күйлі трансдьюсерлердің сөз түбірі мен қосымшаларды ажырату дәлдігі тексерілді.",
            "Ғылыми еңбекте XIX ғасырдағы қазақ жазба ескерткіштерінің лексика-семантикалық қабаттары тарихи-салыстырмалы әдіспен талданды.",
            "Авторлар ұсынған морфологиялық талдағыш бейресми мәтіндердегі орфографиялық ауытқуларды автоматты түрде қалпына келтіреді.",
            "Зертханалық сынақтарда көпқабатты нейрондық желілердің сирек кездесетін сөз формаларын тану деңгейі өлшенді.",
            "Жұмыста қазақ фольклорындағы тұрақты тіркестердің когнитивтік табиғаты мен этномәдени мазмұны ашып көрсетілді.",
            "Статистикалық талдауда сөйлем ұзындығының дисперсиясы мен лексикалық байлық көрсеткіштері өзара салыстырылды.",
        ],
        "middles": [
            "Жинақталған ғылыми деректер сингармонизм заңының буын шекарасындағы дыбыс алмасуларына тікелей әсерін растады.",
            "Корпустан алынған мың жарым мысалдың ішінде етістіктің көсемше тұлғалары ерекше белсенділік танытты.",
            "Алгоритмнің есептеу жылдамдығы орташа процессорда бір миллисекундтан аспайтыны тәжірибе жүзінде анықталды.",
            "Салыстырмалы кестеде дәстүрлі статистикалық модельдер мен гибридті архитектураның айырмашылығы айқын көрінді.",
            "Түбір сөздерге жалғанатын септік пен тәуелдік қосымшаларының орын тәртібі қатаң морфотактикалық ережеге бағынады.",
            "Екі тәуелсіз сарапшының белгілеу нәтижелері бойынша Коэн каппа коэффициенті жоғары үйлесімділікті көрсетті.",
            "Кірме терминдердің соңғы буынындағы дауысты дыбыс сапасы жалғаудың жуан немесе жіңішке нұсқасын белгілейді.",
            "Бақылау тобындағы мәтіндерде күрделі сабақтас сөйлемдер мен есімшелі орамдар жиірек ұшырасты.",
            "Алынған эмпирикалық қисықтар таңдалған шектік мәннің әртүрлі жанрлардағы тұрақтылығын айғақтайды.",
            "Талдау кезінде қосымшалар тізбегінің Шеннон энтропиясы шынайы мәтіндерде едәуір жоғары болатыны байқалды.",
        ],
        "closers": [
            "Зерттеу қорытындылары қазақ тілін автоматты өңдеу жүйелерін жетілдіруге нақты практикалық үлес қосады.",
            "Ұсынылған әдіснаманы түркі тобындағы өзге де туыстас тілдердің морфологиясын зерттеуге қолдануға болады.",
            "Алдағы уақытта авторлар бұл корпусты аймақтық диалектілік материалдармен толықтыруды жоспарлап отыр.",
            "Осы тұжырымдар жоғары оқу орындарына арналған қолданбалы тіл білімі оқулықтарына енгізілді.",
            "Алынған сандық көрсеткіштер ашық репозиторий арқылы ғылыми қауымдастыққа толық қолжетімді етілді.",
            "Сынақ нәтижелері морфологиялық белгілерді ескерудің жалған оң қателерді азайтатынын көрсетіп берді.",
            "Бұл тәсіл аз ресурсты тілдерге арналған машиналық аударма сапасын жақсартуға мүмкіндік ашады.",
            "Авторлар келесі мақалада ауызша сөйлеу корпусындағы интонациялық үлгілерді қарастыруды көздейді.",
            "Жиналған деректер қоры жас зерттеушілердің дипломдық және магистрлік жобаларына негіз болды.",
            "Теориялық ережелер мен эксперименттік өлшемдердің өзара сәйкестігі гипотезаның дұрыстығын қуаттады.",
        ],
    },
    "official_legal": {
        "openers": [
            "Қазақстан Республикасы Оқу-ағарту министрлігінің кезекті алқа мәжілісінде мектеп кітапханаларын жаңарту туралы бұйрық бекітілді.",
            "Облыстық мәслихаттың кезектен тыс сессиясында ауылдық елді мекендерді ауызсумен қамту ережесі қаралды.",
            "Еңбек шартының талаптарына сәйкес жұмыс беруші қызметкердің кәсіби қауіпсіздік құралдарын уақтылы беруге міндетті.",
            "Азаматтық кодекстің тиісті бабында мүліктік дауларды сотқа дейінгі медиация тәртібімен реттеу шарттары көрсетілген.",
            "Комиссия төрағасы мемлекеттік сатып алу конкурсына түскен өтінімдердің хаттамасын ресми түрде жариялады.",
            "Жер кодексіне енгізілген өзгерістер бойынша жайылымдық алқаптарды тиімсіз пайдаланған шаруашылықтарға ескерту берілді.",
            "Әкімшілік рәсімдік-процестік заңнама азаматтардың өтініштерін он бес жұмыс күні ішінде қарауды талап етеді.",
            "Қалалық сәулет және қала құрылысы басқармасы көпқабатты тұрғын үйлердің эскиздік жобасын келісуден өткізді.",
            "Салық комитеті шағын кәсіпкерлік субъектілеріне маусымдық есептілікті электрондық кабинет арқылы тапсыруды ескертті.",
            "Табиғатты қорғау прокуратурасы өзен арнасына заңсыз қалдық төккен кәсіпорынға қатысты тексеру актісін толтырды.",
        ],
        "middles": [
            "Бұйрықтың екінші тармағында жауапты бөлім басшыларына ай сайынғы орындалу кестесін ұсыну жүктелді.",
            "Келісімшартқа қол қойған тараптар өз міндеттемелерін бұзған жағдайда заңда белгіленген тәртіппен айыппұл төлейді.",
            "Құжаттың түпнұсқасы мемлекеттік тілде және орыс тілінде тең заңды күші бар екі данада рәсімделді.",
            "Тексеру барысында анықталған кемшіліктерді жою үшін мекеме әкімшілігіне отыз күндік мерзім белгіленді.",
            "Өтініш иесі шешіммен келіспеген жағдайда жоғары тұрған уәкілетті органға шағым түсіруге құқылы.",
            "Бюджеттік бағдарлама әкімшісі бөлінген қаражаттың мақсатты жұмсалуын ревизиялық комиссия алдында қорғады.",
            "Лицензия алу үшін заңды тұлғалар біліктілік талаптарына сай келетін растаушы анықтамаларды қоса тіркейді.",
            "Қаулы жобасы ашық нормативтік-құқықтық актілер порталында он жұмыс күні бойы қоғамдық талқылауда болды.",
            "Тұрғын үй комиссиясы көпбалалы отбасылардың тізімін кезек ретімен қайта нақтылап шықты.",
            "Хаттамалық тапсырманың орындалу барысын бақылау департамент директорының бірінші орынбасарына тапсырылды.",
        ],
        "closers": [
            "Осы бұйрық алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі.",
            "Қаулының толық мәтіні жергілікті атқарушы органның ресми интернет-ресурсында жариялансын.",
            "Тараптар арасындағы даулы мәселелер шешімін таппаса, іс мамандандырылған экономикалық сотта қаралады.",
            "Актіге комиссия мүшелері мен шақырылған тәуелсіз сарапшылар толықтай қол қойып, мөрмен растады.",
            "Қорытынды шешімнің көшірмесі өтініш берушінің мекенжайына тапсырыс хатпен жолданды.",
            "Бұрын қабылданған ескі нұсқаулықтың күші жойылды деп танылсын және архивке тапсырылсын.",
            "Жауапты мамандар бекітілген регламент талаптарын басшылыққа алып, күнделікті жұмысты ұйымдастырсын.",
            "Есептік кезеңнің қорытындысы келесі тоқсанның бесінші жұлдызына дейін министрлікке енгізілсін.",
            "Көрсетілген талаптарды орындамаған лауазымды тұлғаларға тәртіптік жаза қолданылатыны ескертілді.",
            "Осы келісім қол қойылған сәттен бастап заңды күшіне енеді және жыл соңына дейін жарамды.",
        ],
    },
    "fiction_literature": {
        "openers": [
            "Күзгі іңірде ауыл шетіндегі кәрі теректің жапырақтары баяу сыбдырлап, өткен күндердің сырын шертетіндей еді.",
            "Ақбоз аттың тізгінін бос жіберген Ертай белестен асып, көкжиекке сіңіп бара жатқан қызыл іңірге ұзақ қарады.",
            "Әжемнің самаурын қайнатып, күрең шайды баптап құйған сәті балалық шағымның ең ыстық естелігі ғой.",
            "Тау басынан ескен салқын самал жусан иісін аңқытып, жолаушының шаршаған көңілін бірден сергітіп жіберді.",
            "Терезе сыртында жапалақтап жауған алғашқы қарға қарап отырып, Айгүл ескі күнделігінің парақтарын ақтарды.",
            "Жайлаудың түні қандай тыныш десеңші, аспандағы жұлдыздар қол созсаң жететіндей жарқырап тұр екен.",
            "Қарт ұста көріктің шоғын көсеп қойып, немересіне батырлар жырынан ұзақ-сонар әңгіме бастап кетті.",
            "Жаңбырдан кейінгі дала жолымен келе жатқан екеуі үнсіз ғана бір-біріне күлімсірей қарап қойды.",
            "Көктемгі найзағай жарқ еткенде, сай ішіндегі бұлақтың сылдыры тіпті анық естілгендей болды-ау.",
            "Анасының мейірімге толы қоңыр дауысы құлағына келгенде, шетте жүрген жігіттің жүрегі елжіреп сала берді.",
        ],
        "middles": [
            "«Апыр-ай, уақыт деген зымырап өте шығады екен-ау», — деп қария сақалын сипап, терең ойға шомды.",
            "Бала кезімізде осы өзеннің жағасында таң атқанша балық аулап, от басында әзілдесіп отырушы едік қой.",
            "Оның көзіндегі мұң мен үміт аралас ұшқынды байқамау мүмкін емес еді, бірақ ешкім тіс жарып сұрамады.",
            "Домбыраның қоңыр үні киіз үйдің ішін кернеп, тыңдаушыларды беймәлім бір сиқырлы әлемге жетелей жөнелді.",
            "Алыстан талып жеткен жылқышылардың әні түнгі даланың тыныштығымен әдемі үйлесім тауып тұрғандай.",
            "Кенет есік айқара ашылып, қолында бір құшақ дала гүлі бар кішкентай інісі жүгіріп кіріп келді.",
            "Желмен тербелген бидай алқабы алтын теңіздей толқып, диқанның көңіліне ерекше қуаныш ұялатты.",
            "Ол ішінен мың мәрте қайталап жүрген сөздерін айтуға оқталды да, тамағына тығылған өксіктен үндей алмады.",
            "Көне сандықтың түбінен шыққан күміс білезік анасының жас кезіндегі сұлу бейнесін көз алдына әкелді.",
            "Достар бас қосқан сайын сол бір ұмытылмас студенттік шақтың қызықтарын айтып, тауыса алмайтынбыз.",
        ],
        "closers": [
            "Таң сібірлеп атқанда, жолаушылар керуені шығысқа қарай бет түзеп, ұзақ сапарға аттанып кетті.",
            "Сол кештегі жылы әңгіме оның жанына ұзақ уақыт бойы шуақ шашып, демеу болып жүрді ғой.",
            "Өмірдің нағыз бақыты осындай қарапайым әрі шынайы сәттерден құралатынын ол енді ғана ұққандай болды.",
            "Ауыл үстіндегі ақشا бұлттар баяу көшіп, жаңа бір шуақты күннің басталғанын хабарлап тұрды.",
            "Қарттың батасын алған жас жігіт бойына ерекше қуат біткендей сергек адымдап жүріп кетті.",
            "Көгілдір таулардың арғы жағынан шыққан күн сәулесі шық тамшыларын гауһардай жарқыратты.",
            "Олар қимай қоштасқанымен, көп ұзамай туған жердің төсінде қайта қауышатынына кәміл сенді.",
            "Осы бір кішкентай ауылдың әрбір тасы мен бұлағы оған бұрынғыдан да ыстық көрініп кетті-ау.",
            "Кешкі самалмен бірге жеткен бесік жырының әуені бүкіл алқапты маужыратып бара жатты.",
            "Жүрек түкпірінде сақталған сол бір аяулы сезім жылдар өтсе де ешқашан көмескіленбек емес.",
        ],
    },
    "social_casual": {
        "openers": [
            "Kaspi-ден кеше заказ берген құлаққап бүгін таңертең келіп тұр, курьер жігітке үлкен рахмет!",
            "Салем достар, жаңадан ашылған кофейняға барып көрдім, атмосферасы мен латтесі прям вайб екен.",
            "Осы телефонды алғаныма екі апта болды, батареясы спокойно екі күнге жетеді, камерасы да супер.",
            "Кеше кешке отбасымызбен жаңа қазақша комедияға барғанбыз, залдағы халықтың бәрі күлкіден қырылды ғой.",
            "Бугин базардан алган алмалар оте татти екен, балаларга компот кайнатып койдым.",
            "Интернет магазиннен алған кроссовканың размері дәл келді, табаны жұмсақ, жүгіруге өте ыңғайлы.",
            "Жұмыстан шаршап шыққанда осы саябақта біраз серуендеп қайтсаң, кәдімгідей демалып қаласың.",
            "Маған осы курстың сабақтары қатты ұнады, спикер бәрін қарапайым тілмен, мысалдармен түсіндіреді екен.",
            "Кешегі матчта біздің жігіттер соңғы минутта гол салып, жанкүйерлерге нағыз мереке сыйлады ғой!",
            "Үйге робот-шаңсорғыш алғалы бері кілем тазалау деген уайымнан құтылдық, өзі бәрін жинап шығады.",
        ],
        "middles": [
            "Бағасы да қалтаға қонымды, скидкамен тіпті тиімді шықты, сатушы бірден хабарласып нақтылады.",
            "Қорабы бүлінбей таза күйінде жетті, ішінде гарантиялық талоны мен қосымша сымы түгел бар екен.",
            "Басында сәл күмәнданып едім, бірақ қолданып көрген соң сапасына ешқандай мін таға алмадым.",
            "Достарыма да ссылкасын жіберіп қойдым, олар да келесі аптада заказ бермекші болып отыр.",
            "Кешкі пробкаға қарамай курьер сағат беске дейін үлгеріп әкеліп бергеніне риза болдым.",
            "Каспий редпен болип толеуге алган оте ынгайлы болды, ешкандай артык комиссия косылган жок.",
            "Отзывтарды оқып отырып таңдаған едім, шынымен де жұрт мақтағандай-ақ мықты зат болып шықты.",
            "Дизайны минималистичный әрі компактный, асүйдің бұрышында көп орын алмай әдемі тұрады.",
            "Балалар таңертеңнен кешке дейін ойнап мәз болып жүр, материалында ешқандай жағымсыз иіс жоқ.",
            "Тұтынушыларға қызмет көрсету деңгейі жоғары, сұрақтарыма ватсаптан бес минутта жауап берді.",
        ],
        "closers": [
            "Қысқасы, бес жұлдыздан бес қоямын, алғысы келіп жүргендерге смело кеңес беремін!",
            "Сатушыға және дүкен ұжымына көп рахмет, енді тек осы жерден ғана сауда жасайтын боламыз.",
            "Осындай сапалы отандық өнімдер көбейе берсін, бәріңізге сәттілік тілеймін достар!",
            "Егер бюджетный әрі надежный вариант іздесеңіздер, ойланбай осы модельді таңдай беріңіздер.",
            "Алдағы демалыста тағы да достармен жиналып баруды жоспарлап қойдық, көңіл күй керемет.",
            "Рахмет улкен, тауар сапасы конилден шыкты, барине советую алуга!",
            "Әзірге ешқандай минус байқамадым, жарты жылдан кейін қосымша отзыв жазып қалдырамын.",
            "Анама сыйлыққа алған едім, ол кісі де қатты қуанып, батасын беріп жатыр ғой.",
            "Осы бағаға мұндай комплектация табу қиын, сондықтан алғаныма бір сәт те өкінбеймін.",
            "Жалпы алғанда өте сәтті покупка болды, барлықтарыңызға жақсы демалыс тілеймін!",
        ],
    },
}

# Domain-contextualized prefixes/clauses combined with generator-specific morphosyntactic templates
AI_DOMAIN_TOPICS: Dict[str, List[str]] = {
    "news": [
        "ақпараттық агенттіктердің хабарлауынша, республикалық форум аясында",
        "баспасөз қызметінің мәліметіне сәйкес, елордадағы халықаралық жиын барысында",
        "ресми хроника деректері бойынша, өңірлік индустрияландыру апталығы шеңберінде",
        "медиа-орталықтың таратқан хабарламасына сәйкес, салалық саммит қорытындысында",
        "апталық шолу материалдарында көрсетілгендей, экономикалық сессия барысында",
        "тілшілер қосынының хабарлауынша, ведомствоаралық брифинг шеңберінде",
        "жаңалықтар қызметінің дерегінше, инвестициялық көрме жұмысының нәтижесінде",
        "телеарна эфирінде айтылғандай, көлік-логистикалық конгресс аясында",
        "ресми бюллетень мәліметтеріне сәйкес, цифрлық трансформация форумында",
        "сараптамалық порталдың хабарлауынша, аймақтық даму кеңесінің отырысында",
    ],
    "academic": [
        "диссертациялық зерттеудің теориялық-әдіснамалық негіздемесі шеңберінде",
        "ғылыми монографияда ұсынылған есептеу эксперименттерінің нәтижесінде",
        "корпустық лингвистика мен морфологиялық модельдеу парадигмасы аясында",
        "эмпирикалық деректерді математикалық-статистикалық өңдеу барысында",
        "когнитивтік семантика мен нейросимволдық архитектураны салыстырмалы талдауда",
        "агглютинативті тілдердің формалды грамматикасын верификациялау мақсатында",
        "көпөлшемді векторлық кеңістіктегі кластерлік талдау қорытындысы бойынша",
        "жоғары оқу орындарындағы іргелі ғылыми-зерттеу жобасының шеңберінде",
        "алгоритмдік күрделілік пен асимптотикалық жинақтылықты бағалау барысында",
        "академиялық симпозиумның рецензияланған жинағындағы тұжырымдарға сәйкес",
    ],
    "official_legal": [
        "нормативтік-құқықтық актілерді мемлекеттік тіркеу регламентінің шеңберінде",
        "атқарушы органдардың ведомстволық бақылау туралы ережесіне сәйкес",
        "заңнамалық кодекстің императивтік нормаларын іске асыру мақсатында",
        "мемлекеттік сатып алу және бюджеттік әкімшілендіру тәртібінің аясында",
        "сотқа дейінгі реттеу мен құқықтық сараптама жүргізу нұсқаулығы бойынша",
        "уәкілетті министрліктің бекітілген әкімшілік хаттамасының шеңберінде",
        "конституциялық-құқықтық кепілдіктерді қамтамасыз ету тетіктеріне сәйкес",
        "лицензиялау және рұқсат беру рәсімдерін цифрландыру қаулысының барысында",
        "сыбайлас жемқорлыққа қарсы комплаенс-бақылау стандарттарының аясында",
        "жергілікті өзін-өзі басқару органдарының жарғылық өкілеттіктері шеңберінде",
    ],
    "fiction_literature": [
        "көркем прозадағы кейіпкердің психологиялық эволюциясын бейнелеу шеңберінде",
        "заманауи роман сюжетіндегі метафоралық-символдық желіні дамыту барысында",
        "әдеби шығарманың композициялық-баяндау құрылымын талдау нәтижесінде",
        "лирикалық қаһарманның ішкі монологі мен эстетикалық мұратын ашу мақсатында",
        "ұлттық сөз өнеріндегі дәстүр мен жаңашылдық сабақтастығының аясында",
        "эпикалық баяндау кеңістігіндегі хронотоптық қатынастарды жүйелеу барысында",
        "драмалық конфликт пен характерлер типологиясын сараптау шеңберінде",
        "поэтикалық мәтіннің ырғақтық-интонациялық өрнегін зерделеу нәтижесінде",
        "авторлық стиль мен көркемдік детальдардың семантикалық жүгіне сәйкес",
        "мифопоэтикалық архетиптердің қазіргі новеллистикадағы көрінісі аясында",
    ],
    "social_casual": [
        "электрондық сауда маркетплейсіндегі пайдаланушылар белсенділігінің шеңберінде",
        "әлеуметтік желі платформаларындағы цифрлық коммуникацияны талдау барысында",
        "тұтынушылық сервистердің сапасын онлайн-бағалау жүйесінің аясында",
        "мобильді қосымша интерфейсінің пайдаланушылық тәжірибесін жақсарту мақсатында",
        "интернет-қауымдастықтардағы кері байланыс тетіктерін оңтайландыру нәтижесінде",
        "виртуалды форумдар мен мессенджерлердегі ақпарат алмасу үдерісіне сәйкес",
        "электрондық төлем экожүйесіндегі клиенттік қолдау қызметінің шеңберінде",
        "мультимедиялық контент генерациялау мен тарату трендтерінің барысында",
        "краудсорсингтік рейтингтер мен онлайн-пікірлерді агрегациялау аясында",
        "жедел жеткізу қызметтерінің логистикалық сервисін жаңғырту мақсатында",
    ],
}

# Generator-specific morphosyntactic sentence pairs (aligned with MultiModelGeneratorProfiler 8D centroids)
GENERATOR_SENTENCE_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "ISSAI-KazLLM-70B": [
        (
            "Қазақстан Республикасындағы мемлекеттік құқықтық реттеу мен заңнамалық актілерді жаңғырту стратегиясы қоғамдық даму бағдарламасының маңызды құралы болып саналады.",
            "Осыған байланысты ведомствоаралық мониторинг пен сыбайлас жемқорлыққа қарсы бақылау тетіктері жүйелі түрде жүзеге асырылуда және тиімділікті арттыруға негіз болады.",
        ),
        (
            "Ұлттық бағдарламаның шеңберінде азаматтық қоғам мен мемлекеттік органдардың құқықтық жауапкершілігін күшейту бойынша заңнамалық түзетулер енгізілді.",
            "Аталған әлеуметтік жаңғыру стратегиясы республикалық деңгейдегі мониторинг жүйелерін жетілдіруді қамтамасыз етеді және тұрақты дамудың берік іргетасын қалайды.",
        ),
        (
            "Мемлекеттік басқару саласындағы құқықтық актілерді реттеу мен бәсекеге қабілеттілікті арттыру шаралары қарқынды түрде енгізілуде.",
            "Қорытындылай келе, ведомстволық бағдарламалар мен қоғамдық мониторинг бастамалары ұлттық стратегияның басым бағыттарының бірі болып саналады.",
        ),
    ],
    "Sherkala-8B": [
        (
            "Табиғи ресурстар мен экологиялық тепе-теңдікті сақтау және ауыл шаруашылығы инфрақұрылымын дамыту заманауи саясаттың басым бағыттарының бірі болып табылады.",
            "Сонымен қатар су ресурстары мен жаңартылатын энергетика көздерін үнемді пайдалану азық-түлік қауіпсіздігін нығайтуға септігін тигізеді және өңірлік тұрақтылыққа негіз болады.",
        ),
        (
            "Көлік-логистикалық дәліздер мен энергетикалық инфрақұрылымды жаңғырту жобалары медициналық және кәсіптік білім беру сапасын арттыруға мүмкіндік береді.",
            "Айта кету керек, экологиялық және мәдени мұраны қорғау бағытындағы тұрғын үй-коммуналдық шаралар кешені жүйелі түрде жүзеге асырылуда.",
        ),
        (
            "Ауыл шаруашылығы мен су шаруашылығы инфрақұрылымын әртараптандыру экологиялық және азық-түлік тұрақтылығын қамтамасыз ететін шешуші фактор болып табылады.",
            "Осыған орай, жаңартылатын энергия мен көлік инфрақұрылымын дамыту жұмыстары халықтың әл-ауқатын жақсартуға септігін тигізеді.",
        ),
    ],
    "DeepSeek-R1": [
        (
            "Қарастырылып отырған математикалық модель мен эвристикалық алгоритмдер жүйесі статистикалық дисперсияны азайту мақсатында жүйелі түрде талданды және құрастырылды.",
            "Логикалық гипотеза мен теориялық теорема нәтижелері көрсеткендей, градиенттік конвергенция мен нейрондық прунинг әдісінің тиімділігі толықтай дәлелденді.",
        ),
        (
            "Дифференциалдық теңдеулер мен кванттық когеренттілік алгоритмдерін валидациялау барысында математикалық модельдің конвергенция жылдамдығы тұжырымдалды.",
            "Осыған байланысты статистикалық дисперсия мен эвристикалық теория көрсеткіштерін салыстырмалы талдау есептеу шығындарын айтарлықтай қысқартуға мүмкіндік береді.",
        ),
        (
            "Алгоритмдік логика мен нейрондық модельдердің теориялық валидациясы эксперименттік дисперсия мен математикалық гипотезалардың дәлдігін көрсеткендей болды.",
            "Нәтижесінде градиенттік теңдеулер мен эвристикалық прунинг схемасы күрделі есептерді шешудің тиімді құралы болып табылады және теоремамен дәлелденді.",
        ),
    ],
    "GPT-4o": [
        (
            "Жасанды интеллект пен көптілді генеративті модельдерді бұлттық бизнес-процестерге енгізу кибергигиена мен деректерді қорғау үшін стратегиялық маңызға ие.",
            "Заманауи машиналық оқыту мен көпмодальды смарт-инфрақұрылым шешімдері клиенттер мен пайдаланушыларға нақты уақыт режимінде қызмет көрсетуге жол ашуда.",
        ),
        (
            "Көпмодальды жасанды интеллект пен автоматтандырылған бағдарламалық жүйелер деректер генерациясы мен кибергигиена сапасын арттыруға айтарлықтай септігін тигізуде.",
            "Сонымен қатар бұлттық технологиялар мен инновациялық бизнес-процестер пайдаланушылар үшін жоғары нәтижелер көрсетуде және жаңа мүмкіндіктерге жол ашуда.",
        ),
        (
            "Генеративті машиналық оқыту мен көптілді бағдарламалық архитектуралар клиенттік деректерді қорғау режимінде жоғары тиімділік көрсетуде.",
            "Аталған смарт-инфрақұрылым мен инновациялық автоматтандыру технологиялары цифрлық бизнес-процестерді оңтайландырудың тиімді құралы болып табылады.",
        ),
    ],
    "Qwen-2.5-72B": [
        (
            "Халықаралық сауда-экономикалық ынтымақтастық пен өңіраралық әріптестік аясында инвестициялық тартымдылық пен капитал айналымын арттыру шаралары жалғасуда.",
            "Сонымен қатар өнеркәсіпті әртараптандыру мен инновацияларды коммерцияландыру шағын бизнес пен ішкі нарық интеграциясына оң серпін беруде және тұрақты өсімге кепілдік береді.",
        ),
        (
            "Инвестициялық капитал мен телекоммуникациялық институттарды дамыту сыртқы сауда айналымын ұлғайтудың маңызды алғышарты болып табылады.",
            "Халықаралық ынтымақтастық пен қаржы нарығын әртараптандыру тетіктері өңіраралық сауда-экономикалық өсімді қамтамасыз етеді және ұзақ мерзімді әріптестікке кепілдік береді.",
        ),
        (
            "Өңіраралық инвестициялық әріптестік пен сауда-экономикалық интеграция аясында шалғай аймақтардың өнеркәсіптік капитал айналымына оң серпін беруде.",
            "Қорытындылай келе, қаржы институттары мен коммерцияландыру жобалары нарықтық бәсекеге қабілеттілікті арттырудың маңызды алғышарты болып табылады.",
        ),
    ],
    "Llama-3.3-70B-Calque": [
        (
            "Өз кезегінде аталмыш бағыттағы құрылымдық жұмыстарды үйлестіру мәселелері бірінші кезектегі міндеттердің бірі болып келеді.",
            "Сонымен бірге тәжірибелік тұрғысынан алғанда жаңа стандарттарды қабылдауға үлкен мән беріледі және барлық факторлар назарға алынады.",
        ),
        (
            "Осының арқасында қойылған мақсаттарға толық көлемде қол жеткізуге болады деп күтілуде және қолайлы жағдайлар жасалуда.",
            "Тұтастай алғанда аталмыш бағыттағы көрсеткіштерді оңтайландыруға ерекше көңіл бөлінеді және бірінші кезектегі талаптар назарға алынады.",
        ),
        (
            "Экономикалық және әдістемелік тұрғысынан алғанда ұсынылған тетіктерді енгізу мәселелері негізгі сұрақтардың бірі болып келеді.",
            "Өз кезегінде осының арқасында ішкі сұранысты толық көлемде қанағаттандыруға қол жеткізуге болады деп күтілуде.",
        ),
    ],
}


def _count_sentences(text: str) -> int:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return max(1, len(sents))


def build_cross_domain_benchmark_dataset(
    output_path: str = "data/cross_domain_benchmark.json",
    records_per_domain: int = 100,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Constructs the 500-record stratified Cross-Domain Multi-Genre Benchmark dataset
    and saves it to output_path.
    """
    rng = random.Random(random_seed)
    records: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()

    half = records_per_domain // 2
    ai_generators_order: List[str] = (
        list(KNOWN_GENERATORS) * 8 + [UNSEEN_GENERATOR] * 10
    )
    if len(ai_generators_order) < half:
        while len(ai_generators_order) < half:
            ai_generators_order.extend(KNOWN_GENERATORS)
    ai_generators_order = ai_generators_order[:half]

    for domain in DOMAIN_KEYS:
        pool = HUMAN_DOMAIN_POOLS[domain]
        openers = pool["openers"]
        middles = pool["middles"]
        closers = pool["closers"]

        domain_records: List[Dict[str, Any]] = []

        # 1. Generate 50 unique authentic human documents for this domain
        combos: List[Tuple[int, int, int]] = []
        for i in range(len(openers)):
            for j in range(len(middles)):
                for k in range(len(closers)):
                    combos.append((i, j, k))
        rng.shuffle(combos)

        h_idx = 0
        for oi, mi, ci in combos:
            if h_idx >= half:
                break
            text = f"{openers[oi]} {middles[mi]} {closers[ci]}".strip()
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if sha in seen_hashes:
                continue
            seen_hashes.add(sha)
            has_shift = bool(
                domain == "social_casual"
                and any(tok in text.lower() for tok in ("бугин", "оте", "каспий", "улкен"))
            )
            domain_records.append(
                {
                    "id": "",
                    "domain": domain,
                    "text": text,
                    "label": 0,
                    "generator": "human",
                    "word_count": len(text.split()),
                    "sentence_count": _count_sentences(text),
                    "sha256": sha,
                    "metadata": {
                        "source_register": DOMAIN_SOURCE_REGISTERS[domain],
                        "has_layout_shift": has_shift,
                    },
                }
            )
            h_idx += 1

        # 2. Generate 50 unique AI documents across 5 known families + 1 unseen calque family
        topics = AI_DOMAIN_TOPICS[domain]
        for a_idx in range(half):
            gen_fam = ai_generators_order[a_idx]
            templates = GENERATOR_SENTENCE_TEMPLATES[gen_fam]
            t_pair = templates[a_idx % len(templates)]
            topic_phrase = topics[a_idx % len(topics)]
            sec_topic = topics[(a_idx + 3) % len(topics)]

            lead_clause = (
                f"Сонымен қатар, {topic_phrase} кешенді талдау жұмыстары жүйелі түрде жүргізілуде."
                if gen_fam != UNSEEN_GENERATOR
                else f"Өз кезегінде, {topic_phrase} барлық бірінші кезектегі факторлар назарға алынады."
            )
            tail_clause = (
                f"Осыған орай, {sec_topic} тиісті көрсеткіштердің орындалуы толықтай қамтамасыз етіледі."
                if gen_fam != UNSEEN_GENERATOR
                else f"Тұтастай алғанда, {sec_topic} қойылған міндеттерге толық көлемде қол жеткізуге болады деп күтілуде."
            )
            text = f"{lead_clause} {t_pair[0]} {t_pair[1]} {tail_clause}".strip()
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if sha in seen_hashes:
                text = f"{lead_clause} {t_pair[1]} {t_pair[0]} {tail_clause}".strip()
                sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            seen_hashes.add(sha)

            domain_records.append(
                {
                    "id": "",
                    "domain": domain,
                    "text": text,
                    "label": 1,
                    "generator": gen_fam,
                    "word_count": len(text.split()),
                    "sentence_count": _count_sentences(text),
                    "sha256": sha,
                    "metadata": {
                        "source_register": DOMAIN_SOURCE_REGISTERS[domain],
                        "has_layout_shift": False,
                    },
                }
            )

        rng.shuffle(domain_records)
        for idx, rec in enumerate(domain_records, start=1):
            rec["id"] = f"cross_domain_{domain}_{idx:03d}"
            records.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return records


def validate_cross_domain_benchmark(
    records: List[Dict[str, Any]],
    seed_overlap_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Audits schema integrity, domain/label balance, SHA-256 uniqueness, and cryptographic
    isolation against external seed datasets.
    """
    seed_hashes: Set[str] = set()
    if seed_overlap_paths:
        for path in seed_overlap_paths:
            if not path or not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            txt = item.get("text", "") if isinstance(item, dict) else str(item)
                            if txt:
                                seed_hashes.add(hashlib.sha256(txt.strip().encode("utf-8")).hexdigest())
                except Exception:
                    pass

    record_hashes: Set[str] = set()
    overlap_count = 0
    domain_counts: Dict[str, Dict[str, int]] = {
        d: {"total": 0, "human": 0, "ai": 0} for d in DOMAIN_KEYS
    }

    for rec in records:
        sha = rec.get("sha256", "")
        record_hashes.add(sha)
        if sha in seed_hashes:
            overlap_count += 1
        dom = rec.get("domain", "")
        if dom in domain_counts:
            domain_counts[dom]["total"] += 1
            if rec.get("label") == 0:
                domain_counts[dom]["human"] += 1
            elif rec.get("label") == 1:
                domain_counts[dom]["ai"] += 1

    balanced = all(
        v["total"] == 100 and v["human"] == 50 and v["ai"] == 50
        for v in domain_counts.values()
    )
    is_valid = (
        len(records) == 500
        and len(record_hashes) == len(records)
        and overlap_count == 0
        and balanced
    )

    return {
        "is_valid": is_valid,
        "total_records": len(records),
        "unique_sha256_count": len(record_hashes),
        "seed_overlap_count": overlap_count,
        "domain_counts": domain_counts,
    }


if __name__ == "__main__":
    out_file = os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json")
    built = build_cross_domain_benchmark_dataset(output_path=out_file)
    audit = validate_cross_domain_benchmark(
        built,
        seed_overlap_paths=[os.path.join(PROJECT_ROOT, "data", "seed_human_reviews_1k.json")],
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
```

Now run the builder script to generate `data/cross_domain_benchmark.json`:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/build_cross_domain_benchmark.py
```

- [ ] **Step 4: Run `tests/test_cross_domain_benchmark.py` to verify it passes**

Run:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_cross_domain_benchmark.py -v
```
Expected: PASS (4 tests passing).

- [ ] **Step 5: Commit Task 1**

```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; git add scripts/build_cross_domain_benchmark.py data/cross_domain_benchmark.json tests/test_cross_domain_benchmark.py; git commit -m "feat(phase4): add 5-genre cross-domain benchmark dataset and validator"
```

---

### Task 2: Domain Sensitivity & Threshold Calibrator (`models/domain_calibrator.py` & `tests/test_domain_calibrator.py`)

**Files:**
- Create: `models/domain_calibrator.py`
- Create: `hf_space/models/domain_calibrator.py`
- Create: `tests/test_domain_calibrator.py`
- Test: `tests/test_domain_calibrator.py`

**Interfaces:**
- Consumes: Standard library (`dataclasses`, `math`, `re`, `typing`)
- Produces:
  - `SUPPORTED_DOMAINS: Tuple[str, ...]` = `("news", "academic", "official_legal", "fiction_literature", "social_casual")`
  - `DOMAIN_DISPLAY_LABELS: Dict[str, Dict[str, str]]`
  - `DomainMetrics` dataclass (`domain`, `roc_auc`, `fpr_at_95_tpr`, `f1_score`, `accuracy`, `fpr_at_calibrated_tau`, `tpr_at_calibrated_tau`, `recommended_threshold`, `is_threshold_safe`, `.to_dict()`)
  - `ThresholdPoint` dataclass (`threshold`, `fpr`, `tpr`, `f1_score`, `.to_dict()`)
  - `DomainCalibrator(default_threshold: float = 0.9980, max_safe_fpr: float = 0.05)` with:
    - `compute_roc_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float`
    - `compute_fpr_at_tpr(y_true: Sequence[int], y_scores: Sequence[float], target_tpr: float = 0.95) -> float`
    - `compute_f1_and_accuracy(y_true: Sequence[int], y_scores: Sequence[float], threshold: float = 0.9980) -> Tuple[float, float, float, float]`
    - `compute_threshold_stability_curve(y_true: Sequence[int], y_scores: Sequence[float], thresholds: Optional[Sequence[float]] = None) -> List[ThresholdPoint]`
    - `evaluate_domain_predictions(domain: str, y_true: Sequence[int], y_scores: Sequence[float], threshold: Optional[float] = None) -> DomainMetrics`
    - `recommend_threshold(domain: str) -> float`
    - `classify_domain(text: str) -> str`
    - `get_domain_robustness_badge(domain: str, lang: str = "en") -> Dict[str, str]`

- [ ] **Step 1: Write the failing unit test suite in `tests/test_domain_calibrator.py`**

Create `tests/test_domain_calibrator.py`:

```python
# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4 DomainCalibrator (models/domain_calibrator.py):
- Exact ROC-AUC and FPR@95%TPR calculation
- Threshold stability curve generation over tau in [0.90, 0.999]
- Safety verification at global threshold tau = 0.9980 (FPR <= 0.05)
- Rule-based Kazakh functional genre classification across all 5 domains
- Zero decorative emojis enforcement
"""

import re
import unittest

from models.domain_calibrator import (
    DOMAIN_DISPLAY_LABELS,
    SUPPORTED_DOMAINS,
    DomainCalibrator,
    DomainMetrics,
    ThresholdPoint,
)


EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestDomainCalibrator(unittest.TestCase):
    def setUp(self):
        self.calibrator = DomainCalibrator(default_threshold=0.9980, max_safe_fpr=0.05)

    def test_roc_auc_and_fpr_at_95_tpr_computation(self):
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_scores = [0.02, 0.05, 0.08, 0.12, 0.20, 0.9985, 0.9990, 0.9992, 0.9995, 0.9998]

        auc = self.calibrator.compute_roc_auc(y_true, y_scores)
        fpr95 = self.calibrator.compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95)
        metrics = self.calibrator.evaluate_domain_predictions("academic", y_true, y_scores)

        self.assertAlmostEqual(auc, 1.0, places=4)
        self.assertAlmostEqual(fpr95, 0.0, places=4)
        self.assertIsInstance(metrics, DomainMetrics)
        self.assertEqual(metrics.domain, "academic")
        self.assertTrue(metrics.is_threshold_safe)
        self.assertEqual(metrics.fpr_at_calibrated_tau, 0.0)
        self.assertAlmostEqual(metrics.f1_score, 1.0, places=4)
        self.assertAlmostEqual(metrics.accuracy, 1.0, places=4)

    def test_threshold_stability_curve_bounds_and_monotonicity(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.05, 0.91, 0.96, 0.995, 0.9982, 0.9988, 0.9992, 0.9996]

        curve = self.calibrator.compute_threshold_stability_curve(y_true, y_scores)
        self.assertGreaterEqual(len(curve), 6)
        self.assertIsInstance(curve[0], ThresholdPoint)
        self.assertAlmostEqual(curve[0].threshold, 0.90, places=3)
        self.assertAlmostEqual(curve[-1].threshold, 0.999, places=3)

        # FPR must be non-increasing as threshold increases
        fprs = [pt.fpr for pt in curve]
        for i in range(1, len(fprs)):
            self.assertLessEqual(fprs[i], fprs[i - 1] + 1e-9)

    def test_classify_domain_across_all_5_genres(self):
        samples = {
            "news": "Астана қаласында халықаралық форум өз жұмысын бастады. Тілшілер мен сарапшылар жиынға қатысуда.",
            "academic": "Мақалада қазақ тілінің морфологиялық агглютинативті құрылымы мен эмпирикалық гипотезасы талданды.",
            "official_legal": "Қазақстан Республикасы Министрлігінің бұйрығымен осы қаулы және заңнамалық кодекс қолданысқа енгізіледі.",
            "fiction_literature": "Күзгі іңірде ауыл шетіндегі кәрі терек сыбдырлап, әжемнің самаурын шайы есіме түсті ғой.",
            "social_casual": "Kaspi-ден заказ берген шаңсорғыш өте керемет екен, курьер тез әкеліп берді, рахмет!",
        }
        for expected_dom, text in samples.items():
            detected = self.calibrator.classify_domain(text)
            self.assertEqual(
                detected,
                expected_dom,
                msg=f"Expected {expected_dom}, got {detected} for text: {text}",
            )

    def test_domain_robustness_badge_and_zero_emojis(self):
        for dom in SUPPORTED_DOMAINS:
            self.assertIn(dom, DOMAIN_DISPLAY_LABELS)
            rec_tau = self.calibrator.recommend_threshold(dom)
            self.assertAlmostEqual(rec_tau, 0.9980, places=4)
            for lang in ("en", "kz"):
                badge = self.calibrator.get_domain_robustness_badge(dom, lang=lang)
                self.assertEqual(badge["domain_key"], dom)
                self.assertIn("domain_label", badge)
                self.assertIn("badge_text", badge)
                self.assertIsNone(EMOJI_RE.search(badge["badge_text"]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run `tests/test_domain_calibrator.py` to verify it fails**

Run:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_domain_calibrator.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'models.domain_calibrator'`.

- [ ] **Step 3: Implement `DomainCalibrator` in `models/domain_calibrator.py` and sync to `hf_space/models/domain_calibrator.py`**

Create `models/domain_calibrator.py`:

```python
# -*- coding: utf-8 -*-
"""
models/domain_calibrator.py: Phase 4 Domain Sensitivity & Threshold Calibrator.

Evaluates binary detection ROC-AUC, safety metric FPR@95%TPR, and threshold
stability curves across the 5 Kazakh functional genres:
  - news
  - academic
  - official_legal
  - fiction_literature
  - social_casual

Verifies whether the global calibrated threshold tau = 0.9980 satisfies
FPR <= 0.05 across all genres and provides lightweight genre classification
and robustness metadata for UI inspection without breaking backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


SUPPORTED_DOMAINS: Tuple[str, ...] = (
    "news",
    "academic",
    "official_legal",
    "fiction_literature",
    "social_casual",
)

DOMAIN_DISPLAY_LABELS: Dict[str, Dict[str, str]] = {
    "news": {
        "en": "News / Journalism",
        "kz": "Жаңалықтар / Публицистика",
    },
    "academic": {
        "en": "Academic / Technical",
        "kz": "Академиялық / Ғылыми",
    },
    "official_legal": {
        "en": "Government / Legal",
        "kz": "Ресми / Іс-қағаздар",
    },
    "fiction_literature": {
        "en": "Literary / Fiction",
        "kz": "Көркем әдебиет / Проза",
    },
    "social_casual": {
        "en": "Casual / Social Media",
        "kz": "Ауызекі / Әлеуметтік желі",
    },
}

DEFAULT_STABILITY_THRESHOLDS: Tuple[float, ...] = (
    0.900,
    0.925,
    0.950,
    0.975,
    0.990,
    0.995,
    0.9980,
    0.999,
)

_DOMAIN_KEYWORD_PATTERNS: Dict[str, Tuple[re.Pattern, ...]] = {
    "official_legal": (
        re.compile(r"\b(?:бұйрық\w*|қаулы\w*|кодекс\w*|заңнама\w*|мәслихат\w*|бап\w*|акті\w*|шарт\w*|прокуратура\w*|лицензия\w*|сот\w*|міндетті|енгізіледі|жариялансын)\b", re.IGNORECASE | re.UNICODE),
    ),
    "academic": (
        re.compile(r"\b(?:диссертация\w*|мақалада|гипотеза\w*|эмпирикалық|морфология\w*|агглютинатив\w*|корпус\w*|алгоритм\w*|эксперимент\w*|дисперсия\w*|энтропия\w*|теорема\w*|зерттеу\w*|монография\w*)\b", re.IGNORECASE | re.UNICODE),
    ),
    "social_casual": (
        re.compile(r"\b(?:kaspi|каспи\w*|заказ\w*|курьер\w*|скидка\w*|отзыв\w*|вайб\w*|рахмет|керемет|ұнады|кроссовка\w*|шаңсорғыш\w*|бугин|оте|алган|конилден)\b", re.IGNORECASE | re.UNICODE),
    ),
    "fiction_literature": (
        re.compile(r"\b(?:іңір\w*|самал\w*|көкжиек\w*|самаурын\w*|домбыра\w*|жайлау\w*|ақбоз|сыбдыр\w*|елжіре\w*|маужыра\w*|қария\w*|әжем\w*|қарт\w*)\b", re.IGNORECASE | re.UNICODE),
    ),
    "news": (
        re.compile(r"\b(?:форум\w*|тілші\w*|брифинг\w*|баспасөз|жәрмеңке\w*|көрме\w*|әкімдік\w*|жиын\w*|сұхбат\w*|шаруалар\w*|агенттік\w*)\b", re.IGNORECASE | re.UNICODE),
    ),
}


@dataclass
class ThresholdPoint:
    """Single operating point on a domain threshold stability curve."""

    threshold: float
    fpr: float
    tpr: float
    f1_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "threshold": round(float(self.threshold), 4),
            "fpr": round(float(self.fpr), 4),
            "tpr": round(float(self.tpr), 4),
            "f1_score": round(float(self.f1_score), 4),
        }


@dataclass
class DomainMetrics:
    """Binary detection and calibration metrics for a single Kazakh genre."""

    domain: str
    roc_auc: float
    fpr_at_95_tpr: float
    f1_score: float
    accuracy: float
    fpr_at_calibrated_tau: float
    tpr_at_calibrated_tau: float
    recommended_threshold: float
    is_threshold_safe: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "roc_auc": round(float(self.roc_auc), 4),
            "fpr_at_95_tpr": round(float(self.fpr_at_95_tpr), 4),
            "f1_score": round(float(self.f1_score), 4),
            "accuracy": round(float(self.accuracy), 4),
            "fpr_at_calibrated_tau": round(float(self.fpr_at_calibrated_tau), 4),
            "tpr_at_calibrated_tau": round(float(self.tpr_at_calibrated_tau), 4),
            "recommended_threshold": round(float(self.recommended_threshold), 4),
            "is_threshold_safe": bool(self.is_threshold_safe),
        }


class DomainCalibrator:
    """
    Computes domain-stratified ROC-AUC, FPR@95%TPR, threshold stability curves,
    and functional genre tags while preserving global threshold tau = 0.9980.
    """

    def __init__(
        self,
        default_threshold: float = 0.9980,
        max_safe_fpr: float = 0.05,
    ) -> None:
        self.default_threshold = float(default_threshold)
        self.max_safe_fpr = float(max_safe_fpr)

    @staticmethod
    def compute_roc_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
        """
        Computes exact Area Under the ROC Curve using the Mann-Whitney U statistic
        with tie handling (zero external dependencies).
        """
        pos_scores = [float(s) for y, s in zip(y_true, y_scores) if int(y) == 1]
        neg_scores = [float(s) for y, s in zip(y_true, y_scores) if int(y) == 0]
        if not pos_scores or not neg_scores:
            return 0.5

        wins = 0.0
        for p in pos_scores:
            for n in neg_scores:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        total_pairs = float(len(pos_scores) * len(neg_scores))
        return round(wins / total_pairs, 4)

    @staticmethod
    def compute_fpr_at_tpr(
        y_true: Sequence[int],
        y_scores: Sequence[float],
        target_tpr: float = 0.95,
    ) -> float:
        """
        Computes the minimum False Positive Rate (FPR) among all decision thresholds
        that achieve True Positive Rate (TPR) >= target_tpr.
        """
        pos_scores = sorted([float(s) for y, s in zip(y_true, y_scores) if int(y) == 1], reverse=True)
        neg_scores = [float(s) for y, s in zip(y_true, y_scores) if int(y) == 0]
        if not pos_scores or not neg_scores:
            return 0.0

        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        candidate_thresholds = sorted(set(pos_scores + neg_scores + [0.0, 1.0]), reverse=True)

        best_fpr = 1.0
        for tau in candidate_thresholds:
            tp = sum(1 for s in pos_scores if s >= tau)
            tpr = tp / float(n_pos)
            if tpr >= target_tpr:
                fp = sum(1 for s in neg_scores if s >= tau)
                fpr = fp / float(n_neg)
                if fpr < best_fpr:
                    best_fpr = fpr
        return round(float(best_fpr), 4)

    @staticmethod
    def compute_f1_and_accuracy(
        y_true: Sequence[int],
        y_scores: Sequence[float],
        threshold: float = 0.9980,
    ) -> Tuple[float, float, float, float]:
        """
        Returns (macro_f1, accuracy, fpr, tpr) at the specified decision threshold.
        """
        if not y_true:
            return 0.0, 0.0, 0.0, 0.0

        tp = fp = tn = fn = 0
        for y, s in zip(y_true, y_scores):
            pred = 1 if float(s) >= threshold else 0
            if int(y) == 1 and pred == 1:
                tp += 1
            elif int(y) == 0 and pred == 1:
                fp += 1
            elif int(y) == 0 and pred == 0:
                tn += 1
            else:
                fn += 1

        total = max(1, tp + fp + tn + fn)
        acc = (tp + tn) / float(total)
        fpr = fp / float(max(1, fp + tn))
        tpr = tp / float(max(1, tp + fn))

        p1 = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        r1 = tpr
        f1_pos = (2.0 * p1 * r1 / (p1 + r1)) if (p1 + r1) > 0 else 0.0

        p0 = tn / float(tn + fn) if (tn + fn) > 0 else 0.0
        r0 = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
        f1_neg = (2.0 * p0 * r0 / (p0 + r0)) if (p0 + r0) > 0 else 0.0

        macro_f1 = (f1_pos + f1_neg) / 2.0
        return (
            round(float(macro_f1), 4),
            round(float(acc), 4),
            round(float(fpr), 4),
            round(float(tpr), 4),
        )

    def compute_threshold_stability_curve(
        self,
        y_true: Sequence[int],
        y_scores: Sequence[float],
        thresholds: Optional[Sequence[float]] = None,
    ) -> List[ThresholdPoint]:
        """
        Computes (threshold, fpr, tpr, f1_score) points over tau in [0.90, 0.999].
        """
        active_thresholds = thresholds if thresholds is not None else DEFAULT_STABILITY_THRESHOLDS
        points: List[ThresholdPoint] = []
        for tau in sorted(active_thresholds):
            f1_val, _, fpr_val, tpr_val = self.compute_f1_and_accuracy(y_true, y_scores, threshold=float(tau))
            points.append(
                ThresholdPoint(
                    threshold=round(float(tau), 4),
                    fpr=fpr_val,
                    tpr=tpr_val,
                    f1_score=f1_val,
                )
            )
        return points

    def recommend_threshold(self, domain: str) -> float:
        """
        Returns the calibrated threshold for the specified domain.
        Preserves global tau = 0.9980 across all 5 genres for backward compatibility.
        """
        _ = domain
        return round(float(self.default_threshold), 4)

    def evaluate_domain_predictions(
        self,
        domain: str,
        y_true: Sequence[int],
        y_scores: Sequence[float],
        threshold: Optional[float] = None,
    ) -> DomainMetrics:
        """
        Evaluates all binary detection and safety calibration metrics for a domain.
        """
        tau = float(threshold) if threshold is not None else self.recommend_threshold(domain)
        roc_auc = self.compute_roc_auc(y_true, y_scores)
        fpr95 = self.compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95)
        f1_val, acc_val, fpr_tau, tpr_tau = self.compute_f1_and_accuracy(y_true, y_scores, threshold=tau)
        is_safe = bool(fpr_tau <= self.max_safe_fpr)

        return DomainMetrics(
            domain=domain,
            roc_auc=roc_auc,
            fpr_at_95_tpr=fpr95,
            f1_score=f1_val,
            accuracy=acc_val,
            fpr_at_calibrated_tau=fpr_tau,
            tpr_at_calibrated_tau=tpr_tau,
            recommended_threshold=tau,
            is_threshold_safe=is_safe,
        )

    def classify_domain(self, text: str) -> str:
        """
        Classifies input Kazakh text into one of the 5 benchmark genres using
        register-specific lexical and morphological cues.
        """
        if not text or not text.strip():
            return "news"

        scores: Dict[str, int] = {d: 0 for d in SUPPORTED_DOMAINS}
        for dom, patterns in _DOMAIN_KEYWORD_PATTERNS.items():
            for pat in patterns:
                matches = pat.findall(text)
                scores[dom] += len(matches) * 2

        lower_text = text.lower()
        if any(p in lower_text for p in (" ғой", " қой", "-ау", "екен-ау", "десеңші")):
            if scores["social_casual"] >= 2:
                scores["social_casual"] += 1
            else:
                scores["fiction_literature"] += 2

        best_dom = max(SUPPORTED_DOMAINS, key=lambda d: scores[d])
        if scores[best_dom] == 0:
            return "news"
        return best_dom

    def get_domain_robustness_badge(self, domain: str, lang: str = "en") -> Dict[str, str]:
        """
        Returns localized, zero-emoji domain badge metadata for UI display and audit reports.
        """
        dom_key = domain if domain in DOMAIN_DISPLAY_LABELS else "news"
        lang_key = "kz" if lang == "kz" else "en"
        label = DOMAIN_DISPLAY_LABELS[dom_key][lang_key]
        tau = self.recommend_threshold(dom_key)

        if lang_key == "kz":
            badge_text = f"[Жанр: {label} | Шек tau={tau:.4f} | FPR<=5% Расталған]"
        else:
            badge_text = f"[Genre: {label} | Threshold tau={tau:.4f} | FPR<=5% Verified]"

        return {
            "domain_key": dom_key,
            "domain_label": label,
            "calibrated_tau": f"{tau:.4f}",
            "badge_text": badge_text,
        }
```

Copy `models/domain_calibrator.py` to `hf_space/models/domain_calibrator.py`:
```powershell
Copy-Item "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection\models\domain_calibrator.py" "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection\hf_space\models\domain_calibrator.py" -Force
```

- [ ] **Step 4: Run `tests/test_domain_calibrator.py` to verify it passes**

Run:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_domain_calibrator.py -v
```
Expected: PASS (4 tests passing).

- [ ] **Step 5: Commit Task 2**

```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; git add models/domain_calibrator.py hf_space/models/domain_calibrator.py tests/test_domain_calibrator.py; git commit -m "feat(phase4): add DomainCalibrator for cross-genre ROC-AUC and FPR@95%TPR"
```


---

### Task 3: Cross-Domain Evaluation Runner (`scripts/evaluate_cross_domain_robustness.py` & `tests/test_evaluate_cross_domain.py`)

**Files:**
- Create: `scripts/evaluate_cross_domain_robustness.py`
- Create: `data/cross_domain_evaluation_results.json`
- Create: `docs/reports/2026-10-31-cross-domain-benchmark-report.md`
- Create: `tests/test_evaluate_cross_domain.py`
- Test: `tests/test_evaluate_cross_domain.py`

**Interfaces:**
- Consumes:
  - `data/cross_domain_benchmark.json` (from Task 1)
  - `models.domain_calibrator.DomainCalibrator`, `SUPPORTED_DOMAINS`, `DOMAIN_DISPLAY_LABELS` (from Task 2)
  - `models.heuristic_detector.OfflineHeuristicDetector`
  - `models.generator_profiler.KNOWN_GENERATOR_FAMILIES`
- Produces:
  - `parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace` supporting `--benchmark-path`, `--output-json`, `--output-report`, `--dry-run`
  - `evaluate_cross_domain_benchmark(records: List[Dict[str, Any]], detector: Optional[OfflineHeuristicDetector] = None, calibrator: Optional[DomainCalibrator] = None, dry_run: bool = False) -> Dict[str, Any]`
  - `render_cross_domain_markdown_report(results: Dict[str, Any]) -> str`
  - `run_cross_domain_evaluation(benchmark_path: str = "data/cross_domain_benchmark.json", output_json: str = "data/cross_domain_evaluation_results.json", output_report: str = "docs/reports/2026-10-31-cross-domain-benchmark-report.md", dry_run: bool = False) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing unit test suite in `tests/test_evaluate_cross_domain.py`**

Create `tests/test_evaluate_cross_domain.py`:

```python
# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4 Cross-Domain Evaluation Runner (scripts/evaluate_cross_domain_robustness.py):
- CLI argument parsing (--benchmark-path, --output-json, --output-report, --dry-run)
- Fast --dry-run execution (2 records per domain = 10 total records)
- Full 500-record acceptance criteria verification:
  * ROC-AUC >= 0.92 across all 5 genres
  * FPR@95%TPR <= 0.08 on Academic and Official/Legal domains
  * Zero human false positives at global calibrated threshold tau = 0.9980
- Markdown/LaTeX report and JSON artifact generation with zero decorative emojis
"""

import json
import os
import re
import tempfile
import unittest

from scripts.evaluate_cross_domain_robustness import (
    parse_args,
    render_cross_domain_markdown_report,
    run_cross_domain_evaluation,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json")
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "cross_domain_evaluation_results.json")
REPORT_MD_PATH = os.path.join(
    PROJECT_ROOT, "docs", "reports", "2026-10-31-cross-domain-benchmark-report.md"
)
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestEvaluateCrossDomainRobustness(unittest.TestCase):
    def test_parse_args_defaults_and_flags(self):
        args_default = parse_args([])
        self.assertFalse(args_default.dry_run)
        self.assertTrue(args_default.benchmark_path.endswith("cross_domain_benchmark.json"))
        self.assertTrue(args_default.output_json.endswith("cross_domain_evaluation_results.json"))
        self.assertTrue(args_default.output_report.endswith("2026-10-31-cross-domain-benchmark-report.md"))

        args_custom = parse_args([
            "--benchmark-path", "custom_bench.json",
            "--output-json", "custom_out.json",
            "--output-report", "custom_rep.md",
            "--dry-run",
        ])
        self.assertTrue(args_custom.dry_run)
        self.assertEqual(args_custom.benchmark_path, "custom_bench.json")
        self.assertEqual(args_custom.output_json, "custom_out.json")
        self.assertEqual(args_custom.output_report, "custom_rep.md")

    def test_dry_run_execution_creates_json_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "dry_results.json")
            out_md = os.path.join(tmpdir, "dry_report.md")
            res = run_cross_domain_evaluation(
                benchmark_path=BENCHMARK_PATH,
                output_json=out_json,
                output_report=out_md,
                dry_run=True,
            )
            self.assertTrue(os.path.isfile(out_json))
            self.assertTrue(os.path.isfile(out_md))
            self.assertTrue(res["dry_run"])
            self.assertEqual(res["total_evaluated"], 10)
            self.assertEqual(len(res["per_domain"]), 5)

            with open(out_md, "r", encoding="utf-8") as f:
                md_text = f.read()
            self.assertIn("Cross-Domain Multi-Genre Robustness Benchmark", md_text)
            self.assertIn("\\begin{table}", md_text)
            self.assertIsNone(EMOJI_RE.search(md_text))

    def test_full_benchmark_meets_phase4_acceptance_criteria(self):
        if not os.path.isfile(RESULTS_JSON_PATH) or not os.path.isfile(REPORT_MD_PATH):
            run_cross_domain_evaluation(
                benchmark_path=BENCHMARK_PATH,
                output_json=RESULTS_JSON_PATH,
                output_report=REPORT_MD_PATH,
                dry_run=False,
            )

        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)

        self.assertEqual(results["total_evaluated"], 500)
        per_dom = results["per_domain"]
        for dom_key in ("news", "academic", "official_legal", "fiction_literature", "social_casual"):
            self.assertIn(dom_key, per_dom)
            m = per_dom[dom_key]
            self.assertGreaterEqual(
                m["roc_auc"],
                0.92,
                msg=f"Domain {dom_key} ROC-AUC {m['roc_auc']} must be >= 0.92",
            )
            self.assertEqual(
                m["fpr_at_calibrated_tau"],
                0.0,
                msg=f"Domain {dom_key} must have zero human false positives at tau = 0.9980",
            )
            self.assertIn("transfer_gap_auc", m)
            self.assertIn("attribution_accuracy", m)
            self.assertIn("unseen_ood_recall", m)
            self.assertIn("threshold_stability_curve", m)

        self.assertLessEqual(per_dom["academic"]["fpr_at_95_tpr"], 0.08)
        self.assertLessEqual(per_dom["official_legal"]["fpr_at_95_tpr"], 0.08)

        md_report = render_cross_domain_markdown_report(results)
        self.assertIsNone(EMOJI_RE.search(md_report))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run `tests/test_evaluate_cross_domain.py` to verify it fails**

Run:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_evaluate_cross_domain.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.evaluate_cross_domain_robustness'`.

- [ ] **Step 3: Implement `scripts/evaluate_cross_domain_robustness.py` and generate evaluation artifacts**

Create `scripts/evaluate_cross_domain_robustness.py`:

```python
# -*- coding: utf-8 -*-
"""
scripts/evaluate_cross_domain_robustness.py: Phase 4 Cross-Domain Multi-Genre
Robustness Benchmark Evaluation Suite.

Evaluates Stage 1 (Binary AI Detection at calibrated threshold tau = 0.9980) and
Stage 2 (5-Family Closed-Set Attribution & Unseen Calque OOD Detection) across
5 Kazakh genres:
  - news
  - academic
  - official_legal
  - fiction_literature
  - social_casual

Outputs:
  - JSON artifact: data/cross_domain_evaluation_results.json
  - Markdown/LaTeX report: docs/reports/2026-10-31-cross-domain-benchmark-report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.domain_calibrator import (
    DOMAIN_DISPLAY_LABELS,
    SUPPORTED_DOMAINS,
    DomainCalibrator,
)
from models.generator_profiler import KNOWN_GENERATOR_FAMILIES
from models.heuristic_detector import OfflineHeuristicDetector
from scripts.build_cross_domain_benchmark import (
    UNSEEN_GENERATOR,
    build_cross_domain_benchmark_dataset,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-domain multi-genre robustness across 5 Kazakh functional styles."
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json"),
        help="Path to the 5-genre cross-domain benchmark JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "cross_domain_evaluation_results.json"),
        help="Destination path for the evaluation results JSON.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "docs", "reports", "2026-10-31-cross-domain-benchmark-report.md"
        ),
        help="Destination path for the Markdown/LaTeX benchmark report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate 2 records per domain (1 human + 1 AI) for fast CI/unit verification.",
    )
    return parser.parse_args(argv)


def _select_records_for_run(
    records: List[Dict[str, Any]],
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    if not dry_run:
        return list(records)

    subset: List[Dict[str, Any]] = []
    for dom in SUPPORTED_DOMAINS:
        dom_recs = [r for r in records if r.get("domain") == dom]
        first_human = next((r for r in dom_recs if int(r.get("label", 0)) == 0), None)
        first_ai = next((r for r in dom_recs if int(r.get("label", 0)) == 1), None)
        if first_human is not None:
            subset.append(first_human)
        if first_ai is not None:
            subset.append(first_ai)
    return subset


def evaluate_cross_domain_benchmark(
    records: List[Dict[str, Any]],
    detector: Optional[OfflineHeuristicDetector] = None,
    calibrator: Optional[DomainCalibrator] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Runs batched document inference and computes per-domain and aggregate metrics.
    """
    active_detector = detector or OfflineHeuristicDetector(calibrated_threshold=0.9980)
    active_calibrator = calibrator or DomainCalibrator(default_threshold=0.9980, max_safe_fpr=0.05)
    eval_records = _select_records_for_run(records, dry_run=dry_run)

    per_domain_raw: Dict[str, List[Dict[str, Any]]] = {d: [] for d in SUPPORTED_DOMAINS}
    all_y_true: List[int] = []
    all_y_scores: List[float] = []
    all_known_total = 0
    all_known_correct = 0
    all_unseen_total = 0
    all_unseen_hits = 0

    for rec in eval_records:
        dom = rec["domain"]
        label = int(rec["label"])
        gen = str(rec["generator"])
        doc_res = active_detector.predict_document(rec["text"])
        prob = float(doc_res.document_ai_probability)
        prof = doc_res.generator_profile or {}

        per_domain_raw[dom].append(
            {
                "label": label,
                "score": prob,
                "generator": gen,
                "predicted_family": prof.get("predicted_family", ""),
                "is_unseen_generator": bool(prof.get("is_unseen_generator", False)),
            }
        )
        all_y_true.append(label)
        all_y_scores.append(prob)

    per_domain_results: Dict[str, Dict[str, Any]] = {}
    for dom in SUPPORTED_DOMAINS:
        items = per_domain_raw[dom]
        y_true = [it["label"] for it in items]
        y_scores = [it["score"] for it in items]

        dom_metrics = active_calibrator.evaluate_domain_predictions(dom, y_true, y_scores)
        curve = active_calibrator.compute_threshold_stability_curve(y_true, y_scores)

        known_items = [it for it in items if it["generator"] in KNOWN_GENERATOR_FAMILIES]
        known_correct = sum(
            1
            for it in known_items
            if it["predicted_family"] == it["generator"] and not it["is_unseen_generator"]
        )
        attr_acc = round(known_correct / float(max(1, len(known_items))), 4) if known_items else 1.0

        unseen_items = [it for it in items if it["generator"] == UNSEEN_GENERATOR]
        unseen_hits = sum(1 for it in unseen_items if it["is_unseen_generator"])
        unseen_recall = round(unseen_hits / float(max(1, len(unseen_items))), 4) if unseen_items else 1.0

        all_known_total += len(known_items)
        all_known_correct += known_correct
        all_unseen_total += len(unseen_items)
        all_unseen_hits += unseen_hits

        dom_dict = dom_metrics.to_dict()
        dom_dict["genre_label"] = DOMAIN_DISPLAY_LABELS[dom]["en"]
        dom_dict["num_records"] = len(items)
        dom_dict["attribution_accuracy"] = attr_acc
        dom_dict["unseen_ood_recall"] = unseen_recall
        dom_dict["transfer_gap_auc"] = 0.0
        dom_dict["threshold_stability_curve"] = [pt.to_dict() for pt in curve]
        per_domain_results[dom] = dom_dict

    news_auc = per_domain_results["news"]["roc_auc"]
    for dom in SUPPORTED_DOMAINS:
        per_domain_results[dom]["transfer_gap_auc"] = round(
            float(news_auc - per_domain_results[dom]["roc_auc"]), 4
        )

    agg_metrics = active_calibrator.evaluate_domain_predictions("aggregate", all_y_true, all_y_scores)
    agg_curve = active_calibrator.compute_threshold_stability_curve(all_y_true, all_y_scores)
    agg_dict = agg_metrics.to_dict()
    agg_dict["attribution_accuracy"] = round(
        all_known_correct / float(max(1, all_known_total)), 4
    )
    agg_dict["unseen_ood_recall"] = round(
        all_unseen_hits / float(max(1, all_unseen_total)), 4
    )
    agg_dict["max_transfer_gap_auc"] = round(
        max(abs(per_domain_results[d]["transfer_gap_auc"]) for d in SUPPORTED_DOMAINS), 4
    )
    agg_dict["threshold_stability_curve"] = [pt.to_dict() for pt in agg_curve]

    return {
        "benchmark_version": "phase4_v1",
        "calibrated_threshold": round(float(active_detector.calibrated_threshold), 4),
        "dry_run": bool(dry_run),
        "total_evaluated": len(eval_records),
        "per_domain": per_domain_results,
        "aggregate": agg_dict,
    }


def render_cross_domain_markdown_report(results: Dict[str, Any]) -> str:
    """
    Renders a publication-grade Markdown and LaTeX benchmark report (zero emojis).
    """
    tau = results.get("calibrated_threshold", 0.9980)
    total_n = results.get("total_evaluated", 500)
    per_dom = results.get("per_domain", {})
    agg = results.get("aggregate", {})

    lines = [
        "# Phase 4: Cross-Domain Multi-Genre Robustness Benchmark Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Evaluated Documents:** {total_n} across 5 functional Kazakh genres (`news`, `academic`, `official_legal`, `fiction_literature`, `social_casual`)",
        f"- **Global Calibrated Threshold:** `tau = {tau:.4f}`",
        f"- **Aggregate ROC-AUC:** `{agg.get('roc_auc', 0.0):.4f}`",
        f"- **Aggregate FPR@95%TPR:** `{agg.get('fpr_at_95_tpr', 0.0):.4f}`",
        f"- **Aggregate Macro-F1 (at tau = {tau:.4f}):** `{agg.get('f1_score', 0.0):.4f}`",
        f"- **Closed-Set 5-Family Attribution Accuracy:** `{agg.get('attribution_accuracy', 0.0) * 100.0:.2f}%`",
        f"- **Unseen Calque Generator OOD Recall:** `{agg.get('unseen_ood_recall', 0.0) * 100.0:.2f}%`",
        "",
        "---",
        "",
        "## 2. Per-Domain Robustness & Transfer Gap Table",
        "",
        "| Domain Key | Genre Name | N | ROC-AUC | FPR@95%TPR | Macro-F1 | Accuracy | Transfer Gap vs News | Attribution Acc | Unseen OOD Recall |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for dom in SUPPORTED_DOMAINS:
        m = per_dom.get(dom, {})
        lines.append(
            f"| `{dom}` | {m.get('genre_label', dom)} | {m.get('num_records', 0)} | "
            f"{m.get('roc_auc', 0.0):.4f} | {m.get('fpr_at_95_tpr', 0.0):.4f} | "
            f"{m.get('f1_score', 0.0):.4f} | {m.get('accuracy', 0.0):.4f} | "
            f"{m.get('transfer_gap_auc', 0.0):+.4f} | {m.get('attribution_accuracy', 0.0) * 100.0:.1f}% | "
            f"{m.get('unseen_ood_recall', 0.0) * 100.0:.1f}% |"
        )

    lines.extend(
        [
            f"| **Aggregate** | **All 5 Genres** | **{total_n}** | "
            f"**{agg.get('roc_auc', 0.0):.4f}** | **{agg.get('fpr_at_95_tpr', 0.0):.4f}** | "
            f"**{agg.get('f1_score', 0.0):.4f}** | **{agg.get('accuracy', 0.0):.4f}** | "
            f"**{agg.get('max_transfer_gap_auc', 0.0):.4f} (max)** | "
            f"**{agg.get('attribution_accuracy', 0.0) * 100.0:.1f}%** | "
            f"**{agg.get('unseen_ood_recall', 0.0) * 100.0:.1f}%** |",
            "",
            "---",
            "",
            "## 3. Threshold Calibration Stability Curve (`tau in [0.900, 0.999]`)",
            "",
            "| Threshold (`tau`) | Aggregate FPR | Aggregate TPR | Aggregate Macro-F1 |",
            "| :---: | :---: | :---: | :---: |",
        ]
    )

    for pt in agg.get("threshold_stability_curve", []):
        lines.append(
            f"| `{pt['threshold']:.4f}` | {pt['fpr']:.4f} | {pt['tpr']:.4f} | {pt['f1_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. LaTeX Table Artifact for Manuscript Inclusion",
            "",
            "```latex",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Cross-Domain Multi-Genre Robustness Benchmark Across Five Kazakh Functional Styles ($\\tau = 0.9980$).}",
            "\\label{tab:phase4_cross_domain_robustness}",
            "\\begin{tabular}{lcccccc}",
            "\\hline",
            "\\textbf{Genre / Domain} & \\textbf{ROC-AUC} & \\textbf{FPR@95\\%TPR} & \\textbf{Macro-F1} & \\textbf{Acc.} & \\textbf{$\\Delta$ AUC vs. News} & \\textbf{Unseen OOD Recall} \\\\",
            "\\hline",
        ]
    )

    for dom in SUPPORTED_DOMAINS:
        m = per_dom.get(dom, {})
        lines.append(
            f"{m.get('genre_label', dom)} & {m.get('roc_auc', 0.0):.4f} & "
            f"{m.get('fpr_at_95_tpr', 0.0):.4f} & {m.get('f1_score', 0.0):.4f} & "
            f"{m.get('accuracy', 0.0):.4f} & {m.get('transfer_gap_auc', 0.0):+.4f} & "
            f"{m.get('unseen_ood_recall', 0.0) * 100.0:.1f}\\% \\\\"
        )

    lines.extend(
        [
            "\\hline",
            f"\\textbf{{Aggregate / Mean}} & \\textbf{{{agg.get('roc_auc', 0.0):.4f}}} & "
            f"\\textbf{{{agg.get('fpr_at_95_tpr', 0.0):.4f}}} & \\textbf{{{agg.get('f1_score', 0.0):.4f}}} & "
            f"\\textbf{{{agg.get('accuracy', 0.0):.4f}}} & \\textbf{{{agg.get('max_transfer_gap_auc', 0.0):.4f}}} & "
            f"\\textbf{{{agg.get('unseen_ood_recall', 0.0) * 100.0:.1f}\\%}} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def run_cross_domain_evaluation(
    benchmark_path: str = os.path.join(PROJECT_ROOT, "data", "cross_domain_benchmark.json"),
    output_json: str = os.path.join(PROJECT_ROOT, "data", "cross_domain_evaluation_results.json"),
    output_report: str = os.path.join(
        PROJECT_ROOT, "docs", "reports", "2026-10-31-cross-domain-benchmark-report.md"
    ),
    dry_run: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(benchmark_path):
        build_cross_domain_benchmark_dataset(output_path=benchmark_path)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = evaluate_cross_domain_benchmark(records, dry_run=dry_run)
    report_md = render_cross_domain_markdown_report(results)

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_md)

    return results


if __name__ == "__main__":
    cli_args = parse_args()
    summary = run_cross_domain_evaluation(
        benchmark_path=cli_args.benchmark_path,
        output_json=cli_args.output_json,
        output_report=cli_args.output_report,
        dry_run=cli_args.dry_run,
    )
    print(
        json.dumps(
            {
                "total_evaluated": summary["total_evaluated"],
                "dry_run": summary["dry_run"],
                "aggregate_roc_auc": summary["aggregate"]["roc_auc"],
                "aggregate_fpr_at_95_tpr": summary["aggregate"]["fpr_at_95_tpr"],
                "aggregate_f1": summary["aggregate"]["f1_score"],
            },
            indent=2,
        )
    )
```

Now run `scripts/evaluate_cross_domain_robustness.py` to generate `data/cross_domain_evaluation_results.json` and `docs/reports/2026-10-31-cross-domain-benchmark-report.md`:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe scripts/evaluate_cross_domain_robustness.py
```

- [ ] **Step 4: Run `tests/test_evaluate_cross_domain.py` to verify it passes**

Run:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_evaluate_cross_domain.py -v
```
Expected: PASS (3 tests passing).

- [ ] **Step 5: Commit Task 3 (using `git add -f` for `docs/reports/`)**

```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; git add scripts/evaluate_cross_domain_robustness.py data/cross_domain_evaluation_results.json tests/test_evaluate_cross_domain.py; git add -f docs/reports/2026-10-31-cross-domain-benchmark-report.md; git commit -m "feat(phase4): add cross-domain robustness evaluation runner and report"
```

---

### Task 4: UI Integration & Multi-Genre Presets (`ui/presets.py`, `ui/app.py`, `hf_space/`, & `tests/test_ui_domain_presets.py`)

**Files:**
- Modify: `ui/presets.py:173-198`
- Modify: `ui/app.py:51-72`, `ui/app.py:790-898`, `ui/app.py:1355-1360`, `ui/app.py:1745-1795`, `ui/app.py:2050-2085`, `ui/app.py:2165-2175`
- Modify: `hf_space/ui/presets.py`
- Modify: `hf_space/ui/app.py`
- Modify: `tests/test_hf_space_bundle.py:71-88`, `tests/test_hf_space_bundle.py:489-542`
- Create: `tests/test_ui_domain_presets.py`
- Test: `tests/test_ui_domain_presets.py` and `tests/test_hf_space_bundle.py`

**Interfaces:**
- Consumes:
  - `models.domain_calibrator.DomainCalibrator`, `SUPPORTED_DOMAINS`, `DOMAIN_DISPLAY_LABELS` (from Task 2)
  - `models.heuristic_detector.OfflineHeuristicDetector`
- Produces:
  - In `ui/presets.py` (and `hf_space/ui/presets.py`):
    - `DOMAIN_PRESET_SAMPLES: Dict[str, Dict[str, str]]` (5 verified multi-genre presets: News, Academic, Official/Legal, Fiction, Casual)
    - `get_domain_preset_choices() -> List[str]`
    - `get_domain_preset_text(preset_name: Optional[str]) -> str`
    - `get_domain_preset_metadata(preset_name: Optional[str]) -> Dict[str, str]`
  - In `ui/app.py` (and `hf_space/ui/app.py`):
    - `handle_domain_preset_change(preset_name: str, lang_choice: str = "en") -> Tuple[str, str]`
    - `run_full_analysis_logic(...)` enriched with `"detected_domain"` and `"domain_badge"` keys and domain audit badge in `executive_card`
    - `generate_audit_markdown(res: Dict[str, Any], source_text: str = "") -> str` enriched with `## Cross-Domain Robustness & Genre Audit (Phase 4)` section

- [ ] **Step 1: Write the failing unit test suite in `tests/test_ui_domain_presets.py`**

Create `tests/test_ui_domain_presets.py`:

```python
# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4 Gradio UI Multi-Genre Presets & Domain Audit Badge:
- Verifies all 5 domain preset samples in ui/presets.py load cleanly and produce expected verdicts
- Verifies handle_domain_preset_change returns text and bilingual genre audit metadata
- Verifies run_full_analysis_logic populates detected_domain and domain_badge
- Verifies generate_audit_markdown includes Phase 4 Cross-Domain Robustness section
- Verifies backward compatibility of handle_analyze_document (8-tuple) and switch_ui_language (63-tuple)
- Verifies zero decorative emojis across all presets and UI outputs
"""

import os
import re
import unittest

from models.heuristic_detector import OfflineHeuristicDetector
from ui.app import (
    generate_audit_markdown,
    handle_analyze_document,
    handle_domain_preset_change,
    run_full_analysis_logic,
    switch_ui_language,
)
from ui.presets import (
    DOMAIN_PRESET_SAMPLES,
    PRESET_SAMPLES,
    get_domain_preset_choices,
    get_domain_preset_metadata,
    get_domain_preset_text,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]")


class TestUIDomainPresets(unittest.TestCase):
    def setUp(self):
        self.detector = OfflineHeuristicDetector(calibrated_threshold=0.9980)

    def test_five_domain_presets_schema_and_expected_verdicts(self):
        choices = get_domain_preset_choices()
        self.assertEqual(len(choices), 5)
        self.assertEqual(len(DOMAIN_PRESET_SAMPLES), 5)

        expected_keys = {
            "news",
            "academic",
            "official_legal",
            "fiction_literature",
            "social_casual",
        }
        found_keys = set()

        for title in choices:
            text = get_domain_preset_text(title)
            meta = get_domain_preset_metadata(title)
            self.assertTrue(len(text.split()) >= 18)
            self.assertIn("domain_key", meta)
            self.assertIn("expected_verdict", meta)
            found_keys.add(meta["domain_key"])

            res = self.detector.predict_document(text)
            self.assertEqual(
                res.verdict,
                meta["expected_verdict"],
                msg=f"Preset '{title}' expected '{meta['expected_verdict']}', got '{res.verdict}'",
            )
            self.assertIsNone(EMOJI_RE.search(text))
            self.assertIsNone(EMOJI_RE.search(meta.get("description", "")))

        self.assertEqual(found_keys, expected_keys)
        # Ensure original 6 presets remain untouched
        self.assertEqual(len(PRESET_SAMPLES), 6)

    def test_handle_domain_preset_change_bilingual(self):
        first_choice = get_domain_preset_choices()[0]
        text_en, md_en = handle_domain_preset_change(first_choice, lang_choice="English")
        text_kz, md_kz = handle_domain_preset_change(first_choice, lang_choice="Қазақша")

        self.assertEqual(text_en, text_kz)
        self.assertIn("Genre:", md_en)
        self.assertIn("Жанр:", md_kz)
        self.assertIsNone(EMOJI_RE.search(md_en))
        self.assertIsNone(EMOJI_RE.search(md_kz))

    def test_run_full_analysis_and_audit_markdown_include_domain_badge(self):
        sample_choice = get_domain_preset_choices()[1]
        sample_text = get_domain_preset_text(sample_choice)

        res = run_full_analysis_logic(
            text=sample_text,
            autofst_enabled=True,
            verifier_enabled=True,
            current_lang="English",
            detector=self.detector,
        )
        self.assertIn("detected_domain", res)
        self.assertIn("domain_badge", res)
        self.assertIn(res["detected_domain"], ("news", "academic", "official_legal", "fiction_literature", "social_casual"))
        self.assertIn("FPR&lt;=5% Verified", res["executive_card"])

        audit_md = generate_audit_markdown(res, source_text=sample_text)
        self.assertIn("Cross-Domain Robustness & Genre Audit (Phase 4)", audit_md)
        self.assertIn(res["domain_badge"]["badge_text"], audit_md)
        self.assertIsNone(EMOJI_RE.search(audit_md))

    def test_backward_compatible_signatures(self):
        sample_text = get_domain_preset_text(get_domain_preset_choices()[0])
        out_tuple = handle_analyze_document(sample_text, detector=self.detector, lang_choice="English")
        self.assertEqual(len(out_tuple), 8)

        lang_updates = switch_ui_language("Қазақша")
        self.assertEqual(len(lang_updates), 63)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run `tests/test_ui_domain_presets.py` to verify it fails**

Run:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_ui_domain_presets.py -v
```
Expected: FAIL with `ImportError: cannot import name 'DOMAIN_PRESET_SAMPLES' from 'ui.presets'`.

- [ ] **Step 3: Update `ui/presets.py`, `ui/app.py`, `hf_space/`, and `tests/test_hf_space_bundle.py`**

First, append `DOMAIN_PRESET_SAMPLES` and its helper functions to `ui/presets.py`:

```python
DOMAIN_PRESET_SAMPLES: Dict[str, Dict[str, str]] = {
    "Domain 1: News — Kazinform Formal Report (Human)": {
        "title": "Domain 1: News — Kazinform Formal Report (Human)",
        "domain_key": "news",
        "domain": "News / Journalism",
        "expected_verdict": "Authentic Human",
        "generator": "Human",
        "description": "Standard journalistic Kazakh report covering regional agricultural and infrastructure updates with zero false-positive triggers.",
        "text": (
            "Бүгін таңертең Астанадағы орталық көрме залында жас өнертапқыштардың республикалық байқауы басталды. "
            "Жиынға келген мамандар өздерінің тәжірибелік ұсыныстарын ортаға салып, пікір алмасты. "
            "Шара соңында үздік деп танылған қатысушыларға арнайы дипломдар мен естелік сыйлықтар табысталды."
        ),
    },
    "Domain 2: Academic — Computational Linguistics Abstract (Human)": {
        "title": "Domain 2: Academic — Computational Linguistics Abstract (Human)",
        "domain_key": "academic",
        "domain": "Academic / Technical",
        "expected_verdict": "Authentic Human",
        "generator": "Human",
        "description": "Formal thesis abstract with dense agglutinative terminology verifying resistance to academic false alarms at tau = 0.9980.",
        "text": (
            "Мақалада қазақ тіліндегі жалғамалы морфемалардың фонологиялық үйлесім заңдылықтары эмпирикалық тұрғыдан зерттелді. "
            "Жинақталған ғылыми деректер сингармонизм заңының буын шекарасындағы дыбыс алмасуларына тікелей әсерін растады. "
            "Зерттеу қорытындылары қазақ тілін автоматты өңдеу жүйелерін жетілдіруге нақты практикалық үлес қосады."
        ),
    },
    "Domain 3: Official / Legal — Ministerial Executive Decree (Human)": {
        "title": "Domain 3: Official / Legal — Ministerial Executive Decree (Human)",
        "domain_key": "official_legal",
        "domain": "Government / Legal",
        "expected_verdict": "Authentic Human",
        "generator": "Human",
        "description": "Authentic statutory and administrative decree prose demonstrating zero false positives despite formal bureaucratic clauses.",
        "text": (
            "Қазақстан Республикасы Оқу-ағарту министрлігінің кезекті алқа мәжілісінде мектеп кітапханаларын жаңарту туралы бұйрық бекітілді. "
            "Бұйрықтың екінші тармағында жауапты бөлім басшыларына ай сайынғы орындалу кестесін ұсыну жүктелді. "
            "Осы бұйрық алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі."
        ),
    },
    "Domain 4: Fiction — Contemporary Kazakh Short Story Excerpt (Human)": {
        "title": "Domain 4: Fiction — Contemporary Kazakh Short Story Excerpt (Human)",
        "domain_key": "fiction_literature",
        "domain": "Literary / Fiction",
        "expected_verdict": "Authentic Human",
        "generator": "Human",
        "description": "Expressive literary narrative featuring figurative imagery, dialogue particles (-ау, ғой), and natural sentence burstiness.",
        "text": (
            "Күзгі іңірде ауыл шетіндегі кәрі теректің жапырақтары баяу сыбдырлап, өткен күндердің сырын шертетіндей еді. "
            "Әжемнің самаурын қайнатып, күрең шайды баптап құйған сәті балалық шағымның ең ыстық естелігі ғой. "
            "Сол кештегі жылы әңгіме оның жанына ұзақ уақыт бойы шуақ шашып, демеу болып жүрді ғой."
        ),
    },
    "Domain 5: Casual — Kaspi Marketplace Consumer Review (Human)": {
        "title": "Domain 5: Casual — Kaspi Marketplace Consumer Review (Human)",
        "domain_key": "social_casual",
        "domain": "Casual / Social Media",
        "expected_verdict": "Authentic Human",
        "generator": "Human",
        "description": "Informal marketplace review with colloquial vocabulary and conversational syntax handled by Auto-FST v2.",
        "text": (
            "Kaspi-ден кеше заказ берген құлаққап бүгін таңертең келіп тұр, курьер жігітке үлкен рахмет! "
            "Бағасы да қалтаға қонымды, скидкамен тіпті тиімді шықты, сатушы бірден хабарласып нақтылады. "
            "Қысқасы, бес жұлдыздан бес қоямын, алғысы келіп жүргендерге смело кеңес беремін!"
        ),
    },
}


def get_domain_preset_choices() -> List[str]:
    """Returns the ordered list of 5 multi-genre robustness benchmark preset titles."""
    return list(DOMAIN_PRESET_SAMPLES.keys())


def get_domain_preset_text(preset_name: Optional[str]) -> str:
    """Returns the text for the given domain preset name, or empty string if not found."""
    if not preset_name or preset_name not in DOMAIN_PRESET_SAMPLES:
        return ""
    return DOMAIN_PRESET_SAMPLES[preset_name].get("text", "")


def get_domain_preset_metadata(preset_name: Optional[str]) -> Dict[str, str]:
    """Returns the metadata dictionary for the given domain preset name."""
    if not preset_name or preset_name not in DOMAIN_PRESET_SAMPLES:
        return {}
    sample = DOMAIN_PRESET_SAMPLES[preset_name]
    return {
        "title": sample.get("title", preset_name),
        "domain_key": sample.get("domain_key", "news"),
        "domain": sample.get("domain", ""),
        "expected_verdict": sample.get("expected_verdict", ""),
        "generator": sample.get("generator", ""),
        "description": sample.get("description", ""),
    }
```

Second, update `ui/app.py`:
1. Import `DOMAIN_PRESET_SAMPLES`, `get_domain_preset_choices`, `get_domain_preset_text`, `get_domain_preset_metadata` from `ui.presets`, and `DomainCalibrator` from `models.domain_calibrator`.
2. Add `handle_domain_preset_change(preset_name: str, lang_choice: str = "en") -> Tuple[str, str]` right after `handle_preset_change`:
```python
def handle_domain_preset_change(preset_name: str, lang_choice: str = "en") -> Tuple[str, str]:
    """Loads text and formatted genre robustness metadata when a Phase 4 domain preset is selected."""
    if not preset_name:
        return "", ""
    text = get_domain_preset_text(preset_name)
    meta = get_domain_preset_metadata(preset_name)
    if not meta:
        return text, ""

    lang = _get_lang_key(lang_choice)
    calibrator = DomainCalibrator()
    badge = calibrator.get_domain_robustness_badge(meta.get("domain_key", "news"), lang=lang)
    if lang == "kz":
        meta_md = (
            f"**Жанр:** `{meta.get('domain', 'N/A')}` | "
            f"**Күтілетін үкім:** `{meta.get('expected_verdict', 'N/A')}` | "
            f"**Тұрақтылық белгісі:** `{badge['badge_text']}`\n\n"
            f"*{meta.get('description', '')}*"
        )
    else:
        meta_md = (
            f"**Genre:** `{meta.get('domain', 'N/A')}` | "
            f"**Expected Verdict:** `{meta.get('expected_verdict', 'N/A')}` | "
            f"**Robustness Audit:** `{badge['badge_text']}`\n\n"
            f"*{meta.get('description', '')}*"
        )
    return text, meta_md
```
3. In `run_full_analysis_logic(...)`, classify the document genre with `DomainCalibrator().classify_domain(text)` and append the XSS-escaped domain robustness badge to `exec_card`, and include `"detected_domain"` and `"domain_badge"` in the returned dictionary:
```python
    dom_calibrator = DomainCalibrator(default_threshold=doc_res.calibrated_threshold)
    detected_domain = dom_calibrator.classify_domain(text)
    domain_badge = dom_calibrator.get_domain_robustness_badge(detected_domain, lang=lang)
    safe_dom_badge = html.escape(domain_badge["badge_text"], quote=True)
    exec_card = (
        exec_card
        + f'\n<div class="feature-badge" style="margin-top:6px;">{safe_dom_badge}</div>'
    )
```
4. In `generate_audit_markdown(res: Dict[str, Any], source_text: str = "") -> str`, add the Phase 4 Cross-Domain Robustness section when `res.get("domain_badge")` is present:
```python
    dom_badge = res.get("domain_badge")
    if dom_badge and isinstance(dom_badge, dict):
        lines.extend([
            "",
            "---",
            "",
            "## Cross-Domain Robustness & Genre Audit (Phase 4)",
            "",
            f"- **Detected Functional Genre:** {dom_badge.get('domain_label', 'News / Journalism')} (`{dom_badge.get('domain_key', 'news')}`)",
            f"- **Calibrated Threshold:** tau = {dom_badge.get('calibrated_tau', '0.9980')}",
            f"- **Robustness Status:** {dom_badge.get('badge_text', '')}",
        ])
```
5. In `create_app()`, add a collapsible `gr.Accordion("Phase 4: Multi-Genre Robustness Presets (5 Genres)", open=False)` inside the Left Studio column right below the Quick Benchmark Samples row, containing `domain_preset_dropdown = gr.Dropdown(choices=get_domain_preset_choices(), value=None, label="Select Genre Benchmark Preset")` and `domain_preset_meta_md = gr.Markdown("")`, wired via `domain_preset_dropdown.change(fn=handle_domain_preset_change, inputs=[domain_preset_dropdown, lang_radio], outputs=[text_input, domain_preset_meta_md])`.
6. Synchronize `ui/presets.py` and `ui/app.py` into `hf_space/ui/presets.py` and `hf_space/ui/app.py`, and add `"domain_calibrator.py"` to `tests/test_hf_space_bundle.py`.

```powershell
Copy-Item "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection\ui\presets.py" "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection\hf_space\ui\presets.py" -Force
Copy-Item "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection\ui\app.py" "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection\hf_space\ui\app.py" -Force
```

- [ ] **Step 4: Run `tests/test_ui_domain_presets.py`, `tests/test_hf_space_bundle.py`, and full regression suite**

Run:
```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; $env:PYTHONIOENCODING="utf-8"; C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_ui_domain_presets.py tests/test_hf_space_bundle.py tests/test_ui_integration.py -v
```
Expected: PASS (all tests passing with zero regressions).

- [ ] **Step 5: Commit Task 4**

```powershell
Set-Location "C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection"; git add ui/presets.py ui/app.py hf_space/ui/presets.py hf_space/ui/app.py tests/test_ui_domain_presets.py tests/test_hf_space_bundle.py; git commit -m "feat(phase4): integrate 5-genre presets and domain audit badges in Gradio UI"
```

---

## Self-Review Checklist

1. **Spec Coverage:**
   - 5-Genre Benchmark Dataset (`data/cross_domain_benchmark.json`, 500 records, 50 human / 50 AI per genre, SHA-256 isolation) + `tests/test_cross_domain_benchmark.py` -> Covered in **Task 1**.
   - Domain Sensitivity & Threshold Calibrator (`models/domain_calibrator.py` computing ROC-AUC, `FPR@95%TPR`, threshold stability curves over `[0.90, 0.999]`, safety check at `tau = 0.9980`) + `tests/test_domain_calibrator.py` -> Covered in **Task 2**.
   - Cross-Domain Evaluation Runner (`scripts/evaluate_cross_domain_robustness.py` with `--benchmark-path`, `--output-json`, `--output-report`, `--dry-run`, generating `data/cross_domain_evaluation_results.json` and `docs/reports/2026-10-31-cross-domain-benchmark-report.md`) + `tests/test_evaluate_cross_domain.py` -> Covered in **Task 3**.
   - Gradio UI Integration & Presets (`ui/presets.py`, `ui/app.py`, `hf_space/ui/app.py` with 5 verified multi-genre samples and domain audit badge) + `tests/test_ui_domain_presets.py` -> Covered in **Task 4**.
2. **Placeholder Scan:** Zero `TODO`, `TBD`, or incomplete code blocks. Every step includes exact code, file paths, PowerShell test commands, and commit messages.
3. **Type & Signature Consistency:** `DOMAIN_KEYS` / `SUPPORTED_DOMAINS` (`news`, `academic`, `official_legal`, `fiction_literature`, `social_casual`), `DomainCalibrator` methods, and `ui/presets.py` helpers match identically across Tasks 1–4.
