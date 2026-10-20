# Design Specification: Auto-FST Demo Prototype

**Document Date:** October 20, 2026  
**Document Status:** Approved Design / Ready for Implementation Planning  
**Target Architecture:** Lightweight Neuro-Symbolic Morphology & Orthographic Normalization  
**Execution Environment:** Standard Python 3.10+ (CPU Only, Zero GPU VRAM, < 10ms execution)  

---

## 1. Executive Summary

This specification defines the architecture, data models, linguistic rules, safety guardrails, and interactive CLI demonstration for **Auto-FST**: an adaptive morphological framework designed for informal, code-switched, and diacritic-flattened Kazakh text.

The system addresses two widespread real-world phenomena in digital Central Asian communication:
1. **Russian Keyboard Diacritic Flattening:** Kazakh speakers frequently type on Cyrillic Russian keyboard layouts, omitting authentic Kazakh characters (`ә, і, ң, ғ, ү, ұ, қ, ө, һ`) and substituting standard Russian counterparts (`а, и/ы, н, г, у, к, о, х`), resulting in severe tokenizer fragmentation (e.g., `тұрсыңғой` typed as `турсынгой`).
2. **Emerging Slang and Code-Switched Loanwords:** Unseen internet colloquialisms and English/Russian loan stems (e.g., `вайбтаған`, `донаттау`, `чиллдеп`, `крашыма`) fail static dictionary lookups, inflating subword ratios ($T/W > 3.0$).

Auto-FST resolves these challenges using a deterministic, lightweight pipeline operating with zero GPU overhead, preserving sub-millisecond execution speeds ($O(L)$) while dynamically expanding its lexical inventory.

---

## 2. Global Constraints and Invariance Rules

1. **Conference Manuscript Guard:** Under no circumstances shall `aist2026/paper.tex` or any files in `papers/kazakh_fever_conference/` be touched or modified (strictly 0 diff lines).
2. **Tone and Style:** Strictly zero decorative emojis across all code, tests, documentation, and console interfaces.
3. **File Encoding:** All Python file I/O operations must explicitly specify `encoding='utf-8'`.
4. **Hardware Suitability:** The entire engine and demonstration must run seamlessly on resource-constrained consumer laptops (requiring 0 MB GPU VRAM and standard library dependencies).
5. **Non-Destructive Integration:** The baseline `fst_analyzer.py` module must remain clean and backward-compatible. Dynamic augmentations are applied via modular overlay classes without in-place mutation of the core codebase.

---

## 3. System Architecture and Component Decomposition

```
                               [ Raw Input Text ]
                     ("донаттаған", "турсынгой", "салем")
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ 1. RussianLayoutNormalizer         │
                     │  • High-frequency irregular table │
                     │  • Suffix phonetic equivalences   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ 2. AutoFSTInducer                 │
                     │  • OOV / inflation detector       │
                     │  • Canonical stem extractor       │
                     │  • Phonetic harmony profiler      │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ 3. RegressionGuard Sandbox        │
                     │  • Golden baseline assertions     │
                     │  • Reversibility gate             │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ 4. DynamicFSTOverlay              │
                     │  • Longest-match-first trie/sort  │
                     │  • In-memory lexicon update       │
                     │  • JSON overlay persistence       │
                     └───────────────────────────────────┘
```

### 3.1 Module 1: `RussianLayoutNormalizer` (`src/auto_fst/normalizer.py`)

#### Responsibilities:
- Normalize high-frequency colloquial phrases typed on Russian keyboards.
- Provide dual-layout regular expressions that recognize suffix chains regardless of whether they are spelled with Kazakh diacritics or flattened Russian letters.

#### Suffix Equivalence Schema:
| Authentic Kazakh Affix | Russian Keyboard Variant | Grammatical Function |
| :--- | :--- | :--- |
| `-ған / -ген` | `-ган / -ген` | Past Participle |
| `-қан / -кен` | `-кан / -кен` | Past Participle (Voiceless Coda) |
| `-сың / -сің` | `-сын / -син` | 2nd Person Singular Verbal Agreement |
| `-сыз / -сіз` | `-сыз / -сиз` | 2nd Person Polite Verbal Agreement |
| `-ғой / -қой` | `-гой / -кой` | Emphatic Modal Particle |
| `-ның / -нің` | `-нын / -нин` | Genitive Noun Case |
| `-дың / -дің` | `-дын / -дин` | Genitive Noun Case |
| `-тың / -тің` | `-тын / -тин` | Genitive Noun Case |
| `-ға / -ге` | `-га / -ге` | Dative Noun Case |
| `-қа / -ке` | `-ка / -ке` | Dative Noun Case |

#### Algorithmic Logic:
1. Exact match against curated irregular dictionary (e.g., `турсынгой` -> `тұрсыңғой`, `жаксы` -> `жақсы`, `салем` -> `сәлем`).
2. Suffix regex expansion: Regex patterns compile alternation tokens: e.g., `r'(ған|ген|қан|кен|ган|ген|кан|кен)$'`.

---

### 3.2 Module 2: `AutoFSTInducer` (`src/auto_fst/inducer.py`)

#### Responsibilities:
- Detect unparsed or poorly segmented tokens in input text.
- Reverse-engineer candidate stems by matching valid morphological suffix sequences.
- Compute phonetic harmony profile for the extracted root.

#### Morphological Candidate Sieve:
1. Strip trailing punctuation, symbols, and whitespace.
2. Check if token already exists in standard Kazakh lexicon or `COMMON_LOANWORD_ROOTS`. If yes, pass through.
3. Identify valid trailing Kazakh affix sequences from known tiers:
   - Verbalizers: `-та, -те, -да, -де, -ла, -ле`
   - Participles / Tenses: `-ған, -ген, -қан, -кен, -ды, -ді, -ты, -ті, -ады, -еді`
   - Persons: `-мыз, -міз, -мын, -мін, -сың, -сің, -сыз, -сіз`
   - Cases: `-ның, -нің, -ға, -ге, -ны, -ні, -да, -де, -дан, -ден, -мен, -бен, -пен`
   - Possessives: `-ым, -ім, -ың, -ің, -ы, -і, -сы, -сі, -ымыз, -іміз`
4. Isolate residual stem candidate $R$.
5. Apply validation filters:
   - Minimum stem length: $\text{len}(R) \ge 3$.
   - Cyrillic alphabetic purity: `re.match(r'^[а-яёәіңғүұқөһ]+$', R, re.I)`.
   - Structural sanity: Residual stem must contain at least one vowel.

#### Phonetic Harmony Profiler:
Compute metadata object:
```json
{
  "stem": "вайб",
  "harmony": "hard",
  "coda_voicing": "voiced",
  "suggested_verbalizer": "-та",
  "timestamp": "2026-10-20T10:50:00Z"
}
```
- **Harmony:** Evaluated based on the last vowel of the stem. If vowel in `[а, о, ұ, ы, я, ё]`, harmony is `hard`. If vowel in `[ә, ө, ү, і, е, э, и]`, harmony is `soft`.
- **Coda Voicing:** Evaluated on the final consonant. If in `[б, в, г, д, ж, з, л, м, н, ң, р, у, й]`, voicing is `voiced`. If in `[п, ф, к, қ, т, с, ш, щ, х, ц, ч]`, voicing is `voiceless`.

---

### 3.3 Module 3: `RegressionGuard` (`src/auto_fst/guard.py`)

#### Responsibilities:
- Act as an automated quality control gate preventing lexicon corruption or degradation of existing parses.
- Execute in-memory regression tests against a golden dataset of canonical sentences prior to committing any newly inducted root.

#### Golden Invariance Test Suite:
1. Standard formal sentence:
   `"Кеше кешке орталыққа келгенмін және жұмыстарды көргендіктен қайтарды."`
   -> Must segment identical to baseline.
2. Code-switched loanword sentence:
   `"Каспиден доставкасы өте тез болды, оплатасын жасадым."`
   -> `доставка` and `оплата` must isolate `-сы` and `-сын`.
3. Hard verbal negative sentence:
   `"Бұл жазбаларыңыздан AI арқылы жасалғандықтан, оларды тексеру керек."`
   -> Verb stems and participial endings must remain stable.

#### Rejection Rules:
- If applying the candidate stem causes any of the golden sentences to alter its segmented output, the candidate is rejected with an explanatory regression error log.
- If the candidate stem matches an existing substring of a native closed-class function word (e.g., pronouns, conjunctions), it is rejected.

---

### 3.4 Module 4: `DynamicFSTOverlay` (`src/auto_fst/overlay.py`)

#### Responsibilities:
- Subclass or wrap `AdvancedKazakhFSTAnalyzer` from `fst_analyzer.py`.
- Maintain a dynamic in-memory set and prioritized list of loanword roots.
- Prevent **Substring Shadowing** by enforcing strict `len(root)` descending order (Longest-Match-First).
- Save and load inducted roots cleanly to `data/custom_slang_lexicon.json`.

#### Data Model (`data/custom_slang_lexicon.json`):
```json
{
  "version": "1.0.0",
  "last_updated": "2026-10-20T10:55:00Z",
  "roots": {
    "вайб": {
      "harmony": "hard",
      "coda": "voiced",
      "origin": "slang_loan",
      "example": "вайбтаған"
    },
    "чилл": {
      "harmony": "soft",
      "coda": "voiced",
      "origin": "slang_loan",
      "example": "чиллдеп"
    },
    "донат": {
      "harmony": "hard",
      "coda": "voiceless",
      "origin": "slang_loan",
      "example": "донаттау"
    },
    "чатбот": {
      "harmony": "hard",
      "coda": "voiceless",
      "origin": "compound_loan",
      "example": "чатботтың"
    }
  }
}
```

---

## 4. Interactive CLI Demonstration (`scripts/demo_auto_fst.py`)

The demonstration script provides two execution modes:

### Mode 1: Automated Benchmark Suite
Executes a comparative benchmark across 6 diverse sentences demonstrating:
1. **Russian-Layout Flattened Kazakh:**
   `"салем калайсын, жумысын калай болып жатыр?"`
   `"Ол али жумыста турсынгой, кешке келедигой."`
2. **Modern Youth Slang & English/Russian Loanwords:**
   `"Биз кеше достармен катты вайбтадык жане чиллдедик."`
   `"Ойынга коп донаттау жасады, енди озинин крашына жазады."`
3. **Compound Shadowing Disambiguation:**
   `"Чатботтын жауабын каспиредпен толедик."`
4. **Noise Rejection:**
   `"Тест бббазар123 жокккк соз."`

Outputs a formatted comparative summary in the terminal:
- Original Sentence
- Baseline FST Parse (showing failures on slang / Russian layout)
- Auto-FST Parse (showing clean segmentation)
- Discovered Roots & Inferred Phonetic Tags
- Execution Time (in milliseconds)

### Mode 2: Interactive REPL
Provides an interactive command prompt allowing the user to type arbitrary Kazakh text:
```text
Auto-FST Interactive Console (Type 'exit' to quit)
Auto-FST > Мен бүгін қатты вайбтадым
[Analyzing...] Time: 0.42 ms
[Detected OOV Slang]: 'вайбтадым' -> Extracted Root: 'вайб' | Suffixes: '-та', '-дым'
[Phonetic Profile]: Harmony=Hard, Coda=Voiced
[Regression Guard]: 15 golden assertions passed.
[Updated Lexicon]: 'вайб' committed to active memory.
[Segmented Result]: Мен бүгін қатты вайб -та -дым
```

---

## 5. File Layout and Repository Placement

```text
c:\Users\Roza\Desktop\projects\kazakh-ai-text-detection\
├── data\
│   └── custom_slang_lexicon.json            (Persistent JSON overlay)
├── src\
│   └── auto_fst\
│       ├── __init__.py                      (Package exports)
│       ├── normalizer.py                    (RussianLayoutNormalizer)
│       ├── inducer.py                       (AutoFSTInducer)
│       ├── guard.py                         (RegressionGuard)
│       └── overlay.py                       (DynamicFSTOverlay)
├── scripts\
│   └── demo_auto_fst.py                     (Interactive CLI demo)
└── tests\
    └── test_auto_fst.py                     (Comprehensive unit tests)
```

---

## 6. Verification and Testing Strategy

The test suite (`tests/test_auto_fst.py`) must rigorously verify:
1. **Russian Layout Invariance:** `турсынгой` correctly segments to root + `-сын` + `-гой`.
2. **Slang Induction:** `вайбтаған`, `донаттау`, `чиллдеп`, `крашыма` successfully induce canonical stems `вайб`, `донат`, `чилл`, `краш`.
3. **Longest-Match Sorting:** `чатботтың` is segmented as `чатбот -тың` and never shadowed by `чат`.
4. **Typo and Noise Rejection:** Random alphanumeric strings and malformed words without Kazakh affixes are rejected without polluting the lexicon.
5. **Zero Regression:** All baseline assertions in `tests/test_fst_code_switch.py` and `tests/test_fst_verbal_rules.py` pass with 100% agreement.
6. **Performance Benchmark:** Single-sentence analysis latency must remain under 5.0 milliseconds on a standard CPU.
