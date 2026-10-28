# -*- coding: utf-8 -*-
"""
scripts/evaluate_noisy_fever_ablation.py:
Orthographically Noisy & Morphophonological FEVER Ablation Benchmark.
Evaluates the retrieval robustness of Legacy FST v1 (AdvancedKazakhFSTAnalyzer)
versus Auto-FST v2 (DynamicFSTOverlay) across orthographically corrupted and
morphophonologically inflected Kazakh claims against the 152-passage knowledge corpus.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fst_analyzer import AdvancedKazakhFSTAnalyzer
from src.auto_fst.overlay import DynamicFSTOverlay
from verification.knowledge_store import KnowledgeStore
from verification.evidence import EvidencePassage
from verification.retriever import HybridEvidenceRetriever


NOISY_FEVER_CLAIMS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------
    # 1. Morphophonological Inflected (16 claims)
    # Stems undergo consonant voicing (п->б, қ->ғ, к->г) and vowel syncope (ы/і dropping)
    # -------------------------------------------------------------
    {
        "claim_id": "morph_01",
        "claim_text": "Қазақстанның аймағы Еуразия құрлығының орталығында орналасқан.",
        "target_passage_id": "wiki_kz_001",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_02",
        "claim_text": "Абай Құнанбайұлының кітабы қазақ халқының рухани қазынасы болып табылады.",
        "target_passage_id": "wiki_kz_005",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_03",
        "claim_text": "Қаныш Сәтбаевтың еңбегі Қазақстан ғылымының дамуына үлкен көмегін тигізді.",
        "target_passage_id": "wiki_kz_015",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_04",
        "claim_text": "Астана қаласы мемлекеттің бас қаласы ретінде жаңа орны белгіленді.",
        "target_passage_id": "wiki_kz_002",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_05",
        "claim_text": "Мұхтар Әуезовтің себебі белгілі зерттеулері Абайтану ғылымының негізін салды.",
        "target_passage_id": "wiki_kz_006",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_06",
        "claim_text": "Шоқан Уәлихановтың жазған дерегі Орталық Азия мен Қашқария тарихын ашты.",
        "target_passage_id": "wiki_kz_016",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_07",
        "claim_text": "Ыбырай Алтынсариннің ашқан мектебі қазақ балаларына арналған алғашқы білім ошағы болды.",
        "target_passage_id": "wiki_kz_017",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_08",
        "claim_text": "Балқаш көлінің балығы мен су ресурстары Қазақстанның табиғи байлығы саналады.",
        "target_passage_id": "wiki_kz_012",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_09",
        "claim_text": "Ахмет Байтұрсынұлының тілегі қазақ тілінің оқулығы мен әліпбиін реформалау еді.",
        "target_passage_id": "wiki_kz_011",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_10",
        "claim_text": "Тоқтар Әубәкіров ұшағы және ғарыш кемесі арқылы қазақтан шыққан тұңғыш ғарышкер атанды.",
        "target_passage_id": "wiki_kz_008",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_11",
        "claim_text": "Қазақ хандығының негізін қалаған ұрпағы Жәнібек пен Керей хандар болған.",
        "target_passage_id": "wiki_kz_004",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_12",
        "claim_text": "Түркістан қаласында Қожа Ахмет Ясауи кесенесінің орны тарихи орталық саналады.",
        "target_passage_id": "wiki_kz_018",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_13",
        "claim_text": "Қазақстан тәуелсіздігі туралы заң қабылданғанда халықтың талабы орындалды.",
        "target_passage_id": "hist_005",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_14",
        "claim_text": "Алтын адам жауынгерінің сүйегі мен киімі Есік обасынан табылды.",
        "target_passage_id": "wiki_kz_026",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_15",
        "claim_text": "Каспий теңізінің аймағы мен жағалауы мұнайға өте бай.",
        "target_passage_id": "wiki_kz_019",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },
    {
        "claim_id": "morph_16",
        "claim_text": "Жамбыл Жабаевтың халыққа арнаған жүрегі мен жырлары өшпес мұра қалдырды.",
        "target_passage_id": "wiki_kz_028",
        "expected_verdict": "SUPPORTED",
        "category": "morphophonological_inflected",
    },

    # -------------------------------------------------------------
    # 2. Russian Layout Noisy (16 claims)
    # Cyrillic native characters replaced with Russian keyboard layout equivalents (қ->к, ғ->г, ұ->у, ү->у, ә->а, і->и, ө->о)
    # -------------------------------------------------------------
    {
        "claim_id": "rus_01",
        "claim_text": "Казакстан Республикасынын тауелсиздиги 1991 жылы жарияланды.",
        "target_passage_id": "wiki_kz_001",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_02",
        "claim_text": "Астана каласы Казакстаннын бас каласы болып табылады.",
        "target_passage_id": "wiki_kz_002",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_03",
        "claim_text": "Алматы каласында биик таулар мен асем табигат орналаскан.",
        "target_passage_id": "wiki_kz_003",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_04",
        "claim_text": "Абай Кунанбайулы казак адебиетинин улы акыны жане агартушысы.",
        "target_passage_id": "wiki_kz_005",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_05",
        "claim_text": "Каныш Сатбаев галымдардын косемдеринин бири жане геолог болган.",
        "target_passage_id": "wiki_kz_015",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_06",
        "claim_text": "Байконыр гарыш айлагынан алемдеги алгашкы гарыш кемелери ушкан.",
        "target_passage_id": "wiki_kz_007",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_07",
        "claim_text": "Ахмет Байтурсынулы казак тилинин окулыктарын жазган галым.",
        "target_passage_id": "wiki_kz_011",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_08",
        "claim_text": "Туркистан каласы Казак хандыгынын тарихи астанасы болган.",
        "target_passage_id": "wiki_kz_018",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_09",
        "claim_text": "Шокан Уалиханов казактын тунгыш галымы жане саяхатшысы.",
        "target_passage_id": "wiki_kz_016",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_10",
        "claim_text": "Балкаш колинин суы бир жагы тущы, бир жагы ащы болып келеди.",
        "target_passage_id": "wiki_kz_012",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_11",
        "claim_text": "Токтар Аубакиров тунгыш казак гарышкерлер катарына жатады.",
        "target_passage_id": "wiki_kz_008",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_12",
        "claim_text": "Ыбырай Алтынсарин казак балалары ушин мектептер ашкан.",
        "target_passage_id": "wiki_kz_017",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_13",
        "claim_text": "Мухтар Ауезов Абай жолы романынын авторы ретинде танымал.",
        "target_passage_id": "wiki_kz_006",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_14",
        "claim_text": "Алтын адам Есик обасынын кужаттары мен корганынан табылган.",
        "target_passage_id": "wiki_kz_026",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_15",
        "claim_text": "Каспий тенизи жер бетиндеги ен улкен туйык кол саналады.",
        "target_passage_id": "wiki_kz_019",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
    {
        "claim_id": "rus_16",
        "claim_text": "Казак хандыгы 1465 жылы Жетису ониринде курылган мемлекет.",
        "target_passage_id": "hist_001",
        "expected_verdict": "SUPPORTED",
        "category": "russian_layout_noisy",
    },
]


def build_noisy_fever_stress_set(output_path: str = "data/kazakh_fever_noisy_stress.jsonl") -> List[Dict[str, Any]]:
    """Builds and serializes the 32-claim Orthographically Noisy & Morphophonological FEVER Stress Set."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in NOISY_FEVER_CLAIMS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return NOISY_FEVER_CLAIMS


def _load_passages(corpus_path: str) -> List[EvidencePassage]:
    """Loads EvidencePassage objects from a jsonl file."""
    passages: List[EvidencePassage] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            passages.append(EvidencePassage(
                passage_id=d.get("passage_id", ""),
                title=d.get("title", ""),
                text=d.get("text", ""),
                source_url=d.get("source_url", "")
            ))
    return passages


def _evaluate_retrieval(
    retriever: HybridEvidenceRetriever,
    claims: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Computes Recall@1, Recall@5, and MRR overall and per category."""
    metrics_by_cat: Dict[str, Dict[str, float]] = {
        "morphophonological_inflected": {"r1": 0, "r5": 0, "mrr": 0.0, "total": 0},
        "russian_layout_noisy": {"r1": 0, "r5": 0, "mrr": 0.0, "total": 0},
    }
    total_r1 = 0
    total_r5 = 0
    total_mrr = 0.0
    total_count = len(claims)

    for item in claims:
        q = item["claim_text"]
        target = item["target_passage_id"]
        cat = item["category"]

        results = retriever.retrieve(q, top_k=5)
        pids = [p.passage_id for p in results]

        rank = 0
        if target in pids:
            rank = pids.index(target) + 1

        reciprocal_rank = (1.0 / rank) if rank > 0 else 0.0
        hit_1 = 1 if rank == 1 else 0
        hit_5 = 1 if (1 <= rank <= 5) else 0

        total_r1 += hit_1
        total_r5 += hit_5
        total_mrr += reciprocal_rank

        if cat in metrics_by_cat:
            metrics_by_cat[cat]["r1"] += hit_1
            metrics_by_cat[cat]["r5"] += hit_5
            metrics_by_cat[cat]["mrr"] += reciprocal_rank
            metrics_by_cat[cat]["total"] += 1

    by_cat_summary = {}
    for cat, data in metrics_by_cat.items():
        n = data["total"]
        by_cat_summary[cat] = {
            "total": n,
            "recall_at_1": round(data["r1"] / n, 4) if n > 0 else 0.0,
            "recall_at_5": round(data["r5"] / n, 4) if n > 0 else 0.0,
            "mrr": round(data["mrr"] / n, 4) if n > 0 else 0.0,
        }

    return {
        "total_claims": total_count,
        "recall_at_1": round(total_r1 / total_count, 4) if total_count > 0 else 0.0,
        "recall_at_5": round(total_r5 / total_count, 4) if total_count > 0 else 0.0,
        "mrr": round(total_mrr / total_count, 4) if total_count > 0 else 0.0,
        "by_category": by_cat_summary,
    }


def run_noisy_fever_ablation(
    stress_path: str = "data/kazakh_fever_noisy_stress.jsonl",
    corpus_path: str = "data/kazakh_knowledge_corpus.jsonl",
    output_path: str = "data/noisy_fever_ablation_results.json"
) -> Dict[str, Any]:
    """
    Executes the head-to-head retrieval ablation: Legacy FST v1 vs. Auto-FST v2.
    """
    if not os.path.exists(stress_path):
        build_noisy_fever_stress_set(stress_path)

    claims: List[Dict[str, Any]] = []
    with open(stress_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))

    passages = _load_passages(corpus_path)

    # 1. Setup Legacy FST v1 Retriever
    store_v1 = KnowledgeStore(fst_analyzer=AdvancedKazakhFSTAnalyzer())
    for p in passages:
        store_v1.add_passage(p)
    retriever_v1 = HybridEvidenceRetriever(knowledge_store=store_v1, dense_weight=0.0)

    # 2. Setup Auto-FST v2 Retriever
    store_v2 = KnowledgeStore(fst_analyzer=DynamicFSTOverlay())
    for p in passages:
        store_v2.add_passage(p)
    retriever_v2 = HybridEvidenceRetriever(knowledge_store=store_v2, dense_weight=0.0)

    # 3. Evaluate both systems
    res_v1 = _evaluate_retrieval(retriever_v1, claims)
    res_v2 = _evaluate_retrieval(retriever_v2, claims)

    report = {
        "benchmark": "Kazakh FEVER Orthographic Noise & Morphological Stress Benchmark",
        "total_claims": len(claims),
        "corpus_passages": len(passages),
        "legacy_fst_v1": res_v1,
        "autofst_v2": res_v2,
        "improvement": {
            "recall_at_1_gain": round(res_v2["recall_at_1"] - res_v1["recall_at_1"], 4),
            "recall_at_5_gain": round(res_v2["recall_at_5"] - res_v1["recall_at_5"], 4),
            "mrr_gain": round(res_v2["mrr"] - res_v1["mrr"], 4),
            "relative_r1_improvement_pct": round(
                ((res_v2["recall_at_1"] - res_v1["recall_at_1"]) / max(0.001, res_v1["recall_at_1"])) * 100.0, 2
            ),
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    rep = run_noisy_fever_ablation()
    print("Noisy FEVER Ablation Benchmark Completed Successfully:")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
