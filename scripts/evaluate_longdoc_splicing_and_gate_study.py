# -*- coding: utf-8 -*-
"""
scripts/evaluate_longdoc_splicing_and_gate_study.py:
Stage 2 Empirical Evaluation of Long-Document Sliding-Window Splicing Detection
and Dynamic Gate-Weight (g) Adaptation Across Kazakh Linguistic & Orthographic Regimes.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kaz_mage.aggregator import DocumentAggregator
from kaz_mage.chunker import SentencePreservingChunker
from models.heuristic_detector import OfflineHeuristicDetector
from scripts.build_hybrid_mixed_benchmark import (
    AI_SENTENCES,
    GENERATOR_FAMILIES,
    HUMAN_SENTENCES,
)

INJECTION_RATIOS: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.25, 0.50)

# Clean formal in-domain Kazakh sentences (journalistic & encyclopedic prose, standard orthography)
FORMAL_CLEAN_SENTENCES: List[str] = [
    "Қазақстан Республикасының Ұлттық кітапханасында сирек кездесетін тарихи қолжазбалардың көрмесі ашылды.",
    "Алматы облысындағы Іле Алатауы ұлттық табиғи паркінде жаңа туристік бағыттар іске қосылды.",
    "Халықаралық ғылыми симпозиумға жиналған тарихшылар Орталық Азияның ортағасырлық сәулет ескерткіштерін талқылады.",
    "Қарағанды техникалық университетінің зерттеушілері тау-кен өнеркәсібіне арналған жаңа сенсорлық құрылғыны сынақтан өткізді.",
    "Ертіс өзенінің жағалауын абаттандыру жұмыстары биылғы жаз маусымында толық аяқталды.",
    "Астана опера театрында қазақ композиторларының симфониялық шығармаларынан тұратын мерекелік концерт өтті.",
    "Маңғыстау өңіріндегі жерасты су көздерін зерттеу экспедициясы гидрологиялық картаны жаңартты.",
    "Шығыс Қазақстан облысындағы марал шаруашылығы агротуризм саласының дамуына үлес қосты.",
    "Ұлттық музей қорына археологиялық қазба жұмыстары кезінде табылған қола дәуірінің жәдігерлері тапсырылды.",
    "Түркістан қаласындағы жаңа ботаникалық бақта мыңнан астам ағаш көшеттері отырғызылды.",
    "Қазақ тіл білімі институтының мамандары терминологиялық сөздіктің электрондық нұсқасын жариялады.",
    "Жетісу өңіріндегі суармалы егістік алқаптарында тамшылатып суару әдісі кеңінен қолданыла бастады.",
    "Байқоңыр ғарыш айлағында метеорологиялық спутникті ұшыруға дайындық жұмыстары жүргізілді.",
    "Ақтөбе қаласында жас өнертапқыштар мен робототехника үйірмелерінің республикалық жарысы ұйымдастырылды.",
    "Қостанай облысындағы астық жинау науқаны қолайлы ауа райы жағдайында уақытылы аяқталды.",
    "Балқаш көлінің экожүйесін сақтау жөніндегі ғылыми кеңес жылдық есебін ұсынды.",
    "Атырау мұнай және газ университетінде экологиялық зертхана студенттерге есігін ашты.",
    "Павлодар өңіріндегі өндірістік кәсіпорындар ауаны тазарту сүзгілерін жаңартты.",
    "Қызылорда облысында күріштің құрғақшылыққа төзімді жаңа сұрпы тәжірибелік алқапта сыналды.",
    "Семей қаласында Абай мұрасын зерттеуге арналған республикалық оқушылар олимпиадасы қорытындыланды.",
    "Тараз қаласының көне шаһар аумағында жүргізілген археологиялық зерттеулер керуен сарайдың іргетасын ашты.",
    "Көкшетау ұлттық паркінде орман қорын қалпына келтіру жұмыстары жоспарға сай атқарылды.",
    "Орал қаласындағы өлкетану музейінде Жайық өзенінің тарихына арналған мұрағат құжаттары қойылды.",
    "Алматыдағы астрофизика институтының телескоптары арқылы астероидтар белдеуіне бақылау жасалды.",
]

# Colloquial code-switched OOD Kazakh reviews & social media posts (loanword roots + Kazakh suffixes)
COLLOQUIAL_CODESWITCHED_SENTENCES: List[str] = [
    "Каспиден заказ берген тауарымның доставкасы өте тез болды, курьерге рахметімді айттым ғой!",
    "Приложениеде бонустар мен скидкалар күшті екен, оплатасын картамен бірден жасадым.",
    "Инстаграмдағы страницасынан отзывтарды оқып алып едім, качествосы шынымен керемет шықты ғой.",
    "Ватсаптан операторға заявка жібердім, аккаунтымның паролін бес минутта қалпына келтіріп берді.",
    "Бүгін достармен кафеде жақсы вайбтадық, сервис пен менюдің бағасы қатты ұнады ғой.",
    "Телеграмдағы каналдан жаңа файлдарды жүктеп алдым, интернеттің сетьі жақсы ұстап тұр.",
    "Банктің кабинетінде счет пен чекті тексеріп шықтым, комиссиясы мен проценті төмен екен.",
    "Дүкеннен алған ноутбуктың гарантиясы бар, программаларын орнатып бірден жұмысқа қостым ғой.",
    "Курьер заказды уақытында әкеліп берді, возврат жасаудың мүлде керегі болмады, рахмет!",
    "Тиктоктан көрген видеодағы товарды каспиден таптым, скидкамен алғаным өте тиімді болды ғой.",
    "Клиенттерге арналған акцияда бонустар жиналып қалды, келесі покупкамда соны қолданамын.",
    "Қонақүйдің броньын приложениеден рәсімдедім, профилімде барлық чектер сақталып тұр екен.",
    "Досыммен бірге жаңа проект бастадық, чатботтың кодын жазып тестілеуден өткіздік қой.",
    "Камераның фото мен видео сапасы өте жақсы екен, блогерлердің отзывы рас болып шықты.",
    "Оператормен телефон арқылы сөйлестім, сервистің сапасына жоғары баға беремін, жарады!",
    "Парктегі концерттің атмосферасы күшті болды, достармен сториске фотоларды салдық қой.",
    "Компьютердің файлы ашылмай қалып еді, сервистегі шеберлер тез жөндеп берді, рахмет!",
    "Каспидің доставкасы ауылға да жетті, ата-анама сыйлықты заказбен жібергенім керемет болды ғой.",
    "Онлайн курстың сайтындағы видеолар түсінікті екен, сертификатымды профильден жүктеп алдым.",
    "Жаңа телефонның экраны мен батареясы ұнады, курьер қорабын бүлдірмей таза жеткізді ғой.",
    "Стримде көрермендер донат жіберіп қолдау көрсетті, комьюнитидің белсенділігі қуантты ғой.",
    "Каталогтан керекті размерді оңай таптым, оплатасы өткен соң чегі бірден поштаға келді.",
    "Кредиттің суммасын приложениеден ерте жауып тастадым, ешқандай артық комиссия ұстамады ғой.",
    "Демалыста достармен тауда чиллдадық, ауа райы мен табиғаттың көрінісі керемет ұнады!",
]

# Russian-layout corrupted OOD samples: 12 Human (label=0) + 12 AI (label=1) = 24 balanced samples
RUSSIAN_LAYOUT_NOISY_LABELED: List[Dict[str, Any]] = [
    # 12 Russian-layout Human samples (label = 0)
    {"label": 0, "text": "Бугин каспиден заказ бердим, доставкасы оте жаксы болдыгой рахмет! Сапасы мен багасы катты унады."},
    {"label": 0, "text": "Кеше достармен орталыкка барганбыз, бирак уакыт тыгыз болып уйге ерте кайттык кой."},
    {"label": 0, "text": "Дукеннен алган китаптар оте кызыкты екен, балаларга окитын адеми ангимелер коп."},
    {"label": 0, "text": "Курьер тауарды сагат бесте акелди, корабы таза, сапасы жаксы, бизге оте унады рахмет!"},
    {"label": 0, "text": "Ауылдагы ажемизге демалыста барып кайттык, ауасы таза жане тамагы оте дамди болдыгой."},
    {"label": 0, "text": "Приложениеде скидкамен жаксы телефон алдым, багасы арзан жане камерасы кушти тусиреди."},
    {"label": 0, "text": "Бугин жумыста жана проект бастадык, ариптестердин комегимен барин онай биттик кой."},
    {"label": 0, "text": "Уйдеги интернет кеше ошип калды, операторга заявка берген сон тез косып берди рахмет."},
    {"label": 0, "text": "Анамнын туган кунине адеми сыйлык алдым, каспидин жеткизу кызмети оте жылдам жумыс жасайды."},
    {"label": 0, "text": "Досым усынган фильмди отбасымызбен кордик, мазмуны терен жане оте кызык екен гой."},
    {"label": 0, "text": "Жана патерге кошип заттарды жинадык, коршилер оте жаксы адамдар болып шыкты."},
    {"label": 0, "text": "Базардан татти жемистер алдым, багасы колжетимди жане сапасы керемет унады рахмет!"},
    # 12 Russian-layout AI samples (label = 1)
    {"label": 1, "text": "Корытындылай келе, бугинги танда бул маселе заманауи когам ушин манызды рол аткарады. Осыган орай кешенди шаралар кабылданды."},
    {"label": 1, "text": "Осыган орай, мемлекеттик багдарламанын шенберинде экономикалык осуди камтамасыз етеди жане негиз болады. Корытындылай келе натижеси жогары болып табылады."},
    {"label": 1, "text": "Айта кету керек, цифрландыру урдиси барлык салаларда жогары дарежеде дамып келеди жане манызды рол аткарады болып табылады."},
    {"label": 1, "text": "Корытындылай келе, аталган инновациялык шешимдер ондиристик куаттылыкты арттыруга негиз болады жане басым багыт болып саналады."},
    {"label": 1, "text": "Бир жагынан жане екинши жагынан алганда, бул стратегия азаматтардын омир сапасын арттыру куралы болып табылады. Осыган орай жумыстар жалгасуда."},
    {"label": 1, "text": "Атап откен жон, заманауи алемде акпараттык кауипсиздик манызды рол аткарады. Корытындылай келе бул жуйе негиз болады."},
    {"label": 1, "text": "Корытындылай келе, халыкаралык ынтымактастык елдин инвестициялык алеуетин айтарлыктай арттыруга негиз болады болып табылады."},
    {"label": 1, "text": "Осыган орай, билим беру жуйесин жангырту багдарламасы манызды рол аткарады. Айта кету керек бул бастама болып саналады."},
    {"label": 1, "text": "Айта кету керек, экологиялык мониторинг жургизу туракты дамудын шешуши шарты болып табылады. Осыган орай бакылау кушейтилди."},
    {"label": 1, "text": "Корытындылай келе, жасанды интеллект алгоритмдери деректерди ондеуде манызды рол аткарады жане тиимди курал болып табылады."},
    {"label": 1, "text": "Осыган орай, когамдык институттардын катысуымен кабылданган шешимдер негиз болады. Атап откен жон бул манызды рол аткарады."},
    {"label": 1, "text": "Атап откен жон, логистикалык инфракурылымды дамыту сауда айналымын арттыруга негиз болады. Корытындылай келе бул багыт болып табылады."},
]


def _compose_paragraph(sentences: List[str], rng: random.Random, count: int) -> str:
    chosen = [sentences[rng.randrange(len(sentences))] for _ in range(count)]
    return " ".join(chosen)


def build_longdoc_splicing_corpus(
    docs_per_ratio: int = 20,
    target_words: int = 1100,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Constructs a deterministic multi-paragraph long-document Kazakh benchmark (~1,100 words/doc)
    spanning 5 AI injection ratios (0%, 5%, 10%, 25%, 50%) and 5 LLM generator families.
    """
    rng = random.Random(random_seed)
    human_pool = list(HUMAN_SENTENCES) + list(FORMAL_CLEAN_SENTENCES)
    docs: List[Dict[str, Any]] = []
    doc_idx = 1

    for ratio in INJECTION_RATIOS:
        for i in range(docs_per_ratio):
            gen_family = "human" if ratio == 0.0 else GENERATOR_FAMILIES[i % len(GENERATOR_FAMILIES)]
            target_ai_words = int(round(target_words * ratio))
            target_human_words = max(120, target_words - target_ai_words)

            # Build authentic human paragraphs (~85-110 words per paragraph)
            human_paragraphs: List[str] = []
            acc_human_words = 0
            while acc_human_words < target_human_words:
                para = _compose_paragraph(human_pool, rng, count=7)
                human_paragraphs.append(para)
                acc_human_words += len(para.split())

            injected_text = ""
            inject_char_span: Optional[Tuple[int, int]] = None
            placement = "none"

            if ratio > 0.0:
                ai_pool = AI_SENTENCES[gen_family]
                ai_sents: List[str] = []
                acc_ai_words = 0
                # Ensure at least 3 synthetic sentences so an injected span carries realistic discourse structure
                while acc_ai_words < target_ai_words or len(ai_sents) < 3:
                    s = ai_pool[rng.randrange(len(ai_pool))]
                    ai_sents.append(s)
                    acc_ai_words += len(s.split())
                injected_text = " ".join(ai_sents)

                # Rotate placement across interior, late, and early-interior positions
                # (avoid index 0 on 2/3 of docs to expose 512-token truncation blindspots)
                n_h = len(human_paragraphs)
                slot_mode = i % 3
                if slot_mode == 0:
                    insert_pos = max(1, n_h // 2)
                    placement = "middle"
                elif slot_mode == 1:
                    insert_pos = max(1, n_h - 1)
                    placement = "late"
                else:
                    insert_pos = min(2, max(1, n_h // 3))
                    placement = "early_interior"

                prefix_paras = human_paragraphs[:insert_pos]
                suffix_paras = human_paragraphs[insert_pos:]
                prefix_str = "\n\n".join(prefix_paras)
                suffix_str = "\n\n".join(suffix_paras)

                parts = []
                if prefix_str:
                    parts.append(prefix_str)
                start_char = len(prefix_str) + (2 if prefix_str else 0)
                parts.append(injected_text)
                end_char = start_char + len(injected_text)
                if suffix_str:
                    parts.append(suffix_str)
                full_text = "\n\n".join(parts)
                inject_char_span = (start_char, end_char)
                actual_ai_words = len(injected_text.split())
            else:
                full_text = "\n\n".join(human_paragraphs)
                actual_ai_words = 0

            total_w = len(full_text.split())
            docs.append({
                "doc_id": f"longdoc_{doc_idx:04d}",
                "target_ratio": ratio,
                "actual_ai_word_ratio": round(actual_ai_words / float(max(1, total_w)), 4),
                "has_ai_injection": bool(ratio > 0.0),
                "generator": gen_family,
                "placement": placement,
                "total_words": total_w,
                "injected_text": injected_text,
                "inject_char_span": inject_char_span,
                "text": full_text,
            })
            doc_idx += 1

    return docs


def evaluate_longdoc_splicing(
    docs: List[Dict[str, Any]],
    max_words: int = 120,
    calibrated_threshold: float = 0.9980,
) -> Dict[str, Any]:
    """
    Compares 3 document-level aggregation methods on identical sliding-window chunk scores:
    1. truncation_512: Only inspects the first ~350 words (first 3 chunks), mimicking 512-BPE truncation.
    2. mean_pooling: Arithmetic mean across all M chunks against calibrated_threshold (and lenient 0.50).
    3. proposed_topk_pooling: Dynamic Top-K worst-chunk pooling via DocumentAggregator.
    """
    detector = OfflineHeuristicDetector(calibrated_threshold=calibrated_threshold)
    detector.chunker = SentencePreservingChunker(max_words=max_words, overlap_sentences=1)
    detector.aggregator = DocumentAggregator(calibrated_threshold=calibrated_threshold, top_k_cfg=2)

    by_ratio_raw: Dict[str, List[Dict[str, Any]]] = {f"{r:.2f}": [] for r in INJECTION_RATIOS}
    localization_hits = 0
    localization_total = 0

    for doc in docs:
        res = detector.predict_document(doc["text"])
        chunk_probs = [c.ai_probability for c in res.chunks]
        m_chunks = max(1, len(chunk_probs))

        # Baseline 1: 512-subword truncation (~first 240 Kazakh words -> first 2 windows of 120 words)
        trunc_probs = chunk_probs[:2] if chunk_probs else [0.0]
        p_trunc = max(trunc_probs)
        pred_trunc = p_trunc >= calibrated_threshold

        # Baseline 2: Naive document mean pooling across all sliding windows
        p_mean = sum(chunk_probs) / float(m_chunks)
        pred_mean_calibrated = p_mean >= calibrated_threshold
        pred_mean_lenient = p_mean >= 0.50

        # Proposed: Dynamic Top-K worst-chunk pooling (DocumentAggregator)
        pred_topk = res.document_ai_probability >= calibrated_threshold

        span_hit = False
        if doc["has_ai_injection"] and doc["inject_char_span"] is not None:
            localization_total += 1
            inj_s, inj_e = doc["inject_char_span"]
            if res.worst_chunk is not None and res.worst_chunk.is_ai:
                ov_s = max(inj_s, res.worst_chunk.start_char)
                ov_e = min(inj_e, res.worst_chunk.end_char)
                if ov_e > ov_s:
                    span_hit = True
                    localization_hits += 1

        r_key = f"{doc['target_ratio']:.2f}"
        by_ratio_raw[r_key].append({
            "has_ai": doc["has_ai_injection"],
            "pred_trunc": pred_trunc,
            "pred_mean": pred_mean_calibrated,
            "pred_mean_lenient_050": pred_mean_lenient,
            "pred_topk": pred_topk,
            "est_ai_ratio": res.ai_content_ratio,
            "total_chunks": res.total_chunks,
            "span_hit": span_hit,
        })

    # Compute Pure Human (0.00) FPR
    ctrl_rows = by_ratio_raw["0.00"]
    n_ctrl = max(1, len(ctrl_rows))
    pure_human_fpr = {
        "truncation_512": round(100.0 * sum(1 for r in ctrl_rows if r["pred_trunc"]) / n_ctrl, 2),
        "mean_pooling": round(100.0 * sum(1 for r in ctrl_rows if r["pred_mean"]) / n_ctrl, 2),
        "mean_pooling_lenient_050": round(100.0 * sum(1 for r in ctrl_rows if r["pred_mean_lenient_050"]) / n_ctrl, 2),
        "proposed_topk_pooling": round(100.0 * sum(1 for r in ctrl_rows if r["pred_topk"]) / n_ctrl, 2),
    }

    by_ratio_summary: Dict[str, Any] = {}
    all_injected_rows: List[Dict[str, Any]] = []
    for r_key, rows in by_ratio_raw.items():
        n = max(1, len(rows))
        if r_key != "0.00":
            all_injected_rows.extend(rows)
        by_ratio_summary[r_key] = {
            "num_documents": len(rows),
            "mean_chunks_per_doc": round(sum(r["total_chunks"] for r in rows) / float(n), 2),
            "mean_estimated_ai_ratio_pct": round(100.0 * sum(r["est_ai_ratio"] for r in rows) / float(n), 2),
            "truncation_512": {
                "recall_pct": round(100.0 * sum(1 for r in rows if r["pred_trunc"]) / float(n), 2),
            },
            "mean_pooling": {
                "recall_pct": round(100.0 * sum(1 for r in rows if r["pred_mean"]) / float(n), 2),
                "lenient_050_recall_pct": round(
                    100.0 * sum(1 for r in rows if r["pred_mean_lenient_050"]) / float(n), 2
                ),
            },
            "proposed_topk_pooling": {
                "recall_pct": round(100.0 * sum(1 for r in rows if r["pred_topk"]) / float(n), 2),
            },
        }

    n_inj = max(1, len(all_injected_rows))
    overall = {
        "injected_documents": len(all_injected_rows),
        "control_documents": len(ctrl_rows),
        "truncation_512_macro_recall_pct": round(
            100.0 * sum(1 for r in all_injected_rows if r["pred_trunc"]) / float(n_inj), 2
        ),
        "mean_pooling_macro_recall_pct": round(
            100.0 * sum(1 for r in all_injected_rows if r["pred_mean"]) / float(n_inj), 2
        ),
        "mean_pooling_lenient_050_macro_recall_pct": round(
            100.0 * sum(1 for r in all_injected_rows if r["pred_mean_lenient_050"]) / float(n_inj), 2
        ),
        "proposed_topk_macro_recall_pct": round(
            100.0 * sum(1 for r in all_injected_rows if r["pred_topk"]) / float(n_inj), 2
        ),
        "worst_chunk_localization_recall_pct": round(
            100.0 * localization_hits / float(max(1, localization_total)), 2
        ),
    }

    return {
        "total_documents": len(docs),
        "total_words_evaluated": sum(d["total_words"] for d in docs),
        "chunk_max_words": max_words,
        "calibrated_threshold": calibrated_threshold,
        "pure_human_fpr": pure_human_fpr,
        "by_ratio": by_ratio_summary,
        "overall": overall,
    }


def _macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    p1 = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2.0 * p1 * r1 / (p1 + r1)) if (p1 + r1) > 0 else 0.0

    p0 = tn / float(tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2.0 * p0 * r0 / (p0 + r0)) if (p0 + r0) > 0 else 0.0

    return round(100.0 * (f1_1 + f1_0) / 2.0, 2)


def evaluate_dynamic_gate_regimes() -> Dict[str, Any]:
    """
    Evaluates dynamic semantic-vs-morphological gate weights across 3 linguistic regimes
    (24 samples each = 72 total) and measures Auto-FST v2 F1 recovery on Russian-layout noisy text.
    """
    detector = OfflineHeuristicDetector(calibrated_threshold=0.9980)

    regime_texts = {
        "formal_news_wiki": FORMAL_CLEAN_SENTENCES[:24],
        "colloquial_kaspi_social": COLLOQUIAL_CODESWITCHED_SENTENCES[:24],
        "russian_layout_noisy": [item["text"] for item in RUSSIAN_LAYOUT_NOISY_LABELED[:24]],
    }

    regime_gates: Dict[str, List[float]] = {}
    regimes_summary: Dict[str, Any] = {}

    for r_name, texts in regime_texts.items():
        gates = [detector.compute_dynamic_gate(t) for t in texts]
        regime_gates[r_name] = gates
        mean_g = sum(gates) / float(len(gates))
        variance = sum((g - mean_g) ** 2 for g in gates) / float(max(1, len(gates) - 1))
        std_g = math.sqrt(variance)
        regimes_summary[r_name] = {
            "sample_count": len(gates),
            "mean_gate": round(mean_g, 4),
            "std_gate": round(std_g, 4),
            "mean_semantic_pct": round(100.0 * mean_g, 2),
            "mean_morphological_fst_pct": round(100.0 * (1.0 - mean_g), 2),
            "min_gate": round(min(gates), 4),
            "max_gate": round(max(gates), 4),
        }

    g_formal = regime_gates["formal_news_wiki"]
    g_noisy = regime_gates["russian_layout_noisy"]
    t_stat, p_val = stats.ttest_ind(g_formal, g_noisy, equal_var=False)

    m1 = sum(g_formal) / len(g_formal)
    m2 = sum(g_noisy) / len(g_noisy)
    v1 = sum((x - m1) ** 2 for x in g_formal) / max(1, len(g_formal) - 1)
    v2 = sum((x - m2) ** 2 for x in g_noisy) / max(1, len(g_noisy) - 1)
    pooled_sd = math.sqrt((v1 + v2) / 2.0) if (v1 + v2) > 0 else 1e-8
    cohens_d = (m1 - m2) / pooled_sd

    # Noisy Orthographic Detection Ablation: Legacy surface regex (v1) vs Auto-FST v2 detector
    legacy_ai_regex = [
        re.compile(p, re.IGNORECASE | re.UNICODE)
        for p in (
            r"\bқорытындылай келе\b",
            r"\bосыған орай\b",
            r"\bайта кету керек\b",
            r"\bайтарлықтай\b",
            r"\bмаңызды рөл атқарады\b",
            r"\bжоғары дәрежеде\b",
            r"\bатап өткен жөн\b",
            r"\bжүйелі түрде\b",
            r"\bбір жағынан\b",
            r"\bекінші жағынан\b",
            r"\bзаманауи әлемде\b",
            r"\bбүгінгі таңда\b",
            r"\bайқын көрініс табады\b",
            r"\bтиімділігін арттыру\b",
            r"\bболып табылады\b",
        )
    ]

    y_true = [int(item["label"]) for item in RUSSIAN_LAYOUT_NOISY_LABELED]
    y_pred_v1: List[int] = []
    y_pred_v2: List[int] = []

    for item in RUSSIAN_LAYOUT_NOISY_LABELED:
        raw_t = item["text"]
        v1_hits = sum(1 for rx in legacy_ai_regex if rx.search(raw_t))
        v1_prob = min(0.9998, 0.85 + v1_hits * 0.08) if v1_hits > 0 else 0.15
        y_pred_v1.append(1 if v1_prob >= 0.9980 else 0)

        res_v2 = detector.predict_document(raw_t)
        y_pred_v2.append(1 if res_v2.document_ai_probability >= 0.9980 else 0)

    f1_v1 = _macro_f1(y_true, y_pred_v1)
    f1_v2 = _macro_f1(y_true, y_pred_v2)

    return {
        "regimes": regimes_summary,
        "significance_formal_vs_noisy": {
            "welch_t_statistic": round(float(t_stat), 4),
            "welch_p_value": float(p_val),
            "cohens_d": round(float(cohens_d), 4),
        },
        "noisy_detection_ablation": {
            "sample_count": len(RUSSIAN_LAYOUT_NOISY_LABELED),
            "unnormalized_v1_f1_pct": f1_v1,
            "autofst_v2_f1_pct": f1_v2,
            "absolute_f1_gain_pct": round(f1_v2 - f1_v1, 2),
        },
    }


def render_splicing_and_gate_chart(
    splicing_report: Dict[str, Any],
    gate_report: Dict[str, Any],
    output_png: str,
) -> str:
    """Renders a 300-DPI 2-panel academic figure summarizing splicing recall and gate shifts."""
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

    # Panel (a): Long-Document Splicing Detection Recall across Injection Ratios
    ratio_keys = ["0.05", "0.10", "0.25", "0.50"]
    x_labels = ["5%", "10%", "25%", "50%"]
    x_pos = list(range(len(ratio_keys)))

    rec_trunc = [splicing_report["by_ratio"][k]["truncation_512"]["recall_pct"] for k in ratio_keys]
    rec_mean = [splicing_report["by_ratio"][k]["mean_pooling"]["recall_pct"] for k in ratio_keys]
    rec_topk = [splicing_report["by_ratio"][k]["proposed_topk_pooling"]["recall_pct"] for k in ratio_keys]

    ax1.plot(x_pos, rec_topk, marker="o", linewidth=2.4, color="#1d4ed8", label="Proposed Dynamic Top-K Pooling")
    ax1.plot(x_pos, rec_trunc, marker="s", linewidth=2.0, linestyle="--", color="#d97706", label="512-Token First-Window Truncation")
    ax1.plot(x_pos, rec_mean, marker="^", linewidth=2.0, linestyle=":", color="#dc2626", label="Naive Document Mean Pooling")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(-5, 108)
    ax1.set_xlabel("Injected Synthetic Span Ratio in ~1,100-Word Document", fontsize=10)
    ax1.set_ylabel("Hybrid AI Detection Recall (%)", fontsize=10)
    ax1.set_title("(a) Long-Document Splicing Recall vs. Pooling Strategy", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="center right", fontsize=8.5, frameon=True)

    # Panel (b): Dynamic Gate Allocation Across Kazakh Linguistic Regimes
    regime_order = [
        ("formal_news_wiki", "Formal News /\nWikipedia (ID)"),
        ("colloquial_kaspi_social", "Code-Switched\nReviews (OOD)"),
        ("russian_layout_noisy", "Russian-Layout\nNoisy (OOD)"),
    ]
    r_labels = [lbl for _, lbl in regime_order]
    sem_pcts = [gate_report["regimes"][rk]["mean_semantic_pct"] for rk, _ in regime_order]
    morph_pcts = [gate_report["regimes"][rk]["mean_morphological_fst_pct"] for rk, _ in regime_order]

    y_pos = list(range(len(regime_order)))
    bar_w = 0.52
    ax2.bar(y_pos, sem_pcts, width=bar_w, color="#3b82f6", label="Semantic Context Stream (g)")
    ax2.bar(y_pos, morph_pcts, width=bar_w, bottom=sem_pcts, color="#10b981", label="Morphological Auto-FST v2 Stream (1 - g)")

    for idx, (sp, mp) in enumerate(zip(sem_pcts, morph_pcts)):
        ax2.text(idx, sp / 2.0, f"{sp:.1f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax2.text(idx, sp + mp / 2.0, f"{mp:.1f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    ax2.set_xticks(y_pos)
    ax2.set_xticklabels(r_labels, fontsize=9)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Mean Stream Gate Allocation (%)", fontsize=10)
    ax2.set_title("(b) Dynamic Gate Shift Under Domain & Orthographic Noise", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper center", fontsize=8.5, ncol=2, frameon=True)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_png)


def run_full_study(
    output_json: str = "data/longdoc_splicing_and_gate_results.json",
    output_png: str = "data/chart_longdoc_splicing_and_gate_shift.png",
    docs_per_ratio: int = 20,
) -> Dict[str, Any]:
    """Executes the full Stage 2 splicing + dynamic gate empirical study and writes JSON + PNG."""
    docs = build_longdoc_splicing_corpus(docs_per_ratio=docs_per_ratio, target_words=1100, random_seed=42)
    splicing_report = evaluate_longdoc_splicing(docs, max_words=120, calibrated_threshold=0.9980)
    gate_report = evaluate_dynamic_gate_regimes()
    chart_path = render_splicing_and_gate_chart(splicing_report, gate_report, output_png)

    payload = {
        "study_version": "stage2_autofst_v2_2026_10_30",
        "splicing_study": splicing_report,
        "dynamic_gate_study": gate_report,
        "figure_path": os.path.relpath(chart_path, PROJECT_ROOT).replace("\\", "/"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


if __name__ == "__main__":
    results = run_full_study()
    print("Stage 2 Long-Document Splicing & Dynamic Gate Study Summary:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
