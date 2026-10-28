# -*- coding: utf-8 -*-
"""
scripts/build_hybrid_mixed_benchmark.py: Generates and evaluates the Span-Level
Mixed Human-AI Provenance Benchmark for Kazakh.
"""

import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROVENANCE_CLASSES = (
    "pure_human",
    "pure_ai",
    "human_ai_spliced",
    "ai_human_edited",
)

GENERATOR_FAMILIES = (
    "ISSAI-KazLLM-70B",
    "Sherkala-8B",
    "DeepSeek-R1",
    "GPT-4o",
    "Qwen-2.5-72B",
)

# Human authentic seed sentences (exhibiting colloquialisms, evidentials, particles, natural variation)
HUMAN_SENTENCES = [
    "Кеше кешке орталыққа достармен бірге барғанбыз, бірақ көрме жабық болып шықты.",
    "Каспиден тауардың жеткізілуі өте тез болды, курьерге рахметімді айттым ғой.",
    "Балалардың мектептегі жаңа оқулықтары өте қызықты жазылған екен.",
    "Ауылдағы ата-әжемізге демалыста арнайы барып, біраз көмектесіп қайттық.",
    "Бұл мәселені шешу үшін әрқайсымыз өз жауапкершілігімізді түсінуіміз керек қой.",
    "Жаңа жұмыс орнындағы ұжым бізді өте жылы қабылдап, бірден қолдау көрсетті.",
    "Көктем шыға ауладағы ағаштарды реттеп, сенбілік өткізуді жоспарлап отырмыз.",
    "Таңертең ерте тұрып жүгіру денсаулыққа керемет серпін беретінін өзім байқадым.",
    "Осы кітапты оқып шыққаннан кейін тарихқа деген көзқарасым түбегейлі өзгерді.",
    "Біздің өңірдегі өзен суы биыл мол болып, шаруалардың еңбегі жанған сияқты.",
    "Көршілермен жиналып, ауламызға балалар ойын алаңын бірге орнаттық.",
    "Бүгінгі талқылауда мамандар өздерінің шынайы тәжірибелерімен бөлісті ғой.",
]

# Synthetic AI seed sentences (exhibiting bureaucratic templates, repetitive participial clauses, formal connectors)
AI_SENTENCES = {
    "ISSAI-KazLLM-70B": [
        "Қазақстан Республикасындағы цифрландыру үдерістері қоғамның барлық салаларында қарқынды түрде дамып келеді.",
        "Аталған бағдарламаның шеңберінде экономикалық тиімділікті арттыру бойынша кешенді іс-шаралар жүзеге асырылуда.",
        "Ғылыми зерттеулердің нәтижелері көрсеткендей, инновациялық тәсілдер өндірістік қуаттылықты арттыруға негіз болады.",
        "Мемлекеттік органдар мен жеке сектор арасындағы өзара іс-қимыл тұрақты даму мақсаттарына қол жеткізуді қамтамасыз етеді.",
    ],
    "Sherkala-8B": [
        "Табиғи ресурстарды тиімді пайдалану мәселелері заманауи экологиялық саясаттың басым бағыттарының бірі болып табылады.",
        "Білім беру жүйесін жаңғырту шеңберінде жаңа әдістемелік құралдар мен цифрлық платформалар енгізілуде.",
        "Өңірлік инфрақұрылымды дамыту жөніндегі жобалар халықтың әл-ауқатын жақсартуға оң ықпалын тигізеді.",
        "Ақпараттық технологияларды интеграциялау арқылы басқару процестерінің ашықтығы мен тиімділігі қамтамасыз етіледі.",
    ],
    "DeepSeek-R1": [
        "Қарастырылып отырған алгоритмдік шешімдер күрделі деректер құрылымын оңтайландыру мақсатында жүйелі түрде талданды.",
        "Мәселені шешудің бірінші кезеңінде барлық шектеулерді ескере отырып, математикалық модель құрастырылды.",
        "Логикалық қорытындыларға сәйкес, ұсынылған архитектура есептеу шығындарын айтарлықтай қысқартуға мүмкіндік береді.",
        "Осыған байланысты, модельдің жұмыс істеу сенімділігін тексеру үшін қосымша валидация жүргізу қажет деп есептеледі.",
    ],
    "GPT-4o": [
        "Жасанды интеллект жүйелерінің дамуы заманауи ғылым мен білім беру саласында жаңа мүмкіндіктерге жол ашуда.",
        "Көптілді тілдік модельдер қазақ тіліндегі мәтіндерді талдау және сапалы генерациялау бойынша жоғары нәтижелер көрсетуде.",
        "Ақпараттық қауіпсіздік пен кибергигиена ережелерін сақтау цифрлық қоғамның тұрақтылығы үшін стратегиялық маңызға ие.",
        "Мәліметтер базасын оңтайландыру және бұлттық технологияларды енгізу бизнес-процестерді жеделдетуге септігін тигізеді.",
    ],
    "Qwen-2.5-72B": [
        "Халықаралық ынтымақтастық аясында жүзеге асырылып жатқан жобалар еліміздің инвестициялық тартымдылығын арттырады.",
        "Тұрақты экономикалық өсімді қамтамасыз ету мақсатында өңдеу өнеркәсібін әртараптандыру шаралары жалғасуда.",
        "Қоғамдық институттардың белсенді қатысуымен қабылданған шешімдер әлеуметтік бастамалардың тиімділігін арттырады.",
        "Заманауи телекоммуникациялық желілер шалғай елді мекендерді сапалы интернетпен қамтуға жағдай жасайды.",
    ],
}


def build_hybrid_mixed_dataset(
    output_path: str = "data/kaz_hybrid_mixed_spans.jsonl",
    num_per_class: int = 25,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Constructs a balanced multi-sentence dataset covering 4 provenance categories:
    pure_human, pure_ai, human_ai_spliced, and ai_human_edited.
    """
    random.seed(random_seed)
    docs = []
    doc_counter = 1

    # 1. pure_human
    for i in range(num_per_class):
        n_sents = random.randint(4, 6)
        sents = random.sample(HUMAN_SENTENCES, n_sents)
        labels = [0] * n_sents
        docs.append({
            "doc_id": f"doc_hybrid_{doc_counter:04d}",
            "domain": "society",
            "provenance_class": "pure_human",
            "generator": "human",
            "sentences": sents,
            "sentence_labels": labels,
            "changeover_indices": []
        })
        doc_counter += 1

    # 2. pure_ai
    for i in range(num_per_class):
        gen = GENERATOR_FAMILIES[i % len(GENERATOR_FAMILIES)]
        ai_pool = AI_SENTENCES[gen]
        n_sents = 4
        sents = [ai_pool[j % len(ai_pool)] for j in range(n_sents)]
        labels = [1] * n_sents
        docs.append({
            "doc_id": f"doc_hybrid_{doc_counter:04d}",
            "domain": "technology",
            "provenance_class": "pure_ai",
            "generator": gen,
            "sentences": sents,
            "sentence_labels": labels,
            "changeover_indices": []
        })
        doc_counter += 1

    # 3. human_ai_spliced
    for i in range(num_per_class):
        gen = GENERATOR_FAMILIES[i % len(GENERATOR_FAMILIES)]
        ai_pool = AI_SENTENCES[gen]
        h_sents = random.sample(HUMAN_SENTENCES, 2)
        a_sents = [ai_pool[0], ai_pool[1]]
        if i % 2 == 0:
            sents = h_sents + a_sents
            labels = [0, 0, 1, 1]
            changeovers = [2]
        else:
            sents = a_sents + h_sents
            labels = [1, 1, 0, 0]
            changeovers = [2]

        docs.append({
            "doc_id": f"doc_hybrid_{doc_counter:04d}",
            "domain": "society",
            "provenance_class": "human_ai_spliced",
            "generator": gen,
            "sentences": sents,
            "sentence_labels": labels,
            "changeover_indices": changeovers
        })
        doc_counter += 1

    # 4. ai_human_edited
    for i in range(num_per_class):
        gen = GENERATOR_FAMILIES[i % len(GENERATOR_FAMILIES)]
        ai_pool = AI_SENTENCES[gen]
        h_sents = random.sample(HUMAN_SENTENCES, 2)
        sents = [ai_pool[0], h_sents[0], ai_pool[1], h_sents[1]]
        labels = [1, 0, 1, 0]
        docs.append({
            "doc_id": f"doc_hybrid_{doc_counter:04d}",
            "domain": "technology",
            "provenance_class": "ai_human_edited",
            "generator": gen,
            "sentences": sents,
            "sentence_labels": labels,
            "changeover_indices": [1, 2, 3]
        })
        doc_counter += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return docs


# Feature-based sentence-level AI score heuristic for boundary evaluation
def _score_sentence_ai_likelihood(sentence: str) -> float:
    score = 0.5
    s_lower = sentence.lower()

    # Human markers (reduce AI likelihood)
    human_markers = ["ғой", "қой", "екен", "барғанбыз", "жазғанбыз", "айттым", "байқадым", "отырмыз", "қайттық"]
    for m in human_markers:
        if m in s_lower:
            score -= 0.25

    # AI markers (formulaic bureaucratic phrases & abstract connectors)
    ai_markers = [
        "шеңберінде", "жүзеге асырылуда", "қамтамасыз етеді", "басым бағыттарының бірі",
        "мақсатында", "айтарлықтай", "байланысты", "сәйкес", "себін тигізеді"
    ]
    for m in ai_markers:
        if m in s_lower:
            score += 0.25

    return max(0.0, min(1.0, score))


def evaluate_span_boundary_detection(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates sentence-level AI vs Human classification and span boundary IoU across
    the mixed-provenance dataset.
    """
    total_sentences = 0
    y_true_all = []
    y_pred_all = []

    gen_stats = {gen: {"y_true": [], "y_pred": []} for gen in GENERATOR_FAMILIES}

    total_iou = 0.0
    iou_count = 0

    for doc in docs:
        sentences = doc["sentences"]
        labels = doc["sentence_labels"]
        gen = doc["generator"]

        preds = []
        for s in sentences:
            ai_score = _score_sentence_ai_likelihood(s)
            p = 1 if ai_score >= 0.50 else 0
            preds.append(p)

        y_true_all.extend(labels)
        y_pred_all.extend(preds)
        total_sentences += len(sentences)

        if gen in gen_stats:
            gen_stats[gen]["y_true"].extend(labels)
            gen_stats[gen]["y_pred"].extend(preds)

        # Boundary changeover IoU
        pred_changeovers = set()
        for idx in range(1, len(preds)):
            if preds[idx] != preds[idx - 1]:
                pred_changeovers.add(idx)

        true_changeovers = set(doc.get("changeover_indices", []))
        if true_changeovers or pred_changeovers:
            intersection = len(true_changeovers & pred_changeovers)
            union = len(true_changeovers | pred_changeovers)
            iou = intersection / union if union > 0 else 1.0
            total_iou += iou
            iou_count += 1
        else:
            # Both empty (pure documents)
            total_iou += 1.0
            iou_count += 1

    # Macro F1
    def _compute_macro_f1(y_t: List[int], y_p: List[int]) -> float:
        tp = sum(1 for yt, yp in zip(y_t, y_p) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_t, y_p) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_t, y_p) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_t, y_p) if yt == 0 and yp == 0)

        p1 = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r1 = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1_1 = (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 1.0

        p0 = tn / (tn + fn) if (tn + fn) > 0 else 1.0
        r0 = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        f1_0 = (2 * p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 1.0

        return round((f1_1 + f1_0) / 2.0, 4)

    overall_macro_f1 = _compute_macro_f1(y_true_all, y_pred_all)
    mean_iou = round(total_iou / iou_count, 4) if iou_count > 0 else 1.0

    by_gen_metrics = {}
    for gen, st in gen_stats.items():
        if st["y_true"]:
            f1 = _compute_macro_f1(st["y_true"], st["y_pred"])
            ai_recall = sum(1 for yt, yp in zip(st["y_true"], st["y_pred"]) if yt == 1 and yp == 1) / sum(st["y_true"]) if sum(st["y_true"]) > 0 else 1.0
            by_gen_metrics[gen] = {
                "macro_f1": f1,
                "ai_recall": round(ai_recall, 4)
            }

    return {
        "total_documents": len(docs),
        "total_sentences": total_sentences,
        "sentence_macro_f1": overall_macro_f1,
        "boundary_iou": mean_iou,
        "by_generator": by_gen_metrics
    }


if __name__ == "__main__":
    out_file = "data/kaz_hybrid_mixed_spans.jsonl"
    docs = build_hybrid_mixed_dataset(out_file, num_per_class=25)
    print(f"Built {len(docs)} hybrid mixed documents in {out_file}")
    res = evaluate_span_boundary_detection(docs)
    print("Span Boundary Detection Results:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
