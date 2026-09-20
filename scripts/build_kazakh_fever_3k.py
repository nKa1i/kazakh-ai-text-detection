# -*- coding: utf-8 -*-
"""
scripts/build_kazakh_fever_3k.py: Kazakh-FEVER 3K Data Generation & Validation Pipeline.

Provides tooling for generating candidate claims using few-shot LLM prompts with Hard NEI
enforcement, validating claim schema and linguistic constraints, and stratified dataset partitioning.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import random


def generate_candidate_claims_prompt(article_title: str, text: str, domain: str) -> str:
    """
    Generates a structured few-shot prompt for LLMs (e.g. Qwen-2.5-72B, GPT-4o)
    to generate balanced Kazakh-FEVER claim-evidence pairs with strict Hard NEI enforcement.

    Args:
        article_title: Title of the source article or passage.
        text: Reference passage text.
        domain: Data domain (e.g. 'wikipedia', 'factcheck_kz').

    Returns:
        Formatted prompt string.
    """
    prompt = f"""Сіз — фактілерді тексеруге арналған деректер жинағын (Kazakh-FEVER) дайындайтын сарапшы лингвистсіз.
Берілген мәтін негізінде үш санат бойынша қазақ тіліндегі нақты тұжырымдарды (claims) жасаңыз:
1. SUPPORTS: Мәтіндегі ақпаратпен толық дәлелденетін тұжырым.
2. REFUTES: Мәтіндегі ақпаратқа тікелей қайшы келетін тұжырым (нақты уақыт, сан, тұлға немесе оқиға бұрмаланған).
3. NOT_ENOUGH_INFO (Hard NEI): Мәтіндегі тұлғалар, нысандар мен тақырыпты пайдаланатын, бірақ мәтінде айтылмаған, шындыққа ұқсас қосымша мәліметті қамтитын тұжырым.

МАҢЫЗДЫ ЕРЕЖЕЛЕР (Hard NEI және Span-Grounding талаптары):
- Hard NEI ережесі: NOT_ENOUGH_INFO санатындағы тұжырымдар міндетті түрде берілген мәтіннің тақырыбы мен тұлғаларына қатысты болуы шарт. Мәтінге мүлдем қатысы жоқ жат тақырыптарды (мысалы, ғарыш, басқа мемлекеттер немесе байланыссыз фактілер) қолдануға ҚАТАҢ ТЫЙЫМ САЛЫНАДЫ. Тұжырым шындыққа жанасымды көрінуі тиіс, бірақ берілген мәтін оны растауға да, жоққа шығаруға да жеткіліксіз болуы керек.
- Span-Grounding талабы: SUPPORTS және REFUTES санаттары үшін міндетті түрде нақты дәлел сөйлемді немесе дәйексөзді (`evidence_sentence`) көрсетіңіз.
- Тұжырым ұзындығы: Әр тұжырым 8-ден 35-ке дейінгі сөзден тұруы керек.
- Тіл нормалары: Тұжырымдар табиғи қазақ тілінде, академиялық немесе фактчекингтік стильде құрылуы тиіс.

Кіріс мәліметтері:
- Мақала тақырыбы: {article_title}
- Домен (Domain): {domain}
- Мәтін мазмұны:
\"\"\"{text}\"\"\"

Жауапты төмендегідей JSON форматында қайтарыңыз:
[
  {{
    "claim": "Тұжырым мәтіні осында (8-35 сөз аралығында)",
    "label": "SUPPORTS",
    "evidence_sentence": "Мәтіндегі нақты дәлел сөйлем"
  }},
  {{
    "claim": "Тұжырым мәтіні осында (8-35 сөз аралығында)",
    "label": "REFUTES",
    "evidence_sentence": "Мәтіндегі теріске шығарылатын нақты сөйлем"
  }},
  {{
    "claim": "Тұжырым мәтіні осында (8-35 сөз аралығында)",
    "label": "NOT_ENOUGH_INFO",
    "evidence_sentence": null
  }}
]
"""
    return prompt.strip()


def validate_claim_record(record: dict) -> bool:
    """
    Validates the schema, word count, label legality, and evidence grounding of a claim record.

    Requirements:
        - Required keys: 'id', 'claim', 'evidence_sentences', 'label', 'domain'.
        - Label must be one of: 'SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'.
        - Claim length in tokens (words): 8 <= len(claim.split()) <= 35.
        - If label is 'SUPPORTS' or 'REFUTES', evidence_sentences must be non-empty.

    Args:
        record: Candidate claim dictionary.

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(record, dict):
        return False

    required_keys = {"id", "claim", "evidence_sentences", "label", "domain"}
    if not required_keys.issubset(record.keys()):
        return False

    claim_id = record.get("id")
    if not isinstance(claim_id, str) or not claim_id.strip():
        return False

    domain = record.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        return False

    claim = record.get("claim")
    if not isinstance(claim, str):
        return False

    tokens = claim.strip().split()
    if not (8 <= len(tokens) <= 35):
        return False

    label = record.get("label")
    valid_labels = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
    if label not in valid_labels:
        return False

    evidence = record.get("evidence_sentences")
    if not isinstance(evidence, list):
        return False

    for ev in evidence:
        if not isinstance(ev, str):
            return False

    if label in {"SUPPORTS", "REFUTES"}:
        if len(evidence) == 0:
            return False
        if not any(ev.strip() for ev in evidence):
            return False

    return True


def partition_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.6667,
    dev_ratio: float = 0.1667,
    test_ratio: float = 0.1666,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits claim records into stratified train, dev, and test sets preserving domain and label distributions.

    Args:
        records: List of claim dictionaries.
        train_ratio: Ratio of training split (default: ~0.6667).
        dev_ratio: Ratio of development split (default: ~0.1667).
        test_ratio: Ratio of test split (default: ~0.1666).
        seed: Random seed for reproducible shuffling.

    Returns:
        (train_records, dev_records, test_records)
    """
    if not records:
        return [], [], []

    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        domain = rec.get("domain", "unknown")
        label = rec.get("label", "unknown")
        strata[(str(domain), str(label))].append(rec)

    rng = random.Random(seed)
    train_records: List[Dict[str, Any]] = []
    dev_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for key in sorted(strata.keys()):
        group = list(strata[key])
        rng.shuffle(group)
        n = len(group)
        if n == 0:
            continue

        cut1 = int(round(n * train_ratio))
        cut2 = int(round(n * (train_ratio + dev_ratio)))

        cut1 = max(0, min(n, cut1))
        cut2 = max(cut1, min(n, cut2))

        train_records.extend(group[:cut1])
        dev_records.extend(group[cut1:cut2])
        test_records.extend(group[cut2:])

    return train_records, dev_records, test_records


def _get_mock_candidate_records() -> List[Dict[str, Any]]:
    """Generates a balanced mock dataset for dry-run verification."""
    mock_claims = [
        # Wikipedia - SUPPORTS
        {
            "id": "kz_fever_dry_0001",
            "claim": "Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады.",
            "evidence_sentences": ["Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0002",
            "claim": "Астана қаласы 1997 жылдан бастап Қазақстанның жаңа елордасы ретінде белгіленді.",
            "evidence_sentences": ["1997 жылы елорда Алматы қаласынан Ақмолаға көшіріліп, 1998 жылы қала атауы Астана болып өзгертілді."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0003",
            "claim": "Абай Құнанбайұлы 1845 жылы Семей өңірінде дүниеге келген қазақтың ұлы ақыны болып табылады.",
            "evidence_sentences": ["Абай (Ибраһим) Құнанбайұлы — 1845 жылы Семей өңірінде дүниеге келген ұлы қазақ ақыны, ойшылы және ағартушысы."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0004",
            "claim": "Байқоңыр ғарыш айлағының құрылысы 1955 жылы Қызылорда облысында ресми түрде басталды.",
            "evidence_sentences": ["Оның құрылысы 1955 жылы Қызылорда облысында басталды."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        },
        # Wikipedia - REFUTES
        {
            "id": "kz_fever_dry_0005",
            "claim": "Қазақстан Республикасы 1998 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады.",
            "evidence_sentences": ["Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады."],
            "label": "REFUTES",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0006",
            "claim": "Шымкент қаласы 1997 жылдан бастап Қазақстанның елордасы болып ресми түрде бекітілді.",
            "evidence_sentences": ["1997 жылы елорда Алматы қаласынан Ақмолаға көшіріліп, 1998 жылы қала атауы Астана болып өзгертілді."],
            "label": "REFUTES",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0007",
            "claim": "Абай Құнанбайұлы 1880 жылы Семей өңірінде дүниеге келген ұлы қазақ ойшылы болып табылады.",
            "evidence_sentences": ["Абай (Ибраһим) Құнанбайұлы — 1845 жылы Семей өңірінде дүниеге келген ұлы қазақ ақыны, ойшылы және ағартушысы."],
            "label": "REFUTES",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0008",
            "claim": "Байқоңыр ғарыш айлағының құрылысы 1972 жылы Қызылорда облысында ресми түрде басталған болатын.",
            "evidence_sentences": ["Оның құрылысы 1955 жылы Қызылорда облысында басталды."],
            "label": "REFUTES",
            "domain": "wikipedia"
        },
        # Wikipedia - NOT_ENOUGH_INFO (Hard NEI)
        {
            "id": "kz_fever_dry_0009",
            "claim": "Тәуелсіздік туралы Конституциялық заң жобасын Жоғарғы Кеңестің арнайы құрылған он депутаты әзірледі.",
            "evidence_sentences": [],
            "label": "NOT_ENOUGH_INFO",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0010",
            "claim": "Астана қаласының жаңа бас жоспарын әзірлеу үшін жапондық сәулетші Кисё Курокава шақырылды.",
            "evidence_sentences": [],
            "label": "NOT_ENOUGH_INFO",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0011",
            "claim": "Абай Құнанбайұлы өзінің атақты «Қара сөздері» атты еңбегін Семей қаласындағы баспаханада жазды.",
            "evidence_sentences": [],
            "label": "NOT_ENOUGH_INFO",
            "domain": "wikipedia"
        },
        {
            "id": "kz_fever_dry_0012",
            "claim": "Байқоңыр ғарыш айлағының бас инженері осы нысанға қажетті құрылыс материалдарын Оралдан тасымалдады.",
            "evidence_sentences": [],
            "label": "NOT_ENOUGH_INFO",
            "domain": "wikipedia"
        },
        # Factcheck_kz - SUPPORTS
        {
            "id": "kz_fever_dry_0013",
            "claim": "Қазақстанда коронавирусқа қарсы отандық QazVac вакцинасы 2021 жылдың сәуір айынан бастап салынды.",
            "evidence_sentences": ["Қазақстанда коронавирусқа қарсы QazVac вакцинасы 2021 жылдың сәуір айынан бастап халыққа ресми түрде салына бастады."],
            "label": "SUPPORTS",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0014",
            "claim": "Қазақстанның Ұлттық қорындағы қаржы көлемі экономикалық тұрақтылықты қамтамасыз ету мақсатында құрылған.",
            "evidence_sentences": ["Ұлттық қор болашақ ұрпақтар үшін және қаржылық тұрақтылықты сақтау мақсатында құрылған."],
            "label": "SUPPORTS",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0015",
            "claim": "Әлеуметтік желілерде тараған тегін пәтер беру туралы хабарлама ресми органдармен расталмаған жалған ақпарат.",
            "evidence_sentences": ["Әлеуметтік желіде тараған тегін пәтер беру туралы ақпарат жалған болып шықты."],
            "label": "SUPPORTS",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0016",
            "claim": "Қазақстанда тұрмыстық зорлық-зомбылыққа қарсы заң талаптары соңғы жылдары айтарлықтай қатаңдатылды.",
            "evidence_sentences": ["Қазақстан Республикасында тұрмыстық зорлық-зомбылыққа қатысты жауапкершілік күшейтілді."],
            "label": "SUPPORTS",
            "domain": "factcheck_kz"
        },
        # Factcheck_kz - REFUTES
        {
            "id": "kz_fever_dry_0017",
            "claim": "Қазақстанда коронавирусқа қарсы отандық QazVac вакцинасы 2024 жылдың сәуір айынан бастап салынды.",
            "evidence_sentences": ["Қазақстанда коронавирусқа қарсы QazVac вакцинасы 2021 жылдың сәуір айынан бастап халыққа ресми түрде салына бастады."],
            "label": "REFUTES",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0018",
            "claim": "Қазақстанның Ұлттық қоры тек қана шетелдік компаниялардың инвестициясы есебінен толықтырылып отырады.",
            "evidence_sentences": ["Ұлттық қор болашақ ұрпақтар үшін және қаржылық тұрақтылықты сақтау мақсатында құрылған."],
            "label": "REFUTES",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0019",
            "claim": "Мемлекет барлық азаматтарға кезектен тыс тегін тұрғын үй үлестіру бағдарламасын ресми бекітті.",
            "evidence_sentences": ["Әлеуметтік желіде тараған тегін пәтер беру туралы ақпарат жалған болып шықты."],
            "label": "REFUTES",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0020",
            "claim": "Қазақстанда тұрмыстық зорлық-зомбылыққа қатысты қылмыстық жауапкершілік толығымен заңнан алынып тасталды.",
            "evidence_sentences": ["Қазақстан Республикасында тұрмыстық зорлық-зомбылыққа қатысты жауапкершілік күшейтілді."],
            "label": "REFUTES",
            "domain": "factcheck_kz"
        },
        # Factcheck_kz - NOT_ENOUGH_INFO (Hard NEI)
        {
            "id": "kz_fever_dry_0021",
            "claim": "QazVac вакцинасын дайындаған қазақстандық ғалымдар тобы халықаралық медициналық грантпен толық марапатталды.",
            "evidence_sentences": [],
            "label": "NOT_ENOUGH_INFO",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0022",
            "claim": "Ұлттық қордағы активтердің басым бөлігі швейцариялық мемлекеттік банктердің арнайы есепшотында сақталуда.",
            "evidence_sentences": [],
            "label": "NOT_ENOUGH_INFO",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0023",
            "claim": "Тегін пәтер беру туралы жалған хабарламаны таратқан азаматқа қатысты әкімшілік айыппұл салынды.",
            "evidence_sentences": [],
            "label": "NOT_ENOUGH_INFO",
            "domain": "factcheck_kz"
        },
        {
            "id": "kz_fever_dry_0024",
            "claim": "Тұрмыстық зорлық-зомбылыққа қарсы заң жобасы бойынша қоғамдық тыңдау Алматы қаласында өтті.",
            "evidence_sentences": [],
            "label": "NOT_ENOUGH_INFO",
            "domain": "factcheck_kz"
        },
    ]
    return mock_claims


def validate_file(file_path: str) -> int:
    """
    Validates an existing JSONL dataset file and prints summary statistics.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        0 if all records are valid, 1 if errors found.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    total = 0
    valid_count = 0
    invalid_count = 0
    label_counts: Dict[str, int] = defaultdict(int)
    domain_counts: Dict[str, int] = defaultdict(int)
    token_lengths: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            total += 1
            try:
                rec = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSONDecodeError: {e}", file=sys.stderr)
                invalid_count += 1
                continue

            if validate_claim_record(rec):
                valid_count += 1
                label_counts[rec["label"]] += 1
                domain_counts[rec["domain"]] += 1
                token_lengths.append(len(rec["claim"].split()))
            else:
                invalid_count += 1
                print(f"Line {line_num}: Validation failed for record id='{rec.get('id')}': {rec}", file=sys.stderr)

    print("=" * 60)
    print(f"Dataset Validation Report: {file_path}")
    print(f"Total Records: {total}")
    print(f"Valid Records: {valid_count} ({valid_count / max(1, total) * 100:.1f}%)")
    print(f"Invalid Records: {invalid_count}")
    print("-" * 60)
    print("Label Distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl}: {cnt} ({cnt / max(1, valid_count) * 100:.1f}%)")
    print("-" * 60)
    print("Domain Distribution:")
    for dom, cnt in sorted(domain_counts.items()):
        print(f"  {dom}: {cnt} ({cnt / max(1, valid_count) * 100:.1f}%)")
    if token_lengths:
        print("-" * 60)
        print(f"Token Lengths: min={min(token_lengths)}, max={max(token_lengths)}, avg={sum(token_lengths) / len(token_lengths):.1f}")
    print("=" * 60)

    return 0 if invalid_count == 0 else 1


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for Kazakh-FEVER 3K generation and validation."""
    parser = argparse.ArgumentParser(
        description="Kazakh-FEVER 3K Data Generation & Validation Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/kazakh_knowledge_corpus.jsonl",
        help="Path to input knowledge corpus JSONL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kazakh_fever_3k_candidates.jsonl",
        help="Path to output candidate claims JSONL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate a mock candidate dataset for verification without external APIs"
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate an existing JSONL dataset file and print statistics"
    )
    parser.add_argument(
        "--generate-prompts",
        action="store_true",
        help="Generate sample few-shot prompts using input knowledge corpus"
    )

    parsed_args = parser.parse_args(args if args is not None else sys.argv[1:])

    if parsed_args.validate:
        return validate_file(parsed_args.validate)

    if parsed_args.dry_run:
        records = _get_mock_candidate_records()
        out_dir = os.path.dirname(parsed_args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(parsed_args.output, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Dry run complete: Wrote {len(records)} candidate claims to {parsed_args.output}")
        return 0

    if parsed_args.generate_prompts:
        if not os.path.exists(parsed_args.input):
            print(f"Error: Input file not found: {parsed_args.input}", file=sys.stderr)
            return 1
        with open(parsed_args.input, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 3:
                    break
                p = json.loads(line.strip())
                prompt = generate_candidate_claims_prompt(
                    article_title=p.get("title", ""),
                    text=p.get("text", ""),
                    domain="wikipedia"
                )
                print(f"--- Prompt Sample {idx + 1} ---")
                print(prompt)
                print()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
