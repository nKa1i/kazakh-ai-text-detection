# -*- coding: utf-8 -*-
"""
verification/fever_generator.py: Synthetic Kazakh-FEVER Benchmark Dataset Generator.
Generates balanced SUPPORTED, REFUTED, and NOT ENOUGH INFO claim-evidence pairs
using controlled temporal, negation, and entity perturbations.
"""

import re
import json
from typing import List, Dict, Tuple, Any, Optional

from verification.knowledge_store import KnowledgeStore


YEAR_MUTATION_MAP = {
    "1991": "1998",
    "1997": "2005",
    "1998": "1991",
    "1993": "2001",
    "1995": "1989",
    "1955": "1972",
    "1961": "1975",
    "1845": "1880",
    "1465": "1520",
    "1929": "1940",
    "2018": "2010",
    "2019": "2012",
    "1835": "1870",
    "2015": "2008",
    "1969": "1985",
}

ENTITY_MUTATION_MAP = {
    "Астана": "Шымкент",
    "Алматы": "Павлодар",
    "Байқоңыр": "Сарышаған",
    "Балқаш": "Зайсан",
    "Тоқтар": "Талғат",
    "Шымкент": "Орал",
    "Түркістан": "Тараз",
    "Каспий": "Балқаш",
    "Зайсан": "Алакөл",
    "Алтай": "Қаратау",
    "Ертіс": "Жайық",
    "Сырдария": "Іле",
    "Арал": "Каспий",
    "Мәжіліс": "Сенат",
    "Сенат": "Мәжіліс",
    "Қасым-Жомарт": "Нұрсұлтан",
    "Қаныш": "Мұхтар",
    "Шоқан": "Абай",
    "Ыбырай": "Шоқан",
    "Айдын": "Тоқтар",
}


class KazakhFEVERGenerator:
    """
    Generates balanced, challenging fact verification pairs modeled after the FEVER protocol.
    """

    def __init__(self, knowledge_store: KnowledgeStore):
        self.store = knowledge_store

    def mutate_claim(self, claim_text: str) -> Tuple[str, str]:
        """
        Applies controlled factual mutation (temporal, negation, or entity) to produce a REFUTED claim.
        Returns: (mutated_text, mutation_type)
        """
        text = claim_text.strip()

        # 1. Try temporal mutation first if 4-digit year is present
        years = re.findall(r'\b(1\d{3}|20\d{2})\b', text)
        if years:
            orig_yr = years[0]
            mutated_yr = YEAR_MUTATION_MAP.get(orig_yr, "1995" if orig_yr != "1995" else "1998")
            mutated_text = re.sub(r'\b' + orig_yr + r'\b', mutated_yr, text, count=1)
            return mutated_text, "temporal_mutation"

        # 2. Try entity mutation
        for orig_ent, new_ent in ENTITY_MUTATION_MAP.items():
            if orig_ent in text:
                mutated_text = text.replace(orig_ent, new_ent, 1)
                return mutated_text, "entity_substitution"

        # 3. Fallback: Polar negation mutation
        if " болып табылады" in text:
            mutated_text = text.replace(" болып табылады", " болып табылмайды", 1)
            return mutated_text, "polar_negation"
        elif " емес" in text:
            mutated_text = text.replace(" емес", "", 1)
            return mutated_text, "polar_negation"
        else:
            mutated_text = text.rstrip(".!?") + " емес."
            return mutated_text, "polar_negation"

    def generate_benchmark(self, samples_per_class: int = 12) -> List[Dict[str, Any]]:
        """
        Generates a balanced Kazakh-FEVER benchmark containing:
        - samples_per_class SUPPORTED claims
        - samples_per_class REFUTED claims
        - samples_per_class NOT ENOUGH INFO claims
        """
        passages = list(self.store.passages.values())
        if not passages:
            return []

        benchmark: List[Dict[str, Any]] = []

        # 1. Generate SUPPORTED claims (extract first sentence from passages)
        supported_records = []
        for i in range(samples_per_class):
            p = passages[i % len(passages)]
            first_sent = p.text.split(".")[0].strip() + "."
            supported_records.append({
                "claim": first_sent,
                "evidence_id": p.passage_id,
                "evidence_title": p.title,
                "label": "SUPPORTED",
                "mutation_type": "none"
            })
        benchmark.extend(supported_records)

        # 2. Generate REFUTED claims (mutate sentences)
        refuted_records = []
        for i in range(samples_per_class):
            p = passages[i % len(passages)]
            first_sent = p.text.split(".")[0].strip() + "."
            mutated_text, m_type = self.mutate_claim(first_sent)
            refuted_records.append({
                "claim": mutated_text,
                "evidence_id": p.passage_id,
                "evidence_title": p.title,
                "label": "REFUTED",
                "mutation_type": m_type
            })
        benchmark.extend(refuted_records)

        # 3. Generate NOT ENOUGH INFO claims (unverifiable, out-of-domain propositions)
        nei_records = []
        unrelated_claims = [
            "Марс ғаламшарында тұщы су қоры ресми түрде табылды.",
            "Юпитер планетасының айналасында жүзден астам серік бар екені дәлелденді.",
            "Сахара шөлінде ежелгі алып қаланың қирандылары қазылып алынды.",
            "Антарктида құрлығының астында жылы тұщы көлдер желісі анықталды.",
            "Нью-Йорк қаласында әлемдегі ең биік ағаш ғимарат бой көтерді.",
            "Тынық мұхитының терең түбінен бұрын белгісіз болған балық түрлері табылды.",
            "Күн жүйесінің шетінде тоғызыншы алып ғаламшардың бар екені расталды.",
            "Амазонка ормандарында жаңа емдік өсімдіктер кешені зерттелді.",
            "Токио университетінің ғалымдары сутегімен жүретін ұшақ жасап шығарды.",
            "Венера ғаламшарының атмосферасында фосфин газының белгілері тіркелді.",
            "Мариана шұңғымасында ең терең сүңгуір аппарат жаңа рекорд орнатты.",
            "Африка құрлығында жаңа геологиялық жарылыс пайда болып жатыр.",
        ]
        for i in range(samples_per_class):
            claim_text = unrelated_claims[i % len(unrelated_claims)]
            # Distractor from passages
            distractor = passages[(i + 5) % len(passages)]
            nei_records.append({
                "claim": claim_text,
                "evidence_id": distractor.passage_id,
                "evidence_title": distractor.title,
                "label": "NOT ENOUGH INFO",
                "mutation_type": "out_of_corpus_unverifiable"
            })
        benchmark.extend(nei_records)

        return benchmark

        return benchmark

    def save_to_jsonl(self, records: List[Dict[str, Any]], file_path: str) -> None:
        """Saves generated benchmark records to a JSONL file."""
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
