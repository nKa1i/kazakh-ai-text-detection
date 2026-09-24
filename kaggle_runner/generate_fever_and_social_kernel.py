# -*- coding: utf-8 -*-
"""
Kaggle GPU Kernel: Multi-Domain Generalization & Kazakh-FEVER 3K Dataset Generator.
Runs on Kaggle Dual Tesla T4 GPUs using Qwen-2.5-7B-Instruct (4-bit quantized).
Generates:
1. Kazakh-FEVER 3K: Balanced claim-evidence triples (SUPPORTS, REFUTES, Hard NEI).
2. Social Media Benchmark: Colloquial Kazakh synthetic text with slang and code-switching.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional

EMBEDDED_KNOWLEDGE_CORPUS_B64 = ""

PERSONAS = [
    "student",
    "consumer",
    "tech_enthusiast",
    "casual_chat",
    "news_commenter",
    "entrepreneur",
    "gamer",
    "sports_fan",
]

PLATFORMS = [
    "telegram",
    "twitter",
    "instagram",
    "tiktok",
    "vk",
]


def unpack_embedded_corpus(
    payload_b64: str,
    target_path: str = "kazakh_knowledge_corpus.jsonl",
) -> List[Dict[str, Any]]:
    """
    Safely decompresses a base64-encoded gzip payload into target_path if
    it does not already exist, and returns the loaded list of article dicts.

    Args:
        payload_b64: Base64-encoded gzip string containing JSONL data.
        target_path: Destination path for the unpacked corpus.

    Returns:
        List of parsed article dictionaries.
    """
    if os.path.exists(target_path):
        articles: List[Dict[str, Any]] = []
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        articles.append(json.loads(line))
            if articles:
                return articles
        except Exception as e:
            print(f"Warning: Failed reading existing {target_path} ({e}). Re-unpacking payload.")

    clean_b64 = payload_b64.strip() if payload_b64 else ""
    if not clean_b64:
        return []

    compressed_bytes = base64.b64decode(clean_b64.encode("ascii"))
    decompressed_bytes = gzip.decompress(compressed_bytes)
    decompressed_str = decompressed_bytes.decode("utf-8")

    parent_dir = os.path.dirname(os.path.abspath(target_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(decompressed_str)

    articles = []
    for line in decompressed_str.splitlines():
        line = line.strip()
        if line:
            articles.append(json.loads(line))
    return articles


def generate_aspect_prompt(article_title: str, text: str, aspect_idx: int) -> str:
    """
    Constructs an aspect-diverse prompt for generating Kazakh-FEVER triples.

    Aspect 0: Entity & Event Grounding (persons, locations, core events, actions).
    Aspect 1: Numerical & Chronological Grounding (years, dates, quantities, sequences).
    Aspect 2: Causal, Relational & Attribute Grounding (causes, outcomes, relations, properties).

    Args:
        article_title: Title of the source article.
        text: Text passage of the source article.
        aspect_idx: Integer indicator of aspect (modulo 3).

    Returns:
        Formatted prompt string instructing LLM to output JSON array of 3 claims.
    """
    mode = aspect_idx % 3

    if mode == 0:
        aspect_name = "Аспект 0: Тұлғалар мен негізгі оқиғалар (Entity & Event Grounding)"
        aspect_focus = (
            "Тұжырымдар нақты тұлғаларға (кім?), орындар мен нысандарға (қайда?), "
            "негізгі тарихи/қоғамдық оқиғалар мен іс-әрекеттерге негізделуі тиіс.\n"
            "- SUPPORTS: Мәтіндегі нақты тұлғаны, орынды немесе оқиғаны дәл растайтын тұжырым.\n"
            "- REFUTES: Мәтіндегі тұлғаны немесе оқиғаны басқа атаумен, бөтен адаммен не бұрмаланған әрекетпен теріске шығаратын тұжырым.\n"
            "- NOT_ENOUGH_INFO (Hard NEI): Мәтіндегі кейіпкер немесе нысан туралы, бірақ осы мәтінде МҮЛДЕМ АЙТЫЛМАҒАН, ойдан шығарылған шынайы көрінетін қосымша факт."
        )
    elif mode == 1:
        aspect_name = "Аспект 1: Сандық және хронологиялық деректер (Numerical & Chronological Grounding)"
        aspect_focus = (
            "Тұжырымдар нақты жылдарға (қашан?), мерзімдерге, сандық көрсеткіштерге, "
            "өлшемдерге, пайыздық үлестерге және уақыттық реттілікке негізделуі тиіс.\n"
            "- SUPPORTS: Мәтіндегі нақты сан, жыл, уақыт мерзіміне толық сәйкес келетін тұжырым.\n"
            "- REFUTES: Мәтіндегі датаны, ғасырды, жылды немесе сандық өлшемді бұрмалап жоққа шығаратын тұжырым.\n"
            "- NOT_ENOUGH_INFO (Hard NEI): Мәтіндегі тақырыпқа қатысты, бірақ мәтінде АЙТЫЛМАҒАН тың сандық/хронологиялық көрсеткіш."
        )
    else:
        aspect_name = "Аспект 2: Себеп-салдарлық байланыстар мен қасиеттер (Causal, Relational & Attribute Grounding)"
        aspect_focus = (
            "Тұжырымдар құбылыстардың пайда болу себептеріне, салдарына, нәтижелеріне, "
            "ұйымдық/әлеуметтік байланыстарына немесе ғылыми/құқықтық сипаттамаларына негізделуі тиіс.\n"
            "- SUPPORTS: Мәтіндегі себеп-салдарлық байланысты немесе нақты сипаттаманы растайтын тұжырым.\n"
            "- REFUTES: Себеп пен салдарды теріс ауыстыратын, не болмаса қасиетін теріске шығаратын тұжырым.\n"
            "- NOT_ENOUGH_INFO (Hard NEI): Тақырыпқа қатысты, бірақ мәтінде АЙТЫЛМАҒАН қосымша ұйымдық қатынас немесе себептік байланыс."
        )

    prompt = f"""Сен қазақ тіліндегі академиялық фактчекинг (Kazakh-FEVER) деректер жинағын құрастыратын білікті лингвист-мамансың.
Бағыт: {aspect_name}

{aspect_focus}

Мәтін тақырыбы: {article_title}
Мәтін мазмұны: {text}

Талаптар:
1. Тұжырымдар ұзындығы 8-ден 35 сөзге дейін болуы қажет.
2. SUPPORTS және REFUTES үшін 'evidence_sentence' өрісінде мәтіннен нақты дәлел сөйлем болуы шарт.
3. NOT_ENOUGH_INFO үшін 'evidence_sentence' міндетті түрде бос жол ("") болуы тиіс.
4. Жауапты ТЕК келесі JSON форматында қайтар:
[
  {{"claim": "қазақша тұжырым 1", "label": "SUPPORTS", "evidence_sentence": "мәтіндегі нақты дәлел сөйлем"}},
  {{"claim": "қазақша тұжырым 2", "label": "REFUTES", "evidence_sentence": "мәтіндегі теріске шығарылатын сөйлем"}},
  {{"claim": "қазақша тұжырым 3", "label": "NOT_ENOUGH_INFO", "evidence_sentence": ""}}
]"""
    return prompt


def generate_social_prompt(persona: str, platform: str) -> str:
    """
    Constructs a generation prompt for colloquial Kazakh social media text.
    """
    return f"""Сен қазақша әлеуметтік желілерде ({platform}) белсенді жазатын қолданушысың ({persona}).
Бейресми ауызекі қазақ тілінде (жастар сленгі, '-сың ғой', '-ма екен', '-шы/-ші' қосымшалары, күнделікті диалог, ағылшын/орыс сөздерімен араласқан code-switching) 1 қысқа пікір немесе жазба жаз.
Ұзындығы 15-50 сөз болсын. Тек пікірдің өзін қазақша жаз, басқа түсіндірме жазба."""


def mock_llm_query_fever(aspect_idx: int, article_title: str) -> str:
    """
    Generates a deterministic valid mock JSON response for dry-run testing.
    """
    mode = aspect_idx % 3
    if mode == 0:
        return json.dumps([
            {
                "claim": f"{article_title} бойынша негізгі оқиғалар мен тұлғалар деректері толық расталған факт ретінде ұсынылады.",
                "label": "SUPPORTS",
                "evidence_sentence": f"{article_title} туралы деректер мәтінде нақты көрсетілген."
            },
            {
                "claim": f"{article_title} бойынша оқиғалар басқа өңірде мүлдем өтпеген деп қате теріске шығарылады.",
                "label": "REFUTES",
                "evidence_sentence": f"{article_title} оқиғасы осы аймақта орын алғандығы мәтінде бар."
            },
            {
                "claim": f"{article_title} тақырыбына қатысты халықаралық сарапшылардың қосымша зерттеуі жарияланған еді.",
                "label": "NOT_ENOUGH_INFO",
                "evidence_sentence": ""
            }
        ], ensure_ascii=False)
    elif mode == 1:
        return json.dumps([
            {
                "claim": f"{article_title} дерегінде көрсетілген мерзімдер мен жылдық есептер толықтай сәйкес келеді.",
                "label": "SUPPORTS",
                "evidence_sentence": f"{article_title} мәтініндегі уақыт мерзімі нақты бекітілген."
            },
            {
                "claim": f"{article_title} оқиғасы мәтінде көрсетілген мерзімнен он жыл бұрын басталған болатын.",
                "label": "REFUTES",
                "evidence_sentence": f"{article_title} мәтініндегі нақты дата көрсетілген мерзім болып табылады."
            },
            {
                "claim": f"{article_title} нысанының жалпы аумағы туралы мәлімет елу пайызға артық деп есептелді.",
                "label": "NOT_ENOUGH_INFO",
                "evidence_sentence": ""
            }
        ], ensure_ascii=False)
    else:
        return json.dumps([
            {
                "claim": f"{article_title} бойынша пайда болған салдарлар мәтінде жазылған себептермен тікелей байланысты.",
                "label": "SUPPORTS",
                "evidence_sentence": f"{article_title} себептері мен нәтижелері мәтінде толық жазылған."
            },
            {
                "claim": f"{article_title} дамуының басты себебі ретінде басқа мүлдем қайшы құбылыс көрсетіледі.",
                "label": "REFUTES",
                "evidence_sentence": f"{article_title} нәтижесі мәтіндегі нақты себепке негізделген."
            },
            {
                "claim": f"{article_title} саласындағы жаңа құқықтық ережелер арнайы комиссия шешімімен енгізілген болатын.",
                "label": "NOT_ENOUGH_INFO",
                "evidence_sentence": ""
            }
        ], ensure_ascii=False)


def mock_llm_query_social(persona: str, platform: str) -> str:
    """
    Generates a deterministic valid mock colloquial Kazakh post for dry-run testing.
    """
    return (
        f"Мына {platform} желісіндегі жаңалық өте қызық болды достар! "
        f"Өзім {persona} ретінде айтсам бұл тақырып шынымен маңызды сияқты ғой негізі, "
        f"ертең міндетті түрде бәрін көріп шығайық."
    )


class LLMRunner:
    """
    Manages LLM initialization and inference on Kaggle Dual Tesla T4 GPUs or CPU mock fallback.
    """
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.tokenizer = None
        self.model = None

    def initialize_model(self) -> None:
        if self.dry_run:
            print("Dry-run mode active: skipping GPU model loading.")
            return

        print("\nLoading Qwen/Qwen2.5-7B-Instruct in 4-bit precision on GPU...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model successfully loaded on GPU!")

    def query(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        mock_type: str = "fever",
        aspect_idx: int = 0,
        title: str = "",
        persona: str = "",
        platform: str = "",
    ) -> str:
        if self.dry_run or self.model is None:
            if mock_type == "fever":
                return mock_llm_query_fever(aspect_idx, title)
            else:
                return mock_llm_query_social(persona, platform)

        import torch
        messages = [
            {
                "role": "system",
                "content": "You are a professional computational linguist and Kazakh native speaker specializing in academic NLP datasets.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()


_DEFAULT_RUNNER: Optional[LLMRunner] = None


def query_llm(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Legacy convenience helper for backwards compatibility.
    """
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = LLMRunner(dry_run=True)
    return _DEFAULT_RUNNER.query(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


def run_fever_generation(
    runner: LLMRunner,
    articles: List[Dict[str, Any]],
    output_path: str = "kazakh_fever_3k_generated.jsonl",
    max_queries: int = 1000,
    target_claims: int = 3000,
) -> List[Dict[str, Any]]:
    """
    Executes the aspect-diverse Kazakh-FEVER generation loop with validation and disk flushing.
    """
    print(f"\n[FEVER] Starting Aspect-Diverse Generation Loop (Target: {target_claims} claims, up to {max_queries} queries)...")
    if not articles:
        print("[FEVER] No articles provided. Skipping generation.")
        return []

    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fever_records: List[Dict[str, Any]] = []
    claim_id_counter = 1
    num_articles = len(articles)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for q in range(max_queries):
            if len(fever_records) >= target_claims:
                print(f"[FEVER] Reached target {target_claims} claims at query {q}.")
                break

            art_idx = q % num_articles
            art = articles[art_idx]
            title = art.get("title", f"Тақырып {art_idx + 1}")
            text = art.get("text", "")
            if len(text.strip()) < 30:
                continue

            aspect_idx = (q // num_articles + (q % 3)) % 3
            prompt = generate_aspect_prompt(title, text, aspect_idx)

            try:
                resp = runner.query(
                    prompt,
                    max_new_tokens=450,
                    temperature=0.6,
                    mock_type="fever",
                    aspect_idx=aspect_idx,
                    title=title,
                )
                json_match = re.search(r"\[.*\]", resp, re.DOTALL)
                if json_match:
                    claims_data = json.loads(json_match.group(0))
                    for item in claims_data:
                        claim_text = item.get("claim", "").strip()
                        label = item.get("label", "").strip().upper()
                        ev = item.get("evidence_sentence", "").strip()
                        tokens = claim_text.split()

                        if not (8 <= len(tokens) <= 35):
                            continue
                        if label not in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}:
                            continue

                        # Validation: non-empty evidence for SUPPORTS/REFUTES, empty for Hard NEI
                        if label in {"SUPPORTS", "REFUTES"}:
                            if not ev:
                                continue
                            evidence_list = [ev]
                        else:
                            evidence_list = []

                        record = {
                            "id": f"kz_fever_{claim_id_counter:04d}",
                            "article_title": title,
                            "claim": claim_text,
                            "label": label,
                            "evidence_sentences": evidence_list,
                            "domain": art.get("domain", "wikipedia"),
                        }
                        fever_records.append(record)
                        claim_id_counter += 1

                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if len(fever_records) % 10 == 0:
                            f_out.flush()

                        if len(fever_records) >= target_claims:
                            break

            except Exception as e:
                print(f"Error generating claims for query {q} ('{title}'): {e}")

            if (q + 1) % 50 == 0 or (q + 1) == max_queries:
                print(f"[FEVER] Processed {q + 1}/{max_queries} queries -> {len(fever_records)} validated claims.")

        f_out.flush()

    print(f"[Done] Saved {len(fever_records)} Kazakh-FEVER claims to {output_path}.")
    return fever_records


def run_social_generation(
    runner: LLMRunner,
    output_path: str = "kazakh_social_media_ai_1k.jsonl",
    target_posts: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Executes the scaled colloquial Kazakh social media post generation loop.
    """
    print(f"\n[Social] Starting Colloquial Kazakh Benchmark Generation (Target: {target_posts} posts)...")

    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    social_records: List[Dict[str, Any]] = []
    soc_id_counter = 1

    max_attempts = int(target_posts * 1.5)
    attempt = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        while len(social_records) < target_posts and attempt < max_attempts:
            persona = PERSONAS[attempt % len(PERSONAS)]
            platform = PLATFORMS[(attempt // len(PERSONAS)) % len(PLATFORMS)]
            prompt = generate_social_prompt(persona, platform)
            attempt += 1

            try:
                post_text = runner.query(
                    prompt,
                    max_new_tokens=150,
                    temperature=0.85,
                    mock_type="social",
                    persona=persona,
                    platform=platform,
                )
                clean_text = post_text.strip().replace('"', '').replace('\n', ' ')
                tokens = clean_text.split()
                if 10 <= len(tokens) <= 100:
                    rec = {
                        "id": f"soc_ai_{soc_id_counter:04d}",
                        "text": clean_text,
                        "label": "ai",
                        "platform": platform,
                        "persona": persona,
                    }
                    social_records.append(rec)
                    soc_id_counter += 1

                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if len(social_records) % 25 == 0:
                        f_out.flush()

            except Exception as e:
                print(f"Error generating social post (attempt {attempt}): {e}")

            if len(social_records) % 50 == 0 and len(social_records) > 0:
                print(f"[Social] Generated {len(social_records)}/{target_posts} social media posts.")

        f_out.flush()

    print(f"[Done] Saved {len(social_records)} Social Media AI posts to {output_path}.")
    return social_records


def load_or_unpack_corpus(
    embedded_payload: str = EMBEDDED_KNOWLEDGE_CORPUS_B64,
    target_path: str = "kazakh_knowledge_corpus.jsonl",
) -> List[Dict[str, Any]]:
    """
    Resolves the Kazakh knowledge corpus by checking the embedded payload,
    local paths, Kaggle input mounts, or curated fallback articles.
    """
    print("\nPreparing Reference Knowledge Articles...")

    # 1. Try unpacking embedded payload first
    if embedded_payload.strip():
        articles = unpack_embedded_corpus(embedded_payload, target_path)
        if articles:
            print(f"Unpacked and loaded {len(articles)} articles from embedded payload into {target_path}.")
            return articles

    # 2. Candidate paths
    candidate_paths = [
        target_path,
        "kazakh_knowledge_corpus.jsonl",
        "kaggle_runner/kazakh_knowledge_corpus.jsonl",
        os.path.join(os.path.dirname(__file__), "kazakh_knowledge_corpus.jsonl") if "__file__" in globals() else None,
        "/kaggle/working/kazakh_knowledge_corpus.jsonl",
        "/kaggle/input/kazakh-knowledge-corpus/kazakh_knowledge_corpus.jsonl",
    ]
    for cp in candidate_paths:
        if cp and os.path.exists(cp):
            try:
                articles = []
                with open(cp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            articles.append(json.loads(line))
                if articles:
                    print(f"Loaded {len(articles)} articles from {cp}.")
                    return articles
            except Exception as e:
                print(f"Warning: Failed loading from {cp}: {e}")

    # 3. Fallback generator or seed articles
    try:
        from scripts.expand_knowledge_corpus import build_curated_knowledge_corpus
        articles = build_curated_knowledge_corpus(target_path)
        print(f"Generated and loaded {len(articles)} articles using expand_knowledge_corpus.")
        return articles
    except Exception as e:
        print(f"Warning: Seed corpus not found ({e}). Using built-in high-value Kazakh encyclopedic topics.")
        return [
            {"passage_id": "wiki_kz_001", "domain": "history", "title": "Қазақстан тәуелсіздігі", "text": "Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады. Тәуелсіздік туралы Конституциялық заң Жоғарғы Кеңес тарапынан қабылданды."},
            {"passage_id": "wiki_kz_002", "domain": "geography", "title": "Астана қаласы", "text": "Астана қаласы — Қазақстанның елордасы. 1997 жылы елорда Алматы қаласынан Ақмолаға көшіріліп, 1998 жылы қала атауы Астана болып өзгертілді."},
            {"passage_id": "wiki_kz_003", "domain": "literature", "title": "Абай Құнанбайұлы", "text": "Абай (Ибраһим) Құнанбайұлы — 1845 жылы Семей өңірінде дүниеге келген ұлы қазақ ақыны, ойшылы және ағартушысы. Оның атақты туындыларының бірі — «Қара сөздері»."},
            {"passage_id": "wiki_kz_004", "domain": "history", "title": "Қазақ хандығы", "text": "Қазақ хандығы 1465 жылы Керей мен Жәнібек хандардың бастауымен құрылды. Хандықтың негізі Жетісу және Шу өңірінде қаланды."},
            {"passage_id": "wiki_kz_005", "domain": "science", "title": "Байқоңыр ғарыш айлағы", "text": "Байқоңыр — әлемдегі тұңғыш әрі ең ірі ғарыш айлағы. Оның құрылысы 1955 жылы Қызылорда облысында басталды. 1961 жылы Юрий Гагарин осы жерден ғарышқа ұшты."}
        ]


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for Kaggle GPU execution and local dry runs.
    """
    parser = argparse.ArgumentParser(
        description="Kaggle GPU Kernel: Kazakh-FEVER 3K & Social Media AI Dataset Generator"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run fast 2-step mock generation without GPU")
    parser.add_argument("--fever-queries", type=int, default=1000, help="Max FEVER LLM queries")
    parser.add_argument("--fever-target", type=int, default=3000, help="Target FEVER claims")
    parser.add_argument("--social-target", type=int, default=1000, help="Target social posts")
    parser.add_argument("--output-fever", default="kazakh_fever_3k_generated.jsonl", help="Output FEVER JSONL")
    parser.add_argument("--output-social", default="kazakh_social_media_ai_1k.jsonl", help="Output Social JSONL")
    parser.add_argument("--corpus-target", default="kazakh_knowledge_corpus.jsonl", help="Corpus unpack destination")
    args = parser.parse_args(argv)

    print("=" * 70)
    print("KAZAKH AI RESEARCH: KAGGLE GPU DATASET GENERATION PIPELINE")
    print("Environment: Python", sys.version)
    print("=" * 70)

    # Detect CUDA
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    is_dry_run = args.dry_run or not has_cuda
    if is_dry_run:
        print("Notice: Running in dry-run/CPU mode.")
        if not args.dry_run and not has_cuda:
            print("Notice: CUDA is not available. Falling back to dry-run mock mode.")
        if args.dry_run or not has_cuda:
            if args.fever_queries == 1000:
                args.fever_queries = 2
            if args.fever_target == 3000:
                args.fever_target = 6
            if args.social_target == 1000:
                args.social_target = 2

    # Load / Unpack corpus
    articles = load_or_unpack_corpus(EMBEDDED_KNOWLEDGE_CORPUS_B64, args.corpus_target)

    # Initialize runner
    runner = LLMRunner(dry_run=is_dry_run)
    runner.initialize_model()

    # Run generations
    fever_records = run_fever_generation(
        runner=runner,
        articles=articles,
        output_path=args.output_fever,
        max_queries=args.fever_queries,
        target_claims=args.fever_target,
    )

    social_records = run_social_generation(
        runner=runner,
        output_path=args.output_social,
        target_posts=args.social_target,
    )

    print("\n" + "=" * 70)
    print("KAGGLE GPU PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("Output files ready:")
    print(f"1. {args.output_fever} ({len(fever_records)} records)")
    print(f"2. {args.output_social} ({len(social_records)} records)")
    print("=" * 70)


if __name__ == "__main__":
    main()
