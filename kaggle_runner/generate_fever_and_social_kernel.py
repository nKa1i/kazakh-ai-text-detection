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

EMBEDDED_KNOWLEDGE_CORPUS_B64 = "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"

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

        import subprocess
        import sys

        print("Ensuring bitsandbytes>=0.46.1 and accelerate are installed...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1", "accelerate"],
                check=True,
            )
            for mod in list(sys.modules.keys()):
                if mod.startswith("bitsandbytes"):
                    del sys.modules[mod]
            print("Environment dependencies verified successfully.")
        except Exception as e:
            print(f"Notice during dependency verification: {e}")

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
