# -*- coding: utf-8 -*-
"""
Kaggle GPU Kernel: Multi-Domain Generalization & Kazakh-FEVER 3K Dataset Generator.
Runs on Kaggle Dual Tesla T4 GPUs using Qwen-2.5-7B-Instruct (4-bit quantized).
Generates:
1. Kazakh-FEVER 3K: Balanced claim-evidence triples (SUPPORTS, REFUTES, Hard NEI).
2. Social Media Benchmark: Colloquial Kazakh synthetic text with slang and code-switching.
"""

import os
import sys
import json
import time
import re
import random
from typing import List, Dict, Any

print("=" * 70)
print("KAZAKH AI RESEARCH: KAGGLE GPU DATASET GENERATION PIPELINE")
print("Environment: Python", sys.version)
print("=" * 70)

# Step 1: Install required packages silently if missing
try:
    import torch
    import transformers
    import bitsandbytes
except ImportError:
    print("[1/5] Installing transformers, accelerate, bitsandbytes...")
    os.system("pip install -q transformers accelerate bitsandbytes sentence-transformers wikipedia-api")
    import torch
    import transformers

print(f"[2/5] PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"      Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"      Device {i}: {torch.cuda.get_device_name(i)}")

# Step 2: Load Qwen-2.5-7B-Instruct in 4-bit
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
print(f"\n[3/5] Loading {MODEL_ID} in 4-bit precision on GPU...")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print("Model successfully loaded on GPU!")


def query_llm(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": "You are a professional computational linguist and Kazakh native speaker specializing in academic NLP datasets."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


# Step 3: Load or expand knowledge corpus
print("\n[4/5] Preparing Reference Knowledge Articles...")
candidate_paths = [
    "kazakh_knowledge_corpus.jsonl",
    "kaggle_runner/kazakh_knowledge_corpus.jsonl",
    os.path.join(os.path.dirname(__file__), "kazakh_knowledge_corpus.jsonl") if "__file__" in locals() else None,
    "/kaggle/working/kazakh_knowledge_corpus.jsonl",
    "/kaggle/input/kazakh-knowledge-corpus/kazakh_knowledge_corpus.jsonl"
]
corpus_path = None
for cp in candidate_paths:
    if cp and os.path.exists(cp):
        corpus_path = cp
        break

articles = []
if corpus_path and os.path.exists(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    print(f"Loaded {len(articles)} seed articles from {corpus_path}.")
else:
    try:
        from scripts.expand_knowledge_corpus import build_curated_knowledge_corpus
        corpus_path = "kazakh_knowledge_corpus.jsonl"
        articles = build_curated_knowledge_corpus(corpus_path)
        print(f"Generated and loaded {len(articles)} articles using expand_knowledge_corpus.")
    except Exception as e:
        print(f"Warning: Seed corpus not found ({e}). Using built-in high-value Kazakh encyclopedic topics.")
        articles = [
            {"passage_id": "wiki_kz_001", "domain": "history", "title": "Қазақстан тәуелсіздігі", "text": "Қазақстан Республикасы 1991 жылы 16 желтоқсанда өз тәуелсіздігін ресми түрде жариялады. Тәуелсіздік туралы Конституциялық заң Жоғарғы Кеңес тарапынан қабылданды."},
            {"passage_id": "wiki_kz_002", "domain": "geography", "title": "Астана қаласы", "text": "Астана қаласы — Қазақстанның елордасы. 1997 жылы елорда Алматы қаласынан Ақмолаға көшіріліп, 1998 жылы қала атауы Астана болып өзгертілді."},
            {"passage_id": "wiki_kz_003", "domain": "literature", "title": "Абай Құнанбайұлы", "text": "Абай (Ибраһим) Құнанбайұлы — 1845 жылы Семей өңірінде дүниеге келген ұлы қазақ ақыны, ойшылы және ағартушысы. Оның атақты туындыларының бірі — «Қара сөздері»."},
            {"passage_id": "wiki_kz_004", "domain": "history", "title": "Қазақ хандығы", "text": "Қазақ хандығы 1465 жылы Керей мен Жәнібек хандардың бастауымен құрылды. Хандықтың негізі Жетісу және Шу өңірінде қаланды."},
            {"passage_id": "wiki_kz_005", "domain": "science", "title": "Байқоңыр ғарыш айлағы", "text": "Байқоңыр — әлемдегі тұңғыш әрі ең ірі ғарыш айлағы. Оның құрылысы 1955 жылы Қызылорда облысында басталды. 1961 жылы Юрий Гагарин осы жерден ғарышқа ұшты."}
        ]


# Step 4: Generate Kazakh-FEVER Claims
print("\n[5/5] Generating Kazakh-FEVER 3K Candidate Claims with Hard NEI...")
output_fever_path = "kazakh_fever_3k_generated.jsonl"
output_social_path = "kazakh_social_media_ai_1k.jsonl"

fever_records = []
claim_id_counter = 1

for idx, art in enumerate(articles):
    title = art.get("title", f"Тақырып {idx+1}")
    text = art.get("text", "")
    if len(text.strip()) < 30:
        continue

    prompt = f"""Сен қазақ тіліндегі академиялық фактчекинг (Kazakh-FEVER) деректер жинағын құрастыратын мамансың.
Төменде берілген мәтінге сүйене отырып, 3 түрлі тұжырым (claim) жаса:
1. SUPPORTS: Мәтінге толық сәйкес келетін, дәлелденетін тұжырым.
2. REFUTES: Мәтіндегі ақпаратты (жыл, сан, тұлға, оқиға) бұрмалап жоққа шығаратын тұжырым.
3. NOT_ENOUGH_INFO (Hard NEI): Мәтіндегі кейіпкер немесе тақырып туралы, бірақ мәтінде АЙТЫЛМАҒАН, ойдан шығарылған шынайы көрінетін тың дерек.

Мәтін тақырыбы: {title}
Мәтін мазмұны: {text}

Нәтижені ТЕК келесі JSON форматында шығар:
[
  {{"claim": "қазақша тұжырым 1", "label": "SUPPORTS", "evidence_sentence": "мәтіндегі нақты дәлел сөйлем"}},
  {{"claim": "қазақша тұжырым 2", "label": "REFUTES", "evidence_sentence": "мәтіндегі теріске шығарылатын сөйлем"}},
  {{"claim": "қазақша тұжырым 3", "label": "NOT_ENOUGH_INFO", "evidence_sentence": ""}}
]"""

    try:
        resp = query_llm(prompt, max_new_tokens=400, temperature=0.6)
        json_match = re.search(r"\[.*\]", resp, re.DOTALL)
        if json_match:
            claims_data = json.loads(json_match.group(0))
            for item in claims_data:
                claim_text = item.get("claim", "").strip()
                label = item.get("label", "").upper()
                ev = item.get("evidence_sentence", "").strip()
                tokens = claim_text.split()

                if 8 <= len(tokens) <= 35 and label in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}:
                    record = {
                        "id": f"kz_fever_{claim_id_counter:04d}",
                        "article_title": title,
                        "claim": claim_text,
                        "label": label,
                        "evidence_sentences": [ev] if ev else [],
                        "domain": art.get("domain", "wikipedia")
                    }
                    fever_records.append(record)
                    claim_id_counter += 1
    except Exception as e:
        print(f"Error generating claims for '{title}': {e}")

    if (idx + 1) % 5 == 0 or (idx + 1) == len(articles):
        print(f"Processed {idx+1}/{len(articles)} articles -> {len(fever_records)} claims generated.")

with open(output_fever_path, "w", encoding="utf-8") as f:
    for rec in fever_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"\n[Done] Saved {len(fever_records)} Kazakh-FEVER claims to {output_fever_path}.")


# Step 5: Generate Social Media Synthetic Benchmark
print("\nGenerating Social Media Colloquial Kazakh Benchmark...")
personas = ["student", "consumer", "tech_enthusiast", "casual_chat"]
platforms = ["telegram", "twitter", "forum"]

social_records = []
soc_id_counter = 1

for i in range(100):  # Expandable on Kaggle
    persona = random.choice(personas)
    platform = random.choice(platforms)

    prompt = f"""Сен қазақша әлеуметтік желілерде ({platform}) белсенді жазатын қолданушысың ({persona}).
Бейресми ауызекі қазақ тілінде (жастар сленгі, '-сың ғой', '-ма екен', '-шы' қосымшалары, күнделікті диалог, ағылшын/орыс сөздерімен араласқан code-switching) 1 қысқа пікір немесе жазба жаз.
Ұзындығы 15-50 сөз болсын. Тек пікірдің өзін қазақша жаз, басқа түсіндірме жазба."""

    try:
        post_text = query_llm(prompt, max_new_tokens=150, temperature=0.85)
        clean_text = post_text.strip().replace('"', '').replace('\n', ' ')
        tokens = clean_text.split()
        if 10 <= len(tokens) <= 100:
            rec = {
                "id": f"soc_ai_{soc_id_counter:04d}",
                "text": clean_text,
                "label": "ai",
                "platform": platform,
                "persona": persona
            }
            social_records.append(rec)
            soc_id_counter += 1
    except Exception as e:
        pass

    if (i + 1) % 25 == 0:
        print(f"Generated {len(social_records)} social media posts.")

with open(output_social_path, "w", encoding="utf-8") as f:
    for rec in social_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[Done] Saved {len(social_records)} Social Media AI posts to {output_social_path}.")

print("\n" + "=" * 70)
print("KAGGLE GPU PIPELINE RUN COMPLETED SUCCESSFULLY!")
print(f"Output files ready in /kaggle/working/:")
print(f"1. {output_fever_path} ({len(fever_records)} records)")
print(f"2. {output_social_path} ({len(social_records)} records)")
print("=" * 70)
