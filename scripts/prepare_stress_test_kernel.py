import os
import sys
import json
import zlib
import base64

def generate_kernel():
    print("Reading 2,000 paired benchmark dataset...")
    data_path = "data/kazakh_aigc_paired_2k.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Compress dataset
    raw_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    comp_bytes = zlib.compress(raw_bytes, level=9)
    b64_str = base64.b64encode(comp_bytes).decode("ascii")
    print(f"Compressed dataset: {len(raw_bytes)} bytes -> {len(comp_bytes)} bytes ({len(b64_str)} b64 chars)")

    kernel_code = f'''import os
import sys
import json
import zlib
import base64
import time
import math
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

print("=" * 75)
print("KAZAKH AI-TEXT DETECTION: ZERO-SHOT STRESS-TEST (GPU T4 x 2)")
print("Benchmarking KazRoBERTa (Pure vs. FST) on Qwen-2.5-7B vs. Human KazSAnDRA")
print("=" * 75)

import torch
print(f"PyTorch: {{torch.__version__}}")
print(f"CUDA Available: {{torch.cuda.is_available()}}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU Count: {{torch.cuda.device_count()}}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {{i}}: {{torch.cuda.get_device_name(i)}} ({{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}} GB)")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# -----------------------------------------------------------------------------
# 1. Embedded Hybrid FST Morphological Analyzer (Turkic Nominal + Verbal + Loanwords)
# -----------------------------------------------------------------------------
COMMON_LOANWORD_ROOTS = [
    'доставка', 'оплата', 'возврат', 'заказ', 'бонус', 'акция', 'клиент', 'карта',
    'банк', 'номер', 'приложение', 'аккаунт', 'профиль', 'страница', 'пароль', 'код',
    'каспи', 'инстаграм', 'ватсап', 'телеграм', 'сет', 'сеть', 'чек', 'счет',
    'отзыв', 'товар', 'заявка', 'гранд', 'грант', 'кабинет', 'сумма', 'меню', 'плюс',
    'видео', 'фото', 'файл', 'арна', 'оператор', 'комиссия', 'процент', 'качество',
    'сервис', 'курьер', 'бронь', 'скидка'
]

class AdvancedKazakhFSTAnalyzer:
    def __init__(self):
        self.loanword_roots = COMMON_LOANWORD_ROOTS
        self.cases = [
            'нің', 'ның', 'дың', 'дің', 'тың', 'тің',
            'ға', 'ге', 'қа', 'ке', 'на', 'не',
            'ны', 'ні', 'ды', 'ді', 'ты', 'ті',
            'да', 'де', 'та', 'те', 'нда', 'нде',
            'дан', 'ден', 'тан', 'тен', 'нан', 'нен',
            'мен', 'бен', 'пен',
        ]
        self.possessives = [
            'ларымыз', 'леріміз', 'дарымыз', 'деріміз', 'тарымыз', 'теріміз',
            'ларыңыз', 'леріңіз', 'дарыңыз', 'деріңіз', 'тарыңыз', 'теріңіз',
            'мыз', 'міз', 'ңыз', 'ңіз', 'ымыз', 'іміз', 'ыңыз', 'іңіз',
            'лары', 'лері', 'дары', 'дері', 'тары', 'тері',
            'сын', 'сін', 'мын', 'мін',
            'м', 'ң', 'ы', 'і', 'сы', 'сі',
        ]
        self.plurals = ['лар', 'лер', 'дар', 'дер', 'тар', 'тер']
        self.verbal_tenses = [
            'ғандықтан', 'гендіктен', 'қандықтан', 'кендіктен',
            'атын', 'етін', 'йтын', 'йтін',
            'ған', 'ген', 'қан', 'кен',
            'ады', 'еді', 'йды', 'йді',
            'мақ', 'мек', 'бақ', 'бек', 'пақ', 'пек',
        ]
        self.verbal_persons = [
            'мын', 'мін', 'бын', 'бін', 'пын', 'пін',
            'мыз', 'міз', 'быз', 'біз', 'пыз', 'піз',
            'сың', 'сің', 'сыздар', 'сіздер'
        ]
        self.case_re = re.compile(r'(' + '|'.join(self.cases) + r')$')
        self.poss_re = re.compile(r'(' + '|'.join(self.possessives) + r')$')
        self.plur_re = re.compile(r'(' + '|'.join(self.plurals) + r')$')
        self.verb_tense_re = re.compile(r'(' + '|'.join(self.verbal_tenses) + r')$')
        self.verb_person_re = re.compile(r'(' + '|'.join(self.verbal_persons) + r')$')

    def _segment_loanword(self, word):
        w_lower = word.lower()
        for root in self.loanword_roots:
            if w_lower.startswith(root) and len(w_lower) > len(root):
                sfx_fallback = word[len(root):]
                if len(sfx_fallback) >= 1:
                    return word[:len(root)] + f" -{{sfx_fallback}}"
        return None

    def analyze_and_segment(self, text):
        if not isinstance(text, str):
            return text
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) <= 3:
                processed_words.append(word)
                continue
            loanword_seg = self._segment_loanword(word)
            if loanword_seg:
                processed_words.append(loanword_seg)
                continue
            original = word
            suffixes = []
            m_vper = self.verb_person_re.search(word)
            if m_vper and len(word[:m_vper.start()]) >= 3:
                suffixes.insert(0, m_vper.group(1))
                word = word[:m_vper.start()]
            m_vt = self.verb_tense_re.search(word)
            if m_vt and len(word[:m_vt.start()]) >= 3:
                suffixes.insert(0, m_vt.group(1))
                word = word[:m_vt.start()]
            match = self.case_re.search(word)
            if match and len(word[:match.start()]) >= 3:
                suffixes.insert(0, match.group(1))
                word = word[:match.start()]
            if len(word) > 3:
                match = self.poss_re.search(word)
                if match and len(word[:match.start()]) >= 3:
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]
            if len(word) > 3:
                match = self.plur_re.search(word)
                if match and len(word[:match.start()]) >= 3:
                    suffixes.insert(0, match.group(1))
                    word = word[:match.start()]
            if suffixes:
                processed_words.append(word + " " + " ".join(f"-{{s}}" for s in suffixes))
            else:
                processed_words.append(original)
        return " ".join(processed_words)

fst_analyzer = AdvancedKazakhFSTAnalyzer()

# -----------------------------------------------------------------------------
# 2. Load 2,000 Paired Benchmark Dataset
# -----------------------------------------------------------------------------
DATA_B64 = "{b64_str}"
raw_json = zlib.decompress(base64.b64decode(DATA_B64.encode('ascii'))).decode('utf-8')
benchmark_data = json.loads(raw_json)
print(f"Successfully loaded {{len(benchmark_data)}} paired benchmark test samples.")

df_test = pd.DataFrame(benchmark_data)
df_test["text_fst"] = df_test["text"].apply(fst_analyzer.analyze_and_segment)

print("Sample Benchmark Texts:")
print(f"  Human Raw: {{df_test[df_test['label']==0]['text'].iloc[0][:80]}}")
print(f"  Human FST: {{df_test[df_test['label']==0]['text_fst'].iloc[0][:80]}}")
print(f"  Qwen  Raw: {{df_test[df_test['label']==1]['text'].iloc[0][:80]}}")
print(f"  Qwen  FST: {{df_test[df_test['label']==1]['text_fst'].iloc[0][:80]}}")

# -----------------------------------------------------------------------------
# 3. GPU Batch Inference Function
# -----------------------------------------------------------------------------
def run_batch_inference(model, tokenizer, texts, batch_size=64):
    model = model.to(device)
    model.eval()
    preds, probs_ai = [], []
    text_list = list(texts)
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(batch_preds.tolist())
        probs_ai.extend(batch_probs.tolist())
    return np.array(preds), np.array(probs_ai)

# -----------------------------------------------------------------------------
# 4. Evaluate KazRoBERTa (Pure) Baseline
# -----------------------------------------------------------------------------
print("\\n" + "=" * 50)
print("EVALUATING KAZROBERTA (PURE)")
print("Loading fine-tuned checkpoint: nKa1i/kazroberta-kk-ai-detection...")
print("=" * 50)

pure_repo = "nKa1i/kazroberta-kk-ai-detection"
pure_tok = AutoTokenizer.from_pretrained(pure_repo)
pure_model = AutoModelForSequenceClassification.from_pretrained(pure_repo)

preds_pure, probs_pure = run_batch_inference(pure_model, pure_tok, df_test["text"])
preds_pure_fst_input, probs_pure_fst_input = run_batch_inference(pure_model, pure_tok, df_test["text_fst"])

# -----------------------------------------------------------------------------
# 5. Train & Evaluate KazRoBERTa (Hybrid FST)
# -----------------------------------------------------------------------------
print("\\n" + "=" * 50)
print("TRAINING KAZROBERTA (HYBRID FST)")
print("Pretraining base: kz-transformers/kaz-roberta-conversational")
print("Training split: nKa1i/kazakh-ai-detect (8,848 samples)")
print("=" * 50)

# Fetch training dataset
train_url = "https://huggingface.co/datasets/nKa1i/kazakh-ai-detect/raw/main/data/train.csv"
print(f"Downloading training data from {{train_url}}...")
df_train_raw = pd.read_csv(train_url)
print(f"Loaded {{len(df_train_raw)}} training samples.")

# Apply FST morphological segmentation to training text
print("Applying Hybrid FST analyzer to training split...")
df_train_raw["text"] = df_train_raw["text"].astype(str).apply(fst_analyzer.analyze_and_segment)

fst_base_name = "kz-transformers/kaz-roberta-conversational"
fst_tok = AutoTokenizer.from_pretrained(fst_base_name)

def tokenize_fst(batch):
    return fst_tok(batch["text"], padding="max_length", truncation=True, max_length=128)

split = Dataset.from_pandas(df_train_raw[["text", "label"]]).train_test_split(test_size=0.1, seed=42)
tok_train = split["train"].map(tokenize_fst, batched=True)
tok_val = split["test"].map(tokenize_fst, batched=True)

fst_model = AutoModelForSequenceClassification.from_pretrained(fst_base_name, num_labels=2)

output_fst_dir = "output/fst_native_KazRoBERTa"
os.makedirs(output_fst_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_fst_dir,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=42,
    logging_steps=50
)

def compute_hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {{
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0)
    }}

trainer = Trainer(
    model=fst_model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    compute_metrics=compute_hf_metrics
)

print("Starting FST model training (3 epochs)...")
train_start = time.time()
trainer.train()
print(f"FST training completed in {{time.time() - train_start:.1f}}s.")

trainer.save_model(output_fst_dir)
fst_tok.save_pretrained(output_fst_dir)

print("\\nEvaluating KazRoBERTa (Hybrid FST) on paired benchmark...")
preds_fst, probs_fst = run_batch_inference(fst_model, fst_tok, df_test["text_fst"])

# -----------------------------------------------------------------------------
# 6. Detailed Evaluation & Metrics Computation
# -----------------------------------------------------------------------------
df_test["pred_pure"] = preds_pure
df_test["prob_pure"] = probs_pure
df_test["pred_pure_fst_input"] = preds_pure_fst_input
df_test["prob_pure_fst_input"] = probs_pure_fst_input
df_test["pred_fst"] = preds_fst
df_test["prob_fst"] = probs_fst

# Save raw prediction dataframe
os.makedirs("output", exist_ok=True)
df_test.to_csv("output/zero_shot_predictions_2k.csv", index=False, encoding="utf-8")
print("Saved predictions to output/zero_shot_predictions_2k.csv")

y_true = df_test["label"].values

def compute_all_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    return {{
        "accuracy": round(acc * 100, 2),
        "f1": round(f1 * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": round(fpr * 100, 2),
        "fnr": round(fnr * 100, 2)
    }}

# Overall Metrics
metrics_pure = compute_all_metrics(y_true, preds_pure, probs_pure)
metrics_pure_fst_input = compute_all_metrics(y_true, preds_pure_fst_input, probs_pure_fst_input)
metrics_fst = compute_all_metrics(y_true, preds_fst, probs_fst)

# Length-Stratified Metrics (RAID protocol)
brackets = ["short", "medium", "long"]
stratified_metrics = {{}}

for b in brackets:
    mask = (df_test["length_bracket"] == b).values
    stratified_metrics[b] = {{
        "total_samples": int(mask.sum()),
        "pure": compute_all_metrics(y_true[mask], preds_pure[mask], probs_pure[mask]),
        "fst": compute_all_metrics(y_true[mask], preds_fst[mask], probs_fst[mask]),
        "pure_testtime_fst": compute_all_metrics(y_true[mask], preds_pure_fst_input[mask], probs_pure_fst_input[mask])
    }}

# McNemar Test on Short-Text False Positives
short_human_mask = ((df_test["length_bracket"] == "short") & (df_test["label"] == 0)).values
pure_short_fp = (preds_pure[short_human_mask] == 1)
fst_short_fp = (preds_fst[short_human_mask] == 1)

# Contingency table: [neither_fp, fst_only_fp], [pure_only_fp, both_fp]
b_cell = int(((~pure_short_fp) & fst_short_fp).sum()) # Pure correct, FST error
c_cell = int((pure_short_fp & (~fst_short_fp)).sum()) # Pure error, FST correct

chi2_val = (abs(b_cell - c_cell) - 1)**2 / max(1, b_cell + c_cell) if (b_cell + c_cell) > 0 else 0.0
from scipy.stats import chi2
p_val = chi2.sf(chi2_val, 1)

mcnemar_results = {{
    "short_human_count": int(short_human_mask.sum()),
    "pure_short_fp": int(pure_short_fp.sum()),
    "fst_short_fp": int(fst_short_fp.sum()),
    "pure_fp_rate": round(float(pure_short_fp.mean()) * 100, 2),
    "fst_fp_rate": round(float(fst_short_fp.mean()) * 100, 2),
    "fp_reduction_cases": int(pure_short_fp.sum() - fst_short_fp.sum()),
    "relative_fp_reduction_percent": round(float((pure_short_fp.sum() - fst_short_fp.sum()) / max(1, pure_short_fp.sum())) * 100, 2),
    "chi2_statistic": round(float(chi2_val), 4),
    "p_value": float(p_val),
    "is_statistically_significant": bool(p_val < 0.05)
}}

# Bootstrap 95% Confidence Intervals (1,000 resamples)
print("\\nComputing 1,000-iteration Bootstrap 95% Confidence Intervals...")
np.random.seed(42)
n_boot = 1000
boot_pure_acc, boot_fst_acc = [], []
boot_pure_f1, boot_fst_f1 = [], []
boot_pure_short_fp, boot_fst_short_fp = [], []
boot_fp_red = []

n_total = len(y_true)
indices = np.arange(n_total)

for _ in range(n_boot):
    idx = np.random.choice(indices, size=n_total, replace=True)
    y_b = y_true[idx]
    p_pure_b = preds_pure[idx]
    p_fst_b = preds_fst[idx]

    boot_pure_acc.append(accuracy_score(y_b, p_pure_b) * 100)
    boot_fst_acc.append(accuracy_score(y_b, p_fst_b) * 100)
    boot_pure_f1.append(f1_score(y_b, p_pure_b, zero_division=0) * 100)
    boot_fst_f1.append(f1_score(y_b, p_fst_b, zero_division=0) * 100)

    # Short FP count
    sh_mask_b = (df_test["length_bracket"].values[idx] == "short") & (y_b == 0)
    fp_p = (p_pure_b[sh_mask_b] == 1).sum()
    fp_f = (p_fst_b[sh_mask_b] == 1).sum()
    boot_pure_short_fp.append(fp_p)
    boot_fst_short_fp.append(fp_f)
    red = ((fp_p - fp_f) / max(1, fp_p)) * 100 if fp_p > 0 else 0.0
    boot_fp_red.append(red)

bootstrap_summary = {{
    "pure": {{
        "accuracy": {{
            "mean": round(float(np.mean(boot_pure_acc)), 2),
            "ci_lower": round(float(np.percentile(boot_pure_acc, 2.5)), 2),
            "ci_upper": round(float(np.percentile(boot_pure_acc, 97.5)), 2)
        }},
        "f1": {{
            "mean": round(float(np.mean(boot_pure_f1)), 2),
            "ci_lower": round(float(np.percentile(boot_pure_f1, 2.5)), 2),
            "ci_upper": round(float(np.percentile(boot_pure_f1, 97.5)), 2)
        }},
        "short_fp_count": {{
            "mean": round(float(np.mean(boot_pure_short_fp)), 2),
            "ci_lower": round(float(np.percentile(boot_pure_short_fp, 2.5)), 2),
            "ci_upper": round(float(np.percentile(boot_pure_short_fp, 97.5)), 2)
        }}
    }},
    "fst": {{
        "accuracy": {{
            "mean": round(float(np.mean(boot_fst_acc)), 2),
            "ci_lower": round(float(np.percentile(boot_fst_acc, 2.5)), 2),
            "ci_upper": round(float(np.percentile(boot_fst_acc, 97.5)), 2)
        }},
        "f1": {{
            "mean": round(float(np.mean(boot_fst_f1)), 2),
            "ci_lower": round(float(np.percentile(boot_fst_f1, 2.5)), 2),
            "ci_upper": round(float(np.percentile(boot_fst_f1, 97.5)), 2)
        }},
        "short_fp_count": {{
            "mean": round(float(np.mean(boot_fst_short_fp)), 2),
            "ci_lower": round(float(np.percentile(boot_fst_short_fp, 2.5)), 2),
            "ci_upper": round(float(np.percentile(boot_fst_short_fp, 97.5)), 2)
        }}
    }},
    "relative_fp_reduction_percent": {{
        "mean": round(float(np.mean(boot_fp_red)), 2),
        "ci_lower": round(float(np.percentile(boot_fp_red, 2.5)), 2),
        "ci_upper": round(float(np.percentile(boot_fp_red, 97.5)), 2)
    }}
}}

# Cross-Generator Transfer Delta (Paper 1 Sherkala vs. Qwen-2.5)
paper1_in_distribution = {{
    "pure_accuracy": 96.10,
    "pure_f1": 96.09,
    "pure_short_fpr": 10.6,
    "fst_accuracy": 96.32,
    "fst_f1": 96.32,
    "fst_short_fpr": 6.0
}}

transfer_delta = {{
    "pure_transfer_drop_accuracy": round(metrics_pure["accuracy"] - paper1_in_distribution["pure_accuracy"], 2),
    "pure_transfer_drop_f1": round(metrics_pure["f1"] - paper1_in_distribution["pure_f1"], 2),
    "fst_transfer_drop_accuracy": round(metrics_fst["accuracy"] - paper1_in_distribution["fst_accuracy"], 2),
    "fst_transfer_drop_f1": round(metrics_fst["f1"] - paper1_in_distribution["fst_f1"], 2)
}}

# Assemble full results summary
full_summary = {{
    "experiment": "Zero-Shot Cross-Generator Stress-Test (KazRoBERTa Pure vs. FST)",
    "dataset": "KazSAnDRA (1,000 Human) vs. Qwen-2.5-7B-Instruct (1,000 AI)",
    "total_samples": int(len(df_test)),
    "pure_model": metrics_pure,
    "pure_model_testtime_fst": metrics_pure_fst_input,
    "fst_trained_model": metrics_fst,
    "length_stratification": stratified_metrics,
    "mcnemar_short_text_fp": mcnemar_results,
    "bootstrap_ci_95": bootstrap_summary,
    "paper1_in_distribution_reference": paper1_in_distribution,
    "cross_generator_transfer_delta": transfer_delta
}}

with open("output/zero_shot_stress_test_results.json", "w", encoding="utf-8") as f:
    json.dump(full_summary, f, indent=2, ensure_ascii=False)
print("Saved complete summary to output/zero_shot_stress_test_results.json")

# -----------------------------------------------------------------------------
# 7. Generate Publication-Ready Markdown Report
# -----------------------------------------------------------------------------
report_lines = [
    "# Empirical Results: Zero-Shot Cross-Generator Detector Stress-Test",
    "",
    "**Target Architecture:** KazRoBERTa (`kz-transformers/kaz-roberta-conversational`)",
    "**Training Distribution:** KazAI-Detect (KazSAnDRA Reviews + Sherkala-7B LLM)",
    "**Out-of-Distribution Test Bed:** $N = 2,000$ paired samples (1,000 Human KazSAnDRA + 1,000 Qwen-2.5-7B-Instruct)",
    "",
    "## 1. Primary Zero-Shot Transfer Performance",
    "",
    "| Model Variant | Input Mode | Overall Acc (%) | F1-Score (%) | Precision (%) | Recall (%) | False Positive Rate (FPR %) | False Negative Rate (FNR %) |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    f"| **KazRoBERTa (Pure)** | Raw Text | {{metrics_pure['accuracy']}}% | {{metrics_pure['f1']}}% | {{metrics_pure['precision']}}% | {{metrics_pure['recall']}}% | {{metrics_pure['fpr']}}% | {{metrics_pure['fnr']}}% |",
    f"| **KazRoBERTa (Test-Time FST)** | FST Preprocessed | {{metrics_pure_fst_input['accuracy']}}% | {{metrics_pure_fst_input['f1']}}% | {{metrics_pure_fst_input['precision']}}% | {{metrics_pure_fst_input['recall']}}% | {{metrics_pure_fst_input['fpr']}}% | {{metrics_pure_fst_input['fnr']}}% |",
    f"| **KazRoBERTa (Hybrid FST)** | FST-Trained + FST | {{metrics_fst['accuracy']}}% | {{metrics_fst['f1']}}% | {{metrics_fst['precision']}}% | {{metrics_fst['recall']}}% | {{metrics_fst['fpr']}}% | {{metrics_fst['fnr']}}% |",
    "",
    "## 2. RAID-Compliant Length Bracket Breakdown",
    "",
    "| Length Bracket | Sub-sample Size | Pure Acc (%) | Pure FPR (%) | Hybrid FST Acc (%) | Hybrid FST FPR (%) | FPR Reduction (%) |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
]

for b in brackets:
    p_b = stratified_metrics[b]["pure"]
    f_b = stratified_metrics[b]["fst"]
    sh_red = round(((p_b["fpr"] - f_b["fpr"]) / max(0.01, p_b["fpr"])) * 100, 1)
    label_b = f"**{{b.capitalize()}}** ({{'≤ 60 chars' if b=='short' else ('61–85 chars' if b=='medium' else '> 85 chars')}})"
    report_lines.append(f"| {{label_b}} | {{stratified_metrics[b]['total_samples']}} | {{p_b['accuracy']}}% | {{p_b['fpr']}}% | {{f_b['accuracy']}}% | {{f_b['fpr']}}% | **{{sh_red}}%** |")

report_lines.extend([
    "",
    "## 3. Short-Text Hypothesis Testing & Rigorous Robustness Verification",
    "",
    f"- **Short-Text Human Samples:** {{mcnemar_results['short_human_count']}}",
    f"- **Pure KazRoBERTa False Positives:** {{mcnemar_results['pure_short_fp']}} ({{mcnemar_results['pure_fp_rate']}}%)",
    f"- **Hybrid FST False Positives:** {{mcnemar_results['fst_short_fp']}} ({{mcnemar_results['fst_fp_rate']}}%)",
    f"- **Absolute FP Elimination:** {{mcnemar_results['fp_reduction_cases']}} cases",
    f"- **Relative FP Reduction:** **{{mcnemar_results['relative_fp_reduction_percent']}}%** (95% CI: [{{bootstrap_summary['relative_fp_reduction_percent']['ci_lower']}}%, {{bootstrap_summary['relative_fp_reduction_percent']['ci_upper']}}%])",
    f"- **McNemar's Test Statistic:** $\\\\chi^2 = {{mcnemar_results['chi2_statistic']}}$, $p = {{mcnemar_results['p_value']:.6f}}$",
    f"- **Conclusion:** {{'Statistically significant false positive reduction confirmed (p < 0.05).' if mcnemar_results['is_statistically_significant'] else 'Not statistically significant at alpha=0.05.'}}",
    "",
    "## 4. In-Distribution (Sherkala-7B) vs. Out-of-Distribution (Qwen-2.5-7B) Transfer Degradation",
    "",
    "| Metric | In-Distribution (Sherkala-7B) | Out-of-Distribution (Qwen-2.5-7B) | Generalization Delta (Δ) |",
    "| :--- | :---: | :---: | :---: |",
    f"| **Pure Accuracy** | {{paper1_in_distribution['pure_accuracy']}}% | {{metrics_pure['accuracy']}}% | {{transfer_delta['pure_transfer_drop_accuracy']:+.2f}}% |",
    f"| **Pure F1-Score** | {{paper1_in_distribution['pure_f1']}}% | {{metrics_pure['f1']}}% | {{transfer_delta['pure_transfer_drop_f1']:+.2f}}% |",
    f"| **Hybrid FST Accuracy** | {{paper1_in_distribution['fst_accuracy']}}% | {{metrics_fst['accuracy']}}% | {{transfer_delta['fst_transfer_drop_accuracy']:+.2f}}% |",
    f"| **Hybrid FST F1-Score** | {{paper1_in_distribution['fst_f1']}}% | {{metrics_fst['f1']}}% | {{transfer_delta['fst_transfer_drop_f1']:+.2f}}% |",
    ""
])

report_md = "\\n".join(report_lines)
with open("output/zero_shot_stress_test_report.md", "w", encoding="utf-8") as f:
    f.write(report_md)
print("Saved markdown report to output/zero_shot_stress_test_report.md")

print("\\n" + "=" * 75)
print("ZERO-SHOT STRESS-TEST COMPLETED SUCCESSFULLY!")
print("=" * 75)
print(report_md)
'''

    output_path = "kaggle_runner/diagnostic_kernel.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(kernel_code)
    print(f"Generated {output_path} successfully ({os.path.getsize(output_path)} bytes).")

if __name__ == "__main__":
    generate_kernel()
