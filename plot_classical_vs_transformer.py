import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 11, 'figure.dpi': 300, 'font.family': 'sans-serif'})

# Detailed Benchmark Data across Classical ML vs Deep Transformers
models = [
    'TF-IDF +\nNaive Bayes',
    'TF-IDF +\nLogistic Reg',
    'TF-IDF +\nSVM',
    'mBERT\n(Pure)',
    'XLM-RoBERTa\n(Pure)',
    'KazRoBERTa\n(Pure)',
    'KazRoBERTa +\nHybrid FST (Ours)'
]

acc_scores = [52.38, 52.38, 38.10, 95.42, 95.45, 96.10, 96.32]
f1_scores = [61.54, 61.54, 31.58, 95.46, 95.52, 96.09, 96.32]
fpr_scores = [47.62, 47.62, 61.90, 11.60, 13.40, 10.60, 6.00]

colors = ['#d9534f', '#d9534f', '#d9534f', '#f0ad4e', '#f0ad4e', '#0275d8', '#5cb85c']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Accuracy & F1-Score Comparison
x = np.arange(len(models))
width = 0.35

rects1 = ax1.bar(x - width/2, acc_scores, width, label='Accuracy (%)', color='#2C3E50')
rects2 = ax1.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#16A085')

ax1.set_ylabel('Percentage (%)', fontweight='bold')
ax1.set_title('Classification Performance (Accuracy & F1)\nClassical ML vs Deep Transformers', pad=15, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=25, ha='right', fontweight='bold', fontsize=9)
ax1.set_ylim(0, 115)
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

for i in range(len(models)):
    ax1.annotate(f"{acc_scores[i]:.1f}%", (x[i] - width/2, acc_scores[i] + 1.5), ha='center', fontsize=8, fontweight='bold')
    ax1.annotate(f"{f1_scores[i]:.1f}%", (x[i] + width/2, f1_scores[i] + 1.5), ha='center', fontsize=8, fontweight='bold', color='#0e6655')

# Subplot 2: False Positive Rate (FPR %) Comparison
bars = ax2.bar(models, fpr_scores, color=colors, width=0.55)
ax2.set_ylabel('False Positive Rate (FPR %)', fontweight='bold')
ax2.set_title('False Positive Rate on Authentic Kazakh Text\nLower is Better (Hybrid FST Cuts Accusations)', pad=15, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=25, ha='right', fontweight='bold', fontsize=9)
ax2.set_ylim(0, 75)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

for bar, fpr in zip(bars, fpr_scores):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{fpr:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
os.makedirs("data", exist_ok=True)
out_img = os.path.join("data", "chart_classical_vs_transformer_baselines.png")
plt.savefig(out_img, bbox_inches='tight', dpi=300)
print(f"Successfully generated comparison diagram at: {out_img}")

# Detailed JSON report for Professor
detailed_data = {
    "classical_baselines": {
        "TF-IDF + Naive Bayes": {"accuracy": 52.38, "f1_score": 61.54, "fpr": 47.62},
        "TF-IDF + Logistic Regression": {"accuracy": 52.38, "f1_score": 61.54, "fpr": 47.62},
        "TF-IDF + Support Vector Machine (SVM)": {"accuracy": 38.10, "f1_score": 31.58, "fpr": 61.90}
    },
    "transformer_baselines": {
        "mBERT (Pure)": {"accuracy": 95.42, "f1_score": 95.46, "fpr": 11.60},
        "XLM-RoBERTa (Pure)": {"accuracy": 95.45, "f1_score": 95.52, "fpr": 13.40},
        "KazRoBERTa (Pure)": {"accuracy": 96.10, "f1_score": 96.09, "fpr": 10.60}
    },
    "proposed_method": {
        "KazRoBERTa + Hybrid FST (Ours)": {"accuracy": 96.32, "f1_score": 96.32, "fpr": 6.00}
    },
    "statistical_superiority": {
        "transformer_vs_classical_gain": "+43.94% Accuracy Gain over Classical ML (p < 0.0001)",
        "fst_false_positive_reduction": "-43.40% FPR Reduction over Pure KazRoBERTa (p = 0.000004)"
    }
}

out_json = os.path.join("data", "detailed_model_benchmark_comparison.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(detailed_data, f, ensure_ascii=False, indent=2)
print(f"Saved detailed benchmark JSON at: {out_json}")
