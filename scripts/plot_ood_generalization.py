import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 11, 'figure.dpi': 300, 'font.family': 'sans-serif'})

domains = ['Consumer Reviews\n(Informal)', 'Formal News\n(Informburo / Egemen)', 'Wikipedia\n(Academic)']
pure_fpr = [12.8, 4.2, 3.6]
hybrid_fpr = [7.2, 2.4, 1.8]

x = np.arange(len(domains))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))

rects1 = ax.bar(x - width/2, pure_fpr, width, label='KazRoBERTa (Pure)', color='#d9534f')
rects2 = ax.bar(x + width/2, hybrid_fpr, width, label='KazRoBERTa + Hybrid FST', color='#5cb85c')

ax.set_ylabel('False Positive Rate (FPR %)', fontweight='bold')
ax.set_title('Out-of-Distribution (OOD) Cross-Domain Generalization\nHybrid FST Consistently Reduces False Accusations Across Domains', pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(domains, fontweight='bold')
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
ax.set_ylim(0, 16)

# Annotations
for i in range(len(domains)):
    red_pct = ((pure_fpr[i] - hybrid_fpr[i]) / pure_fpr[i]) * 100
    ax.annotate(f"{pure_fpr[i]}%", (x[i] - width/2, pure_fpr[i] + 0.3), ha='center', fontsize=9, fontweight='bold')
    ax.annotate(f"{hybrid_fpr[i]}%\n(-{red_pct:.1f}%)", (x[i] + width/2, hybrid_fpr[i] + 0.3), ha='center', fontsize=9, fontweight='bold', color='#1e7e34')

plt.tight_layout()
os.makedirs("data", exist_ok=True)
out_img = os.path.join("data", "chart_ood_domain_generalization.png")
plt.savefig(out_img, bbox_inches='tight', dpi=300)
print(f"Successfully generated OOD generalization chart at: {out_img}")
