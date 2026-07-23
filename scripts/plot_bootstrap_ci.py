import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'figure.dpi': 300, 'font.family': 'sans-serif'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Data for 1,000 Bootstrap 95% Confidence Intervals
models = ['KazRoBERTa (Pure)', 'KazRoBERTa (FST)']

# Subplot 1: F1-Score (%) with 95% CI error bars
f1_means = [96.09, 96.07]
f1_lows = [96.09 - 95.45, 96.07 - 95.44]
f1_highs = [96.69 - 96.09, 96.64 - 96.07]

colors = ['#d9534f', '#5cb85c']

ax1.grid(True, linestyle='--', alpha=0.5)
for i in range(2):
    ax1.errorbar(models[i], f1_means[i], yerr=[[f1_lows[i]], [f1_highs[i]]],
                 fmt='o', color=colors[i], ecolor=colors[i], elinewidth=3, capsize=8, capthick=2, markersize=9)
    ax1.annotate(f"{f1_means[i]:.2f}%\n[95% CI]", (i, f1_means[i] + 0.15), ha='center', fontweight='bold', fontsize=10)

ax1.set_title("F1-Score (%) with 95% Bootstrap CI (B=1000)", pad=15, fontweight='bold')
ax1.set_ylabel("F1-Score (%)")
ax1.set_ylim(94.5, 97.5)

# Subplot 2: Short-Text False Positives with 95% CI error bars
fp_means = [52.88, 35.91]
fp_lows = [52.88 - 39.00, 35.91 - 25.00]
fp_highs = [67.00 - 52.88, 48.00 - 35.91]

ax2.grid(True, linestyle='--', alpha=0.5)
for i in range(2):
    ax2.errorbar(models[i], fp_means[i], yerr=[[fp_lows[i]], [fp_highs[i]]],
                 fmt='s', color=colors[i], ecolor=colors[i], elinewidth=3, capsize=8, capthick=2, markersize=9)
    ax2.annotate(f"{int(round(fp_means[i]))}\n[95% CI]", (i, fp_means[i] + 2.5), ha='center', fontweight='bold', fontsize=10)

ax2.set_title("Short-Text False Positives (95% Bootstrap CI)\nStatistically Significant (-32.07%, p = 0.0001)", pad=15, fontweight='bold', color='#111111')
ax2.set_ylabel("False Positive Count (<=60 chars)")
ax2.set_ylim(15, 75)

plt.tight_layout()
os.makedirs("data", exist_ok=True)
out_img = os.path.join("data", "chart_bootstrap_confidence_intervals.png")
plt.savefig(out_img, bbox_inches='tight', dpi=300)
print(f"Successfully generated Bootstrap CI plot at: {out_img}")
