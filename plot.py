import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set(style="whitegrid", context="paper")
colors = sns.color_palette("colorblind")

def parse_log(filepath):
    steps, test_seq = [], []
    best = None
    with open(filepath) as f:
        for line in f:
            m = re.search(r'\[FT\] Step\s+(\d+).*?test seq\s+([0-9.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                test_seq.append(float(m.group(2)))
            b = re.search(r'best test seq acc\s*:\s*([0-9.]+)', line)
            if b:
                best = float(b.group(1))
    return pd.DataFrame({'step': steps, 'test_seq': test_seq}), best

files = {
    'seed_1':    '/home/ss95332/src/pycharmprojects/Smallest-GPT-for-addition/logs/toygpt_lawa_seed=1.txt',
    'seed_42':   '/home/ss95332/src/pycharmprojects/Smallest-GPT-for-addition/logs/toygpt_lawa_seed=42.txt',
    'seed_222':  '/home/ss95332/src/pycharmprojects/Smallest-GPT-for-addition/logs/toygpt_lawa_seed=222.txt',
    'seed_1337': '/home/ss95332/src/pycharmprojects/Smallest-GPT-for-addition/logs/toygpt_lawa_seed=1337.txt',
    'seed_0':    '/home/ss95332/src/pycharmprojects/Smallest-GPT-for-addition/logs/toygpt_lawa_seed=0.txt',
}

dfs = {}
bests = []
for k, v in files.items():
    df, best = parse_log(v)
    dfs[k] = df
    if best is not None:
        bests.append(best)

best_overall = max(bests)

# Align on common steps
common_steps = sorted(set.intersection(*[set(df['step']) for df in dfs.values()]))
aligned = np.array([dfs[k].set_index('step').loc[common_steps, 'test_seq'].values for k in dfs])

mean = aligned.mean(axis=0)
std  = aligned.std(axis=0)
steps_k = np.array(common_steps) / 1000  # scale to thousands

window = 2
def smooth(arr):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values

mean_s = smooth(mean)
std_s  = smooth(std)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(steps_k, mean_s, color=colors[0], linewidth=3, label='GPT (296 Params)', zorder=3)
ax.fill_between(steps_k, mean_s - std_s, mean_s + std_s,
                color=colors[0], alpha=0.2, zorder=2)

# Big yellowish-brown star at the start point
ax.scatter(steps_k[0], mean_s[0], marker='*', s=600, color='#B8860B', zorder=5)
ax.annotate('Pre-trained', xy=(steps_k[0], mean_s[0]),
            xytext=(steps_k[0] + 1.5, mean_s[0]),
            fontsize=14, color='#B8860B', va='center', fontweight='bold')

ax.set_title('ToyGPT 10 Digit Addition  — 5 Seeds', fontsize=16)
ax.set_xlabel('Steps (in K)', fontsize=20)
ax.set_ylabel('Test Seq Accuracy', fontsize=20)
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# Main legend
legend = ax.legend(fontsize=18, loc='lower right')

# Separate box above legend for best acc
from matplotlib.patches import FancyBboxPatch
ax.annotate(f'Best Test Seq: {best_overall:.4f}',
            xy=(1, 0), xycoords='axes fraction',
            xytext=(-10, 10 + legend.get_window_extent(fig.canvas.get_renderer()).height),
            textcoords='offset points',
            fontsize=15, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

plt.tight_layout()
pdf_filename = '/home/ss95332/src/pycharmprojects/Smallest-GPT-for-addition/Figure/toyGPT-296params.pdf'
plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', dpi=600)
print(f"Saved: {pdf_filename}")

png_filename = '/home/ss95332/src/pycharmprojects/Smallest-GPT-for-addition/Figure/toyGPT-296params.png'
plt.savefig(png_filename, format='png', bbox_inches='tight', dpi=600)
plt.show()