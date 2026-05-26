"""
统计 papers.json（+ classic_papers.json）中论文按月份的分布，绘制直方图。
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_ROOT = Path(__file__).parent

# 读取论文数据
papers = json.loads((_ROOT / "data" / "papers.json").read_text(encoding="utf-8"))

classic_path = _ROOT / "data" / "classic_papers.json"
if classic_path.exists():
    classic = json.loads(classic_path.read_text(encoding="utf-8"))
    existing = {p.get("arxiv_id") for p in papers}
    papers += [p for p in classic if p.get("arxiv_id") not in existing]

# 提取年月（YYYY-MM），跳过 date 为空的
months = [p["date"][:7] for p in papers if p.get("date") and len(p["date"]) >= 7]
counts = Counter(months)

# 按时间排序
labels = sorted(counts.keys())
values = [counts[m] for m in labels]

# ── 绘图 ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 5))

bars = ax.bar(range(len(labels)), values, color="#4C8EDA", edgecolor="white", linewidth=0.5)

# 数值标签
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=7)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xlabel("Month")
ax.set_ylabel("Papers")
ax.set_title(f"Paper Distribution by Month  (total: {len(months):,})")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
out = _ROOT / "hist.png"
plt.savefig(out, dpi=150)
print(f"已保存 → {out}  ({len(labels)} 个月份，{len(months):,} 篇)")
plt.show()
