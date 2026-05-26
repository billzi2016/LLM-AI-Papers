"""
工具模块：处理 classic.csv，从 arxiv 快照提取经典论文的标题、摘要和日期。
不要直接运行，从 main.py 调用 process_classic()。

输入：  ../classic.csv（arxiv_id, title, year, category）
输出：  classic_papers.json（与 papers.json 相同 schema）

arxiv_id 为空的行（如 LSTM、AlexNet 等未上 arxiv 的论文）直接跳过。
"""

import csv
import json
from pathlib import Path

from check_arxiv import get_abstracts, SNAPSHOT_FILE

_DATA_DIR    = Path(__file__).parent
_REPO_ROOT   = _DATA_DIR.parent
CLASSIC_CSV  = _REPO_ROOT / "classic.csv"
CLASSIC_JSON = _DATA_DIR  / "classic_papers.json"


def process_classic() -> list[dict]:
    """
    读取 classic.csv，扫描 arxiv 快照获取规范标题、摘要和日期，
    返回与 papers.json 相同 schema 的列表。

    返回
    ─────
    [{"title": arxiv规范标题, "arxiv_id": ..., "summary": "", "abstract": ..., "date": ...}, ...]
    """
    # 读取 CSV，只保留有 arxiv_id 的行
    rows = []
    with open(CLASSIC_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            aid = row.get('arxiv_id', '').strip()
            if aid:
                rows.append(aid)

    if not rows:
        print("[process_classic] classic.csv 中无有效 arxiv_id，跳过。")
        return []

    print(f"[process_classic] 从 classic.csv 读取 {len(rows)} 个 arxiv_id，扫描快照...")
    info_map = get_abstracts(set(rows))  # {arxiv_id: {title, abstract, date}}

    results: list[dict] = []
    missing = []
    for aid in rows:
        info = info_map.get(aid)
        if not info:
            missing.append(aid)
            continue
        results.append({
            "title":    info.get('title', ''),    # arxiv 快照中的规范标题
            "arxiv_id": aid,
            "summary":  "",                        # 经典论文无 raw.txt 描述
            "abstract": info.get('abstract', ''),
            "date":     info.get('date', ''),
        })

    if missing:
        print(f"  [process_classic] 在快照中未找到的 ID（{len(missing)} 个）：{missing}")

    CLASSIC_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f"[process_classic] 完成 — {len(results)} 篇 → {CLASSIC_JSON}")
    return results
