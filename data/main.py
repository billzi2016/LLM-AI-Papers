"""
主入口：整合 process_raw 和 check_arxiv，完成完整的论文提取和匹配流程。

流程
─────────────────────────────────────────────────────────────
步骤1  process_raw.extract_papers()
       用 Ollama 大模型从 raw.txt 提取 (title, summary) 列表
       → 中间结果保存到 papers_extracted.json

步骤2  check_arxiv.build_index()
       从 arxiv 快照构建标题索引（首次约 5 分钟，之后缓存）

步骤3  匹配
       对每篇提取的论文，在 arxiv 索引中三级查找
       找到 → 记录 arxiv_id 和 matched_title
       找不到 → 记录为失败

步骤4  check_arxiv.get_abstracts()
       对命中的论文，流式扫描快照获取 abstract

步骤5  输出
       命中的论文 → papers.json       {title, summary, abstract}
       未命中的论文 → failed.json     {title, summary, abstract: ""}
─────────────────────────────────────────────────────────────

输出文件说明
─────────────────────────────────────────────────────────────
papers.json
  - title    : arxiv 中的规范标题（保证大小写和标点正确）
  - summary  : raw.txt 中作者写的描述
  - abstract : arxiv 原始摘要

failed.json
  - title    : raw.txt 提取的标题（arxiv 中未找到对应论文）
  - summary  : raw.txt 中的描述
  - abstract : ""（空字符串，因为未找到）
─────────────────────────────────────────────────────────────
"""

import json
from pathlib import Path

from process_raw     import extract_papers
from check_arxiv     import build_index, find_paper, get_abstracts
from process_classic import process_classic

DATA_DIR = Path(__file__).parent

# 中间文件：Ollama 提取结果，可用于调试或单独重跑步骤 2-5
EXTRACTED_FILE = DATA_DIR / "papers_extracted.json"

# 最终输出：在 arxiv 中找到匹配的论文
FINAL_FILE = DATA_DIR / "papers.json"

# 失败输出：在 arxiv 中未找到匹配的论文
FAILED_FILE = DATA_DIR / "failed.json"


def main() -> None:
    # ──────────────────────────────────────────────────────────────────── #
    # 步骤1：从 raw.txt 提取论文                                            #
    # extract_papers 每处理完一块就保存进度，支持 Ctrl+C 后从断点续跑。       #
    # ──────────────────────────────────────────────────────────────────── #
    raw_file = DATA_DIR / "raw.txt"
    papers = extract_papers(raw_file)

    # 保存中间结果，方便单独调试后续步骤
    EXTRACTED_FILE.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"[main] 提取完成：{len(papers):,} 篇论文 → {EXTRACTED_FILE}\n")

    # ──────────────────────────────────────────────────────────────────── #
    # 步骤2：构建 arxiv 标题索引                                            #
    # 首次运行读取 4.9 GB 快照约需 5 分钟，构建完成后缓存到 .arxiv_index.pkl。 #
    # 之后再运行直接加载缓存，几秒内完成。                                    #
    # ──────────────────────────────────────────────────────────────────── #
    index = build_index()

    # ──────────────────────────────────────────────────────────────────── #
    # 步骤3：匹配论文                                                       #
    # 三级策略（详见 check_arxiv.py 注释）：                                 #
    #   L1 标准化精确匹配  → 处理标点/大小写/空格差异                         #
    #   L2 词袋排序匹配    → 处理词序差异                                    #
    #   L3 倒排索引+模糊   → 容忍约 1 个词的差异                             #
    # ──────────────────────────────────────────────────────────────────── #
    print(f"\n[main] 开始匹配 {len(papers):,} 篇论文 ...")
    matched: dict[int, dict] = {}   # {论文下标 → {arxiv_id, matched_title, score}}
    failed_indices: list[int] = []  # 未命中的论文下标

    for i, paper in enumerate(papers):
        result = find_paper(paper['title'], index)
        if result:
            matched[i] = result
        else:
            failed_indices.append(i)

    print(f"  命中：{len(matched):,} 篇，未命中：{len(failed_indices):,} 篇\n")

    # ──────────────────────────────────────────────────────────────────── #
    # 步骤4：获取命中论文的摘要                                              #
    # 流式扫描快照，只提取命中论文的 abstract，避免把 240 万条摘要全载入内存。  #
    # ──────────────────────────────────────────────────────────────────── #
    arxiv_ids = {m['arxiv_id'] for m in matched.values()}
    abstracts = get_abstracts(arxiv_ids)   # {arxiv_id: abstract 文本}

    # ──────────────────────────────────────────────────────────────────── #
    # 步骤5：写出结果                                                       #
    # papers.json  ← 命中论文，title 使用 arxiv 规范标题                    #
    # failed.json  ← 未命中论文，abstract 留空                              #
    # ──────────────────────────────────────────────────────────────────── #

    # 命中论文
    final: list[dict] = []
    for i, paper in enumerate(papers):
        if i not in matched:
            continue
        m = matched[i]
        info = abstracts.get(m['arxiv_id'], {})
        final.append({
            "title":     m['matched_title'],          # arxiv 规范标题
            "arxiv_id":  m['arxiv_id'],               # arxiv 论文 ID
            "summary":   paper['summary'],             # raw.txt 中的描述
            "abstract":  info.get('abstract', ''),    # arxiv 摘要
            "date":      info.get('date', ''),         # 首次投稿日期 YYYY-MM-DD
        })

    FINAL_FILE.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"[main] papers.json  → {len(final):,} 篇  ({FINAL_FILE})")

    # 未命中论文（保留供人工核查或后续处理）
    failed: list[dict] = []
    for i in failed_indices:
        paper = papers[i]
        failed.append({
            "title":     paper['title'],   # raw.txt 提取的标题（可能有误差）
            "arxiv_id":  "",              # 未找到，留空
            "summary":   paper['summary'], # raw.txt 中的描述
            "abstract":  "",              # 未找到，留空
            "date":      "",              # 未找到，留空
        })

    FAILED_FILE.write_text(
        json.dumps(failed, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"[main] failed.json  → {len(failed):,} 篇  ({FAILED_FILE})")

    # ──────────────────────────────────────────────────────────────────── #
    # 步骤6：处理 classic.csv                                               #
    # 从 arxiv 快照提取经典论文，写入 classic_papers.json                    #
    # ──────────────────────────────────────────────────────────────────── #
    print()
    process_classic()

    print(f"\n[main] 全部完成。")


if __name__ == '__main__':
    main()
