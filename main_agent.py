"""
主入口：读取 papers.json，为每篇论文生成中文解读并提交 git commit。

流程
─────────────────────────────────────────────────────────────────
1. 读取 data/papers.json
2. 加载 .reader_progress.json（断点续跑，跳过已完成的论文）
3. 对每篇未处理的论文：
   a. 调用 reader_agent.generate_reading() 生成中文解读
   b. 调用 committer.write_and_commit() 写文件 + git commit
   c. 保存进度
4. 全部完成，打印统计

运行方式
─────────────────────────────────────────────────────────────────
  cd /Users/bizi/Desktop/GitHub/LLM-AI-Papers
  python3 main_agent.py

完成后手动 push：
  git push
─────────────────────────────────────────────────────────────────
"""

import json
import time
from datetime import timedelta
from pathlib import Path

from reader_agent import generate_reading
from committer   import write_and_commit

_ROOT         = Path(__file__).parent
PAPERS_FILE   = _ROOT / "data" / "papers.json"
PROGRESS_FILE = _ROOT / ".reader_progress.json"
FAILED_FILE   = _ROOT / ".reader_failed.json"


def _load_progress() -> dict:
    """读取上次保存的进度，若不存在则返回空状态。"""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
    return {"done": []}


def _save_progress(state: dict) -> None:
    """将当前进度写入文件，供中断后续跑使用。"""
    PROGRESS_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def _key(paper: dict) -> str:
    """用 arxiv_id 作为论文唯一标识；无 id 时退化到 title。"""
    return paper.get('arxiv_id') or paper.get('title', '')


def main() -> None:
    papers = json.loads(PAPERS_FILE.read_text(encoding='utf-8'))
    total  = len(papers)
    print(f"[main_agent] papers.json 共 {total:,} 篇")

    state = _load_progress()
    done  = set(state["done"])
    todo  = [p for p in papers if _key(p) not in done]
    print(f"  已完成：{len(done):,}  待处理：{len(todo):,}\n")

    failed: list[dict] = []
    elapsed_times: list[float] = []
    session_start = time.time()

    for i, paper in enumerate(todo):
        remaining = len(todo) - i - 1

        # ETA 计算
        if elapsed_times:
            avg     = sum(elapsed_times) / len(elapsed_times)
            eta_str = str(timedelta(seconds=int(avg * remaining)))
            avg_str = f"{avg:.1f}s/篇"
        else:
            eta_str, avg_str = "计算中", "--"

        pct = ((len(done) + i) / total) * 100
        print(f"  [{i+1:>5}/{len(todo)}] {pct:5.1f}%  {paper['title'][:48]:<48}"
              f"  均速:{avg_str}  ETA:{eta_str}  ", end='', flush=True)

        t0     = time.time()
        output = generate_reading(paper)
        elapsed = time.time() - t0

        if output is None:
            print(f"→ 生成失败  [{elapsed:.1f}s]")
            failed.append(paper)
        else:
            try:
                path = write_and_commit(paper, output)
                rel  = path.relative_to(_ROOT)
                print(f"→ {rel}  [{elapsed:.1f}s]")
                elapsed_times.append(elapsed)
            except Exception as e:
                print(f"→ commit 失败：{e}  [{elapsed:.1f}s]")
                failed.append(paper)

        done.add(_key(paper))
        _save_progress({"done": list(done)})

    # 保存失败列表，供人工检查或重跑
    if failed:
        FAILED_FILE.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n  失败列表 → {FAILED_FILE}")

    total_sec = time.time() - session_start
    success   = len(todo) - len(failed)
    print(f"\n[main_agent] 完成 — 成功：{success:,}  失败：{len(failed):,}"
          f"  总耗时：{str(timedelta(seconds=int(total_sec)))}")
    print("  请手动执行 git push 推送到远端。")

    # 全部成功时清理进度文件
    if not failed:
        PROGRESS_FILE.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
