"""
主入口：读取 papers.json + classic_papers.json，多线程并行生成中文解读并提交。

架构
─────────────────────────────────────────────────────────────────
- 每个 Ollama 端点（load_balancer.ENDPOINTS）对应一个独立线程
- 所有线程共享同一个论文队列，各自独立调用模型
- git commit 通过锁串行化，避免 .git/index.lock 冲突
- 进度保存也通过锁串行化，保证写入安全

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
import threading
from collections import deque
from datetime import timedelta
from pathlib import Path

from load_balancer import ENDPOINTS
from reader_agent  import generate_reading
from committer     import write_and_commit

_ROOT         = Path(__file__).parent
PAPERS_FILE   = _ROOT / "data" / "papers.json"
CLASSIC_FILE  = _ROOT / "data" / "classic_papers.json"
PROGRESS_FILE = _ROOT / ".reader_progress.json"
FAILED_FILE   = _ROOT / ".reader_failed.json"


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
    return {"done": []}


def _key(paper: dict) -> str:
    return paper.get('arxiv_id') or paper.get('title', '')


def main() -> None:
    # ── 读取论文列表 ───────────────────────────────────────────────────────
    papers = json.loads(PAPERS_FILE.read_text(encoding='utf-8'))
    if CLASSIC_FILE.exists():
        classic     = json.loads(CLASSIC_FILE.read_text(encoding='utf-8'))
        existing    = {p.get('arxiv_id') for p in papers}
        classic     = [p for p in classic if p.get('arxiv_id') not in existing]
        papers      = papers + classic
        print(f"[main_agent] papers.json + classic_papers.json 合并，共 {len(papers):,} 篇")
    else:
        print(f"[main_agent] papers.json 共 {len(papers):,} 篇")
    total = len(papers)

    state = _load_progress()
    done  = set(state["done"])
    todo  = [p for p in papers if _key(p) not in done]
    print(f"  已完成：{len(done):,}  待处理：{len(todo):,}")
    print(f"  并行端点数：{len(ENDPOINTS)}\n")

    if not todo:
        print("[main_agent] 无待处理论文，退出。")
        return

    # ── 共享状态（所有线程共用） ──────────────────────────────────────────
    work_queue  = deque(todo)           # 论文队列，deque.popleft() 是原子操作
    queue_lock  = threading.Lock()      # 保护 popleft + 计数
    git_lock    = threading.Lock()      # git commit 串行化
    print_lock  = threading.Lock()      # 控制台输出不交错
    progress_lock = threading.Lock()    # 进度文件写入串行化

    failed:        list[dict]  = []
    elapsed_times: list[float] = []
    completed      = [0]                # 已完成数（列表包装以便线程修改）
    session_start  = time.time()

    def worker(endpoint: str, worker_id: int) -> None:
        """单个线程的工作循环：不断从队列取论文，生成解读，串行 commit。"""
        while True:
            # 从队列取一篇论文
            with queue_lock:
                if not work_queue:
                    break
                paper     = work_queue.popleft()
                local_idx = len(todo) - len(work_queue)  # 当前是第几篇

            t0     = time.time()
            output = generate_reading(paper, endpoint)
            elapsed = time.time() - t0

            title_short = paper['title'][:45]

            if output is None:
                with print_lock:
                    print(f"  [w{worker_id}] ✗  {title_short}  [{elapsed:.1f}s]", flush=True)
                with progress_lock:
                    failed.append(paper)
            else:
                # git 操作串行，避免锁文件冲突
                commit_ok = True
                with git_lock:
                    try:
                        path = write_and_commit(paper, output)
                        rel  = path.relative_to(_ROOT)
                        with print_lock:
                            print(f"  [w{worker_id}] ✓  {title_short}  → {rel}  [{elapsed:.1f}s]",
                                  flush=True)
                    except Exception as e:
                        commit_ok = False
                        with print_lock:
                            print(f"  [w{worker_id}] ✗ commit失败  {title_short}  {e}  [{elapsed:.1f}s]",
                                  flush=True)

                if not commit_ok:
                    with progress_lock:
                        failed.append(paper)
                else:
                    with progress_lock:
                        elapsed_times.append(elapsed)

            # 更新进度
            with progress_lock:
                done.add(_key(paper))
                completed[0] += 1
                pct = (completed[0] / len(todo)) * 100
                if elapsed_times:
                    avg     = sum(elapsed_times) / len(elapsed_times)
                    # 剩余队列长度 / 线程数 估算 ETA
                    remaining_est = len(work_queue) / max(len(ENDPOINTS), 1)
                    eta_str = str(timedelta(seconds=int(avg * remaining_est)))
                else:
                    eta_str = "计算中"
                with print_lock:
                    print(f"  [进度] {completed[0]:>5}/{len(todo)}  {pct:5.1f}%  ETA:{eta_str}",
                          flush=True)
                _save_progress(done)

    def _save_progress(done_set: set) -> None:
        PROGRESS_FILE.write_text(
            json.dumps({"done": list(done_set)}, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    # ── 启动线程，每端点一个 ──────────────────────────────────────────────
    threads = []
    for wid, ep in enumerate(ENDPOINTS):
        t = threading.Thread(target=worker, args=(ep, wid), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # ── 收尾 ──────────────────────────────────────────────────────────────
    if failed:
        FAILED_FILE.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n  失败列表 → {FAILED_FILE}")

    total_sec = time.time() - session_start
    success   = len(todo) - len(failed)
    print(f"\n[main_agent] 完成 — 成功：{success:,}  失败：{len(failed):,}"
          f"  总耗时：{str(timedelta(seconds=int(total_sec)))}")
    print("  请手动执行 git push 推送到远端。")

    if not failed:
        PROGRESS_FILE.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
