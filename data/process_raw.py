"""
工具模块：通过 Ollama 大模型从 raw.txt 中提取论文的 title 和 summary。
不要直接运行本文件，从 main.py 调用 extract_papers()。
"""

import re
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

# ── Ollama 配置 ──────────────────────────────────────────────────────────────
# OLLAMA_API      = "http://localhost:11434/api/generate"
OLLAMA_API      = "http://10.54.79.131:11434/api/generate"
MODEL           = "gpt-oss:120b"
PAPERS_PER_CHUNK = 10   # 每批发给模型的论文数量，小块保证边界准确、内容干净
MAX_RETRIES     = 5     # JSON 解析失败时的最大重试次数

_DATA_DIR     = Path(__file__).parent
PROGRESS_FILE = _DATA_DIR / ".progress.json"  # 中间进度文件，支持中断续跑

# ── 提示词 ───────────────────────────────────────────────────────────────────
# 说明了三种情况：
#   1. 有 summary → 正常提取
#   2. 没有 summary → summary 留空字符串
#   3. 完全无关的元文字 → 直接忽略（但与论文相关的元文字可保留到紧邻论文的 summary 末尾）
PROMPT = """从以下文本中提取所有研究论文条目，返回JSON数组。
格式：[{{"title": "论文标题", "summary": "摘要描述"}}]
规则：
- 只提取真实的研究论文条目，跳过纯章节标题（如"1. 相关综述"）和分隔线
- title：论文名称本身，去掉"（1）""633."等序号前缀
- summary：紧随标题后的描述/介绍文字，多段合并为一段；如果该论文没有描述，summary 留空字符串 ""
- **summary 必须逐字复制原文，禁止改写、缩减、润色、翻译或加入任何自己的理解**
- 对于与论文内容完全无关的元文字（如"这个系列包含1000篇论文…""因为字数太多所以拆开…"等），直接忽略
- 如果某段元文字对紧邻论文有一定补充说明，可酌情保留在该论文 summary 末尾
- 只输出合法 JSON 数组，不要任何其他文字

文本：
{chunk}"""


def _is_false_positive(raw_line: str) -> bool:
    """
    判断一个「（N）...」或「NNN....」行是否是假阳性（列表项/步骤描述，而非论文标题）。

    规则1：去掉序号前缀后内容以「。」或「…」结尾
          → 是完整句子，不是标题
          例：「（4）从全部输出中选择分数最高的一个作为最终的输出。」

    规则2：内容含行内文献引用「[数字]」
          → 是描述文字中的参考文献标注，不是独立标题
          例：「（5）MDM[24]（这个方法…）」
    """
    content = re.sub(r'^（\d+）', '', raw_line.strip()).strip()
    content = re.sub(r'^\d{3,}\.', '', content).strip()
    if content.endswith('。') or content.endswith('…'):
        return True
    if re.search(r'\[\d+\]', content):
        return True
    return False


def _find_paper_starts(lines: list[str]) -> list[int]:
    """
    扫描所有行，找到论文条目的起始行索引。
    支持两种格式：
      格式1：（1）Title     ← 新版笔记格式，中文全角括号序号
      格式2：633.Title      ← 旧版笔记格式，三位及以上数字加句点
    过滤掉列表项、步骤描述等假阳性，保证分块边界落在真实论文标题上。
    """
    fmt1 = re.compile(r'^（\d+）.{3,}')
    fmt2 = re.compile(r'^\d{3,}\..{3,}')
    return [
        i for i, line in enumerate(lines)
        if (fmt1.match(line.strip()) or fmt2.match(line.strip()))
        and not _is_false_positive(line)
    ]


def _build_chunks(lines: list[str],
                  starts: list[int],
                  per_chunk: int) -> list[tuple[int, str]]:
    """
    按论文边界切块，每块包含 per_chunk 篇论文。
    返回 [(起始行号_1索引, 文本块), ...]
    按论文边界切割可避免单篇论文被截断在两个块之间。
    """
    chunks = []
    for i in range(0, len(starts), per_chunk):
        start = starts[i]
        # 下一块的起点就是当前块的终点，最后一块取到文件末尾
        end = starts[i + per_chunk] if i + per_chunk < len(starts) else len(lines)
        chunks.append((start + 1, ''.join(lines[start:end])))
    return chunks


def _call_ollama(text: str) -> str:
    """
    调用本地 Ollama API，返回模型的原始文本响应。
    timeout=600 是为了给 120B 大模型留足推理时间。
    """
    resp = requests.post(OLLAMA_API, json={
        "model":  MODEL,
        "prompt": PROMPT.format(chunk=text),
        "stream": False,
        "options": {
            "temperature": 0.1,   # 低温度保证输出格式稳定
            "num_ctx":     131072, # 128k 上下文，足够容纳 50 篇论文
        },
    }, timeout=600)
    resp.raise_for_status()
    return resp.json().get("response", "")


def _parse_json(text: str) -> list[dict] | None:
    """
    从模型响应中提取 JSON 数组。
    先去掉 <think>…</think> 思考块，再用正则找 [...] 片段。
    如果 JSON 不完整（如被截断），尝试修复后再解析。
    返回解析成功的列表，或 None（表示本次响应无法解析，需要重试）。
    """
    # 去掉可能存在的思考链标签
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    m = re.search(r'\[.*\]', text, re.DOTALL)
    if not m:
        return None

    # 先尝试原始片段，再尝试截断修复版本
    for candidate in (m.group(), m.group().rstrip(',') + ']'):
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            continue
    return None


def _normalize(title: str) -> str:
    """标准化标题用于去重：小写 + 合并多余空格。"""
    return re.sub(r'\s+', ' ', title.lower().strip())


def _load_progress() -> dict:
    """读取上次保存的进度文件，若不存在则返回空状态。"""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
    return {"done_chunks": [], "papers": []}


def _save_progress(state: dict) -> None:
    """将当前进度写入文件，供中断后续跑使用。"""
    PROGRESS_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


class _Tee:
    """
    同时向控制台和日志文件写入输出。
    用 sys.stdout = _Tee(log_path) 替换标准输出，
    所有 print() 调用自动同步写入文件，无需修改其他代码。
    """
    def __init__(self, log_path: Path):
        self._console = sys.stdout
        self._file = open(log_path, 'a', encoding='utf-8', buffering=1)  # 行缓冲，实时写入

    def write(self, data: str):
        self._console.write(data)
        self._file.write(data)

    def flush(self):
        self._console.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._console  # 恢复原始 stdout


def extract_papers(input_file: str | Path,
                   per_chunk: int = PAPERS_PER_CHUNK) -> list[dict]:
    """
    主函数：从 raw.txt 提取论文列表，返回 [{title, summary}, ...] 去重结果。

    流程：
      1. 检测论文起始行 → 按边界分块
      2. 每块发给 Ollama，解析返回的 JSON
      3. 若 JSON 无效，最多重试 MAX_RETRIES 次；仍失败则跳过该块
      4. 每块处理完后立即保存进度（支持 Ctrl+C 后从断点继续）
      5. 全部完成后删除进度文件并返回结果

    参数：
      input_file  : raw.txt 路径
      per_chunk   : 每批处理的论文数，默认 10
    """
    input_file = Path(input_file)

    # 日志文件：process_raw_YYYYMMDD_HHMMSS.log
    log_path = _DATA_DIR / f"process_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"[process_raw] 日志 → {log_path}")
    print(f"[process_raw] 读取 {input_file} ...")
    lines = input_file.read_text(encoding='utf-8').splitlines(keepends=True)
    print(f"  共 {len(lines):,} 行")

    starts = _find_paper_starts(lines)
    print(f"  检测到 {len(starts):,} 篇论文条目")

    chunks = _build_chunks(lines, starts, per_chunk)
    total  = len(chunks)
    print(f"  分为 {total} 块（每块约 {per_chunk} 篇）\n")

    # 加载进度：done_chunks 记录已完成的块 ID，papers 是已提取的结果
    state     = _load_progress()
    done      = set(state["done_chunks"])
    seen      = {_normalize(p["title"]) for p in state["papers"]}
    all_papers: list[dict] = state["papers"]

    # 已完成的块数（用于 ETA 计算，跳过的不计入耗时）
    completed_times: list[float] = []
    session_start = time.time()

    for idx, (line_num, chunk) in enumerate(chunks):
        chunk_id = f"{idx}_{line_num}"
        if chunk_id in done:
            continue  # 已处理，直接跳过

        # ── 进度前缀 ──────────────────────────────────────────────────
        remaining = total - len(done) - 1   # 本块之后还剩多少块
        if completed_times:
            avg_sec  = sum(completed_times) / len(completed_times)
            eta_sec  = avg_sec * remaining
            eta_str  = str(timedelta(seconds=int(eta_sec)))
            avg_str  = f"{avg_sec:.1f}s/块"
        else:
            eta_str = "计算中"
            avg_str = "--"

        pct = (len(done) / total) * 100
        print(f"  [{idx+1:>4}/{total}] {pct:5.1f}%  行~{line_num:<6}"
              f"  均速:{avg_str}  ETA:{eta_str}  ", end='', flush=True)

        # ── 带重试的模型调用 ──────────────────────────────────────────
        t0     = time.time()
        papers = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw    = _call_ollama(chunk)
                papers = _parse_json(raw)
                if papers is not None:
                    break
                print(f"(重试{attempt}) ", end='', flush=True)
            except Exception as e:
                print(f"(错误{attempt}:{e}) ", end='', flush=True)
        elapsed = time.time() - t0

        if papers is None:
            print(f"→ 重试耗尽，跳过  [{elapsed:.1f}s]")
        else:
            added = 0
            for p in papers:
                title   = (p.get('title')   or '').strip()
                summary = (p.get('summary') or '').strip()
                if not title:
                    continue
                key = _normalize(title)
                if key not in seen:
                    seen.add(key)
                    all_papers.append({"title": title, "summary": summary})
                    added += 1
            print(f"→ +{added} 篇  累计:{len(all_papers)}  [{elapsed:.1f}s]")
            completed_times.append(elapsed)

        # 无论成功与否都标记为完成，避免卡在同一块反复失败
        done.add(chunk_id)
        _save_progress({"done_chunks": list(done), "papers": all_papers})

    # 全部完成，清理进度文件
    PROGRESS_FILE.unlink(missing_ok=True)
    total_sec = time.time() - session_start
    print(f"\n[process_raw] 完成 — {len(all_papers):,} 篇去重论文"
          f"  总耗时:{str(timedelta(seconds=int(total_sec)))}")
    tee.close()
    return all_papers
