"""
工具模块：在 arxiv 元数据快照中检索论文。
不要直接运行本文件，从 main.py 调用 build_index() / find_paper() / get_abstracts()。

检索策略（三级递进，速度由快到慢）
─────────────────────────────────────────────
L1 标准化精确匹配
   去掉所有标点、小写、合并多空格后做 dict 查找。
   处理：标点差异、大小写差异、多余空格等，O(1)。

L2 词袋排序匹配（token-set）
   将标准化后的词排序拼成字符串再做 dict 查找。
   处理：词序不同（"Attention Is All You Need" vs "Is Attention All You Need"），O(1)。

L3 倒排索引 + rapidfuzz 模糊打分
   先用倒排索引筛出共享关键词最多的 100 个候选，
   再用 rapidfuzz.token_sort_ratio 打分，阈值 ≥ 90 才算命中。
   处理：缺少一个词、单词拼写微差等。
─────────────────────────────────────────────
索引第一次构建约需 5 分钟（读 4.9 GB 快照），之后缓存到 .arxiv_index.pkl，
下次加载只需几秒。
"""

import re
import json
import pickle
import email.utils
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz

_DATA_DIR     = Path(__file__).parent
SNAPSHOT_FILE = _DATA_DIR / "arxiv-metadata-oai-snapshot.json"
INDEX_CACHE   = _DATA_DIR / ".arxiv_index.pkl"

# 停用词：这些词在标题中过于常见，不适合作为倒排索引的关键词
_STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'for', 'on', 'with', 'and', 'or',
    'to', 'is', 'are', 'by', 'at', 'from', 'as', 'via', 'its', 'using',
    'we', 'our', 'be', 'it', 'this', 'that', 'not', 'but', 'can', 'do',
}


def _normalize(title: str) -> str:
    """
    标准化标题：小写 → 去掉所有标点符号和特殊字符 → 合并多余空格。
    例：'LLM-JEPA: Large...' → 'llm jepa large...'
    """
    t = re.sub(r'[^\w\s]', ' ', title.lower())  # 非字母数字/空格的字符替换为空格
    return re.sub(r'\s+', ' ', t).strip()


def _token_set_key(title: str) -> str:
    """
    词袋排序键：标准化后将所有词排序再拼接。
    两个词序不同的标题会产生相同的 key，实现词序无关匹配。
    例：'A Survey on LLMs' 和 'LLMs A Survey on' → 相同 key
    """
    return ' '.join(sorted(_normalize(title).split()))


def _parse_date(date_str: str) -> str:
    """
    将 arxiv versions[0].created 的日期字符串转换为 YYYY-MM-DD。
    输入格式例：'Mon, 13 Jan 2025 00:00:00 GMT'
    解析失败时返回空字符串。
    """
    try:
        parsed = email.utils.parsedate(date_str)
        if parsed:
            return f"{parsed[0]:04d}-{parsed[1]:02d}-{parsed[2]:02d}"
    except Exception:
        pass
    return ''


def _content_words(title: str) -> list[str]:
    """
    提取标题中的关键词（去除停用词，长度 > 2）。
    这些词用于构建和查询倒排索引。
    """
    return [
        w for w in _normalize(title).split()
        if w not in _STOPWORDS and len(w) > 2
    ]


def build_index(snapshot_file: Path = SNAPSHOT_FILE,
                cache_file: Path = INDEX_CACHE) -> dict:
    """
    构建 arxiv 标题索引，若缓存文件存在则直接加载。

    索引结构
    ─────────
    entries    : list[(arxiv_id, original_title)]  所有论文的 ID 和原始标题
    exact      : {标准化标题 → entries 下标}        L1 精确匹配用
    token_set  : {词袋排序键 → entries 下标}        L2 词序无关匹配用
    inverted   : {关键词 → [entries 下标列表]}       L3 候选筛选用

    参数
    ─────
    snapshot_file : arxiv JSONL 快照路径
    cache_file    : pickle 缓存路径，构建完成后自动写入
    """
    if cache_file.exists():
        print(f"[check_arxiv] 加载缓存索引 {cache_file} ...")
        with open(cache_file, 'rb') as f:
            idx = pickle.load(f)
        print(f"  已加载 {len(idx['entries']):,} 篇论文。")
        return idx

    print(f"[check_arxiv] 从 {snapshot_file} 构建索引")
    print("  首次构建约需 5 分钟，完成后缓存到 .arxiv_index.pkl")

    entries   = []                    # [(arxiv_id, title), ...]
    exact     = {}                    # 标准化标题 → 下标
    token_set = {}                    # 词袋排序键 → 下标
    inverted  = defaultdict(list)     # 关键词 → [下标, ...]

    with open(snapshot_file, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过损坏行

            arxiv_id = obj.get('id', '').strip()
            title    = obj.get('title', '').replace('\n', ' ').strip()
            if not arxiv_id or not title:
                continue

            i = len(entries)
            entries.append((arxiv_id, title))

            # L1 索引：标准化标题（若有重复标题只保留第一个）
            exact.setdefault(_normalize(title), i)

            # L2 索引：词袋排序键
            token_set.setdefault(_token_set_key(title), i)

            # L3 索引：每个关键词都指向本论文
            for word in _content_words(title):
                inverted[word].append(i)

            if (lineno + 1) % 500_000 == 0:
                print(f"  已处理 {lineno + 1:,} 行，{len(entries):,} 篇论文 ...")

    idx = {
        'entries':   entries,
        'exact':     exact,
        'token_set': token_set,
        'inverted':  dict(inverted),
    }

    print(f"  保存索引到 {cache_file} ...")
    with open(cache_file, 'wb') as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[check_arxiv] 索引构建完成 — {len(entries):,} 篇论文。")
    return idx


def find_paper(title: str, index: dict, threshold: int = 90) -> dict | None:
    """
    在索引中查找论文，三级递进策略。

    参数
    ─────
    title     : 待查标题（来自 raw.txt 提取结果）
    index     : build_index() 返回的索引字典
    threshold : L3 模糊匹配最低分数（0-100），默认 90

    返回
    ─────
    命中时：{'arxiv_id': ..., 'matched_title': ..., 'score': ...}
    未命中：None
    """
    entries = index['entries']

    def _hit(i: int, score: int) -> dict:
        """封装命中结果。"""
        return {
            'arxiv_id':      entries[i][0],   # arxiv 论文 ID（如 "2301.00001"）
            'matched_title': entries[i][1],   # arxiv 中的原始标题
            'score':         score,           # 匹配分数（100=精确，90+=模糊）
        }

    # ── L1：标准化精确匹配 ───────────────────────────────────────────────
    # 处理标点差异、大小写差异、多余空格等，速度 O(1)
    norm = _normalize(title)
    if norm in index['exact']:
        return _hit(index['exact'][norm], 100)

    # ── L2：词袋排序匹配 ─────────────────────────────────────────────────
    # 对词序不同的标题也能命中，速度 O(1)
    tsk = _token_set_key(title)
    if tsk in index['token_set']:
        return _hit(index['token_set'][tsk], 98)

    # ── L3：倒排索引候选筛选 + rapidfuzz 模糊打分 ────────────────────────
    words = _content_words(title)
    if not words:
        return None  # 无有效关键词，无法检索

    # 统计每篇候选论文与查询共享的关键词数量
    counts: dict[int, int] = {}
    for word in words:
        for i in index['inverted'].get(word, []):
            counts[i] = counts.get(i, 0) + 1

    if not counts:
        return None

    # 至少要有一半的关键词匹配才纳入候选（过滤明显不相关的论文）
    min_overlap = max(1, len(words) // 2)
    candidates = sorted(
        [(i, c) for i, c in counts.items() if c >= min_overlap],
        key=lambda x: -x[1]   # 按共享词数从多到少排序
    )[:100]                    # 只取前 100 个候选，避免计算量过大

    # 用 rapidfuzz token_sort_ratio 对候选打分，找最高分
    # token_sort_ratio 会先对词排序再比较，对词序差异鲁棒
    best_score, best_i = 0, -1
    for i, _ in candidates:
        score = fuzz.token_sort_ratio(norm, _normalize(entries[i][1]))
        if score > best_score:
            best_score, best_i = score, i

    if best_i >= 0 and best_score >= threshold:
        return _hit(best_i, best_score)

    return None  # 三级均未命中


def get_abstracts(arxiv_ids: set[str],
                  snapshot_file: Path = SNAPSHOT_FILE) -> dict[str, dict]:
    """
    流式扫描 arxiv 快照，提取指定 ID 列表对应的 abstract 和首次投稿日期。

    之所以不在 build_index 时一并存储 abstract，是因为 240 万篇论文的摘要
    全部载入内存约需 2-3 GB，而实际只有几千篇命中，单独扫一遍代价更低。

    参数
    ─────
    arxiv_ids     : 需要获取摘要的 arxiv ID 集合
    snapshot_file : 快照文件路径

    返回
    ─────
    {arxiv_id: {"title": str, "abstract": str, "date": str}}
    date 格式为 YYYY-MM-DD（首次投稿时间），找不到的 ID 不会出现在结果中
    """
    if not arxiv_ids:
        return {}

    remaining = set(arxiv_ids)  # 剩余未找到的 ID，找完就停止扫描
    result: dict[str, dict] = {}

    print(f"[check_arxiv] 获取 {len(remaining):,} 篇论文的摘要和日期（扫描快照）...")
    with open(snapshot_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            aid = obj.get('id', '').strip()
            if aid in remaining:
                # 摘要和标题中可能含有换行符，统一替换为空格
                title    = obj.get('title',    '').replace('\n', ' ').strip()
                abstract = obj.get('abstract', '').replace('\n', ' ').strip()
                # 取 versions 数组中第一个元素的 created 字段作为首次投稿日期
                versions = obj.get('versions', [])
                date_raw = versions[0].get('created', '') if versions else ''
                result[aid] = {
                    'title':    title,
                    'abstract': abstract,
                    'date':     _parse_date(date_raw),
                }
                remaining.discard(aid)
                if not remaining:
                    break  # 所有 ID 都找到了，提前终止

    print(f"  获取到 {len(result):,} 篇摘要和日期。")
    return result
