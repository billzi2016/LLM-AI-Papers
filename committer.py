"""
工具模块：将论文解读写入 Markdown 文件并提交 git commit。
不要直接运行，从 main_agent.py 调用 write_and_commit()。

文件组织：
  {repo_root}/{YYYY-MM}/{slug}.md
  按论文首次投稿的年月分文件夹，避免单目录文件过多。

Commit 时间：
  日期固定为论文 date 字段（首次投稿当天），时间在 08:00-21:30 内随机，
  通过 GIT_AUTHOR_DATE / GIT_COMMITTER_DATE 注入。

Markdown 结构：
  前半部分（英文 title + Abstract）：代码硬拼，不过模型
  后半部分（中文标题 + 解读正文）：reader_agent 的原始输出，直接追加
"""

import os
import re
import random
import subprocess
from datetime import datetime
from pathlib import Path

# 仓库根目录（此文件所在目录）
REPO_ROOT = Path(__file__).parent


def make_slug(title: str) -> str:
    """
    将英文标题转换为合法的文件名（不含扩展名）。
    小写 → 去掉非字母数字字符 → 空格换连字符 → 合并连续连字符 → 截断 60 字符。
    """
    t = title.lower().strip()
    t = re.sub(r'[^\w\s-]', '', t)
    t = re.sub(r'[\s_]+', '-', t)
    t = re.sub(r'-+', '-', t)
    return t[:60].rstrip('-')


def _build_prefix(paper: dict) -> str:
    """
    生成 Markdown 文件的硬拼前缀：英文标题 + 元信息 + 英文摘要 + 分隔线。
    这部分直接从 papers.json 字段拼接，不经过模型。
    """
    arxiv_id = paper.get('arxiv_id', '').strip()
    date     = paper.get('date', '').strip()
    abstract = paper.get('abstract', '').strip()
    title    = paper.get('title', '').strip()

    meta = [f"> **Date**：{date}"] if date else []
    if arxiv_id:
        meta.append(f"> **arXiv**：https://arxiv.org/abs/{arxiv_id}")

    lines = [f"# {title}", ""]
    if meta:
        lines += meta + [""]
    lines += ["## Abstract", "", abstract, "", "---", "", ""]

    return '\n'.join(lines)


def _commit_datetime(date_str: str) -> str:
    """
    以论文发布日期为基准，随机选取当天 08:00-21:30 的时间。
    返回 ISO 8601 格式字符串，供 GIT_AUTHOR_DATE 使用。
    若日期字符串无效，退化为当前时间。
    """
    try:
        base = datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        base = datetime.now()

    hour   = random.randint(8, 21)
    minute = random.choice([0, 30])
    return base.replace(hour=hour, minute=minute, second=0).strftime('%Y-%m-%dT%H:%M:%S')


def write_and_commit(paper: dict, model_output: str) -> Path:
    """
    将论文解读写入对应月份目录，并以论文发布日期提交 git commit。

    参数
    ─────
    paper        : papers.json 中的一条记录，需含 title / arxiv_id / abstract / date
    model_output : reader_agent 返回的字符串，第一行为 # 中文标题 论文详细解读

    返回
    ─────
    写入的文件绝对路径
    """
    date_str = paper.get('date', '')
    folder   = date_str[:7] if len(date_str) >= 7 else 'unknown'
    out_dir  = REPO_ROOT / "papers" / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # 拼接最终 Markdown：硬拼前缀 + 模型输出
    slug     = make_slug(paper['title']) + '.md'
    out_path = out_dir / slug
    content  = _build_prefix(paper) + model_output
    out_path.write_text(content, encoding='utf-8')

    # git add 只加这一个文件，绝不用 git add .
    rel_path = out_path.relative_to(REPO_ROOT)
    subprocess.run(
        ['git', 'add', str(rel_path)],
        cwd=REPO_ROOT, check=True,
    )

    # commit 消息
    title_short = paper['title'][:60]
    msg = f"feat: {title_short}\n\nDate: {date_str or 'unknown'}"

    # 注入发布当天的随机时间
    commit_dt = _commit_datetime(date_str)
    env = {**os.environ,
           'GIT_AUTHOR_DATE':    commit_dt,
           'GIT_COMMITTER_DATE': commit_dt}

    subprocess.run(
        ['git', 'commit', '-m', msg],
        cwd=REPO_ROOT, env=env, check=True,
        capture_output=True,  # 不把 git 输出混入进度条
    )

    return out_path
