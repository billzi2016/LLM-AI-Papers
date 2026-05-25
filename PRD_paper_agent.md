# PRD：自动化论文精读 AI Agent

## 1. 项目背景与目标

**背景**：AI 领域论文产出速度远超人工阅读能力，研究者难以实时追踪前沿进展。

**目标**：构建一套自动化论文精读 Agent，将 `papers.json` 中的论文逐篇转化为结构清晰、深入浅出的中文解读文章，并按论文发表时间自动提交到 GitHub，实现**高质量学术舆情的持续、实时、低人工干预的追踪与沉淀**。

**核心价值**：
- 科研人员：快速了解论文核心贡献和方法，无需逐字阅读原文
- 入门者：遇到专业术语自动解释，零门槛理解前沿研究
- 团队：形成可检索的知识库，积累科研资产

---

## 2. 系统架构

```
papers.json
    │
    ▼
[reader_agent.py]  ←→  负载均衡双 Ollama API
    │                   localhost:11434  /  10.54.79.131:11434
    │   为每篇论文生成解读 Markdown
    ▼
[committer.py]
    │   按 date 字段分文件夹存放，用论文发布日期提交 git commit
    ▼
GitHub 仓库（按月份目录组织）
  2024-01/
    attention-is-all-you-need.md
    ...
  2024-02/
    ...
```

---

## 3. 模块详细设计

### 3.1 `reader_agent.py`（核心 Agent）

#### 3.1.1 输入输出

| 字段 | 来源 | 用途 |
|------|------|------|
| `title` | papers.json | 作为文章标题，也传给模型帮助理解 |
| `abstract` | papers.json（来自 arxiv） | 传给模型，作为最权威的论文描述 |
| `summary` | papers.json（来自 raw.txt，博客数据） | **不放入最终 Markdown**（版权风险），但传给模型作参考辅助理解 |
| `date` | papers.json（首次投稿日期） | 决定输出目录和 git commit 时间 |

**说明**：`summary` 来源于互联网博客，有版权风险，仅用于辅助模型理解，不写入发布文件。

#### 3.1.2 负载均衡策略

使用**轮询 + 阻塞式**负载均衡（而非并发），原因：
- `gpt-oss:120b` 是单卡大模型，并发请求会排队，反而变慢
- 阻塞式确保一个请求返回后再发下一个，避免模型侧积压
- 两个 API 端点轮流使用，均摊负载

```python
OLLAMA_ENDPOINTS = [
    "http://localhost:11434/api/generate",
    "http://10.54.79.131:11434/api/generate",
]
# 每次调用后 index = (index + 1) % 2，交替使用
```

若某端点连续失败 N 次，自动降级到另一个端点（故障转移）。

#### 3.1.3 输出 Markdown 结构

最终 Markdown = **硬拼前缀** + **模型原始输出**，无需解析。

```markdown
# {英文 title（papers.json 原文）}

> **Date**：{date}  
> **arXiv**：https://arxiv.org/abs/{arxiv_id}

## Abstract

{英文 abstract 原文，papers.json 原文，不做任何修改}

---

{← 以上为代码硬拼，以下直接追加模型返回的字符串 →}

# {中文 title} 论文详细解读

{模型输出的中文解读正文...}
```

**说明**：
- 前半部分（英文 title / Abstract）：纯字符串拼接，不过模型
- 后半部分：Prompt 要求模型第一行输出 `# {中文标题} 论文详细解读`，后面跟正文；代码拿到模型返回字符串直接追加，无需任何解析
- `summary` 不出现在文件中（版权风险）

#### 3.1.4 Prompt 设计（核心）

```
你是一位 AI 研究领域的资深科普作者，正在为「有编程基础但刚入门 AI 研究」的读者写一篇论文解读。

请根据下面提供的论文标题、摘要（abstract）和参考描述（仅供你理解，不要照抄）来撰写解读。

---
【论文标题】
{title}

【摘要（abstract，权威来源，请优先参考）】
{abstract}

【参考描述（来自博客，仅供理解参考，不得照抄）】
{summary}
---

**输出格式要求（必须严格遵守）**：
第一行必须是：# {论文中文译名} 论文详细解读
然后空一行，再按下面的结构写正文。
不要输出任何其他前缀、解释或额外文字，直接从第一行的标题开始。

正文结构如下，每个部分都要言之有物，不要写废话和套话：

### 这篇论文解决了什么问题？
用 2-3 句话说清楚：在它出现之前，这个领域遇到了什么困难或者局限？读者需要理解"它为什么有必要存在"。

### 关键概念速览
列举本文涉及的 3-6 个核心术语，每个术语用 1-2 句白话解释清楚（不要假设读者知道这些词）。
格式：**术语名**：解释。
示例：**CoT（思维链）**：让模型在给出答案之前先把推理步骤写出来，就像人解数学题时打草稿一样，能显著提升复杂问题的准确率。

### 核心创新点
这篇论文和之前的方法相比，究竟做了什么新的事情？要具体，不要说"提出了一种新方法"这种废话。
用"之前的方法 → 本文的做法 → 带来的改变"这个逻辑来写，1-3 个创新点。

### 方法怎么做的？
用类比或流程图（文字版）把方法讲清楚，让一个没读过原文的人能理解这篇论文的核心操作步骤。
不需要面面俱到，抓住最关键的 1-2 个设计就好。

### 效果如何？
这篇论文在什么任务上验证了效果？达到了什么数字？和谁对比？用具体数据说话，没有数据就说"论文声称……"。

### 一句话总结
如果只能记住这篇论文的一件事，应该是什么？写成一句话，像推特简介一样简洁有力。

---
写作要求：
- 语言：中文，口语化但不失准确，像讲给朋友听一样
- 专业术语首次出现时必须解释，之后可以直接使用
- 禁止出现：「本文」「综上所述」「值得注意的是」「不难看出」「显而易见」等 AI 八股文
- 禁止照抄参考描述的内容，那是别人写的有版权风险
- 如果摘要信息不足以支撑某个部分，坦诚说"原文未详细描述"，不要编造
- 总长度控制在 600-1000 字，宁可少写也不要凑字数
```

#### 3.1.5 进度管理

- 进度文件：`.reader_progress.json`，结构 `{"done": ["title_hash1", ...], "papers": [...]}`
- 每篇完成后立即写入进度，支持 Ctrl+C 后续跑
- title_hash 用 `md5(title)[:8]` 生成，避免路径问题

---

### 3.2 `committer.py`（Git 提交器）

#### 3.2.1 文件组织

```
{repo_root}/
  2024-01/
    {slug}.md          # slug = title 转小写，空格换连字符，去掉特殊符号，截断 60 字符
  2024-02/
    ...
  2025-03/
    ...
```

slug 生成规则：
```python
import re
def make_slug(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r'[^\w\s-]', '', t)     # 去掉特殊符号
    t = re.sub(r'\s+', '-', t)          # 空格换连字符
    return t[:60].rstrip('-') + '.md'
```

#### 3.2.2 Commit 时间策略

- **基准时间**：使用论文的 `date` 字段（首次投稿日期）
- **随机偏移**：在基准日期 ±3 天内随机偏移，时间随机选 08:00-22:00 的整点或半点
- **目的**：让 commit 历史看起来更自然，不像批量脚本生成

```python
import random
from datetime import datetime, timedelta

def jitter_date(date_str: str) -> datetime:
    base = datetime.strptime(date_str, '%Y-%m-%d')
    offset_days = random.randint(-3, 3)
    offset_hours = random.choice(range(8, 22))
    offset_minutes = random.choice([0, 30])
    return base + timedelta(days=offset_days, hours=offset_hours, minutes=offset_minutes)
```

使用 `GIT_AUTHOR_DATE` 和 `GIT_COMMITTER_DATE` 环境变量覆盖 commit 时间：

```bash
GIT_AUTHOR_DATE="2024-01-13T14:30:00" \
GIT_COMMITTER_DATE="2024-01-13T14:30:00" \
git commit -m "feat: add paper reading - {title[:50]}"
```

#### 3.2.3 Commit 消息格式

```
feat: {title[:60]}

Date: {original_date}
```

简洁，不含 AI 标记，不含 Co-Author。

#### 3.2.4 批量提交防 reject 策略

GitHub 对单次 push 有限制（文件数量、大小）。策略：
- 每完成一篇就立即 `git add` + `git commit`（单独提交，历史干净）
- Push 策略：每提交 50 篇 push 一次，或运行结束时最终 push 一次
- 不允许在 `committer.py` 内自动 push，由用户手动决定何时 push

---

### 3.3 `main_agent.py`（主入口）

```
流程：
1. 读取 papers.json
2. 加载 .reader_progress.json（断点续跑）
3. 对每篇未处理的论文：
   a. 调用 reader_agent 生成解读（带重试，最多 3 次）
   b. committer 写入 md 文件并 git commit
   c. 保存进度
4. 全部完成，打印统计（总数、成功、失败、耗时）
```

---

## 4. 文件清单

| 文件 | 职责 |
|------|------|
| `reader_agent.py` | Ollama 调用、负载均衡、Prompt、解读生成、进度管理 |
| `committer.py` | Markdown 生成、文件写入、git commit（含时间偏移） |
| `main_agent.py` | 主入口，串联两个模块 |

以上文件均放在 `/Users/bizi/Desktop/GitHub/LLM-AI-Papers/data/` 下。

---

## 5. 非功能性要求

| 项目 | 要求 |
|------|------|
| 中断恢复 | Ctrl+C 后重新运行，跳过已完成论文 |
| 日志 | 实时打印进度（篇数/总数、当前标题、耗时、ETA） |
| 故障隔离 | 单篇失败不影响整体，跳过后继续，失败列表写入 `reader_failed.json` |
| 注释语言 | 全部中文 |
| 输出语言 | 解读文章为中文 |

---

## 6. 已确认事项

| 事项 | 决定 |
|------|------|
| Markdown 结构 | 英文 title + 英文 abstract（硬拼）+ 模型输出（含中文标题和正文） |
| 模型输出格式 | 第一行固定为 `# {中文标题} 论文详细解读`，代码直接追加，无需解析 |
| Push 时机 | 脚本只做 `git add` + `git commit`，**不自动 push**，由用户手动决定 |
| `arxiv_id` 字段 | `papers.json` 已包含，解读文章中拼入 arxiv 链接 |
| `summary` 处理 | 仅传给模型辅助理解，不写入 Markdown 文件 |

## 7. 待确认事项

1. **仓库路径**：md 文件提交到哪个 git 仓库？当前的 `LLM-AI-Papers` 还是另建一个？
