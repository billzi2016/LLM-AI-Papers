"""
工具模块：调用 Ollama 大模型，为单篇论文生成中文解读。
不要直接运行，从 main_agent.py 调用 generate_reading()。

输出约定：
  模型必须以 "# {中文标题} 论文详细解读" 开头，
  committer 拿到字符串后直接追加到 Markdown 前缀，无需任何解析。
"""

import re
from load_balancer import balancer

MODEL       = "gpt-oss:120b"
MAX_RETRIES = 3   # 格式不符或请求异常时最大重试次数

# ── Prompt ───────────────────────────────────────────────────────────────────
# 核心约束：
#   1. 第一行固定格式，方便代码直接拼接
#   2. 每个章节结构明确，避免模型输出泛泛而谈的废话
#   3. 禁止 AI 八股文套话，要求具体数据和类比解释
PROMPT = """你是一位 AI 研究领域的资深科普作者，正在为「有编程基础但刚入门 AI 研究」的读者写一篇论文解读。

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
第一行必须是：# {{论文中文译名}} 论文详细解读
然后空一行，再按下面的结构写正文。
不要输出任何其他前缀、解释或多余文字，直接从第一行的标题开始。

正文结构如下，每个部分都要言之有物，不要写废话和套话：

### 这篇论文解决了什么问题？
用 2-3 句话说清楚：在它出现之前，这个领域遇到了什么困难或者局限？读者需要理解「它为什么有必要存在」。

### 关键概念速览
列举本文涉及的 3-6 个核心术语，每个术语用 1-2 句白话解释清楚（不要假设读者知道这些词）。
格式：**术语名**：解释。
示例：**CoT（思维链）**：让模型在给出答案之前先把推理步骤写出来，就像人解数学题时打草稿一样，能显著提升复杂问题的准确率。

### 核心创新点
这篇论文和之前的方法相比，究竟做了什么新的事情？要具体，不要说「提出了一种新方法」这种废话。
用「之前的方法 → 本文的做法 → 带来的改变」这个逻辑来写，1-3 个创新点。

### 方法怎么做的？
用类比或流程图（文字版）把方法讲清楚，让一个没读过原文的人能理解这篇论文的核心操作步骤。
不需要面面俱到，抓住最关键的 1-2 个设计就好。

### 效果如何？
这篇论文在什么任务上验证了效果？达到了什么数字？和谁对比？用具体数据说话，没有数据就说「论文声称……」。

### 一句话总结
如果只能记住这篇论文的一件事，应该是什么？写成一句话，像推特简介一样简洁有力。

---
写作要求：
- 语言：中文，口语化但不失准确，像讲给朋友听一样
- 专业术语首次出现时必须解释，之后可以直接使用
- 禁止出现：「本文」「综上所述」「值得注意的是」「不难看出」「显而易见」等 AI 八股文
- 禁止照抄参考描述的内容，那是别人写的有版权风险
- 如果摘要信息不足以支撑某个部分，坦诚说「原文未详细描述」，不要编造
- 总长度控制在 600-1000 字，宁可少写也不要凑字数"""


def _strip_think(text: str) -> str:
    """去掉模型可能输出的 <think>…</think> 思考链，只保留正文。"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def generate_reading(paper: dict) -> str | None:
    """
    为一篇论文生成中文解读，返回模型原始输出字符串（以 # 标题行开头）。
    重试 MAX_RETRIES 次仍失败则返回 None。

    参数
    ─────
    paper : papers.json 中的一条记录，需含 title / abstract / summary
    """
    prompt = PROMPT.format(
        title    = paper.get('title',    ''),
        abstract = paper.get('abstract', ''),
        summary  = paper.get('summary',  ''),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = balancer.post(
                json={
                    "model":  MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,    # 略高温度让文章更自然流畅
                        "num_ctx":     131072,
                    },
                },
                timeout=600,
            )
            text = _strip_think(resp.json().get("response", ""))
            if text.startswith('#'):
                return text   # 格式正确，直接返回给 committer 拼接
            print(f"(重试{attempt}:输出未以#开头) ", end='', flush=True)
        except Exception as e:
            print(f"(重试{attempt}:{e}) ", end='', flush=True)

    return None
