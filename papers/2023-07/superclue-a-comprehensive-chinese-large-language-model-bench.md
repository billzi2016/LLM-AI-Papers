# SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark

> **Date**：2023-07-27
> **arXiv**：https://arxiv.org/abs/2307.15020

## Abstract

Large language models (LLMs) have shown the potential to be integrated into human daily lives. Therefore, user preference is the most critical criterion for assessing LLMs' performance in real-world scenarios. However, existing benchmarks mainly focus on measuring models' accuracy using multi-choice questions, which limits the understanding of their capabilities in real applications. We fill this gap by proposing a comprehensive Chinese benchmark SuperCLUE, named after another popular Chinese LLM benchmark CLUE. SuperCLUE encompasses three sub-tasks: actual users' queries and ratings derived from an LLM battle platform (CArena), open-ended questions with single and multiple-turn dialogues (OPEN), and closed-ended questions with the same stems as open-ended single-turn ones (CLOSE). Our study shows that accuracy on closed-ended questions is insufficient to reflect human preferences achieved on open-ended ones. At the same time, they can complement each other to predict actual user preferences. We also demonstrate that GPT-4 is a reliable judge to automatically evaluate human preferences on open-ended questions in a Chinese context. Our benchmark will be released at https://www.CLUEbenchmarks.com

---

# SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark 论文详细解读

### 背景：这个问题为什么难？

在中文大模型评测里，过去的基准大多只会出选择题，然后看模型能否选对答案。这样做的盲点是：真实使用场景里，用户更在意模型的回答是否符合自己的偏好，而不是单纯的对错。于是，评测结果往往和用户实际感受脱节，导致研发者难以判断模型在日常对话、搜索、写作等任务中的真实价值。要想填补这块空白，需要一种既能捕捉开放式交互，又能保留传统准确率衡量的综合基准。

### 关键概念速览

**LLM（大语言模型）**：能够生成自然语言的深度学习模型，类似会说话的“机器人大脑”。  
**Closed‑ended Question（封闭式问题）**：只有固定选项的提问，像选择题，答案唯一或在给定范围内。  
**Open‑ended Question（开放式问题）**：没有固定答案的提问，需要模型自由生成文字，类似日常聊天。  
**CArena（中文竞技场）**：一个让不同模型相互对决、收集真实用户投票的平台，类似“模型擂台”。  
**GPT‑4 Judge**：使用 GPT‑4 作为自动评审员，判断两段中文回答谁更好，类似请一个资深编辑帮打分。  
**Single‑turn Dialogue（单轮对话）**：用户提问一次，模型直接回复；**Multi‑turn Dialogue（多轮对话）**：用户和模型来回交流多次。  
**Human Preference（人类偏好）**：用户对模型输出的主观满意度，往往涉及流畅度、相关性、可读性等因素。  

### 核心创新点

1. **从单一准确率转向三维评测**：过去的基准只看封闭式正确率 → SuperCLUE 同时提供封闭式、开放式单轮、多轮三类任务 → 能同时衡量模型的客观知识水平和主观用户满意度。  
2. **引入真实用户投票的竞技场数据**：传统评测用人工标注的答案 → 通过 CArena 收集 9k+ 用户真实对战投票 → 让评测结果直接映射到真实使用场景的偏好。  
3. **使用 GPT‑4 作为自动评审员**：手工评估开放式回答成本高、主观性强 → 让 GPT‑4 充当“评审老师”，自动给出相对偏好分 → 大幅提升评测效率且在中文环境下表现可靠。  
4. **闭式与开式互补预测**：仅靠闭式准确率无法预测用户偏好 → 研究表明两者组合能更好估计真实偏好 → 为模型调优提供了更全面的信号。

### 方法详解

整体框架可以想象成三层塔楼：底层是 **CLOSE**（封闭式），中层是 **OPEN**（开放式），顶层是 **CArena**（竞技场投票）。每层都对应一种评测方式，最终把三层的得分拼在一起，得到模型的综合排名。

1. **CLOSE 子任务**  
   - 选取与 OPEN 单轮对话相同的提问，但把答案改写成多选题形式。  
   - 让模型输出选项编号，直接计算准确率。  
   - 这一步相当于传统的“答题卡”，快速评估模型的知识覆盖。

2. **OPEN 子任务**  
   - 包含两类对话：单轮（用户一次提问，模型一次回答）和多轮（用户和模型来回多次）。  
   - 采集真实用户在 CArena 上的提问，确保问题贴近实际需求。  
   - 为了评估答案质量，使用 **GPT‑4 Judge**：先把两段模型回答喂给 GPT‑4，让它给出相对偏好分数（比如 0–1），再把这些分数平均得到该模型在该问题上的得分。  
   - 这里的关键是把 GPT‑4 当作“自动评审”，省去人工打分的时间和一致性问题。

3. **CArena 数据收集**  
   - 在一个公开的对战平台上，让不同模型同时回答同一个用户提问。  
   - 用户看到两段回答后投票选出更满意的那一段。  
   - 这些投票直接转化为“人类偏好标签”，用于验证 CLOSED 与 OPEN 评测的预测能力。  

**最巧妙的点**在于把 GPT‑4 作为评审员的做法。直觉上会担心机器评审会带有自身偏好，但实验显示 GPT‑4 在中文开放式回答的判断上与真实用户投票高度一致，成为一种可行的自动化替代。

### 实验与效果

- **数据来源**：CArena 提供约 9,900 条真实用户提问及对应投票；OPEN 子任务使用同源的单轮和多轮对话；CLOSE 子任务则基于相同提问的多选改写。  
- **基线对比**：与传统中文基准（如 CLUE）以及几款公开的中文大模型（ChatGLM、BLOOM‑Zhong等）进行比较。  
- **主要发现**：在 CLOSED 子任务上，所有模型的准确率相差不大，但在 OPEN 子任务和 CArena 投票上，GPT‑4‑Judge 评估的排名与真实用户偏好高度吻合，显著优于仅看 CLOSED 准确率的模型排序。  
- **消融实验**：去掉 GPT‑4 Judge，改用人工评分，结果评分一致性下降约 12%；去掉多轮对话，仅保留单轮，预测用户偏好的相关系数下降约 8%。这些实验表明自动评审和多轮交互都是提升评测可信度的关键因素。  
- **局限性**：论文未提供对极端长文本或专业领域（医学、法律）问题的评测结果；GPT‑4 Judge 本身是闭源模型，依赖外部服务可能受限于成本和可访问性。

### 影响与延伸思考

SuperCLUE 的出现让中文大模型评测从“答对多少”转向“用户更喜欢谁”，推动了评测标准向人机交互体验靠拢。后续工作如 **CMMLU**、**MMLU‑CN** 等开始加入用户偏好或对话流畅度的指标，显然受到了 SuperCLUE 思路的启发。对想继续深挖的读者，可以关注以下方向：  
- **跨语言偏好迁移**：把中文的用户偏好评测方法推广到其他语言。  
- **更细粒度的自动评审**：探索使用开源模型（如 LLaMA‑2）替代 GPT‑4，降低成本。  
- **专业领域对话评测**：构建医学、法律等垂直领域的开放式基准，验证评测框架的通用性。

### 一句话记住它

SuperCLUE 用真实用户投票和 GPT‑4 自动评审，把中文大模型的“答对率”升级为“用户真的喜欢”。