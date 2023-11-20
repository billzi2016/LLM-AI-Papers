# GPQA: A Graduate-Level Google-Proof Q&A Benchmark

> **Date**：2023-11-20
> **arXiv**：https://arxiv.org/abs/2311.12022

## Abstract

We present GPQA, a challenging dataset of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry. We ensure that the questions are high-quality and extremely difficult: experts who have or are pursuing PhDs in the corresponding domains reach 65% accuracy (74% when discounting clear mistakes the experts identified in retrospect), while highly skilled non-expert validators only reach 34% accuracy, despite spending on average over 30 minutes with unrestricted access to the web (i.e., the questions are "Google-proof"). The questions are also difficult for state-of-the-art AI systems, with our strongest GPT-4 based baseline achieving 39% accuracy. If we are to use future AI systems to help us answer very hard questions, for example, when developing new scientific knowledge, we need to develop scalable oversight methods that enable humans to supervise their outputs, which may be difficult even if the supervisors are themselves skilled and knowledgeable. The difficulty of GPQA both for skilled non-experts and frontier AI systems should enable realistic scalable oversight experiments, which we hope can help devise ways for human experts to reliably get truthful information from AI systems that surpass human capabilities.

---

# GPQA：面向研究生水平的防谷歌检索问答基准 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理里，问答基准大多是面向大众或中学水平的，模型只要能检索到公开网页上的答案就能得高分。于是出现了“只要Google一下就能答对”的现象，导致评估结果并不能真实反映模型在真正科研场景下的推理能力。研究者们缺少一种能够逼迫模型走出“搜索‑复制”模式、必须进行深度学科推理的测评工具。正因为如此，构建一个既要专业又要防止简单网络搜索的高难度基准，成为了迫切需求。

### 关键概念速览
- **Graduate‑Level MCQ（研究生水平选择题）**：题目深度相当于硕士或博士课程的考核，需要跨章节、跨概念的综合理解，而不是记忆单一事实。  
- **Google‑Proof（防谷歌）**：即使给出无限的上网时间，普通人也很难凭借搜索找到正确答案，题目设计时会排除可直接检索的表述。  
- **Scalable Oversight（可扩展监督）**：在模型输出可能超出人类专家能力时，仍能让较弱的监督者有效检测并纠正错误的技术框架。  
- **Baseline LLM（基准大语言模型）**：指用于评估基准难度的现有最强模型，如 GPT‑4，通常会配合特定提示（prompt）进行测试。  
- **Human‑in‑the‑Loop（人机协同）**：让人类在模型推理过程中参与检查或引导的模式，是实现可扩展监督的核心思路。  
- **Difficulty Calibration（难度校准）**：通过让不同背景的评审者答题，量化题目真实难度并据此筛选进入最终数据集。  

### 核心创新点
1. **专家全程编写 → 采用拥有或正在攻读 PhD 的领域专家亲自出题**  
   过去的 QA 数据集多依赖众包或教材摘录，导致质量参差不齐。GPQA 让真正的科研人员从零设计题干、选项和正确答案，确保每道题都具备学术深度和严谨性。

2. **Google‑Proof 设计 → 验证题目在开放网络搜索下仍保持低可解率**  
   传统基准往往忽视搜索可得性，导致模型只要“上网”就能轻松拿分。GPQA 为每道题安排了非专家验证者，他们可以自由使用搜索引擎，平均耗时 30 分钟，却只能得到约 34% 的正确率，这直接证明了防谷歌的有效性。

3. **面向监督实验的基准定位 → 把题目难度作为人类监督可行性的压力测试**  
   作者把数据集的核心价值定位在“让弱监督者也能发现强模型错误”。这与以往单纯评估模型准确率的做法不同，提供了一个实验平台来探索可扩展监督的上限。

4. **系统化基准评估 → 用多种提示方式对 GPT‑4 进行测试，报告最高 39% 的准确率**  
   通过零提示、Few‑Shot、Chain‑of‑Thought 等多种策略，系统评估了当前最强大语言模型的上限，展示了即使是最前沿模型也难以突破研究生水平的壁垒。

### 方法详解
**整体思路**：先让领域专家独立出题 → 再让同领域的 PhD 级评审进行盲测 → 接着让非专家在无限制网络搜索条件下答题 → 最后用前沿大模型进行自动评估。整个流程像是“层层筛网”，每一步都在过滤掉容易被搜索或直觉猜到的题目，只留下真正需要深度推理的样本。

**关键步骤**：

1. **题目创作**  
   - 每位专家在自己熟悉的子领域（生物、物理、化学）撰写 10–20 道多选题。  
   - 题目结构统一为：题干、4–5 个选项、唯一正确答案以及简短的解释。  
   - 为防止直接检索，作者会刻意使用专业术语、交叉概念或实验情境，而不是直接引用教材中的定义。

2. **专家盲测（难度校准）**  
   - 另一批拥有 PhD 学位的专家在不知答案的情况下答题。  
   - 统计整体正确率：约 65%，排除事后发现的明显错误后提升至 74%。  
   - 这一步确保题目对真正的学科专家仍具挑战性。

3. **非专家 Google‑Proof 验证**  
   - 招募具备普通大学理科背景的验证者，允许他们使用任意搜索引擎、维基百科、教材等资源。  
   - 每题平均搜索时间 30 分钟，最终正确率仅 34%。  
   - 通过对比两组结果，作者确认了题目在“搜索可得性”上的低可解度。

4. **模型基准测试**  
   - 选取 GPT‑4 作为代表性大模型，使用三种提示策略：  
     a. **Zero‑Shot**：直接给出题目和选项。  
     b. **Few‑Shot**：提供 2–3 例相似的已解题目作为示例。  
     c. **Chain‑of‑Thought**：要求模型先写出推理过程再给出答案。  
   - 最高准确率 39%，显著低于人类专家，却略高于非专家验证者，说明模型在此类高阶推理上仍有明显差距。

**最巧妙的设计**：在非专家验证阶段，作者并没有限制搜索时间或资源，而是让验证者“自由上网”。这种“放水”式的设置让任何依赖外部信息的解法都难以提升准确率，从而真正检验模型是否具备内部推理能力。

### 实验与效果
- **数据规模**：448 道多选题，覆盖生物、物理、化学三大自然科学领域。  
- **人类表现**：PhD 级专家 65%（排除错误后 74%），普通非专家 34%（每题平均 30 分钟搜索）。  
- **模型表现**：GPT‑4 在最佳 Chain‑of‑Thought 提示下 39% 的准确率，仍低于非专家的 34% 但高于随机猜测（约 20%）。  
- **对比基线**：论文未提供其他大模型（如 Claude、Llama）对比，只列出 GPT‑4 作为最强基线。  
- **消融实验**：原文未详细报告各提示策略的单独贡献，只给出整体最高值。  
- **局限性**：题目数量相对有限（仅 448 条），且仅覆盖自然科学，尚未涉及工程、社会科学等领域；此外，非专家验证者的背景差异可能影响“Google‑Proof”评估的普适性。

### 影响与延伸思考
GPQA 把“防搜索”与“高阶学科推理”结合起来，为评估大模型在科研助理角色上的可靠性提供了新视角。自发布后，多个团队开始围绕“可扩展监督”设计实验，尝试让弱监督者（如普通科研助理）利用模型的中间推理过程进行错误检测。后续工作（如 **Self‑Check GPT**、**OversightBench**）在引用 GPQA 时，常把它当作检验人机协同系统鲁棒性的“压力锅”。如果想进一步探索，可关注以下方向：  
- 将基准扩展到跨学科或工程设计题目；  
- 开发自动化的“难度预测器”，在训练阶段就筛除易检索的样本；  
- 研究如何让模型在给出答案的同时生成可供人类审查的证据链（类似 Chain‑of‑Thought 的可解释化）。

### 一句话记住它
GPQA 用专家亲自出题、放水式搜索验证，打造了首个让强大语言模型也只能拿到 40% 左右正确率的研究生级防谷歌问答基准。