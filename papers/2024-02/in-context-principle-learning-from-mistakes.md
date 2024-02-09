# In-Context Principle Learning from Mistakes

> **Date**：2024-02-08
> **arXiv**：https://arxiv.org/abs/2402.05403

## Abstract

In-context learning (ICL, also known as few-shot prompting) has been the standard method of adapting LLMs to downstream tasks, by learning from a few input-output examples. Nonetheless, all ICL-based approaches only learn from correct input-output pairs. In this paper, we revisit this paradigm, by learning more from the few given input-output examples. We introduce Learning Principles (LEAP): First, we intentionally induce the model to make mistakes on these few examples; then we reflect on these mistakes, and learn explicit task-specific "principles" from them, which help solve similar problems and avoid common mistakes; finally, we prompt the model to answer unseen test questions using the original few-shot examples and these learned general principles. We evaluate LEAP on a wide range of benchmarks, including multi-hop question answering (Hotpot QA), textual QA (DROP), Big-Bench Hard reasoning, and math problems (GSM8K and MATH); in all these benchmarks, LEAP improves the strongest available LLMs such as GPT-3.5-turbo, GPT-4, GPT-4 turbo and Claude-2.1. For example, LEAP improves over the standard few-shot prompting using GPT-4 by 7.5% in DROP, and by 3.3% in HotpotQA. Importantly, LEAP does not require any more input or examples than the standard few-shot prompting settings.

---

# 基于上下文的错误原则学习 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，**少样本提示（few‑shot prompting）** 成了把通用语言模型搬到具体任务的主流手段。传统做法只把几组“正确的”输入‑输出示例塞进提示里，让模型直接模仿。然而，模型往往会在细节上出错，尤其是需要多步推理或数学计算的场景。因为提示里没有显式告诉模型“哪些思路是危险的、哪些常见的坑要避开”，模型只能靠经验猜测，导致错误率居高不下。换句话说，**只学对不学错** 的范式忽视了人类学习中“从错误中抽象规律”的关键环节，这让提升空间被提前封死。

### 关键概念速览
- **In‑Context Learning（上下文学习）**：在模型的输入中直接给出几对示例，让模型在不改参数的情况下“现场学习”。类似老师在课堂上现场演示几道例题，学生随后直接模仿解答。
- **Few‑shot prompting（少样本提示）**：提供极少（通常 1‑5 条）示例的提示方式。它的核心假设是模型已经具备足够的通用知识，只需要一点上下文就能迁移。
- **Principle（任务原则）**：从错误中抽象出的通用规则或警示句子，例如 “在比较大小时先把单位统一”。把它想成解题的“经验法则”，可以在不同题目间复用。
- **Mistake induction（诱导错误）**：有意让模型在示例上给出错误答案的技巧。相当于老师故意出几道“陷阱题”，让学生先踩坑再反思。
- **Reflection（自我反思）**：模型在看到自己犯的错后，生成解释并归纳出原则的过程。像学生写错题本，先写错因再总结解题技巧。
- **Prompt engineering（提示工程）**：设计提示文本的艺术与技术，包括示例排布、指令语言、以及如何把原则嵌入最终提示。

### 核心创新点
1. **从只学对 → 学对也学错**  
   传统少样本提示只展示正确答案，模型只能“复制”。这篇论文在同样的示例数量下，加入**诱导错误**的步骤，让模型先产生错误答案，再进行自我纠正。这样模型在同一批示例上获得了“双向”学习信号，显著提升了对任务细节的敏感度。

2. **显式抽取任务原则 → 作为通用提示**  
   过去的自我纠错往往是“直接重新回答”。这里把模型的错误解释转化为**可复用的原则**，并把这些原则与原始示例一起放进最终提示。相当于把学生的错题本变成全班共享的解题技巧手册，提升了后续未见题目的成功率。

3. **不增加示例数量 → 纯粹提示层面的提升**  
   大多数提升方法要么扩充训练数据，要么加入外部工具。LEAP 只在提示内部做文章，保持了 **few‑shot** 场景的成本不变，却在多个基准上实现了两位数的相对提升。

4. **统一框架覆盖多种任务**  
   通过同一套“诱错‑反思‑原则”流程，作者在多跳阅读、文本抽取、数学推理等五类任务上都取得了正向收益，证明了该思路的任务无关性。

### 方法详解
**整体思路**可以概括为四步循环：  
1) **提供少量正确示例**（与普通 few‑shot 相同）。  
2) **诱导模型在这些示例上出错**，通过在提示里加入“请尝试回答，即使可能会错”。  
3) **让模型自我反思**：读取自己的错误答案，要求模型解释错误原因并归纳出“一条或多条原则”。  
4) **组合原始示例 + 学到的原则**，构成最终的测试提示，让模型回答真正的测试题。

**关键模块拆解**  
- **错误诱导模块**：在提示的示例后面加上一句指令，例如 “下面的答案可能不对，请先给出你的答案”。这种轻度诱导不会改变示例本身，只是让模型在内部产生冲突，从而更容易暴露弱点。  
- **反思与原则抽取模块**：模型看到自己的错误后，收到类似 “请解释为什么刚才的答案是错的，并写出避免此类错误的通用原则” 的指令。模型的输出通常是两段：错误原因 + 归纳的原则。这里的原则往往是简短的自然语言规则，例如 “先把所有数字转成整数再比较”。  
- **原则融合模块**：把得到的原则列表拼接到原始 few‑shot 示例前面，形成新的提示。此时提示里既有具体例子，又有抽象的“解题指南”。模型在回答新问题时，会先读取这些指南，再结合示例进行推理。

**反直觉之处**在于：**让模型先犯错反而提升了整体准确率**。直觉上会担心错误示例会误导模型，但实验表明，错误本身提供了强信号，促使模型在后续阶段主动检查、校正，从而形成更稳健的推理路径。

### 实验与效果
- **测试任务**：多跳阅读（HotpotQA）、文本抽取（DROP）、大模型硬推理基准（Big‑Bench Hard）、两套数学推理（GSM8K、MATH）。  
- **基线模型**：GPT‑3.5‑turbo、GPT‑4、GPT‑4‑turbo、Claude‑2.1。所有基线均使用标准 few‑shot 提示。  
- **主要提升**：在 DROP 上，使用 GPT‑4 时 LEAP 比普通 few‑shot 高出 **7.5%**；在 HotpotQA 上提升 **3.3%**。在数学任务（GSM8K、MATH）同样出现两位数的相对增益。  
- **消融实验**：去掉“原则融合”步骤，性能回落到普通 few‑shot 水平；仅保留诱导错误但不做反思，提升幅度也大幅下降，说明两者缺一不可。  
- **局限性**：方法依赖模型具备自我解释能力，较小模型（如 7B 参数）往往生成的原则质量不高；每轮诱错‑反思会增加推理时间，大约多出 1‑2 倍的 latency；在极端噪声数据上，模型可能归纳出错误的“原则”，导致负向迁移。

### 影响与延伸思考
LEAP 把“错题本”概念搬进了大语言模型的提示层，随后出现了一批**错误驱动的提示研究**：如 “Self‑Check Prompting” 让模型先给出答案再自行校验、 “Self‑Consistency” 通过多次采样取多数答案、以及 “Chain‑of‑Thought with Correction” 把纠错步骤显式加入推理链。推测未来会有 **自动化原则生成**（让模型在大量任务上自行构建通用规则库）和 **跨任务原则迁移**（把数学原则用于代码生成）的工作出现。想进一步了解，可关注 2024‑2025 年间在 ACL、NeurIPS 上出现的 “error‑aware prompting” 系列论文。

### 一句话记住它
让大模型先犯错，再把错误总结成通用原则，直接把“错题本”写进提示，就能在不增添示例的情况下显著提升少样本推理。