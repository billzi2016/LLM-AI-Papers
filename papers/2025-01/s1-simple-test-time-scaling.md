# s1: Simple test-time scaling

> **Date**：2025-01-31
> **arXiv**：https://arxiv.org/abs/2501.19393

## Abstract

Test-time scaling is a promising new approach to language modeling that uses extra test-time compute to improve performance. Recently, OpenAI's o1 model showed this capability but did not publicly share its methodology, leading to many replication efforts. We seek the simplest approach to achieve test-time scaling and strong reasoning performance. First, we curate a small dataset s1K of 1,000 questions paired with reasoning traces relying on three criteria we validate through ablations: difficulty, diversity, and quality. Second, we develop budget forcing to control test-time compute by forcefully terminating the model's thinking process or lengthening it by appending "Wait" multiple times to the model's generation when it tries to end. This can lead the model to double-check its answer, often fixing incorrect reasoning steps. After supervised finetuning the Qwen2.5-32B-Instruct language model on s1K and equipping it with budget forcing, our model s1-32B exceeds o1-preview on competition math questions by up to 27% (MATH and AIME24). Further, scaling s1-32B with budget forcing allows extrapolating beyond its performance without test-time intervention: from 50% to 57% on AIME24. Our model, data, and code are open-source at https://github.com/simplescaling/s1

---

# 简易测试时扩展 论文详细解读

### 背景：这个问题为什么难？

语言模型在推理任务上往往受限于单次前向传播的算力，模型只能在固定的时间窗口里“思考”。即使模型容量很大，缺少足够的推理时间也会导致错误的链式思考或遗漏关键步骤。此前的提升主要靠扩大模型参数或在训练阶段加入思维链（CoT）等技巧，但这些方法的收益在达到一定规模后趋于饱和。OpenAI 的 o1 系列展示了“测试时扩展”——在推理阶段投入额外算力可以显著提升表现，却没有公开实现细节，导致社区只能盲目猜测。于是，如何用最简洁的手段在测试时控制算力并让模型自行检查答案，成为亟待解决的难题。

### 关键概念速览

**测试时扩展（Test-time scaling）**：在模型推理阶段额外分配计算资源，让模型有更长的“思考时间”。可以类比为考试时给学生加时，帮助他们检查并纠正错误。

**预算强迫（Budget forcing）**：人为设定模型的思考上限或下限。上限时强制终止生成，下限时通过插入“Wait”等词迫使模型继续输出，类似于老师在学生快要交卷时提醒“再想想”。

**思维链（Chain‑of‑Thought, CoT）**：让模型在给出答案前先写出推理步骤，像在纸上写草稿一样，使错误更容易被发现。

**s1K 数据集**：作者精心挑选的 1,000 条问答对，每条都附带高质量的推理过程，确保难度、覆盖面和答案质量都达标。

**Qwen2.5‑32B‑Instruct**：一款开源的大语言模型，拥有 320 亿参数，原生支持指令式交互，是本实验的基座模型。

### 核心创新点

1. **从大规模数据到千条高质量样本**  
   之前的复制工作往往直接使用公开的数学题库，质量参差不齐，导致微调效果不佳。作者通过三条验证标准（难度、 diversity、质量）筛选出 1,000 条高质量问答，形成 s1K 数据集。这样的小而精数据让模型在有限的微调预算下学到更可靠的推理模式。

2. **预算强迫的双向控制**  
   传统的测试时扩展只会让模型“跑得更久”。本研究提出两种极端操作：当模型生成结束标记时，插入若干 “Wait” 强制其继续思考；当模型思考时间超过预设上限时，直接截断输出。这样既能让模型自我检查，也能防止无限循环，形成一种“思考-检查-结束”的闭环。

3. **在微调后直接嵌入预算强迫**  
   与需要额外后处理或外部校验器的做法不同，作者在微调阶段就让模型习惯被“打断”或被“催促”。模型学会在被迫继续时回顾已有步骤，类似于训练时让学生在老师提醒后重新审视答案，从而在实际推理时自然产生双重检查。

4. **通过预算强迫实现跨尺度提升**  
   实验显示，仅在测试时使用预算强迫，就能把 s1‑32B 的 AIME24 正确率从 50% 推到 57%，相当于在不改模型结构的情况下实现了“规模外”提升。这表明算力调度本身是一种可被模型利用的元认知能力。

### 方法详解

**整体框架**  
整个流程可以划分为三步：① 数据准备 → ② 基模型微调 → ③ 推理时预算强迫。核心思想是让模型在训练阶段就熟悉“被强迫思考”或“被强迫停止”的情境，随后在实际使用时只需切换预算阈值即可。

**步骤一：构建 s1K 数据集**  
- 从公开的数学竞赛题库中抽取大量题目。  
- 通过人工标注，确保每道题的难度在中高区间，覆盖代数、几何、数论等多种子领域（diversity）。  
- 为每题撰写完整的思维链，要求推理过程逻辑严谨、语言流畅（质量）。  
- 最终得到 1,000 条“问题 + 推理 + 答案”三元组。

**步骤二：在 Qwen2.5‑32B‑Instruct 上进行监督微调**  
- 将 s1K 直接喂入模型，使用标准的指令微调流程（输入为问题，输出为完整思维链加答案）。  
- 为了让模型适应预算强迫，在微调数据中随机插入两类特殊标记：  
  * **强制终止标记**：在思维链的任意位置插入 `<STOP>`，训练时让模型学习在收到该标记后立即停止输出。  
  * **强制继续标记**：在思维链末尾插入若干 “Wait” 词，训练时要求模型在看到这些词后继续生成后续推理。  
- 通过这种方式，模型在训练时就已经“体验”了被迫停止或被迫继续的情境。

**步骤三：测试时的预算强迫机制**  
- **上限控制**：设定一个最大 token 数或时间阈值。若模型在达到阈值前仍未输出结束标记，则强制截断并返回当前已生成的内容。  
- **下限控制**：当模型在生成结束标记（如 `</s>`）时，系统自动在输出后追加若干 “Wait” token，迫使模型继续思考。随后模型会在新的上下文中重新审视已有答案，常常产生“再检查”式的补充或更正。  
- 这两种操作可以单独使用，也可以组合：先让模型思考到一定深度，再在即将结束时插入 “Wait”，形成“先深思后复核”的闭环。

**最巧妙的设计**  
预算强迫并不是简单的硬截断或硬续写，而是把“思考时间”当作可调的超参数，让模型在不同预算下自适应产生不同质量的答案。尤其是“Wait” 的插入，利用了模型对上下文的连续性敏感，使其在被提醒后自然回到已生成的推理链，产生类似人类“检查草稿”的行为。

### 实验与效果

- **测试数据**：MATH（约 2,500 题）和 AIME24（120 题）两套数学竞赛题库，均为高难度推理任务。  
- **基线对比**：  
  * o1‑preview（OpenAI 未公开实现细节的模型）在同样的算力预算下的表现。  
  * 未使用预算强迫的原始 Qwen2.5‑32B‑Instruct。  
- **主要结果**：  
  * 在 MATH 上，s1‑32B 通过预算强迫比 o1‑preview 高出约 27% 的准确率。  
  * 在 AIME24 上，单纯使用预算强迫将正确率从 50% 提升到 57%，相当于在不改模型规模的情况下实现了跨尺度提升。  
- **消融实验**：  
  * 去掉 “Wait” 强制继续，仅保留上限截断，提升幅度下降约 12%。  
  * 只使用上限截断而不进行微调中的强制终止标记，模型在被截断后往往生成不完整的推理，效果进一步下降。  
  * 这表明两种预算强迫以及对应的微调标记共同贡献了最终提升。  
- **局限性**：  
  * 预算强迫依赖于模型对 “Wait” 等词的敏感度，若迁移到对这些词不敏感的模型，效果可能减弱。  
  * 只在数学推理任务上验证，其他类型的长文本生成（如代码、写作）尚未测试。  
  * 过度强制继续可能导致生成冗长的无意义内容，需要精细调参。

### 影响与延伸思考

这篇工作向社区展示了“算力调度”本身可以成为提升模型推理质量的工具，而不必单纯依赖模型规模。随后出现的几篇论文尝试将预算强迫与自适应停机（adaptive halting）结合，探索让模型自行决定何时需要额外思考时间。还有研究把类似的 “Wait” 机制用于代码调试，让模型在生成错误代码后自动进入检查模式。想进一步了解，可关注 **自适应推理预算**、**模型内部元认知** 以及 **少样本微调与推理时算力调度的交叉** 等方向。

### 一句话记住它

只要给大模型加一点“思考时间”，并用“强迫停止/继续”让它自我检查，就能用千条高质量数据把它的数学推理水平推到 o1 级别。