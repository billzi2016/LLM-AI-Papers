# Self-Refine: Iterative Refinement with Self-Feedback

> **Date**：2023-03-30
> **arXiv**：https://arxiv.org/abs/2303.17651

## Abstract

Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improving initial outputs from LLMs through iterative feedback and refinement. The main idea is to generate an initial output using an LLMs; then, the same LLMs provides feedback for its output and uses it to refine itself, iteratively. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner, and feedback provider. We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5, ChatGPT, and GPT-4) LLMs. Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, improving by ~20% absolute on average in task performance. Our work demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test time using our simple, standalone approach.

---

# 自我精炼：迭代自我反馈 论文详细解读

### 背景：这个问题为什么难？
生成式大语言模型（LLM）一次性输出往往会出现逻辑漏洞、细节遗漏或表达不够精准。传统做法是直接把模型的首轮答案交给用户，或者在训练阶段加入大量标注数据让模型学会更好地回答。但这些方法都有根本限制：①训练成本高，标注数据难以覆盖所有细分任务；②即使是最强的模型（如 GPT‑4）在推理时仍会出现“思路不连贯”或“算错一步”的情况；③现有的后处理手段（如检索增强、链式思考）需要额外的检索模块或专门的提示工程，增加系统复杂度。于是，如何在不改动模型本身、也不依赖外部数据的情况下，让模型在使用时自行提升答案质量，成为一个迫切的需求。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，像 ChatGPT、GPT‑4 那样可以完成对话、写作、推理等任务。  
**一次性生成**：模型在收到提示后直接输出最终答案，类似一次性写完作文。  
**自我反馈**：模型对自己刚刚输出的内容进行评价或批评，就像人写完段落后自己读一遍找错。  
**迭代精炼**：把自我反馈当作新提示，再让模型重新生成答案，循环多次直到满意，类似编辑稿件时的“写‑改‑写‑改”。  
**零监督（Zero‑Shot）**：不需要额外的标注数据或专门的微调，只凭模型已有的通用能力完成任务。  
**人类偏好评估**：让真实用户对不同版本的答案进行打分，判断哪一个更好，类似让读者投票选出最佳稿子。  
**自动指标**：使用 BLEU、ROUGE、Exact Match 等机器可计算的分数来衡量答案质量，提供客观对比。

### 核心创新点
1. **同一个模型兼任生成、评估、改写** → 过去的工作往往需要两个或更多模型（一个生成，一个评分）或额外的检索/推理模块。Self‑Refine 让同一个 LLM 先输出答案，再用同样的 LLM 生成对该答案的批评，最后再让它依据批评自行改写。 → 省去额外模型和数据，部署成本几乎为零，且可以直接在任何支持的 LLM 上复现。

2. **纯零监督的迭代循环** → 传统提升方法依赖有标签的强化学习（RLHF）或人类标注的对齐数据。Self‑Refine 完全不做任何额外训练，只在推理阶段循环几次。 → 让最前沿的商用模型（GPT‑4）在测试时仍有提升空间，突破了“模型已经最优” 的误区。

3. **统一的提示模板** → 通过精心设计的 Prompt，把“生成答案”“给出反馈”“依据反馈改写”三步包装成自然语言指令，模型无需内部修改即可理解任务切换。 → 这种提示工程的通用性使得方法可以快速迁移到对话、代码、数学等多种任务。

4. **实证验证跨任务提升约 20%** → 在 7 类任务上（对话、摘要、数学推理等）进行人类偏好和自动指标双重评估，Self‑Refine 的输出普遍优于单轮生成，平均提升约 20% 的绝对分数。 → 证明了迭代自我反馈的效果不是偶然，而是可复制的提升手段。

### 方法详解
整体思路可以概括为三步循环：**生成 → 反馈 → 精炼**，循环若干轮后输出最终答案。下面把每一步拆开讲。

1. **初始生成**  
   - 输入：用户的原始任务描述（例如“解释量子纠缠”）。  
   - Prompt：使用普通的指令式提示，让模型直接给出答案。  
   - 输出：第一版答案 A₀。

2. **自我反馈**  
   - 输入：把 A₀ 与原始任务一起喂回模型，提示模型“请评估上面的答案，指出不足之处”。  
   - 关键是让模型把自己当成审稿人，而不是继续写内容。  
   - 输出：一段批评文字 F₀，可能包括“缺少例子”“推理步骤不完整”“用词不够准确”等。

3. **基于反馈的精炼**  
   - 输入：把原始任务、A₀ 和 F₀ 合并，提示模型“请根据上述批评重新写一遍答案”。  
   - 模型在同一轮对话中读取自己的错误，然后生成改进版 A₁。  
   - 这一步相当于人类编辑稿件时把评审意见写进新稿。

4. **迭代**  
   - 将 A₁ 再送入反馈模块，得到 F₁，继续生成 A₂……  
   - 实际实验中作者发现 2~3 轮已经足够，更多轮往往收益递减甚至出现“过度修改”。  

**提示设计的巧妙之处**  
- 使用明确的角色指令（如“你是一个审稿人”）帮助模型切换视角。  
- 在反馈阶段加入“请用简洁的句子列出 3 条最关键的问题”，限制批评长度，防止模型跑题。  
- 精炼阶段的提示中明确要求“保留原有正确部分，只修改错误”，让模型更倾向于局部修改而不是全盘重写。

**算法层面的解释**  
虽然论文没有给出正式的数学公式，但可以把整个过程视作一个函数迭代：  
`A_{k+1} = Refine( Prompt, A_k, Feedback_k )`，其中 `Feedback_k = Critique( Prompt, A_k )`。  
这里的 `Critique` 和 `Refine` 都是同一个 LLM 的调用，只是提示不同。整个循环在推理时完成，不涉及梯度更新。

### 实验与效果
- **任务覆盖**：对话生成、问答、摘要、情感分类、代码生成、数学推理、事实检索共 7 类。  
- **基准模型**：GPT‑3.5、ChatGPT（基于 GPT‑3.5 的聊天版）以及 GPT‑4。  
- **对比对象**：同模型的单轮生成（即不使用 Self‑Refine），以及公开的几种强化学习或检索增强的基线（原文未给出全部名称）。  
- **提升幅度**：在人类偏好投票中，Self‑Refine 的答案平均比单轮答案多获得约 20% 的绝对偏好分；在自动指标上也出现相似的正向提升。  
- **消融实验**：作者分别去掉反馈环节、只做一次精炼、或把反馈改成随机噪声，结果显示没有反馈的版本几乎回到单轮水平，说明自我反馈是核心驱动力。  
- **局限性**：迭代次数过多会导致答案漂移或冗余；对极其开放式的创意任务（如诗歌）反馈质量不稳定；模型本身的错误如果在第一轮未被识别，后续循环也难以纠正。原文承认这些场景仍需进一步研究。

### 影响与延伸思考
Self‑Refine 把“自我审视”这一人类写作习惯搬进了 LLM 推理流程，打开了“推理时自我改进” 的新思路。随后的工作（如 *Iterative Self‑Correction*、*Self‑Consistency with Feedback*）纷纷在此基础上加入更复杂的信念融合或多模型投票，进一步提升鲁棒性。对想深入的读者，可以关注以下方向：① 如何让模型产生更可靠的批评（比如结合外部知识库）；② 多模态或代码任务中自我反馈的适配方式；③ 将自我迭代与强化学习结合，实现“在线学习”。这些都是当前研究的热点。

### 一句话记住它
让同一个大语言模型先自评再自改，几轮迭代即可在测试时显著提升答案质量，几乎不需要额外数据或模型。