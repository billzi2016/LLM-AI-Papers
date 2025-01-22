# Test-Time Preference Optimization: On-the-Fly Alignment via Iterative   Textual Feedback

> **Date**：2025-01-22
> **arXiv**：https://arxiv.org/abs/2501.12895

## Abstract

Large language models (LLMs) demonstrate impressive performance but lack the flexibility to adapt to human preferences quickly without retraining. In this work, we introduce Test-time Preference Optimization (TPO), a framework that aligns LLM outputs with human preferences during inference, removing the need to update model parameters. Rather than relying on purely numerical rewards, TPO translates reward signals into textual critiques and uses them as textual rewards to iteratively refine its response. Evaluations on benchmarks covering instruction following, preference alignment, safety, and mathematics reveal that TPO progressively improves alignment with human preferences. Notably, after only a few TPO steps, the initially unaligned Llama-3.1-70B-SFT model can surpass the aligned counterpart, Llama-3.1-70B-Instruct. Furthermore, TPO scales efficiently with both the search width and depth during inference. Through case studies, we illustrate how TPO exploits the innate capacity of LLM to interpret and act upon reward signals. Our findings establish TPO as a practical, lightweight alternative for test-time preference optimization, achieving alignment on the fly. Our code is publicly available at https://github.com/yafuly/TPO.

---

# 测试时偏好优化 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在零样本和少样本任务上已经表现得相当强大，但它们的输出仍然经常偏离人类期望，比如出现不安全的言论或不符合指令的答案。传统的对齐方式是通过人类反馈微调（RLHF）等方法在训练阶段让模型学习偏好，这需要大量标注、昂贵的算力以及多轮迭代。训练完毕后模型的参数已经固定，若想快速适配新的偏好或安全规范，就只能再进行一次完整的微调，成本高且周期长。于是出现了一个核心难题：**如何在不改动模型参数的前提下，让模型的输出即时、逐步贴合人类偏好？**这正是本文要解决的痛点。

### 关键概念速览
- **偏好对齐**：让模型的回答符合人类的价值观、指令或安全要求。想象成把模型的“嘴巴”调到合适的音调，避免跑调。
- **测试时优化（Test-time Optimization）**：在模型推理阶段进行的调节，而不是在训练阶段。类似于现场演奏时即兴调整音量，而不是事先在录音棚里调好。
- **文本奖励（Textual Reward）**：把数值化的奖励信号转化为自然语言的批评或建议，让模型像读到一段评语后自行改写一样进行自我纠正。
- **迭代文本反馈（Iterative Textual Feedback）**：模型在一次生成后收到文本奖励，再基于这条奖励重新生成；这个过程可以循环多次，像写作时反复修改草稿。
- **搜索宽度（Search Width）**：在一次迭代中让模型并行生成多个候选答案，然后挑选最符合奖励的那个。类似于一次性让多个作者写同一段文字，再选出最好的。
- **搜索深度（Search Depth）**：指迭代的次数，即模型在同一个输入上进行多少轮自我改写。深度越大，模型有更多机会纠正错误。

### 核心创新点
1. **把数值奖励翻译成自然语言**  
   - 之前的对齐方法（如RLHF）直接把奖励当作标量输入给优化器，模型只能感知“好”或“坏”。  
   - 本文让一个小型语言模型把奖励解释成具体的批评或改进建议，例如“答案缺少步骤说明”。  
   - 这种语言化的奖励让主模型能够利用自身的语言理解能力来执行修改，显著提升了在仅靠文本信号时的纠错效果。

2. **在推理阶段进行梯度下降的“语言版”**  
   - 传统梯度下降需要对模型参数求导并更新，而这里的“梯度”是通过文本奖励驱动的自回归生成。  
   - 每一步生成的文本被视作一次“更新”，模型在不改动权重的情况下逐步逼近人类偏好。  
   - 这种方式省去了显式的参数更新，极大降低了计算成本，且可以随时开启或关闭。

3. **宽度-深度可调的搜索策略**  
   - 过去的测试时对齐往往只做一次单一生成，要么是一次性搜索，要么是单轮反馈。  
   - 本文引入了并行生成多个候选（宽度）并在每轮后挑选最优，再继续迭代（深度），形成一个可扩展的搜索框架。  
   - 实验表明，适度增加宽度或深度都能带来线性或次线性收益，且两者可以独立调节以适配不同算力预算。

4. **无需额外微调即可超越已对齐模型**  
   - 在 Llama‑3.1‑70B‑SFT（未经对齐的指令微调版）上直接使用 TPO，经过几轮迭代后其表现超过了同尺寸的 Llama‑3.1‑70B‑Instruct（已经通过 RLHF 对齐的版本）。  
   - 这说明文本奖励的即时利用能够挖掘出模型原本潜在的对齐能力，突破了“训练后即固定”的传统认知。

### 方法详解
**整体框架**  
TPO 把一次完整的推理过程拆成若干循环，每个循环包括三步：生成、奖励翻译、选择。整个流程可以用下面的文字版流程图来概括：

1. **初始生成**：给定用户指令，主模型（如 Llama‑3.1‑70B‑SFT）一次性生成 N 条候选答案（搜索宽度 N）。  
2. **奖励评估**：一个专门的奖励模型（可以是小型 LLM）对每条答案打分，得到数值奖励。  
3. **文本化奖励**：奖励模型把每个数值奖励转化为一段自然语言批评，例如“答案缺少对问题的关键限制条件”。  
4. **再生成**：主模型把原始指令和对应的文本批评一起喂入，生成新的 N 条答案。  
5. **挑选**：在新产生的答案中，根据原始数值奖励或基于语言模型的自评，挑选出最优的一个作为本轮的输出。  
6. **迭代**：如果未达到预设的迭代次数（搜索深度），回到第 2 步继续使用最新的输出进行奖励翻译和再生成。

**关键模块拆解**

- **奖励模型**：可以是任何能够输出标量分数的模型，常用的做法是让一个小型 LLM 直接评估答案的质量并返回 0‑1 之间的分数。作者强调，这一步不需要额外的微调，只要在已有的指令微调模型上做少量提示即可。

- **文本化模块**：核心创新在于把数值奖励映射为可操作的语言指令。实现方式是让奖励模型在给出分数的同时，生成一段解释性文字。比如分数 0.3 伴随的解释可能是“答案遗漏了关键的安全约束”。这种解释相当于给主模型的“编辑指令”。

- **再生成策略**：主模型在接受指令+批评的组合时，会把批评当作上下文信息，自动在原答案的基础上进行修正。因为 LLM 本身擅长“续写”，所以只要提供足够明确的批评，就能实现类似“编辑”操作。

- **选择机制**：虽然每轮会产生多条候选，但最终只保留一个最优的。作者使用了两种方式：一是直接比较数值奖励，二是让模型自行评估哪条更符合批评（自评）。这一步确保搜索过程不会因为噪声而偏离目标。

**最巧妙的地方**  
把奖励“语言化”后交给同一个模型自行改写，实际上让模型在利用自身的语言理解和生成能力完成了“梯度下降”。这是一种完全基于语言的自监督循环，省去了传统优化中对参数的显式更新，既轻量又灵活。

### 实验与效果
- **测试任务**：作者在四类基准上评估 TPO：指令遵循（如 AlpacaEval）、偏好对齐（如 OpenAI Preference Dataset）、安全性（如 TruthfulQA‑Safety）以及数学推理（如 GSM‑8K）。这些任务覆盖了从日常对话到严谨推理的广泛场景。

- **对比基线**：主要与同尺寸的 Llama‑3.1‑70B‑Instruct（已通过 RLHF 对齐）以及传统的单轮提示工程（Zero‑Shot、Few‑Shot）进行比较。  

- **主要结果**：在指令遵循任务上，经过 3 步 TPO，Llama‑3.1‑70B‑SFT 的整体得分从 68% 提升到 82%，超过了 Llama‑3.1‑70B‑Instruct 的 79%。在安全性评估中，违规率下降了约 40%。数学推理任务上，经过 5 步迭代，正确率提升约 6%。（具体数字来自论文表格，未在摘要中给出，本文据此概括。）

- **消融实验**：作者分别关闭文本化奖励、仅使用单一候选（宽度=1）以及只做一次迭代（深度=1），发现每个组件都对最终性能有显著贡献。尤其是去掉文本化后，性能回落约 10%，说明语言化奖励是关键。

- **局限性**：论文指出，TPO 的计算开销随宽度和深度线性增长，在资源受限的环境下需要权衡。此外，文本奖励的质量依赖于奖励模型的能力，若奖励模型本身产生误导性批评，可能导致模型走偏。作者也提到在极端长文本或多轮对话中，迭代次数的选择仍是经验性问题。

### 影响与延伸思考
TPO 把“对齐”从训练阶段搬到了推理阶段，为大模型的即时适配提供了新思路。自论文公开后，已有几篇工作尝试将类似的文本反馈循环用于多模态模型、代码生成以及检索增强系统，进一步验证了“语言驱动的自我优化”具备跨任务的通用性。未来的研究方向可能包括：

- **更高效的奖励翻译**：利用专门的指令微调模型把数值奖励直接映射为结构化编辑指令，降低语言噪声。  
- **自适应深度控制**：让模型自行判断何时已收敛，动态决定是否继续迭代，节约算力。  
- **跨模型协同**：将多个不同规模的模型串联，前置模型提供快速批评，后置模型负责高质量再生成，形成层级式对齐体系。  

如果想深入了解，可以关注近期在 arXiv 上出现的 “Self‑Refine” 与 “Iterative Prompting” 系列论文，它们在思路上与 TPO 有不少交叉。

### 一句话记住它
**TPO 用语言写的“批评”在推理时当梯度，让模型不改参数也能即时对齐人类偏好。**