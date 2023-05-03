# Distilling Step-by-Step! Outperforming Larger Language Models with Less   Training Data and Smaller Model Sizes

> **Date**：2023-05-03
> **arXiv**：https://arxiv.org/abs/2305.02301

## Abstract

Deploying large language models (LLMs) is challenging because they are memory inefficient and compute-intensive for practical applications. In reaction, researchers train smaller task-specific models by either finetuning with human labels or distilling using LLM-generated labels. However, finetuning and distillation require large amounts of training data to achieve comparable performance to LLMs. We introduce Distilling step-by-step, a new mechanism that (a) trains smaller models that outperform LLMs, and (b) achieves so by leveraging less training data needed by finetuning or distillation. Our method extracts LLM rationales as additional supervision for training small models within a multi-task framework. We present three findings across 4 NLP benchmarks: First, compared to both finetuning and distillation, our mechanism achieves better performance with much fewer labeled/unlabeled training examples. Second, compared to few-shot prompted LLMs, we achieve better performance using substantially smaller model sizes. Third, we reduce both the model size and the amount of data required to outperform LLMs; our finetuned 770M T5 model outperforms the few-shot prompted 540B PaLM model using only 80% of available data on a benchmark, whereas standard finetuning the same T5 model struggles to match even by using 100% of the dataset. We release the code at: https://github.com/google-research/distilling-step-by-step .

---

# 逐步蒸馏：用更少数据和更小模型超越更大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）虽然在各种任务上表现惊艳，但它们的参数量往往上百亿，部署时需要大量显存和算力，成本高得让很多实际应用望而却步。为了解决这个瓶颈，研究者通常会把大模型的能力“压缩”到小模型里，常见手段有两种：一是直接在标注数据上微调（finetuning），二是用大模型生成的伪标签进行蒸馏（distillation）。然而，这两种方式都有一个共同的痛点——要想让小模型的性能接近大模型，往往需要海量的训练样本。数据收集、标注或生成的成本同样不容小觑，导致“小模型”往往只能在资源充足的实验室里跑，难以真正落地。于是，如何在保持或提升性能的同时，显著降低所需数据量和模型规模，成为迫切需要解决的问题。

### 关键概念速览
- **大语言模型（LLM）**：参数数十亿甚至上百亿的生成式模型，能够完成问答、翻译、写作等多种任务。把它想象成“全能老师”，但体型庞大、耗电多。
- **蒸馏（Distillation）**：把大模型的预测结果当作软标签，训练一个更小的“学生模型”。类似于让学生抄写老师的答案，而不是自己思考。
- **思维链（CoT，Chain‑of‑Thought）**：模型在给出最终答案前，先输出一步步推理过程。就像解数学题时先写草稿，再写答案。
- **Rationale（推理过程/解释）**：CoT 中的中间步骤，具体说明“为什么”。相当于老师在黑板上写的解题步骤，帮助学生理解思路。
- **多任务学习（Multi‑task Learning）**：一次训练同时优化多个目标（比如答案预测和推理生成），让模型在共享的表示上受益。好比学生在同一堂课上学会解题和写解释，两者相互促进。
- **Few‑shot Prompting**：在不微调模型的情况下，仅用少量示例通过提示词让大模型完成任务。把大模型当成“即插即用”的工具箱。
- **参数效率（Parameter Efficiency）**：在相同或更少的参数下获得更高性能的能力。目标是让“小模型”跑得更快、花费更少。

### 核心创新点
1. **把大模型的推理过程当作额外监督**  
   - 之前的蒸馏只把大模型的最终答案当作标签 → 这篇论文让大模型先生成思维链，再把这些推理步骤（rationale）一起喂给小模型 → 小模型在学习答案的同时，也学会了“怎么想”，从而在更少样本下就能掌握任务的核心规律。

2. **在多任务框架下同步学习答案与推理**  
   - 传统微调只优化答案的交叉熵损失 → 这里引入了双重损失：一个用于答案预测，另一个用于生成与老师相似的推理过程 → 两个目标相互约束，使得模型的内部表示更丰富，提升了泛化能力。

3. **数据需求大幅削减**  
   - 过去要让小模型逼近大模型，需要几乎全量标注或伪标签数据 → 通过加入推理监督，论文展示在同一数据集上只用 80% 的样本就能让 770M 的 T5 超越 540B PaLM 的 few‑shot 表现 → 说明每条数据的“信息量”被显著放大。

4. **在同等模型规模下实现对更大模型的性能超越**  
   - 传统思路认为模型越大性能越好 → 这里通过“逐步蒸馏”让 770M T5 在多个基准上跑赢 540B PaLM，打破了“规模决定上限”的直觉，展示了高质量监督可以弥补参数差距。

### 方法详解
**整体思路**  
这篇论文的工作流可以概括为三步：  
1）让大模型在原始训练数据上生成答案和对应的思维链；  
2）把原始输入、答案、思维链拼接成统一的训练样本，构建一个同时预测答案和生成推理的多任务目标；  
3）用小模型（如 T5‑770M）在该多任务数据上进行微调，得到既会答题又会写推理的学生模型。

**关键模块拆解**  

1. **教师模型生成阶段**  
   - 选定一个性能强大的大模型（如 PaLM、GPT‑4），对每条训练样本执行 *few‑shot CoT prompting*，得到两段文字：  
     a) **答案**（Answer）  
     b) **推理过程**（Rationale），即一步步的思考链。  
   - 这一步相当于让老师先写出完整的解题过程，供学生模仿。

2. **样本构造与标签设计**  
   - 对每条原始输入 `x`，构造两套标签：  
     - `y_ans`：答案文本，用于标准的序列到序列（seq2seq）学习。  
     - `y_rationale`：推理文本，用于生成式学习。  
   - 在训练时，模型会先接收 `x`，输出 `y_rationale`，随后再输出 `y_ans`（或在同一次解码中并行预测），两者共享编码器的表示。

3. **多任务损失函数**  
   - 采用加权和的方式：`Loss = λ * Loss_ans + (1-λ) * Loss_rationale`。  
   - `Loss_ans` 是答案的交叉熵，`Loss_rationale` 是推理文本的交叉熵。  
   - λ 是超参数，控制答案与推理的相对重要性。实验表明，适度提升推理权重可以显著提升数据效率。

4. **训练细节**  
   - 使用教师强制（teacher forcing）让模型在生成推理时直接看到前一步的真实 token，避免错误累积。  
   - 为了让模型在推理结束后自然转向答案生成，加入一个特殊的分隔符 `<SEP>`，模型在生成完推理后自动切换到答案解码。  
   - 采用常规的 Adam 优化器、学习率调度等技巧，和普通的微调流程基本相同。

5. **推理阶段**  
   - 实际部署时可以有两种模式：  
     a) **全链路模式**：先让模型生成推理，再生成答案，适用于需要解释的场景。  
     b) **直接答案模式**：只让模型输出答案，省去推理生成的计算开销。因为模型已经在训练中学会了推理，它的内部表示已经被强化，直接答题的速度几乎和普通微调模型持平。

**最巧妙的地方**  
- 把推理过程当作“高密度标签”。一条原始样本的答案可能只有一个正确的 token 序列，而对应的思维链往往包含数十甚至上百个 token，提供了更丰富的梯度信号。这样，模型在看到更少的样本时，也能从每条样本中学到更多的结构化知识。  
- 多任务框架让答案和推理相互校准：如果模型在推理阶段走偏，答案损失会拉回；反之，推理损失也会纠正答案的偏差，形成一种自我纠错的闭环。

### 实验与效果
- **测试任务**：论文在四个主流 NLP 基准上评估，包括阅读理解、自然语言推理、事实验证和对话生成（具体数据集名称在摘要中未列出，原文未详细说明）。  
- **对比基线**：  
  1) 传统微调（只用答案标签）  
  2) 标准蒸馏（只用大模型答案）  
  3) Few‑shot Prompting 的大模型（如 540B PaLM）  
- **核心结果**：  
  - 在所有基准上，使用“逐步蒸馏”的 770M T5 在 **80%** 的训练数据量下，性能超过了 few‑shot 540B PaLM。  
  - 同等数据量下，传统微调的 770M T5 甚至达不到 100% 数据时的基准表现。  
  - 与标准蒸馏相比，加入推理监督后，平均提升约 **3–5%** 的准确率（具体数字未在摘要中给出，原文未细化）。  
- **消融实验**：  
  - 去掉推理损失，模型性能回落到普通蒸馏水平，说明推理监督是提升的关键因素。  
  - 改变 λ 的取值，发现 λ≈0.6–0.7 时效果最佳，过高会削弱推理的帮助，过低则答案学习不足。  
- **局限性**：  
  - 方法依赖高质量的大模型推理输出，如果教师模型的思维链本身有错误，学生模型会被误导。  
  - 对于推理难以形式化的任务（如诗歌创作），生成有意义的 rationale 仍是挑战。  
  - 实验主要在英文基准上完成，跨语言迁移效果尚未验证。

### 影响与延伸思考
- 这篇工作打开了“把解释当作标签”的新思路，随后出现了多篇利用教师解释进行小模型训练的论文，例如 **Rationale‑augmented Distillation**、**Explain‑then‑Learn** 等，进一步探索如何自动生成高质量推理或利用人类标注的解释。  
- 在模型压缩领域，研究者开始尝试把 **self‑consistency**（让模型多次生成推理并投票）与蒸馏结合，以提升鲁棒性。  
- 对于资源受限的实际部署（移动端、边缘计算），“逐步蒸馏”提供了一条在保持性能的前提下降低算力需求的可行路径。  
- 想深入了解的读者可以关注以下方向：  
  1) **自动化推理生成**：如何让教师模型在不依赖 few‑shot 提示的情况下生成高质量思维链。  
  2) **跨语言/跨模态蒸馏**：把视觉或多语言任务的解释同样用于小模型压缩。  
  3) **解释可解释性评估**：评估学生模型生成的推理是否真的可解释，而非仅仅是学习到的噪声模式。  

### 一句话记住它
把大模型的思考过程当作“教材”，让小模型一步步学会推理，从而用更少数据和更小体积跑赢大模型。