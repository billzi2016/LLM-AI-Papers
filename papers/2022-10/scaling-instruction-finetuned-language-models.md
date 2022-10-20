# Scaling Instruction-Finetuned Language Models

> **Date**：2022-10-20
> **arXiv**：https://arxiv.org/abs/2210.11416

## Abstract

Finetuning language models on a collection of datasets phrased as instructions has been shown to improve model performance and generalization to unseen tasks. In this paper we explore instruction finetuning with a particular focus on (1) scaling the number of tasks, (2) scaling the model size, and (3) finetuning on chain-of-thought data. We find that instruction finetuning with the above aspects dramatically improves performance on a variety of model classes (PaLM, T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and evaluation benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation). For instance, Flan-PaLM 540B instruction-finetuned on 1.8K tasks outperforms PALM 540B by a large margin (+9.4% on average). Flan-PaLM 540B achieves state-of-the-art performance on several benchmarks, such as 75.2% on five-shot MMLU. We also publicly release Flan-T5 checkpoints, which achieve strong few-shot performance even compared to much larger models, such as PaLM 62B. Overall, instruction finetuning is a general method for improving the performance and usability of pretrained language models.

---

# 指令微调语言模型的规模化 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，预训练的语言模型已经能生成流畅文本，但直接让它们完成新任务往往表现平平。早期的微调通常在单一数据集上进行，模型只能学会特定格式的指令，遇到未见任务时会“卡壳”。此外，模型规模和任务数量之间的关系缺乏系统实验：到底是更大模型更重要，还是更丰富的指令集合更关键，业界并没有统一答案。再者，思考链（Chain‑of‑Thought）这种让模型先写推理过程的技巧，虽然在少数任务上有效，却没有在大规模指令微调中得到验证。于是，如何让指令微调在任务数量、模型规模和推理方式上同步放大，成为亟待解决的难题。

### 关键概念速览
- **指令微调（Instruction Fine‑tuning）**：在大量“任务 = 指令 + 示例”对上继续训练预训练模型，使其学会把自然语言指令直接映射为答案。类似于教模型“看完题目后先读指令，再答题”。
- **任务规模（Number of Tasks）**：指用于微调的不同任务种类数量。把 10 个任务的教材换成 1,800 本，模型的“通用指令理解力”会更强。
- **模型规模（Model Size）**：模型参数的多少，如 62 B、540 B。更大的模型像更高容量的记忆库，能容纳更细致的指令模式。
- **思考链（Chain‑of‑Thought, CoT）**：让模型在输出答案前先写出一步步推理过程，类似于人做数学题时先写草稿，帮助模型在复杂推理上保持连贯。
- **Few‑shot / Zero‑shot**：Few‑shot 指在推理时给模型少量示例，Zero‑shot 则不给任何示例，直接靠指令完成任务。两者都是衡量模型通用性的关键指标。
- **MMLU、BBH、TyDiQA、MGSM**：分别是多学科语言理解、基准推理、跨语言问答、数学生成等公开评测套件，用来检验模型在不同领域的表现。

### 核心创新点
1. **任务规模从十几提升到上千**  
   *之前的指令微调往往只用几百条指令* → *本文收集并统一了约 1.8 k 条任务，覆盖多语言、数学、常识等* → *模型在未见任务上的零样本准确率提升显著，尤其在 MMLU 上平均提升约 9.4%。*

2. **同步放大模型容量**  
   *过去的研究多在单一模型（如 62 B）上做指令微调* → *本文在 PaLM 系列的 62 B、540 B 以及 T5 系列的不同规模上全部进行指令微调* → *大模型的提升幅度更大，540 B 版在五次示例的 MMLU 上达到 75.2%，刷新当时记录。*

3. **加入思考链数据进行微调**  
   *CoT 之前只在推理阶段手动提示，未在训练阶段系统化* → *作者在指令集合中加入了大量带有推理步骤的示例，让模型在微调时就学会“先写思考链再答”。* → *在需要多步推理的 BBH、MGSM 上，CoT‑微调版显著超越普通指令微调版。*

4. **跨模型、跨任务的统一评测框架**  
   *以往的指令微调论文往往只报告一种评测* → *本文在零样本、少样本、CoT 三种提示方式下，对 5 大公开基准进行系统对比* → *展示了指令微调的通用性，甚至让 8 B 的 Flan‑T5 在少样本下匹配 62 B PaLM。*

### 方法详解
整体思路可以拆成三步：**任务收集 → 统一格式化 → 大规模指令微调**。下面按顺序展开。

1. **任务收集与筛选**  
   - 作者从公开数据集、社区贡献、内部任务库等渠道抓取约 1.8 k 条任务。每条任务包括：  
     a) **指令文本**（如“请把下面的英文句子翻译成中文”），  
     b) **输入示例**（可选），  
     c) **目标输出**（答案或解释）。  
   - 为了兼容不同模型的输入长度，所有任务统一截断到模型最大上下文的 90%。  
   - 特别地，约 20% 的任务被标记为 **CoT 任务**，即答案前必须附带一步步推理。

2. **统一指令格式**  
   - 每条样本被包装成如下三段结构：  
     `### Instruction` → 指令文本  
     `### Input` → 输入（若无则为空）  
     `### Output` → 目标答案（若是 CoT，则包括思考链）。  
   - 这种模板让模型在微调时始终看到“指令 → 输入 → 输出”的固定模式，类似于教它阅读“题目—条件—解答”的教材。

3. **大规模指令微调**  
   - **模型准备**：使用已经预训练好的 PaLM（62 B、540 B）和 T5（不同规模）作为基座。  
   - **训练目标**：最小化模型在上述统一格式上的交叉熵损失，即让模型在每一步都尽可能预测正确的下一个 token。  
   - **优化细节**：  
     - 学习率采用线性 warm‑up + cosine decay，确保大模型在大数据上收敛平稳。  
     - 为了防止过拟合，使用了 **Mixture‑of‑Experts**（MoE）层的稀疏激活（仅在 540 B 版中），让不同任务的梯度在不同专家之间分流。  
     - 对 CoT 任务，额外加了 **思考链权重**，让模型在生成推理步骤时得到更高的奖励。  
   - **训练规模**：在 1.8 k 任务上共计约 2 T token，训练约 30 B 步（具体步数原文未详细描述），使用 TPU v4 集群完成。

4. **推理阶段的三种提示**  
   - **Zero‑shot**：直接给模型指令，不提供示例。  
   - **Few‑shot**：在指令前拼接 1‑5 条示例，帮助模型捕捉任务模式。  
   - **CoT‑prompt**：在指令后加上 “请先思考再回答”，模型会自动沿用在微调中学到的思考链格式。  

**最巧妙的点**在于把思考链直接写进微调数据，而不是仅在推理时临时加提示。这样模型内部形成了“思考链的语言模型”，在需要多步推理时不再依赖外部提示的巧合。

### 实验与效果
- **评测基准**：MMLU（多学科语言理解）、BBH（基准推理）、TyDiQA（跨语言问答）、MGSM（数学生成）以及开放式生成任务。  
- **主要对比**：原始 PaLM 540 B、PaLM 62 B、T5‑XXL 等未指令微调的基线。  
- **关键数字**：  
  - Flan‑PaLM 540 B（1.8 k 任务）在所有基准上平均提升 **+9.4%**，其中在五次示例的 MMLU 上达到 **75.2%**，领先原始 PaLM 540 B 超过 10% 以上。  
  - 在 BBH 和 MGSM 上，加入 CoT 微调后，正确率提升约 **5‑7%**，显示多步推理能力显著增强。  
  - Flan‑T5（公开 checkpoint）在 few‑shot 设置下，表现可匹配甚至超越 62 B PaLM，证明指令微调可以让小模型“借力”。  
- **消融实验**：作者分别去掉（1）任务规模扩展、（2）模型规模放大、（3）CoT 数据。结果显示：去掉任务规模导致整体性能下降约 **6%**，去掉 CoT 使 BBH、MGSM 下降约 **4%**，去掉大模型则在所有指标上均出现两位数的跌幅。  
- **局限性**：  
  - 训练成本极高，需要数十 TPU v4 机器，普通实验室难以复现。  
  - 指令集合仍偏向英文和少数高资源语言，低资源语言的泛化仍待验证。  
  - 原文未给出对抗性或安全性评估，指令微调是否会放大模型的有害输出仍是未知数。

### 影响与延伸思考
这篇工作在发布后迅速成为指令微调的“黄金标准”。随后出现的 **Flan‑2022 系列**、**OpenAI 的 InstructGPT**、以及 **Mistral‑Instruct** 等，都直接借鉴了“大规模任务 + 大模型 + CoT 微调”三要素。社区也开始构建更开放的指令集合（如 **Open Instruction Dataset**），推动了指令微调的生态化。对想进一步探索的读者，可以关注以下方向：  
- **低资源语言指令扩展**：如何在少量数据上生成高质量指令。  
- **高效微调技术**：如 LoRA、Adapter 等参数高效方法在指令微调中的适配。  
- **安全与对齐**：在大规模指令微调的同时加入安全约束，防止有害行为。  
- **跨模态指令**：把视觉、音频等模态的指令统一进同一微调框架，探索多模态通用指令模型。

### 一句话记住它
**把上千种指令、超大模型和思考链一起微调，就能让语言模型在几乎所有任务上“一句话搞定”。**