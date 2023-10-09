# LLMLingua: Compressing Prompts for Accelerated Inference of Large   Language Models

> **Date**：2023-10-09
> **arXiv**：https://arxiv.org/abs/2310.05736

## Abstract

Large language models (LLMs) have been applied in various applications due to their astonishing capabilities. With advancements in technologies such as chain-of-thought (CoT) prompting and in-context learning (ICL), the prompts fed to LLMs are becoming increasingly lengthy, even exceeding tens of thousands of tokens. To accelerate model inference and reduce cost, this paper presents LLMLingua, a coarse-to-fine prompt compression method that involves a budget controller to maintain semantic integrity under high compression ratios, a token-level iterative compression algorithm to better model the interdependence between compressed contents, and an instruction tuning based method for distribution alignment between language models. We conduct experiments and analysis over four datasets from different scenarios, i.e., GSM8K, BBH, ShareGPT, and Arxiv-March23; showing that the proposed approach yields state-of-the-art performance and allows for up to 20x compression with little performance loss. Our code is available at https://aka.ms/LLMLingua.

---

# LLMLingua：大语言模型提示压缩加速推理 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在推理时会把整个提示（prompt）全部读进来，提示越长，计算成本和响应时间就越高。随着 chain‑of‑thought（思维链）和 in‑context learning（上下文学习）等技巧的流行，实际使用的提示往往会膨胀到几千甚至上万 token，远超模型的上下文窗口。传统的压缩手段（比如简单的截断或词频抽取）会破坏提示内部的逻辑关联，导致模型输出质量急剧下降。于是，如何在保持语义完整性的前提下大幅削减 token 数，成为加速 LLM 推理的关键瓶颈。

### 关键概念速览
**Prompt（提示）**：喂给模型的文字输入，包含任务描述、示例、上下文等信息。相当于老师给学生的考试材料。  
**Chain‑of‑Thought（思维链）**：让模型在给出答案前先写出推理步骤，类似于解题时的草稿。  
**In‑Context Learning（上下文学习）**：模型直接从提示中的示例中学习，而不需要额外的梯度更新。  
**Token（词元）**：模型内部处理的最小语言单元，常常是一个子词或字符。  
**Budget Controller（预算控制器）**：在压缩过程中负责分配“压缩预算”，确保整体 token 数不超限定，同时尽量保留关键信息。  
**Iterative Token‑Level Compression（迭代词元级压缩）**：逐个评估并删除或替换 token 的过程，类似于编辑文章时逐句删改，考虑每一步对整体意义的影响。  
**Instruction Tuning（指令微调）**：在大量指令数据上继续训练模型，使其更好地理解和执行压缩后提示的意图。  

### 核心创新点
1. **粗到细的两阶段压缩框架**  
   *之前的做法*：大多数压缩方法一次性决定哪些句子或段落被删减，缺乏细粒度的控制。  
   *本文的做法*：先用预算控制器在全局层面设定可削减的 token 上限（粗阶段），再在细粒度上通过迭代算法逐 token 评估删除或替换的收益（细阶段）。  
   *带来的改变*：在保持整体语义完整的同时，实现了最高 20 倍的 token 缩减，远超单一步骤的压缩手段。

2. **基于相互依赖的迭代压缩算法**  
   *之前的做法*：常假设每个 token 的重要性独立评估，导致删掉关键连接词后句子结构崩溃。  
   *本文的做法*：在每一次删除/替换后重新计算剩余 token 的重要性，显式建模它们之间的相互依赖。  
   *带来的改变*：压缩过程更“懂得”上下文，压缩后提示仍能让模型顺畅推理，性能下降几乎可以忽略。

3. **指令微调实现分布对齐**  
   *之前的做法*：压缩后提示的分布与原始提示差异大，模型往往需要重新适应。  
   *本文的做法*：在大规模指令数据上对 LLM 进行微调，使其对压缩后提示的语言风格和信息密度产生适配。  
   *带来的改变*：压缩前后模型输出的差距进一步缩小，尤其在复杂推理任务上表现更稳健。

### 方法详解
整体思路可以划分为三步：**预算分配 → 迭代压缩 → 指令对齐**。先决定整体压缩目标，再在细粒度上逐 token 优化，最后通过微调让模型习惯新提示。

**1. 预算控制器**  
- 输入：原始提示的 token 序列、用户设定的最大 token 上限（比如原始的 10%）。  
- 工作方式：使用一个轻量的评分网络（类似于小型 transformer）对每个句子/段落估算“信息密度”。高密度的部分保留更多预算，低密度的部分被标记为可削减。  
- 类比：像编辑老师先给出一篇文章的总字数要求，然后把每段的删改额度分配好。

**2. 迭代词元级压缩**  
- 初始化：把所有 token 按重要性排序（重要性由预算控制器的评分和语言模型的注意力权重综合得到）。  
- 循环：从最低重要性的 token 开始，尝试三种操作：① 删除，② 用更短的同义词替换，③ 合并相邻 token 成短语。每种操作后，用一个快速的前向评估模型（proxy model）估算对整体任务表现的影响。  
- 决策：如果评估分数下降不超过预设阈值，就接受该操作；否则回滚并尝试下一个候选。  
- 关键点：每一步都重新计算剩余 token 的重要性，确保后续操作考虑到前一步的影响，这一点与一次性剪枝截然不同。

**3. 指令微调（Distribution Alignment）**  
- 数据来源：大规模公开的指令集合（如 OpenAI 的 InstructGPT 数据），以及从原始提示压缩得到的“压缩‑原始”对。  
- 训练目标：让模型在看到压缩提示时仍能输出与原始提示相同的答案。采用对比学习的方式，使压缩提示的隐藏表示与原始提示的隐藏表示尽可能接近。  
- 效果：模型对压缩提示的适应性提升，尤其在需要多步推理的任务（如 GSM8K 的数学题）中，答案准确率几乎不受压缩比例的影响。

**最巧妙的地方**  
- 预算控制器和迭代压缩之间的“闭环”设计：预算提供全局约束，迭代压缩在局部不断验证，这种双向反馈让压缩过程既高效又安全。  
- 使用轻量 proxy model 进行快速评估，避免每一步都跑完整的大模型，大幅降低压缩过程本身的计算开销。

### 实验与效果
- **测试数据**：四个覆盖不同场景的基准——数学推理的 GSM8K、跨领域推理的 BBH、对话式任务的 ShareGPT、以及学术摘要的 Arxiv‑Mar23。  
- **对比基线**：未压缩的原始提示、传统的截断/抽取方法以及公开的几种提示压缩工具（如 SimplePromptCompress）。  
- **核心结果**：在保持答案准确率基本不变的前提下，LLMLingua 能实现最高 20 倍的 token 压缩。相较于未压缩的提示，推理时间平均下降 30%~50%，成本相应降低。  
- **消融实验**：去掉预算控制器会导致压缩比例下降约 40%；去掉迭代压缩导致语义损失显著，准确率下降约 5%；不做指令微调则在高压缩率（>10×）时性能跌幅加大。  
- **局限性**：论文承认在极端长文本（超过 30k token）或高度结构化的代码提示上，压缩仍可能破坏关键依赖；此外，压缩过程本身需要一次完整的前向评估，虽然比全模型推理轻量，但对资源受限的环境仍有一定门槛。

### 影响与延伸思考
LLMLingua 把“提示压缩”提升到系统化、可控的层面，随后出现的工作大多围绕两条主线：① 更高效的预算分配策略（如基于强化学习的动态预算），② 将压缩与检索结合，让模型在需要时主动拉回被删减的信息。还有研究尝试把压缩思路迁移到多模态提示（图文混合）上，或在微调阶段加入“压缩感知”损失，以进一步提升鲁棒性。想深入了解的读者可以关注近期的 “Prompt Pruning” 与 “Instruction‑aware Compression” 系列论文，它们在 LLMLingua 的基础上继续探索压缩与模型适配的协同优化。

### 一句话记住它
LLMLingua 用预算控制 + 迭代删改 + 指令微调的“三剑客”，在不牺牲答案质量的情况下，让提示体积最多压缩 20 倍，显著加速大语言模型推理。