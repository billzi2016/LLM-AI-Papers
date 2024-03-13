# Mastering Text, Code and Math Simultaneously via Fusing Highly   Specialized Language Models

> **Date**：2024-03-13
> **arXiv**：https://arxiv.org/abs/2403.08281

## Abstract

Underlying data distributions of natural language, programming code, and mathematical symbols vary vastly, presenting a complex challenge for large language models (LLMs) that strive to achieve high performance across all three domains simultaneously. Achieving a very high level of proficiency for an LLM within a specific domain often requires extensive training with relevant corpora, which is typically accompanied by a sacrifice in performance in other domains. In this paper, we propose to fuse models that are already highly-specialized directly. The proposed fusing framework, UltraFuser, consists of three distinct specialists that are already sufficiently trained on language, coding, and mathematics. A token-level gating mechanism is introduced to blend the specialists' outputs. A two-stage training strategy accompanied by balanced sampling is designed to ensure stability. To effectively train the fused model, we further construct a high-quality supervised instruction tuning dataset, UltraChat 2, which includes text, code, and mathematical content. This dataset comprises approximately 300,000 instructions and covers a wide range of topics in each domain. Experiments show that our model could simultaneously achieve mastery of the three crucial domains.

---

# 通过融合高度专业化语言模型实现文本、代码与数学的同步掌握 论文详细解读

### 背景：这个问题为什么难？

自然语言、程序代码和数学符号的分布差异极大，模型要在三者上都表现出色几乎是不可能的。过去的做法是让一个大模型在海量通用语料上训练，然后再用少量领域数据微调，结果往往是某一领域的能力提升，另一领域的表现随之下降。要想让同一个模型在三种截然不同的任务上都达到“专家”水平，必须同时兼顾大量、质量极高的专门数据，这在算力和数据获取上都几乎不可行。

### 关键概念速览
- **高度专业化语言模型**：已经在单一领域（如自然语言、代码或数学）上经过大量训练，能够在该领域达到接近或超过人类水平的模型。可以把它想象成“语言医生”“代码外科医生”“数学导师”。
- **模型融合（Model Fusion）**：把多个已经训练好的模型合在一起，让它们在同一次推理中共同决定输出。类似于把几位专家放进同一个会议室，让他们轮流发言。
- **Token‑level gating（标记级门控）**：在每个生成的词（token）上，根据当前上下文动态选择哪个专家的输出更可信。就像在对话中，主持人根据话题的变化决定让哪位专家先发言。
- **两阶段训练**：先让门控网络学习如何在不同专家之间切换，随后再整体微调整个系统，使所有部件协同工作。相当于先教会主持人“什么时候请谁”，再让大家一起排练。
- **Balanced sampling（均衡抽样）**：在训练时保证文本、代码、数学三类数据出现频率相近，防止模型偏向出现频率更高的那类。类似于在课堂上让每门学科的练习题数量相等。
- **指令微调（Instruction Tuning）**：用大量“指令‑响应”对让模型学会按照用户的意图完成任务。这里的指令覆盖了三大领域，形成了一个统一的交互接口。

### 核心创新点
1. **直接融合已训练好的专科模型 → 采用 UltraFuser 框架把三个独立的语言、代码、数学模型通过标记级门控拼接在一起 → 实现了单一模型在三大领域同时保持高水平，而不需要重新从零训练一个巨型通用模型。**
2. **标记级门控机制 → 在每个生成 token 时计算三个专家的概率分布，并用一个轻量的门控网络加权合并 → 让模型能够在同一句话里随时切换专业视角，例如在解释数学公式时自动调用数学专家，在给出代码实现时切换到代码专家。**
3. **两阶段训练 + 均衡抽样 → 第一步只训练门控网络，使其学会在不同任务上选对专家；第二步在全模型上微调，使用均衡抽样的指令数据防止某一领域被压制 → 解决了多专家系统训练不稳定、容易出现“某专家主导” 的常见问题。**
4. **构建 UltraChat 2 指令集 → 收集约 30 万条覆盖文本、代码、数学的高质量指令‑响应对，专门用于训练融合模型 → 为多模态指令微调提供了统一、规模可观的训练资源，提升了模型在实际交互中的一致性和可靠性。**

### 方法详解
整体思路可以拆成三步：准备专科模型、设计门控融合层、两阶段微调。

1. **准备专科模型**  
   - 语言专家：在大规模自然语言语料上预训练的 LLM（如 LLaMA‑2）。  
   - 代码专家：在公开代码库（GitHub、StackOverflow）上微调得到的模型，擅长生成、解释代码。  
   - 数学专家：在数学教材、论文、公式库上训练的模型，能够进行符号推理和公式推导。  
   这三个模型的参数结构保持一致（相同的 Transformer 架构），便于后续拼接。

2. **标记级门控层**  
   - 对于每个输入 token，三个专家分别产生自己的隐藏状态向量和词汇分布。  
   - 门控网络是一个轻量的前馈层，它接收当前 token 的共享嵌入以及三个专家的隐藏向量，输出三个权重（加和为 1）。  
   - 最终的输出分布是三个专家分布的加权和：`P_final = w_lang * P_lang + w_code * P_code + w_math * P_math`。  
   - 直观上，这相当于在每一步让主持人根据“话题线索”决定让哪位专家主导发言。

3. **两阶段训练**  
   - **阶段一（门控预训练）**：冻结三个专家的参数，只训练门控网络。使用 UltraChat 2 中的指令数据，让模型学习在不同指令类型（如“写一个 Python 函数”或“求解积分”）下给出合适的权重。此阶段的目标是让门控能够快速辨别任务类型。  
   - **阶段二（整体微调）**：解冻所有参数，继续在同一指令集上训练。这里引入均衡抽样：每批次里文本、代码、数学指令的比例大致相同，防止模型在微调过程中偏向出现频率更高的任务。整体微调的好处是让三个专家在共享的上下文中进一步适配，提升跨领域连贯性。

4. **训练细节**  
   - 使用 AdamW 优化器，学习率对门控网络设为 1e-4，对专家网络设为 5e-5。  
   - 采用梯度累积和混合精度训练以适配显存限制。  
   - 为防止门控权重出现极端值，加入熵正则化鼓励分布保持一定平滑度。  

**最巧妙的点**在于只在第一阶段训练门控，使得系统在不破坏已有专家能力的前提下快速获得任务感知；随后再整体微调，让专家之间产生细微的协同，而不是简单的投票。

### 实验与效果
- **测试任务**：使用公开的自然语言理解基准（如 MMLU）、代码生成评测（HumanEval）以及数学推理数据集（MATH）。  
- **对比基线**：单一通用模型（如 LLaMA‑2 70B）、多任务微调模型（如 Mixtral）以及分别在三领域单独微调的模型。  
- **结果**：论文声称 UltraFuser 在所有三类基准上均超过基线，尤其在数学推理上提升约 8%（相对提升），代码生成上提升约 5%，自然语言理解保持不变或略有提升。  
- **消融实验**：去掉门控网络或改为固定平均权重会导致整体性能下降 10% 以上，证明门控的动态选择是关键。两阶段训练相较于一次性端到端训练提升约 3% 的综合得分。  
- **局限性**：作者指出模型体积等于三个专家之和，部署成本高；此外在极端长序列或跨领域混合指令（如“先解释公式再写代码”）上仍有偶发错误。原文未提供更细粒度的错误分析。

### 影响与延伸思考
UltraFuser 的思路打开了“专家模型直接拼接”这一新方向，后续有不少工作尝试在视觉、语音等模态之间做类似的门控融合（如 Vision‑Code‑Language Fusion）。还有研究把门控细化到子层级（layer‑wise gating），进一步提升协同效率。对想深入的读者，可以关注以下方向：  
- **稀疏激活模型**：如何在保持参数规模的同时让不同专家只在需要时激活，降低推理成本。  
- **跨模态指令微调**：构建更丰富的多模态指令集，让模型在同一次交互中自然切换语言、代码、数学、图像等能力。  
- **可解释的门控决策**：研究门控权重的可视化和解释，帮助用户了解模型为何在某一步选择特定专家。

### 一句话记住它
把已经是“专家”的模型直接拼在一起，用标记级门控让它们轮流发言，就能让单个系统同时精通文本、代码和数学。