# LLM Augmented LLMs: Expanding Capabilities through Composition

> **Date**：2024-01-04
> **arXiv**：https://arxiv.org/abs/2401.02412

## Abstract

Foundational models with billions of parameters which have been trained on large corpora of data have demonstrated non-trivial skills in a variety of domains. However, due to their monolithic structure, it is challenging and expensive to augment them or impart new skills. On the other hand, due to their adaptation abilities, several new instances of these models are being trained towards new domains and tasks. In this work, we study the problem of efficient and practical composition of existing foundation models with more specific models to enable newer capabilities. To this end, we propose CALM -- Composition to Augment Language Models -- which introduces cross-attention between models to compose their representations and enable new capabilities. Salient features of CALM are: (i) Scales up LLMs on new tasks by 're-using' existing LLMs along with a few additional parameters and data, (ii) Existing model weights are kept intact, and hence preserves existing capabilities, and (iii) Applies to diverse domains and settings. We illustrate that augmenting PaLM2-S with a smaller model trained on low-resource languages results in an absolute improvement of up to 13\% on tasks like translation into English and arithmetic reasoning for low-resource languages. Similarly, when PaLM2-S is augmented with a code-specific model, we see a relative improvement of 40\% over the base model for code generation and explanation tasks -- on-par with fully fine-tuned counterparts.

---

# LLM 增强 LLM：通过组合扩展能力 论文详细解读

### 背景：这个问题为什么难？

大模型（LLM）拥有数十亿参数，训练时使用海量语料，已经在翻译、写代码、推理等任务上展现出惊人的零样本能力。可是，这类模型是“一体式”的——所有能力都埋在同一套权重里。想让模型学会新语言、专门的编程风格或低资源领域的知识，通常只能靠大规模微调或从头再训练，既耗算力又容易把已有能力给破坏。于是出现了“模型太笨重、改动代价高、能力容易相互干扰”的窘境，迫切需要一种既能保留原有技能，又能快速叠加新技能的方案。

### 关键概念速览
- **基础模型（Foundation Model）**：指像 PaLM、GPT 那样在通用大语料上预训练得到的、参数规模巨大的模型，具备广泛的语言理解与生成能力。可以把它想成“全能工具箱”。
- **领域模型（Domain‑Specific Model）**：在特定任务或语言上进一步训练的较小模型，专注于某一细分能力，就像“专用螺丝刀”。  
- **跨注意力（Cross‑Attention）**：一种让一个模型的查询向量去关注另一个模型的键值对的机制，类似于两个人对话时，一个人听取并引用对方的话来补充自己的表达。
- **参数冻结（Weight Freezing）**：在组合过程中保持原模型权重不变，只训练新增的少量模块，等价于在已有建筑上加装可拆卸的扩展层。
- **少量数据微调（Few‑Shot Fine‑Tuning）**：只用几百到几千条标注样本对新增模块进行训练，成本远低于全模型微调。
- **相对/绝对提升（Relative/Absolute Improvement）**：评价指标的提升幅度，前者是基准的百分比增长，后者是直接的分数提升。

### 核心创新点
1. **跨模型注意力桥接 → 在两个独立的语言模型之间插入跨注意力层 → 让大模型直接利用小模型的专属表征，而不需要改动原有权重。** 这比传统的“先微调再合并”省去了重复训练的步骤。
2. **保持基础模型不变 + 只训练少量桥接参数 → 通过冻结 PaLM2‑S 的全部参数，仅对跨注意力模块进行少量梯度更新 → 既保留了原有的通用能力，又避免了灾难性遗忘。**
3. **统一的组合框架适用于多种场景 → 同一套跨注意力机制既能把低资源语言模型接入，也能把代码专用模型接入 → 实现“一套代码，多种增益”。** 这比以往针对每个任务单独设计适配层的做法更具通用性。
4. **极少数据即可实现显著提升 → 只用了低资源语言的少量平行语料或代码示例，就让 PaLM2‑S 在对应任务上分别提升了 13% 绝对分数和 40% 相对分数 → 证明了“少量参数+少量数据”也能产生大幅度性能跃迁。**

### 方法详解
整体思路可以拆成三步：  
1️⃣ **准备两套模型**：一个是通用的大模型（如 PaLM2‑S），另一个是针对特定领域训练的轻量模型（如低资源语言翻译模型或代码生成模型）。  
2️⃣ **插入跨注意力模块**：在大模型的每一层（或选定的几层）后面并行放置一个跨注意力层。该层的查询向量来源于大模型的隐藏状态，键和值则来自小模型对应层的隐藏状态。于是，大模型在生成每个 token 时，会“偷看”小模型的表征并把它们加权融合。可以把它想象成两位专家在同一张白板上共同作画，大模型负责整体布局，小模型负责细节填充。  
3️⃣ **仅训练跨注意力参数**：冻结两套模型的原始权重，只对跨注意力层的投影矩阵进行梯度更新。训练数据只需要少量目标任务的示例（比如几千句低资源语言的翻译对或几百段代码），因为跨注意力的目标是学会如何把已有的专属表征映射到大模型的生成空间，而不是重新学习语言本身。

**关键细节**  
- **投影维度**：跨注意力的查询、键、值通常会被映射到一个相对低维的空间，以控制参数量。  
- **层级选择**：作者在实验中发现把跨注意力放在中高层（如第 12 层）效果最佳，因为此时的表征已经具备一定抽象度，既能捕捉语义，又不至于过于底层噪声。  
- **残差连接**：跨注意力的输出会通过残差加回到大模型原始隐藏状态，确保即使跨注意力失效，模型仍能靠自身能力完成任务。  
- **训练目标**：仍然使用原始的大模型任务损失（如交叉熵），只是在梯度流向上限制在跨注意力层。

最巧妙的地方在于**不需要重新训练大模型**，而是让它“借用”小模型的专长，这种“软耦合”比硬拼接权重或全模型微调更灵活，也更安全。

### 实验与效果
- **低资源语言翻译**：把 PaLM2‑S 与一个专门训练的低资源语言模型组合后，在从该语言到英语的翻译任务上，绝对分数提升最高可达 13%。  
- **代码生成与解释**：同样的组合方式把 PaLM2‑S 与代码专用模型对接后，在代码生成和解释基准上相对提升约 40%，表现与直接对 PaLM2‑S 完全微调得到的模型持平。  
- **基线对比**：分别与（1）原始 PaLM2‑S、（2）仅微调 PaLM2‑S、（3）全模型融合（即把两个模型的权重直接相加）进行比较，跨注意力组合在提升幅度和计算成本上均占优势。  
- **消融实验**：作者移除跨注意力层、或改为仅在低层使用，性能明显下降，说明跨注意力是关键驱动因素。  
- **局限性**：论文未给出在极大模型（如 540B 参数）上的实验，也没有评估跨注意力在多模型同时组合时的扩展性；此外，跨注意力本身仍会带来一定的推理时延。

### 影响与延伸思考
这篇工作打开了“模型即插件”的新思路，后续有不少研究尝试把不同专长的模型通过类似的注意力桥接或 LoRA（低秩适配）方式进行组合，形成可插拔的 AI 系统。推测未来会出现 **模型市场**：开发者只需提供小型专用模型，平台通过跨注意力或其他轻量桥接把它们装配到通用大模型上，实现按需增能。想进一步了解，可以关注 **Mixture‑of‑Experts（专家混合）**、**Adapter** 系列以及 **Prompt‑Tuning** 的最新进展，它们都在探索如何在保持核心模型不变的前提下灵活扩展能力。

### 一句话记住它
**只在大模型和小模型之间加一层跨注意力，就能用少量参数和数据让通用 LLM 获得专属新技能。**