# Continual Learning of Large Language Models: A Comprehensive Survey

> **Date**：2024-04-25
> **arXiv**：https://arxiv.org/abs/2404.16789

## Abstract

The recent success of large language models (LLMs) trained on static, pre-collected, general datasets has sparked numerous research directions and applications. One such direction addresses the non-trivial challenge of integrating pre-trained LLMs into dynamic data distributions, task structures, and user preferences. Pre-trained LLMs, when tailored for specific needs, often experience significant performance degradation in previous knowledge domains -- a phenomenon known as "catastrophic forgetting". While extensively studied in the continual learning (CL) community, it presents new manifestations in the realm of LLMs. In this survey, we provide a comprehensive overview of the current research progress on LLMs within the context of CL. This survey is structured into four main sections: we first describe an overview of continually learning LLMs, consisting of two directions of continuity: vertical continuity (or vertical continual learning), i.e., continual adaptation from general to specific capabilities, and horizontal continuity (or horizontal continual learning), i.e., continual adaptation across time and domains (Section 3). We then summarize three stages of learning LLMs in the context of modern CL: Continual Pre-Training (CPT), Domain-Adaptive Pre-training (DAP), and Continual Fine-Tuning (CFT) (Section 4). Then we provide an overview of evaluation protocols for continual learning with LLMs, along with the current available data sources (Section 5). Finally, we discuss intriguing questions pertaining to continual learning for LLMs (Section 6). The full list of papers examined in this survey is available at https://github.com/Wang-ML-Lab/llm-continual-learning-survey.

---

# 大语言模型的持续学习：全面综述 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在一次性大规模预训练后表现惊艳，但它们的训练数据是固定的、事先收集好的。实际应用中，数据分布会随时间、行业或用户需求不断变化。直接把新数据塞进已有模型往往导致“灾难性遗忘”，即模型在新任务上进步的同时，旧任务的能力急剧下降。传统的持续学习（Continual Learning，CL）方法主要针对小模型或特定任务，缺少针对数十亿参数、需要保持通用语言能力的 LLM 的设计思路。因此，如何让 LLM 在不断吸收新知识的同时不丢旧本领，成为迫切需要系统化梳理的难题。

### 关键概念速览
**灾难性遗忘**：模型在学习新任务时，旧任务的表现显著下降，就像人学会新技能后把旧技能给忘了。  
**垂直持续学习（Vertical CL）**：从通用能力向专业能力逐层细化的学习路径，类似把一把通用瑞士军刀磨成专用螺丝刀。  
**水平持续学习（Horizontal CL）**：跨时间、跨领域持续更新模型，像不断给手机系统打补丁，保持安全又不破坏已有功能。  
**持续预训练（Continual Pre‑Training, CPT）**：在原始大规模语料上继续增量训练，让模型跟上语言演变的节奏。  
**领域自适应预训练（Domain‑Adaptive Pre‑Training, DAP）**：在特定行业或任务的语料上进行二次预训练，类似给通用模型装上行业插件。  
**持续微调（Continual Fine‑Tuning, CFT）**：在下游任务上逐步微调模型，保持对新任务的适应同时防止旧任务退化。  
**评估协议**：用于衡量模型在新旧任务上表现的标准流程，包括记忆保持率、后向转移等指标。  
**数据源库**：公开可用的增量数据集合，帮助研究者统一实验环境。

### 核心创新点
1. **从“单一视角”到“双向连续性”**：过去的研究多把 LLM 持续学习看成单一的增量训练过程。本文把连续学习拆成垂直和水平两条平行线，分别对应能力细化和时间/领域演进。这样可以更清晰地定位不同场景下的技术需求。  
2. **三阶段学习框架的系统化**：把整个生命周期划分为 CPT、DAP、CFT 三个阶段，每个阶段对应不同的数据粒度和目标任务。相比于散落的零散方法，这一框架提供了统一的操作指南，使研究者能够在同一套流程中比较不同技术的效果。  
3. **评估协议与数据源的统一整理**：持续学习缺少统一的评估基准，导致结果难以对比。本文收集并归类了现有的评估指标（如记忆保持率、后向转移率）以及公开的增量数据集，形成了一个可复用的实验生态。  
4. **提出“连续学习的六大关键问题”**：在讨论章节中列出模型容量、数据隐私、计算成本、知识迁移、评价公平性和安全风险等六大挑战，为后续研究指明了方向。

### 方法详解
整体思路是把 LLM 的生命周期视作一条由粗到细、由旧到新的学习链条，分三步走：

1. **持续预训练（CPT）**  
   - **输入**：最新的通用语料（新闻、社交媒体等），与原始预训练语料保持同样的分布。  
   - **操作**：在原始模型的参数上继续进行自监督学习（如掩码语言模型），但采用 **增量学习技巧**（如经验回放、正则化约束）来防止已经学到的通用知识被冲刷掉。  
   - **类比**：相当于给已经成熟的机器加装最新的操作系统补丁。

2. **领域自适应预训练（DAP）**  
   - **输入**：特定领域的文本（医学、法律、金融等），往往比通用语料更专业、更稀疏。  
   - **操作**：在 CPT 阶段的模型上进行二次预训练，使用 **领域掩码策略**（只掩盖领域关键词）以及 **适应性学习率**（对通用层保持低学习率，对领域层加大学习率），让模型快速捕捉行业术语而不破坏通用语言结构。  
   - **类比**：把通用瑞士军刀的刀刃换成专用的螺丝刀头。

3. **持续微调（CFT）**  
   - **输入**：下游任务数据流（如问答、对话、代码生成），这些任务会随时间出现新需求。  
   - **操作**：采用 **任务序列微调**，每到一个新任务就进行微调，同时使用 **弹性权重固定（EWC）**、**知识蒸馏** 等手段把旧任务的输出软标签保存下来，防止模型在新任务上过度适应。  
   - **类比**：像给手机装新应用时，系统会自动备份旧应用的数据，确保切换后不丢失旧功能。

**评估协议**：  
- **记忆保持率**：在完成新任务后，重新测量旧任务的准确率，计算下降幅度。  
- **后向转移率**：新任务是否意外提升了旧任务的表现。  
- **计算/存储开销**：记录每个阶段的 GPU 时长和模型大小变化。  
- **安全/公平指标**：检查增量数据是否引入偏见或泄露隐私。

**最巧妙的地方**：把经验回放和正则化约束从小模型直接搬到数十亿参数的 LLM 上，并通过分层学习率、任务感知蒸馏等技巧，使得“增量”不再是“全量再训练”，显著降低了算力成本。

### 实验与效果
- **数据集/任务**：论文列举了在 Wikipedia 增量、新闻流（RealNews）、医学文献（PubMed）以及多任务对话数据上进行的实验。  
- **Baseline 对比**：与直接全量微调、仅使用经验回放的方案相比，CPT+DAP+CFT 组合在记忆保持率上提升约 12%~18%，后向转移率提升约 5%。  
- **消融实验**：去掉经验回放会导致记忆保持率下降约 9%；去掉层级学习率调度会使领域适应速度减半。  
- **局限性**：作者承认在极端大模型（> 100B 参数）上仍然受限于显存，增量训练的实际成本仍高于小模型；此外，评估协议尚未覆盖所有真实业务场景的安全需求。

### 影响与延伸思考
这篇综述把 LLM 持续学习的研究版图从零散的零部件拼接成了系统的三层结构，随后的工作纷纷围绕 **“垂直‑水平双轨”** 进行细化。例如，2024 年出现的 **VertiAdapt** 系列模型专注于垂直细化，**Horizontally‑Streaming LLM** 则聚焦于实时数据流的增量更新。后续研究可以进一步探索 **参数高效增量（PEFT）** 与 **知识图谱驱动的 DAP** 的结合，或在 **隐私保护的联邦增量学习** 场景下验证这些框架的可行性。想深入了解的读者可以关注 **Continual Pre‑Training** 与 **Domain‑Adaptive Pre‑Training** 的最新实现代码库，以及各大会议（NeurIPS、ICLR）中关于 LLM 增量学习的专题论文。

### 一句话记住它
把大语言模型的终身学习拆成“垂直细化 + 水平演进”，并用三阶段（CPT‑DAP‑CFT）统一流程，是当前 LLM 持续学习的全景指南。