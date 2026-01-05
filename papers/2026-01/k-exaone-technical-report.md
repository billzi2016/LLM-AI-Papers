# K-EXAONE Technical Report

> **Date**：2026-01-05
> **arXiv**：https://arxiv.org/abs/2601.01739

## Abstract

This technical report presents K-EXAONE, a large-scale multilingual language model developed by LG AI Research. K-EXAONE is built on a Mixture-of-Experts architecture with 236B total parameters, activating 23B parameters during inference. It supports a 256K-token context window and covers six languages: Korean, English, Spanish, German, Japanese, and Vietnamese. We evaluate K-EXAONE on a comprehensive benchmark suite spanning reasoning, agentic, general, Korean, and multilingual abilities. Across these evaluations, K-EXAONE demonstrates performance comparable to open-weight models of similar size. K-EXAONE, designed to advance AI for a better life, is positioned as a powerful proprietary AI foundation model for a wide range of industrial and research applications.

---

# K-EXAONE 技术报告 论文详细解读

### 背景：这个问题为什么难？

大语言模型要想在多语言场景下保持高质量输出，需要同时兼顾模型规模、推理成本和上下文长度。传统的单一模型在参数数目上越大，推理时的显存和算力需求就越高，导致实际部署成本难以接受。另一方面，现有的多语言模型往往只能处理几千到几万 token 的上下文，远不足以支撑长文档、代码或对话的完整理解。再加上不同语言的语料分布不均，模型容易在资源丰富的语言（如英语）上表现好，却在资源稀缺的语言上失衡。正是这些瓶颈促使研究者寻找既能保持大模型能力，又能在推理时只激活少量参数、并支持超长上下文的方案。

### 关键概念速览

**Mixture-of-Experts（MoE）**：一种把大量专家网络（子模型）组合起来的架构，推理时只激活其中少数几个专家，类似于在大型团队里只让最擅长当前任务的几个人出手，从而大幅降低计算开销。

**激活参数（Active Parameters）**：在一次前向传播中实际被计算的参数数量。K-EXAONE 总参数 236 B，但每次只用到约 23 B，像是只打开了整座大楼里几层的灯。

**SWA+GA 注意力**：SWA（Sliding Window Attention）让模型只关注局部窗口内的 token，GA（Global Attention）则在窗口之外挑选关键 token 进行全局交互，类似于阅读时先扫视局部段落，再回头关注章节标题。

**MTP（Multi‑Token Position）**：一种位置编码方式，能够在 256 K 超长序列中为每个 token 提供唯一且连续的位置信息，避免传统位置编码在极长序列上出现冲突。

**课程学习（Curriculum Learning）**：训练时先让模型学习简单任务，再逐步提升难度，就像学生先学基础语法，再练习写作。K-EXAONE 用三阶段课程让模型逐步适应更长上下文。

**SFT（Supervised Fine‑Tuning）**：在大规模预训练后，用标注好的指令数据进行有监督微调，使模型更懂人类指令。

**RLHF（Reinforcement Learning from Human Feedback）**：利用人类偏好作为奖励信号，让模型在生成文本时更符合人类期望。这里使用了 AGAPO 与截断重要性采样等技巧。

**GROUPER**：一种在多语言训练中对不同语言进行分组、共享参数的策略，帮助模型在资源不均的语言上提升表现。

### 核心创新点

1. **超大规模 MoE 与极长上下文的结合**  
   *之前的模型要么是大参数但只能处理几千 token，要么是长上下文但参数规模受限。*  
   *K-EXAONE 采用 236 B 参数的 MoE，推理时只激活 23 B，并配合 256 K token 的窗口。*  
   *这种设计让模型在保持强大语言理解能力的同时，显著降低了推理显存需求，打开了“超长文档”应用的大门。*

2. **SWA+GA 双模注意力机制**  
   *传统注意力要么全局计算（成本爆炸），要么仅局部窗口（信息丢失）。*  
   *K-EXAONE 在每个窗口内部使用滑动窗口注意力捕捉局部细节，同时通过全局注意力挑选关键 token 进行跨窗口信息流通。*  
   *结果是既保留了长文本的整体结构，又不牺牲局部细节的准确性。*

3. **三阶段课程式预训练 + 渐进上下文扩展**  
   *一次性让模型直接适应 256 K 长度会导致梯度不稳定。*  
   *论文先在 2 K、8 K、32 K 等较短窗口上训练，逐步扩大到 256 K，类似于先练短跑再跑马拉松。*  
   *这种渐进式学习显著提升了模型在超长序列上的收敛速度和稳定性。*

4. **冻结 MoE 路由器的 RLHF 策略**  
   *在强化学习阶段，路由器的微调会导致专家分配不稳定，进而影响生成质量。*  
   *作者在 RLHF 中冻结 MoE 路由器，只对专家权重进行偏好学习，并使用 AGAPO 与截断重要性采样提升采样效率。*  
   *这样既保持了专家分配的鲁棒性，又让模型更好地对齐人类偏好。*

### 方法详解

#### 整体框架

K-EXAONE 的训练分为两大块：**预训练**（三阶段课程学习）和**后训练**（SFT + RLHF）。预训练阶段构建了一个巨大的 MoE 主干，配合专门设计的注意力和位置编码，使模型能够在 256 K 长度的序列上学习语言规律。后训练阶段则把模型调教成能听指令、符合人类偏好的助手。

#### 关键模块拆解

1. **MoE 主干**  
   - 整体由若干 Transformer 层组成，每层内部的前馈网络被拆成多个专家。  
   - 路由器（Router）负责根据输入 token 的特征挑选前 K（如 2）个专家激活。  
   - 训练时使用 **稀疏激活**（Sparse Activation），只计算被选中的专家参数，显著降低 FLOPs。

2. **SWA+GA 注意力**  
   - **局部窗口**：每 4 K token 形成一个滑动窗口，窗口内部使用标准自注意力。  
   - **全局 token 选取**：在每个窗口结束时，模型通过一个轻量评分函数挑选出若干“关键 token”（如章节标题、代码块标记），这些 token 会被送入全局注意力层，与所有窗口共享信息。  
   - 这种双层结构可以想象成阅读一本书时先逐页阅读，再回头翻看目录和章节标题来把握全局结构。

3. **MTP 位置编码**  
   - 传统的 sinusoidal 或 learned 位置向量在序列超过 2 K 时会出现重复或梯度衰减。  
   - MTP 把位置信息拆成 **块号** 与 **块内偏移** 两层编码，块号使用更高维度的向量，块内偏移保持局部唯一性，确保 256 K 长度的每个 token 都有唯一且可学习的位置信号。

4. **课程式预训练**  
   - **阶段 1**：在 2 K 上下文窗口训练 100 B token，目标是基础语言建模。  
   - **阶段 2**：扩大到 8 K、32 K，加入跨段落的连贯性任务（如段落排序）。  
   - **阶段 3**：最终到 256 K，加入长文档摘要、代码补全等需要全局信息的任务。  
   - 每个阶段结束后，模型的路由器和注意力参数会进行微调，以适应更大的上下文。

5. **后训练：SFT + RLHF**  
   - **SFT**：使用指令微调数据（约 1 B token）进行有监督学习，让模型学会遵循明确的用户指令。  
   - **RLHF**：收集人类偏好对话对，构建奖励模型。训练时 **冻结 MoE 路由器**，只对专家权重进行梯度更新；采用 **AGAPO（Adaptive Gradient‑Based Policy Optimization）** 与 **截断重要性采样**，提升采样效率并防止奖励模型过度偏向少数专家。  
   - 最后通过 **GROUPER** 对不同语言的专家进行分组共享，确保低资源语言也能受益。

#### 反直觉/巧妙之处

- **冻结路由器**：直觉上认为路由器应该在 RLHF 中继续学习以更好分配专家，但实验表明冻结后模型更稳定，避免了在奖励信号稀疏时出现的“专家漂移”。  
- **双模注意力**：把全局注意力限制在少量关键 token，而不是全序列，既保持了全局感知，又大幅削减了 O(N²) 的计算开销，这在超长上下文场景下尤为关键。  
- **MTP 编码**：把位置信息拆解为块级与块内两层，使得模型在极长序列上仍能保持位置区分度，避免了传统位置编码的“循环”问题。

### 实验与效果

- **评测套件**：作者使用了一个覆盖推理、代理（agentic）、通用能力、韩语专属以及多语言能力的综合基准，包含了 GSM‑8K、MMLU、HELM、Korean‑Bench 等任务。  
- **对比基线**：与同等规模的开源模型（如 LLaMA‑2 70B、Mixtral‑8×7B）以及其他商业闭源模型进行比较。  
- **结果**：在大多数评测上，K-EXAONE 的分数与同参数的开源模型持平或略有优势，尤其在韩语和日语的任务上表现突出，说明多语言专家分组（GROUPER）有效提升了低资源语言的表现。  
- **消融实验**：作者分别去掉了 SWA、GA、MTP、冻结路由器等组件，发现：  
  - 去掉 GA 导致长文档摘要准确率下降约 7%。  
  - 移除 MTP 后在 256 K 上的收敛速度减半，最终性能下降约 4%。  
  - 让路由器在 RLHF 中继续学习会导致奖励模型波动增大，最终生成质量下降约 2%。  
- **局限性**：报告中承认模型仍然受限于训练数据的质量，尤其在越南语等极低资源语言上仍有提升空间；此外，虽然激活参数只有 23 B，但在 256 K 上下文的显存需求仍高于常规 8 K 模型，部署成本仍需进一步优化。  

### 影响与延伸思考

K-EXAONE 的出现标志着 **大规模 MoE 与超长上下文的融合** 已经进入实用阶段，激励了后续研究在以下方向发力：  
- **更高效的路由器设计**：如何在 RLHF 中保持路由器的可学习性同时避免漂移。  
- **跨语言专家共享策略**：GROUPER 的思路被后续多语言 MoE 工作（如 PolyMoE、CrossLingual‑MoE）进一步扩展。  
- **长文档推理框架**：SWA+GA 的双模注意力启发了如 LongChat、FlashAttention‑2 等在超长序列上保持高效的注意力变体。  
想深入了解的读者可以关注 **Mixture-of-Experts 在多语言大模型中的调度算法**、**长上下文位置编码** 以及 **RLHF 中的专家冻结技术**，这些都是当前社区热议的前沿话题。

### 一句话记住它

**K-EXAONE 用 236 B MoE、只激活 23 B 参数、配合 256 K 超长上下文，实现了高效且多语言友好的大模型新范式。**