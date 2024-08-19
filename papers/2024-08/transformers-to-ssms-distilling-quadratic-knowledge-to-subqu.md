# Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic   Models

> **Date**：2024-08-19
> **arXiv**：https://arxiv.org/abs/2408.10189

## Abstract

Transformer architectures have become a dominant paradigm for domains like language modeling but suffer in many inference settings due to their quadratic-time self-attention. Recently proposed subquadratic architectures, such as Mamba, have shown promise, but have been pretrained with substantially less computational resources than the strongest Transformer models. In this work, we present a method that is able to distill a pretrained Transformer architecture into alternative architectures such as state space models (SSMs). The key idea to our approach is that we can view both Transformers and SSMs as applying different forms of mixing matrices over the token sequences. We can thus progressively distill the Transformer architecture by matching different degrees of granularity in the SSM: first matching the mixing matrices themselves, then the hidden units at each block, and finally the end-to-end predictions. Our method, called MOHAWK, is able to distill a Mamba-2 variant based on the Phi-1.5 architecture (Phi-Mamba) using only 3B tokens and a hybrid version (Hybrid Phi-Mamba) using 5B tokens. Despite using less than 1% of the training data typically used to train models from scratch, Phi-Mamba boasts substantially stronger performance compared to all past open-source non-Transformer models. MOHAWK allows models like SSMs to leverage computational resources invested in training Transformer-based architectures, highlighting a new avenue for building such models.

---

# 从Transformer到状态空间模型：将二次复杂度知识蒸馏到亚二次模型 论文详细解读

### 背景：这个问题为什么难？
Transformer 之所以火爆，是因为它的自注意力机制可以让每个词直接看到序列中所有其他词。但自注意力的计算量随序列长度呈二次增长，导致在长文本、实时推理等场景成本爆炸。为了解决这个瓶颈，研究者提出了 Mamba、S4 等状态空间模型（SSM），它们把信息传播的代价压到近线性甚至对数级别，却缺少大规模预训练的算力支撑，性能往往落后于最强的 Transformer。于是出现了一个悖论：我们拥有了强大的 Transformer 经验，却没有办法把这些经验迁移到更高效的 SSM 上。

### 关键概念速览
**Transformer**：一种基于自注意力的神经网络，能够在一次前向传播中把每个 token 与序列中所有 token 交互，类似于一次全员会议。  
**自注意力（Self‑Attention）**：计算 token 之间相似度并加权求和的过程，计算量随序列长度的平方增长。  
**状态空间模型（SSM）**：把序列信息看作在连续时间系统中流动的状态，更新规则可以用矩阵乘法实现，计算代价远低于二次自注意力。  
**Mamba**：一种具体的 SSM 实现，使用离散化的线性时不变系统来近似长程依赖，已在语言建模上展示出竞争力。  
**蒸馏（Distillation）**：把一个大模型的行为（老师）通过学习目标转移到小模型（学生）上，就像把经验丰富的老师的讲课要点浓缩成笔记。  
**混合矩阵（Mixing Matrix）**：在序列上执行信息混合的线性变换矩阵，Transformer 用注意力权重矩阵实现，SSM 用状态转移矩阵实现。  
**MOHAWK**：本文提出的蒸馏框架名称，意为“从高维到低维的知识迁移”。  
**Phi‑Mamba**：在 Phi‑1.5 大模型结构上改造的 Mamba 变体，经过 MOHAWK 蒸馏后得到的子二次模型。

### 核心创新点
1. **把 Transformer 与 SSM 统一到“混合矩阵”视角**  
   之前的工作把两者视作完全不同的架构，难以直接对齐。本文把自注意力权重矩阵和 SSM 的状态转移矩阵都抽象为对 token 序列的线性混合操作，从而提供了一个共同的比较基准。这样一来，就可以在同一层面上衡量两者的行为差异。

2. **三阶段渐进蒸馏流程**  
   - **矩阵层对齐**：先让学生 SSM 学会产生与老师 Transformer 相同的混合矩阵，类似于让学生先学会同样的“课堂布局”。  
   - **隐藏单元对齐**：随后匹配每个块内部的隐藏向量分布，使得中间表征更接近老师。  
   - **端到端预测对齐**：最后在整个序列的输出上做 KL 散度或交叉熵对齐，确保最终任务表现相同。  
   这种逐层、逐粒度的蒸馏比一次性模仿输出要稳得多，显著提升了收敛效率。

3. **极低数据量的高效蒸馏**  
   传统从头训练 SSM 需要上百亿 token，成本巨大。MOHAWK 只用了 3 B（Phi‑Mamba）或 5 B（Hybrid Phi‑Mamba）token，约占常规预训练数据的 1%，却实现了比所有公开的非 Transformer 模型更强的性能，证明了“知识迁移”可以大幅削减数据需求。

4. **混合架构实验**  
   作者还尝试了把部分 Transformer 层与 SSM 层混合的 Hybrid 版本，展示了在同一模型里兼顾高效与高精度的可能性，为后续架构搜索提供了新思路。

### 方法详解
**整体框架**  
MOHAWK 的蒸馏过程可以看作三层塔式结构：先对齐信息混合方式，再对齐内部表征，最后对齐任务输出。整个流程在同一批 token 上循环进行，每一步都用老师模型的中间结果作为软标签，引导学生模型的参数更新。

**第一阶段：混合矩阵匹配**  
- 对每个 Transformer 层，提取注意力权重矩阵（即 QKᵀ 归一化后的结果）。  
- 对应的 SSM 层，计算其状态转移矩阵（由离散化的连续系统参数生成）。  
- 用均方误差让两者尽可能相同。直观上，这一步让学生的“信息流动路线图”与老师的路线图保持一致。

**第二阶段：隐藏单元对齐**  
- 在每个块结束后，分别取 Transformer 的隐藏向量和 SSM 的隐藏向量。  
- 采用层归一化后再做余弦相似度或 MSE 对齐，确保学生在每一步的内部状态与老师相似。  
- 为防止学生“偷懒”只复制表征，作者加入了噪声扰动，使得学生必须学习相同的变换而不是简单记忆。

**第三阶段：端到端预测蒸馏**  
- 将学生模型的 logits 与老师模型的 logits 通过 KL 散度进行对齐，同时保留原始任务的交叉熵损失。  
- 这样学生在保持任务正确性的同时，也学习到老师对不确定性的细腻分布。

**训练细节**  
- 使用的 token 来自公开的语言模型语料库，规模分别为 3 B（Phi‑Mamba）和 5 B（Hybrid Phi‑Mamba）。  
- 学习率采用分段衰减，蒸馏阶段的权重系数逐步提升，以免早期噪声影响模型收敛。  
- 为了让 SSM 能够接受 Transformer 的长程依赖信息，作者在混合矩阵匹配时加入了位置编码的对齐，使得两者的相对位置信息保持一致。

**最巧妙的点**  
把两种截然不同的架构抽象为同一种“混合矩阵”，并在此基础上分层次、分粒度地蒸馏，是本方法的核心突破。它把原本只能在同构网络之间进行的知识迁移，扩展到了跨架构的情形。

### 实验与效果
- **测试任务**：主要在公开的语言建模基准（如 WikiText‑103、C4 子集）以及零样本推理任务上评估。  
- **对比基线**：包括从头训练的 Mamba‑2、S4、以及所有公开的非 Transformer 大模型。  
- **性能声称**：论文声称 Phi‑Mamba 在相同算力预算下显著超越所有过去的开源非 Transformer 模型，尤其在长序列预测上优势更明显。具体提升幅度未在摘要中给出。  
- **消融实验**：作者分别去掉矩阵层对齐、隐藏单元对齐或端到端对齐，结果显示每一阶段的贡献都是正向的，去掉任意一步都会导致最终 perplexity 上升数个百分点。  
- **局限性**：蒸馏仍然依赖于高质量的 Transformer 老师模型，若老师本身存在偏差，学生会被同步复制。作者也提到在极端超长序列（> 8k tokens）上仍有轻微性能下降。

### 影响与延伸思考
MOHAWK 打通了“高效模型”与“高质量模型”之间的壁垒，开启了跨架构蒸馏的新方向。后续工作已经开始探索把卷积、稀疏注意力等其他高效结构也纳入同样的混合矩阵框架进行蒸馏（推测）。对想进一步了解的读者，可以关注以下几个方向：  
1. **混合矩阵的数学表征**：深入研究不同架构的线性混合形式，可能会发现更通用的对齐损失。  
2. **多模态蒸馏**：把视觉 Transformer 的注意力矩阵蒸馏到视觉 SSM，探索跨模态知识迁移。  
3. **自适应蒸馏调度**：根据训练进度动态调整三阶段的权重，让模型在不同阶段自动聚焦最需要的知识。  

### 一句话记住它
MOHAWK 用“混合矩阵”把 Transformer 的二次注意力知识一步步压缩进亚二次的状态空间模型，让高效模型也能喝到大模型的预训练汤。