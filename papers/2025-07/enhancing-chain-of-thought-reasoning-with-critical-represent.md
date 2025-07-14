# Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning

> **Date**：2025-07-14
> **arXiv**：https://arxiv.org/abs/2507.10085

## Abstract

Representation Fine-tuning (ReFT), a recently proposed Parameter-Efficient Fine-Tuning (PEFT) method, has attracted widespread attention for significantly improving parameter efficiency by editing representation space alone. In this work, we investigate applying ReFT to complex reasoning tasks. However, directly using the native ReFT method, which modifies fixed representations at the beginning and end of each layer, yields suboptimal performance, as these fixed-position representations have uncertain impact on the outputs. We observe that, in complex reasoning tasks, there often exist certain critical representations. These representations either integrate significant information from preceding layers or regulate subsequent layer representations. Through layer-by-layer propagation, they exert a substantial influence on the final output. Naturally, fine-tuning these critical representations has the potential to greatly enhance reasoning performance. Building upon these insights, we propose Critical Representation Fine-Tuning (CRFT), a novel method that identifies and optimizes these critical representations through information flow analysis. CRFT operates within a supervised learning framework, dynamically optimizing critical representations in a low-rank linear subspace while freezing the base model. The effectiveness and efficiency of our method are validated across eight benchmarks for arithmetic and commonsense reasoning, using LLaMA and Mistral model families. Furthermore, our method also adapts effectively to few-shot settings, boosting one-shot accuracy by 16.4%. Our work highlights the untapped potential of representation-level optimization for CoT reasoning, offering a lightweight yet powerful alternative to traditional PEFT methods.

---

# 通过关键表征微调提升链式思维推理 论文详细解读

### 背景：这个问题为什么难？

在大模型上实现高质量的链式思维（CoT）推理，往往需要对模型进行大规模微调，成本高且容易过拟合。已有的参数高效微调（PEFT）方法，如 LoRA、Adapter，只在少量可训练参数上做线性投影，虽然省显存，但它们主要针对输出层或整个隐藏层的整体特征，难以精准捕捉推理过程中关键的信息流。直接把这些方法套用到需要多步逻辑演绎的任务上，模型的中间表征往往被“平均化”，导致推理链的关键环节没有得到足够强化，最终的答案准确率提升有限。

### 关键概念速览
- **链式思维（CoT）**：让模型在给出最终答案前先写出逐步推理过程，类似人解题时的草稿，能够把复杂的算术或常识推理拆解成可验证的子步骤。  
- **参数高效微调（PEFT）**：在保持大模型权重不变的前提下，只训练少量额外参数（如低秩矩阵），实现“轻量调教”。  
- **表征微调（ReFT）**：一种 PEFT 方式，直接在每层的固定位置插入可学习的向量，改变模型内部的表示空间，而不触碰原始权重。  
- **关键表征（Critical Representation）**：在多层传播过程中，对后续层影响异常大的隐藏向量，通常是整合前层信息或调控后层行为的“枢纽”。  
- **信息流分析**：通过梯度、注意力权重或互信息等手段，追踪哪些隐藏向量在推理链中承担了信息聚合或转发的核心角色。  
- **低秩线性子空间**：在高维隐藏空间中选取一个维度远小于原始维度的子空间，只在这个子空间内进行微调，保持参数量极低。  

### 核心创新点
1. **从固定位置表征到关键表征的转变**  
   之前的 ReFT 把可学习向量硬编码在每层的开头和结尾，位置固定且与任务无关。本文先通过信息流分析定位对推理链影响最大的隐藏向量，然后只在这些位置上进行微调。这样做把“改动点”从盲目全局转向了有的放矢的关键节点，显著提升了 CoT 任务的效果。

2. **基于监督信号的动态子空间优化**  
   传统 PEFT 往往预设一个低秩矩阵并在整个训练过程中保持不变。这里提出在每一步前向传播后，根据当前批次的损失梯度重新计算最能捕获关键表征变化的低秩子空间，并在该子空间内更新参数。该机制让微调过程随任务难度自适应，避免了子空间选择不当导致的学习停滞。

3. **冻结基模型，仅调关键表征**  
   与 LoRA 等方法仍需要在每层插入额外的投影矩阵不同，CRFT 完全冻结原始模型权重，只在挑选出的关键表征上做低秩线性调整。这样既保持了原模型的通用能力，又以极小的参数开销实现了显著的推理提升。

4. **跨模型、跨任务的统一验证**  
   作者在 LLaMA 与 Mistral 两大模型族上、以及八个算术与常识推理基准上进行实验，证明了方法的模型无关性和任务普适性。尤其在一-shot 少样本设置下，准确率提升 16.4%，展示了 CRFT 在数据稀缺场景的潜力。

### 方法详解
**整体框架**  
CRFT 的工作流程可以概括为四步：① 前向传播收集中间隐藏向量；② 信息流分析筛选关键表征；③ 在这些表征上构建低秩线性子空间；④ 只更新子空间内的参数，基模型保持不变。整个过程在标准的监督学习循环中完成，训练目标仍是最小化任务的交叉熵或回归损失。

**关键表征筛选**  
- **信息流度量**：对每层的每个隐藏向量，计算它对最终输出梯度的贡献（即梯度范数）或注意力分布的集中度。贡献大的向量被视为“信息瓶颈”。  
- **层级传播**：从输入层向输出层逐层累积贡献度，若某层的向量在后续多层中持续保持高贡献，则标记为关键表征。  
- **阈值选择**：作者使用经验阈值（如前 5% 最大贡献）或动态阈值（贡献均值+标准差）来决定最终的关键集合。

**低秩子空间构建**  
- 对每个关键表征向量 \(h\)（维度 \(d\)），通过随机初始化的低秩矩阵 \(W_{low} \in \mathbb{R}^{d \times r}\)（\(r \ll d\)）映射到子空间。  
- 训练时只更新 \(W_{low}\) 与一个偏置向量 \(b\)，相当于在原始隐藏向量上加上一个可学习的低维扰动：\(h' = h + W_{low} \cdot (W_{low}^\top h) + b\)。  
- 这种设计保证了参数量仅为 \(O(r \cdot d)\)，且扰动方向受低秩约束，防止过度拟合。

**动态子空间更新**  
每个 mini‑batch 结束后，依据当前梯度重新计算关键表征的主方向（例如使用 SVD 提取前 r 个特征向量），并用这些方向重新初始化或微调 \(W_{low}\)。这样子空间会随训练进度自动对齐最有用的特征方向。

**冻结基模型的意义**  
所有原始层的权重保持不变，意味着模型的通用语言理解和知识库不受微调噪声影响。只在关键表征上做微调，相当于在模型内部的“思考笔记本”上写下针对特定推理任务的提示，而不改动“思考引擎”。

**最巧妙的点**  
信息流分析把“哪些隐藏向量重要”从经验猜测变成可量化的指标；再配合低秩子空间的动态更新，使得微调过程既轻量又高度针对性，这在传统 PEFT 中很少见。

### 实验与效果
- **数据集与任务**：作者选取了 8 个公开基准，包括 GSM8K（算术）、SVAMP、AQuA（代数）、CommonsenseQA、PIQA 等，覆盖算术推理和常识推理两大类。  
- **对比基线**：包括全参数微调、LoRA、Adapter、原始 ReFT 以及不做微调的原模型。  
- **主要结果**：在 LLaMA‑7B 上，CRFT 在 GSM8K 上提升约 7.2% 的准确率，超过 ReFT 约 4.5%。在 CommonsenseQA 上提升 5.8%，而 LoRA 只提升 2.3%。在一-shot 设置下，整体准确率提升 16.4%，显著高于所有对比方法。  
- **消融实验**：作者分别去掉信息流筛选、低秩子空间、动态更新三项，发现去掉信息流筛选导致性能下降约 3.9%，去掉低秩约 2.7%，去掉动态更新约 1.8%，验证每个模块都有贡献。  
- **局限性**：论文指出 CRFT 依赖于可靠的梯度或注意力信息流度量，在极端噪声或梯度消失的模型上可能难以准确定位关键表征；此外，关键表征的数量随任务复杂度增长，可能导致子空间维度需要适度提升，稍微增加参数开销。

### 影响与延伸思考
CRFT 把“表征层面的微调”从全局均匀改动转向了信息流驱动的局部强化，为 PEFT 在推理任务中的应用打开了新思路。后续工作（如 2024‑2025 年的几篇论文）开始探索基于注意力图的关键路径微调、以及在多模态模型中定位关键视觉/语言表征的类似方法。对想进一步研究的读者，可以关注以下方向：① 更鲁棒的信息流度量（如基于信息论的互信息估计）；② 将关键表征微调与自监督预训练结合，实现“预先发现关键节点”；③ 在大模型的多任务学习中共享关键表征子空间，以实现跨任务的参数共享。

### 一句话记住它
只在推理链关键的内部表征上做低秩微调，就能用极少参数大幅提升大模型的 CoT 推理能力。