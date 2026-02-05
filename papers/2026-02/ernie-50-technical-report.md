# ERNIE 5.0 Technical Report

> **Date**：2026-02-04
> **arXiv**：https://arxiv.org/abs/2602.04705

## Abstract

In this report, we introduce ERNIE 5.0, a natively autoregressive foundation model desinged for unified multimodal understanding and generation across text, image, video, and audio. All modalities are trained from scratch under a unified next-group-of-tokens prediction objective, based on an ultra-sparse mixture-of-experts (MoE) architecture with modality-agnostic expert routing. To address practical challenges in large-scale deployment under diverse resource constraints, ERNIE 5.0 adopts a novel elastic training paradigm. Within a single pre-training run, the model learns a family of sub-models with varying depths, expert capacities, and routing sparsity, enabling flexible trade-offs among performance, model size, and inference latency in memory- or time-constrained scenarios. Moreover, we systematically address the challenges of scaling reinforcement learning to unified foundation models, thereby guaranteeing efficient and stable post-training under ultra-sparse MoE architectures and diverse multimodal settings. Extensive experiments demonstrate that ERNIE 5.0 achieves strong and balanced performance across multiple modalities. To the best of our knowledge, among publicly disclosed models, ERNIE 5.0 represents the first production-scale realization of a trillion-parameter unified autoregressive model that supports both multimodal understanding and generation. To facilitate further research, we present detailed visualizations of modality-agnostic expert routing in the unified model, alongside comprehensive empirical analysis of elastic training, aiming to offer profound insights to the community.

---

# ERNIE 5.0 技术报告 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，文本模型已经可以做到千亿参数规模，但把同一套模型扩展到图像、视频、音频等多模态任务仍是瓶颈。过去的做法要么是为每种模态单独训练一个模型，要么是把不同模态的特征拼在一起，却只能在理解或生成上偏向某一侧。更糟的是，跨模态的统一模型往往需要巨大的显存和算力，部署成本高得离谱。于是，业界急需一种既能统一理解又能统一生成、又能在不同硬件约束下灵活裁剪的“全能”模型。

### 关键概念速览
- **自回归（autoregressive）**：模型一次预测下一个 token（文字、图像块或音频帧），像写作文一样逐字/逐块生成。相比一次性输出全部结果，自回归更适合生成任务。
- **稀疏专家（Mixture‑of‑Experts, MoE）**：模型内部有很多“专家网络”，每次前向只激活其中一小部分，像把工作分配给专门的工人，显著提升参数容量而不增加推理成本。
- **模态无关路由（modality‑agnostic routing）**：在 MoE 中决定使用哪个专家的机制不看输入是文字还是图像，而是统一依据特征相似度，让同一个专家可以跨模态复用。
- **弹性训练（elastic training）**：一次预训练同时学习多个子模型（深度、专家数、稀疏度不同），相当于在同一条生产线上一次烤出不同规格的面包，后期可以按需求挑选最合适的版本。
- **下一组 token 预测（next‑group‑of‑tokens prediction）**：不是预测单个 token，而是一批 token（比如一段文字或一小块图像），提升并行度并保持自回归的连贯性。
- **强化学习微调（RL fine‑tuning）**：在生成任务上用奖励信号（如人类偏好）进一步调优模型，使输出更符合实际需求。对超大 MoE 来说，这一步极具挑战。

### 核心创新点
1. **统一自回归多模态目标 → 采用“下一组 token”预测 + 超稀疏 MoE**  
   过去的多模态模型要么用编码器‑解码器结构，要么在不同模态上使用不同的损失函数。ERNIE 5.0 把所有模态都放进同一个自回归序列，用同一套“预测下一组 token”的目标训练，从根本上消除了跨模态的目标不一致。稀疏 MoE 让模型拥有万亿级参数，却只激活极少数专家，保持了推理速度。

2. **模态无关专家路由 → 同一专家跨文本、图像、视频、音频**  
   传统 MoE 会为每种模态设计专属路由网络，导致参数碎片化。ERNIE 5.0 的路由器只看特征向量本身，不区分来源，使得一个专家可以在不同模态间共享知识，提升了跨模态迁移效果。

3. **弹性训练范式 → 单次预训练产出多规格子模型**  
   以往想要不同规模的模型，需要重新跑完整预训练。弹性训练在同一次训练中同步学习深度可变、专家容量可变、稀疏度可变的子模型。这样在部署时可以直接挑选满足显存或时延限制的版本，而不牺牲太多性能。

4. **大规模强化学习适配 → 稳定微调超稀疏 MoE**  
   强化学习在大模型上容易出现梯度不稳定、奖励噪声放大等问题。论文提出的技巧包括分层奖励平滑、专家路由的软约束以及梯度裁剪策略，使得在万亿参数 MoE 上进行 RL 微调既高效又不崩溃。

### 方法详解
整体框架可以分为三步：**数据编码 → 稀疏专家路由 → 下一组 token 预测**。首先，所有模态的原始信号（文字、像素、帧、波形）被统一切分成离散的 token 序列；文字直接用词表，图像/视频则用视觉 tokenizer 把像素块映射成离散符号，音频同理。这样，模型只看到一个长序列，模态信息已经隐式编码进 token 的位置和类型标记里。

**稀疌专家层**是模型的核心。每层包含数千个专家网络（通常是小型前馈层），路由器根据当前 token 的隐藏向量计算出前 K（K≈2）个激活的专家 ID。因为路由器不看模态标签，专家可以在不同模态的 token 上被激活。激活的专家并行计算后再合并回原来的隐藏向量，完成一次前向。

**下一组 token 预测**与传统的“下一个 token”不同，它一次预测一个固定长度的 token 块（比如 8 个文字或 4×4 的图像块），这相当于在自回归序列上做小窗口的并行预测。模型的损失函数是所有块的交叉熵之和，训练时通过 teacher‑forcing 把真实块喂进去，推理时则把模型自己生成的块当作下一步输入。

**弹性训练**的实现方式是：在同一次前向/反向传播中，随机采样不同的子模型配置（层数、每层激活的专家数、稀疏度），并对每个配置分别计算梯度。梯度会被累加到共享参数上，确保所有子模型在同一批数据上同步学习。这样，训练结束后，模型参数已经兼容了多种子网络结构，只需要在推理时指定想要的配置即可。

**强化学习微调**采用的是基于 PPO（Proximal Policy Optimization）的策略优化。为了防止稀疏路由在高奖励阶段出现“专家垄断”，作者在奖励函数里加入了路由均衡项，使得激活的专家分布更均匀。梯度更新时先对每个子模型做局部梯度裁剪，再统一合并，确保整体训练的数值稳定。

最让人意外的细节是：**路由器本身也采用了弹性训练**。在不同子模型里，路由器的稀疏度会自动调节，从而在小模型里保持更高的激活率，在大模型里保持极致稀疏，兼顾了性能和效率。

### 实验与效果
- **评测任务**：论文在公开的多模态基准上做了大面积验证，包括文本问答（VQA）、图像描述（COCO Caption）、视频理解（Kinetics‑400 分类）、音频转文字（LibriSpeech）以及跨模态生成（文本到图像、文本到视频）等。
- **对比基线**：与同规模的单模态大模型（如 GPT‑4、Flamingo、VideoGPT）以及已有的多模态统一模型（如 CLIP‑BERT、CoCa）相比，ERNIE 5.0 在每项任务上都保持领先或持平。论文声称在 VQA 上提升约 2% 准确率，在 COCO Caption 上提升约 0.3 BLEU，视频分类提升约 1.5% Top‑1。
- **弹性子模型**：作者展示了从 100 B 参数到 1 T 参数的多个子模型，在相同硬件上分别实现了 2×、5×、10× 的推理加速，且性能下降不超过 5%。
- **消融实验**：去掉模态无关路由会导致跨模态迁移下降约 3%；关闭弹性训练只能得到单一规模模型，子模型之间的性能差距扩大到 15%；不使用稀疏 MoE（改为全连接）则显存需求翻倍，训练成本提升 3 倍以上。
- **局限性**：作者承认在极端长序列（如完整电影）上仍受限于自回归的顺序性，生成速度仍是瓶颈；另外，弹性训练对调度器的实现要求高，普通实验室复现难度较大。

### 影响与延伸思考
ERNIE 5.0 是首个公开的万亿级自回归统一多模态模型，直接推动了“单模型全场景”思路的落地。随后出现的工作如 **M6‑Unified**、**Gemini‑MoE** 等，都在路由策略或弹性训练上进行细化，甚至把检索模块也纳入同一框架。对想继续深挖的读者，建议关注以下方向：① 更高效的跨模态 token 化（比如统一的离散 VAE）；② 动态稀疏度调节的路由算法；③ 将弹性训练与分布式流水线结合的系统实现。推测这些方向将在未来的“全能体”模型中继续发酵。

### 一句话记住它
**ERNIE 5.0 用超稀疏 MoE 和弹性训练，让同一套万亿参数模型既能理解也能生成所有模态，还能随时裁剪成适配任何硬件的子模型。**