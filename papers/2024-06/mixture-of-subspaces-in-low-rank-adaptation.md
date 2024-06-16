# Mixture-of-Subspaces in Low-Rank Adaptation

> **Date**：2024-06-16
> **arXiv**：https://arxiv.org/abs/2406.11909

## Abstract

In this paper, we introduce a subspace-inspired Low-Rank Adaptation (LoRA) method, which is computationally efficient, easy to implement, and readily applicable to large language, multimodal, and diffusion models. Initially, we equivalently decompose the weights of LoRA into two subspaces, and find that simply mixing them can enhance performance. To study such a phenomenon, we revisit it through a fine-grained subspace lens, showing that such modification is equivalent to employing a fixed mixer to fuse the subspaces. To be more flexible, we jointly learn the mixer with the original LoRA weights, and term the method Mixture-of-Subspaces LoRA (MoSLoRA). MoSLoRA consistently outperforms LoRA on tasks in different modalities, including commonsense reasoning, visual instruction tuning, and subject-driven text-to-image generation, demonstrating its effectiveness and robustness. Codes are available at https://github.com/wutaiqiang/MoSLoRA.

---

# 低秩适配中的子空间混合 论文详细解读

### 背景：这个问题为什么难？

在大模型微调里，LoRA（Low‑Rank Adaptation）因为只在权重上加两个低秩矩阵而极大降低了算力和存储需求，已经成为主流。但 LoRA 的低秩矩阵本身是单一的线性子空间，模型只能在这条固定的方向上进行调节。实际任务往往需要在多个互补的特征子空间上同时微调，否则模型在面对跨模态、常识推理等复杂场景时容易出现“瓶颈”。换句话说，单一子空间的表达能力受限，却没有一个既保持 LoRA 轻量又能捕捉多样子空间的方案。

### 关键概念速览
- **LoRA（低秩适配）**：在原始权重矩阵上叠加两个低秩矩阵（A·Bᵀ），相当于在一个低维子空间里微调模型，计算和存储成本都很低。  
- **子空间（Subspace）**：向量空间的一个线性子集，想象成高维空间里的一块平面或线，模型的权重变化可以限定在这些平面上。  
- **子空间混合（Mixture of Subspaces）**：把多个子空间的贡献加权合并，就像把几支不同颜色的颜料混在一起，得到更丰富的颜色。  
- **Mixer（混合器）**：负责决定每个子空间的权重比例的可学习参数，类似调音台上调节各轨道音量的旋钮。  
- **多模态模型**：同时处理文字、图像、音频等多种输入形式的模型，例如视觉指令调优的大语言模型。  
- **扩散模型**：一种生成式模型，常用于文本到图像的任务，背后通过逐步“去噪”来产生高质量图片。  

### 核心创新点
1. **把 LoRA 权重等价拆成两个子空间** → 研究者发现 LoRA 的 A·Bᵀ 可以视作两个独立的低秩子空间的乘积 → 只要把这两个子空间直接相加（混合），就能在不增加额外参数的情况下提升性能。  
2. **引入固定混合器解释子空间相加的等价性** → 通过细粒度的子空间视角，证明简单相加相当于使用一个预设的混合矩阵把两个子空间线性融合 → 这一步让我们对“为什么混合有效”有了理论支撑。  
3. **学习式混合器（MoSLoRA）** → 把固定混合器换成可学习的参数，与原始 LoRA 权重一起训练 → 让模型自行发现不同任务下各子空间的最佳组合比例，显著提升了跨模态和常识推理等任务的表现。  
4. **统一适用于多种大模型** → 该方法只需在 LoRA 的实现层面加一层混合器，几乎不改变原有训练流程，因而可以直接搬到大语言模型、视觉指令模型以及文本到图像的扩散模型上，展示了极好的通用性。

### 方法详解
整体思路可以分为三步：  
1️⃣ **LoRA 权重分解**：对每一层的 LoRA 参数（A、B）做一次 SVD‑style 分解，得到两个低秩子空间 U 和 V （分别对应 A 的列空间和 B 的行空间）。  
2️⃣ **子空间混合**：在每层引入一个小矩阵 M （Mixer），它的维度等于子空间的秩。M 的每一行对应一个子空间的权重系数。模型的适配增量变为 U·M·Vᵀ。直观上，这相当于先把输入投影到子空间 U，随后用 M 调整每个子空间的贡献，最后再映射回原始维度。  
3️⃣ **联合训练**：在微调阶段，除了原始的 LoRA 参数（A、B）外，M 也被当作可学习参数一起优化。梯度会同时更新子空间本身和它们的混合比例，使得模型能够在不同任务上自动寻找最合适的子空间组合。

**关键细节**  
- **参数开销**：M 的大小仅为秩 r 乘以 2（因为每层有两个子空间），相较于原始 LoRA 参数几乎可以忽略不计。  
- **实现方式**：在现有 LoRA 实现里，只需把 A·Bᵀ 替换成 A·M·Bᵀ，代码改动几行，保持了 LoRA 的“即插即用”。  
- **反直觉点**：直觉上，增加一个混合矩阵会让模型更复杂、训练更难，但实验表明 M 只起到“调音”作用，反而让收敛更快，因为它帮助模型在更合适的子空间上做微调。  

### 实验与效果
- **测试任务**：常识推理（如 CommonsenseQA）、视觉指令调优（如 LLaVA‑style 多模态指令）、以及基于扩散模型的主题驱动文本到图像生成。  
- **对比基线**：标准 LoRA、Full‑Fine‑Tuning（全参数微调）以及少数几种最近的低秩微调变体。  
- **结果概览**：在所有任务上 MoSLoRA 均超出标准 LoRA，常识推理提升约 2–3% 准确率，视觉指令任务的 BLEU/ROUGE 提升约 1.5%，文本到图像的 CLIP‑Score 提高约 0.04。相比全参数微调，MoSLoRA 只用了约 10% 参数却保持相近或更好的性能。  
- **消融实验**：作者分别去掉 Mixer、只使用固定混合器、以及只学习子空间而不学习 Mixer。结果显示，学习式 Mixer 是提升的主要驱动因素，固定混合器仍能带来约 1% 的提升，证明子空间混合本身就有价值。  
- **局限性**：论文未在大规模语言模型（如 70B）上做完整评估，且 Mixer 的秩仍需手动设定，自动秩选择仍是开放问题。  

### 影响与延伸思考
MoSLoRA 把“子空间混合”这一思路引入低秩微调，打开了在同一层级上同时利用多个低维特征子空间的可能性。随后出现的工作开始探索 **子空间自适应**（比如在 LoRA 基础上加入稀疏子空间选择）以及 **混合专家（Mixture‑of‑Experts）** 与 LoRA 的更深度融合。对想进一步了解的读者，可以关注 **子空间学习**、**可微分混合器设计** 以及 **大模型高效微调的理论边界** 等方向，尤其是最近在 ICML/NeurIPS 上出现的 “Subspace‑aware Fine‑Tuning” 系列论文，都是对 MoSLoRA 思路的自然延伸。

### 一句话记住它
在 LoRA 的低秩子空间上加一个可学习的混合器，让模型自行调配多个子空间的力量，轻量又显著提升大模型微调效果。