# MoH: Multi-Head Attention as Mixture-of-Head Attention

> **Date**：2024-10-15
> **arXiv**：https://arxiv.org/abs/2410.11842

## Abstract

In this work, we upgrade the multi-head attention mechanism, the core of the Transformer model, to improve efficiency while maintaining or surpassing the previous accuracy level. We show that multi-head attention can be expressed in the summation form. Drawing on the insight that not all attention heads hold equal significance, we propose Mixture-of-Head attention (MoH), a new architecture that treats attention heads as experts in the Mixture-of-Experts (MoE) mechanism. MoH has two significant advantages: First, MoH enables each token to select the appropriate attention heads, enhancing inference efficiency without compromising accuracy or increasing the number of parameters. Second, MoH replaces the standard summation in multi-head attention with a weighted summation, introducing flexibility to the attention mechanism and unlocking extra performance potential. Extensive experiments on ViT, DiT, and LLMs demonstrate that MoH outperforms multi-head attention by using only 50%-90% of the attention heads. Moreover, we demonstrate that pre-trained multi-head attention models, such as LLaMA3-8B, can be further continue-tuned into our MoH models. Notably, MoH-LLaMA3-8B achieves an average accuracy of 64.0% across 14 benchmarks, outperforming LLaMA3-8B by 2.4% by utilizing only 75% of the attention heads. We believe the proposed MoH is a promising alternative to multi-head attention and provides a strong foundation for developing advanced and efficient attention-based models.

---

# MoH：将多头注意力视为头部混合模型 论文详细解读

### 背景：这个问题为什么难？

Transformer 的核心是多头注意力（Multi‑Head Attention），它把同一个查询向量分别送进若干个注意力头，再把所有头的输出加和。虽然这种设计让模型可以并行捕捉不同的关系，但也带来了两大痛点：  
1）并不是所有头在每个 token 上都同等重要，很多头在实际推理时几乎是“闲置”的，导致算力浪费。  
2）所有头的输出都被强行等权相加，缺少对不同头重要性的动态调节，限制了注意力机制的表达灵活性。于是，提升效率的同时不牺牲甚至提升精度，成了一个亟待突破的难题。

### 关键概念速览
- **多头注意力（Multi‑Head Attention）**：把同一查询拆成若干子查询，分别算注意力后再相加，类似把一张图片切成多块分别看，再把结果拼回去。  
- **注意力头（Attention Head）**：每个子查询对应的注意力计算单元，负责捕捉一种特定的模式或关系。  
- **Mixture‑of‑Experts（MoE）**：让多个专家模型并行工作，每个输入只激活一部分专家，像餐厅里点菜让厨师挑选擅长的菜式。  
- **专家路由（Expert Routing）**：决定哪个专家（或注意力头）被激活的机制，常用稀疏 gating 网络实现。  
- **加权求和（Weighted Sum）**：对多个向量求和时赋予不同权重，类似给不同配料不同的调味比例。  
- **继续微调（Continue‑tuning）**：在已有预训练模型上再训练，使其适配新结构或新任务。  
- **ViT / DiT / LLM**：分别指视觉 Transformer、扩散模型的 Transformer 以及大语言模型，都是 Transformer 的不同应用场景。  

### 核心创新点
1. **把多头注意力写成求和形式** → 作者把原本的并行计算显式化为“所有头的输出相加”，为后续引入加权提供数学依据。 → 这一步让我们可以直接在求和上套上 MoE 的 gating，打开了对头部重要性进行动态选择的大门。  
2. **将注意力头视作 MoE 中的专家** → 传统 MoE 在层级或子网络上划分专家，这里把每个注意力头当作专家，并用稀疏路由让每个 token 只激活最合适的头。 → 结果是推理时只用到 50%‑90% 的头，算力大幅下降，却不牺牲精度。  
3. **用加权求和取代统一求和** → 在每个 token 的输出上，引入由路由网络产生的权重，对不同头的贡献进行调节。 → 这让注意力机制拥有了“可调的混合比例”，在实验中进一步提升了性能。  
4. **直接在已有的多头注意力模型上继续微调** → 作者证明了预训练好的 LLaMA3‑8B 等模型可以无缝迁移到 MoH 结构，只需继续微调即可。 → 这意味着不必从头训练，大幅降低了研发成本。

### 方法详解
**整体思路**：先把标准多头注意力的计算拆解为“每个头的输出 + 求和”。随后在求和前插入一个稀疏路由模块，让每个 token 根据自己的特征挑选并加权若干头的输出，最后再把加权后的结果相加得到最终的注意力向量。

**步骤拆解**：

1. **标准多头注意力的显式求和**  
   - 输入的查询、键、值先分别线性映射成 H 组（H 为头数）子空间。  
   - 每组子空间内部做 Scaled Dot‑Product Attention，得到 H 个向量。  
   - 传统做法是把这 H 个向量直接相加（或拼接后再线性变换），这里改写为显式的求和公式，为后面的加权提供空间。

2. **稀疏路由网络（Gate）**  
   - 对每个 token，使用一个轻量的 MLP（或线性层+softmax）产生 H 维的路由分数。  
   - 通过 Top‑k 或阈值筛选，只保留最重要的 k（如 0.5‑0.9H）个头的分数，其余置零，实现稀疏激活。  
   - 这些分数随后归一化，成为该 token 对每个被激活头的**权重**。

3. **加权求和**  
   - 把每个头的注意力输出乘以对应的权重（若该头未被激活则权重为 0），再相加。  
   - 这一步相当于“让每个 token 自己决定该听哪几位专家的建议，并给出信任度”。  

4. **继续微调**  
   - 对已有的预训练模型，只需把原来的多头注意力层替换成上述 MoH 结构，保持其它参数不变。  
   - 使用原任务或下游任务的少量数据继续训练，让路由网络学会在新任务上挑选合适的头。

**关键细节**：  
- 路由网络的参数量极小（通常只占整体模型的 <1%），所以整体参数几乎不变。  
- 稀疏激活带来的算力节省主要体现在注意力矩阵的计算上，因为未被激活的头可以直接跳过。  
- 作者在实验中发现，权重的学习比硬性“只用前 k 个头”更稳健，模型能够在不同层、不同 token 上自适应选择。

### 实验与效果
- **测试平台**：视觉 Transformer（ViT）、扩散模型的 Transformer（DiT）以及大语言模型（LLM）如 LLaMA3‑8B。  
- **基准对比**：与原始多头注意力模型直接对比，MoH 在使用 50%‑90% 头的情况下整体精度不降反升。  
- **具体数字**：在 LLaMA3‑8B 上继续微调后，MoH‑LLaMA3‑8B 在 14 项基准测试上平均得到 64.0% 的准确率，比原始 LLaMA3‑8B 高出 2.4%，而只用了 75% 的注意力头。  
- **消融实验**：论文报告了去掉路由权重（仅做稀疏激活）和去掉稀疏（全头加权）两种设置，发现两者单独使用均能提升效率，但只有两者结合才达到最佳的精度‑效率平衡。  
- **局限性**：作者承认路由网络在极端低头数（如 <30%）时会出现不稳定；此外，稀疏激活在硬件上仍需特定的实现才能完全发挥算力优势。  

### 影响与延伸思考
MoH 把 MoE 思想搬进注意力头部，为“细粒度专家选择”提供了可行路径。自论文发布后，已有工作尝试在跨模态 Transformer、稀疏自注意力以及大模型压缩领域加入类似的头部路由机制（如“Head‑Prune‑MoE”）。未来可以进一步探索：  
- **更高效的路由实现**（比如基于硬件的稀疏矩阵加速）。  
- **动态头数调节**：让模型在不同输入长度或任务难度下自动改变激活头的比例。  
- **与结构化剪枝结合**：把 MoH 的稀疏选择与后续的硬件剪枝一起使用，进一步压缩模型体积。  

如果想深入了解，建议关注近期在 NeurIPS、ICLR 上出现的 “Sparse Mixture of Experts for Transformers” 系列论文，它们在 MoH 的基础上加入了更复杂的负载均衡和专家多样性约束。

### 一句话记住它
**MoH 把每个注意力头当成专家，让每个 token 用路由挑选并加权少数关键头，从而在更少计算下实现更高精度。**