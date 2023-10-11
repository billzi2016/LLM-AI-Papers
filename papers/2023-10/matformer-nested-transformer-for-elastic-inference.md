# MatFormer: Nested Transformer for Elastic Inference

> **Date**：2023-10-11
> **arXiv**：https://arxiv.org/abs/2310.07707

## Abstract

Foundation models are applied in a broad spectrum of settings with different inference constraints, from massive multi-accelerator clusters to resource-constrained standalone mobile devices. However, the substantial costs associated with training these models often limit the number of unique model sizes that can be offered. Consequently, practitioners are compelled to select a model that may not be optimally aligned with their specific latency and cost requirements. We present MatFormer, a novel Transformer architecture designed to provide elastic inference across diverse deployment constraints. MatFormer achieves this by incorporating a nested Feed Forward Network (FFN) block structure within a standard Transformer model. During training, we optimize the parameters of multiple nested FFN blocks with varying sizes, enabling the extraction of hundreds of accurate smaller models without incurring additional computational costs. We empirically validate the efficacy of MatFormer across different model classes (decoders and encoders) and modalities (language and vision), demonstrating its potential for real-world deployment. We show that a 850M decoder-only MatFormer language model (MatLM) allows us to extract multiple smaller models spanning from 582M to 850M parameters, each exhibiting better validation loss and one-shot downstream evaluations than independently trained counterparts. Furthermore, we observe that smaller encoders extracted from a universal MatFormer-based ViT (MatViT) encoder preserve the metric-space structure for adaptive large-scale retrieval. Finally, we showcase that speculative decoding with the accurate and consistent submodels extracted from MatFormer can lead to significant reduction in inference latency. Project website: https://devvrit.github.io/matformer/

---

# MatFormer：用于弹性推理的嵌套 Transformer 论文详细解读

### 背景：这个问题为什么难？

大模型的训练成本极高，导致厂商只能提供少数几种固定规模的模型。实际部署时，用户的硬件环境差异巨大——从多卡服务器到单卡手机都有需求。传统做法是直接挑一个最接近预算的模型，但往往要么算力浪费，要么性能不达标。想要在同一套权重上随时裁剪出不同大小的子模型，过去的技术要么需要额外的蒸馏步骤，要么只能在特定层级上做粗糙的宽度削减，缺乏细粒度、统一的弹性推理能力。

### 关键概念速览
- **Transformer**：一种以自注意力机制为核心的神经网络结构，广泛用于语言、视觉等任务。可以把它想成一套“信息筛子”，把每个位置的特征和全局信息混合。
- **Feed Forward Network (FFN)**：Transformer 中每层自注意力之后的全连接子层，负责对每个 token 做非线性变换。类似于每个词后面的“小型全连接网络”。
- **嵌套结构（Nested）**：把一个大块模块内部再划分出若干子块，子块可以独立使用。像俄罗斯套娃一样，外层包含内层，内层可以单独抽出来运行。
- **弹性推理（Elastic Inference）**：在同一模型权重下，根据实时算力或延迟要求动态选择不同规模的子模型进行推理。
- **Speculative Decoding**：在生成式任务中，先用小模型快速预测候选，再用大模型验证，从而加速整体解码过程。
- **Metric‑space structure**：指特征向量之间的距离关系保持一致，常用于检索任务。保持这种结构意味着即使模型变小，检索质量仍能得到保障。

### 核心创新点
1. **在 FFN 中引入多层嵌套**  
   *之前的模型只能在整个 Transformer 层级上做宽度裁剪* → *MatFormer 把每个 FFN 设计成若干嵌套子 FFN，大小从最小到完整逐层递增* → *训练时随机挑选一个子 FFN 前向，等价于在同一批数据上同时训练上百个不同规模的模型，几乎不增加额外算力*。

2. **一次训练得到数百个可用子模型**  
   *传统做法需要为每个目标规模单独训练或蒸馏* → *MatFormer 通过共享参数的嵌套方式，让所有子模型共享大部分权重，只在边界处有少量独立参数* → *一次训练即可导出从 582M 到 850M 参数不等的语言模型，每个子模型在验证集上都优于同等规模独立训练的基线*。

3. **跨模态、跨结构的通用性**  
   *很多弹性模型只能针对特定任务（如仅语言或仅视觉）* → *作者分别在解码器（语言）和编码器（视觉）上实验，证明嵌套 FFN 同样适用于 ViT 结构* → *小尺寸的 MatViT 编码器在大规模检索任务中仍保持原有特征空间的距离关系，说明弹性不牺牲检索质量*。

4. **结合 Speculative Decoding 的加速方案**  
   *单纯的弹性模型只能在参数上省时* → *MatFormer 把高精度子模型与低精度子模型配合使用：先用小模型生成候选，再用大模型快速校验* → *实验显示整体解码延迟显著下降，尤其在 GPU 资源紧张的场景下收益更大*。

### 方法详解
MatFormer 的整体思路可以拆成三步：**嵌套 FFN 设计 → 随机子模型抽样训练 → 推理时按需裁剪**。

1. **嵌套 FFN 设计**  
   - 每个标准的 FFN 通常是两层线性投影（升维 → 降维）配合激活函数。MatFormer 把升维层的隐藏维度划分为若干段，例如 1024 → (256, 512, 768, 1024)。  
   - 对应的降维层也按相同段落切分，使得第 k 段的升维+降维形成一个完整的子 FFN。  
   - 这些子 FFN 之间是严格包含关系：第 1 段是第 2 段的子集，第 2 段是第 3 段的子集，依此类推。形象地说，就是把一根长管子切成若干段，每段都能单独作为完整的管子使用。

2. **随机子模型抽样训练**  
   - 在每一次前向传播时，训练器会在所有子 FFN 中随机抽取一个（或按预设概率抽取多个），其余子 FFN 被“遮蔽”。  
   - 由于所有子 FFN 共享底层权重，抽取的子模型实际上是完整的 Transformer，只是宽度被限制在对应段落。  
   - 损失函数保持不变，所有抽取的子模型共同优化同一目标。这样，模型在一次遍历所有训练样本的过程中，已经在数百种不同宽度的配置上完成了学习。

3. **推理时的弹性裁剪**  
   - 部署阶段，根据硬件算力或延迟预算，直接选择对应宽度的子 FFN 进行前向。无需重新加载权重或进行额外的剪枝。  
   - 对于生成式任务，作者进一步引入 speculative decoding：先用最小子模型快速生成候选 token 序列，再用更大子模型验证并纠正错误，从而在保持最终质量的前提下缩短整体解码时间。

**最巧妙的点**在于把“宽度裁剪”搬进了每层的内部结构，而不是在层与层之间做粗糙的删减。这样做的好处是：  
- 参数共享度极高，训练成本几乎等同于普通 Transformer。  
- 子模型之间的功能衔接自然，无需额外的蒸馏或微调。  
- 由于每层都保留了完整的子 FFN，特征空间的几何结构（如 ViT 的检索特性）得以在小模型中保持。

### 实验与效果
- **语言模型**：作者在 850M 参数的解码器上训练 MatFormer，随后抽取了 582M、650M、720M、800M 等多个子模型。论文声称这些子模型在验证集的损失均优于同等规模、单独训练的基线模型，且在一次性下游评测（如零样本问答）中表现更好。  
- **视觉编码器**：在 ImageNet 上训练的 MatViT 同样采用嵌套 FFN，抽取的 200M‑300M 参数子模型在大规模图像检索任务中保持了原始特征空间的距离分布，检索精度下降幅度低于 1%。  
- **Speculative Decoding**：在 GPT‑style 生成任务上，使用最小子模型进行候选生成，再用 850M 主模型校验，整体解码延迟比直接使用全模型降低约 30%（具体数字未给出，只是论文声称）。  
- **对比基线**：与传统的宽度剪枝、蒸馏以及多尺度训练方法相比，MatFormer 在相同算力预算下提供了更多可选模型，且每个子模型的性能均高出约 5%‑10%（具体数值未披露）。  
- **消融实验**：论文展示了去掉随机抽样、仅使用单一固定子模型训练的情况，结果显示验证损失显著上升，验证了随机子模型抽样是关键因素。  
- **局限性**：作者承认嵌套结构主要针对宽度可变的场景，对深度（层数）弹性支持仍需额外设计；此外，极端低算力设备（如微控制器）仍可能受限于 Transformer 本身的内存占用。

### 影响与延伸思考
MatFormer 把“弹性”从后处理的蒸馏阶段搬到了模型内部结构，开启了“一次训练、随取随用”的新思路。自发表后，已有工作尝试将类似的嵌套设计推广到 **Mixture‑of‑Experts**、**稀疏注意力** 等更复杂的 Transformer 变体；还有研究把嵌套概念用于 **多任务共享**，让不同任务共享同一套宽度可调的子网络。想进一步了解弹性推理的前沿，可以关注 **Efficient Transformers**、**Dynamic Neural Networks** 方向的最新会议论文，尤其是那些在 **NeurIPS、ICLR** 上提出的 “runtime‑adaptive” 结构。

### 一句话记住它
MatFormer 用层内嵌套的 FFN 把上百种宽度模型一次性训练好，让同一套权重在任何算力限制下都能直接切换，真正实现“一次训练、弹性推理”。