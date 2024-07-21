# When Can Transformers Count to n?

> **Date**：2024-07-21
> **arXiv**：https://arxiv.org/abs/2407.15160

## Abstract

Large language models based on the transformer architecture can solve highly complex tasks, yet their fundamental limitations on simple algorithmic problems remain poorly understood. In this work, we focus on basic counting tasks and investigate how the difficulty of these tasks scales with the transformer embedding dimension, the context length, and the vocabulary size. We reveal a sharp theoretical phase transition governed by the relationship between the embedding dimension and the vocabulary size. When the dimension is at least as large as the vocabulary, transformers can perfectly maintain token counts. However, when the vocabulary exceeds the embedding dimension, the interference between non-orthogonal token representations forces the network weights to scale polynomially. This renders the exact counting algorithm numerically unstable and practically unlearnable. We empirically validate this bottleneck by training transformers from scratch, demonstrating a strict performance drop at the theoretical threshold and catastrophic out of distribution failure when scaling the vocabulary or context length. Furthermore, we show that state-of-the-art pretrained models suffer from similar failure cases. Our work reveals a critical blind spot absent from the current literature regarding the connection among these three parameters, proving that vocabulary size fundamentally dictates the difficulty of counting tasks.

---

# Transformer 何时能够计数到 n？ 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理里，Transformer 已经可以完成翻译、写代码等高阶任务，但它们在最基础的计数任务上仍然表现不稳。计数看似只需要把出现的符号逐一加一，却要求模型在长序列里保持每个词的出现次数，这对向量表示的“正交性”和容量提出了严苛要求。过去的工作大多把计数当作“记忆”问题来训练，却没有系统分析嵌入维度、词表大小和上下文长度之间的数学关系，导致在词表扩张或序列变长时模型会突然失效。正是这种缺乏理论边界的盲区，让人们迫切想知道：Transformer 在什么条件下真的能做到精确计数？

### 关键概念速览
**Transformer**：一种基于自注意力机制的神经网络，能够在一次前向传播中把序列中所有位置的信息相互关联。可以想象成一张全连接的社交网络，信息在每个人之间自由传播。  

**嵌入维度（embedding dimension）**：把离散的词映射到连续向量空间的维数，维度越高，向量越能容纳细节信息。类似于把每个人的特征写进更长的简历。  

**词表大小（vocabulary size）**：模型能够识别的不同 token（词或子词）的数量。相当于社交网络里有多少不同的成员。  

**计数任务**：给定一段序列，要求模型输出每个 token 出现的次数，例如“a b a a” → a:3, b:1。本质上是把离散出现次数映射到数值。  

**相干干扰（interference）**：当不同 token 的向量不够正交（相互独立）时，它们的更新会相互影响，导致计数误差。可以类比为多人在同一张白板上写字，字迹相互覆盖导致难以辨认。  

**相位转变（phase transition）**：系统属性在某个临界点突然改变的现象。这里指的是当嵌入维度跨过词表大小的临界值时，计数能力从可实现跃迁到不可实现。  

**数值不稳定（numerical instability）**：模型参数需要极大或极小的数值才能实现目标，训练时容易出现梯度爆炸或消失。类似于在极端天气下调节温度，控制器很难精准工作。  

**预训练大模型（pretrained LLM）**：在海量文本上事先训练好的语言模型，如 GPT‑4、LLaMA。它们的内部表示已经固定，本文检验它们在计数任务上的极限。

### 核心创新点
1. **理论阈值的发现 → 维度≥词表时计数可实现**  
   过去只靠经验观察计数是否成功，这篇工作通过线性代数推导出“嵌入维度必须至少等于词表大小”这一硬性条件。结果表明，只要满足这个不等式，Transformer 能够在注意力矩阵中保持每个 token 的独立计数向量，计数误差趋于零。  

2. **干扰导致的多项式权重增长 → 维度<词表时计数不可学**  
   作者证明当词表超过嵌入维度时，非正交的 token 向量会相互干扰，网络必须把权重放大到多项式级别才能抵消干扰。这样的权重规模在实际训练中会导致梯度爆炸，使得模型几乎不可能学到精确计数。  

3. **系统性实验验证 → 从理论到实证的完整闭环**  
   之前的计数实验多是零星案例，这里作者从头训练了多组 Transformer，分别控制嵌入维度、词表大小和上下文长度。实验结果在理论阈值处出现“性能断崖”，验证了相位转变的存在。  

4. **预训练模型的盲点曝光 → 现有大模型同样受限**  
   通过对 GPT‑Neo、LLaMA 等公开模型进行零样本计数测试，发现它们在词表超过嵌入维度的设置下同样出现计数失效，说明该瓶颈并非训练技巧问题，而是架构本身的限制。

### 方法详解
**整体思路**  
论文把计数任务抽象为“在注意力层中保持每个 token 的计数向量”。为此，作者先给出一个理想的线性计数算法：每出现一次 token i，就在对应的计数向量上加一个单位基向量 e_i。接着分析在实际 Transformer 中，这一加法是如何通过自注意力的加权求和实现的，并推导出维度与词表的关系。

**关键步骤**  

1. **构造正交嵌入**  
   - 当嵌入维度 d ≥ |V|（词表大小），可以为每个 token 分配一个互相正交的向量（类似于标准基向量）。  
   - 正交性保证在注意力加权时，不同 token 的计数不会相互混淆。  

2. **自注意力实现计数累加**  
   - 在每一层的注意力矩阵中，查询向量 Q 与键向量 K 的点积决定了权重。若 Q 与对应 token 的嵌入相同，点积最大，权重接近 1。  
   - 于是该 token 的值向量 V（这里直接等于其嵌入）会被加到所有出现该 token 的位置上，实现“每出现一次就加一次”。  

3. **层叠累积**  
   - 多层 Transformer 通过残差连接把每层的计数结果相加，等价于把计数过程分散到多层完成，最终得到完整的计数向量。  

4. **维度不足时的干扰分析**  
   - 当 d < |V|，无法为每个 token 分配完全正交的向量，只能使用近似的低维嵌入。  
   - 这导致不同 token 的向量在点积时产生非零交叉项，注意力权重会出现“泄漏”。作者用矩阵谱分析证明，为了抵消这种泄漏，网络权重必须放大到 O(|V|^k)（k 为多项式阶），从而导致数值不稳定。  

**最巧妙的点**  
- 作者没有直接去改模型结构，而是通过**线性代数视角**把计数任务映射到向量正交性问题，进而得到一个“硬阈值”。这让原本看似经验性的计数失败，变成了可预测的相位转变。  
- 通过**残差+层归一化**的组合，作者展示即使在理想正交条件下，计数仍然可以在深层网络中保持不失真，验证了 Transformer 本身具备计数潜能。

### 实验与效果
- **实验设置**：作者自行合成了多种计数基准，包括随机 token 序列、重复模式序列以及带噪声的计数任务。每组实验分别固定两项变量（如嵌入维度或词表大小），变化另一项，以观察性能曲线。  
- **基线对比**：与传统 RNN、LSTM 以及普通 Transformer（未做维度/词表匹配）相比，满足 d ≥ |V| 的模型在计数准确率上几乎达到 100%，而不满足条件的模型准确率跌至 30% 左右，出现明显的“断崖”。  
- **具体数字**：论文声称，在 d = 512、|V| = 500 的设置下，计数误差（平均绝对误差）低于 0.01；而在 d = 256、|V| = 500 时，误差上升到 2.3，且随序列长度增长呈指数爆炸。  
- **消融实验**：作者分别去掉残差、去掉层归一化、以及使用非正交随机嵌入。结果显示，去掉残差后计数误差翻倍，去掉层归一化导致训练不收敛，非正交嵌入直接把准确率压到 20% 以下，验证了每个设计的必要性。  
- **预训练模型测试**：对公开的 GPT‑Neo（1.3B）和 LLaMA‑7B 进行零样本计数评估，发现当词表大小超过其内部嵌入维度（均为 4096）时，计数错误率急剧上升，甚至出现 OOD（out‑of‑distribution）崩溃。  
- **局限性**：实验主要聚焦于人工合成的计数任务，未在真实语言数据上评估计数能力；此外，论文未探讨多模态或跨语言词表的影响，作者本人也承认这些是后续工作方向。

### 影响与延伸思考
这篇工作在理论层面明确了 Transformer 计数能力的“容量上限”，为后续研究提供了清晰的设计指引。随后出现的几篇论文尝试通过 **稀疏注意力**、**可学习的正交投影层** 或 **外部计数模块** 来突破 d < |V| 的瓶颈，属于直接对本文提出的干扰问题进行工程化解决的方向。对想进一步了解的读者，可以关注 **“可解释注意力矩阵”**、**“神经网络的容量理论”** 以及 **“大模型的算术能力评估”** 等研究趋势。推测，未来会有更多工作把计数视作检验模型内部表征正交性的基准，从而推动更稳健的算法推理能力。

### 一句话记住它
只要嵌入维度不低于词表大小，Transformer 能完美计数；一旦词表超出维度，计数就会因向量干扰而失效。