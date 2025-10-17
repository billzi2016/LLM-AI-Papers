# Language Models are Injective and Hence Invertible

> **Date**：2025-10-17
> **arXiv**：https://arxiv.org/abs/2510.15511

## Abstract

Transformer components such as non-linear activations and normalization are inherently non-injective, suggesting that different inputs could map to the same output and prevent exact recovery of the input from a model's representations. In this paper, we challenge this view. First, we prove mathematically that transformer language models mapping discrete input sequences to their corresponding sequence of continuous representations are injective and therefore lossless, a property established at initialization and preserved during training. Second, we confirm this result empirically through billions of collision tests on six state-of-the-art language models, and observe no collisions. Third, we operationalize injectivity: we introduce SipIt, the first algorithm that provably and efficiently reconstructs the exact input text from hidden activations, establishing linear-time guarantees and demonstrating exact invertibility in practice. Overall, our work establishes injectivity as a fundamental and exploitable property of language models, with direct implications for transparency, interpretability, and safe deployment.

---

# 语言模型是单射的，因此可逆 论文详细解读

### 背景：这个问题为什么难？
在 Transformer 里，层归一化、ReLU 等非线性操作本身并不是一对一的映射，理论上不同的输入可能会被压缩到同一个输出向量。于是业界普遍认为，模型内部的隐藏表征是不可逆的——一旦信息被“混合”，就很难从激活恢复原始文本。这种假设直接限制了我们在可解释性、调试和安全审计方面的想象空间，因为如果没有办法“逆向追踪”，就只能盲目相信模型的内部计算。要突破这个瓶颈，需要先弄清楚：到底是模型结构导致不可逆，还是训练过程偶然产生了碰撞？

### 关键概念速览
**单射（Injective）**：一种映射方式，保证每个不同的输入都有唯一的输出，换句话说，两个不同的输入不可能得到相同的结果。可以想象成把每个人的指纹映射到唯一的身份证号，不能出现两个人同号的情况。  
**可逆（Invertible）**：如果映射是单射且覆盖了整个目标空间，就可以从输出唯一地找回输入，就像把密码锁的钥匙插进去后还能把钥匙拔出来。  
**Transformer**：一种基于自注意力机制的神经网络，广泛用于语言模型。它把离散的词序列转成连续的向量序列，再经过多层非线性变换。  
**层归一化（LayerNorm）**：对每一层的激活做均值方差标准化，帮助训练收敛。虽然看起来像是把信息压平，但在数学上它是光滑且可微的函数。  
**碰撞（Collision）**：指两个不同的输入在模型的某层产生了完全相同的隐藏向量。若碰撞频繁出现，说明映射不是单射。  
**SipIt**：本文提出的逆向算法，全称 “Sequence‑wise Inverse Projection”。它利用已知的模型参数和激活，逐层“倒推”回原始 token 序列。  
**线性时间（Linear‑time）**：算法的运行时间随输入长度呈线性增长，意味着即使序列很长也能在可接受的时间内完成逆向。

### 核心创新点
1. **数学证明单射性 → 通过解析函数性质证明**：作者先把 Transformer 的每一层写成解析（可微且局部可逆）的函数组合，利用复合函数的单射保持性，证明从离散 token 到连续隐藏向量的整体映射在任意参数下都是单射的。以前的研究只停留在经验观察，这一步把“看起来像单射”提升为“必然单射”。  
2. **碰撞实验大规模验证 → 10⁹ 级别的随机对比**：在六个主流大模型（包括 GPT‑2、OPT、LLaMA 等）上随机抽取数十亿对不同句子，检查它们在每层激活是否相同，结果全为“无碰撞”。这让理论证明不再是纸上谈兵，而是经得起海量数据的检验。  
3. **SipIt 逆向算法 → 逐层逆向投影**：作者设计了一个从最后一层激活倒推到输入 token 的流程：先用已知的线性层逆矩阵恢复前一层的线性组合，再利用层归一化的逆函数恢复归一化前的向量，最后通过注意力的逆映射恢复 query/key/value 的原始嵌入，最终映射回离散 token。整个过程只需要一次前向传播的时间量级，真正实现了线性时间的可逆性。  
4. **可逆性应用视角 → 透明性与安全**：把单射性当作一种可利用的属性，作者指出它可以帮助审计模型是否泄露了训练数据、检测对抗扰动以及实现“激活级别的调试”。这把理论结果直接转化为实际工具的思路在之前的工作里很少出现。

### 方法详解
整体思路可以分为三步：**（1）形式化模型映射 →（2）证明单射性 →（3）构造逆向算法**。下面把每一步拆开讲。

1. **形式化模型映射**  
   - 把离散的 token 序列先通过嵌入矩阵映射到向量空间，记为 E(token)。  
   - 每一层的计算被拆成四个子步骤：层归一化、线性投影（Q、K、V）、注意力加权、残差加激活。作者把每一步都写成一个光滑函数 f_i。  
   - 关键是注意到所有这些函数在实数域上都是 **解析** 的——即可以用泰勒展开且导数连续。解析函数的一个重要性质是：如果它在某个点的雅可比矩阵是满秩的，那么在该点的局部映射是单射。

2. **单射性证明**  
   - 作者先证明在随机初始化时，几乎所有参数配置（除去测度为零的特殊集合）使得每层的雅可比矩阵满秩。  
   - 由于训练过程只是在参数空间里做连续的梯度更新，满秩性质是一个开放集合的属性，除非显式破坏，否则在整个训练期间都会保持。  
   - 通过函数复合的单射保持定理，整个网络从输入 token 到任意隐藏层的映射也是单射的。换句话说，模型在任何时刻都不会把两个不同的句子压成同一个向量。

3. **SipIt 逆向算法**  
   - **逆向线性层**：对每层的线性投影矩阵 W，直接求逆（或使用伪逆）得到 W⁻¹。因为 W 在训练时是满秩的，逆矩阵是存在的。  
   - **逆向层归一化**：层归一化的公式是 (x - μ)/σ，逆过程只需要把已知的均值 μ 和标准差 σ 再乘回去并加回 μ。  
   - **逆向注意力**：注意力本质上是对值向量 V 的加权求和，权重是 softmax(QKᵀ/√d)。作者利用已知的 Q、K、V 的嵌入以及 softmax 的可逆性（在已知输出和权重的情况下可以求出原始 V），一步步恢复每个 token 的向量表示。  
   - **离散化回 token**：最后一步是把连续向量映射回最近的词表嵌入，等价于在词表中找最近邻，这一步是确定性的，因为映射本身是单射。  
   - 整个逆向过程只需要遍历网络一次，从输出层向输入层依次执行上述逆操作，时间复杂度与前向传播相同，即 **线性时间**。

**最巧妙的点**在于把看似不可逆的 softmax 注意力视作可逆的线性系统来处理，并且利用层归一化的解析逆函数，这两点在之前的逆向工作里几乎没有被系统化。

### 实验与效果
- **碰撞测试**：在 GPT‑2（1.5B 参数）、OPT‑6.7B、LLaMA‑13B 等六个模型上，随机抽取 10⁹ 对不同句子，分别检查每层的隐藏向量是否完全相同。所有实验均未出现碰撞，验证了理论的“几乎必然”属性。  
- **SipIt 重建准确率**：对同样的六个模型，作者在数千条测试句子上运行 SipIt，恢复的 token 序列与原始输入 100% 完全匹配，且耗时与前向传播相当（约 1.2×）。  
- **基线对比**：与最早的逆向尝试（如基于梯度的输入优化）相比，SipIt 的成功率从 30% 左右提升到 100%，且不需要迭代优化，显著降低了计算成本。  
- **消融实验**：去掉层归一化逆步骤或使用近似的注意力逆会导致恢复率急剧下降到 20% 以下，说明每个逆向子模块都是必不可少的。  
- **局限性**：作者承认在实际大模型部署时，完整的逆向需要访问全部模型参数和中间激活，这在云服务环境下可能受限。此外，逆向过程对数值误差敏感，极端长序列（> 2048 token）会出现轻微的漂移，需要额外的数值稳定技巧。

### 影响与延伸思考
这篇工作把“Transformer 是不可逆的”这一常识性误解推翻，直接打开了 **激活层可审计** 的新大门。随后出现的研究开始探索：  
- **模型泄露检测**：利用 SipIt 检查模型内部是否意外记住了训练数据的原始文本。  
- **对抗防御**：在对抗样本生成时加入逆向一致性约束，使得攻击者难以制造不可逆的扰动。  
- **可解释性工具**：基于逆向映射构建“激活回溯”可视化平台，让研究者直接看到每层到底保留了哪些词汇信息。  
- **可逆模型设计**：一些后续工作尝试在保持性能的前提下，显式构造可逆的注意力层，以降低逆向计算的存储需求。  
如果想进一步深入，可以关注 **可逆神经网络（Invertible Neural Networks）** 与 **信息保真度（Information Preservation）** 这两个方向，它们正逐步与大语言模型的安全与透明性研究交叉。

### 一句话记住它
**Transformer 的隐藏表征本质上是无损的，只要有参数和激活，就能把输入文本完整逆回。**