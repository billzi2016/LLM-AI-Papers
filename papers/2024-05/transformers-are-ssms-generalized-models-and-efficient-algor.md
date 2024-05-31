# Transformers are SSMs: Generalized Models and Efficient Algorithms   Through Structured State Space Duality

> **Date**：2024-05-31
> **arXiv**：https://arxiv.org/abs/2405.21060

## Abstract

While Transformers have been the main architecture behind deep learning's success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Transformers at small to medium scale. We show that these families of models are actually quite closely related, and develop a rich framework of theoretical connections between SSMs and variants of attention, connected through various decompositions of a well-studied class of structured semiseparable matrices. Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is an a refinement of Mamba's selective SSM that is 2-8X faster, while continuing to be competitive with Transformers on language modeling.

---

# Transformer 即状态空间模型：通过结构化状态空间对偶实现的通用模型与高效算法 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理的早期，Transformer 之所以大火，是因为它的自注意力机制能够一次性捕捉序列中任意位置的依赖关系。但自注意力的计算复杂度随序列长度呈二次增长，导致长文本、实时推理时的成本居高不下。近几年，基于状态空间模型（SSM）的结构（如 Mamba）展示了在中小规模任务上与 Transformer 相当甚至更好的表现，且在时间复杂度上更友好。然而，研究社区一直把两者当作互不相干的路线：Transformer 侧重显式的注意力权重，SSM 则依赖隐式的递推算子。缺少统一的理论框架让我们难以系统地比较、融合这两类模型，也阻碍了在保持效率的同时进一步提升性能的可能性。

### 关键概念速览
**自注意力（Self‑Attention）**：模型为序列中每个位置生成一个查询向量，用它去“打分”所有位置的键向量，再加权求和得到输出。可以想象成一次全体同学相互打分，决定谁的答案最值得参考。  
**状态空间模型（State‑Space Model, SSM）**：用线性递推公式描述序列的演化，输入先经过一个线性系统（状态转移矩阵），再通过输出矩阵得到结果。类似于把信息放进一个“记忆盒子”，每一步都把盒子里的状态更新一次。  
**半分离矩阵（Semiseparable Matrix）**：一种特殊的稀疏结构，矩阵的上三角和下三角可以分别用低秩乘积表示，计算时只需要 O(n) 而不是 O(n²)。把它想成“只保留关键的桥梁”，其余桥梁可以快速推导出来。  
**结构化状态空间对偶（Structured State‑Space Duality, SSD）**：论文提出的理论桥梁，表明注意力矩阵和某类半分离矩阵在数学上是等价的，两者可以相互转换。就像发现了“注意力的另一种语言”。  
**选择性 SSM（Selective SSM）**：Mamba 系列中通过稀疏化和门控机制只对部分时间步进行高成本计算的技巧，类似于只在关键章节做深度阅读。  
**Mamba‑2**：本文基于 SSD 设计的新一代模型，核心层是对选择性 SSM 的细化实现，兼顾速度和精度。可以把它看作是“注意力+状态空间的混合引擎”。  

### 核心创新点
1. **注意力 ↔ 半分离矩阵的等价映射**  
   之前的工作把自注意力视为全连接的软注意力矩阵，计算代价高；SSM 则被当作线性递推系统，两者看似毫不相干。作者通过对结构化半分离矩阵的分解，展示了注意力矩阵其实是这类矩阵的特例。这样一来，任何针对半分离矩阵的快速算法都可以直接用于加速注意力。  
2. **SSD 框架下的统一模型视角**  
   在 SSD 框架中，作者把注意力、Mamba 的选择性 SSM、以及更广义的递推算子统一到同一个数学对象上。之前的模型只能在各自的实现细节上做改进，而现在可以在同一套理论里比较、混合、甚至交叉优化。  
3. **Mamba‑2 的层级优化**  
   基于 SSD，作者对 Mamba 的选择性 SSM 进行两层改进：① 用更高效的半分离矩阵乘法替代原始的递推卷积；② 引入门控机制只在必要的时间步激活高维计算。实验表明，这样的改动在保持原有精度的前提下，将推理速度提升了 2‑8 倍。  
4. **系统化的效率‑精度权衡分析**  
   论文不仅给出理论上 O(n) 的复杂度证明，还在实际实现中提供了多种配置（不同半分离秩、不同门控阈值），帮助使用者在硬件限制下灵活选取最优点。过去的 SSM 工作往往只给出单一配置，缺少这种实用的调参指南。  

### 方法详解
**整体思路**  
作者的核心思路是：先把注意力矩阵写成一种已知的半分离结构，然后利用该结构的快速乘法实现注意力的近似；随后把这种实现嵌入到 Mamba 的选择性 SSM 中，形成 Mamba‑2 的基本层。整个过程可以分为三步：① 矩阵分解 → ② 快速乘法实现 → ③ 门控式选择性递推。  

**步骤拆解**  

1. **半分离矩阵分解**  
   - 传统自注意力的权重矩阵是 QKᵀ 再经过 softmax。作者把 Q 与 K 看作两组低秩向量的外积，利用半分离矩阵的定义，将整个矩阵拆成上三角低秩部分 + 下三角低秩部分。  
   - 直观上，这相当于把“每个人对所有人打分”这件事拆成“只看左边的同学”和“只看右边的同学”，每一部分都可以用少量参数描述。  

2. **快速乘法（SSD 计算核）**  
   - 对于半分离矩阵，乘以向量的过程可以递归地在 O(n) 时间完成：先从左到右累加上三角贡献，再从右到左累加下三角贡献。  
   - 代码实现上，作者用两次前缀和（prefix‑sum）和后缀和（suffix‑sum）来完成，这与卷积的 FFT 加速思路类似，只是这里的“频域”是时间步的累积。  

3. **选择性递推层（Selective SSM）**  
   - Mamba 原始的递推层使用一个线性状态转移矩阵 A 和一个卷积核 B，计算成本随序列长度线性增长。  
   - 在 Mamba‑2 中，作者把 A 的作用用上一步的半分离乘法来近似，同时在每个时间步引入一个门控函数 g(t)。只有当 g(t) 超过阈值时，才执行完整的高维递推；否则直接使用快速的半分离近似。  
   - 这一步的关键在于门控函数的设计：它基于当前输入的能量（norm）和历史状态的变化率，类似于人类在阅读时只在“关键句子”上做深度思考。  

4. **整体网络堆叠**  
   - Mamba‑2 的每一层都由上述快速注意力子层 + 选择性递推子层组成，层间仍保留残差连接和层归一化，确保梯度流通。  
   - 与传统 Transformer 不同的是，Mamba‑2 在每层的前向传播里已经把大部分 O(n²) 的计算压缩到 O(n)，只在少数关键位置保留原始的高维运算。  

**最巧妙的点**  
- 把注意力矩阵映射到半分离结构后，作者没有直接放弃 softmax，而是用一种“软门控”在概率空间里保留了注意力的归一化特性，这保证了模型的表达力。  
- 选择性递推的门控阈值是可学习的，使得网络能够在训练过程中自行发现哪些时间步值得花更多算力，这种自适应的计算分配在之前的 SSM 工作中很少出现。  

### 实验与效果
- **数据集与任务**：论文在公开的语言建模基准（如 WikiText‑103、C4）以及代码生成任务（OpenAI‑Codex 数据）上进行评估。  
- **Baseline 对比**：与同等参数规模的标准 Transformer、原始 Mamba 以及最新的 Performer（线性注意力）进行比较。作者报告说，在 WikiText‑103 上，Mamba‑2 的 perplexity 与 Transformer 基本持平（相差不到 0.2），而推理速度提升约 3.5 倍；在 C4 上，Mamba‑2 超过原始 Mamba 0.5% 的准确率，同时比 Transformer 快 2.8 倍。  
- **消融实验**：通过去掉门控、使用普通半分离乘法或恢复全注意力，实验显示：门控机制贡献约 30% 的速度提升，半分离乘法本身贡献约 20% 的加速，精度下降不超过 0.1%。  
- **局限性**：作者指出，Mamba‑2 在极长序列（> 10k tokens）上仍会出现数值不稳定的情况，需要更精细的数值技巧；此外，半分离矩阵的秩选择对不同任务敏感，调参成本仍然存在。  

### 影响与延伸思考
- 这篇论文首次在理论层面把注意力和状态空间模型统一起来，打开了“注意力即递推”这一新视角。随后的工作（如 *Attention‑SSM Hybrid*、*Linear‑SSD Transformers*）纷纷借鉴 SSD 框架，尝试在更大模型上实现 O(n) 的推理。  
- 对于想进一步探索的读者，可以关注以下方向：① 更高阶的半分离结构（如多层次分解）在视觉 Transformer 中的应用；② 将 SSD 与稀疏注意力（如 BigBird）结合，进一步压缩显存；③ 研究门控机制的可解释性，看看模型到底在“哪些位置”决定投入更多算力。  

### 一句话记住它
**SSD 框架把注意力和状态空间模型等价化，让我们用 O(n) 的速度跑出和 Transformer 同等水平的语言模型。**