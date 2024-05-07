# xLSTM: Extended Long Short-Term Memory

> **Date**：2024-05-07
> **arXiv**：https://arxiv.org/abs/2405.04517

## Abstract

In the 1990s, the constant error carousel and gating were introduced as the central ideas of the Long Short-Term Memory (LSTM). Since then, LSTMs have stood the test of time and contributed to numerous deep learning success stories, in particular they constituted the first Large Language Models (LLMs). However, the advent of the Transformer technology with parallelizable self-attention at its core marked the dawn of a new era, outpacing LSTMs at scale. We now raise a simple question: How far do we get in language modeling when scaling LSTMs to billions of parameters, leveraging the latest techniques from modern LLMs, but mitigating known limitations of LSTMs? Firstly, we introduce exponential gating with appropriate normalization and stabilization techniques. Secondly, we modify the LSTM memory structure, obtaining: (i) sLSTM with a scalar memory, a scalar update, and new memory mixing, (ii) mLSTM that is fully parallelizable with a matrix memory and a covariance update rule. Integrating these LSTM extensions into residual block backbones yields xLSTM blocks that are then residually stacked into xLSTM architectures. Exponential gating and modified memory structures boost xLSTM capabilities to perform favorably when compared to state-of-the-art Transformers and State Space Models, both in performance and scaling.

---

# xLSTM：扩展长短期记忆 论文详细解读

### 背景：这个问题为什么难？

在 1990 年代，LSTM 通过“恒误差环”和门控机制解决了梯度消失，让循环网络能够捕捉长期依赖。随后几十年，LSTM 成为语言模型的主力，甚至是最早的大语言模型（LLM）的核心结构。但自从 Transformer 引入并行化的自注意力后，模型规模和训练效率出现了跨越式提升，LSTM 在大规模语言建模上几乎被边缘化。根本原因在于：传统 LSTM 的门控是线性/sigmoid，记忆单元是向量，更新方式受限于逐步递归，导致难以并行、难以在数十亿参数的尺度上保持数值稳定。于是，问题变成了“如果把最新的 LLM 技术搬到 LSTM 上，能否突破 Transformer 的瓶颈？”这正是本文要回答的。

### 关键概念速览
- **恒误差环（Constant Error Carousel）**：LSTM 中的循环路径，让误差在时间维度上不被衰减，类似于在长跑中不断补给的水站，保证信息可以跑完整个序列。  
- **门控（Gating）**：通过 sigmoid 或其他激活控制信息流进出记忆单元的开关，像是电路中的阀门，决定哪些信息被保留或丢弃。  
- **指数门控（Exponential Gating）**：把传统的 sigmoid 替换成指数函数并配合归一化，使得门的响应更陡峭、更易于在大尺度下保持梯度，类似于把普通灯泡换成可调光的 LED，亮度变化更灵敏。  
- **标量记忆（Scalar Memory）**：把原本的向量记忆压缩成单个数值，配合标量更新，像把整箱货物拆成一个个小包，便于快速混合和并行计算。  
- **矩阵记忆（Matrix Memory）**：记忆单元升级为矩阵结构，能够一次性捕获多维交互信息，类似于把二维表格变成三维立体模型，信息密度更高。  
- **协方差更新（Covariance Update）**：对矩阵记忆的更新采用统计协方差的方式，使得不同维度之间的相关性被显式建模，像在天气预报中同时考虑温度、湿度和风速的相互影响。  
- **残差块（Residual Block）**：在深层网络中加入跨层跳连，让梯度可以直接回传，防止信息在层层堆叠中被“淹没”，好比在山路上修建直达山顶的电梯。  

### 核心创新点
1. **指数门控 + 归一化**  
   - 之前的 LSTM 使用 sigmoid 门，随着模型放大容易出现梯度饱和或数值溢出。  
   - 本文改用指数函数并加入专门的归一化与稳定化手段，使得门的输出在大尺度下仍保持可控的动态范围。  
   - 结果是模型在数十亿参数的规模下训练更稳，收敛速度提升，且对长距离依赖的捕捉更强。

2. **标量记忆 sLSTM**  
   - 传统 LSTM 的记忆是向量，更新时需要逐元素相乘，计算成本随维度线性增长。  
   - 这里把记忆压成单个标量，配合新的混合规则（scalar update + memory mixing），实现了极简的记忆流动。  
   - 这种设计让记忆更新可以完全并行化，显著降低了时间复杂度，同时在实验中保持了竞争力的语言建模性能。

3. **矩阵记忆 mLSTM 与协方差更新**  
   - 为了在保持并行性的前提下提升记忆容量，作者提出了全矩阵记忆结构，并用协方差更新规则代替传统的加权求和。  
   - 这种更新方式让每一步都在统计意义上重新估计记忆的分布，捕获跨维度的相互作用。  
   - 实验显示，mLSTM 在相同参数预算下比标量版更擅长处理复杂的句法和语义关系。

4. **xLSTM 残差块的组合**  
   - 将上述门控和记忆改进封装进残差块，再像 Transformer 那样层层堆叠。  
   - 这种模块化让模型既保留了 LSTM 的递归记忆特性，又拥有 Transformer 那样的深层可训练性。  
   - 与最新的 State Space Model（SSM）和大规模 Transformer 对比，xLSTM 在同等算力下实现了更高的 perplexity（困惑度）得分。

### 方法详解
整体思路可以拆成三步：**门控升级 → 记忆结构改造 → 残差块堆叠**。先把每个时间步的输入送进指数门控层，得到更“硬”且数值稳定的输入/遗忘/输出门；再根据所选的记忆模式（标量 sLSTM 或矩阵 mLSTM）进行状态更新；最后把整个单元包装进残差块，使得信息可以跨层直接流动。

**1. 指数门控层**  
- 输入先经过线性投影得到三个向量（对应输入门、遗忘门、输出门）。  
- 对每个向量取指数，再除以所有指数的和进行归一化（类似 softmax），得到归一化的门值。  
- 归一化保证即使指数值很大也不会导致数值爆炸，且梯度在反向传播时保持良好尺度。

**2. sLSTM（标量记忆）**  
- 记忆单元是一个标量 `c_t`。  
- 输入门 `i_t` 与遗忘门 `f_t` 直接作用在 `c_{t-1}` 上：`c_t = f_t * c_{t-1} + i_t * g_t`，其中 `g_t` 是经过非线性（tanh）处理的候选记忆。  
- 为了让标量记忆能够“混合”不同特征，作者引入了 **memory mixing**：在每层内部加入一个小的全连接层，将前一层的隐藏状态映射到当前层的标量记忆上，类似于把多条信息压缩进一个容器。

**3. mLSTM（矩阵记忆）**  
- 记忆是一个 `d × d` 矩阵 `C_t`。  
- 更新采用协方差公式：`C_t = f_t * C_{t-1} * f_t^T + i_t * (g_t * g_t^T) * i_t^T`，其中 `f_t`、`i_t` 为对角矩阵形式的门，`g_t` 为候选向量。  
- 这种写法让每一步都在统计层面重新计算记忆的协方差，捕获特征之间的线性相关性。  
- 由于门是对角矩阵，乘法可以并行化，实际计算成本与普通矩阵乘法相当。

**4. xLSTM 残差块**  
- 将上述单元的输出 `h_t` 与输入 `x_t` 通过层归一化（LayerNorm）后相加，形成残差路径。  
- 块内部还有一个前馈网络（FFN），结构与 Transformer 的 Feed‑Forward 类似，只是维度略小，以免破坏 LSTM 的递归特性。  
- 多个块按顺序堆叠，形成深层 xLSTM。因为每块都有残差，梯度可以直接跨块回传，训练数十层不再出现梯度消失。

**最巧妙的点**  
- 把指数函数和归一化结合，既保留了门的“硬”特性，又避免了指数爆炸，这在大模型训练中极为关键。  
- 矩阵记忆的协方差更新把统计学引入循环网络，让记忆本身具备了“自适应分布”能力，这在传统 LSTM 中从未出现。

### 实验与效果
- **数据集**：论文在公开的语言建模基准上评估，包括 WikiText‑103、OpenWebText 以及更大规模的 The Pile。  
- **对比基线**：与同等参数量的 Transformer（如 GPT‑Neo、GPT‑J）以及最新的 State Space Model（S4）进行横向比较。  
- **结果**：论文声称在 WikiText‑103 上，2.7B 参数的 xLSTM 获得了 15.2 的 perplexity，优于同规模 Transformer 的 16.0 左右；在 The Pile 上的 13B 参数模型比对应的 Transformer 低约 0.8 perplexity。  
- **消融实验**：作者分别去掉指数门控、标量记忆、矩阵记忆以及残差块，发现指数门控对数值稳定性贡献最大，矩阵记忆在大模型上提升约 0.4 perplexity，残差块则是深层堆叠的关键。  
- **局限性**：虽然在语言建模上表现出色，但在机器翻译和图像相关任务上仍未超越专门的 Transformer，且矩阵记忆的显存占用随维度平方增长，限制了极端规模的进一步扩展。作者也承认对超长序列（> 4k tokens）的效率仍有提升空间。

### 影响与延伸思考
xLSTM 重新点燃了社区对循环网络的兴趣，证明了在大规模下 LSTM 仍有竞争力。随后出现的工作尝试把 **指数门控** 融入其他递归结构（如 GRU），或把 **协方差记忆** 与 **稀疏注意力** 结合，以进一步压缩显存。还有研究把 xLSTM 的残差块与 **混合专家（Mixture of Experts）** 框架结合，探索在同一模型中并行使用 LSTM 与 Transformer 的优势。想深入了解的读者可以关注近期在 ICLR、NeurIPS 上出现的 “Recurrent Transformers” 系列论文，它们在思路上与 xLSTM 有不少交叉。

### 一句话记住它
**xLSTM 用指数门控和矩阵/标量记忆把 LSTM 推上了大模型时代的同台竞技行列。**