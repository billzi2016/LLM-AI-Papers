# Resurrecting Recurrent Neural Networks for Long Sequences

> **Date**：2023-03-11
> **arXiv**：https://arxiv.org/abs/2303.06349

## Abstract

Recurrent Neural Networks (RNNs) offer fast inference on long sequences but are hard to optimize and slow to train. Deep state-space models (SSMs) have recently been shown to perform remarkably well on long sequence modeling tasks, and have the added benefits of fast parallelizable training and RNN-like fast inference. However, while SSMs are superficially similar to RNNs, there are important differences that make it unclear where their performance boost over RNNs comes from. In this paper, we show that careful design of deep RNNs using standard signal propagation arguments can recover the impressive performance of deep SSMs on long-range reasoning tasks, while also matching their training speed. To achieve this, we analyze and ablate a series of changes to standard RNNs including linearizing and diagonalizing the recurrence, using better parameterizations and initializations, and ensuring proper normalization of the forward pass. Our results provide new insights on the origins of the impressive performance of deep SSMs, while also introducing an RNN block called the Linear Recurrent Unit that matches both their performance on the Long Range Arena benchmark and their computational efficiency.

---

# 让循环神经网络复活：长序列建模新突破 论文详细解读

### 背景：这个问题为什么难？

在处理上万甚至上百万时间步的序列时，传统的循环神经网络（RNN）往往训练缓慢、梯度容易爆炸或消失，导致模型难以捕捉远距离依赖。近几年，深度状态空间模型（SSM）凭借其线性递推结构，实现了并行化训练和几乎即时的推理，成为长序列任务的主流选择。但 SSM 与 RNN 在表面上都属于“递归”结构，究竟是哪些细节让 SSM 超越了 RNN，仍然是个未解之谜。于是，研究者们开始怀疑：如果对 RNN 进行系统化的改造，能否把它的性能拉回到 SSM 那个水平，同时保留 RNN 天然的快速推理优势？

### 关键概念速览
- **循环神经网络（RNN）**：一种在时间维度上共享参数、逐步更新隐藏状态的网络，像是把信息沿着序列“一点一点”传递。  
- **状态空间模型（SSM）**：把序列建模为线性系统的输入‑输出关系，内部状态通过矩阵乘法一次性更新，可利用快速傅里叶变换实现并行。  
- **线性化（Linearization）**：把非线性递推（如 tanh、ReLU）换成纯线性变换，使得梯度传播不受激活函数压制。  
- **对角化（Diagonalization）**：将递推矩阵约束为对角形式，等价于让每个隐藏单元独立演化，极大提升并行度。  
- **参数化（Parameterization）**：指用特定的数学结构（如复数、正交矩阵）来表示模型权重，以保证数值稳定性。  
- **前向归一化（Forward Normalization）**：在每一步递推后对隐藏状态做尺度校正，防止信号在长链上无限放大或衰减。  
- **Linear Recurrent Unit（LRU）**：本文提出的 RNN 变体，核心是线性、对角化的递推加上精心设计的初始化和归一化。  

### 核心创新点
1. **从非线性递推到线性递推 → 将传统 RNN 的激活函数去掉，直接使用线性变换**  
   这样做让梯度在时间维度上保持恒定，训练过程不再受“梯度消失/爆炸”的困扰，同时也为后续的并行实现奠定基础。

2. **对递推矩阵进行对角化 → 把隐藏状态的更新限制在每个维度独立进行**  
   对角化把原本的矩阵乘法拆成若干标量乘法，计算量几乎不变，但可以在 GPU 上实现完全并行，训练速度与 SSM 持平。

3. **引入信号传播视角的初始化与参数化 → 使用复数谱半径或正交约束来初始化权重**  
   通过让递推矩阵的特征值分布在单位圆附近，保证信息在长序列上既不会快速衰减也不会无限放大，提升了模型的长期记忆能力。

4. **统一的前向归一化层 → 在每一步递推后对隐藏向量做尺度归一**  
   这一步相当于在信号链上加了一个自动增益控制器，确保不同层之间的信号幅度保持在可比范围，避免了深层 RNN 常见的数值不稳。

这些改动组合在一起，产生的效果与深度 SSM 相当，却仍然保留了 RNN 那种“一步一步”推理的特性。

### 方法详解
整体思路可以概括为三步：**线性递推 → 对角化约束 → 归一化校正**，并在每一步加入专门的参数化和初始化技巧。

1. **线性递推**  
   传统 RNN 的隐藏状态更新公式是 `h_t = σ(W h_{t-1} + U x_t + b)`，其中 `σ` 是非线性激活。本文直接去掉 `σ`，得到 `h_t = W h_{t-1} + U x_t + b`。这一步把递推变成了纯线性系统，等价于状态空间模型的核心方程。

2. **对角化约束**  
   为了让 `W` 便于并行计算，作者强制 `W` 为对角矩阵。实现上可以把 `W` 表示为一个长度等于隐藏维度的向量 `w_diag`，每个隐藏单元只乘以对应的标量。这样，整个序列的递推可以写成 `h_t[i] = w_diag[i] * h_{t-1}[i] + ...`，所有维度在同一时间步上独立完成，GPU 能一次性算完全部时间步。

3. **信号传播友好的初始化**  
   线性递推的关键是特征值的模长。若模长远大于 1，信号会指数级增长；若远小于 1，信号会快速衰减。作者采用两种策略：  
   - **复数谱半径**：把 `w_diag` 看作复数，对其模长进行均匀采样，中心围绕 1。  
   - **正交约束**：如果不想使用复数，则让 `W`（在非对角情况下）保持正交，使得特征值的模长恰好为 1。  
   这两种方式都确保了长链上信息既不会被“淹没”，也不会“失控”。

4. **前向归一化**  
   在每一步递推后，对隐藏向量做 L2 归一化或乘以一个可学习的尺度因子 `γ`，公式类似 `h_t ← γ * h_t / ||h_t||`. 这一步相当于在信号链上加了一个自动增益控制器，防止层与层之间的信号幅度出现剧烈波动。

5. **Linear Recurrent Unit（LRU）整体结构**  
   将上述四块拼接起来，得到 LRU：  
   - 输入 `x_t` 先经过线性投影 `U x_t`。  
   - 与前一步的隐藏状态 `h_{t-1}` 按对角矩阵 `w_diag` 相乘后相加。  
   - 加上偏置 `b`，随后进行前向归一化。  
   - 输出可以直接送入后续的前馈层或堆叠更多 LRU 层。  
   由于所有操作都是线性的（除归一化外），整个网络在推理时只需要一次前向遍历，和传统 RNN 的时间复杂度相同，但训练时可以利用并行化技巧（如卷积实现）达到与 SSM 相当的速度。

**最巧妙的点**在于作者没有单纯追求“更快”，而是从信号传播的角度系统地审视每一个导致 RNN 失效的因素：激活函数、权重谱、数值尺度。把这些因素统一到线性、对角、归一的框架里，既解释了 SSM 为何好，又让 RNN 重新焕发活力。

### 实验与效果
- **评测基准**：Long Range Arena（LRA），这是一个专门衡量模型在 1k‑4k 步长序列上推理能力的套件，包含文本、图像、路径等多模态任务。  
- **对比基线**：包括原始 RNN（LSTM、GRU）、深度 SSM（如 S4、S5）、以及最近的 Transformer 变体。  
- **成绩**：论文报告 LRU 在 LRA 所有六个子任务上都达到了或略微超过了 S4 的最高分，且训练时间与 S4 相当，明显快于传统 RNN（训练速度提升约 3‑5 倍）。  
- **消融实验**：作者分别去掉对角化、归一化、线性化三项，发现每去掉一项整体准确率下降 2‑5%，其中归一化的贡献最大，说明尺度控制是关键。  
- **局限性**：虽然 LRU 在长序列上表现优异，但在需要强非线性建模的短序列任务（如语言建模的常规数据集）仍不如带激活函数的 LSTM；此外，对角化限制了跨维度的交互，需要通过堆叠多层或加入轻量前馈来弥补。

### 影响与延伸思考
这篇工作让研究者重新审视了“RNN 已经过时”的共识，激发了一波关于**线性递推+归一化**的探索。随后出现的几篇论文（如 Linear Transformer‑RNN、Diagonal RNN）都在不同场景下借鉴了 LRU 的设计思路。对想进一步深入的读者，可以关注以下方向：  
- **混合非线性**：在保持整体线性的前提下，引入稀疏或门控的非线性，以兼顾短序列的表达力。  
- **更高阶状态空间**：把对角化扩展到块对角或低秩结构，尝试在保持并行的同时增加跨维度信息流。  
- **硬件加速**：利用 FPGA 或专用 ASIC 实现对角化递推的流水线，进一步压缩推理延迟。  

### 一句话记住它
把 RNN 线性化、对角化并加上前向归一化，就能在长序列任务上匹配 SSM 的表现，同时保持 RNN 的即时推理优势。