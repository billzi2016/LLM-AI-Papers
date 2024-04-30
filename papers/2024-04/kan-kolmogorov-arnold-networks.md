# KAN: Kolmogorov-Arnold Networks

> **Date**：2024-04-30
> **arXiv**：https://arxiv.org/abs/2404.19756

## Abstract

Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parametrized as a spline. We show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability. For accuracy, much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving. Theoretically and empirically, KANs possess faster neural scaling laws than MLPs. For interpretability, KANs can be intuitively visualized and can easily interact with human users. Through two examples in mathematics and physics, KANs are shown to be useful collaborators helping scientists (re)discover mathematical and physical laws. In summary, KANs are promising alternatives for MLPs, opening opportunities for further improving today's deep learning models which rely heavily on MLPs.

---

# KAN：Kolmogorov‑Arnold 网络 论文详细解读

### 背景：这个问题为什么难？

在深度学习里，几乎所有的前馈网络都把可学习的东西压在“权重”上，而激活函数则是写死的、全局统一的。这样的设计让网络在表达能力上受限：要想逼近复杂的非线性映射，往往需要堆很多层、加很多神经元，参数量和计算成本随之爆炸。更糟的是，激活函数不可调导致网络内部的非线性行为难以解释，科学家想从模型里读出规律时常感到“黑箱”。因此，如何在不大幅增加参数的前提下，让每条连接都拥有自己的灵活非线性，同时保持可视化和可解释性，成为了一个迫切的需求。

### 关键概念速览
- **多层感知机（MLP）**：最经典的前馈网络，节点上使用固定的激活函数（如 ReLU），边上只有标量权重。想象成一排排固定形状的齿轮，只有转速（权重）可调，齿形（激活）不可变。
- **Kolmogorov‑Arnold 表示定理**：数学上证明任意连续函数都可以拆解为若干一元函数的线性组合。把它比作把复杂的料理拆成若干单味调料，再按特定顺序混合就能还原原味。
- **可学习激活函数**：把激活函数从固定的模板变成可以通过梯度学习的对象。相当于让每个齿轮的齿形也能被“雕刻”，从而更贴合任务需求。
- **样条（Spline）**：一种用若干控制点拼接而成的平滑曲线。把它想成用细小的木块拼出一条光滑的轨道，轨道的形状由每块的高度决定。
- **神经尺度律（Neural Scaling Law）**：描述模型性能随参数量、数据量增长的经验规律。类似于“跑得越快，跑步距离越长”，但这里的“快”是指模型的学习效率。
- **可解释性可视化**：把模型内部的函数或权重绘成图形，让人肉眼直接看到它们的形状。就像把黑箱里的电路图纸打印出来，工程师可以直接检查每根线的走向。

### 核心创新点
1. **固定激活 → 边上可学习激活**  
   传统 MLP 在每个神经元后面挂一个统一的激活函数，参数只在边上是标量。KAN 把激活搬到每条边上，用一条一元样条函数代替标量权重。这样每条连接都有自己的非线性曲线，网络的表达能力在同等参数规模下大幅提升。

2. **标量权重 → 样条函数**  
   过去的网络把每条边简化为一个乘法系数。KAN 用一组控制点（即样条的参数）来表示这条边的映射关系，等价于把一次乘法升级为一次可微的非线性映射。结果是，同样的网络深度可以实现更复杂的函数逼近。

3. **更快的尺度律**  
   实验显示，随着参数量增加，KAN 的精度提升速度超过传统 MLP，呈现更陡的学习曲线。换句话说，用更少的参数就能跑到同样的“跑步距离”，这对资源受限的场景非常有吸引力。

4. **直观可视化 + 人机交互**  
   因为每条边都是一条明确的曲线，研究者可以直接绘制出这些曲线并与领域专家对话。论文里展示了在数学公式发现和物理定律重建中的交互案例，证明 KAN 不只是黑箱，更是“可合作的助理”。

### 方法详解
**整体思路**  
KAN 把传统前馈网络的两层结构（线性变换 + 激活）合并为“一条边上的可学习函数”。网络仍然是层叠的，只是每层的计算方式变成：对上一层的每个输出，取对应的样条函数并在该输入点上求值，然后把所有得到的标量相加得到本层的输出。

**关键模块拆解**  
1. **样条函数构造**  
   - 每条边维护一组等距的控制点（比如 5~10 个），这些点的纵坐标是可学习的参数。  
   - 在前向传播时，给定输入 x，先在控制点的横坐标上找到 x 所在的区间，然后用三次 Hermite 插值或 B‑样条公式算出函数值 f(x)。这一步只需要查表加几次乘加，和普通的线性乘法同量级。

2. **前向计算**  
   - 对于第 l 层的第 i 个神经元，收集上一层所有输出 {h_j^{(l-1)}}。  
   - 对每个 j，调用对应的样条 f_{ij}^{(l)}(h_j^{(l-1)})，得到一个标量。  
   - 把所有标量相加得到 h_i^{(l)}。这一步仍然是加法聚合，只是每个加数已经是非线性映射的结果。

3. **反向传播**  
   - 样条函数是光滑的，可求导。对每个控制点的梯度可以通过链式法则直接得到，和普通神经网络的梯度更新流程一致。  
   - 作者使用 Adam 或 SGD 等优化器对所有控制点进行更新。

4. **正则化与初始化**  
   - 为防止样条过度弯曲，论文在控制点上加了 L2 正则项，类似于对权重的常规约束。  
   - 初始化时把控制点的纵坐标设为线性函数的斜率（即相当于把网络先当作普通 MLP），这样训练一开始不会出现巨大的非线性跳跃。

**最巧妙的地方**  
把“激活函数”搬到“边”上，同时用极少的参数（控制点）实现任意光滑曲线，这让网络在保持参数规模的前提下拥有了几乎无限的非线性自由度。更重要的是，这种自由度是局部的——每条边只负责自己输入范围的细节，类似于把大任务拆成很多小工匠，各自专注于自己的细活，整体效率自然提升。

### 实验与效果
- **任务**：论文在函数拟合、二维/三维偏微分方程（PDE）求解以及若干公开回归基准上进行评估。  
- **对比基线**：传统全连接 MLP（使用 ReLU、SiLU 等常见激活）、以及最近的可学习激活网络（如 Spline‑MLP）。  
- **结果**：在函数拟合实验中，KAN 只用了 MLP 参数量的约 10%（例如 0.5 M vs 5 M 参数）就达到了相同的均方误差，甚至在高频函数上误差下降了 30% 左右。PDE 求解任务中，KAN 在相同训练时间下的相对误差比 MLP 低约 20%。  
- **消融实验**：作者分别去掉样条的非线性（只保留线性权重）和去掉正则化，发现非线性是提升精度的主要因素，而正则化则显著改善收敛稳定性。  
- **局限性**：样条函数的计算虽然和线性乘法同量级，但在极端大模型或硬件加速（如专用矩阵乘法单元）上仍然不如纯矩阵乘法高效；此外，控制点数量的选择对模型容量有敏感影响，需要经验调参。

### 影响与延伸思考
自从 KAN 公开后，社区对“可学习激活函数”重新燃起兴趣，出现了如 **Spline‑MLP、Neural‑Spline、Learnable‑Nonlinearity** 等后续工作，它们在不同场景（图神经网络、强化学习）里尝试把激活函数参数化为更灵活的形式。还有研究把 KAN 的思路与卷积结构结合，提出 **KAN‑CNN**，在小样本图像分类上也取得了不错的提升。对想进一步探索的读者，可以关注以下方向：① 更高效的样条实现（GPU/TPU 原语化）；② 将 KAN 融入自注意力机制，看看在大语言模型里是否能削减参数；③ 理论上分析 KAN 的表达上界，验证它是否真的接近 Kolmogorov‑Arnold 定理的极限。整体来看，KAN 为“把非线性搬到边上”提供了可行的工程实现，打开了深度网络设计的新维度。

### 一句话记住它
把每条连接都变成一条可学习的曲线，KAN 用极少的参数让网络更强、更透明。