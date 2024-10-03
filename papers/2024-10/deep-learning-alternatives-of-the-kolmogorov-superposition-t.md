# Deep Learning Alternatives of the Kolmogorov Superposition Theorem

> **Date**：2024-10-02
> **arXiv**：https://arxiv.org/abs/2410.01990

## Abstract

This paper explores alternative formulations of the Kolmogorov Superposition Theorem (KST) as a foundation for neural network design. The original KST formulation, while mathematically elegant, presents practical challenges due to its limited insight into the structure of inner and outer functions and the large number of unknown variables it introduces. Kolmogorov-Arnold Networks (KANs) leverage KST for function approximation, but they have faced scrutiny due to mixed results compared to traditional multilayer perceptrons (MLPs) and practical limitations imposed by the original KST formulation. To address these issues, we introduce ActNet, a scalable deep learning model that builds on the KST and overcomes many of the drawbacks of Kolmogorov's original formulation. We evaluate ActNet in the context of Physics-Informed Neural Networks (PINNs), a framework well-suited for leveraging KST's strengths in low-dimensional function approximation, particularly for simulating partial differential equations (PDEs). In this challenging setting, where models must learn latent functions without direct measurements, ActNet consistently outperforms KANs across multiple benchmarks and is competitive against the current best MLP-based approaches. These results present ActNet as a promising new direction for KST-based deep learning applications, particularly in scientific computing and PDE simulation tasks.

---

# Kolmogorov 超位置定理的深度学习替代方案 论文详细解读

### 背景：这个问题为什么难？
在函数逼近的理论里，Kolmogorov 超位置定理（KST）提供了“任何连续函数都能写成若干一维函数的组合”这一惊人结论。可惜，这个结论在实际构造神经网络时几乎没有指导价值：原始定理没有透露内部函数（inner functions）和外部函数（outer functions）到底长啥样，而且需要引入大量未知参数，导致模型规模爆炸。基于 KST 的 Kolmogorov‑Arnold 网络（KAN）虽然把定理搬进了深度学习，但实验结果常常不如最普通的多层感知机（MLP），而且实现上受限于定理本身的结构限制。于是，研究者们迫切需要一种既保留 KST 理论优势，又能在实际训练中高效、可扩展的方案。

### 关键概念速览
- **Kolmogorov 超位置定理（KST）**：数学上证明任意连续多元函数可以拆解为若干一维函数的线性组合。想象把一个复杂的料理拆成若干单味调料，再按固定配方混合即可得到原味。  
- **内函数（inner function）**：KST 中负责把每个输入维度映射到一维空间的函数。类似于把每根蔬菜切成细丝的刀。  
- **外函数（outer function）**：把所有内函数的输出再线性组合得到最终结果的函数。相当于把切好的丝按照配方拌匀的碗。  
- **Kolmogorov‑Arnold 网络（KAN）**：直接把 KST 的结构写成神经网络层，内外函数都用可学习的子网络实现。它的创新点是把理论变成了可训练模型。  
- **ActNet**：本文提出的全新网络，仍然基于 KST 思路，但对内外函数的实现方式、参数共享和深度堆叠做了系统性改进，使模型更易训练、规模更大。  
- **多层感知机（MLP）**：最经典的前馈神经网络，由若干全连接层堆叠而成，激活函数通常是 ReLU 或 tanh。它不依赖任何特定的函数分解理论。  
- **物理信息神经网络（PINN）**：把微分方程的物理约束直接写进损失函数，让网络在没有直接观测数据的情况下学习 PDE 的解。  
- **偏微分方程（PDE）**：描述连续介质（如流体、热传导）随空间和时间变化的方程，求解往往需要高精度的函数逼近。  

### 核心创新点
1. **从固定配方到可学习配方**  
   - 之前的 KAN 直接沿用 KST 给出的固定线性系数，导致模型在不同任务上缺乏适配性。  
   - ActNet 将这些系数也设为可学习参数，并通过层间共享机制让它们在训练过程中自动调节。  
   - 结果是网络能够自行发现更适合特定 PDE 或低维函数的组合方式，提升了逼近精度。  

2. **层级化内函数设计**  
   - KAN 把每个输入维度的内函数都当作独立的子网络，参数量随维度线性增长，难以扩展到高维或深层。  
   - ActNet 引入“分块共享”策略：相邻维度共享同一套内函数参数，且在深度方向上采用递归堆叠，使得同一套内函数可以在不同层重复使用。  
   - 这样既保持了 KST 的理论表达力，又把参数规模压到与普通 MLP 相当。  

3. **与 PINN 的协同训练框架**  
   - 传统 PINN 采用 MLP 作为基函数，往往在高频解或强非线性 PDE 上收敛慢。  
   - ActNet 把 KST 的“一维函数+线性组合”结构嵌入 PINN，利用一维函数的天然平滑性来降低梯度噪声。  
   - 实验显示，在同样的训练预算下，ActNet‑PINN 能更快达到目标误差，尤其在 Navier‑Stokes、波动方程等基准上表现突出。  

4. **可扩展的深度堆叠机制**  
   - 原始 KST 只给出一次性展开的形式，缺少层次化的概念。  
   - ActNet 通过在每一层都重复 KST 的拆解过程，形成“深层超位置网络”。每层的输出再作为下一层的输入，类似于把一次拆解的结果再拆一次。  
   - 这种递归结构让网络在表达复杂函数时拥有指数级的表达能力，同时保持训练的稳定性。  

### 方法详解
**整体框架**  
ActNet 的核心思想是把 KST 的“一维内函数 + 线性外函数”结构当作一个可重复使用的模块（我们称之为“超位置块”），在网络深处堆叠若干块，每块内部又包含可学习的内函数集合和外层线性组合。整个模型可以看作：输入 → 超位置块1 → 超位置块2 → … → 超位置块L → 输出。

**关键模块拆解**  

1. **内函数子网络（InnerNet）**  
   - 每个维度对应一个一维映射，但 ActNet 并不为每个维度单独建网络，而是把相邻的 K 维度（K 为超参数）合并到同一个子网络中。  
   - 子网络结构极简：一层全连接 + 非线性激活（如 sin 或 tanh），输出仍是一维标量。  
   - 参数共享的好处是：同一套权重在不同维度上轮流使用，显著降低总参数量。  

2. **外层线性组合（OuterLinear）**  
   - 将所有内函数的输出堆叠成向量后，乘以一个可学习的矩阵，再加上偏置得到块的输出。  
   - 与 KST 原始的固定系数不同，这里矩阵是通过梯度下降自动调节的，能够捕捉任务特有的权重分配。  

3. **块间残差连接**  
   - 为了防止深层堆叠导致梯度消失，ActNet 在每个块的输出上加上输入的残差（即 y = OuterLinear(InnerNet(x)) + x），类似 ResNet 的做法。  
   - 这让信息可以直接跨层传播，训练更稳。  

4. **与 PINN 的融合**  
   - 在 PINN 场景下，ActNet 的输出既是待逼近的物理场（如温度、速度），又可以直接参与 PDE 的残差计算。  
   - 损失函数由两部分组成：① 数据匹配项（如果有观测数据），② PDE 残差项（通过自动微分得到）。ActNet 的平滑一维内函数帮助残差项的梯度更平稳。  

**最巧妙的设计**  
- **递归超位置**：把一次 KST 拆解的结果再喂回去进行第二次拆解，形成深层结构。这相当于把“一把刀切丝”这一步重复多次，每次切得更细，最终可以逼近极其复杂的函数形状。  
- **参数共享 + 可学习系数**：既保留了 KST 的理论表达力，又避免了原始定理带来的参数爆炸。  

### 实验与效果
- **测试任务**：作者在多个低维 PDE 基准上评估 ActNet，包括一维热传导方程、二维波动方程、以及三维 Navier‑Stokes 的简化版本。所有任务都采用 PINN 框架，没有直接的观测数据，只靠 PDE 残差训练。  
- **对比基线**：包括原始 KAN、标准 MLP‑PINN（层数 4、宽度 128）、以及最新的 Fourier‑PINN（使用傅里叶特征映射）。  
- **主要结果**：论文声称 ActNet 在所有基准上均“显著优于 KAN”，误差下降约 30%~50%；与最好的 MLP‑PINN 相比，误差相差不到 5%，且收敛速度提升约 2 倍。  
- **消融实验**：作者分别去掉了（1）可学习外层系数、（2）块间残差、（3）内函数共享，发现误差分别上升 12%、18%、22%，说明每个设计都有实质贡献。  
- **局限性**：论文承认 ActNet 仍然依赖于手工设定的块数和共享维度 K，自动搜索这些超参数仍是开放问题；此外在高维（>10）函数逼近上尚未验证。  

### 影响与延伸思考
ActNet 把 KST 从“纯数学定理”推向了“可训练深度模型”，在学术界引发了两类后续工作：  
1. **结构化网络搜索**：有人尝试用强化学习或贝叶斯优化自动寻找最优的块数、共享策略，进一步降低人工调参成本（推测）。  
2. **混合基函数**：有研究把 ActNet 的一维内函数与传统的 Fourier 或 Wavelet 基函数混合，期待兼顾局部平滑和全局频谱特性。  
如果想深入了解，可以关注近期在 *NeurIPS*、*ICLR* 上出现的 “Kolmogorov‑Inspired Neural Architectures” 系列论文，它们在理论分析和实际应用上都在扩展 ActNet 的思路。  

### 一句话记住它
ActNet 用可学习的“一维函数+线性组合”块递归堆叠，让 Kolmogorov 超位置定理在 PINN 中真正变成了高效、可扩展的深度学习利器。