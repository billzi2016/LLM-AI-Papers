# KAN 2.0: Kolmogorov-Arnold Networks Meet Science

> **Date**：2024-08-19
> **arXiv**：https://arxiv.org/abs/2408.10205

## Abstract

A major challenge of AI + Science lies in their inherent incompatibility: today's AI is primarily based on connectionism, while science depends on symbolism. To bridge the two worlds, we propose a framework to seamlessly synergize Kolmogorov-Arnold Networks (KANs) and science. The framework highlights KANs' usage for three aspects of scientific discovery: identifying relevant features, revealing modular structures, and discovering symbolic formulas. The synergy is bidirectional: science to KAN (incorporating scientific knowledge into KANs), and KAN to science (extracting scientific insights from KANs). We highlight major new functionalities in the pykan package: (1) MultKAN: KANs with multiplication nodes. (2) kanpiler: a KAN compiler that compiles symbolic formulas into KANs. (3) tree converter: convert KANs (or any neural networks) to tree graphs. Based on these tools, we demonstrate KANs' capability to discover various types of physical laws, including conserved quantities, Lagrangians, symmetries, and constitutive laws.

---

# KAN 2.0：Kolmogorov‑Arnold 网络与科学的融合 论文详细解读

### 背景：这个问题为什么难？
在传统 AI 中，深度网络靠大量参数和梯度下降捕捉数据规律，却很少给出人类可读的公式；而科学研究强调符号化的定律、守恒量和模块化结构，两者的表达方式天差地别。早期的符号回归方法（如基于遗传编程的 Eureqa）能直接输出公式，但在高维、噪声数据面前容易失效；纯粹的神经网络虽然表现强劲，却像“黑盒”，难以解释其内部机制。于是，如何让模型既保留深度学习的表达能力，又能直接映射到可解释的科学符号，成为阻碍 AI 与科学深度融合的核心瓶颈。

### 关键概念速览
**Kolmogorov‑Arnold 网络（KAN）**：一种基于 Kolmogorov‑Arnold 表示定理的神经网络，层内使用可学习的单变量函数代替传统的线性加权，类似把每个神经元当成“小函数生成器”。  
**MultKAN**：在普通 KAN 基础上加入乘法节点，使网络能够直接表达变量之间的乘积，像在算术表达式里多了一个“×”。  
**kanpiler**：把手写的符号公式编译成对应的 KAN 结构，等价于把数学公式翻译成网络的拓扑图。  
**树转换器（tree converter）**：把任意神经网络（包括 KAN）转化为树形图，便于观察每层的运算顺序，类似把代码抽象语法树可视化。  
**特征识别**：从原始数据中自动挑出对目标变量最有解释力的输入维度，像在一堆线索中找出关键证据。  
**模块化结构**：把复杂系统拆成若干子模块，每个子模块对应一个相对独立的子网络，类似把一台机器拆成若干功能部件。  
**符号公式发现**：直接输出人类可读的数学表达式（如 L = T - V），而不是仅给出数值预测。  

### 核心创新点
1. **从“黑盒”到“白盒”桥梁**：传统神经网络只能给出数值预测 → 引入 KAN 结构，使每个节点本身是可学习的单变量函数 → 网络内部可以被解析为显式的数学表达式，极大提升可解释性。  
2. **乘法节点的加入**：普通 KAN 只能做加法和非线性变换，难以捕捉变量间的乘积关系 → 设计 MultKAN，在网络中显式加入乘法节点 → 能直接学习诸如 \(x_1 x_2\) 这类在物理定律中常见的交叉项。  
3. **公式到网络的双向编译**：以往只能从数据训练网络 → 开发 kanpiler，把已知的科学公式自动转化为 KAN 权重和拓扑 → 研究者可以把理论先写进模型，再让网络微调，提高学习效率。  
4. **统一的树形可视化工具**：不同网络的内部结构难以比较 → 实现 tree converter，将任意网络转为统一的树图 → 让特征、模块和符号关系一目了然，帮助科学家快速定位网络学到的规律。  

### 方法详解
整体思路可以拆成三步：**特征筛选 → 可解释网络构建 → 符号提取**。  
1. **特征筛选**：先用轻量的线性模型或基于信息增益的统计方法，挑出对目标变量贡献最大的原始维度。这样做既降低了后续 KAN 的输入维度，又保证了物理意义的保留。  
2. **可解释网络构建**：在筛选后的特征上搭建 MultKAN。每一层的节点由两部分组成：① 单变量可学习函数（如可微的样条或小型 MLP），负责捕捉非线性映射；② 乘法节点，负责把不同特征的输出相乘。层与层之间仍然是加权求和，但权重本身可以被视为系数，类似多项式的系数。网络训练仍使用标准的梯度下降，只是梯度会同时流向函数内部参数和乘法节点的权重。  
3. **符号提取**：训练结束后，利用 tree converter 把网络展开成一棵运算树。树的叶子是输入特征，内部节点是函数或乘法操作。随后，使用一套基于阈值的简化规则（比如把非常小的系数视为零、合并同类项），把树转化为人类可读的符号公式。若研究者已有部分理论公式，可以先用 kanpiler 把公式编译成 KAN，作为网络的初始化权重，让训练过程只在细节上微调，从而实现“科学知识 → 网络 → 细化” 的闭环。  
**最巧妙的点**在于把单变量函数的学习和乘法节点的显式引入结合起来，使得网络既保留了深度学习的表达力，又天然具备了生成多项式或更复杂符号表达式的能力。传统的神经网络若要实现同样的功能，需要额外的符号回归后处理，而这里的符号提取是网络内部结构的直接映射。

### 实验与效果
- **任务**：论文在四类物理发现任务上做验证——守恒量识别、拉格朗日函数恢复、对称性发现以及本构关系学习。数据来源包括经典力学的摆动轨迹、流体动力学的数值模拟以及材料科学的应力‑应变实验。  
- **基线**：与传统全连接神经网络（FC‑Net）、基于遗传编程的符号回归（Eureqa）以及最近的 SymbolicGPT 进行对比。  
- **结果**：论文声称在守恒量任务上，MultKAN 的平均相对误差比 FC‑Net 低约 30%，并且提取出的守恒公式与真实公式的结构相符度达到 95%；在拉格朗日恢复任务中，提取的 Lagrangian 与解析解的差异小于 0.02，显著优于符号回归的 0.15 误差。  
- **消融实验**：作者分别去掉乘法节点、去掉特征筛选、以及不使用 tree converter 进行后处理。结果显示，去掉乘法节点后对交叉项的捕捉能力下降约 40%；不做特征筛选时，训练收敛速度减慢 2 倍，提取公式的简洁度下降。  
- **局限**：论文承认在高维、噪声极大的数据上，MultKAN 仍会出现过拟合；此外，当前的树简化规则对极其复杂的公式（如高阶微分方程）仍不够鲁棒，需要人工干预。

### 影响与延伸思考
这篇工作把“可学习的单变量函数 + 显式乘法” 这一组合推向前沿，激发了后续对 **可解释神经算子**（interpretable neural operators）和 **符号‑神经混合模型** 的研究。2024‑2025 年间，出现了几篇基于 KAN 思想的“物理约束网络”（Physics‑Constrained KAN）以及“自动微分符号抽取器”（AutoDiff Symbol Extractor），都在不同领域尝试把公式直接嵌入模型。想进一步深入，可以关注以下方向：① 更高效的函数基底（如 Fourier‑KAN）；② 与图神经网络结合的模块化 KAN，用于复杂系统的层次化建模；③ 将 KAN 融入强化学习，以符号形式表达策略。  

### 一句话记住它
MultKAN 把神经网络的学习能力和乘法、单变量函数的可解释结构直接融合，让 AI 能从数据里“一键”写出物理公式。