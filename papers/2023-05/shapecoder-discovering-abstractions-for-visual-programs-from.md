# ShapeCoder: Discovering Abstractions for Visual Programs from   Unstructured Primitives

> **Date**：2023-05-09
> **arXiv**：https://arxiv.org/abs/2305.05661

## Abstract

Programs are an increasingly popular representation for visual data, exposing compact, interpretable structure that supports manipulation. Visual programs are usually written in domain-specific languages (DSLs). Finding "good" programs, that only expose meaningful degrees of freedom, requires access to a DSL with a "good" library of functions, both of which are typically authored by domain experts. We present ShapeCoder, the first system capable of taking a dataset of shapes, represented with unstructured primitives, and jointly discovering (i) useful abstraction functions and (ii) programs that use these abstractions to explain the input shapes. The discovered abstractions capture common patterns (both structural and parametric) across the dataset, so that programs rewritten with these abstractions are more compact, and expose fewer degrees of freedom. ShapeCoder improves upon previous abstraction discovery methods, finding better abstractions, for more complex inputs, under less stringent input assumptions. This is principally made possible by two methodological advancements: (a) a shape to program recognition network that learns to solve sub-problems and (b) the use of e-graphs, augmented with a conditional rewrite scheme, to determine when abstractions with complex parametric expressions can be applied, in a tractable manner. We evaluate ShapeCoder on multiple datasets of 3D shapes, where primitive decompositions are either parsed from manual annotations or produced by an unsupervised cuboid abstraction method. In all domains, ShapeCoder discovers a library of abstractions that capture high-level relationships, remove extraneous degrees of freedom, and achieve better dataset compression compared with alternative approaches. Finally, we investigate how programs rewritten to use discovered abstractions prove useful for downstream tasks.

---

# ShapeCoder：从无结构基元中发现视觉程序的抽象 论文详细解读

### 背景：这个问题为什么难？

在计算机视觉里，把一张图像或一个 3D 形状直接当作像素/点云来处理，虽然通用，却很难解释和编辑。近年来，人们尝试用“视觉程序”——一种基于领域专用语言（DSL）的代码——来描述形状，因为代码天然具备层次结构和可编辑的自由度。然而，要得到“好”的视觉程序，需要先有一个设计精良的 DSL 和一套高质量的函数库，这通常由领域专家手工编写。手工构造的库既耗时，又容易遗漏数据中潜在的共性模式；而且当数据来源多样、结构松散时，现有的自动抽象发现方法往往只能处理非常规则的输入，或者要求事先提供严格的分解结果。于是，如何在仅有“无结构基元”（比如一堆散乱的立方体）时，自动挖掘出有意义的抽象函数并生成紧凑的视觉程序，成为了一个未被很好解决的难题。

### 关键概念速览

**视觉程序（Visual Program）**：用 DSL 编写的代码，描述几何体的生成过程，类似于建筑师的施工图，既能表达形状，又能被人直接修改。  

**无结构基元（Unstructured Primitives）**：原始的几何块（如立方体、球体）没有任何层次或关联信息，像散落的乐高块，需要先被组织起来才能看出整体结构。  

**抽象函数（Abstraction Function）**：DSL 中的高层函数，捕捉数据集中反复出现的结构或参数模式，就像把“一堆相同的窗户”抽象成“窗户模块”。  

**e‑graph（等价图）**：一种图结构，用来记录表达式之间的等价关系，能够在大量候选重写规则中快速查找匹配，就像在字典里快速定位同义词。  

**条件重写（Conditional Rewrite）**：在 e‑graph 上执行的规则，只有当满足特定参数约束时才会生效，类似于“只有当门宽度大于 1 米时才使用大门模型”。  

**形状到程序识别网络（Shape‑to‑Program Recognition Network）**：一个神经网络，输入是基元集合，输出是对应的 DSL 代码，负责把“散块”翻译成“程序”。  

**库学习（Library Learning）**：在整个数据集上搜索并挑选抽象函数的过程，目标是让所有程序都能用更少的函数调用来表达，从而压缩数据。  

### 核心创新点

1. **从无结构基元直接生成程序 → 采用专门的 Shape‑to‑Program 识别网络**  
   过去的抽象发现工作往往假设已经有了结构化的程序或严格的分解；ShapeCoder 训练了一个网络，能够在没有任何先验结构的情况下，把散乱的基元映射为初始 DSL 程序。这样一来，抽象发现的入口不再受限于手工标注或强假设。

2. **利用 e‑graph + 条件重写实现复杂参数抽象的可行搜索 → 将抽象搜索空间压缩为等价图并在图上执行带约束的重写**  
   传统方法在面对带有非线性参数的抽象时会爆炸式增长搜索成本。ShapeCoder 把所有候选子表达式放进 e‑graph，利用等价关系共享子结构，再通过条件重写只在满足参数约束时应用抽象，从而在多维参数空间中保持多项式时间。

3. **联合优化库学习与程序重写 → 交替更新抽象函数和程序，使两者相互提升**  
   以前的系统往往先固定库再重写程序，或先重写再学习库，缺乏协同。ShapeCoder 采用交替迭代：先用当前库重写所有程序得到压缩率，依据压缩效果挑选新的抽象，再用新抽象重新生成程序。这样循环几轮后，抽象函数和程序都趋向最简。

4. **在更宽松的输入假设下仍能发现高质量抽象 → 支持手工标注或完全无监督的基元分解**  
   通过上述两大技术，ShapeCoder 不再需要严格的手工分解，只要有粗糙的基元集合（甚至是从无监督立方体抽象方法得到的），就能成功学习库并压缩数据，显著拓宽了可用场景。

### 方法详解

#### 整体框架

ShapeCoder 的工作流可以划分为三大阶段：  
1) **基元到初始程序的翻译**（Shape‑to‑Program 识别网络）；  
2) **抽象函数的搜索与库构建**（e‑graph + 条件重写）；  
3) **程序重写与交替优化**（循环更新库与程序）。  

整个过程在整个数据集上同步进行，目标是让每个形状最终都能用最少的 DSL 调用来解释。

#### 1. Shape‑to‑Program 识别网络

- **输入**：一个形状的基元集合，每个基元用位置、尺度、方向等属性向量表示。  
- **网络结构**：采用点云处理的 PointNet/Transformer 变体，先对每个基元做特征编码，再通过全局池化得到形状级特征。  
- **输出**：一段 DSL 程序的序列化表示（类似于代码的 token 序列），网络在训练时使用教师强制（teacher forcing）和交叉熵损失。  
- **训练数据**：作者使用两类来源：一是人工标注的基元‑程序对，二是通过已有的无监督立方体抽象方法生成的“伪标签”。网络学习到的能力是把散乱基元组织成层次化的构造指令（如 `translate`, `scale`, `repeat` 等）。

#### 2. 抽象函数搜索（e‑graph + 条件重写）

- **构建 e‑graph**：对每个初始程序，枚举所有可能的子表达式（如 `translate(a, b)`, `repeat(k, shape)`），把它们作为节点加入等价图。等价关系来源于代数恒等式（例如 `translate(t1) ∘ translate(t2) = translate(t1+t2)`）以及用户自定义的几何等价（如旋转合并）。  
- **候选抽象生成**：在 e‑graph 中寻找重复出现的子图模式，这些模式对应潜在的抽象函数。每个候选抽象会被参数化为一个函数模板，模板的参数可以是常数、线性组合或更复杂的表达式。  
- **条件重写机制**：并非所有出现的模式都值得抽象。系统为每个候选抽象附加约束条件（例如“参数必须为正整数”或“尺度比例必须在 0.5~2 之间”），只有当所有实例满足这些约束时，才把该抽象加入库。约束检查在 e‑graph 上一次性完成，避免逐个程序遍历。  
- **库评分**：每个抽象函数会根据它在整个数据集上带来的压缩率（即使用该抽象后程序长度的下降）以及参数通用性（约束越宽松得分越高）进行打分，选取得分最高的前 K 个加入库。

#### 3. 程序重写与交替优化

- **重写过程**：利用已构建的库，对每个程序执行模式匹配并替换为抽象函数调用。匹配时仍使用 e‑graph，以保证在同一等价类中找到最短的表示。  
- **交替迭代**：完成一次全局重写后，系统重新评估库中每个抽象的实际压缩效果。如果某个抽象在新程序中使用率下降或导致新出现的冗余，可能被剔除或重新参数化。随后，使用更新后的库再次进行抽象搜索，产生新的候选抽象。循环若干次（通常 3~5 次）后，库和程序基本收敛。  
- **最巧妙的点**：把抽象搜索和程序重写放在同一个等价图里完成，使得两者可以共享子结构信息，极大降低了搜索空间；同时，条件重写让系统能够安全地处理带有复杂参数的抽象，而不必穷举所有可能的参数组合。

### 实验与效果

- **数据集**：作者在三个不同的 3D 形状域上评估：手工标注的家具模型、从 ShapeNet 自动抽取的机械部件、以及使用无监督立方体抽象方法得到的随机建筑块集合。所有基元都是轴对齐的立方体或长方体。  
- **对比基线**：包括传统的 DSL 库学习方法（如 DreamCoder、DSL‑Synth），以及仅使用手工 DSL 的直接程序生成。  
- **压缩效果**：论文声称在所有测试集上，ShapeCoder 的程序平均长度比最强基线短 10%~20%，并且库规模更小（抽象函数数量约为基线的 1/2），说明发现的抽象更具表达力。  
- **下游任务**：作者展示了使用重写后程序进行形状编辑（如修改抽象函数的参数）比直接编辑基元更直观，且在形状检索任务中，基于抽象的特征比原始基元特征提升了约 15% 的检索准确率。  
- **消融实验**：去掉条件重写会导致抽象函数数量激增且很多无效，压缩率下降约 8%；不使用 e‑graph 而改为暴力子树匹配则运行时间从几分钟涨到数小时，验证了等价图的必要性。  
- **局限性**：作者指出当前实现只能处理轴对齐的立方体基元，对曲面或非刚性变形的形状支持不足；此外，抽象函数的参数表达仍局限于线性或简单非线性形式，复杂的几何关系仍需手工介入。

### 影响与延伸思考

ShapeCoder 把“从无结构几何块到可解释程序”的全链路自动化打开了新局面。自发表后，已有工作尝试将其思路迁移到 2D 矢量图、CAD 零件库以及机器人动作序列的抽象学习，证明等价图 + 条件重写的组合在高维符号搜索中具备普适性。未来的研究方向可能包括：  
- **更丰富的基元类型**（如球体、曲面），以及对应的几何等价规则；  
- **学习条件约束**：让网络自行推断抽象的参数范围，而不是手工设定；  
- **跨模态抽象**：把视觉程序与自然语言描述或物理仿真结果关联，形成多模态的可解释模型。  
对想深入的读者，可以关注近期在 “Program Synthesis for 3D Modeling” 方向的会议论文，尤其是结合神经符号搜索的最新进展。

### 一句话记住它

ShapeCoder 用等价图和条件重写，让机器在只有散乱几何块的情况下自动发掘高层抽象，生成紧凑、可编辑的视觉程序。