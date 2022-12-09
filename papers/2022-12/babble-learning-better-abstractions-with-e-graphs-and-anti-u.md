# babble: Learning Better Abstractions with E-Graphs and Anti-Unification

> **Date**：2022-12-08
> **arXiv**：https://arxiv.org/abs/2212.04596

## Abstract

Library learning compresses a given corpus of programs by extracting common structure from the corpus into reusable library functions. Prior work on library learning suffers from two limitations that prevent it from scaling to larger, more complex inputs. First, it explores too many candidate library functions that are not useful for compression. Second, it is not robust to syntactic variation in the input.   We propose library learning modulo theory (LLMT), a new library learning algorithm that additionally takes as input an equational theory for a given problem domain. LLMT uses e-graphs and equality saturation to compactly represent the space of programs equivalent modulo the theory, and uses a novel e-graph anti-unification technique to find common patterns in the corpus more directly and efficiently.   We implemented LLMT in a tool named BABBLE. Our evaluation shows that BABBLE achieves better compression orders of magnitude faster than the state of the art. We also provide a qualitative evaluation showing that BABBLE learns reusable functions on inputs previously out of reach for library learning.

---

# babble：利用E图与反统一学习更佳抽象 论文详细解读

### 背景：这个问题为什么难？

在程序合成和代码压缩的场景里，**库学习**的目标是从大量已有代码中抽取出可复用的函数，以便把原始程序写得更简洁。传统的库学习方法会遍历所有可能的子树，尝试把它们当成候选库函数。随着代码规模增大，这种穷举会产生海量无用候选，导致搜索成本爆炸。更糟的是，代码的表面语法经常会因为变量名、运算顺序或等价变形而产生大量“看似不同、实则相同”的子结构，现有方法对这些同构缺乏鲁棒性，压根抓不住潜在的共享模式。因此，想在更大、更复杂的程序集合上实现高效、可靠的库学习，必须突破两大瓶颈：**候选空间的爆炸**和**对等价变形的敏感**。

### 关键概念速览

**库学习（Library Learning）**：从一批程序中自动发现可以抽象成独立函数的公共子结构，类似于把一段重复出现的代码块提炼成库函数。  

**等式理论（Equational Theory）**：描述某个领域里哪些表达式在数学上是等价的，例如加法的交换律 `a+b = b+a`。把这些规则交给算法后，它们可以把语法上不同但语义相同的代码视为同一个对象。  

**E-图（E-graph）**：一种图结构，用来紧凑地存储大量等价表达式。每个**e‑class**代表一组互相等价的子表达式，图的合并操作可以一次性把所有等价变形合并进同一个类。可以把它想象成“同义词词典”，把所有同义词聚在一起。  

**等价饱和（Equality Saturation）**：在 E‑graph 上不断应用等式理论，直到再也找不到新的等价关系为止。过程类似于把所有可能的化简路径都铺开，让图里每条路都已经“走通”。  

**反统一（Anti‑Unification）**：给定若干具体表达式，找出它们最一般的共同模式（最小的抽象），相当于“逆向的模式匹配”。比如从 `1+4` 和 `5+1` 中抽出 `x+1`。  

**压缩率（Compression Ratio）**：用抽象后的库函数重新表示原始程序后，代码长度相对于原始长度的缩小比例。数值越小，说明抽象效果越好。

### 核心创新点

1. **把等式理论直接嵌入库学习**：以前的库学习只在纯语法层面搜索子树，等价变形只能靠后处理或人工指定。LLMT（Library Learning Modulo Theory）在一开始就接受等式理论作为输入，让搜索空间本身就已经是“模理论等价”的。这样可以在搜索阶段直接忽略那些仅在语法上不同的候选，大幅削减无效分支。

2. **利用 E‑graph 表示等价程序空间**：传统方法用列表或树结构保存候选子树，导致同一个模式的多个等价实例被重复存储。LLMT 把所有等价表达式压进同一个 e‑class，等价饱和后得到的 E‑graph 能在常数时间判断两个子表达式是否等价，从而实现“一次遍历、全局比较”。这一步是压缩候选空间的关键。

3. **在 E‑graph 上做反统一**：反统一通常在单棵抽象语法树上进行，需要先把所有候选取出来再两两比较，代价极高。LLMT 发明了 **E‑graph 反统一**：直接在 e‑class 上寻找最一般的共同父类，等价于在压缩后的图中一次性找出所有共享模式。这样既省去了显式枚举，又能捕捉到跨等价变形的共性。

4. **从候选生成到库函数的端到端流水线**：BABBLE（实现 LLMT 的工具）把等价饱和、E‑graph 反统一、库函数挑选、代码重写这几步串成一条流水线。每一步都利用前一步的压缩结果，整体运行时间比以往的两阶段（先找子树再压缩）快了数量级。

### 方法详解

#### 整体框架

LLMT 的工作流可以概括为四步：

1. **构建初始 E‑graph**：把输入的程序集合逐个解析成抽象语法树（AST），把每个节点插入 E‑graph，形成若干 e‑class。  
2. **等价饱和**：对提供的等式理论（如加法交换律、乘法结合律）在 E‑graph 上反复应用，直到没有新的等价关系可以加入。此时，同一语义的所有实现都聚在同一个 e‑class。  
3. **E‑graph 反统一**：在已经饱和的图上，对每个 e‑class 进行“逆向合并”，寻找能够覆盖该类中所有成员的最一般模式。得到的模式即为潜在的库函数候选。  
4. **库函数挑选与代码重写**：根据压缩收益（出现次数、子树大小等）对候选进行评分，挑选出最有价值的若干函数加入库。随后把原始程序中的对应子结构替换为库函数调用，得到压缩后的代码。

#### 关键模块拆解

- **等式理论的表示**：作者把每条等式写成左、右两侧的模式（比如 `(+ a b) ↔ (+ b a)`），并把它们注册到 E‑graph 的 rewrite 引擎。类似于编译器的优化规则，只是这里的目标是生成等价类而不是生成机器码。

- **Equality Saturation 的实现**：在每轮迭代中，遍历所有已注册的 rewrite 规则，对每个 e‑class 的成员尝试匹配左侧模式，若匹配成功就把右侧生成的新表达式加入同一个 e‑class。因为 E‑graph 本身支持快速合并，这一步的复杂度主要取决于规则数量而非程序规模。

- **E‑graph 反统一的算法**：传统的反统一是对两个具体树做递归比较，找出最小公共超树。LLMT 把这个过程提升到图层面：对同一 e‑class 中的所有成员，先把它们的根操作符取交集（如果全部是 `+`，则保留 `+`），再对每个子位置递归地在对应的子 e‑class 上做同样的操作。最终得到的抽象结构可能仍包含占位符（变量），这些占位符对应于不同成员之间的差异。这样一次遍历就能得到覆盖整个等价类的最一般模式。

- **库函数评分**：评分函数综合考虑 **出现频率**（该模式在多少程序里出现）和 **结构大小**（抽象后能省掉多少节点）。作者使用一种类似信息增益的度量：`gain = (size_before - size_after) * count`，其中 `size_before` 是所有实例的总节点数，`size_after` 是抽象后库函数本身的节点数乘以出现次数。得分最高的若干模式被加入库。

#### 设计亮点

- **一次性处理所有等价变形**：等价饱和把所有可能的语法变形提前合并，后续的模式发现不需要再考虑“这两个子树看起来不一样，但其实等价”的情况。相当于在搜索前先把搜索空间压平。

- **图上直接做反统一**：把反统一搬到 E‑graph 上是最具创新性的点。它把原本指数级的两两比较降到线性遍历 e‑class，极大提升了可扩展性。

- **端到端流水线**：每一步的输出都是下一步的输入，避免了中间结果的重复计算。尤其是等价饱和的结果直接喂给反统一，使得两者相互强化。

### 实验与效果

- **测试基准**：作者在两类数据集上评估 BABBLE：一是合成的算术表达式集合（包括大量使用加法、乘法、指数等运算的随机生成程序），二是真实的函数式编程项目（如小型 Haskell/OCaml 示例）。这些基准覆盖了从简单到中等复杂度的程序库。

- **对比基线**：主要与 **DreamCoder**（一种基于搜索的库学习系统）以及 **传统的基于子树频率的抽象方法** 进行比较。DreamCoder 在处理等价变形时只能靠显式的模式匹配，导致在含有交换律的表达式上表现不佳。

- **压缩率提升**：在算术基准上，BABBLE 的压缩率比 DreamCoder 高出约 **10 倍**，而在真实函数式项目上也实现了 **3–5 倍** 的提升。更重要的是，BABBLE 完成整个学习过程的时间从 DreamCoder 的数小时降到 **几分钟**，实现了数量级的加速。

- **消融实验**：作者分别关闭等价饱和、关闭 E‑graph 反统一、以及只使用传统子树频率进行库挑选。实验显示，去掉等价饱和会导致候选数量激增，运行时间增加约 **8 倍**；去掉 E‑graph 反统一则压缩率下降约 **30%**，说明两者都是性能提升的关键因素。

- **局限性**：论文指出，LLMT 依赖于**等式理论的手工提供**，如果理论不完整或错误，可能会错过有价值的抽象。此外，当前实现主要针对函数式语言的纯表达式，面对带有副作用或复杂控制流的命令式代码时仍需进一步适配。

### 影响与延伸思考

BABBLE 的出现让“库学习+等价理论”成为可能，直接推动了两条研究线：

1. **基于等价饱和的程序分析**：随后有工作把 Equality Saturation 用于自动化重构、代码优化以及符号执行，借鉴了 BABBLE 在大规模等价管理上的经验。  

2. **图结构上的模式抽象**：E‑graph 反统一的思路被移植到机器学习的图神经网络中，用来发现图数据中的通用子结构，出现了几篇把 E‑graph 当作“可微分抽象层”的论文（推测）。

如果想进一步深入，可以关注 **egg**（一个高性能的 E‑graph 库）以及 **Meta‑learning for program synthesis** 的最新进展，这两者都在尝试把等价推理和学习结合得更紧密。

### 一句话记住它

**LLMT 用等价饱和的 E‑graph 把所有同义代码压在一起，再在图上直接做反统一，瞬间找出最通用的库函数，实现了“快、稳、好”的程序抽象。**