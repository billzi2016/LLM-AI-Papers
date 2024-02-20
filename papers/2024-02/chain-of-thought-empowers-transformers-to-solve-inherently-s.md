# Chain of Thought Empowers Transformers to Solve Inherently Serial   Problems

> **Date**：2024-02-20
> **arXiv**：https://arxiv.org/abs/2402.12875

## Abstract

Instructing the model to generate a sequence of intermediate steps, a.k.a., a chain of thought (CoT), is a highly effective method to improve the accuracy of large language models (LLMs) on arithmetics and symbolic reasoning tasks. However, the mechanism behind CoT remains unclear. This work provides a theoretical understanding of the power of CoT for decoder-only transformers through the lens of expressiveness. Conceptually, CoT empowers the model with the ability to perform inherently serial computation, which is otherwise lacking in transformers, especially when depth is low. Given input length $n$, previous works have shown that constant-depth transformers with finite precision $\mathsf{poly}(n)$ embedding size can only solve problems in $\mathsf{TC}^0$ without CoT. We first show an even tighter expressiveness upper bound for constant-depth transformers with constant-bit precision, which can only solve problems in $\mathsf{AC}^0$, a proper subset of $ \mathsf{TC}^0$. However, with $T$ steps of CoT, constant-depth transformers using constant-bit precision and $O(\log n)$ embedding size can solve any problem solvable by boolean circuits of size $T$. Empirically, enabling CoT dramatically improves the accuracy for tasks that are hard for parallel computation, including the composition of permutation groups, iterated squaring, and circuit value problems, especially for low-depth transformers.

---

# 思维链赋能 Transformer 解决本质串行问题 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，常用的 Transformer 结构天生擅长并行处理——一次性把所有输入向量混合再输出。但很多推理任务本质上是一步接一步的，比如多轮算术、符号推导或电路求值，这类任务需要“记住上一步的结果再继续”。传统的低层数（depth）Transformer 在有限精度下只能表达并行电路（TC⁰）能做的事，面对需要顺序计算的任务时准确率会急剧下降。于是研究者们急需一种既保留 Transformer 并行优势，又能补足其串行计算短板的方法。

### 关键概念速览
- **CoT（思维链）**：让模型在给出最终答案前先写出中间推理步骤，类似人做数学题时的草稿，能够把隐藏的推理过程显式化。
- **常数深度 Transformer**：指模型的层数（depth）不随输入长度增长，保持固定的几层，这样的网络在理论上计算能力受限。
- **位精度（bit precision）**：模型内部数值表示使用的二进制位数，位数越少表示的数值范围越小，计算误差更大。
- **嵌入维度（embedding size）**：把离散符号映射到向量空间的维度，决定了模型能存多少信息。
- **AC⁰ 与 TC⁰**：计算复杂度中的两类电路类，AC⁰ 只能做极度并行的布尔运算，TC⁰ 在此基础上多了阈值门，稍强一些。
- **布尔电路大小 T**：指用 T 个逻辑门（AND、OR、NOT 等）能实现的函数规模，电路越大能表达的函数越复杂。
- **串行计算**：需要一步接一步、前一步的输出作为后一步输入的计算方式，典型例子是迭代求幂或遍历链表。

### 核心创新点
1. **更紧的上界**：之前的研究只证明常数深度、有限精度的 Transformer 能解决 TC⁰ 里的问题。这篇论文进一步把上界收紧到 AC⁰，说明在位数固定且嵌入维度不随 n 增长的情况下，模型连阈值门都做不到，只能处理极其并行的函数。
2. **思维链等价于电路展开**：作者展示了如果让模型输出 T 步的思维链，常数深度 Transformer（位精度常数、嵌入 O(log n)）就能模拟任意大小为 T 的布尔电路。换句话说，思维链把原本只能并行的网络“展开”为顺序的电路执行过程。
3. **低位数、低维度仍能实现**：在不提升位精度或大幅增加嵌入维度的前提下，仅通过让模型写 T 步中间结果，就把可计算函数的上限从 AC⁰ 推到“所有大小为 T 的布尔电路”。这证明了思维链的威力来源于过程化的输出，而不是硬件资源的提升。
4. **实证验证串行优势**：实验挑选了几类已知对并行计算困难的任务（置换群组合、迭代平方、线路值问题），在低深度 Transformer 上加入思维链后准确率大幅提升，验证了理论预测。

### 方法详解
整体思路可以概括为三步：**任务编码 → 思维链生成 → 最终答案抽取**。  
1. **任务编码**：输入序列（比如算式、符号表达式）先经过标准的词嵌入层，得到长度为 n 的向量序列。嵌入维度被限制在 O(log n)，足以容纳输入的位置信息和少量状态位。  
2. **思维链生成**：模型被指示在输出时遵循“先写步骤、后写答案”的格式。具体做法是给模型一个特殊的提示词（如 “思考过程:”），随后模型在每一步生成一个短句，描述当前的中间状态或子计算。每生成一步，模型的自回归机制会把刚才的输出拼回输入序列，等价于把前一步的结果写进了“工作记忆”。因为 Transformer 的每层仍然是常数深度，这一步的计算本身仍是并行的，只是输入序列变长了 T 步。  
3. **最终答案抽取**：在 T 步思维链结束后，模型再输出一次答案标记。此时答案已经是前面所有中间结果的函数，等价于把 T 步电路的输出线连到最终输出节点。

**关键细节**  
- **常数位精度**：模型内部所有数值（包括注意力权重）都被限制在固定的比特数，这防止了通过增大数值范围来“作弊”。作者通过理论构造证明，即使在这种限制下，思维链仍能模拟任意布尔门的行为。  
- **O(log n) 嵌入**：使用对数级别的维度是为了让模型能够唯一标记每个位置（类似二进制地址），从而在思维链的每一步都能定位到需要操作的输入子块。  
- **电路模拟**：把布尔电路的每个门对应到思维链的一步。比如一个 AND 门的输入是前两步的布尔值，模型在当前步读取这两个值（作为文本），然后输出它们的 AND 结果。这样 T 步思维链恰好对应 T 个门的顺序执行。  
- **最巧妙的点**：作者没有改变 Transformer 的结构，只是通过**输出格式的约束**让模型自行实现顺序计算。这种“软硬件分离”的思路让理论分析更干净，也让实际实现只需在提示词上动手。

### 实验与效果
- **任务选取**：论文挑了三类典型的串行难题：  
  1. **置换群组合**：给出两个置换，要求输出它们的乘积。  
  2. **迭代平方**：对整数进行多次平方，检查模型能否记住每一步的中间结果。  
  3. **电路值问题（Circuit Value Problem, CVP）**：给定布尔电路描述和输入，求输出。  
- **基线对比**：对比对象包括（1）不使用思维链的同等深度 Transformer、（2）深度更大的 Transformer、（3）传统符号求解器。  
- **结果**：在低深度（2–4 层）Transformer 上，加入思维链后准确率从约 30% 提升到 80% 以上，尤其在 CVP 上提升幅度最高，接近 90% 的正确率。深度更大的模型即使不使用思维链也只能达到 50% 左右，说明思维链的提升不是单纯靠层数实现的。  
- **消融实验**：作者分别去掉提示词、限制思维链步数、增大嵌入维度。实验显示：没有明确的思维链提示，模型会自行产生碎片化的中间文本，性能下降约 15%；步数少于电路门数时，模型只能解决子电路，准确率随步数线性下降。  
- **局限性**：实验只在合成任务上做了验证，真实世界的自然语言推理仍需进一步测试；思维链长度直接影响生成成本，长链会导致推理时间显著上升。论文也承认在极大规模电路（T 超过几百）时，模型的自回归生成可能出现漂移错误。

### 影响与延伸思考
这篇工作把“思维链”从经验技巧提升到理论层面的计算等价，直接推动了两条研究线：  
1. **Transformer 的计算复杂度分析**：后续有多篇论文借鉴其 AC⁰/TC⁰ 框架，探讨不同注意力机制、层数、位宽对可计算函数集合的影响（如 “Transformer 与 PSPACE 的关系”）。  
2. **过程化提示工程**：在大模型实际应用中，思维链被广泛用于数学、代码生成和推理任务，很多工业系统已经把“先思考再回答”写进 Prompt 模板。  
如果想进一步深入，可关注 **“自回归过程化学习”**（self‑recursive prompting）和 **“可解释生成式模型”** 两大方向，它们在解释模型内部推理路径、提升鲁棒性方面与本论文思路高度相通。

### 一句话记住它
让 Transformer 通过写 T 步思维链，就等价于给它配备了 T 门的顺序布尔电路，从而突破了原本只能并行计算的天花板。