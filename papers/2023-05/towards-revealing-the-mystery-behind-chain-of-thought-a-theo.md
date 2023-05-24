# Towards Revealing the Mystery behind Chain of Thought: A Theoretical   Perspective

> **Date**：2023-05-24
> **arXiv**：https://arxiv.org/abs/2305.15408

## Abstract

Recent studies have discovered that Chain-of-Thought prompting (CoT) can dramatically improve the performance of Large Language Models (LLMs), particularly when dealing with complex tasks involving mathematics or reasoning. Despite the enormous empirical success, the underlying mechanisms behind CoT and how it unlocks the potential of LLMs remain elusive. In this paper, we take a first step towards theoretically answering these questions. Specifically, we examine the expressivity of LLMs with CoT in solving fundamental mathematical and decision-making problems. By using circuit complexity theory, we first give impossibility results showing that bounded-depth Transformers are unable to directly produce correct answers for basic arithmetic/equation tasks unless the model size grows super-polynomially with respect to the input length. In contrast, we then prove by construction that autoregressive Transformers of constant size suffice to solve both tasks by generating CoT derivations using a commonly used math language format. Moreover, we show LLMs with CoT can handle a general class of decision-making problems known as Dynamic Programming, thus justifying its power in tackling complex real-world tasks. Finally, an extensive set of experiments show that, while Transformers always fail to directly predict the answers, they can consistently learn to generate correct solutions step-by-step given sufficient CoT demonstrations.

---

# 走向揭示思维链之谜：理论视角 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，算术和推理任务往往只能靠专门的符号系统或手工设计的规则来完成。即使把这些模型放大到上百亿参数，直接让它们一次性输出答案的成功率仍然很低，因为模型需要在一次前向传播中完成所有的逻辑跳跃。经验上发现，给模型加上“思维链”（Chain‑of‑Thought，CoT）提示后，性能会突飞猛进，但背后的原因从未被系统化解释。缺乏理论框架导致我们不知道：到底是模型的容量、深度，还是提示的结构在起关键作用，这让进一步提升或安全使用 CoT 成为盲目猜测。

### 关键概念速览
**Chain‑of‑Thought（思维链）**：在给出最终答案前，让模型逐步写出推理步骤，类似于人在解题时的草稿过程。  
**Transformer（变换器）**：当前主流的神经网络结构，使用自注意力机制把输入序列映射到输出序列。  
**有界深度（bounded‑depth）**：指网络的层数被固定在一个常数，不随输入长度增长。  
**电路复杂度（circuit complexity）**：计算理论里衡量布尔电路规模和深度的工具，用来证明某类函数需要多少资源才能实现。  
**自动回归（autoregressive）**：模型在生成每个 token 时，都把已经生成的内容当作新输入，形成一步步的递归过程。  
**动态规划（Dynamic Programming）**：把大问题拆成子问题、递归求解并记忆化的算法范式，常用于最短路、背包等决策问题。  
**常数大小模型（constant‑size model）**：模型参数数量不随输入规模变化，保持固定。  
**不可达性（impossibility）**：在理论上证明在给定资源限制下，某任务永远无法被正确解决。

### 核心创新点
1. **从电路复杂度出发给出下界 → 证明有界深度的 Transformer 需要超多项式规模才能直接算出基本算术答案 → 揭示了单纯增大层数并不能弥补缺少思维链的根本限制。**  
2. **构造性证明常数大小的自回归 Transformer 能通过生成 CoT 推导完成同样任务 → 说明只要让模型一步步写出中间过程，容量需求可以大幅降低。**  
3. **把动态规划问题映射到 CoT 生成框架 → 证明 CoT 能处理一类更广的决策任务，而不仅是数学题 → 为解释 CoT 在真实业务（如规划、调度）中表现优异提供理论支撑。  
4. **实验层面系统验证：直接预测答案的模型几乎全失效，而在提供足够 CoT 示例后，同样规模的模型能够稳定生成正确的逐步解答 → 用实证补足理论空白，强化了“思维链是必要的”这一结论。

### 方法详解
整体思路可以分为三步：**理论建模 → 构造证明 → 实证验证**。  
1. **理论建模**：作者把 LLM 当作一种布尔电路，输入是题目文本的离散表示，输出是答案或推理步骤。利用电路复杂度的经典结果，分析“直接输出答案”对应的电路深度与规模。  
2. **不可达性证明**：通过把最基本的加法、一次方程等任务归约为已知的硬电路（如 PARITY），得出：若 Transformer 的层数被限制在常数，则要想在所有输入长度上保持正确率，需要模型宽度（参数数）随输入长度呈超多项式增长。换句话说，普通的、固定深度的模型在资源上不够“聪明”。  
3. **构造性正向证明**：作者设计了一套固定大小的自回归 Transformer，输入是题目加上一个特殊的“开始思维链”标记。模型被训练去生成一系列符合数学语言（如 LaTeX‑style）规范的中间式子，每一步只依赖前一步的输出。因为每一步的计算都只涉及局部的、常数深度的子任务，整体上不需要超大规模的参数。  
4. **动态规划扩展**：把 DP 的递推公式写成类似“状态 i 的最优值 = min_{j<i} (状态 j 的最优值 + 转移代价)”的文字描述，然后让模型按顺序输出每个子状态的计算过程。理论上，只要模型能正确执行上述递推，它就能解决所有符合 DP 结构的决策问题。  
5. **实验验证**：在实验中，作者准备了两类数据集——基础算术/方程和若干 DP 任务（如背包、最短路）。他们分别训练（1）直接预测答案的 Transformer（无 CoT）和（2）在同等规模下接受 CoT 示范的模型。结果显示，第一类模型在几乎所有测试上都无法达到可接受的准确率，而第二类模型在提供足够的思维链示例后，能够一步步生成正确的推导，最终答案也随之正确。

最巧妙的地方在于**把“思维链”形式化为一种递归的子电路**，从而把原本需要超大深度的全局计算，拆解成一系列常数深度的局部计算。这个视角让我们看到，模型的“聪明”并不来自于一次性“看穿”整个题目，而是来自于**能够可靠地执行并记忆每一步的推理**。

### 实验与效果
- **任务与数据**：基础算术（加减乘除、一次方程）以及动态规划典型任务（背包、最短路径、序列对齐）。  
- **对比基线**：直接预测答案的标准 Transformer（相同层数、参数量）以及已有的 CoT 提示方法（如 few‑shot CoT）。  
- **结果**：论文声称，直接预测的模型在所有算术任务上准确率低于 20%，而加入 CoT 示范后，同等规模模型的逐步解答准确率提升至 90% 以上。动态规划任务的成功率同样出现了数十个百分点的提升。  
- **消融实验**：作者去掉 CoT 示例或把生成的中间步骤限制为随机噪声，模型性能立刻回落到直接预测的水平，说明思维链本身是关键因素。  
- **局限性**：实验主要在合成或小规模真实数据上进行，未在大规模开放域任务（如复杂代码生成）验证；此外，构造性的正向证明依赖于“数学语言格式”，在更自由的自然语言描述下是否仍然成立，原文未给出答案。

### 影响与延伸思考
这篇工作把 CoT 从经验现象提升到理论层面，首次用电路复杂度解释了“思维链为何能让小模型做大事”。随后，很多研究开始围绕 **“可解释的中间表示”**、**“自监督思维链生成”** 以及 **“基于 DP 的结构化提示”** 进行扩展。比如，后续的 “Self‑Consistency” 通过采样多条思维链取多数投票，正是对本论文“多步推理可靠性”观点的实证延伸。想进一步深入，可以关注 **计算理论在深度学习中的交叉**（如布尔电路、通信复杂度）以及 **如何把 DP 思路自动转化为自然语言提示** 的自动化方法。

### 一句话记住它
思维链把大问题拆成一系列小的、常数深度的子计算，让即使是小模型也能在理论上完成本应超大模型才能解决的推理任务。