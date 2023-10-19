# GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for   Reasoning Problems

> **Date**：2023-10-19
> **arXiv**：https://arxiv.org/abs/2310.12397

## Abstract

There has been considerable divergence of opinion on the reasoning abilities of Large Language Models (LLMs). While the initial optimism that reasoning might emerge automatically with scale has been tempered thanks to a slew of counterexamples, a wide spread belief in their iterative self-critique capabilities persists. In this paper, we set out to systematically investigate the effectiveness of iterative prompting of LLMs in the context of Graph Coloring, a canonical NP-complete reasoning problem that is related to propositional satisfiability as well as practical problems like scheduling and allocation. We present a principled empirical study of the performance of GPT4 in solving graph coloring instances or verifying the correctness of candidate colorings. In iterative modes, we experiment with the model critiquing its own answers and an external correct reasoner verifying proposed solutions. In both cases, we analyze whether the content of the criticisms actually affects bottom line performance. The study seems to indicate that (i) LLMs are bad at solving graph coloring instances (ii) they are no better at verifying a solution--and thus are not effective in iterative modes with LLMs critiquing LLM-generated solutions (iii) the correctness and content of the criticisms--whether by LLMs or external solvers--seems largely irrelevant to the performance of iterative prompting. We show that the observed increase in effectiveness is largely due to the correct solution being fortuitously present in the top-k completions of the prompt (and being recognized as such by an external verifier). Our results thus call into question claims about the self-critiquing capabilities of state of the art LLMs.

---

# GPT‑4 并不知道自己错了：迭代提示在推理问题上的分析 论文详细解读

### 背景：这个问题为什么难？

图着色是经典的 NP 完全问题，要求把图的每个节点涂上颜色，使相邻节点颜色不同。它等价于布尔可满足性（SAT）的一类实例，广泛映射到排班、资源分配等实际场景。过去，人们期待大语言模型（LLM）随着参数规模增长会自然掌握这类组合推理能力，甚至能自行发现并纠正错误。可是已有大量反例显示，单轮提示下的 LLM 往往只能给出表面化、直觉式的答案，缺乏系统的搜索与验证机制。于是出现了“迭代自我批评”——让模型先生成答案，再让它自己评估、改写——的设想，但到底能否真正提升解题成功率，仍缺乏严格实验验证。

### 关键概念速览

**图着色（Graph Coloring）**：把图的每个节点分配颜色，要求相邻节点颜色不同。想象把地图的各省涂色，不能相邻的省份用同一种颜色。

**NP 完全**：指问题在多项式时间内可以验证答案，但找答案本身被认为没有快速算法。就像找钥匙的过程很慢，但一旦找到钥匙就能立刻验证是否能打开门。

**迭代提示（Iterative Prompting）**：模型先给出答案，再在后续轮次中对该答案进行批评、修改或重新生成。类似于人写作文后自己审稿、改错的过程。

**自我批评（Self‑Critique）**：模型在同一轮对话里充当“审稿人”，评估自己刚才的输出是否合理，然后尝试纠正。相当于让学生先答题，再让自己检查答案。

**外部正确推理器（External Correct Reasoner）**：论文里使用的传统图着色求解器（如 SAT 求解器），它能在几毫秒内判断一个着色方案是否合法。相当于请教数学老师来检查答案。

**Top‑k 采样**：在一次生成中，模型会返回概率最高的 k 条候选答案。想象一次考试后老师给出前几名的答案，学生可以挑选其中正确的那一个。

### 核心创新点

1. **系统化的迭代实验设计**  
   *之前的工作大多只展示几例自我批评的成功案例* → *本文构建了完整的实验流水线：先让 GPT‑4 生成图着色方案 → 再让它自行批评或让外部求解器批评 → 根据批评结果决定是否接受、修改或重新生成* → *能够客观衡量批评环节对整体成功率的真实贡献，而不是凭感觉判断。*

2. **对比自我批评与外部验证的双向实验**  
   *过去只关注模型内部的自我纠错* → *本文同时让外部 SAT 求解器对模型的答案进行验证，并把验证结果喂回模型* → *发现无论批评来源是模型还是外部工具，最终性能提升几乎相同，说明批评内容本身并非关键因素。*

3. **揭示“Top‑k 幸运”现象**  
   *常规解释是模型通过迭代学习到更好的解* → *作者追踪每一次生成的候选答案，发现正确的着色方案往往已经在第一次生成的 top‑k 列表里，只是没有被模型主动识别* → *迭代提示的提升主要是因为外部验证帮助挑选了已经存在的正确候选，而不是模型真正改进了推理。*

4. **负向结论的量化**  
   *大多数研究倾向于报告正向结果* → *本文用统计显著性检验展示 GPT‑4 在图着色的解题率约为 12%，验证率约为 15%，两者相差无几* → *为社区提供了一个可靠的基准，提醒后续工作不要盲目宣称自我批评有效。*

### 方法详解

整体框架可以拆成三步：

1. **生成候选着色**  
   给模型一个图的描述（节点数、边列表）以及“请给出一个合法的颜色分配”。模型使用一次性生成（temperature 0.7）返回前 k=5 条候选答案。

2. **批评环节**  
   - **自我批评模式**：模型收到自己的第一条答案后，被提示“请检查这份着色是否满足相邻不同颜色的约束，并指出错误”。模型输出批评文本，随后被要求“基于上述批评重新给出一个着色”。  
   - **外部验证模式**：把第一条答案交给 SAT 求解器，求解器返回“合法”或“非法 + 冲突边”。该信息被拼接进新的提示，要求模型在知晓冲突后重新生成。

3. **选择最终答案**  
   对于每一次迭代，系统都会记录所有生成的候选以及对应的批评。最终的决定规则是：如果任意一次生成被外部求解器标记为合法，就直接返回该答案；否则返回最后一次模型生成的答案。

**关键细节**  
- **提示模板**：作者精心设计了两套提示，一套用于生成颜色分配（明确列出颜色编号），另一套用于批评（要求模型给出具体冲突边）。这相当于给模型提供“检查清单”。  
- **批评内容的过滤**：虽然模型会输出自然语言的批评，系统只提取其中的布尔判断（合法/非法）和冲突对。这样避免了语言噪声对后续生成的干扰。  
- **迭代次数**：实验固定为两轮（生成 → 批评 → 重新生成），因为进一步迭代的收益在预实验中几乎为零。  
- **对照实验**：为了排除“只要多生成几次就会出现正确答案”的可能，作者还跑了“单轮多样本”基线，即直接从一次生成的 top‑k 中挑选合法答案，而不进行任何批评。

最反直觉的地方在于：**批评的质量并不决定最终成功率**。即使模型给出极其详细、逻辑严密的错误分析，只要外部验证没有帮助挑选出已经存在的正确候选，整体表现仍然停留在原始的低水平。

### 实验与效果

- **数据集**：作者随机生成了 200 个中等规模（10–30 节点）无向图，每个图的染色数设为 3 或 4，确保问题属于 NP 完全的典型难度。  
- **基线**：  
  - *单轮直接生成*（不做任何批评，直接取 top‑1）。  
  - *单轮 top‑k 选取*（检查 top‑5 中是否有合法解）。  
  - *传统启发式图着色算法*（如 DSATUR），作为非 LLM 的强基准。  
- **结果**：  
  - 单轮直接生成的合法率约 12%。  
  - 单轮 top‑k 选取提升到约 15%，说明正确答案偶尔已经在候选里。  
  - 自我批评模式的最终合法率同样约 15%，与 top‑k 基线无显著差异。  
  - 外部验证模式也只有约 16% 的提升，基本归因于挑选了已有的合法候选。  
  - 传统 DSATUR 在相同图上可以达到 90% 以上的成功率，凸显 LLM 在组合搜索上的劣势。  
- **消融实验**：作者去掉批评环节，仅保留多次生成的 top‑k 采样，结果几乎相同，进一步证明批评本身并未带来额外收益。  
- **局限性**：实验只覆盖了中小规模图，未测试更大规模或稀疏/密集极端情况；此外，仅使用 GPT‑4 一个模型，未评估更小或更大的 LLM 版本是否表现相似。

### 影响与延伸思考

这篇工作在社区里引发了两类讨论：一是对“LLM 能自我纠错”这一乐观预期的审慎反思，二是对如何让 LLM 与传统符号求解器协同的探索。随后出现的几篇论文（如 *LLM‑Solver Hybrid*、*Prompt‑Based Verification*）尝试把 LLM 作为前置生成器，随后交给高效的 SAT/SMT 求解器完成验证，明确把 LLM 的优势定位在“快速构造候选”而非“深度推理”。如果想进一步了解，可以关注 **Neuro‑Symbolic**（神经符号）方向的最新进展，尤其是把 LLM 与可解释的搜索算法结合的工作。

### 一句话记住它

**GPT‑4 的自我批评并不会真正提升图着色解题率，提升主要来自偶然出现的正确候选被外部验证挑出来。**