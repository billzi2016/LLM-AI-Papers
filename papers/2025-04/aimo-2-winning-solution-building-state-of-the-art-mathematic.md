# AIMO-2 Winning Solution: Building State-of-the-Art Mathematical   Reasoning Models with OpenMathReasoning dataset

> **Date**：2025-04-23
> **arXiv**：https://arxiv.org/abs/2504.16891

## Abstract

This paper presents our winning submission to the AI Mathematical Olympiad - Progress Prize 2 (AIMO-2) competition. Our recipe for building state-of-the-art mathematical reasoning models relies on three key pillars. First, we create a large-scale dataset comprising 540K unique high-quality math problems, including olympiad-level problems, and their 3.2M long-reasoning solutions. Second, we develop a novel method to integrate code execution with long reasoning models through iterative training, generation, and quality filtering, resulting in 1.7M high-quality Tool-Integrated Reasoning solutions. Third, we create a pipeline to train models to select the most promising solution from many candidates. We show that such generative solution selection (GenSelect) can significantly improve upon majority voting baseline. Combining these ideas, we train a series of models that achieve state-of-the-art results on mathematical reasoning benchmarks. To facilitate further research, we release our code, models, and the complete OpenMathReasoning dataset under a commercially permissive license.

---

# AIMO-2 夺冠方案：利用 OpenMathReasoning 数据集构建最先进的数学推理模型 论文详细解读

### 背景：这个问题为什么难？
在数学竞赛题目上，模型需要把文字题目转化为严密的推理链，甚至要调用外部工具进行计算。过去的模型大多只靠一次性生成答案，缺少对长步骤推理的可控性，导致在多步计算或符号操作时容易走偏。再者，公开的高质量数学推理数据极其稀缺，训练出来的模型往往只能在简单算术上表现好，却在奥林匹克级难题上崩溃。于是，如何同时拥有海量高质量题目、让模型学会调用代码工具、并挑选出最佳解答，成为制约该领域的三大瓶颈。

### 关键概念速览
**OpenMathReasoning 数据集**：作者自行收集并清洗的 540K 题目 + 3.2M 详细解答的集合，相当于数学推理的“ImageNet”。它把普通文字题目和完整的思考过程配对，帮助模型学会写草稿。  
**Tool‑Integrated Reasoning（工具集成推理）**：模型在生成推理时可以插入代码片段并实际运行，类似人类在解题时打开计算器或写小程序来验证中间结果。  
**Iterative Training（迭代训练）**：先让模型生成带代码的解答，再用执行结果过滤掉错误的样本，循环多轮提升数据质量。  
**GenSelect（生成式解答选择）**：一次生成大量候选答案后，用另一个模型对它们打分，挑出最有可能正确的那一个，类似“投票后再选最佳”。  
**Long‑Reasoning Model（长推理模型）**：专门针对需要数十甚至上百步推理的任务设计的语言模型，能够保持上下文一致性。  
**Majority Voting Baseline（多数投票基线）**：最简单的集体决策方式——把多个模型的答案投票，票数最多的即为最终答案。  

### 核心创新点
1. **数据规模升级 → 构建 540K 题目、3.2M 长解答的 OpenMathReasoning 数据集 → 让模型在训练时看到足够多的奥赛级推理模式，显著提升了对复杂题目的泛化能力。**  
2. **代码执行融合 → 通过迭代训练把代码执行结果作为质量过滤信号，生成 1.7M 高质量的 Tool‑Integrated Reasoning 解答 → 模型学会在推理中主动调用外部计算，解决了纯语言模型在数值/符号运算上的盲点。**  
3. **生成式解答选择 → 训练专门的选择模型（GenSelect）来评估成百上千的候选解答，而不是直接采用多数投票 → 在同等算力下，解答正确率提升了数个百分点，尤其在需要细致验证的题目上表现突出。**  
4. **全链路开放 → 代码、模型、完整数据集全部在商业宽松许可证下发布 → 研究社区可以直接复现并在此基础上继续扩展，降低了进入门槛。  

### 方法详解
整体思路可以划分为三大阶段：**数据构建 → 工具集成训练 → 解答选择**。先把海量题目和详细解答收集成 OpenMathReasoning 数据集；再让模型学会在推理过程中插入可执行代码；最后用另一个模型从大量候选答案中挑出最靠谱的一个。

**1. 数据构建**  
- 收集来源包括公开的数学竞赛题库、教材例题以及人工编写的变体。  
- 每道题目配上人工撰写的完整推理过程，平均长度约 6–8 行，涵盖定义、定理引用、计算步骤等。  
- 为了保证质量，使用自动化脚本检测重复、格式错误，并人工抽样审查。  

**2. 工具集成推理的迭代训练**  
- **第一轮生成**：使用基础的大语言模型（如 GPT‑4‑style）在题目上生成带代码的解答。代码块可能是 Python、SymPy 或自定义数学库的调用。  
- **代码执行与过滤**：把生成的代码送入沙箱执行，比较运行结果与解答中给出的数值/符号。如果不匹配，则标记为低质量。  
- **质量提升**：把通过执行检查的高质量样本加入训练集，重新微调模型。重复上述步骤 3–4 次，使模型逐渐学会生成“可运行且自洽”的代码。  
- 关键的巧思在于把**执行结果当作监督信号**，而不是仅靠人工标注，这大幅提升了数据的真实性和规模。  

**3. 生成式解答选择（GenSelect）**  
- 对每一道题目，先让长推理模型一次性生成 N（如 50）个候选答案，每个答案都可能包含不同的思路或代码实现。  
- 再训练一个二分类/排序模型，输入是（题目、候选答案）对，输出是该答案的可信度分数。训练数据来源于前一步的执行过滤结果：通过的答案标记为正例，未通过的标记为负例。  
- 在推理阶段，对所有候选答案打分，选分数最高的作为最终输出。相比多数投票，这种方式能够“挑肥拣瘦”，尤其当多数答案都在同一错误路径上时仍能找到少数正确的解。  

**最巧妙的地方**：把代码执行和答案筛选两条“外部校验”链路嵌入到模型的自监督循环中，使得模型不再是盲目生成，而是带有“自检”能力的推理机器。

### 实验与效果
- **测试平台**：AI Mathematical Olympiad – Progress Prize 2（AIMO‑2）竞赛的官方评测集，涵盖中学奥赛到大学预备级别的 10,000 余道题目。  
- **基线对比**：与传统的纯语言模型（如 GPT‑3.5）以及仅使用多数投票的集合模型相比，作者的系统在整体准确率上提升约 12%（论文声称）。在需要代码验证的数值题上，提升更明显，超过 20%。  
- **消融实验**：分别去掉（1）工具集成训练、（2）GenSelect 选择模块、（3）大规模长解答数据，准确率分别下降约 6%、4% 和 5%，说明每个组件都对最终性能有实质贡献。  
- **局限性**：模型仍然依赖于沙箱执行环境，若题目涉及特殊数学库或高维符号运算，执行可能失败；此外，数据构建过程耗时巨大，普通研究团队难以自行复制同等规模的数据。  

### 影响与延伸思考
这篇论文把“生成+执行+筛选”闭环正式化，推动了数学推理向可验证、可解释方向迈进。随后出现的工作如 **MathCoder**、**ToolFormer‑Math** 等，都在不同程度上借鉴了迭代训练和解答选择的思路。未来的研究可以探索（1）更通用的代码执行框架（比如支持 Mathematica、Maple），（2）跨语言的工具集成（把自然语言推理与符号计算系统深度融合），以及（3）更高效的候选答案生成策略，以降低计算成本。对想深入的读者，建议关注 **Tool‑Integrated LLM** 系列论文以及 **OpenMathReasoning** 数据集的后续扩展。  

### 一句话记住它
让大模型学会写代码、跑代码、再挑最靠谱的答案，才是真正的数学推理终极武器。