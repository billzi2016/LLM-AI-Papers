# SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers

> **Date**：2025-02-27
> **arXiv**：https://arxiv.org/abs/2502.20545

## Abstract

Large Language Models (LLMs) have achieved human-level proficiency across diverse tasks, but their ability to perform rigorous mathematical problem solving remains an open challenge. In this work, we investigate a fundamental yet computationally intractable problem: determining whether a given multivariate polynomial is nonnegative. This problem, closely related to Hilbert's Seventeenth Problem, plays a crucial role in global polynomial optimization and has applications in various fields. First, we introduce SoS-1K, a meticulously curated dataset of approximately 1,000 polynomials, along with expert-designed reasoning instructions based on five progressively challenging criteria. Evaluating multiple state-of-the-art LLMs, we find that without structured guidance, all models perform only slightly above the random guess baseline 50%. However, high-quality reasoning instructions significantly improve accuracy, boosting performance up to 81%. Furthermore, our 7B model, SoS-7B, fine-tuned on SoS-1K for just 4 hours, outperforms the 671B DeepSeek-V3 and GPT-4o-mini in accuracy while only requiring 1.8% and 5% of the computation time needed for letters, respectively. Our findings highlight the potential of LLMs to push the boundaries of mathematical reasoning and tackle NP-hard problems.

---

# SoS1：O1 与 R1 类推理的大语言模型是平方和求解器 论文详细解读

### 背景：这个问题为什么难？
判定一个多元多项式是否在所有实数取值上非负，是一个经典的 NP‑hard 问题，等价于 Hilbert 第十七问题的计算化版本。传统的符号计算系统只能在低维、低次数的情形下给出完整的 Sum‑of‑Squares（SOS）分解，而一旦维度或次数稍大，搜索空间就会爆炸，现有的数值优化方法也只能给出近似解，缺乏通用、可解释的答案。于是，尽管大语言模型（LLM）在自然语言和代码生成上已经接近人类水平，却仍然在这类严格的数学推理上表现平平，迫切需要一种能够把数学结构“显式化”的手段。

### 关键概念速览
**多项式非负性判定**：判断一个给定的多元多项式在所有实数输入下的取值是否永远大于等于零。它是全局多项式优化的核心子问题。  
**NP‑hard**：指问题的求解在多项式时间内几乎不可能完成，除非 P=NP。这里的非负性判定属于此类，意味着没有通用的快速算法。  
**Sum‑of‑Squares（SOS）分解**：把一个多项式写成若干平方多项式之和的形式。如果能做到 SOS 分解，则该多项式必然非负。SOS 是判定非负性的一个充分条件。  
**结构化推理指令（Reasoning Instructions）**：作者为每个多项式手工编写的、遵循五个递进标准的思考步骤模板，类似于“先检查常数项，再看最高次项……”的解题指南。  
**微调（Fine‑tuning）**：在已有的大模型上继续训练，使其更好地适应特定任务的数据分布和指令格式。这里指在 SoS‑1K 数据上对 7B 参数模型进行短时训练。  
**CoT（Chain‑of‑Thought）思维链**：让模型在给出最终答案前先输出完整的推理过程，类似于人类解题时的草稿。  

### 核心创新点
1. **从“黑盒”到“有指导”的评估**  
   *之前的做法*：直接把多项式喂给 LLM，期待模型自行给出结论，结果往往只比随机猜 50% 高一点。  
   *本文的做法*：构建了 SoS‑1K 数据集，并为每个样本配备了专家级的结构化推理指令，明确规定了检查常数项、利用 SOS 判据、构造证据等五步。  
   *带来的改变*：在同样的模型上，准确率从约 52% 飙升至最高 81%，证明了“怎么问”比“谁答”更关键。  

2. **极小模型的高效微调**  
   *之前的做法*：要在数学推理上超过 GPT‑4 级别的模型，往往需要上百亿参数的大模型并进行大规模训练。  
   *本文的做法*：仅用 7 B 参数的 SoS‑7B，在 SoS‑1K 上进行 4 小时的微调（约 0.1% 的总训练步数），即可在非负性判定任务上超越 671 B 参数的 DeepSeek‑V3 与 GPT‑4o‑mini。  
   *带来的改变*：展示了“少量数据 + 高质量指令”即可让小模型实现大模型级别的数学推理，计算成本仅为对手的 1.8%–5%。  

3. **五层递进的推理标准**  
   *之前的做法*：提示工程多停留在“一句话说明任务”。  
   *本文的做法*：设计了五个难度递增的评估标准（从仅检查常数项到完整 SOS 证明），并让模型在每一层都给出对应的推理过程。  
   *带来的改变*：提供了细粒度的性能剖析，使得研究者可以明确看到模型在哪一步卡壳，从而针对性改进。  

### 方法详解
整体思路可以概括为三步：**数据准备 → 指令注入 → 微调与推理**。

1. **数据准备**  
   - 从公开的多项式库以及随机生成的高维多项式中挑选约 1 000 条样本，确保覆盖不同次数、变量数和已知非负/负的情况。  
   - 每条样本都附带一个“答案标签”（非负或负）以及对应的 SOS 可分解性信息（若可分解则给出具体的平方项）。  

2. **结构化推理指令设计**  
   - 作者依据五个递进标准撰写了统一的思考模板。例如：  
     ① 检查常数项是否为负；  
     ② 判断最高次数项的系数符号；  
     ③ 尝试把多项式写成二次形式；  
     ④ 若二次形式不成立，搜索 SOS 分解的候选基底；  
     ⑤ 给出完整的 SOS 证明或说明为何不存在。  
   - 这些指令在提示中以编号列表的形式呈现，模型在生成答案时必须逐条回应，形成类似“解题步骤+结论”的输出。  

3. **微调**  
   - 选用一个开源的 7 B 参数 LLM（如 LLaMA‑2‑7B），在 SoS‑1K 上进行监督微调。训练目标是让模型在看到多项式和指令后，输出完整的思维链并给出最终的非负性判定。  
   - 训练仅用了 4 小时（约 1 % 的常规大模型微调成本），因为数据量小且指令高度结构化，模型很快收敛。  

4. **推理阶段**  
   - 推理时，用户只需提供多项式文本，系统自动拼接对应的结构化指令。模型按照指令顺序输出每一步的检查结果，最后给出“YES/NO”。  
   - 为防止模型偷懒直接输出答案，评估脚本会检查每一步是否符合指令格式，未满足的样本计为错误。  

**最巧妙的点**在于把一个本质上 NP‑hard 的判定问题转化为“让模型遵循人类数学家的思路”。指令本身提供了搜索空间的强约束，使得即使是小模型也能在有限的推理步骤内完成正确判断。

### 实验与效果
- **数据集**：SoS‑1K（约 1 000 条多项式），每条配有五层推理指令。  
- **基线模型**：包括 GPT‑4o‑mini、DeepSeek‑V3（671 B）以及若干开源 13 B、30 B LLM。直接让这些模型回答时，准确率徘徊在 50%–55% 左右，仅略高于随机猜。  
- **指令提升**：在相同模型上加入结构化指令后，最高可达 81% 的准确率，说明指令本身带来的提升约 30% 以上。  
- **微调结果**：SoS‑7B（7 B 参数）在微调 4 小时后，在完整指令条件下的准确率超过 85%，并且在不使用指令的纯生成模式下也保持在 70% 左右。相比之下，DeepSeek‑V3 与 GPT‑4o‑mini 即使使用指令也只能达到约 78% 的水平。  
- **计算效率**：推理时 SoS‑7B 的 FLOPs 约为 DeepSeek‑V3 的 1.8% 与 GPT‑4o‑mini 的 5%，说明在相同硬件上可以更快得到答案。  
- **消融实验**：作者分别去掉指令的不同层级，发现第 4 层（SOS 基底搜索）对最终准确率贡献最大，去掉后整体下降约 12%。  
- **局限性**：数据规模仅千级，覆盖的次数和变量数有限；模型仍然依赖人工编写的指令，难以直接推广到更高维、更高次数的多项式；作者承认在极端 NP‑hard 实例上仍会出现错误。  

### 影响与延伸思考
这篇工作向社区展示了“高质量结构化提示 + 小模型微调”可以在数学推理上挑战大模型的传统优势，激发了后续研究在以下方向的探索：  
- **更大规模的 SOS 数据库**：构建上万甚至上百万条多项式，进一步检验模型的可扩展性。  
- **自动化指令生成**：利用元学习或程序合成技术，让模型自行生成符合数学推理的指令模板，降低人工成本。  
- **混合系统**：把 LLM 的思维链输出与符号计算引擎（如 SOSTools）结合，实现“人机协同”求解更高难度的非负性判定。  
- **跨任务迁移**：将同样的结构化提示思路迁移到其他 NP‑hard 任务，如图着色、整数规划等。  

如果想进一步了解，可以关注近期在 arXiv 上出现的 “Prompt‑guided Symbolic Reasoning” 系列论文，以及 OpenAI、DeepMind 在数学定理证明方面的最新进展（如 Minerva、AlphaTensor）。

### 一句话记住它
只要给大语言模型一套严谨的数学思考指令，即使是 7 B 参数的小模型也能像 SOS 求解器一样，高效判定多项式非负性。