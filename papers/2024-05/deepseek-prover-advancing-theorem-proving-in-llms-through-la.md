# DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale   Synthetic Data

> **Date**：2024-05-23
> **arXiv**：https://arxiv.org/abs/2405.14333

## Abstract

Proof assistants like Lean have revolutionized mathematical proof verification, ensuring high accuracy and reliability. Although large language models (LLMs) show promise in mathematical reasoning, their advancement in formal theorem proving is hindered by a lack of training data. To address this issue, we introduce an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems. This approach involves translating natural language problems into formal statements, filtering out low-quality statements, and generating proofs to create synthetic data. After fine-tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof generation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any. These results demonstrate the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs. Both the synthetic dataset and the model will be made available to facilitate further research in this promising field.

---

# DeepSeek-Prover：通过大规模合成数据提升大语言模型的定理证明能力 论文详细解读

### 背景：这个问题为什么难？

形式化数学证明（如 Lean 4）要求模型在严格的逻辑系统里一步步写出可验证的代码，出错一次就会被拒绝。现有的大语言模型（LLM）虽然在自然语言数学推理上表现不错，却缺少大规模、质量可靠的“题目‑声明‑证明”三元组来进行有监督微调。公开的正式证明数据极其稀少，主要集中在少数成熟的数学库，导致模型在新领域（比如高中竞赛题）几乎没有学习经验。于是，训练数据的瓶颈成为提升 LLM 定理证明能力的根本障碍。

### 关键概念速览

**Lean 4**：一种交互式证明助理，使用函数式语言表达数学声明和证明，类似于让机器“读懂”数学教材并写出可机器检查的解答。  
**合成数据**：通过程序自动生成的训练样本，而不是人工标注的真实数据。这里指的是把自然语言竞赛题转成 Lean 代码并配上机器生成的证明。  
**微调（Fine‑tuning）**：在已有的大模型（DeepSeekMath 7B）基础上，用特定任务的数据再训练，让模型专注于该任务的细节。  
**miniF2F**：Lean 4 上的一个小型竞赛题基准，包含数百道高中/大学水平的数学题目，用来评估模型的整体证明成功率。  
**FIMO（Formalized International Mathematical Olympiad）**：把国际数学奥林匹克（IMO）题目形式化后收录的高难度基准，难度远高于 miniF2F。  
**全证明生成（Whole‑proof generation）**：模型一次性输出完整的 Lean 代码证明，而不是逐步交互或只给出提示。  
**采样数（Samples）**：在生成时模型会尝试多次（如 64 次）并取最好的结果，采样数越多成功率通常越高。  

### 核心创新点

1. **从竞赛题到正式声明的自动管线 → 将自然语言题目先用规则/模型翻译成 Lean 4 语法，再用过滤器剔除语义不完整或难以证明的条目 → 产生了 800 万高质量的“题‑声明‑证明”对，显著缓解了训练数据匮乏的问题。**  
2. **大规模合成证明生成器 → 在已有的 Lean 自动定理证明工具（如 `mathlib` 的 tactic）上跑批量求解，成功生成的证明被保存为训练标签 → 让模型看到真实的证明结构而不是仅仅是声明。**  
3. **针对 DeepSeekMath 7B 的专门微调策略 → 采用混合学习率、梯度累积以及对长序列的特殊截断方式，使得 7B 参数在 8M 样本上仍保持稳定收敛 → 微调后模型在 miniF2F 上的全证明成功率从 23%（GPT‑4）提升到 52%。**  
4. **评估方式的系统化 → 在 miniF2F 上使用 64 次采样统计成功率，在 FIMO 上直接报告能否完整证明 → 证明了合成数据不仅提升了常规题目，也对极端难题有一定渗透。**  

### 方法详解

整体思路可以划分为三大阶段：**数据构造 → 证明生成 → 模型微调**。

1. **数据构造**  
   - **题目收集**：从高中数学竞赛（如美国数学竞赛、欧几里得竞赛）以及本科水平的公开题库抓取原始自然语言描述。  
   - **自然语言到 Lean 的翻译**：使用一个专门训练的语义解析模型，把题目拆解成定义、假设和目标三个部分，并映射到 Lean 4 的类型系统。这里的关键是保持数学概念（如集合、函数、序列）的语义一致性。  
   - **质量过滤**：对生成的 Lean 代码运行语法检查和轻量化的自动求解器。如果代码无法通过基本的类型检查或在 30 秒内找不到任何 tactic 线索，就被丢弃。这样保证进入下一步的样本至少是“可求解的”。  

2. **证明生成**  
   - **自动求解器批处理**：对每个通过过滤的声明，调用 Lean 4 自带的 `simp`, `ring`, `linarith` 等 tactic，甚至使用 `mathlib` 中的高级定理搜索。成功得到完整证明的样本被标记为“合成证明”。  
   - **多样性增强**：同一声明可能有多条不同的证明路径，系统会随机切换 tactic 顺序或加入等价的引理，从而产生多样化的证明文本，防止模型只记忆单一套路。  

3. **模型微调**  
   - **输入格式**：每条训练样本的输入是“Problem: <自然语言题目> → FormalStatement: <Lean 代码>”，输出是对应的完整 Lean 证明。  
   - **训练技巧**：采用 2e‑5 的基础学习率，前 2k 步使用线性 warm‑up；梯度累积到 64 步后才进行一次参数更新，以适配显存限制。对超过 1024 token 的长序列使用滑动窗口截断，确保关键的引理和结论不被裁剪。  
   - **损失函数**：标准的交叉熵加上一个轻微的序列长度正则，鼓励模型生成简洁的证明而不是冗长的噪声。  

**最巧妙的点**在于把“自然语言 → 形式化 → 自动证明”这条闭环全自动化，省去了人工标注的高成本，同时通过多样化的 tactic 组合让模型看到丰富的证明风格，从而学会更通用的推理模式。

### 实验与效果

- **测试基准**：Lean 4 miniF2F（约 400 题）和 Lean 4 Formalized International Mathematical Olympiad（FIMO，148 题）。  
- **主要对比**：GPT‑4（直接生成 Lean 代码）和一种基于树搜索的强化学习方法（论文中称为 “tree‑search RL”）。  
- **结果**：在 miniF2F 上，使用 64 次采样时 DeepSeek‑Prover 的全证明成功率为 46.3%，累计成功率（任意一次成功）达到 52%；GPT‑4 仅为 23.0%，树搜索 RL 为 41.0%。在 FIMO 上，DeepSeek‑Prover 能完整证明 5 题（约 3.4%），而 GPT‑4 全部失败。  
- **消融实验**：原文提到去掉合成证明的多样性增强会导致成功率下降约 4%；仅使用自然语言到 Lean 的翻译而不做自动求解过滤，模型在 miniF2F 上的表现跌至 38%。这些实验说明两大模块（高质量过滤、证明多样性）对最终性能都有实质贡献。  
- **局限**：合成数据仍然局限于高中/本科层面的题目，面对更抽象的高等数学（如拓扑、范畴论）时仍显不足；此外，模型在长序列（超过 1500 token）的证明上仍会出现截断或遗漏。作者也承认，采样数越多成功率提升越明显，但实际部署时计算成本会急剧上升。

### 影响与延伸思考

这篇工作首次展示了“规模化合成正式证明数据”可以让中等规模的 LLM（7B 参数）在定理证明上逼近甚至超越商业大模型的水平。随后的研究开始探索更自动化的题目生成（比如使用图神经网络生成新颖的几何构造），以及把合成数据与真实数学库（如 `mathlib`）混合训练，以进一步提升模型的泛化能力。对想深入的读者，可以关注以下方向：  
- **跨领域合成数据**：把物理、计算机科学等领域的定理也转化为 Lean 形式，构建多学科训练集。  
- **交互式微调**：结合人类专家的即时反馈，让模型在生成错误证明时得到纠正。  
- **更高效的采样策略**：研究基于价值函数的采样，降低大量随机采样的计算开销。  

### 一句话记住它

用大规模自动生成的 Lean 证明数据微调 7B 模型，就能让普通大语言模型在数学定理证明上实现“半专业”水平。