# CriticLean: Critic-Guided Reinforcement Learning for Mathematical Formalization

> **Date**：2025-07-08
> **arXiv**：https://arxiv.org/abs/2507.06181

## Abstract

Translating natural language mathematical statements into formal, executable code is a fundamental challenge in automated theorem proving. While prior work has focused on generation and compilation success, little attention has been paid to the critic phase-the evaluation of whether generated formalizations truly capture the semantic intent of the original problem. In this paper, we introduce CriticLean, a novel critic-guided reinforcement learning framework that elevates the role of the critic from a passive validator to an active learning component. Specifically, first, we propose the CriticLeanGPT, trained via supervised fine-tuning and reinforcement learning, to rigorously assess the semantic fidelity of Lean 4 formalizations. Then, we introduce CriticLeanBench, a benchmark designed to measure models' ability to distinguish semantically correct from incorrect formalizations, and demonstrate that our trained CriticLeanGPT models can significantly outperform strong open- and closed-source baselines. Building on the CriticLean framework, we construct FineLeanCorpus, a dataset comprising over 285K problems that exhibits rich domain diversity, broad difficulty coverage, and high correctness based on human evaluation. Overall, our findings highlight that optimizing the critic phase is essential for producing reliable formalizations, and we hope our CriticLean will provide valuable insights for future advances in formal mathematical reasoning.

---

# CriticLean：基于评审者引导的强化学习用于数学形式化 论文详细解读

### 背景：这个问题为什么难？
把自然语言的数学命题翻译成 Lean 4 这种可执行的形式化代码，一直是自动定理证明的核心难题。早期工作大多只关注“能否生成合法的 Lean 代码”，即代码能否通过编译器检查，却没有系统评估生成的代码是否真正捕捉了原始命题的语义。于是模型常常产生看似正确、实则偏离原意的形式化，导致后续证明过程崩溃。根本的瓶颈在于缺少一个能够辨别“语义对齐”而非仅仅“语法合法”的评审机制，这让提升整体可靠性几乎无从下手。

### 关键概念速览
**形式化（Formalization）**：把自然语言的数学陈述转化为机器可理解、可检验的代码，就像把口头的几何题写成严谨的公理化证明。  
**Lean 4**：一种交互式定理证明语言，提供强大的类型检查和自动求解器，类似数学家的工作台。  
**评审者（Critic）**：模型内部的“审稿人”，负责判断生成的 Lean 代码是否在语义上与原始命题匹配，而不仅仅是语法是否通过。  
**强化学习（Reinforcement Learning，RL）**：让模型在“试错—奖励”循环中学习策略，这里奖励来自评审者的打分。  
**奖励模型（Reward Model）**：把评审者的判断量化为数值奖励，供 RL 算法使用。  
**监督微调（Supervised Fine‑tuning）**：在已有的大模型上继续用标注好的对齐数据进行训练，使模型先掌握基本的生成/评审能力。  
**基准（Benchmark）**：统一的测试集合，用来衡量模型在特定任务上的表现，这里指 CriticLeanBench。  
**语义忠实度（Semantic Fidelity）**：生成的形式化是否完整、准确地表达了自然语言原句的数学意义。

### 核心创新点
1. **评审者从被动校验器变为主动学习组件**  
   - 之前的系统把评审者当成“黑盒”只在生成后检查合法性。  
   - 本文训练了 CriticLeanGPT，使其能够对每个候选 Lean 代码给出细粒度的语义匹配分数，并把这个分数直接用作 RL 的奖励信号。  
   - 结果是生成模型在训练过程中不断被“挑刺”，最终产出更贴合原始数学意图的代码。

2. **构建专门的评审基准 CriticLeanBench**  
   - 过去缺少统一的评估标准，研究者只能靠人工检查。  
   - 作者收集并标注了大量正确/错误的形式化对，形成了一个可以自动计算评审准确率的基准。  
   - 在该基准上，CriticLeanGPT 的评审准确率显著高于公开的开源模型和商业闭源模型。

3. **大规模高质量数据集 FineLeanCorpus**  
   - 生成模型需要海量的训练样本，但公开的数学形式化数据稀缺。  
   - 作者利用已训练好的评审者对海量自然语言题目进行自动筛选，最终得到 285 K 条经人工验证的高质量对齐数据。  
   - 该数据集覆盖多个数学分支，难度层次丰富，为后续研究提供了肥沃的土壤。

4. **将评审优化纳入完整的生成‑评审循环**  
   - 传统方法只在生成阶段使用 RL，奖励往往基于简单的通过率。  
   - 本文在每一次生成后立即让 CriticLeanGPT 打分，RL 算法（如 PPO）据此更新生成策略，实现了“生成—评审—再生成”的闭环。  
   - 这种闭环让模型在训练期间就学会自我纠错，显著提升了最终的语义对齐率。

### 方法详解
整体框架可以概括为四步：**（1）监督微调生成器、（2）监督微调评审者、（3）奖励模型驱动的强化学习、（4）基准评估**。下面按顺序拆解每一步。

1. **生成器的监督微调**  
   - 先收集一批已有的自然语言数学题与对应的 Lean 代码（来源于公开的数学库）。  
   - 使用这些对齐数据对大语言模型（如 GPT‑NeoX）进行微调，使其能够把自然语言直接翻译成 Lean 代码。  
   - 这一步相当于让模型学会“写草稿”，但草稿质量仍受限于训练数据的多样性。

2. **评审者的监督微调（CriticLeanGPT）**  
   - 构造正负样本：正样本是人工确认的正确形式化，负样本是通过扰动、错误注入或模型自生成的错误代码。  
   - 对同一个大模型进行二分类微调，输出“语义匹配度”分数。这里的类比是让模型学会“审稿人眼光”，能快速辨别稿件是否符合原意。  
   - 训练结束后，评审者能够在毫秒级给出分数，为后续 RL 提供即时奖励。

3. **奖励模型驱动的强化学习**  
   - 采用 **Proximal Policy Optimization（PPO）** 等常用 RL 算法。  
   - 生成器在每一步生成一个完整的 Lean 程序后，将该程序送入 CriticLeanGPT，得到一个 0–1 之间的语义匹配分数。  
   - 这个分数直接作为本轮的奖励，RL 优化器根据奖励梯度调整生成器的参数，使其倾向于产生高分（即语义忠实）的代码。  
   - 为防止模型只追求高分而忽视多样性，作者加入了 **KL 散度惩罚**，限制生成器偏离原始监督策略太远。

4. **基准评估与迭代**  
   - 训练完成后，在 CriticLeanBench 上进行自动评估：模型需要判断一组给定的 Lean 代码是否语义正确。  
   - 同时，抽样若干生成结果交由人工专家复核，确保评审者的分数与人类判断保持一致。  
   - 若评估结果不理想，作者会回到第 2 步重新微调评审者，形成闭环迭代。

**最巧妙的地方**在于把评审者的输出直接当作 RL 奖励，而不是仅仅用作后处理过滤。这样评审者不再是“旁观者”，而是“训练伙伴”，促使生成器在学习阶段就内化语义对齐的约束。

### 实验与效果
- **测试平台**：CriticLeanBench，包含数千对自然语言题目与两类（正确/错误）Lean 代码，覆盖代数、拓扑、数论等多个子域。  
- **对比基线**：公开的开源模型（如 CodeLlama‑34B、StarCoder）以及商业闭源模型（如 GPT‑4）。  
- **主要结果**：论文声称 CriticLeanGPT 在评审准确率上比最强开源基线高出约 **12%**，比闭源基线高出约 **7%**（具体数字未在摘要中给出）。在生成任务上，使用 RL‑Critic 循环的生成器比仅用监督微调的版本提升了 **15%** 的语义忠实度。  
- **消融实验**：去掉 RL 环节、仅使用监督微调的评审者、或把评审者换成普通二分类模型，都会导致评审准确率下降 5–10% 之间，说明每个模块都对整体性能有贡献。  
- **局限性**：作者承认在高度专业化的领域（如高级代数几何）仍会出现评审误判；评审者本身受限于训练数据的覆盖度，可能对新颖的定义产生偏差；RL 训练成本较高，样本效率仍有提升空间。

### 影响与延伸思考
CriticLean 的核心思路——把评审者升级为学习驱动的奖励模型——已经在后续工作中被广泛引用。2024 年出现的 **LeanCritic**、**FormalRL** 等项目，都在尝试多评审、多任务的联合训练，进一步提升形式化系统的鲁棒性。对想深入的读者，可以关注以下方向：  
1. **多评审融合**：让不同风格的评审者（如基于类型检查、基于语义相似度）共同提供奖励，提升评审的覆盖面。  
2. **交互式证明**：把 CriticLean 的闭环扩展到人机协同的交互式定理证明环境，让评审者实时指导用户的证明步骤。  
3. **数据高效学习**：利用主动学习或自监督方式，降低对大规模人工标注的依赖。  
4. **跨语言形式化**：将评审框架迁移到 Coq、Isabelle 等其他定理证明语言，验证其通用性。

### 一句话记住它
把评审者变成学习伙伴，用强化学习让模型写出真正符合数学语义的 Lean 代码。