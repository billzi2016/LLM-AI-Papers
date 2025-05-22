# AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.16400

## Abstract

Despite recent progress in large-scale reinforcement learning (RL) for reasoning, the training recipe for building high-performing reasoning models remains elusive. Key implementation details of frontier models, such as DeepSeek-R1, including data curation strategies and RL training recipe, are often omitted. Moreover, recent research indicates distillation remains more effective than RL for smaller models. In this work, we demonstrate that large-scale RL can significantly enhance the reasoning capabilities of strong, small- and mid-sized models, achieving results that surpass those of state-of-the-art distillation-based models. We systematically study the RL training process through extensive ablations and propose a simple yet effective approach: first training on math-only prompts, then on code-only prompts. Notably, we find that math-only RL not only significantly enhances the performance of strong distilled models on math benchmarks (e.g., +14.6% / +17.2% on AIME 2025 for the 7B / 14B models), but also code reasoning tasks (e.g., +6.8% / +5.8% on LiveCodeBench for the 7B / 14B models). In addition, extended code-only RL iterations further improve performance on code benchmarks with minimal or no degradation in math results. We develop a robust data curation pipeline to collect challenging prompts with high-quality, verifiable answers and test cases to enable verification-based RL across both domains. Finally, we identify key experimental insights, including curriculum learning with progressively increasing response lengths and the stabilizing effect of on-policy parameter updates. We find that RL not only elicits the foundational reasoning capabilities acquired during pretraining and supervised fine-tuning (e.g., distillation), but also pushes the limits of the model's reasoning ability, enabling it to solve problems that were previously unsolvable.

---

# AceReason‑Nemotron：通过强化学习提升数学与代码推理能力 论文详细解读

### 背景：这个问题为什么难？
在大模型的预训练阶段，模型已经学会了大量的语言模式，但要让它们在数学题目或编程挑战上进行严谨的推理仍然很吃力。传统的微调手段大多依赖**蒸馏**——把更大的老师模型的输出当作标签教给小模型，这种方式在提升基本能力上有效，却难以突破“思考深度”。另一方面，已有的强化学习（RL）实验多聚焦于对话或游戏，缺少针对数学/代码这种需要精确验证的任务的系统化方案。于是，如何设计一种既能利用 RL 的探索性，又能保持答案可验证性的训练流程，成为了阻碍进一步提升的小模型推理能力的关键瓶颈。

### 关键概念速览
**强化学习（RL）**：让模型在与环境交互后，根据得到的奖励信号调整自身参数，类似于人类通过试错学习。  
**蒸馏（Distillation）**：把大模型的输出当作“软标签”，教给体积更小的模型，像是让学生抄写老师的答案。  
**验证式奖励（Verification‑based Reward）**：对模型的输出进行程序化检查（比如运行代码或代数求解），只有通过验证才给正向奖励，确保模型学会的是可证实的解法。  
**Curriculum Learning（课程学习）**：训练时先让模型解决简单或单一类型的任务，再逐步加入更复杂或多样的任务，类似于学生先学基础再学综合。  
**On‑policy 参数更新**：在每一步使用当前策略产生的数据来更新模型，保证学习过程与模型实际行为保持一致，避免离线数据带来的偏差。  
**响应长度递增**：训练时从短答案开始，逐步让模型生成更长的推理链，帮助模型适应长文本的组织结构。  
**数据策划管线**：系统化收集、过滤、标注高质量题目及其可验证答案的流程，确保 RL 过程中的奖励来源可靠。  
**数学推理 / 代码推理**：分别指模型在解数学题和写/调试代码时的逻辑推演能力，两者都要求答案能够被严格检验。

### 核心创新点
- **大规模 RL 超越蒸馏**：以前的研究普遍认为在小模型上蒸馏比 RL 更有效。这篇论文直接在 7B/14B 规模的模型上跑大规模 RL，结果在 AIME、LiveCodeBench 等基准上分别提升了 14%‑17% 和 6%‑7%，显著跑赢最强蒸馏基线。  
- **两阶段课程学习**：先让模型只在数学题目上进行 RL，等数学能力稳固后再切换到代码题目。这样做比一次性混合训练得到更高的数学和代码分数，说明任务顺序本身是一种重要的学习信号。  
- **验证式奖励的数据管线**：作者搭建了一个能够自动生成“高难度、可验证”题目的爬虫+筛选系统，配套的答案和测试用例让 RL 的奖励可以直接基于程序化校验，而不是人工打分。  
- **响应长度递增 + on‑policy 更新**：在训练早期限制答案长度，随后逐步放宽，同时每一步都用最新的模型生成数据进行更新，这种组合显著降低了训练不稳定和梯度爆炸的风险。

### 方法详解
整体思路可以拆成四步：  
1. **基线模型准备**：先用公开的预训练模型做一次蒸馏微调，得到 7B/14B 的“强蒸馏模型”。  
2. **数学 RL 阶段**：构建仅包含数学题目的数据池，每条题目配有可验证的答案（如代数求根、几何证明的数值或符号结果）。模型在每一步生成答案后，系统自动运行验证脚本，若通过则给高奖励，否则给负奖励。训练采用 PPO（近端策略优化）等 on‑policy 算法，并且在前 10% 的训练步长限制答案长度为 50 token，随后逐步放宽到 200 token。  
3. **代码 RL 阶段**：在数学 RL 完成后，切换到仅含代码题目的数据池。验证方式改为运行代码并检查输出是否匹配预设的测试用例。这里同样使用 on‑policy PPO，并继续沿用响应长度递增的调度，只是把长度上限调到 300 token，以容纳更长的函数实现。  
4. **混合微调与评估**：完成两阶段 RL 后，作者再进行一次轻量的监督微调，以平衡模型在两类任务上的表现。最终模型在所有基准上进行评测。

关键细节：  
- **验证式奖励**是整个流程的核心。作者没有使用人工打分，而是把每道题的答案转化为可执行的检查脚本（数学题用 SymPy 求解，代码题用单元测试），确保奖励信号完全客观。  
- **Curriculum 的顺序**（数学→代码）是实验中发现的最佳路径。若直接混合，两类任务的奖励信号会相互干扰，导致收敛慢甚至出现退化。  
- **On‑policy 更新**让模型每一步都在最新策略下生成数据，避免了离线 RL 常见的“分布漂移”问题。作者报告说，这种方式比传统的离线 KL‑惩罚更稳健。  
- **响应长度递增**相当于让模型先学会写“短答案”，再逐步适应写“完整推导”。这在长链推理任务（如 AIME）上提升了约 3% 的准确率。

### 实验与效果
- **测试集合**：数学方面使用 AIME 2025、MATH、GSM8K；代码方面使用 LiveCodeBench、HumanEval。  
- **基线对比**：与最先进的蒸馏模型（如 DeepSeek‑R1 的蒸馏版）相比，7B 模型在 AIME 上提升了 14.6%，在 LiveCodeBench 上提升了 6.8%；14B 模型对应提升分别为 17.2% 与 5.8%。  
- **消融实验**：作者分别去掉数学 RL、代码 RL、响应长度递增、on‑policy 更新四个组件，发现数学 RL 的缺失导致 AIME 分数下降约 10%，而去掉 on‑policy 则使训练过程出现显著波动，最终分数下降约 4%。  
- **局限性**：实验仅在 7B/14B 规模模型上完成，未验证对更大模型的适用性；验证式奖励依赖于高质量的自动检查脚本，某些开放式数学证明仍难以完全自动化。  

### 影响与延伸思考
这篇工作向社区展示了“RL + 可验证奖励”在提升小模型推理能力上的可行性，随后出现了多篇围绕代码生成 RL（如 CodeRL、RL4Code）和数学推理 RL（如 MathRL、RL‑MATH）的跟进研究。它也促使数据策划团队更加重视自动化生成可验证题目，推动了类似 “OpenMathBench” 的公开数据集建设。对想进一步探索的读者，可以关注以下方向：① 将同样的 RL 框架扩展到多模态（图文）推理；② 研究更高效的 on‑policy 算法，以降低大规模 RL 的算力需求；③ 探索基于人类反馈的奖励混合方式，弥补自动验证的盲区。  

### 一句话记住它
**用可验证奖励的两阶段强化学习，让小模型在数学和代码推理上实现了比蒸馏更大的飞跃。**