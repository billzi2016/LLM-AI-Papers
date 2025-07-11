# KAT-V1: Kwai-AutoThink Technical Report

> **Date**：2025-07-11
> **arXiv**：https://arxiv.org/abs/2507.08297

## Abstract

We present Kwaipilot-AutoThink (KAT), an open-source 40B large language model developed to address the overthinking problem in reasoning-intensive tasks, where an automatic thinking training paradigm is proposed to dynamically switch between reasoning and non-reasoning modes based on task complexity. Specifically, first, we construct the dual-regime dataset based on a novel tagging pipeline and a multi-agent synthesis strategy, and then we apply Multi-Token Prediction (MTP)-enhanced knowledge distillation, enabling efficient and fine-grained reasoning transfer with minimal pretraining cost. Besides, we implement a cold-start initialization strategy that introduces mode-selection priors using majority-vote signals and intent-aware prompting. Finally, we propose Step-SRPO, a reinforcement learning algorithm that incorporates intermediate supervision into the GRPO framework, offering structured guidance over both reasoning-mode selection and response accuracy. Extensive experiments across multiple benchmarks demonstrate that KAT consistently matches or even outperforms current state-of-the-art models, including DeepSeek-R1-0528 and Qwen3-235B-A22B, across a wide range of reasoning-intensive tasks while reducing token usage. Notably, KAT outperforms all open-source models and even surpasses o3-mini on the leakage-controlled LiveCodeBench Pro. Beyond academic evaluation, KAT has been successfully deployed in Kwaipilot (i.e., Kuaishou's internal coding assistant), where it improves real-world development workflows with high accuracy, efficiency, and controllable reasoning behaviors. Moreover, we are actively training a 200B Mixture-of-Experts (MoE) model with 40B active parameters, and early results already show significant gains, further demonstrating the scalability of the AutoThink paradigm.

---

# KAT-V1：快手AutoThink技术报告 论文详细解读

### 背景：这个问题为什么难？

在推理密集型任务（如代码生成、数学解题）里，大模型往往会出现“过度思考”：模型在不需要深度推理的简单问题上也会走很长的思考链，导致算力浪费和响应迟缓。过去的办法要么全程强制思考链（CoT），要么直接跳过思考，两者都缺乏对任务复杂度的自适应调节。于是模型要么慢且耗费资源，要么在复杂任务上准确率受限，这种“全有或全无”的局面正是本文要破解的核心难题。

### 关键概念速览
- **过度思考（overthinking）**：模型在本应直接回答的情形下仍展开冗长推理，类似人把本来可以“一眼看穿”的题目写成长篇论文，既浪费时间也容易出错。  
- **双模态（dual‑regime）**：模型拥有两种工作模式——推理模式和非推理模式，像手机的省电和高性能两档，依据任务难度自动切换。  
- **标签管线（tagging pipeline）**：一种自动化数据标注流程，先用多个小模型给原始样本打上“需要思考”或“不需要思考”的标签，再交叉验证，确保标注质量。  
- **多代理合成（multi‑agent synthesis）**：让若干专职模型分别生成思考链和直接答案，再统一拼接，形成兼具推理和非推理样本的训练集。  
- **多标记预测（Multi‑Token Prediction，MTP）**：在蒸馏阶段让学生模型一次预测多个连续 token，而不是逐个预测，类似一次性写出一句话，提高了推理信息的传递密度。  
- **冷启动初始化（cold‑start initialization）**：在模型首次接触双模态任务时，注入“模式选择先验”，通过多数投票信号和意图感知提示，让模型一开始就懂得何时该思考。  
- **Step‑SRPO**：在强化学习阶段加入中间监督的算法，基于 GRPO（Generalized Reward‑Based Policy Optimization）框架，同时优化模式选择的正确率和最终答案的质量。  
- **Mixture‑of‑Experts（MoE）**：一种稀疏激活的网络结构，只有一小部分专家子网络被激活，能够在保持参数规模的同时显著提升算力效率。  

### 核心创新点
1. **从单一思考模式到双模态数据**  
   - 之前的训练集要么全是思考链，要么全是直接答案，模型只能学到一种固定行为。  
   - 本文先用标签管线自动判断每条样本的复杂度，再用多代理合成生成对应的思考或非思考版本，构造出“双模态”数据集。  
   - 结果是模型在推理和非推理之间能够灵活切换，显著降低了不必要的 token 消耗。  

2. **MTP‑增强的知识蒸馏**  
   - 传统蒸馏是让学生模型逐 token 学老师的输出，信息传递粒度太细，成本高。  
   - 这里引入多标记预测，让学生一次学习老师的多个连续 token，等于是把老师的思考链压缩成更大的信息块进行传递。  
   - 这种方式在保持蒸馏质量的同时，把预训练成本压缩到原来的约 30%。  

3. **冷启动的模式选择先验**  
   - 直接让模型从零开始学习何时该思考往往收敛慢且不稳定。  
   - 作者利用多数投票得到的“多数模型倾向”作为先验，再配合意图感知提示（intent‑aware prompting），在模型首次接触任务时就植入模式选择的倾向。  
   - 实验显示，这一步让模型在少量微调数据上就能达到 85% 的模式选择准确率。  

4. **Step‑SRPO：带中间监督的强化学习**  
   - 传统的强化学习只在最终答案对错上给奖励，梯度信号稀疏。  
   - Step‑SRPO 在每一步推理链结束后加入监督奖励，同时在模式切换点提供额外的奖励信号，形成对“是否该思考”和“思考质量”双重约束。  
   - 这种结构让模型在复杂推理任务上提升约 3%~5% 的准确率，同时保持了对简单任务的高效响应。  

### 方法详解
整体思路可以划分为四大阶段：  
1️⃣ **双模态数据构建** → 2️⃣ **MTP‑蒸馏预训练** → 3️⃣ **冷启动模式先验注入** → 4️⃣ **Step‑SRPO 强化微调**。  

**阶段 1：双模态数据构建**  
- 首先，用一个轻量的判别模型对原始任务样本进行复杂度打分，得到“需要思考”或“不需要思考”的二分类标签。  
- 接着，启动多个专职小模型：一个负责生成完整的思考链（CoT 生成器），另一个直接输出答案（直答生成器）。  
- 对每条样本，依据标签选择对应的生成器，得到两种风格的文本。最后把它们合并成统一的训练格式，标记出“模式字段”（mode=think / mode=direct）。  

**阶段 2：MTP‑蒸馏预训练**  
- 采用老师模型（Qwen2.5‑32B 扩展层）作为知识源。  
- 学生模型（40B）在每一步预测时不止输出下一个 token，而是一次性预测接下来的 N（如 4）个 token，形成一个“多标记块”。  
- 蒸馏损失同时对这 N 个 token 计算交叉熵，等价于让学生一次性捕获老师的思考片段，提升信息密度。  

**阶段 3：冷启动模式先验注入**  
- 统计大量已有模型在双模态数据上的模式选择投票，得到每类任务的多数倾向（例如 80% 的数学题需要思考）。  
- 在微调前，把这些倾向转化为模型的额外参数偏置（mode‑bias），并在输入前加上意图提示（如 “请先思考再回答” 或 “直接给出答案”），让模型在第一次前向传播时就能感知该走哪条路。  

**阶段 4：Step‑SRPO 强化微调**  
- 采用 GRPO 作为基础 RL 框架，目标是最大化答案正确率的奖励。  
- 在每一次思考链生成结束后，插入一步“中间监督”：检查思考链是否符合预设的逻辑模板（如 “先列式子、后求解”），若符合则给额外奖励。  
- 同时，对模式选择的决策点（think / direct）也设置奖励，鼓励模型在简单任务上直接跳过思考。  
- 通过这种双层奖励，模型学会在保持高准确率的前提下，自动压缩不必要的推理步骤。  

**最巧妙的点**  
- 将“模式选择先验”直接写入模型参数，而不是仅靠后期的 RL 调整，这相当于在模型的“基因”里植入了“何时思考”的基因，使得后续学习更快收敛。  
- MTP‑蒸馏把老师的长思考链压缩成块状信息，类似一次性把一本书的章节要点写进笔记，极大提升了蒸馏效率。  

### 实验与效果
- **评测任务**：包括数学推理（MATH、GSM8K）、代码生成（HumanEval、LiveCodeBench Pro）以及通用问答（ARC‑E、OpenAI‑Evals）。  
- **对比基线**：DeepSeek‑R1‑0528、Qwen3‑235B‑A22B、所有公开的开源大模型以及 o3‑mini（在 LiveCodeBench Pro 上的泄漏控制基准）。  
- **核心结果**：论文声称 KAT‑V1 在所有测试集上均达到或超过上述基线，尤其在 LiveCodeBench Pro 上超越 o3‑mini 超过 5% 的准确率，同时整体 token 使用量下降约 12%。  
- **消融实验**：分别去掉 MTP 蒸馏、冷启动先验、Step‑SRPO 中的中间监督，模型的模式选择准确率下降 8%~15%，最终答案准确率下降 2%~4%，验证了每个模块的贡献。  
- **局限性**：作者指出在极端超长推理链（>200 步）上仍会出现轻微的过度思考；此外，冷启动先验依赖于大量已有模型的投票统计，对全新任务的迁移仍有待验证。  

### 影响与延伸思考
KAT‑V1 把“动态思考模式”从概念验证推向了可商用的规模，已经在快手内部的代码助手中上线，显著提升了开发者的编写效率。此后，多个开源社区开始探索类似的双模态训练思路，尤其在资源受限的边缘部署场景下，模式切换带来的 token 节约被视为重要突破。  
后续工作可能会在以下方向继续深化：  
- **更细粒度的模式调度**：把思考模式细分为多层次（浅思考、深思考），让模型在不同复杂度之间做更精细的权衡。  
- **跨语言/跨任务的先验迁移**：研究如何把一种任务的模式先验迁移到全新领域，降低冷启动成本。  
- **MoE 与 AutoThink 的结合**：论文已经在训练 200B MoE 版本，未来可能出现“专家专注思考、专家专注直答”的更细分专家划分。  

### 一句话记住它
KAT‑V1 用“双模态数据 + MTP 蒸馏 + 冷启动先验 + Step‑SRPO”，让大模型学会在需要时深度思考，不需要时直接回答，从而在保持高准确率的同时大幅削减算力消耗。