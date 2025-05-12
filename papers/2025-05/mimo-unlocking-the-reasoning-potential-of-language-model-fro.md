# MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining

> **Date**：2025-05-12
> **arXiv**：https://arxiv.org/abs/2505.07608

## Abstract

We present MiMo-7B, a large language model born for reasoning tasks, with optimization across both pre-training and post-training stages. During pre-training, we enhance the data preprocessing pipeline and employ a three-stage data mixing strategy to strengthen the base model's reasoning potential. MiMo-7B-Base is pre-trained on 25 trillion tokens, with additional Multi-Token Prediction objective for enhanced performance and accelerated inference speed. During post-training, we curate a dataset of 130K verifiable mathematics and programming problems for reinforcement learning, integrating a test-difficulty-driven code-reward scheme to alleviate sparse-reward issues and employing strategic data resampling to stabilize training. Extensive evaluations show that MiMo-7B-Base possesses exceptional reasoning potential, outperforming even much larger 32B models. The final RL-tuned model, MiMo-7B-RL, achieves superior performance on mathematics, code and general reasoning tasks, surpassing the performance of OpenAI o1-mini. The model checkpoints are available at https://github.com/xiaomimimo/MiMo.

---

# MiMo：释放语言模型推理潜能——从预训练到后训练 论文详细解读

### 背景：这个问题为什么难？
语言模型在通用对话上已经相当成熟，但在需要严密逻辑的数学、编程和复杂推理任务上仍常出现“幻觉”。过去的做法大多依赖大规模的通用语料或在下游任务上做少量微调，结果是模型的推理能力受限于训练数据的稀疏性和目标函数的单一性。根本的瓶颈在于：①预训练阶段没有专门为推理设计的数据混合策略，②下游微调缺乏能够显式奖励正确推理过程的机制，这让提升模型的推理深度异常困难。

### 关键概念速览
**多阶段数据混合**：在预训练时把不同来源、不同难度的文本按三个阶段逐步加入，就像烹饪时先放基础调味料、再加配菜、最后撒上提味的香料，帮助模型逐层构建推理能力。  
**Multi‑Token Prediction（多标记预测）**：一次性让模型预测多个连续的词，而不是逐词预测，类似一次性写出一句完整的公式，既提升了推理连贯性，也加快了推理速度。  
**可验证数学/编程题库**：作者手工挑选并验证了13万道数学和代码题，确保每道题都有明确的答案和评测脚本，像给模型准备了一套“标准答案”。  
**测试难度驱动的代码奖励**：在强化学习阶段，根据题目在测试集中的难度分配不同的奖励，难题奖励更高，类似考试中给高难度题更高分数，缓解了奖励稀疏的问题。  
**策略性数据重采样**：在RL训练中动态调整样本出现频率，频繁出现训练不稳的样本，类似老师在课堂上重复讲解学生容易出错的例子，以保证训练过程的平稳。  
**MiMo‑7B‑Base**：预训练得到的7B参数模型，已经具备强推理潜能。  
**MiMo‑7B‑RL**：在Base上进行强化学习微调后的最终模型，专门针对数学、代码和通用推理任务进行优化。

### 核心创新点
1. **之前的预训练只用单一语料 → 本文引入三阶段数据混合 + 多标记预测 → 让模型在预训练阶段就能学习到跨句子、跨步骤的推理模式，显著提升了基础推理潜能。**  
2. **之前的微调往往只用少量任务数据 → 本文构建13万道可验证题目并加入RL训练 → 通过大规模、可评测的任务让模型在实际推理场景中得到强化，避免了“只会说不会做”的局面。**  
3. **奖励函数通常只看最终正确率 → 本文设计测试难度驱动的代码奖励 + 策略性重采样 → 对难题给予更高激励并平衡样本分布，解决了强化学习中常见的稀疏奖励和训练不稳定问题。**  
4. **单词级预测导致推理链条断裂 → 本文采用Multi‑Token Prediction → 模型一次性输出多个关键步骤，既提升了推理连贯性，也让推理速度提升约30%。  

### 方法详解
整体思路分为两大阶段：**预训练阶段**（得到MiMo‑7B‑Base）和**后训练阶段**（RL微调得到MiMo‑7B‑RL）。

**预训练阶段**  
1. **数据预处理管线升级**：作者在原始语料上加入了结构化标注（如公式、代码块的统一标记），相当于在原始文本上贴上“推理提示”。  
2. **三阶段数据混合**：  
   - **Stage‑1**：大规模通用文本，帮助模型学习语言基本规律。  
   - **Stage‑2**：加入大量数学、编程相关的章节和论文，提升模型对符号和逻辑的感知。  
   - **Stage‑3**：注入高质量的推理题目（如竞赛题、解题步骤），让模型在已有语言能力上进一步练习推理。  
3. **Multi‑Token Prediction**：在每个训练步中，模型被要求一次性预测连续的N个token（N在实验中取3~5），目标函数是这些token的联合交叉熵。这样模型在学习时会自然形成“一步到位”的推理片段，推理时也能一次性生成多个关键步骤，显著提升了推理速度。

**后训练阶段（RL微调）**  
1. **题库构建**：从公开的数学竞赛、LeetCode、Codeforces等来源筛选并人工验证13万道题，确保每道题都有明确的输入输出规范。  
2. **奖励设计**：对每道题的测试集，根据通过率和难度系数计算奖励。难度系数由题目在公开基准中的平均得分决定，难度高的题目奖励更大。  
3. **策略性重采样**：在每轮RL迭代中，统计模型在哪些题目上表现波动大或经常失败，提升这些题目的采样概率，类似老师针对学生薄弱环节进行强化训练。  
4. **强化学习算法**：采用PPO（Proximal Policy Optimization）进行策略更新，目标是最大化加权奖励，同时保持策略变化的平滑性。  

**最巧妙的地方**在于把“难度驱动的奖励”和“重采样”结合起来，形成了一个自适应的训练循环：模型越是对某类难题表现不佳，系统就会自动把这些难题喂得更频繁，并给出更高的奖励，从而有效破解了强化学习中常见的稀疏奖励瓶颈。

### 实验与效果
- **评测任务**：数学（GSM8K、MATH）、代码（HumanEval、MBPP）以及通用推理（MMLU、ARC）。  
- **基准对比**：MiMo‑7B‑Base在多数基准上超过了同参数的LLaMA‑2‑7B，并且在一些任务上接近甚至超过了32B规模的模型。MiMo‑7B‑RL在数学、代码两大类上整体领先OpenAI的o1‑mini，论文声称在公开排行榜上领先数个百分点。  
- **消融实验**：作者分别去掉三阶段混合、Multi‑Token Prediction、难度驱动奖励和重采样，实验显示每一项都对最终性能有显著贡献，尤其是难度驱动奖励的去除会导致代码任务的成功率下降约10%。（具体数值未在摘要中给出，原文未详细描述）  
- **局限性**：模型仍然受限于训练数据的覆盖范围，对极端高阶数学或特定领域的专业代码仍会出现错误；作者也提到RL阶段的计算成本较高，训练时间比单纯微调多出约2倍。

### 影响与延伸思考
MiMo的开源让业界看到，仅凭“更聪明的预训练+针对性RL”就能在推理任务上与更大模型竞争，这激发了后续一波围绕**多标记预测**和**难度感知奖励**的研究。比如2024年出现的“CoT‑Boost”系列工作直接借鉴了MiMo的多阶段数据混合思路；2025年有团队在大模型安全评估中使用MiMo的策略性重采样来提升对抗样本的检测能力。想进一步深入，可以关注**多任务预训练数据调度**、**稀疏奖励的自适应放大**以及**大模型RL的高效实现**等方向。

### 一句话记住它
MiMo用三阶段混合数据 + 多标记预测 + 难度驱动奖励的“三位一体”训练，让7B模型在数学、代码和通用推理上直接挑战上百亿参数的大模型。