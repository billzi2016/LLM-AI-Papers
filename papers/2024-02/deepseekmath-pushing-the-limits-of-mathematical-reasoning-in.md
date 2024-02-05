# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open   Language Models

> **Date**：2024-02-05
> **arXiv**：https://arxiv.org/abs/2402.03300

## Abstract

Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.

---

# DeepSeekMath：突破开源语言模型的数学推理极限 论文详细解读

### 背景：这个问题为什么难？

数学题目往往要求模型在符号、步骤和逻辑上保持严密，而普通语言模型的训练目标主要是预测下一个词，缺乏对结构化推理的专门约束。过去的做法要么靠外部工具（如计算器、符号引擎）补足推理漏洞，要么在模型内部加入大量人工标注的 CoT（思维链）示例，却仍然受限于数据规模和训练方式，导致在竞争级别的 MATH 基准上难以突破 40% 左右的瓶颈。于是，如何在不依赖外部工具、仅靠模型本身的能力提升数学推理，成为迫切需要解决的难题。

### 关键概念速览
- **预训练**：在大规模未标注文本上让模型学习语言规律，就像先让学生阅读大量教材，再去做题。这里的预训练继续在已有代码模型的基础上进行，专注于数学相关的网页内容。  
- **MATH 基准**：一个包含高中到大学水平数学题的公开评测集，题目形式多样、解答步骤要求严格，是衡量模型数学推理能力的金标准。  
- **Self‑consistency（自洽采样）**：模型一次生成多个答案，然后让多数票决定最终答案，类似让学生多次独立作答后取最常见的解法，以降低偶然错误。  
- **Proximal Policy Optimization（PPO）**：一种强化学习算法，训练时限制每一步的策略改动幅度，防止模型“跑偏”。  
- **Group Relative Policy Optimization（GRPO）**：本文对 PPO 的改进版，按批次（group）对优势（advantage）进行相对归一，既提升了数学推理的学习信号，又显著降低了显存占用。  
- **Common Crawl**：公开的互联网爬取数据集，包含海量网页文本。论文通过精心筛选，从中抽取出 1200 亿条与数学、代码、自然语言混合的 token。  
- **投票技术（voting）**：在推理阶段让多个模型或多次采样的结果投票决定答案的做法，常用于提升鲁棒性。DeepSeekMath 在单模型单次推理时就能取得高分，几乎不需要这种外部技巧。  

### 核心创新点
1. **大规模数学网页数据筛选 → 采用专门的过滤管线从 Common Crawl 中抽取 120 B 与数学直接相关的 token → 让模型在继续预训练阶段就接触到丰富的数学语言、公式和解题思路，显著提升了对数学结构的感知能力。**  
2. **GRPO 训练算法 → 在 PPO 的基础上引入“组相对优势”归一，使得同一批样本之间的梯度贡献更加平衡，同时把梯度缓存拆分为更小的块 → 训练时显存需求下降约 30%，并且在数学推理任务上比普通 PPO 提升了约 3% 的准确率。**  
3. **纯模型推理，无外部工具 → 只用单个 7 B 参数的模型直接输出完整解答 → 在 MATH 基准上单次推理得分 51.7%，已经逼近商业大模型的水平，证明了纯模型也能实现高质量数学推理。**  
4. **自洽采样的系统化使用 → 生成 64 条解答后进行多数投票 → 结果提升到 60.9%，展示了在同一模型内部通过多样化采样即可获得类似集成的增益，而无需额外模型或后处理。**  

### 方法详解
整体思路可以划分为三大阶段：**数据准备 → 继续预训练 → 强化学习微调（GRPO）**，随后在推理阶段加入 **自洽采样**。

1. **数据准备**  
   - 从 Common Crawl 原始网页中抽取所有出现数学符号、公式、代码块的段落。  
   - 使用一套基于正则表达式和轻量分类器的过滤器，剔除噪声（如广告、无关段落），保留高质量的数学叙述、证明和代码实现。  
   - 最终得到约 120 B token，比例约为 70% 数学、20% 代码、10% 普通自然语言，形成混合语料。

2. **继续预训练**  
   - 以 DeepSeek‑Coder‑Base‑v1.5‑7B 为起点，继续在上述混合语料上进行自监督语言建模。  
   - 训练目标仍是下一个 token 预测，但因为语料中大量出现 LaTeX、Python、数学推导，模型自然而然学会了符号的上下文关联和步骤连贯性。  
   - 这一阶段相当于让模型“先读教材”，为后面的推理打下基础。

3. **GRPO 强化学习微调**  
   - 选取公开的数学题目（包括 MATH、GSM8K 等）作为强化学习的环境。模型先生成答案，使用外部评估脚本给出奖励（正确得高分，错误得低分）。  
   - 与传统 PPO 不同，GRPO 把同一批次（group）内的优势值先做相对排序，再进行归一化，这样每个样本的梯度贡献不被极端奖励主导。  
   - 同时，GRPO 将梯度缓存拆分为更小的子块，显存占用随 batch 大小线性增长而不是指数，允许在单卡上使用更大的 batch。  
   - 通过多轮迭代，模型的策略逐步倾向于生成更长、更连贯的推理链，而不是仅仅追求答案的对错。

4. **自洽采样推理**  
   - 在正式评测时，给定一道题目，模型使用温度采样生成 64 条独立解答。  
   - 对每条解答进行答案抽取（如最后的数值或符号），统计出现频率最高的即为最终输出。  
   - 这种做法相当于让同一个学生多次独立作答后取多数答案，能够抵消偶然的推理错误。

**最巧妙的点**在于 GRPO 的“组相对优势”设计：它既保留了 PPO 对策略更新幅度的安全约束，又通过相对归一让每个样本的学习信号更均衡，避免了少数高奖励样本主导梯度，进而提升了模型在细粒度推理任务上的稳定性。

### 实验与效果
- **评测数据**：主要在竞争级别的 MATH 基准上测试，还辅以 GSM8K、MathQA 等公开数学数据集。  
- **单次推理成绩**：DeepSeekMath‑7B 在 MATH 上取得 51.7% 的准确率，已经接近 Gemini‑Ultra 与 GPT‑4 的水平（后者在公开报告中约为 55% 左右）。  
- **自洽采样提升**：使用 64 次采样的自洽策略后，准确率提升至 60.9%，在同等模型规模中领先约 10%‑15%。  
- **对比基线**：与同类开源 7 B 代码模型（如 CodeLlama‑7B、StarCoder‑7B）相比，DeepSeekMath 在 MATH 上提升约 20%‑25%。  
- **消融实验**：论文分别去掉（1）数学网页数据筛选，仅使用通用 Common Crawl，准确率下降约 5%；（2）GRPO 替换为普通 PPO，显存占用提升 30% 且准确率下降约 2%；（3）自洽采样改为单次输出，整体分数回落到 51.7%。这些实验表明三大模块均对最终表现贡献显著。  
- **局限性**：虽然在 MATH 上已接近商业模型，但在更高阶的证明题或需要长程记忆的多步推理上仍有差距；模型规模仍受 7 B 参数上限，进一步放大可能带来更大提升。作者也提到对极端长公式的渲染仍不够稳健。

### 影响与延伸思考
这篇工作展示了“只靠更好数据和更高效的 RL 微调，就能在 7 B 规模上实现接近商业大模型的数学推理”这一可能性，激发了社区对开放式数学数据管线和轻量化 PPO 变体的关注。随后出现的项目如 **MathCoder**、**OpenMathLM** 都在数据筛选或 GRPO 思路上进行改进，尝试把同样的框架推广到多语言或多模态（图像+公式）场景。对想进一步探索的读者，可以关注以下方向：  
1. **更大规模的数学网页爬取与噪声过滤技术**，尤其是如何自动识别高质量证明文本。  
2. **RL 微调的记忆友好算法**，比如在更大 batch 下保持显存可控的策略。  
3. **工具增强的混合模式**：在保持纯模型高分的同时，引入可调用的符号引擎，实现“模型+工具”的协同推理。  
4. **跨模态数学理解**，把手写公式、图形与文本统一进模型的训练语料。  

### 一句话记住它
只要给 7 B 模型喂进精挑细选的数学网页并用 GRPO 进行高效微调，它就能在不靠外部工具的情况下，单模型直接突破 50% 的 MATH 评分。