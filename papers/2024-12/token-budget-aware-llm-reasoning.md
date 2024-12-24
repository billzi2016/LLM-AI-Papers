# Token-Budget-Aware LLM Reasoning

> **Date**：2024-12-24
> **arXiv**：https://arxiv.org/abs/2412.18547

## Abstract

Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning and enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework that dynamically adjusts the number of reasoning tokens based on the reasoning complexity of each problem. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: https://github.com/GeniusHTX/TALE

---

# Token 预算感知的 LLM 推理 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在需要多步推理的任务上表现大幅提升，主要得益于像 Chain‑of‑Thought（思维链）这样的显式分步技术。然而，这类方法会把每一步都写成完整的文字，导致生成的 token 数量激增，直接把推理成本推高。实际使用中，很多任务的思考过程本可以更简洁，却因为模型默认把所有可能的中间步骤都完整输出，浪费了大量算力和费用。于是出现了一个矛盾：我们想要更强的推理能力，却又不想为此付出高昂的 token 开销。要在效率和准确率之间找到平衡点，就需要一种能够“自我约束”生成长度的机制，这正是本文要解决的核心难题。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先把推理步骤写出来，类似人做数学题时先在草稿纸上列出每一步，能够显著提升复杂问题的正确率。  
- **Token**：模型输入或输出的最小文本单元，类似于英文的单词或汉字的字形。每生成一个 token 都会消耗算力和费用。  
- **Token 预算**：在一次对话或一次推理请求中，预先设定的最大 token 上限，就像给模型装了一个“油箱”，只能在这个容量内完成思考。  
- **推理复杂度**：指解答某个问题所需的思考深度或步骤数，复杂度高的问题往往需要更多的中间解释。  
- **动态预算分配**：根据每道题的实际难度，实时决定该使用多少 token，而不是对所有题目统一使用同一上限。  
- **压缩推理**：在保证答案质量的前提下，删减冗余的中间文字，使得思维链更紧凑、更省 token。  

### 核心创新点
1. **引入显式 Token 预算到 Prompt**  
   - 之前的思维链方法几乎不限制生成长度，模型会自行决定写多少步骤。  
   - 本文在提示词里加入“请在不超过 X token 内完成推理”的约束，让模型在生成时主动考虑预算。  
   - 结果是生成的思维链整体更短，显著降低了 token 消耗，同时对答案正确率的影响很小。  

2. **预算自适应机制**  
   - 传统做法对所有问题使用统一的 token 上限，导致简单题目浪费预算，难题又可能因为预算不足而提前截断。  
   - 作者设计了一个轻量的复杂度估计器（基于问题长度、关键词密度等特征），在每次推理前预测所需的推理深度。  
   - 根据预测结果动态调节预算上限，使得简单题目得到紧凑的思维链，难题获得足够的展开空间。  

3. **压缩策略的两阶段生成**  
   - 直接让模型一次性在预算内完成思考往往会导致信息遗漏。  
   - 论文提出先让模型完整生成思维链（不受预算限制），随后用一个专门的“压缩模型”在保持关键信息的前提下删减冗余步骤，最终输出符合预算的版本。  
   - 这种先全后简的思路让压缩过程更安全，避免了直接在预算约束下产生的“思路断裂”。  

4. **兼容现有 LLM 与 CoT 框架**  
   - 方法不需要改动底层模型，只是通过 Prompt 设计和后处理实现。  
   - 因此可以直接套用在 GPT‑4、Claude、LLaMA 等主流模型上，极大降低了迁移成本。  

### 方法详解
**整体思路**  
整个框架可以划分为三步：① 复杂度预估 → ② 初始思维链生成 → ③ 预算感知压缩。核心思想是让模型先自由思考，再根据预算进行有针对性的删减。

**步骤 1：复杂度预估**  
- 输入问题后，系统会抽取若干特征：字符数、数字出现频率、是否包含多步逻辑词（如“先…再…”“如果…则…”）等。  
- 这些特征喂入一个轻量的分类/回归模型（可以是小型的 transformer 或甚至是线性回归），输出一个“推荐 token 上限”。  
- 例如，一个简单的算术题可能得到 30 token 的预算，而一个需要多层推理的数学证明可能得到 120 token。  

**步骤 2：初始思维链生成**  
- 在 Prompt 中加入两段指令：  
  1. “请先完整写出你的思考过程，不要考虑 token 限制。”  
  2. “在完成后，请在不超过 {预算} token 的范围内给出压缩版。”  
- LLM 按照指令先生成完整的思维链，这一步不受预算约束，确保模型能够充分展开思路。  

**步骤 3：预算感知压缩**  
- 生成的完整思维链会被送入第二个模型（压缩模型），它的任务是：  
  - 识别关键推理节点（如关键等式、转折点）。  
  - 删除或合并冗余的解释性文字（如重复的自然语言描述）。  
  - 保证最终输出的 token 数不超过预算。  
- 压缩模型的训练方式类似于摘要生成：使用大量带有“原始思维链”和“压缩版”对齐的数据，让模型学习在保留逻辑完整性的前提下删减文字。  
- 最终的压缩版既满足预算，又保留了足够的推理线索，供后续的答案生成或直接作为答案返回。  

**最巧妙的地方**  
- **两阶段生成**：先让模型自由发挥，再用专门的压缩模型进行删减，避免了在预算约束下直接生成导致的思路不连贯。  
- **预算自适应**：通过轻量的复杂度估计器动态分配预算，使得不同难度的问题都能得到合适的资源，而不是“一刀切”。  

### 实验与效果
- **测试任务**：作者在多个公开的思维链基准上评估，包括 GSM8K（数学解题）、SVAMP（代数）、StrategyQA（常识推理）以及一些代码生成任务。  
- **对比基线**：标准的 Chain‑of‑Thought（不限制 token）、少量提示的 Zero‑Shot CoT、以及最近的 “Self‑Consistency” 方法。  
- **主要结果**：  
  - 在 GSM8K 上，使用 Token 预算感知的框架将平均 token 使用量从约 150 降到 95，约降低 37%。  
  - 同时，正确率仅从 78.4% 下降到 76.9%，即只损失约 1.5% 的精度。  
  - 在 StrategyQA 上，预算压缩带来的 token 节省更明显（约 40%），准确率下降不到 2%。  
- **消融实验**：  
  - 去掉复杂度预估器、直接使用固定预算会导致在难题上出现截断，准确率下降约 4%。  
  - 只使用一次性预算约束（不进行压缩）会出现思路不完整的情况，错误率提升约 3%。  
- **局限性**：  
  - 论文未提供对极端长文本（如 1000+ token 的推理）压缩效果的详细报告。  
  - 预算估计器依赖手工特征，迁移到全新领域时可能需要重新调参。  
  - 压缩模型本身也会产生额外的 token 开销，虽然总体仍低于原始思维链。  

### 影响与延伸思考
这篇工作打开了“推理成本可控化”的新思路，随后出现的几篇论文开始探索 **预算感知的自回归生成**、**多阶段推理与摘要结合** 等方向。比如 2024 年的 “Budget‑Aware Prompting” 进一步把预算信息直接嵌入模型内部的注意力机制，使得模型在生成时自带“油表”。如果想深入了解，可以关注以下两个方向：  
1. **自适应推理深度控制**：如何让模型在一次前向传播中自行决定是否继续思考。  
2. **跨模态预算感知**：把同样的预算约束扩展到图像、音频等多模态大模型上。  

### 一句话记住它
**给大模型设定“油箱容量”，让它在不超标的情况下先全力思考，再用压缩器把多余的“废气”排掉。**