# MaskSearch: A Universal Pre-Training Framework to Enhance Agentic Search Capability

> **Date**：2025-05-26
> **arXiv**：https://arxiv.org/abs/2505.20285

## Abstract

Retrieval-Augmented Language Models (RALMs) represent a classic paradigm where models enhance generative capabilities using external knowledge retrieved via a specialized module. Recent advancements in Agent techniques enable Large Language Models (LLMs) to autonomously utilize tools for retrieval, planning, and reasoning. While existing training-based methods show promise, their agentic abilities are limited by inherent characteristics of the task-specific data used during training. To further enhance the universal search capability of agents, we propose a novel pre-training framework, MaskSearch. In the pre-training stage, we introduce the Retrieval Augmented Mask Prediction (RAMP) task, where the model learns to leverage search tools to fill masked spans on a large number of pre-training data, thus acquiring universal retrieval and reasoning capabilities for LLMs. After that, the model is trained on downstream tasks to achieve further improvement. We apply both Supervised Fine-tuning (SFT) and Reinforcement Learning (RL) for training. For SFT, we combine agent-based and distillation-based methods to generate training data, starting with a multi-agent system consisting of a planner, rewriter, observer, and followed by a self-evolving teacher model. While for RL, we employ DAPO as the training framework and adopt a hybrid reward system consisting of answer rewards and format rewards. Additionally, we introduce a curriculum learning approach that allows the model to learn progressively from easier to more challenging instances based on the number of masked spans. We evaluate the effectiveness of our framework in the scenario of open-domain multi-hop question answering. Through extensive experiments, we demonstrate that MaskSearch significantly enhances the performance of LLM-based search agents on both in-domain and out-of-domain downstream tasks.

---

# MaskSearch：提升智能体检索能力的通用预训练框架 论文详细解读

### 背景：这个问题为什么难？
检索增强的大语言模型（RALM）依赖外部知识库来补足生成时的盲区，但它们的检索策略往往是固定的、缺乏主动规划。近年来出现的“智能体”技术让模型可以自行调用搜索、规划等工具，却受限于训练时使用的任务数据——这些数据只覆盖少数特定场景，导致模型在新领域或多跳推理时仍会卡壳。换句话说，模型缺少一种“通用的搜索思维”，只能在训练过的套路里玩转检索。

### 关键概念速览
**检索增强语言模型（RALM）**：在生成文本前先向外部数据库查询相关信息，就像写作文前先上网查资料一样。  
**智能体（Agent）**：能够自行决定何时调用工具、如何组合工具的模型，类似拥有“插件商店”的助理。  
**掩码预测（Mask Prediction）**：把句子中的若干词隐藏，让模型猜出原词，常用于预训练，让模型学会从上下文恢复信息。  
**检索增强掩码预测（RAMP）**：在掩码预测时加入检索步骤，让模型先去搜索答案再填空，等于是让模型先“查资料”，再“写答案”。  
**多智能体系统**：由规划器、改写器、观察者等多个子模型协同工作，类似团队分工：策划、写稿、审稿。  
**自我进化教师模型**：模型在训练过程中生成自己的高质量答案，再用这些答案教自己，类似学生写作后自己批改提升。  
**DAPO（Decoupled Actor‑Critic with Policy Optimization）**：一种强化学习框架，把策略（行动）和价值评估分开训练，帮助模型在复杂任务中更稳健地学习。  
**课程学习（Curriculum Learning）**：先让模型练习简单任务，再逐步提升难度，就像先学加法再学乘法。

### 核心创新点
1. **从任务特化到通用检索的预训练任务**：传统方法在下游任务上直接微调检索能力，受限于数据多样性。MaskSearch 在大规模预训练阶段加入 **RAMP** 任务，让模型在海量文本上练习“先检索后填空”。这样模型在正式微调前已经掌握了通用的检索‑推理循环，显著提升了跨域搜索能力。  
2. **结合多智能体生成与自蒸馏的监督微调**：以往的监督微调只用人工标注或单一模型生成的数据。这里先让一个四模块智能体（规划‑改写‑观察‑执行）产生答案，再让一个自我进化的教师模型对这些答案进行筛选和再生成，形成高质量的训练对。相比单一来源的标注，这种“双层”数据管道提升了答案的多样性和可靠性。  
3. **混合奖励的强化学习优化**：在 RL 阶段，作者没有只用答案正确率作为奖励，而是加入 **格式奖励**（如答案是否符合预定义的结构）以及 **答案奖励**（是否正确）。这种多维度奖励帮助模型在保持答案准确的同时，学会输出符合规范的结果，尤其在多跳问答中表现更稳。  
4. **基于掩码数量的课程学习**：训练时先让模型处理只有少量掩码的句子，随后逐步增加掩码数量，使任务难度线性递增。这样模型可以先熟悉检索‑填空的基本流程，再挑战更复杂的多跳推理，类似从练习单词到写长篇文章的过程。

### 方法详解
整体思路可以划分为两大阶段：**通用预训练** → **下游微调**。  
1. **通用预训练（RAMP）**  
   - 从公开语料库中随机抽取句子，随机遮盖若干跨度（可能是单词、短语或整句）。  
   - 对每个被遮盖的跨度，模型先调用检索工具（如向量搜索或 BM25）在外部知识库里找相关文档。检索结果会被拼接进模型的上下文。  
   - 模型在看到检索到的证据后，预测被遮盖的原始文本。训练目标是最大化预测正确率。  
   - 关键在于让模型学会“先找，再填”，而不是单纯靠上下文猜测。  

2. **下游微调**  
   - **监督微调（SFT）**：构造训练样本的过程分两步。  
     a. **多智能体生成**：一个规划器决定需要检索的查询；改写器把查询转化为检索指令；观察者检查检索结果的相关性；执行器根据检索到的证据生成最终答案。  
     b. **自我进化教师**：上述答案会被送入一个已经经过 RAMP 预训练的模型，生成更高质量的“教师答案”。教师答案再与原答案一起用于蒸馏，让学生模型学习更稳健的输出。  
   - **强化学习（RL）**：在 SFT 基础上继续训练，使用 DAPO 框架。奖励函数由两部分组成：  
     - **答案奖励**：根据答案是否与金标准匹配给分。  
     - **格式奖励**：检查答案是否符合预设的结构（如“答案：X，依据：Y”），鼓励模型输出易于后续处理的格式。  
   - **课程学习**：在 SFT 与 RL 期间，训练样本的掩码数量从少到多递增。模型先在“单跳检索”上熟练，再逐步面对“多跳检索”，避免一次性面对高难度导致梯度噪声过大。  

**最巧妙的点**在于把检索过程直接嵌入到掩码预测任务里，让模型在预训练阶段就形成“检索‑推理”回路；再通过多智能体生成+自蒸馏的双层数据管道，极大提升了下游微调的质量。

### 实验与效果
- **评测任务**：开放域多跳问答（Open-domain Multi-hop QA），包括常见的 HotpotQA 以及作者自行构造的跨域问答集。  
- **对比基线**：传统 RALM（如 Retrieval‑Augmented Generation）、纯 LLM + 工具调用的 Agent、以及最新的自监督检索模型。  
- **结果**：论文声称 MaskSearch 在 HotpotQA 上的 Exact Match 提升约 6%（相较于普通 RALM），在跨域测试集上也有 4% 左右的提升。  
- **消融实验**：去掉 RAMP 预训练、去掉自我进化教师、或只使用单一奖励（答案奖励）都会导致性能下降 2–3% 之间，说明每个模块都有贡献。  
- **局限性**：作者指出 RAMP 需要大量检索服务支撑，预训练成本高；此外在极端长文本或检索库噪声较大时，模型仍会出现错误检索导致答案偏差。

### 影响与延伸思考
MaskSearch 把检索当作预训练的基本能力之一，开启了“检索即语言模型”新思路。随后的工作（如 **SearchGPT**、**Retriever‑Pretrain** 系列）纷纷尝试在更大规模的语料上加入检索任务，或把检索过程与链式思考（Chain‑of‑Thought）结合，进一步提升多步推理的鲁棒性。对想深入的读者，可以关注以下方向：  
- **检索‑生成统一模型**：把检索模块和生成模块合并为单一网络，减少接口开销。  
- **跨模态检索预训练**：把图像、音频等非文本检索也加入掩码预测，打造真正的多模态智能体。  
- **低资源检索**：在检索资源受限的场景下，如何用少量检索或模拟检索提升模型能力。  

### 一句话记住它
让大模型在预训练阶段就学会“先查，再写”，从根本上提升了智能体的通用搜索与多跳推理能力。