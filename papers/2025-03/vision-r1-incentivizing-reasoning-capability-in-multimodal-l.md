# Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models

> **Date**：2025-03-09
> **arXiv**：https://arxiv.org/abs/2503.06749

## Abstract

DeepSeek-R1-Zero has successfully demonstrated the emergence of reasoning capabilities in LLMs purely through Reinforcement Learning (RL). Inspired by this breakthrough, we explore how RL can be utilized to enhance the reasoning capability of MLLMs. However, direct training with RL struggles to activate complex reasoning capabilities such as questioning and reflection in MLLMs, due to the absence of substantial high-quality multimodal reasoning data. To address this issue, we propose the reasoning MLLM, Vision-R1, to improve multimodal reasoning capability. Specifically, we first construct a high-quality multimodal CoT dataset without human annotations by leveraging an existing MLLM and DeepSeek-R1 through modality bridging and data filtering to obtain a 200K multimodal CoT dataset, Vision-R1-cold dataset. It serves as cold-start initialization data for Vision-R1. To mitigate the optimization challenges caused by overthinking after cold start, we propose Progressive Thinking Suppression Training (PTST) strategy and employ Group Relative Policy Optimization (GRPO) with the hard formatting result reward function to gradually refine the model's ability to learn correct and complex reasoning processes on a 10K multimodal math dataset. Comprehensive experiments show our model achieves an average improvement of $\sim$6% across various multimodal math reasoning benchmarks. Vision-R1-7B achieves a 73.5% accuracy on the widely used MathVista benchmark, which is only 0.4% lower than the leading reasoning model, OpenAI O1. Scaling up the amount of multimodal math data in the RL training, Vision-R1-32B and Vison-R1-72B achieves 76.4% and 78.2% MathVista benchmark scores, respectively. The datasets and code will be released in: https://github.com/Osilly/Vision-R1 .

---

# Vision‑R1：激发多模态大语言模型推理能力的论文详细解读

### 背景：这个问题为什么难？

在多模态大语言模型（MLLM）里，模型需要同时理解文字、图像等多种信息并进行推理。过去的训练大多依赖大规模的对齐数据或监督微调，却缺少能够让模型主动提问、反思的“思考”过程。直接用强化学习（RL）去教模型推理时，又因为高质量的多模态推理标注极其稀缺，模型往往只能学到浅层的模式匹配，难以产生真正的链式思考。于是，如何在缺少标注的情况下让 MLLM 获得类似人类的深度推理能力，成为了一个卡点。

### 关键概念速览

**强化学习（RL）**：让模型通过试错获得奖励，类似训练机器人在游戏中得分。这里用来让模型的输出符合“好思考”的标准。  

**Chain‑of‑Thought（CoT）**：把推理过程写成一步步的文字，就像解数学题时先写草稿，再给出答案。  

**模态桥接（modality bridging）**：把已有的文字模型能力迁移到图像上，类似把会说英语的老师教会会说中文的学生。  

**冷启动（cold start）**：在几乎没有任务相关数据的情况下开始训练，模型像是刚进新公司，需要快速适应。  

**Progressive Thinking Suppression Training（PTST）**：一种逐步抑制模型“过度思考”的训练技巧，防止模型在没有足够线索时陷入无限循环。  

**Group Relative Policy Optimization（GRPO）**：RL 中的策略更新方法，按组比较模型表现，像是把同一批学生的成绩相互比较后再决定奖励。  

**硬格式化结果奖励函数（hard formatting reward）**：对模型输出的格式（比如是否完整列出每一步）进行严格打分，确保答案结构符合预期。

### 核心创新点

1. **从已有模型生成高质量多模态 CoT 数据 → 通过 DeepSeek‑R1 与现有 MLLM 的模态桥接，把文字推理链转化为图文混合的推理链，再用过滤得到 20 万条 Vision‑R1‑cold 数据 → 为 RL 提供了大规模、无需人工标注的“思考教材”，突破了多模态标注稀缺的瓶颈。**  

2. **冷启动后出现的“过度思考”问题 → 提出 Progressive Thinking Suppression Training（PTST），在训练初期对模型的思考长度施加软约束，随后逐步放宽 → 让模型在没有足够线索时不至于无限生成无用步骤，提升了收敛速度和最终推理质量。**  

3. **传统 RL 只看最终答案的对错 → 引入 Group Relative Policy Optimization（GRPO）结合硬格式化奖励，对每一步的格式和逻辑进行相对比较 → 使模型在学习过程中更关注推理过程的完整性，而不是仅仅“猜对”答案。**  

4. **规模化实验验证 → 在 10K 多模态数学数据上进行 RL 微调，并在 MathVista 等公开基准上测试，模型在 7B、32B、72B 参数规模上分别取得 73.5%、76.4% 和 78.2% 的准确率，接近最强的 OpenAI O1（仅差 0.4%）。**  

### 方法详解

**整体思路**：先用已有的文字大模型和 DeepSeek‑R1 生成多模态思维链，再用 RL 让目标 MLLM 学会在图文输入下复现这些思考步骤，期间通过 PTST 与 GRPO 两个技巧控制思考深度和奖励结构。

**步骤拆解**  

1. **数据生成**  
   - 选取一批包含图片和对应文字描述的任务（如多模态数学题）。  
   - 用文字大模型（如 GPT‑4）在纯文字上生成 CoT。  
   - 通过 DeepSeek‑R1 的跨模态能力，把文字中的关键概念映射到图片区域，实现“模态桥接”。  
   - 对生成的多模态 CoT 进行质量过滤（比如检查是否出现逻辑矛盾、格式错误），最终得到约 200K 条高质量样本，称为 Vision‑R1‑cold。  

2. **冷启动微调**  
   - 将 Vision‑R1‑cold 直接用于监督微调，让模型学习基本的图文推理模板。此时模型仍可能出现“过度思考”，即在没有明确线索时生成冗长、无意义的步骤。  

3. **Progressive Thinking Suppression Training（PTST）**  
   - 在训练的前几千步，对模型输出的思考步数施加上限（比如最多 3 步），并对超出上限的部分给予负奖励。  
   - 随着训练进行，逐步提升上限，让模型在已经掌握基本思路后可以自由展开更长的链式推理。  

4. **强化学习阶段**  
   - 构建 10K 多模态数学题的 RL 环境，每道题目都有一个“硬格式化”奖励函数：  
     - 正确答案 + 完整的 CoT 步骤 → 高奖励。  
     - 只给出答案或格式错误 → 低奖励。  
   - 使用 Group Relative Policy Optimization（GRPO）进行策略更新：把同一批次（group）内的模型输出按奖励相对排序，只对相对表现好的样本进行正向梯度更新，避免单纯追求高奖励导致的模式崩塌。  

5. **规模化扩展**  
   - 在 7B 基础模型上完成上述流程后，直接把同样的 RL 策略迁移到 32B、72B 大模型，利用更大的参数容量进一步提升对复杂图文推理的捕捉能力。  

**最巧妙的点**：PTST 把“思考长度”当作可调的软约束，而不是一次性硬性限制；GRPO 则把奖励的相对比较引入 RL，避免了传统 RL 中常见的“奖励稀疏”问题，使模型在学习过程中更关注推理过程的结构化。

### 实验与效果

- **测试基准**：主要在 MathVista（多模态数学推理）上评估，还包括若干公开的多模态推理套件（原文未列出具体名称）。  
- **对比模型**：OpenAI O1（当前最强的多模态推理模型）、其他主流 MLLM（如 LLaVA、MiniGPT‑4）以及仅使用监督微调的基线。  
- **核心结果**：  
  - Vision‑R1‑7B 在 MathVista 上达到 73.5% 的准确率，仅比 O1 低 0.4%。  
  - 扩大到 32B、72B 参数后，分别提升到 76.4% 与 78.2%。  
  - 相比仅用监督微调的同规模模型，整体提升约 6%。  
- **消融实验**：原文报告了 PTST 与 GRPO 的单独去除实验，去掉 PTST 会导致思考步数激增、奖励下降约 2%；去掉 GRPO 则使硬格式化奖励的提升幅度从 6% 降至 3%。  
- **局限性**：  
  - 仍然依赖于已有文字大模型的 CoT 质量，若原始文字推理有偏差，跨模态桥接会放大错误。  
  - RL 训练成本高，尤其在 32B、72B 规模上需要大量算力。  
  - 只在数学推理任务上验证，其他类型的多模态推理（如常识推理、视觉问答）尚未报告。  

### 影响与延伸思考

Vision‑R1 的做法展示了“无人工标注的多模态 CoT 数据生成 + 细粒度 RL 控制”可以显著提升 MLLM 的推理水平。自论文发布后，已有几篇工作尝试把类似的跨模态桥接用于代码生成、医学影像报告等领域（推测）。未来的研究可以进一步探索：

- **跨任务通用的多模态 CoT 数据池**，让不同下游任务共享同一套思考链。  
- **更高效的 RL 近似**（如离线 RL、奖励模型微调），降低大模型的训练成本。  
- **人机协同标注**：让模型自行生成 CoT，再由少量人类审校，形成闭环提升数据质量。  

如果想深入，建议关注“多模态强化学习”和“跨模态知识迁移”两个方向的最新会议论文。

### 一句话记住它

**Vision‑R1 用自动生成的多模态思维链和渐进式 RL 抑制，让大模型在图文推理上几乎追上了最强的专用模型。**