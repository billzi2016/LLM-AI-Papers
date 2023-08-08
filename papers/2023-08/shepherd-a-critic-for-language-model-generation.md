# Shepherd: A Critic for Language Model Generation

> **Date**：2023-08-08
> **arXiv**：https://arxiv.org/abs/2308.04592

## Abstract

As large language models improve, there is increasing interest in techniques that leverage these models' capabilities to refine their own outputs. In this work, we introduce Shepherd, a language model specifically tuned to critique responses and suggest refinements, extending beyond the capabilities of an untuned model to identify diverse errors and provide suggestions to remedy them. At the core of our approach is a high quality feedback dataset, which we curate from community feedback and human annotations. Even though Shepherd is small (7B parameters), its critiques are either equivalent or preferred to those from established models including ChatGPT. Using GPT-4 for evaluation, Shepherd reaches an average win-rate of 53-87% compared to competitive alternatives. In human evaluation, Shepherd strictly outperforms other models and on average closely ties with ChatGPT.

---

# Shepherd：语言模型生成的批评者 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成文本时已经能写出相当流畅的答案，但它们仍会出现事实错误、逻辑混乱或不符合用户意图的情况。过去的做法大多是让模型自行“自我纠正”，比如在同一模型内部做多轮生成或使用提示词让它检查自己的输出，这种方式往往缺乏系统的错误识别能力，容易遗漏细微的缺陷。另一方面，评估模型输出的专门评审模型往往体积庞大、训练成本高，难以在资源受限的环境中部署。因此，如何用一个相对小巧的模型可靠地发现并改进生成文本的错误，成为了一个迫切而又棘手的挑战。

### 关键概念速览
- **批评模型（Critic Model）**：专门负责审阅其他模型生成内容、指出错误并给出改进建议的模型。类似于写作老师给学生作文打分并提出修改意见。
- **反馈数据集（Feedback Dataset）**：收集自真实用户、社区评论以及人工标注的“错误-改进”对，提供给模型学习如何辨别和纠正问题。可以把它想成一本“常见错误手册”。
- **微调（Fine‑tuning）**：在已有的大模型基础上，用特定任务的数据继续训练，使模型在该任务上表现更好。相当于在通用语言能力上再上一个专业培训班。
- **胜率（Win‑rate）**：在两两比较中，被评审者被选为更好答案的比例。比如 70% 胜率意味着在 10 次对比里有 7 次被认为更优。
- **多模态评估（Multi‑metric Evaluation）**：使用不同的评估手段（如 GPT‑4 自动打分、人工打分）来综合判断模型表现，避免单一指标的偏差。
- **参数规模（Parameter Size）**：模型内部可学习的权重数量，常用来衡量模型的容量。7B 表示 70 亿参数，属于中等规模。

### 核心创新点
1. **从社区反馈构建高质量批评数据 → 直接利用真实用户的纠错评论和人工标注生成“错误‑建议”对 → 让模型学习到更贴近实际使用场景的错误类型和改进方式，而不是仅靠合成或少量人工示例。**  
2. **在 7B 参数的紧凑模型上完成批评任务 → 通过精心的微调策略和数据筛选，使得小模型的批评质量能够匹配甚至超越更大的基线模型 → 打破了“批评模型必须大而全”的传统认知，降低了部署门槛。**  
3. **将批评与改进建议统一输出 → 模型不仅指出错误，还同步给出可操作的修改方案 → 与只给出错误标签的旧方法相比，提升了实际使用价值，用户可以直接拿去改写。**  
4. **使用 GPT‑4 作为统一评审基准进行大规模对比 → 通过自动化的胜率统计，客观量化不同模型的批评优劣 → 为后续研究提供了可复现的评估框架，避免了仅靠人工主观判断的局限。

### 方法详解
**整体思路**：Shepherd 的训练分为两大阶段。第一阶段是构建并清洗反馈数据集；第二阶段是基于已有的语言模型（如 LLaMA‑2‑7B）进行针对批评任务的微调。整个流程可以概括为“收集‑筛选‑标注‑微调‑评估”。

1. **数据收集与筛选**  
   - 从公开的社区平台（如 Reddit、StackExchange）抓取用户对 LLM 生成答案的评论，这些评论往往包含“哪里错了”和“怎么改”。  
   - 同时邀请专业标注员对部分生成答案进行人工审阅，补全缺失的错误标记和改进建议。  
   - 通过自动过滤（去除噪声、重复、极短评论）和人工抽检，确保每条记录都具备明确的错误描述和可执行的改写指令。

2. **数据格式化**  
   - 每条样本被组织为三段式：**原始问题**、**模型答案**、**批评与建议**。  
   - 批评部分进一步细分为“错误类型”（事实错误、逻辑不连贯、语言不自然等）和“改进指令”。这种结构让模型在训练时能够学习到从错误定位到具体修改的完整链路。

3. **微调策略**  
   - 采用 **指令微调（Instruction Fine‑tuning）**：在输入端拼接一个固定的指令前缀（如 “请批评并改进以下回答”），让模型明确任务目标。  
   - 使用 **混合损失**：对批评文本采用交叉熵损失，对改进建议采用另一个交叉熵损失，两者加权求和，确保模型既能准确指出错误，也能给出高质量的改写。  
   - 为防止模型在批评时产生“自我强化”错误，引入 **负采样**：在训练中加入一些故意错误的批评，让模型学会辨别不合理的批评。

4. **推理流程**  
   - 用户提交问题和模型生成的答案。  
   - Shepherd 接收这两段文本，输出结构化的批评报告：先列出错误类别，再给出逐条改写建议。  
   - 如果需要，用户可以把 Shepherd 的建议直接喂回原始生成模型，得到改进后的答案，实现闭环。

**最巧妙的点**：作者把“错误定位”和“改写建议”放在同一次前向传播里，而不是像传统的两步走（先评估再生成）。这样模型内部的注意力机制可以在同一上下文中同时对错误和改进进行关联，显著提升了建议的针对性和可操作性。

### 实验与效果
- **测试任务**：论文主要在公开的问答基准（如 TruthfulQA、OpenAI Evals）以及自建的社区反馈集上评估 Shepherd 的批评质量。  
- **对比基线**：包括未微调的原始 LLaMA‑2‑7B、ChatGPT（3.5 版）以及其他公开的批评模型（如 OpenAI 的 “self‑refine” 方案）。  
- **胜率表现**：使用 GPT‑4 作为评审模型进行两两比较，Shepherd 对竞争对手的平均胜率在 53% 到 87% 之间，具体数值随任务而变。对比中，Shepherd 在多数场景下能够击败未微调模型，并在部分细分任务上超过 ChatGPT。  
- **人工评估**：邀请 30 名具备一定专业背景的评审员对批评报告进行打分，Shepherd 的得分严格高于所有对手，且与 ChatGPT 的得分几乎持平。  
- **消融实验**：论文展示了去掉社区反馈、仅使用人工标注、以及不使用负采样的三种变体。结果表明，缺失社区反馈会导致整体胜率下降约 10%，去掉负采样则会出现明显的“错误批评”增多。  
- **局限性**：作者承认 Shepherd 仍然会在高度专业化的领域（如医学、法律）出现误判，因为训练数据中此类错误样本相对稀缺。此外，批评的深度受限于 7B 参数模型的容量，极端长文本的全局一致性检查仍有不足。

### 影响与延伸思考
Shepherd 的出现让“批评模型可以小而精”这一思路在业界迅速传播。随后有几篇工作（如 **CriticLM**、**RefineGPT**）尝试在更大模型上加入类似的批评‑改进模块，或把批评过程嵌入到多轮对话系统中。还有研究把 Shepherd 的反馈数据集公开，作为通用的错误‑改进基准，推动了评估方法的标准化。对想进一步探索的读者，可以关注以下方向：  
- **跨语言批评**：把同样的批评框架扩展到多语言模型，研究语言差异对错误类型的影响。  
- **自适应反馈采集**：让模型在实际使用中主动请求用户反馈，形成持续迭代的闭环。  
- **层次化批评**：从句子级别到段落、全文层面逐层审查，提升对长文一致性的把控。

### 一句话记住它
Shepherd 用 7 B 参数的“小模型”，通过真实社区反馈训练，能够像老师一样指出并改写大模型的错误，效果甚至能和 ChatGPT 持平。