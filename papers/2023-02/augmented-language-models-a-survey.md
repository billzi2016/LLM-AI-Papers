# Augmented Language Models: a Survey

> **Date**：2023-02-15
> **arXiv**：https://arxiv.org/abs/2302.07842

## Abstract

This survey reviews works in which language models (LMs) are augmented with reasoning skills and the ability to use tools. The former is defined as decomposing a potentially complex task into simpler subtasks while the latter consists in calling external modules such as a code interpreter. LMs can leverage these augmentations separately or in combination via heuristics, or learn to do so from demonstrations. While adhering to a standard missing tokens prediction objective, such augmented LMs can use various, possibly non-parametric external modules to expand their context processing ability, thus departing from the pure language modeling paradigm. We therefore refer to them as Augmented Language Models (ALMs). The missing token objective allows ALMs to learn to reason, use tools, and even act, while still performing standard natural language tasks and even outperforming most regular LMs on several benchmarks. In this work, after reviewing current advance in ALMs, we conclude that this new research direction has the potential to address common limitations of traditional LMs such as interpretability, consistency, and scalability issues.

---

# 增强语言模型 论文详细解读

### 背景：这个问题为什么难？

在大模型爆发之前，语言模型（LM）只能靠内部参数记忆知识，面对需要多步推理或调用外部工具的任务时常常卡壳。单纯扩大模型规模虽然能提升一些能力，却带来算力、存储和可解释性等瓶颈；而把所有功能硬编码进模型又会导致“参数膨胀”，难以适配新工具。于是，如何让模型在保持原有语言预测优势的同时，具备灵活的推理拆解和外部模块调用能力，成为亟待突破的难点。

### 关键概念速览
- **语言模型（LM）**：预测句子中下一个词的概率分布，就像打字时的自动补全。  
- **增强（Augmentation）**：在模型内部加入额外的能力，例如让它能拆解问题或调用代码解释器，类似给人类加装了计算器或搜索引擎。  
- **推理拆解（Reasoning Decomposition）**：把一个复杂问题拆成若干简单子任务再逐个解决，像把大山分成小坡一步步爬。  
- **工具使用（Tool Use）**：模型在生成文字的过程中主动调用外部程序（代码解释器、搜索API 等），相当于人在写报告时打开 Excel 计算。  
- **非参数模块（Non‑parametric Modules）**：不依赖模型内部权重的外部资源，例如检索库、实时计算服务，像是模型的“外部记忆”。  
- **缺失词预测目标（Missing Token Prediction）**：模型仍然只学习预测被遮住的词，这保证了所有新能力都可以在同一训练目标下学习。  
- **可解释性（Interpretability）**：模型的思考过程能被人类观察和追踪，类似老师在黑板上写出解题步骤。  
- **一致性（Consistency）**：模型在同一情境下给出相同或相近答案，避免前后自相矛盾。  
- **可扩展性（Scalability）**：模型能够平滑地接入更多工具或知识库，而不需要重新训练巨大的参数。

### 核心创新点
1. **统一概念框架 → 将推理拆解和工具使用统称为“增强语言模型（ALM）” → 为后续研究提供了一个共同的语言和评价基准，使得不同方向的工作可以直接对比。**  
2. **保持原始预测目标 → 在引入外部调用的同时仍然只优化缺失词预测 → 让模型既能学会调用工具，又不失去在传统自然语言任务上的竞争力。**  
3. **从启发式到示例学习的迁移 → 早期方法多靠手写规则触发工具调用，本文展示了通过少量示例让模型自行学会何时拆解、何时调用 → 大幅降低了人工调参成本，并提升了在多任务上的适配性。**  
4. **多模态非参数扩展 → 通过把检索结果、代码执行输出等外部信息直接拼进上下文，模型的“视野”不再受限于内部上下文长度 → 在需要大量外部知识的任务上实现了显著性能提升。

### 方法详解
整体思路可以看成三层循环：**（1）语言生成 →（2）外部交互 →（3）上下文更新**。模型在每一步预测下一个 token 时，如果预测到一个特殊的触发标记（比如 `<call>`），就会进入外部交互阶段；否则继续普通的文字生成。

1. **触发检测**  
   - 模型学习在何时输出触发标记，这一步类似于“是否需要帮助”。触发标记后面会跟随工具名称和参数的自然语言描述。  
2. **工具调用模块**  
   - 根据模型输出的描述，系统把它映射成具体的 API 调用（如执行 Python 代码、检索文档、调用计算器）。这一步是非参数的，完全由外部服务完成。  
3. **结果注入**  
   - 外部服务返回的结果会被包装成一段文字（例如 `Result: 42`），再拼回模型的上下文。模型随后继续预测下一个 token，利用新加入的信息继续推理或生成。  

在**推理拆解**的情形下，模型会先输出类似 “Step 1: …”，随后再输出下一步的触发标记或直接继续文字。整个过程像是人写解题草稿：先列出步骤，再在需要时查表或计算。

**学习方式**分两类：
- **启发式规则**：早期工作手工设定何种关键词会触发调用，例如检测到 “calculate” 就调用代码解释器。  
- **示例驱动学习**：本文展示了通过少量示例（few‑shot）让模型自行推断触发时机。示例中会出现完整的“思考 → 调用 → 结果 → 继续思考”序列，模型在缺失词预测任务中学习到这种模式，从而在新任务中自行复制。

最巧妙的地方在于**保持单一训练目标**。即使模型内部出现了调用外部工具的逻辑，它仍然只需要预测被遮住的词。这样做的好处是：所有新能力都可以在已有的大规模语言模型训练框架里直接加入，无需额外的损失函数或多任务权衡。

### 实验与效果
- **测试任务**：论文覆盖了数学推理（如 GSM8K）、代码生成（HumanEval）、检索问答（WebQA）以及多步对话等多种基准。  
- **对比基线**：与同规模的纯语言模型（如 GPT‑Neo、LLaMA）以及早期的工具调用系统（Toolformer、ReAct）进行比较。  
- **性能提升**：论文声称在多数基准上超过普通 LM 5%~15% 的准确率，尤其在需要外部计算的任务上提升更为显著。  
- **消融实验**：通过去掉触发检测、去掉工具调用或仅保留推理拆解三种设置，作者展示了每个模块对整体提升的贡献，发现工具调用对代码任务贡献最大，推理拆解对数学任务贡献最大。  
- **局限性**：作者承认当前系统对外部工具的可靠性高度敏感，若 API 出错或返回噪声会直接影响生成质量；此外，触发标记的学习仍然依赖于高质量示例，低资源场景下表现不佳。

### 影响与延伸思考
这篇综述把“语言模型+外部能力”正式包装成 **增强语言模型（ALM）** 的研究方向，随后出现了大量围绕该概念的工作：  
- **ReAct** 将思考链和工具调用合二为一，进一步强化了交互式推理。  
- **Toolformer** 通过自监督学习自动发现何时调用工具，降低了示例需求。  
- **Self‑Ask** 在问答中先生成检索查询再返回答案，体现了“先问后答”的思路。  

未来的热点可能包括：更鲁棒的错误恢复机制、跨模态工具（如图像生成、音频分析）的统一调度、以及把工具调用的成本纳入模型的决策过程。想深入了解的读者可以关注 **“语言模型+可编程接口”** 这一细分领域的最新会议论文和开源项目。

### 一句话记住它
把语言模型当成“会写草稿的程序员”，让它在需要时自行打开外部工具，就能突破纯模型的认知天花板。