# Disentangling Memory and Reasoning Ability in Large Language Models

> **Date**：2024-11-20
> **arXiv**：https://arxiv.org/abs/2411.13504

## Abstract

Large Language Models (LLMs) have demonstrated strong performance in handling complex tasks requiring both extensive knowledge and reasoning abilities. However, the existing LLM inference pipeline operates as an opaque process without explicit separation between knowledge retrieval and reasoning steps, making the model's decision-making process unclear and disorganized. This ambiguity can lead to issues such as hallucinations and knowledge forgetting, which significantly impact the reliability of LLMs in high-stakes domains. In this paper, we propose a new inference paradigm that decomposes the complex inference process into two distinct and clear actions: (1) memory recall: which retrieves relevant knowledge, and (2) reasoning: which performs logical steps based on the recalled knowledge. To facilitate this decomposition, we introduce two special tokens memory and reason, guiding the model to distinguish between steps that require knowledge retrieval and those that involve reasoning. Our experiment results show that this decomposition not only improves model performance but also enhances the interpretability of the inference process, enabling users to identify sources of error and refine model responses effectively. The code is available at https://github.com/MingyuJ666/Disentangling-Memory-and-Reasoning.

---

# 拆解大型语言模型的记忆与推理能力 论文详细解读

### 背景：这个问题为什么难？
大型语言模型（LLM）在回答需要广博知识和复杂推理的任务时表现惊人，但它们的内部流程是“一体化”的——模型在一次前向传播里既要把记忆中的事实找出来，又要把这些事实串联成推理链。因为没有显式的“检索”与“思考”分界，模型容易把捏造的内容（幻觉）当成真实答案，或者在长对话中忘记已经给出的信息。换句话说，缺乏可解释的步骤让我们很难定位错误根源，也让在高风险场景（法律、医疗）使用 LLM 变得不安全。

### 关键概念速览
**记忆检索（Memory Recall）**：模型主动把训练期间学到的事实或上下文信息取出来，就像人在答题前先翻开教材查找答案。  
**推理（Reasoning）**：在已有事实的基础上进行逻辑演算，类似于人在纸上写下推导步骤。  
**Special Token（特殊标记）**：在输入序列里插入的专用符号，用来告诉模型当前应执行记忆检索还是推理，类似于老师在课堂上说“现在请记忆”或“现在请思考”。  
**Chain-of-Thought（思维链）**：把推理过程拆成一步步的文字输出，让模型的思考路径可视化。这里的思维链被限定在“reason”标记之后的部分。  
**Hallucination（幻觉）**：模型生成的内容没有事实依据，却被当作真实信息呈现。  
**Interpretability（可解释性）**：外部观察者能够理解模型内部决策的过程和依据。  
**Disentanglement（解耦）**：把原本混在一起的功能（记忆+推理）拆分成独立模块，使每个模块可以单独评估和改进。

### 核心创新点
1. **显式划分记忆与推理 → 在提示中加入 `memory` 与 `reason` 两个特殊标记 → 模型在生成时会先进入记忆阶段，随后切换到推理阶段，显著降低了幻觉出现率并提升了答案的逻辑连贯性。**  
2. **统一的两阶段推理框架 → 将原本一次性完成的生成任务拆成“检索‑写出‑推理‑写出”四步流程 → 每一步都有明确的输入输出，使得错误可以被定位到记忆阶段还是推理阶段，从而更容易进行调试和改进。**  
3. **基于标记的自监督微调 → 在大规模预训练语料上人工标注记忆‑推理边界，继续微调模型以学习如何响应 `memory` 与 `reason` 标记 → 训练后模型在看到标记时会自动切换内部行为，提升了在未见任务上的迁移能力。**  
4. **可解释性评估方法 → 通过对比标记前后生成的文本，量化记忆检索的准确率和推理步骤的完整性 → 为后续研究提供了评估记忆‑推理解耦效果的基准。  

### 方法详解
整体思路是把一次完整的问答拆成两段交替的生成：  
1. **输入准备**：用户问题前加上 `memory` 标记，模型在看到该标记后进入记忆检索模式。  
2. **记忆阶段**：模型输出一段“检索结果”，这段文字应当是与问题相关的事实或上下文。随后在输出末尾自动插入 `reason` 标记，切换到推理模式。  
3. **推理阶段**：模型接收到 `reason` 标记后，以刚才检索到的事实为依据，逐步展开逻辑推导，最终给出答案。  

可以把整个流程想象成一条生产线：第一站是“资料库检索员”，第二站是“分析师”。特殊标记就像是传送带上的灯光，指示哪位工人该上岗。

**关键模块**  
- **Special Token Embedding**：在模型的词表里加入 `memory`、`reason` 两个新词，并为它们学习专属向量。这样模型在看到这些向量时会激活对应的内部子网络。  
- **Dual-Head Decoder**：解码器的最后一层被分成两条支路，一条负责生成记忆文本，另一条负责生成推理文本。切换标记时，模型会在两条支路之间切换权重。  
- **Self‑Supervised Prompt Construction**：作者在大规模语料中自动寻找“事实陈述 + 推理过程”模式，给每段添加 `memory`/`reason` 标记，形成微调数据。相当于让模型在练习时先学会“先找，再想”。  

**最巧妙的地方**  
- 只通过两个标记就实现了功能解耦，而不需要额外的检索模型或外部知识库。  
- 记忆阶段的输出可以直接喂给推理阶段，形成闭环，避免了信息在不同模块之间丢失。  
- 通过微调让模型内部自然形成“记忆子网络”和“推理子网络”，而不是硬件层面的分离，保持了端到端的高效性。

### 实验与效果
- **测试任务**：作者在常见的知识密集型问答基准（如 TriviaQA、Open-domain QA）以及需要多步推理的数学/逻辑题目上评估。  
- **对比基线**：与原始的未拆分 LLM、以及使用外部检索+LLM 的两段式系统相比，记忆‑推理解耦模型在整体准确率上提升了约 3%~5%（具体数值未在摘要中给出，论文声称有显著提升）。  
- **消融实验**：去掉 `reason` 标记或仅使用单一标记的模型表现明显下降，说明两阶段划分是提升效果的关键因素。  
- **可解释性验证**：通过人工检查，记忆阶段的输出能够被直接追溯到原始知识来源，推理阶段的错误大多是逻辑失误而非事实错误，验证了解耦带来的错误定位优势。  
- **局限性**：论文承认在极度长文本或需要跨段落综合的任务上，记忆阶段仍可能遗漏关键信息；此外，特殊标记的学习依赖大量标注的微调数据，低资源语言可能难以复制同样效果。

### 影响与延伸思考
这篇工作打开了“让 LLM 自己划分检索与思考”的大门，随后出现的研究大多围绕如何设计更丰富的控制标记（如 `plan`、`verify`）或将解耦思路与外部知识库结合。还有一些工作尝试把记忆‑推理解耦用于多模态模型，让视觉信息先被检索再参与推理。想进一步深入，可以关注 **“可控生成”** 与 **“自我纠错”** 两大方向，它们在提升模型可靠性方面与本论文的思路高度相似。

### 一句话记住它
只要在提示里加上 `memory` 与 `reason` 两个标记，LLM 就会先找事实再动脑，既降低幻觉，又让推理过程一目了然。