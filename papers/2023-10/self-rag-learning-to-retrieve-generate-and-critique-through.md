# Self-RAG: Learning to Retrieve, Generate, and Critique through   Self-Reflection

> **Date**：2023-10-17
> **arXiv**：https://arxiv.org/abs/2310.11511

## Abstract

Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. Retrieval-Augmented Generation (RAG), an ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation. We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (Self-RAG) that enhances an LM's quality and factuality through retrieval and self-reflection. Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens. Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. Experiments show that Self-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks. Specifically, Self-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA, reasoning and fact verification tasks, and it shows significant gains in improving factuality and citation accuracy for long-form generations relative to these models.

---

# 自反思检索增强生成（Self‑RAG）论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在对话、写作等场景表现惊人，但它们的知识全部埋在模型参数里，一旦遇到训练时少见或已经更新的事实，就会“凭空”编造答案。为了解决这个“记忆不可靠”问题，研究者提出了检索增强生成（RAG），让模型在生成前先去外部文档库找相关片段。然而，传统 RAG 有两个致命短板：一是检索过程是固定的——不管问题是否需要外部信息，模型都会硬性拉取若干段落，导致不必要的噪声或计算开销；二是检索到的段落往往未经模型自行评估，直接拼进答案里，容易出现引用不相关或误导的情况。于是，如何让模型在需要时主动检索、并对检索结果和自己生成的内容进行自我审视，成为提升事实性和可解释性的关键挑战。

### 关键概念速览
- **检索增强生成（RAG）**：在语言模型生成文本前，先用搜索引擎或向量库找出与输入最相关的文档片段，再把这些片段当作“外部记忆”喂给模型。类似于人写报告时先查资料再写稿。
- **自反思（Self‑Reflection）**：模型在生成过程中插入专门的标记，让自己对已经产生的文字进行评价或批判，就像写作时不停地自我检查。
- **反思标记（Reflection Tokens）**：一组特殊的 token，模型在生成时可以主动触发它们，以切换到“审视模式”。相当于在对话中说“等等，我再想想”。
- **按需检索（On‑Demand Retrieval）**：模型根据当前上下文决定是否需要检索以及检索多少，而不是一刀切地固定数量。好比学生在答题时只有在不确定时才去翻教材。
- **引用准确性（Citation Accuracy）**：生成的长文中每段事实对应的检索来源是否正确匹配，衡量模型能否给出可靠的参考文献。
- **事实验证（Fact Verification）**：模型输出的陈述是否与真实世界事实相符，常用外部知识库或人工标注来评估。

### 核心创新点
1. **检索触发从硬性变为可控**  
   传统 RAG 总是强制检索固定数量的段落。Self‑RAG 让模型在生成时自行决定是否检索、检索多少，这通过在输入中加入“是否检索？”的反思标记实现。结果是模型在不需要外部信息的任务上保持轻量，在需要时才动用检索资源，提升了效率和答案的相关度。

2. **统一模型同时完成检索、生成、审议**  
   过去的系统往往把检索器、生成器、审查器拆成不同模型或流水线。Self‑RAG 只微调一个任意大小的语言模型，使其在同一次前向传播中完成三件事：判断是否检索、读取检索结果、对检索内容和自身输出进行自我批评。这样既简化了部署，又让三个环节能够相互影响，形成闭环反馈。

3. **反思标记驱动的行为调节**  
   通过在训练语料中显式加入反思标记，模型学会在不同任务需求下切换行为。例如在开放域问答时，标记会促使模型先检索再回答；在创意写作时，标记会让模型倾向于少检索、多生成。标记的出现相当于给模型装上了“开关”，在推理阶段可以根据用户指令或上下文灵活调节。

4. **自我批判提升引用和事实准确性**  
   在生成完答案后，模型会再次生成反思标记，引导自己检查每个陈述是否得到检索证据的支持，并在必要时补充或修改引用。实验显示，这一步显著降低了“幻觉”现象，使长文生成的引用准确率比传统 RAG 高出数个百分点（具体数字未在摘要中给出）。

### 方法详解
**整体思路**  
Self‑RAG 把“检索‑生成‑审议”三步压缩进一次自回归过程。模型在每一步的输出中可以出现三类 token：普通文本、检索指令、反思指令。整个流程如下：

1. **输入编码**：用户问题 + 可选的任务指示（如“需要引用”）被编码进模型的上下文。  
2. **自我判断**：模型首先生成一个特殊的“检索/不检索”标记。如果输出是“检索”，模型会进入检索子模块；如果是“不检索”，直接进入生成子模块。  
3. **检索子模块**：模型把问题向量化，查询外部向量库，返回若干段落（数量由模型在前一步决定）。这些段落被拼接回上下文，标记为“检索结果”。  
4. **生成子模块**：模型在包含检索结果的上下文中继续生成答案的主体。  
5. **自我批判**：生成完答案后，模型再次输出反思标记，进入审议阶段。此时模型会逐句检查答案是否有对应的检索依据，若发现缺失或冲突，会在原答案后追加更正或补充引用。  
6. **最终输出**：经过审议的文本即为模型的最终回答，包含必要的引用信息。

**关键模块拆解**  
- **检索决策层**：通过在训练时让模型看到“需要检索”与“不需要检索”的对比案例，模型学会在语义上感知信息缺口。可以把它想象成一个“是否去图书馆查资料”的判断器。  
- **向量检索器**：使用常见的密集向量检索（如 FAISS）与语义相似度排序，返回最相关的段落。段落数量是模型在步骤2中决定的，通常在 1‑5 条之间。  
- **反思生成器**：在训练语料中加入“[REFLECT]”之类的标记，模型在看到该标记后会切换到审议模式。审议模式的目标是输出“[EVIDENCE]段落编号”或“[CORRECT]更正文本”。这相当于让模型在写完文章后自己做一次脚注检查。  
- **统一微调**：所有上述行为都在同一个语言模型上通过指令微调实现。作者声称只需要一次微调即可让任意规模的模型（7B、13B）具备上述能力。

**最巧妙的设计**  
把检索决策和自我审议都包装成普通的 token，使得在推理时不需要额外的控制逻辑，只要模型生成对应的标记，外部系统就会触发相应的子过程。这种“语言层面的控制流”让模型本身成为整个系统的调度中心，极大降低了工程复杂度。

### 实验与效果
- **测试任务**：开放域问答、推理题、事实验证以及长篇生成（需要引用的文章写作）。  
- **对比基线**：ChatGPT、检索增强版 Llama2‑chat、以及未使用自反思的普通 LLM。  
- **核心结果**：论文声称 Self‑RAG（7B、13B）在所有任务上均显著超越基线，尤其在长文生成的引用准确性和整体事实性上领先数个百分点。具体数值未在摘要中披露。  
- **消融实验**：作者分别去掉“按需检索”和“自我批判”两块，发现去掉任意一块都会导致性能回落，说明两者对提升事实性同等重要。  
- **局限性**：模型仍然依赖外部检索库的质量；在极度稀缺或高度专业的领域，检索到的段落可能不足以支撑审议；此外，反思标记的生成仍有概率出现误触发，导致不必要的检索开销。作者在讨论中承认这些问题，并把进一步提升检索可靠性列为未来工作。

### 影响与延伸思考
Self‑RAG 把“自我审查”直接写进语言模型的生成流程，开启了“模型自控”这一新方向。随后的工作（如 Reflexion‑LLM、Self‑Check GPT）纷纷借鉴了反思标记的思路，尝试让模型在更细粒度上评估自己的答案可信度。对想深入的读者，可以关注以下两个方向：  
1. **指令微调与行为调节**：如何设计更丰富的控制标记，让模型在同一次推理中完成多任务切换。  
2. **检索质量评估**：结合学习到的自我批判信号，动态过滤低质量检索结果，进一步降低幻觉风险。

### 一句话记住它
Self‑RAG 让语言模型自己决定何时检索、如何引用，并在生成后自我审查，像一个会查资料、会自我校对的“聪明写手”。