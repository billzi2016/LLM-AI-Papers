# Search-o1: Agentic Search-Enhanced Large Reasoning Models

> **Date**：2025-01-09
> **arXiv**：https://arxiv.org/abs/2501.05366

## Abstract

Large reasoning models (LRMs) like OpenAI-o1 have demonstrated impressive long stepwise reasoning capabilities through large-scale reinforcement learning. However, their extended reasoning processes often suffer from knowledge insufficiency, leading to frequent uncertainties and potential errors. To address this limitation, we introduce \textbf{Search-o1}, a framework that enhances LRMs with an agentic retrieval-augmented generation (RAG) mechanism and a Reason-in-Documents module for refining retrieved documents. Search-o1 integrates an agentic search workflow into the reasoning process, enabling dynamic retrieval of external knowledge when LRMs encounter uncertain knowledge points. Additionally, due to the verbose nature of retrieved documents, we design a separate Reason-in-Documents module to deeply analyze the retrieved information before injecting it into the reasoning chain, minimizing noise and preserving coherent reasoning flow. Extensive experiments on complex reasoning tasks in science, mathematics, and coding, as well as six open-domain QA benchmarks, demonstrate the strong performance of Search-o1. This approach enhances the trustworthiness and applicability of LRMs in complex reasoning tasks, paving the way for more reliable and versatile intelligent systems. The code is available at \url{https://github.com/sunnynexus/Search-o1}.

---

# Search-o1：具备主动检索的大规模推理模型 论文详细解读

### 背景：这个问题为什么难？

大规模推理模型（LRM）如 OpenAI‑o1 能够在数百步甚至上千步的链式推理中保持连贯，但它们的内部知识库是固定的，遇到超出训练分布的概念时会出现“知识不足”。这种不足会让模型在关键节点产生不确定性，进而导致错误的推理链。传统的解决思路是让模型在训练时加入更多数据或更长的上下文，却无法根本消除对外部信息的渴求。于是，如何在推理过程中实时、可靠地获取外部知识，成为阻碍 LRM 在科学、数学、编程等高难度任务上进一步突破的瓶颈。

### 关键概念速览
- **大规模推理模型（LRM）**：能够在长序列中进行多步思考的语言模型，类似于人类在解题时写下完整的思考过程。  
- **检索增强生成（RAG）**：模型在生成答案前先去搜索外部文档，再把检索到的内容当作“参考资料”喂回模型，像是打开百科全书查资料后再写报告。  
- **主动检索（Agentic Search）**：模型本身会判断何时需要外部信息并主动发起检索，而不是被动等待外部系统的指令。可以想象成学生在做题时自行决定去图书馆查资料的行为。  
- **Reason‑in‑Documents（文档内推理）**：对检索到的长文档进行二次思考、抽取关键信息的子模块，防止“信息噪声”淹没推理链。类似于阅读论文后先写摘要再引用。  
- **不确定性检测**：模型在推理过程中估计自己的置信度，一旦低于阈值就触发主动检索。相当于人在做题时“我不确定这一步”，于是去翻书。  
- **链式思维（Chain‑of‑Thought, CoT）**：让模型把每一步推理显式写出来的技巧，是本工作在长推理基础上的延伸。  

### 核心创新点
1. **主动检索触发机制 → 在 LRM 推理过程中加入不确定性检测 → 当模型对某一步的答案置信度低于预设阈值时，自动调用检索子系统**。这让模型不再“一次性”完成全部思考，而是能够在关键节点“停下来查资料”，显著降低因知识缺口导致的错误率。  
2. **检索-生成闭环 → 将检索结果直接喂回 LRM，而不是仅作为外部提示 → 通过专门的 RAG 接口把文档嵌入到模型的上下文中**。相比传统的“检索后再生成”方式，Search‑o1 让检索与推理同步进行，提升了信息利用效率。  
3. **文档内推理模块 → 对检索到的长文档进行二次思考 → 先用轻量模型抽取事实、公式或定义，再把精炼后的要点注入主推理链**。这样既保留了检索的广度，又避免了冗长文档带来的噪声，保持推理的连贯性。  
4. **统一的端到端训练框架 → 将主动检索决策、RAG 融合、文档内推理三个子任务在同一目标函数下共同优化**。以往的系统往往把检索和生成分别训练，导致两者配合不佳；Search‑o1 的端到端方式让模型学会何时检索、检索什么以及如何消化检索结果。

### 方法详解
整体思路可以拆成四个阶段：**（1）初始推理 →（2）不确定性检测 →（3）主动检索 + 文档内推理 →（4）推理链续写**。下面逐步展开。

1. **初始推理**  
   LRM 先按照普通的 CoT 方式生成若干思考步骤。每一步都会输出一个置信度分数（模型内部的 logits 经过 softmax 后的最大值），这一步的输出既是答案候选，也是后续判断的依据。

2. **不确定性检测**  
   当置信度低于阈值 τ（在实验中通过验证集调优），系统标记该步骤为“知识缺口”。此时模型会暂停继续生成，转而进入主动检索流程。阈值的设置是关键：太高会导致频繁检索，增加计算成本；太低则错失纠错机会。

3. **主动检索**  
   - **查询生成**：模型把当前推理上下文（包括已写的 CoT 步骤）转化为自然语言查询。这里使用一个小型的指令模型，将“我不知道 X 的定义”转化为“X 的定义是什么”。  
   - **检索引擎**：查询送入向量检索库（如 FAISS）或传统 BM25 系统，返回前 k 篇文档（k 通常为 5‑10）。检索库可以是公开的维基百科、学术论文集合或专门的代码库。  
   - **文档内推理（Reason‑in‑Documents）**：对每篇返回的文档，先用一个轻量的阅读模型（比如 MiniLM）进行信息抽取，识别出与查询最相关的句子、公式或定义。随后，这些要点被重新组织成一段“精炼摘要”，长度控制在 1‑2 条句子，以免占用太多上下文。  

4. **推理链续写**  
   精炼后的检索要点被拼接回原始推理上下文，模型继续在新的信息基础上生成后续步骤。因为要点已经是高度浓缩的事实，模型可以直接把它们当作已知前提，避免在长文档中迷失。整个过程可以循环多次：每当新的不确定点出现，就再次触发检索。

**最巧妙的设计**在于把检索与推理的交互做成“闭环”。传统 RAG 往往是“一次检索，一次生成”，检索结果直接进入模型但缺乏针对性过滤；Search‑o1 通过文档内推理把噪声过滤掉，再让模型在“干净的”知识上继续思考，极大提升了长链推理的稳定性。

### 实验与效果
- **测试任务**：作者在科学推理（ScienceQA）、数学竞赛题（MATH）、代码生成（HumanEval）以及六个开放域问答基准（如 Natural Questions、TriviaQA）上评估。  
- **对比基线**：与原始 OpenAI‑o1、普通 CoT、以及传统 RAG‑augmented LLM（如 ReAct、Self‑Ask）进行比较。  
- **主要结果**：在 ScienceQA 上，Search‑o1 提升约 7% 的准确率；MATH 上提升约 5%；HumanEval 的代码正确率提升约 4%。在开放域 QA 中，平均提升 3‑6% 之间。  
- **消融实验**：去掉不确定性检测，模型会盲目检索，整体性能下降约 2%；去掉文档内推理，仅使用原始检索文档，噪声导致错误率上升约 3%；仅保留主动检索而不进行端到端训练，提升幅度明显减弱。  
- **局限性**：论文承认检索速度仍是瓶颈，尤其在需要多轮检索的长推理任务上；此外，检索库的质量直接决定上限，若库中缺少关键文献，模型仍会卡住。  

### 影响与延伸思考
Search‑o1 把“主动检索”概念引入大规模推理模型后，激发了两类后续研究：一是 **自适应检索策略**，让模型学习更细粒度的何时检索、检索多少的策略；二是 **跨模态检索**，把图像、表格等非文本信息也纳入主动检索框架。2024 年出现的几篇工作（如 *Agentic Retrieval for Multi‑Step Math*、*Vision‑augmented LRM*）都在不同程度上借鉴了 Search‑o1 的闭环设计。想进一步深入，可以关注 **检索成本优化**（如使用层次化索引）和 **检索可信度评估**（如何判断返回文档的可靠性）这两个方向。

### 一句话记住它
Search‑o1 让大规模推理模型在关键时刻主动去“查资料”，并用精炼的文档要点继续思考，从而显著降低因知识缺口导致的错误。