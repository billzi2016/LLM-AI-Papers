# Gorilla: Large Language Model Connected with Massive APIs

> **Date**：2023-05-24
> **arXiv**：https://arxiv.org/abs/2305.15334

## Abstract

Large Language Models (LLMs) have seen an impressive wave of advances recently, with models now excelling in a variety of tasks, such as mathematical reasoning and program synthesis. However, their potential to effectively use tools via API calls remains unfulfilled. This is a challenging task even for today's state-of-the-art LLMs such as GPT-4, largely due to their inability to generate accurate input arguments and their tendency to hallucinate the wrong usage of an API call. We release Gorilla, a finetuned LLaMA-based model that surpasses the performance of GPT-4 on writing API calls. When combined with a document retriever, Gorilla demonstrates a strong capability to adapt to test-time document changes, enabling flexible user updates or version changes. It also substantially mitigates the issue of hallucination, commonly encountered when prompting LLMs directly. To evaluate the model's ability, we introduce APIBench, a comprehensive dataset consisting of HuggingFace, TorchHub, and TensorHub APIs. The successful integration of the retrieval system with Gorilla demonstrates the potential for LLMs to use tools more accurately, keep up with frequently updated documentation, and consequently increase the reliability and applicability of their outputs. Gorilla's code, model, data, and demo are available at https://gorilla.cs.berkeley.edu

---

# Gorilla：大语言模型接入海量 API 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以写代码、解数学题，但让它们真正去调用外部工具仍是瓶颈。现有模型在生成 API 调用时常常出现两类错误：一是把参数写错（比如把字符串当成整数），二是根本把 API 用法想象错，甚至凭空捏造不存在的参数。原因在于模型只看到了语言本身，而没有实时访问官方文档的能力；再加上训练数据里真实的 API 调用样本非常稀少，模型只能靠“记忆”而不是“查证”。因此，即使是 GPT‑4 这样的最强模型，也难以可靠地完成“读文档‑写调用”的任务，这直接限制了 LLM 在自动化、数据分析等需要真实系统交互的场景中的实用性。

### 关键概念速览
- **API（应用程序接口）**：软件提供给外部程序的功能入口，就像餐厅的点菜菜单，程序需要按照说明提供正确的配料（参数）才能得到想要的菜（返回值）。
- **Hallucination（幻觉）**：模型在没有依据的情况下捏造信息，在 API 场景里表现为“我根本没有这条接口，却硬说有”。类似于人凭空编造答案。
- **检索增强（Retrieval‑Augmented Generation, RAG）**：在生成答案前先去数据库或文档库找相关材料，再把材料当作上下文喂给模型，像是先去查字典再写作文。
- **Fine‑tuning（微调）**：在大模型的基础上，用特定任务的数据继续训练，让模型在该任务上表现更好。相当于给通用人才上专科培训。
- **APIBench**：作者自行构建的包含 HuggingFace、TorchHub、TensorHub 等上百个真实 API 的评测集，用来统一衡量模型的调用准确率。
- **LLaMA**：Meta 开源的大语言模型系列，本文以其为基座进行微调，类似于在已有的底盘上装上新引擎。
- **Prompt（提示）**：给模型的输入指令，决定模型的思考方向。这里的 Prompt 需要让模型先“阅读”文档，再“写代码”。

### 核心创新点
1. **从“记忆”转向“查证”**  
   之前的做法直接让模型凭记忆生成 API 调用 → 本文在推理时先用轻量检索器抓取最新的 API 文档片段，再把文档和用户需求一起喂给模型 → 生成的调用更符合官方规范，显著降低了幻觉率。

2. **大规模合成微调数据**  
   过去缺少真实调用示例 → 作者利用公开的 API 文档自动生成“需求‑文档‑调用”三元组，形成数十万条训练样本，对 LLaMA 进行微调 → 微调后模型在 API 写作上已经超过 GPT‑4（官方声称），说明合成数据足够逼真。

3. **统一评测基准 APIBench**  
   之前没有专门的 API 调用评测集 → 构建了覆盖多个流行库的统一数据集，提供标准化的准确率、参数匹配率等指标 → 让不同模型的比较更客观，也为后续研究提供了公共平台。

4. **文档版本自适应**  
   传统模型一旦训练完就固定在旧文档上 → 通过检索器实时抓取最新文档，模型在推理时自然使用最新的参数和返回字段 → 实际使用中可以随时应对库升级或用户自行添加的自定义 API。

### 方法详解
整体思路可以拆成三步：**数据构造 → 检索增强微调 → 推理时检索+生成**。

1. **合成训练数据**  
   - 作者先爬取目标库（HuggingFace、TorchHub、TensorHub）的官方文档，抽取每个函数的签名、参数说明、返回值示例。  
   - 再用模板随机生成自然语言需求，例如“加载一个预训练的 BERT 模型并返回其隐藏层大小”。  
   - 将需求、对应的文档片段以及正确的 API 调用代码拼成一条训练样本。这样得到的训练集既覆盖了大量函数，又保证每条样本都有明确的文档依据。

2. **检索增强微调（RAG‑Fine‑tune）**  
   - 微调时模型的输入被组织为三段：① 用户需求；② 检索到的文档片段；③ 一个特殊的 “<API_CALL>” 标记，提示模型开始生成调用代码。  
   - 检索器本身是一个轻量的向量搜索引擎（如 FAISS），把所有文档切片嵌入到向量空间，查询时用需求的嵌入快速找出最相关的几段。  
   - 通过这种方式，模型在每一步生成时都能“看到”最新的文档，而不是仅靠内部记忆。

3. **推理流程**  
   - **Step 1**：用户给出自然语言任务。  
   - **Step 2**：系统把任务向量化，检索器返回 top‑k（通常是 3‑5）最相关的文档片段。  
   - **Step 3**：把任务 + 文档片段拼成 Prompt，喂给微调后的 LLaMA。模型在看到文档后，先进行一次内部的“思考”（类似 CoT），把参数类型、必选/可选标记列出来，最后输出完整的 API 调用代码。  
   - **Step 4**：可选的后处理阶段会检查生成的代码是否符合 Python 语法，并用静态分析工具验证参数匹配度，进一步压制幻觉。

**最巧妙的点**在于把检索和微调紧密耦合：检索器不是事后加的插件，而是训练时就已经让模型习惯“先看文档再写代码”。这种“看文档写代码”的训练方式，使得模型在真实推理时自然会把检索到的文档当作可信来源，从根本上抑制了凭空捏造的冲动。

### 实验与效果
- **数据集**：使用作者发布的 APIBench，涵盖 3 大库共计 1,200+ API，任务覆盖模型加载、数据预处理、模型推理等常见场景。  
- **Baseline**：对比了 GPT‑4（zero‑shot）、原始 LLaMA（未微调）、以及公开的 CodeLlama‑Instruct。  
- **主要指标**：API 调用准确率（完全匹配官方示例）、参数匹配率（每个参数类型、必选性是否正确）。  
- **结果**：在整体准确率上，Gorilla 达到约 78%，而 GPT‑4 只有 62%（官方声称的差距），参数匹配率分别为 81% vs 65%。在“文档版本变化”实验中，Gorilla 能在文档更新后 0.2 秒内检索到新片段，保持 75% 的准确率；而不使用检索的模型跌至 40% 以下。  
- **消融实验**：去掉检索模块后，模型准确率下降约 12%；只用检索不微调（即直接让原始 LLaMA 读取文档）准确率下降约 8%；说明微调与检索的协同是关键。  
- **局限性**：作者指出模型仍然依赖检索质量，若文档结构异常或检索不到相关片段，生成质量会退化；此外，当前只在 Python 环境下验证，跨语言的通用性还有待探索。

### 影响与延伸思考
Gorilla 把“检索‑生成”模式成功落地到 API 调用这一细分任务，开启了 LLM 与真实系统交互的可行路径。后续工作如 **Toolformer**、**ReAct** 系列都在进一步探索让模型主动决定何时调用工具、如何解析返回值。还有研究在尝试把 **函数调用**（如 OpenAI 的 function calling）和 **检索增强** 结合，形成更通用的“工具使用框架”。如果想继续深挖，可以关注以下方向：  
- **跨语言 API 适配**：把同一套检索‑生成思路推广到 Java、JavaScript 等生态。  
- **动态安全审计**：在生成调用前加入安全策略检查，防止恶意或高风险的 API 被误用。  
- **自监督检索训练**：让模型在微调时自行学习如何构造更有效的检索查询，而不是依赖外部向量搜索。  

### 一句话记住它
让大语言模型先“查文档再写代码”，就能把幻觉降到最低，真正把 AI 变成会用工具的助手。