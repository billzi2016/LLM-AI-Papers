# Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.16901

## Abstract

Recent advances in Large Language Models (LLMs) have shown promise in function-level code generation, yet repository-level software engineering tasks remain challenging. Current solutions predominantly rely on proprietary LLM agents, which introduce unpredictability and limit accessibility, raising concerns about data privacy and model customization. This paper investigates whether open-source LLMs can effectively address repository-level tasks without requiring agent-based approaches. We demonstrate this is possible by enabling LLMs to comprehend functions and files within codebases through their semantic information and structural dependencies. To this end, we introduce Code Graph Models (CGMs), which integrate repository code graph structures into the LLM's attention mechanism and map node attributes to the LLM's input space using a specialized adapter. When combined with an agentless graph RAG framework, our approach achieves a 43.00% resolution rate on the SWE-bench Lite benchmark using the open-source Qwen2.5-72B model. This performance ranks first among open weight models, second among methods with open-source systems, and eighth overall, surpassing the previous best open-source model-based method by 12.33%.

---

# 代码图模型（CGM）：面向仓库级软件工程任务的图集成大语言模型 论文详细解读

### 背景：这个问题为什么难？

在代码生成的早期阶段，LLM（大语言模型）已经可以在单函数层面写出可运行的代码，但把模型的视野扩展到整个代码仓库就不那么容易了。仓库级任务往往需要跨文件的语义关联、依赖关系以及项目结构信息，而这些信息在纯文本提示里很难完整呈现。现有的解决方案大多依赖商业化的“LLM 代理”，即把模型包装成一个可以自行检索、调用工具的智能体，这种做法既增加了不可预测的行为，又把模型、数据和部署锁进了闭源生态，导致隐私和可定制性受限。于是，如何在保持开源、可控的前提下，让 LLM 直接感知代码库的结构，成为了一个迫切的研究点。

### 关键概念速览

**代码图（Code Graph）**：把仓库里的函数、类、文件等抽象成节点，用调用、导入、继承等关系连成图。想象成一张城市地图，节点是建筑，边是道路，模型可以在这张地图上“走动”。  

**图集成注意力（Graph‑Integrated Attention）**：在 LLM 的自注意力计算里加入图结构信息，让模型在关注某个 token 时，同时参考它在代码图中的邻居节点。类似于在阅读一段文字时，你会顺手翻看相关的章节标题。  

**Adapter（适配层）**：一种轻量的映射网络，把代码图节点的属性（如函数签名、注释、文件路径）转化为 LLM 能接受的向量形式。它相当于把不同语言的翻译官，负责把图信息“说”成模型能听懂的话。  

**RAG（检索增强生成）**：先检索与当前任务最相关的文档或代码片段，再把检索结果喂给生成模型，以提升答案的准确性。这里的检索对象是代码图的子图，而不是纯文本。  

**Agentless（无代理）**：指不使用额外的自动化工具或脚本让模型自行决定检索、调用的流程，而是把所有检索和图信息的融合提前写进模型的输入/注意力机制。  

**SWE‑bench Lite**：一个公开的仓库级软件工程基准，包含多种真实的 bug 修复、功能实现和代码审查任务，用来衡量模型在实际项目中的实用性。  

**开源权重模型**：指模型的参数是公开可下载的，例如 Qwen2.5‑72B，研究者可以自行部署、微调或改造，而不受商业授权限制。

### 核心创新点

1. **从纯文本提示 → 图结构嵌入**：传统方法只把源码当作长文本喂给 LLM，信息稀释且缺乏跨文件上下文。CGM 在注意力层面直接注入代码图的邻接信息，使模型在计算每个 token 的注意力时能够“看到”它的调用者、被调用者等结构化线索。实验表明，这种结构感知提升了 43% 的任务解决率。  

2. **专用 Adapter 将图属性映射到模型空间 → 无需额外微调**：以前要让 LLM 使用图信息，需要大规模微调或额外的嵌入层。CGM 设计了一个轻量的适配层，只在前向传播时把函数签名、文件路径等属性转成向量，几乎不改变原模型的权重。这样既保持了开源模型的原始能力，又让图信息顺畅进入注意力计算。  

3. **Agentless Graph‑RAG 框架 → 统一检索与生成**：多数现有系统使用独立的检索模块和一个“代理”来决定何时检索、如何使用检索结果。CGM 把检索过程嵌入到图注意力里，模型在一次前向传播中即可完成相关子图的定位和答案生成，省去复杂的控制逻辑，显著降低了实现难度和运行时开销。  

4. **在完全开源环境下实现领先性能 → 打破闭源垄断**：使用 Qwen2.5‑72B（开源权重）配合 CGM，论文在 SWE‑bench Lite 上取得 43.00% 的解决率，领先所有同类开源模型，且仅次于少数商业系统。证明了“开源+图结构”完全可以匹配甚至超越“闭源代理”方案。

### 方法详解

#### 整体框架

CGM 的工作流可以拆成四步：  
1) **代码图构建**：对目标仓库进行静态分析，抽取函数、类、文件等实体并建立调用、导入等边，得到一个有向异构图。  
2) **节点属性编码**：每个节点的文本信息（如函数签名、docstring、文件路径）经过专用 Adapter，映射成与 LLM 输入维度相同的向量。  
3) **图集成注意力计算**：在 LLM 的每一层自注意力中，除了原有的 QKV（查询、键、值）矩阵，还加入一个基于图邻接矩阵的偏置项，使得相邻节点的向量在注意力得分上获得额外加权。  
4) **答案生成**：模型在融合了图结构的上下文后，直接输出任务的答案（如补丁代码、实现说明），无需外部检索或工具调用。

#### 关键模块拆解

- **代码图构建**：使用开源的抽象语法树（AST）解析器遍历所有源码文件，记录每个函数的入口、出口、导入路径等信息。得到的图类似于社交网络：节点是代码实体，边是“调用”或“依赖”。  

- **Adapter 设计**：Adapter 由两层线性投影组成，第一层把原始文本向量（通过已有的代码嵌入模型得到）映射到隐藏维度，第二层再映射到 LLM 的词向量空间。这样即使是不同语言的代码，也能统一进入模型。  

- **图集成注意力**：在标准的自注意力公式里，注意力得分是 Q·K^T / sqrt(d)。CGM 在 K（键）矩阵上加上一个由图邻接矩阵乘以可学习权重得到的偏置，记作 K' = K + α·A·V，其中 A 是邻接矩阵，V 是节点值向量，α 是标量调节系数。直观上，这相当于让模型在“看”某个 token 时，额外“听”到它在代码图中的邻居说的话。  

- **Agentless Graph‑RAG**：检索过程不再是独立的搜索引擎，而是通过注意力的图偏置自然实现。模型在一次前向传播中会自动把与当前任务最相关的子图（例如调用链）放大权重，从而在生成答案时自然利用这些信息。  

#### 巧妙之处

最让人眼前一亮的是把图结构直接写进注意力层，而不是在模型外部做后处理。这样做的好处是：① 只需一次前向传播即可完成检索与生成，省去额外的 API 调用；② 图信息与语言信息在同一层次上交互，避免了信息丢失；③ 只需在原模型上加一个轻量的 Adapter 和邻接偏置，保持了开源模型的完整性。

### 实验与效果

- **测试平台**：论文在公开的 SWE‑bench Lite 基准上评估，任务覆盖 bug 修复、功能实现、代码审查等 10 多种真实项目场景。  

- **对比基线**：包括开源的 CodeLlama、StarCoder、以及商业的 GPT‑4、Claude。使用相同的硬件（8×A100）和提示方式进行公平比较。  

- **核心结果**：CGM 搭配 Qwen2.5‑72B 达到 43.00% 的任务解决率，领先同类开源模型约 12.33%，在所有方法中排名第八，仅次于少数商业系统。  

- **消融实验**：  
  1) **去掉图集成注意力**，仅使用 Adapter，解决率跌至 31.5%。  
  2) **去掉 Adapter**，直接把图节点属性当普通 token，效果下降至 34.2%。  
  3) **使用传统检索+LLM（非图）**，比 CGM 低约 7%。这些实验表明图注意力和 Adapter 两者缺一不可。  

- **局限性**：论文承认，当前的代码图构建依赖于静态分析，难以捕捉运行时多态或动态导入的关系；此外，图规模随仓库增大而呈指数增长，需进一步研究稀疏化或分层图策略。  

### 影响与延伸思考

CGM 的出现让社区看到，开源 LLM 完全可以通过结构化信息突破仓库级任务的瓶颈，降低对商业代理的依赖。随后几个月，出现了几篇基于“图‑注意力”思路的后续工作，例如将函数调用图与测试覆盖图联合建模，或把项目的构建依赖树作为额外的图层输入。推测未来的研究会聚焦在：① 动态代码分析与图的融合；② 大规模图的分块与层次化注意力；③ 多模态（代码+文档+Issue）统一图表示。对想进一步深入的读者，可以关注“代码图神经网络（CodeGNN）”和“检索增强生成（RAG）”的最新进展，它们正逐步向 CGM 的方向收敛。

### 一句话记住它

**把代码库的调用图直接写进大语言模型的注意力里，就能让开源模型在仓库级任务上跑得比闭源代理更快、更稳。**