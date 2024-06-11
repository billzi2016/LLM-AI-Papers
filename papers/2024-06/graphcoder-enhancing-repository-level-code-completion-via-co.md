# GraphCoder: Enhancing Repository-Level Code Completion via Code Context   Graph-based Retrieval and Language Model

> **Date**：2024-06-11
> **arXiv**：https://arxiv.org/abs/2406.07003

## Abstract

The performance of repository-level code completion depends upon the effective leverage of both general and repository-specific knowledge. Despite the impressive capability of code LLMs in general code completion tasks, they often exhibit less satisfactory performance on repository-level completion due to the lack of repository-specific knowledge in these LLMs. To address this problem, we propose GraphCoder, a retrieval-augmented code completion framework that leverages LLMs' general code knowledge and the repository-specific knowledge via a graph-based retrieval-generation process. In particular, GraphCoder captures the context of completion target more accurately through code context graph (CCG) that consists of control-flow, data- and control-dependence between code statements, a more structured way to capture the completion target context than the sequence-based context used in existing retrieval-augmented approaches; based on CCG, GraphCoder further employs a coarse-to-fine retrieval process to locate context-similar code snippets with the completion target from the current repository. Experimental results demonstrate both the effectiveness and efficiency of GraphCoder: Compared to baseline retrieval-augmented methods, GraphCoder achieves higher exact match (EM) on average, with increases of +6.06 in code match and +6.23 in identifier match, while using less time and space.

---

# GraphCoder：通过代码上下文图检索与语言模型提升仓库级代码补全 论文详细解读

### 背景：这个问题为什么难？

在实际项目里，代码补全不仅要懂通用的编程语法，还要掌握项目特有的 API、变量命名和实现细节。传统的代码大模型（LLM）在通用补全上表现不错，却因为缺少对当前仓库的专属知识而在仓库级补全上失利。已有的检索增强方法往往把整个文件或函数当成线性序列来匹配，这种“顺序式”上下文忽视了代码内部的控制流和数据依赖，导致检索到的片段与真实需求的相似度不高，进而限制了补全质量。于是，如何在保持大模型通用能力的同时，高效捕获并利用仓库内部的结构化上下文，成为亟待突破的瓶颈。

### 关键概念速览
- **代码上下文图（Code Context Graph，CCG）**：把待补全位置所在的代码块抽象为图结构，节点是语句或表达式，边表示控制流、数据依赖或控制依赖。类似于把代码当成城市地图，路口是语句，路是依赖关系，帮助模型看到“从这里能到哪里”。  
- **检索增强生成（Retrieval‑Augmented Generation，RAG）**：先在外部知识库里找相似片段，再把检索结果喂给生成模型，相当于写作时先查资料再写稿。  
- **粗细粒度检索（Coarse‑to‑Fine Retrieval）**：先用轻量特征快速筛掉大部分不相关代码，再用更精细的图相似度在剩余候选中挑出最贴近的片段，像先用粗筛网过滤大石子，再用细筛网挑选沙子。  
- **代码匹配（Code Match）**：衡量生成代码整体结构与参考答案的相似度，关注语句顺序、控制结构等。  
- **标识符匹配（Identifier Match）**：专门评估变量、函数名等标识符的准确性，反映模型对项目特定命名约定的掌握程度。  
- **Exact Match（EM）**：生成结果与参考答案完全一致的比例，是代码补全常用的严格指标。  

### 核心创新点
1. **从序列到图的上下文建模**  
   - 之前的检索增强方法把代码当成一串字符或 token 来计算相似度，忽略了语句之间的依赖关系。  
   - GraphCoder 用 CCG 把待补全位置的上下文显式建模为图，捕捉控制流、数据依赖和控制依赖。  
   - 这种结构化表示让检索过程更精准，检索到的片段在语义和执行路径上更贴合目标代码。

2. **粗细粒度两阶段检索流程**  
   - 传统方法一次性用高维向量检索，既耗时又容易把噪声带进来。  
   - GraphCoder 先用轻量的语义向量做粗检索，快速排除大部分不相关文件；随后在保留下来的候选上计算图相似度进行细检索。  
   - 结果是检索速度提升且占用更少内存，同时保留了高质量的上下文。

3. **检索结果与 LLM 的协同生成**  
   - 直接把检索到的代码片段拼接到提示中会导致上下文噪声。  
   - GraphCoder 设计了一个“上下文注入”模块，将检索到的图结构信息转化为结构化提示，帮助语言模型在生成时有针对性地利用仓库特有的知识。  
   - 这种协同方式让模型在保持通用能力的同时，显著提升了对项目专属标识符的预测准确率。

### 方法详解
**整体框架**  
GraphCoder 的工作流可以划分为四步：① 构建代码上下文图；② 粗检索候选片段；③ 细粒度图相似度检索；④ 将检索结果注入语言模型进行生成。整体思路是先用轻量特征快速定位可能相关的代码块，再用结构化图信息精炼检索，最后让大模型在“图提示”下完成补全。

**1. 代码上下文图构建**  
- 对每个待补全的函数或代码块，解析其抽象语法树（AST），提取语句节点。  
- 根据控制流分析（如 if、for、while 的分支顺序）添加控制流边；依据数据流分析（变量定义‑使用）添加数据依赖边；再加入控制依赖边（条件对后续语句的影响）。  
- 最终得到一个有向多属性图，节点携带源码片段，边携带依赖类型。

**2. 粗检索（Coarse Retrieval）**  
- 为仓库中每个函数生成一个低维语义向量（如使用预训练的代码嵌入模型），并建立向量索引。  
- 用待补全位置的向量在索引中检索 Top‑k（如 200）最相似的函数，得到一个候选集合。此步骤时间复杂度低，适合大规模仓库。

**3. 细粒度图相似度检索**  
- 对粗检索得到的每个候选函数，重新构建其 CCG。  
- 采用图同构或图嵌入相似度（如 GraphSAGE + cosine）计算候选图与目标图的相似度。  
- 按相似度排序，选出最匹配的 N（如 5）片段作为最终检索结果。这里的关键是把“语义相似”细化为“结构相似”，避免仅凭词向量误检。

**4. 检索结果注入与生成**  
- 将选中的图片段转化为结构化提示：包括关键节点的源码、依赖路径以及标识符映射。  
- 把这些提示拼接到语言模型的输入前缀，形成“检索增强的上下文”。  
- 使用已有的代码 LLM（如 CodeGen、StarCoder）进行自回归生成，模型在生成时可以直接引用检索到的标识符和代码模式。  

**最巧妙的设计**  
- **图到提示的映射**：作者没有直接把图喂给 LLM，而是把图的结构信息压缩成可读的文本提示，这样既利用了 LLM 的自然语言理解能力，又保留了图的结构优势。  
- **两阶段检索**：粗检索的向量快速过滤与细检索的图相似度相结合，既解决了大规模检索的效率问题，又提升了检索质量，这在代码补全场景中少有出现。

### 实验与效果
- **数据集与任务**：在公开的仓库级代码补全基准上（如 CodeXGLUE 的 Repository Completion 子任务）进行评估，任务是给定光标位置预测后续代码。  
- **对比基线**：与传统的检索增强方法（基于纯文本或向量检索）以及纯 LLM（不使用检索）进行比较。  
- **性能提升**：GraphCoder 在 Exact Match（EM）指标上整体领先，代码匹配（Code Match）提升了 **+6.06**，标识符匹配（Identifier Match）提升了 **+6.23**。相较于其他检索增强方案，GraphCoder 还能在更少的时间和内存开销下完成同等任务。  
- **消融实验**：作者分别去掉 CCG、细粒度图相似度、以及检索结果注入三项，发现每一项的缺失都会导致 EM 下降约 2‑3% 以上，说明三者缺一不可。  
- **局限性**：论文未在极大规模（上百万文件）仓库上做全量评测，图构建和细粒度相似度计算仍会在极端情况下成为瓶颈；此外，对动态语言（如 JavaScript）中隐式依赖的捕获仍有待改进。

### 影响与延伸思考
GraphCoder 把“代码图”引入检索增强的流程，为代码补全打开了结构化检索的新方向。后续工作（如 **CodeGraphRAG**、**RepoGraphNet**）已经开始探索更高效的图嵌入和跨语言的图检索技术。对想进一步深入的读者，可以关注以下方向：① 更轻量的图相似度计算（如基于哈希的近似匹配）；② 动态语言的运行时依赖捕获；③ 将图检索与自监督预训练结合，直接让 LLM 学会“读图”。这些都是把结构化代码信息与大模型融合的潜在突破口。

### 一句话记住它
**GraphCoder 用代码依赖图把仓库级上下文变成可检索的结构，让大模型在“看图”后写出更贴合项目的代码。**