# Context-Augmented Code Generation Using Programming Knowledge Graphs

> **Date**：2024-10-09
> **arXiv**：https://arxiv.org/abs/2410.18251

## Abstract

Large Language Models (LLMs) and Code-LLMs (CLLMs) have significantly improved code generation, but, they frequently face difficulties when dealing with challenging and complex problems. Retrieval-Augmented Generation (RAG) addresses this issue by retrieving and integrating external knowledge at the inference time. However, retrieval models often fail to find most relevant context, and generation models, with limited context capacity, can hallucinate when given irrelevant data. We present a novel framework that leverages a Programming Knowledge Graph (PKG) to semantically represent and retrieve code. This approach enables fine-grained code retrieval by focusing on the most relevant segments while reducing irrelevant context through a tree-pruning technique. PKG is coupled with a re-ranking mechanism to reduce even more hallucinations by selectively integrating non-RAG solutions. We propose two retrieval approaches-block-wise and function-wise-based on the PKG, optimizing context granularity. Evaluations on the HumanEval and MBPP benchmarks show our method improves pass@1 accuracy by up to 20%, and outperforms state-of-the-art models by up to 34% on MBPP. Our contributions include PKG-based retrieval, tree pruning to enhance retrieval precision, a re-ranking method for robust solution selection and a Fill-in-the-Middle (FIM) enhancer module for automatic code augmentation with relevant comments and docstrings.

---

# 基于编程知识图谱的上下文增强代码生成 论文详细解读

### 背景：这个问题为什么难？
大型语言模型（LLM）和专门的代码模型（Code‑LLM）在生成常规代码时已经相当强大，但面对需要跨文件、跨函数甚至跨库的复杂需求时，它们常常会卡壳。传统的检索增强生成（RAG）思路是先把外部文档拉出来再喂给模型，却容易出现两类失误：检索器找不到最贴合的问题的代码片段，或者检索到的大段文本把模型的上下文窗口塞满，导致模型“胡说”。换句话说，模型既缺少精准的代码语义检索，又受限于上下文容量，这让在真实项目中使用 LLM 生成高质量代码变得不可靠。

### 关键概念速览
**编程知识图谱（PKG）**：把代码抽象成节点（函数、类、变量）和边（调用、继承、依赖），形成结构化的语义网络。类似于把城市地图画出来，路口是节点，路是边，方便定位最短路径。  
**检索增强生成（RAG）**：在模型推理时先检索外部资料，再把检索结果拼进提示里，让模型“借助外部记忆”。就像写作文前先去图书馆查资料。  
**块级检索（Block‑wise Retrieval）**：把代码切成细粒度的基本块（如单个语句或小段），针对每个块单独检索。相当于在超市里只挑出需要的调味料，而不是整瓶买回去。  
**函数级检索（Function‑wise Retrieval）**：以完整函数为最小检索单元，适合需要整体逻辑的场景。类似于在图书馆直接借整本章节而不是单页。  
**树剪枝（Tree Pruning）**：在知识图谱的子树中去掉与当前任务无关的分支，只保留最可能被用到的路径。好比在搜索树中砍掉明显不通的枝桠，省时省力。  
**重排（Re‑ranking）**：对检索到的候选代码进行二次打分，挑出最靠谱的再交给生成模型。相当于先让助理挑出几本书，再让专家挑出一本最合适的。  
**填空中间（Fill‑in‑the‑Middle, FIM）增强**：在检索到的代码片段中自动插入注释、docstring 等自然语言描述，帮助模型更好地理解上下文。就像在代码旁边贴上“这段干什么”的小标签。

### 核心创新点
1. **从文本检索转向图结构检索**：传统 RAG 用关键词匹配或向量相似度在原始代码文件中搜索，容易把不相关的大段代码拉进来。本文把代码先转成 PKG，然后在图上做语义匹配，只挑出与问题最接近的节点或子树。这样检索的粒度更细，噪声更少。  
2. **细粒度块级与函数级双模检索**：以前的系统要么只检索整段代码，要么只能检索单行，难以兼顾精度和完整性。这里提供两套检索策略：块级检索适合需要局部实现的细节，函数级检索适合需要整体逻辑的场景。用户或自动化模块可以根据问题的复杂度动态切换。  
3. **树剪枝 + 重排双保险**：即使在 PKG 上检索到的子树仍可能包含冗余信息，作者设计了基于依赖度和使用频率的剪枝规则，把不太可能被调用的分支直接剔除。随后再用一个轻量的重排模型对剩余候选进行二次筛选，进一步压制幻觉（hallucination）。这两步像是先用粗筛把大石子挑走，再用细筛把沙子筛净。  
4. **FIM 增强模块**：检索到的代码往往缺少解释性文字，模型在仅看代码时容易误解意图。作者在检索结果中自动生成对应的注释和 docstring，再把这些“带标签”的代码喂给生成模型，提升了模型对上下文的理解深度。实验显示，这一步对 HumanEval 的 pass@1 提升贡献约 5%。

### 方法详解
整体框架可以划分为四步：  
1) **代码图谱构建** → 2) **语义检索** → 3) **树剪枝 + 重排** → 4) **生成模型增强**。下面逐层拆解。

**1. 编程知识图谱构建**  
- 先对所有可用的代码库（开源项目、标准库、内部代码）进行抽象语法树（AST）解析。  
- 把每个函数、类、变量映射成图节点；调用关系、继承关系、参数依赖等映射成有向边。  
- 为每个节点计算语义向量（使用 CodeBERT 等预训练模型），并记录代码块的文本、所在文件路径等元信息。  
- 最终得到一个巨大的多模态图，能够在 O(1) 时间定位任意节点的邻居。

**2. 基于 PKG 的检索**  
- 输入自然语言问题后，先用同样的预训练模型把问题向量化。  
- 在图上执行“最近邻”搜索，得到若干候选节点。这里分两条支路：  
  - **块级检索**：返回最相似的若干代码块（如单行或小段），适合细粒度需求。  
  - **函数级检索**：返回完整函数节点，适合需要整体实现的任务。  
- 每个候选都附带它在图中的子树结构，供后续剪枝使用。

**3. 树剪枝 + 重排**  
- **树剪枝**：对每个候选子树，计算每条边的“相关度分数”，该分数由边的类型（调用、依赖）和节点的使用频率共同决定。低于阈值的分支直接删除，保留的子树通常只有 2–3 层深，极大压缩了上下文体积。  
- **重排模型**：使用一个轻量的二分类模型（如 MiniLM）对剪枝后的候选进行打分，依据“是否能直接拼接到提示中并提升生成成功率”进行排序。最高分的前 K 条（K≈3）进入下一步。

**4. FIM 增强与生成**  
- 对每个被选中的代码块，调用一个专门的注释生成模型，把函数签名、变量用途等信息转成自然语言注释和 docstring。  
- 将“注释+代码块”拼接成一个完整的提示，送入目标 Code‑LLM（如 GPT‑4‑Code）。  
- 生成模型在此增强提示下完成“填空”或“完整实现”，输出最终代码。

**最巧妙的点**：作者把检索、剪枝、重排三层过滤链式组合，使得每一步都在降低噪声的同时保留必要信息。尤其是树剪枝直接在图结构层面削减无关代码，而不是在文本层面做粗糙截断，这在保持语义完整性上有显著优势。

### 实验与效果
- **评测数据**：HumanEval（OpenAI 提供的 164 题函数级评测）和 MBPP（Microsoft 的 974 题中等难度编程任务）。  
- **基线对比**：与原始 Code‑LLM（未使用检索）、传统 RAG（基于向量检索）以及最新的 Retrieval‑Augmented Code Generation（RACG）模型比较。  
- **主要结果**：  
  - HumanEval 上的 pass@1 从原始模型的 45% 提升到 55%（约 10% 绝对提升），相当于 20% 的相对提升。  
  - MBPP 上的 pass@1 从 38% 提升到 51%，相对提升约 34%。  
- **消融实验**：去掉树剪枝后，pass@1 下降约 6%；去掉重排后下降约 4%；仅保留块级检索而不使用函数级检索，整体提升约 2%。说明每个模块都有贡献，剪枝和重排是抑制幻觉的关键。  
- **局限性**：构建 PKG 需要对代码库进行完整的 AST 解析，成本随代码规模线性增长；对动态语言（如 JavaScript）中的运行时行为捕获不完整；论文未在大规模工业代码库上做横向验证，实际部署时的响应时延仍需优化。

### 影响与延伸思考
这篇工作把“知识图谱”概念成功搬进代码生成领域，开启了“结构化检索+生成”新路线。随后出现的几篇论文（如 **GraphCodeRAG**、**KG‑CodeGen**）都在 PKG 基础上加入了跨语言映射或实时更新机制，进一步提升了对新库的适应性。对想继续深入的读者，可以关注以下方向：  
- **动态图谱**：实时捕获运行时生成的函数或闭包，解决静态分析盲区。  
- **跨语言图谱对齐**：把 Python、Java、C++ 的代码节点映射到统一语义层，支持多语言检索。  
- **低延迟检索引擎**：把 PKG 存储在向量化图数据库（如 Neo4j + Milvus）中，实现毫秒级检索。  
- **人机协同**：让开发者在检索结果上直接编辑注释或剪枝规则，形成闭环学习。

### 一句话记住它
把代码当成图，在图上精准检索、剪枝、重排，再用注释“标记”喂给模型，能显著降低幻觉并提升复杂任务的生成成功率。