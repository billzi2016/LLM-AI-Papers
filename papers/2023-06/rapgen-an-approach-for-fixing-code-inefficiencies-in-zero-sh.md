# RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot

> **Date**：2023-06-29
> **arXiv**：https://arxiv.org/abs/2306.17077

## Abstract

Performance bugs are non-functional bugs that can even manifest in well-tested commercial products. Fixing these performance bugs is an important yet challenging problem. In this work, we address this challenge and present a new approach called Retrieval-Augmented Prompt Generation (RAPGen). Given a code snippet with a performance issue, RAPGen first retrieves a prompt instruction from a pre-constructed knowledge-base of previous performance bug fixes and then generates a prompt using the retrieved instruction. It then uses this prompt on a Large Language Model (such as Codex) in zero-shot to generate a fix. We compare our approach with the various prompt variations and state of the art methods in the task of performance bug fixing. Our evaluation shows that RAPGen can generate performance improvement suggestions equivalent or better than a developer in ~60% of the cases, getting ~42% of them verbatim, in an expert-verified dataset of past performance changes made by C# developers.

---

# RAPGen：零样本代码性能缺陷修复方法 论文详细解读

### 背景：这个问题为什么难？

性能缺陷（performance bugs）往往不影响功能正确性，却会导致程序运行慢、资源浪费，甚至在大规模系统里酿成灾难。传统的性能调优依赖经验丰富的工程师手工分析，成本高且难以系统化。已有的自动化修复方法大多基于“生成‑测试”循环，需要大量的运行时评估或专门的性能基准，导致修复过程慢且不易迁移到新语言或新库。更关键的是，现有的语言模型提示（prompt）往往是通用的，缺少针对性能问题的专业指引，导致零样本（zero‑shot）情况下的修复质量不稳定。于是，如何在不进行大量运行时实验的前提下，让大模型直接给出高质量的性能改进建议，成为亟待突破的难点。

### 关键概念速览
- **性能缺陷（performance bug）**：代码在功能上没有错误，但执行效率低下，例如不必要的循环或资源泄漏。类似于跑步时穿了沉重的鞋子，跑得对，却慢得不合理。  
- **零样本（zero‑shot）**：模型在没有看到任何同类任务示例的情况下直接完成任务。就像让一个从未见过钢琴的画家即兴演奏。  
- **提示（prompt）**：给大语言模型的输入指令，决定模型的思考方向。相当于在对话前给出一段背景信息，让对方更容易回答。  
- **检索增强（retrieval‑augmented）**：在生成答案前先从外部知识库中找出相关案例或指令，再把它们拼进提示里。好比写作文前先查阅参考文献，提升内容的专业度。  
- **知识库（knowledge base）**：预先收集的历史性能修复案例及对应的修复指令。可以想象成一本“性能调优手册”。  
- **大型语言模型（LLM）**：如 OpenAI Codex、GPT‑4 等，能够理解代码并生成代码的深度学习模型。它们的能力类似于拥有海量编程经验的“虚拟程序员”。  
- **指令生成（instruction generation）**：把检索到的案例转化为适用于当前代码的提示文本。相当于把参考文献的要点重新组织成自己的论点。  

### 核心创新点
1. **检索‑增强提示 → 直接生成 → 性能提升**  
   以前的零样本修复只靠手工编写通用提示，效果有限。RAPGen 先在专门的性能修复知识库里检索最相似的历史案例，然后把对应的修复指令拼进提示，最后交给 LLM 生成代码。这样模型得到的上下文更贴近性能调优，生成的修复质量显著提升。  

2. **从“案例‑指令”到“可执行提示”的自动化管线**  
   传统方法需要人工把历史提交信息转化为提示，工作量大且易出错。RAPGen 设计了一个自动化的指令生成模块：先用相似度搜索定位案例，再用模板把案例的关键改动抽象为指令，最后拼接成完整提示。该管线把人工经验编码成机器可读的指令，降低了人为偏差。  

3. **在零样本环境下实现接近人工水平的改进**  
   通过上述两步，RAPGen 在不进行任何微调或示例学习的前提下，能够让 LLM 产生与经验丰富的 C# 开发者相当的性能改进。实验显示约 60% 的代码片段得到的改进与人工相当，其中 42% 完全复现了原开发者的改动。  

### 方法详解
**整体框架**  
RAPGen 的工作流可以概括为三步：①检索、②指令生成、③提示驱动的代码生成。整个过程不需要对 LLM 进行额外训练，只是通过精心构造的提示让模型在零样本下发挥最大潜力。

**步骤一：检索**  
- 输入：待修复的代码片段 `C_target`。  
- 系统先把 `C_target` 用语义向量化（比如使用 CodeBERT），得到向量 `v_target`。  
- 在预先构建的性能修复知识库中，每条历史记录都有对应的向量 `v_i`。系统计算 `v_target` 与所有 `v_i` 的相似度，选出前 `k`（论文未明确给出具体数值）最相似的记录。  
- 这一步的核心是把“代码相似度”当作检索信号，确保找到的案例在算法结构或 API 使用上与目标代码相近。

**步骤二：指令生成**  
- 每条检索到的记录包含两部分：原始代码 `C_i` 与修复后代码 `C_i'`，以及开发者在提交信息里写的简短说明。  
- 系统对 `C_i` 与 `C_i'` 做差分，抽取出具体的改动（如“把 `foreach` 换成 `for` 循环并提前计算长度”）。  
- 这些改动被映射成统一的指令模板，例如：“将 `X` 替换为 `Y`，以减少 `Z`”。  
- 多条指令会被合并成一个“检索增强提示” `P_retrieved`，其结构类似：“基于以下历史改动：① … ② …，请对下面的代码进行性能优化”。  

**步骤三：提示驱动的代码生成**  
- 最终提示 `P_final` = `P_retrieved` + `C_target`（代码本体）。  
- 将 `P_final` 送入大型语言模型（如 Codex），使用零样本的 `completion` 接口，让模型直接输出改进后的代码。  
- 生成的代码随后可以交给静态分析或基准测试进行验证（论文中未详细描述验证环节，但在实验里使用了人工评审）。  

**巧妙之处**  
- **检索‑增强而非检索‑后置**：很多 RAG（检索增强生成）系统是先检索再让模型自行决定是否使用信息，RAPGen 把检索结果硬编码进提示，使模型几乎必然关注这些指令。  
- **指令抽象层**：直接把差分代码转成自然语言指令，而不是把完整的历史代码塞进去，降低了提示的噪声，提高了模型对关键改动的聚焦度。  

### 实验与效果
- **数据集**：作者收集了一个由 C# 开发者提交的性能改动历史组成的专家验证数据集，包含数千条真实的性能修复案例。  
- **评估方式**：对每条待修复代码，RAPGen 生成的改进与原开发者的改动进行人工对比，判断是否等价或更好。  
- **基线**：包括（1）直接使用通用提示的 Codex 零样本修复、（2）基于手工编写的性能提示、（3）最新的几种基于微调的代码修复模型。  
- **结果**：RAPGen 在约 60% 的案例中产生的改进被评审认为“等同或优于”人工修复，其中约 42% 完全复现了原开发者的改动。相比最强基线，等价或更好率提升约 15%~20%。  
- **消融实验**：作者分别去掉检索步骤、去掉指令抽象、仅使用原始案例代码进行提示，发现性能下降显著，验证了检索和指令生成的必要性。  
- **局限**：实验仅在 C# 代码上完成，跨语言的迁移效果未知；此外，生成的代码仍需人工或自动化测试确认其性能提升，论文未给出完整的自动评估流水线。  

### 影响与延伸思考
RAPGen 把“检索增强”直接搬到代码性能修复的零样本场景，展示了在缺少大量标注数据时，如何利用历史案例提升大模型的实用性。自发表后，已有工作尝试把相同思路扩展到安全漏洞修复、代码风格统一等方向，甚至把指令生成层级化为多步“思考链”。如果想进一步探索，可以关注以下几个方向：  
- **跨语言知识库构建**：如何在多语言代码库中统一检索表示。  
- **自动化性能验证**：把基准测试嵌入生成闭环，实现全自动的性能回归。  
- **指令模板学习**：让模型自行学习如何把差分抽象成自然语言指令，而不是手工设计模板。  

### 一句话记住它
把历史性能改动检索成指令，塞进提示，让大模型在零样本下直接写出等同于人工的性能优化代码。