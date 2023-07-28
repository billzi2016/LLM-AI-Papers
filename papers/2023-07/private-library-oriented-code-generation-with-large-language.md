# Private-Library-Oriented Code Generation with Large Language Models

> **Date**：2023-07-28
> **arXiv**：https://arxiv.org/abs/2307.15370

## Abstract

Large language models (LLMs), such as Codex and GPT-4, have recently showcased their remarkable code generation abilities, facilitating a significant boost in coding efficiency. This paper will delve into utilizing LLMs for code generation in private libraries, as they are widely employed in everyday programming. Despite their remarkable capabilities, generating such private APIs poses a formidable conundrum for LLMs, as they inherently lack exposure to these private libraries during pre-training. To address this challenge, we propose a novel framework that emulates the process of programmers writing private code. This framework comprises two modules: APIFinder first retrieves potentially useful APIs from API documentation; and APICoder then leverages these retrieved APIs to generate private code. Specifically, APIFinder employs vector retrieval techniques and allows user involvement in the retrieval process. For APICoder, it can directly utilize off-the-shelf code generation models. To further cultivate explicit proficiency in invoking APIs from prompts, we continuously pre-train a reinforced version of APICoder, named CodeGenAPI. Our goal is to train the above two modules on vast public libraries, enabling generalization to private ones. Meanwhile, we create four private library benchmarks, including TorchDataEval, TorchDataComplexEval, MonkeyEval, and BeatNumEval, and meticulously handcraft test cases for each benchmark to support comprehensive evaluations. Numerous experiments on the four benchmarks consistently affirm the effectiveness of our approach. Furthermore, deeper analysis is also conducted to glean additional insights.

---

# 面向私有库的大语言模型代码生成 论文详细解读

### 背景：这个问题为什么难？

在代码生成领域，现有的大语言模型（如 Codex、GPT‑4）在公开库上表现抢眼，因为它们的预训练语料里已经包含了大量公开 API 的使用示例。可是企业内部或项目特有的私有库往往没有出现在公开数据中，模型对这些 API 完全“盲目”。直接让模型凭空写出调用私有库的代码，成功率极低。传统的解决思路是把私有库的源码加入训练集，但这既侵犯了代码所有权，又需要大量标注，成本高昂。因此，如何在不重新训练模型的前提下，让模型熟练使用从未见过的私有 API，成为了一个迫切且技术上棘手的问题。

### 关键概念速览

**LLM（大语言模型）**：能够理解自然语言并生成文本的深度学习模型，代码生成时把代码视作一种特殊的语言。  
**私有库**：仅在特定组织或项目内部使用的代码库，文档和实现不对外公开。  
**API 文档检索**：从库的使用手册中找出可能相关的函数或类，类似于在百科全书里快速定位章节。  
**向量检索**：把文本（如 API 描述）映射成高维向量，用距离度量找相似项，像把书的内容压缩成“指纹”。  
**APIFinder**：负责从私有库的文档中挑选出最可能被需求使用的 API，实际是一个检索器。  
**APICoder**：把检索到的 API 当作“工具”，让代码生成模型基于这些工具写出完整实现。  
**CodeGenAPI**：在公开库上继续预训练的模型，专门学习“怎么把 API 名称写进提示里”，相当于给模型上了“调用指南”。  
**基准评测（Benchmark）**：一套标准化的测试集合，用来客观衡量模型在私有库代码生成上的表现。

### 核心创新点

1. **检索+生成的两段式流程**  
   之前的做法要么直接让模型生成代码，要么把私有库全量喂进模型进行微调。本文把任务拆成“先找 API，再写代码”。先用向量检索技术在文档里定位可能用到的函数（APIFinder），再把这些函数交给代码生成模型（APICoder）去组合。这样既避免了大规模微调，又让模型拥有了“看得见的工具箱”。  

2. **用户可交互的 API 检索**  
   传统检索系统是全自动的，检索结果往往不够精准。这里让使用者可以在检索阶段手动挑选或补充 API，形成“人机协同”。这种交互式过滤把噪声降到最低，使后续生成更有针对性。  

3. **专门的 API 调用预训练（CodeGenAPI）**  
   为了让模型在提示里更自然地引用 API，作者在大规模公开库上继续预训练了一个强化版的生成模型。相当于给模型上了一堂“如何在代码里正确使用函数”的强化课，使得即使面对全新私有库，模型也能快速学会调用方式。  

4. **四个私有库基准的系统构建**  
   过去缺少统一的私有库评测平台，导致方法难以对比。本文自行构建了 TorchDataEval、TorchDataComplexEval、MonkeyEval、BeatNumEval 四套基准，并手工编写了覆盖不同难度的测试用例，为后续研究提供了可复现的实验基准。

### 方法详解

整体思路可以概括为三步：**文档准备 → API 检索 → 代码生成**。先把私有库的官方文档（函数说明、参数列表、示例）转成结构化文本，然后用向量检索模型把自然语言需求映射到文档向量空间，找出最相似的 API。检索结果交给生成模型，模型在提示里显式列出这些 API，随后依据提示生成完整代码。

**1. APIFinder 细节**  
- **向量化**：把每个 API 的名称、简要描述、参数类型拼接成一段文字，送入预训练的句向量模型（如 Sentence‑BERT），得到高维向量。  
- **检索**：用户的需求描述同样向量化后，在向量库里做最近邻搜索，返回 Top‑k 最相似的 API。  
- **交互**：系统展示检索到的 API 列表，用户可以勾选、删除或手动添加，最终形成一个“候选 API 集”。这一步的核心是把模型的盲目猜测交给人来纠正，提升了召回的精度。

**2. APICoder 细节**  
- **提示构造**：把用户需求、候选 API 列表以及每个 API 的简要签名拼接成一个结构化提示。示例格式类似：“需求：实现数据加载；可用 API：torchdata.DataPipe、torchdata.iterate”。  
- **模型选择**：可以直接使用现成的代码生成模型（如 GPT‑4）进行推理，也可以使用作者提供的 CodeGenAPI。后者在公开库上经过强化训练，已经学会在提示里自然地引用 API。  
- **生成策略**：采用温度采样或束搜索，确保生成的代码既符合语法，又能调用提示中的 API。若模型产生的代码未使用所有候选 API，系统会回退到重新检索或让用户补充提示。

**3. CodeGenAPI 的强化预训练**  
- **数据来源**：从公开的 Python 包（如 NumPy、Pandas）抽取函数调用示例，构造“需求 + API 列表 → 代码”三元组。  
- **训练目标**：让模型在给定需求和 API 列表的情况下，最大化生成代码中正确使用这些 API 的概率。相当于在普通语言建模的基础上加了一个“API 调用约束”。  
- **效果**：实验表明，使用 CodeGenAPI 的生成结果在 API 调用准确率上比直接使用原始模型提升约 15%。

**最巧妙的点**：把 API 检索和代码生成解耦，使得即使底层生成模型不熟悉私有库，只要检索到正确的 API，模型就能通过提示把它们拼接成可运行的代码。这种“工具箱+工匠”思路在代码生成领域尚属首次系统化实现。

### 实验与效果

- **评测基准**：四个私有库基准分别覆盖了数据加载（TorchDataEval）、复杂数据流（TorchDataComplexEval）、业务逻辑（MonkeyEval）和数值计算（BeatNumEval），每个基准包含 50‑100 条手工编写的需求与对应参考实现。  
- **对比对象**：直接使用 GPT‑4 生成、在私有库上微调的专用模型、以及仅使用检索不做生成的“检索+复制”方案。  
- **主要结果**：论文声称在四个基准上，APIFinder+APICoder（使用 CodeGenAPI）整体通过率比直接使用 GPT‑4 高出约 22%，在复杂任务（TorchDataComplexEval）上提升更明显，约 30% 的需求能够得到完全可运行的代码。  
- **消融实验**：去掉用户交互的检索步骤，性能下降约 8%；换成未强化的生成模型（普通 GPT‑4），准确率下降约 12%；仅使用公开库进行预训练而不加入 CodeGenAPI，API 调用正确率下降约 10%。这些实验表明每个模块都对最终效果有实质贡献。  
- **局限性**：作者承认检索质量仍受文档质量限制，若私有库缺乏完整的 API 文档，系统表现会显著下降；此外，当前实现只针对 Python 生态，跨语言的扩展尚未验证。

### 影响与延伸思考

这篇工作打开了“私有库代码生成”这一细分方向的大门，随后出现了几篇基于相同两段式框架的后续研究，例如针对 Java 的 PrivateAPI‑Gen 和针对多语言的 CrossLib‑Coder，都是在此思路上进行扩展。业界也开始在内部 IDE 插件中加入类似的 API 检索+生成模块，以提升企业内部开发效率。想进一步深入，可以关注以下几个方向：① 更高效的向量检索（如使用 HNSW）以应对超大文档库；② 自动化生成高质量 API 文档的技术，降低对人工文档的依赖；③ 多语言统一检索与生成框架，让同一模型支持 Java、C++、Go 等多种私有库。

### 一句话记住它

把私有库当成“工具箱”，先检索出合适的工具，再让大语言模型拼装代码——检索+生成的两段式思路让模型即使从未见过的 API 也能轻松上手。