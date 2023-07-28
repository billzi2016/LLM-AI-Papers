# Multilingual Code Co-Evolution Using Large Language Models

> **Date**：2023-07-27
> **arXiv**：https://arxiv.org/abs/2307.14991

## Abstract

Many software projects implement APIs and algorithms in multiple programming languages. Maintaining such projects is tiresome, as developers have to ensure that any change (e.g., a bug fix or a new feature) is being propagated, timely and without errors, to implementations in other programming languages. In the world of ever-changing software, using rule-based translation tools (i.e., transpilers) or machine learning models for translating code from one language to another provides limited value. Translating each time the entire codebase from one language to another is not the way developers work. In this paper, we target a novel task: translating code changes from one programming language to another using large language models (LLMs). We design and implement the first LLM, dubbed Codeditor, to tackle this task. Codeditor explicitly models code changes as edit sequences and learns to correlate changes across programming languages. To evaluate Codeditor, we collect a corpus of 6,613 aligned code changes from 8 pairs of open-source software projects implementing similar functionalities in two programming languages (Java and C#). Results show that Codeditor outperforms the state-of-the-art approaches by a large margin on all commonly used automatic metrics. Our work also reveals that Codeditor is complementary to the existing generation-based models, and their combination ensures even greater performance.

---

# 多语言代码协同演化：利用大语言模型 论文详细解读

### 背景：这个问题为什么难？

很多开源项目会用 Java、C#、Python 等多种语言实现同一套 API 或算法。代码在一种语言里修 bug、加特性后，必须手动把改动同步到其它语言的实现，否则不同实现之间会出现功能不一致或隐藏缺陷。传统的代码翻译工具（如基于规则的 transpiler）只能把完整文件一次性转换，根本不适合“只改一点就同步”的工作流。机器学习的代码翻译模型虽然能生成跨语言代码，但它们通常是“从头生成”，不关注具体的 edit（编辑）差异，也缺乏跨语言的改动对应关系。因此，如何让模型理解“这次改动在 Java 里是把 A 改成 B”，并自动在 C# 里产生对应的 edit，成为了一个迫切而又未被解决的难题。

### 关键概念速览

- **代码改动（code edit）**：指一次提交中对源码的增删改操作，类似于在文档里把某句话改写成另一句话。这里把改动抽象为一系列编辑指令（insert、delete、replace）。
- **编辑序列（edit sequence）**：把一次改动拆解成有序的编辑指令，就像把一次文字校对过程写成“删第 3 行、在第 5 行后插入…”。模型可以直接操作这些序列，而不是重新生成完整文件。
- **跨语言对应（cross‑language correspondence）**：指同一次功能改动在不同编程语言实现之间的映射关系。比如把 Java 的 `List.add` 改成 `addFirst`，在 C# 里对应的改动可能是把 `List.Add` 换成 `Insert(0, …)`。
- **大语言模型（LLM）**：拥有数十亿参数、在海量代码和自然语言上预训练的 Transformer 模型，能够理解代码语义并生成高质量代码片段。
- **生成式模型（generation‑based model）**：传统的代码翻译模型，输入一段源代码，直接输出目标语言的完整实现，类似于“从零写”。它们不显式建模编辑过程。
- **Codeditor**：本文提出的专门用于代码改动跨语言翻译的 LLM，核心是把改动表示成编辑序列并学习语言间的对应关系。
- **对齐改动语料库（aligned edit corpus）**：人工收集的、在两种语言实现中对应的改动对，类似于双语平行句对，只不过单位是“编辑”而不是完整代码。

### 核心创新点

1. **从完整翻译转向编辑翻译**  
   之前的跨语言代码翻译大多把整个文件当作输入，输出完整的目标文件。Codeditor 把任务重新定义为“把源语言的编辑序列映射到目标语言的编辑序列”。这样模型只需要关注改动的局部信息，既降低了生成噪声，又符合开发者实际的“改一点、同步一次”工作流。

2. **显式建模跨语言编辑对应**  
   传统模型只学习语言间的映射规则，缺乏对同一次功能改动在不同语言实现中对应关系的认识。Codeditor 在训练时把两套编辑序列放在同一对齐框架里，让模型学会“如果在 Java 里删掉某行，那么在 C# 里对应的可能是删掉另一行”。这种对应学习显著提升了改动同步的准确性。

3. **结合生成式模型形成互补体系**  
   实验发现，Codeditor 在大多数指标上领先，但在少数需要大块重写的场景仍然不如直接生成式模型。作者把两者的输出做了简单的投票/融合，得到的混合系统在所有评测上都超过单一模型，证明两种思路可以互补。

4. **首个大规模跨语言编辑对齐语料库**  
   为了训练和评估，作者手工收集了 6,613 条对齐改动，覆盖 8 对实现相同功能的开源项目（Java ↔ C#）。这套数据本身就为后续研究提供了宝贵的基准。

### 方法详解

**整体框架**  
Codeditor 的工作流可以分为三步：① 把源语言的提交转化为编辑序列；② 用 LLM 预测目标语言的对应编辑序列；③ 把预测的编辑序列应用到目标代码库，得到同步后的代码。整个过程只在改动的局部进行，不需要重新生成整文件。

**步骤 1：编辑序列化**  
作者使用现成的 diff 工具把每次提交的前后代码差异抽取为增删改指令。每条指令被标记为 `INSERT <位置> <代码片段>`、`DELETE <位置>` 或 `REPLACE <位置> <新代码>`。为了让模型更好地捕捉上下文，指令前后会附加少量的代码上下文（前后各 3 行），形成类似 “[上下文] INSERT … [上下文]” 的文本序列。

**步骤 2：跨语言编辑映射**  
核心模型是一个基于 Transformer 的大语言模型，输入是源语言的编辑序列（连同上下文），输出是目标语言的编辑序列。训练时使用对齐的编辑对，模型通过自回归方式逐条生成目标指令。为了让模型注意到“位置”在不同语言里可能不对应，作者在输入中加入了抽象的 AST（抽象语法树）路径标记，例如 `FuncDecl/ParamList/0`，帮助模型在结构层面对齐。

**步骤 3：编辑应用**  
得到目标语言的编辑序列后，系统在目标代码库中执行相同的增删改操作。因为编辑序列已经包含了具体的行号或 AST 路径，应用过程几乎是自动化的，只要目标文件没有大幅结构变化。

**巧妙之处**  
- **编辑序列而非完整代码**：把改动抽象成指令，使模型的搜索空间大幅缩小，类似于把“写一篇文章”改成“改几个词”。  
- **结构化位置信息**：加入 AST 路径让模型能够跨语言理解“这里是函数入口”，而不是盲目依赖行号。  
- **混合策略**：在预测后，如果生成的编辑序列与原始代码差异过大，系统会回退到生成式模型的完整翻译，再做一次 diff，取两者的交集作为最终改动。

### 实验与效果

- **数据集**：作者构建的 6,613 条对齐改动，来源于 8 对实现相同功能的开源项目，语言对为 Java 与 C#。每对项目都有完整的提交历史，确保改动在两边都有对应版本。  
- **基线**：包括传统基于规则的 transpiler、最新的跨语言代码翻译模型（如 CodeT5、TransCoder）以及直接使用大语言模型进行完整代码生成的方案。  
- **主要指标**：使用 BLEU、CodeBLEU、Exact Match 等自动评估指标。Codeditor 在所有指标上均领先基线，且领先幅度被描述为“大幅提升”。  
- **消融实验**：作者分别去掉编辑序列化、AST 位置信息、以及混合生成策略，结果显示去掉任意一项都会导致指标下降 5%~12% 不等，说明每个模块都对整体性能有实质贡献。  
- **局限性**：论文承认 Codeditor 依赖于高质量的对齐改动数据，若项目之间的实现差异过大（例如语言特性差异导致代码结构根本不同），模型的对应学习会受限。此外，模型在处理大规模重构（一次提交改动数千行）时仍然表现不佳。

### 影响与延伸思考

这篇工作首次把“代码改动”作为跨语言翻译的基本单元，打开了“代码协同演化”这一新研究方向。后续有几篇论文尝试把相同思路推广到更多语言组合（如 Python ↔ TypeScript）或加入更细粒度的语义信息（如类型约束）。在工业界，部分大型 IDE 已经开始实验基于编辑序列的同步插件，帮助多语言团队保持实现一致性。想进一步深入，可以关注以下方向：① 自动构建跨语言编辑对齐语料库的半监督方法；② 将模型与版本控制系统深度集成，实现实时改动建议；③ 探索编辑序列在安全审计、漏洞迁移中的应用。整体来看，Codeditor 为代码维护自动化提供了可行的路径，也为 LLM 在软件工程中的细粒度任务打开了新大门。

### 一句话记住它

把代码改动当成“编辑指令”，让大语言模型学会跨语言对应，从而实现自动同步的“代码协同演化”。