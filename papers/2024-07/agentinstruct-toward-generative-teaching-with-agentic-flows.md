# AgentInstruct: Toward Generative Teaching with Agentic Flows

> **Date**：2024-07-03
> **arXiv**：https://arxiv.org/abs/2407.03502

## Abstract

Synthetic data is becoming increasingly important for accelerating the development of language models, both large and small. Despite several successful use cases, researchers also raised concerns around model collapse and drawbacks of imitating other models. This discrepancy can be attributed to the fact that synthetic data varies in quality and diversity. Effective use of synthetic data usually requires significant human effort in curating the data. We focus on using synthetic data for post-training, specifically creating data by powerful models to teach a new skill or behavior to another model, we refer to this setting as Generative Teaching. We introduce AgentInstruct, an extensible agentic framework for automatically creating large amounts of diverse and high-quality synthetic data. AgentInstruct can create both the prompts and responses, using only raw data sources like text documents and code files as seeds. We demonstrate the utility of AgentInstruct by creating a post training dataset of 25M pairs to teach language models different skills, such as text editing, creative writing, tool usage, coding, reading comprehension, etc. The dataset can be used for instruction tuning of any base model. We post-train Mistral-7b with the data. When comparing the resulting model Orca-3 to Mistral-7b-Instruct (which uses the same base model), we observe significant improvements across many benchmarks. For example, 40% improvement on AGIEval, 19% improvement on MMLU, 54% improvement on GSM8K, 38% improvement on BBH and 45% improvement on AlpacaEval. Additionally, it consistently outperforms other models such as LLAMA-8B-instruct and GPT-3.5-turbo.

---

# AgentInstruct：面向生成式教学的代理流框架 论文详细解读

### 背景：这个问题为什么难？

合成数据已经成为加速大语言模型（LLM）研发的关键手段，但它的质量参差不齐。传统做法往往让强大的模型直接模仿已有数据，结果容易出现“模型坍塌”：生成的文本缺乏多样性、创新性甚至出现错误。要让合成数据真正提升下游模型，需要人工筛选、标注，这在规模上几乎不可行。于是，如何在不大量投入人工成本的前提下，自动生成既多样又高质量的教学数据，成为制约生成式教学（Generative Teaching）的大难题。

### 关键概念速览
- **生成式教学（Generative Teaching）**：用强模型生成的指令‑响应对来“教”另一模型新技能，类似老师用自编教材给学生上课。  
- **Agentic Flow（代理流）**：把多个智能体（agent）串联成流水线，每个智能体负责特定子任务，像装配线上的工人各司其职。  
- **Prompt Generation（提示生成）**：让模型自行创造出提问或任务描述，相当于让老师先想出课堂练习题。  
- **Response Generation（响应生成）**：在给定提示下让模型输出答案或行为，类似学生完成作业。  
- **Instruction Tuning（指令微调）**：在已有的基础模型上继续训练，使其更好地遵循自然语言指令。  
- **Synthetic Data Diversity（合成数据多样性）**：指生成的数据在主题、风格、难度等维度上的丰富程度，类似课堂上不同类型的练习题。  
- **Model Collapse（模型坍塌）**：合成数据质量低导致模型输出单调、错误率升高的现象，就像学生只会背答案而不懂思考。  

### 核心创新点
1. **从原始文档到完整指令对的全自动流水线**  
   之前的合成数据大多依赖人工编写提示或使用单一模型一次性生成问答。AgentInstruct 把“提示生成”和“答案生成”拆成两个独立的智能体，并在两者之间加入过滤、重写等环节，形成端到端的代理流。这样既提升了数据的多样性，又让每一步都有专门的质量控制。

2. **只需原始文本/代码种子，无需人工标注**  
   传统方法往往需要先准备好标注好的指令对作为模板。AgentInstruct 只要求提供未加工的文档或代码文件，系统会自动抽取关键概念、生成任务描述，再让强模型完成回答，实现了“零标注”合成。

3. **大规模合成 2500 万对指令数据并直接用于后训练**  
   过去的合成数据规模通常在几百万甚至更少，受限于计算成本或质量控制。作者通过高效的代理流并行化，成功生成 2500 万对多技能指令数据，覆盖编辑、创作、工具使用、编程、阅读理解等多种场景，直接用于对 Mistral‑7B 的后训练。

4. **显著提升多项基准，验证“教学”有效性**  
   与同基座的 Mistral‑7B‑Instruct 对比，使用 AgentInstruct 合成数据微调得到的 Orca‑3 在 AGIEval、MMLU、GSM8K、BBH、AlpacaEval 等基准上分别提升 40%、19%、54%、38%、45%。这表明自动生成的教学数据能够真正让模型学到新技能，而不是简单的复制。

### 方法详解
**整体框架**  
AgentInstruct 把合成指令对的过程划分为四大阶段：种子抽取、提示生成、响应生成、质量过滤。每一阶段都是一个独立的智能体，前后通过统一的 JSON 接口传递信息，形成一条“代理流”。整个流水线可以在多 GPU 上并行运行，极大提升吞吐量。

**1. 种子抽取（Seed Extraction）**  
输入是原始文本或代码文件。抽取智能体使用轻量的检索模型（如 MiniLM）扫描文档，挑选出句子、函数、注释等信息密集的片段作为“种子”。这些种子相当于课堂上老师挑选的教材章节。

**2. 提示生成（Prompt Agent）**  
提示智能体基于强大的 LLM（如 GPT‑4）接收种子并生成多样的任务描述。它会随机选择任务类型（编辑、翻译、代码实现等），并在提示中加入变体（不同难度、不同风格），从而实现数据的多样性。这里的关键技巧是让模型在“任务空间”上进行采样，而不是固定模板。

**3. 响应生成（Response Agent）**  
响应智能体同样使用强模型，但在提示的约束下生成答案或行为。为了防止模型直接复制种子内容，作者在提示中加入“禁止直接引用”之类的指令，并让模型进行推理或创作。此阶段还会调用工具调用能力（如代码解释器）来完成更复杂的任务。

**4. 质量过滤（Filter Agent）**  
过滤智能体负责检查生成的对是否符合质量标准。它会评估答案的完整性、逻辑一致性、是否出现违禁内容等。若不合格，流水线会回到提示生成或响应生成环节重新生成。过滤策略采用多模型投票和规则检查相结合的方式，确保最终数据既多样又可靠。

**巧妙之处**  
- **代理流的模块化**：每个智能体都可以单独升级或替换，例如把 Prompt Agent 换成更强的模型，只需保持接口不变。  
- **自适应采样**：Prompt Agent 会根据过滤结果的反馈动态调整任务类型的采样概率，类似老师根据学生的错误率调整练习题难度。  
- **并行批处理**：种子抽取后，提示生成和响应生成可以在数千个并行实例上同时进行，显著降低了生成 2500 万对数据的时间成本。

### 实验与效果
- **数据规模**：使用公开的文本语料和开源代码库，AgentInstruct 生成了约 2500 万对指令‑响应数据，覆盖 10+ 技能类别。  
- **后训练模型**：在 Mistral‑7B 基础上进行 2‑3 个 epoch 的指令微调，得到模型 Orca‑3。  
- **基准评测**：  
  - AGIEval（通用人工智能评估）提升约 40%。  
  - MMLU（多任务语言理解）提升约 19%。  
  - GSM8K（数学推理）提升约 54%。  
  - BBH（大语言模型基准）提升约 38%。  
  - AlpacaEval（指令遵循）提升约 45%。  
- **对比基线**：与同基座的 Mistral‑7B‑Instruct、LLAMA‑8B‑Instruct、GPT‑3.5‑Turbo 等模型相比，Orca‑3 在多数指标上均保持领先。  
- **消融实验**：论文中展示了去掉过滤模块、仅使用单一智能体生成等设置的性能下降，说明每个代理流环节都对最终质量有贡献。  
- **局限性**：合成数据仍然依赖于强模型的能力，若底层模型出现偏见或错误，可能会被放大。作者也提到在极端专业领域（如医学、法律）仍需人工审校。

### 影响与延伸思考
AgentInstruct 把“自动生成教学材料”提升到可以大规模、低成本实现的层次，打开了后训练（post‑training）新范式。随后的工作开始探索更细粒度的代理流（例如加入检索‑增强、对抗过滤），以及把合成数据与真实数据混合训练的策略。对想进一步研究的读者，可以关注以下方向：  
- **自适应代理流调度**：让系统根据实时质量反馈自动分配计算资源。  
- **跨模态合成**：把文本、图像、音频等多模态种子一起纳入代理流，生成多模态指令对。  
- **安全与偏见控制**：在过滤阶段加入更细致的伦理审查模型，防止有害内容的系统性生成。  

### 一句话记住它
AgentInstruct 用一条可拆解、可并行的“代理流水线”，把原始文档自动转化为大规模高质量指令数据，让模型通过“自教自学”显著提升多项能力。