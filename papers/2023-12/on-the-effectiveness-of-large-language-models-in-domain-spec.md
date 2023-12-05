# On the Effectiveness of Large Language Models in Domain-Specific Code   Generation

> **Date**：2023-12-04
> **arXiv**：https://arxiv.org/abs/2312.01639

## Abstract

Large language models (LLMs) such as ChatGPT have shown remarkable capabilities in code generation. Despite significant achievements, they rely on enormous training data to acquire a broad spectrum of open-domain knowledge. Besides, their evaluation revolves around open-domain benchmarks like HumanEval, which primarily consist of programming contests. Therefore, it is hard to fully characterize the intricacies and challenges associated with particular domains (e.g., web, game, and math). In this paper, we conduct an in-depth study of the LLMs in domain-specific code generation. Our results demonstrate that LLMs exhibit sub-optimal performance in generating domain-specific code, due to their limited proficiency in utilizing domain-specific libraries. We further observe that incorporating API knowledge as prompts can empower LLMs to generate more professional code. Based on these findings, we further investigate how to effectively incorporate API knowledge into the code generation process. We experiment with three strategies for incorporating domain knowledge, namely, external knowledge inquirer, chain-of-thought prompting, and chain-of-thought fine-tuning. We refer to these strategies as a new code generation approach called DomCoder. Experimental results show that all strategies of DomCoder lead to improvement in the effectiveness of domain-specific code generation under certain settings.

---

# 大语言模型在领域特定代码生成中的有效性 论文详细解读

### 背景：这个问题为什么难？

通用的大语言模型（LLM）在公开的代码生成基准上表现抢眼，但它们的训练数据主要是开源仓库和竞赛题目，缺少对特定行业库的深度覆盖。实际开发中，像 Web 框架、游戏引擎这类专业库的 API 使用方式非常细致，错误一次就可能导致编译或运行时崩溃。过去的评测几乎只看“写对一个函数”，忽视了对这些领域专有库的调用熟练度，导致模型在真实业务场景里常常“写得好看却用不对”。因此，评估并提升 LLM 在特定领域代码生成的能力成为亟待解决的难题。

### 关键概念速览
- **大语言模型（LLM）**：在海量文本上预训练的深度网络，能够根据提示生成自然语言或代码，类似会“写作文”的机器人。  
- **领域特定代码生成**：让模型输出使用特定行业库（如 Gin、Unreal Engine）的代码，而不是仅仅实现通用算法。可以比作厨师不仅要会做菜，还要熟练使用某家餐厅的专属调味料。  
- **API 知识提示（API Prompt）**：在给模型的输入里加入库的函数签名、使用示例等信息，帮助模型“查阅手册”。相当于在考试前给学生一本参考手册。  
- **外部知识查询器（External Knowledge Inquirer）**：模型在生成过程中主动向外部检索系统（如文档搜索）请求相关 API 信息，再把检索结果拼回提示中。像是程序员在写代码时打开浏览器搜索官方文档。  
- **思维链（Chain‑of‑Thought, CoT）**：让模型先列出解题思路或调用步骤，再正式写代码。类似先写草稿再落笔，能够让模型的推理过程更透明。  
- **思维链微调（CoT‑Fine‑Tuning）**：在带有思维链标注的训练数据上继续训练模型，使其自然形成“先思考后编码”的习惯。相当于给学生大量带有解题步骤的例题，让他们养成写草稿的好习惯。  
- **DomCoder**：本文提出的整体方案名称，集合了上述三种把领域 API 融入生成过程的策略。  

### 核心创新点
1. **从“纯提示”到“主动查询”**  
   之前的工作只在一次性提示里塞进 API 文档，往往因为篇幅或上下文限制导致信息不全。本文引入 **外部知识查询器**，让模型在生成过程中主动发起检索，把最新、最完整的 API 片段动态注入提示。这样模型不再受限于一次性提示的容量，生成的代码更贴合真实库的使用方式。  

2. **把思维链搬进代码生成**  
   传统的 CoT 主要用于数学或推理题目，代码生成领域少有尝试。作者把 **思维链提示** 应用于调用库的步骤，让模型先列出“先创建对象 → 再调用方法 → 最后返回结果”等逻辑链，再写出具体代码。实验显示，这种先思考后落笔的方式显著降低了 API 调用错误率。  

3. **思维链微调提升内部化**  
   仅靠提示仍然依赖模型的即时推理能力。作者在带有思维链标注的领域代码数据上继续微调模型，使其内部“记住”了思考过程。微调后模型在没有外部检索的情况下，也能自行生成完整的调用链，提升了在资源受限环境下的鲁棒性。  

4. **统一的 DomCoder 框架**  
   将上述三种策略统一包装成 **DomCoder**，并在同一实验平台上进行对比。框架的模块化设计让研究者可以灵活组合查询、思维链提示或微调，快速评估哪种方式最适合特定领域。  

### 方法详解
整体思路可以概括为三步：**（1）准备领域 API 知识 → （2）让模型获取或内部化这些知识 → （3）生成代码**。DomCoder 把这三步拆成可插拔的模块。

1. **API 知识准备**  
   - 收集目标领域的官方文档、示例代码和常用函数签名。  
   - 将每条 API 信息结构化为 “函数名 + 参数类型 + 返回值 + 简短说明”。  
   - 对于 Go 的 Gin、Prometheus、gRPC-go 以及 C++ 的 Unreal、Cocos2d-x、Bgfx，分别建立独立的知识库。

2. **知识获取策略**  
   - **外部查询器**：在模型生成每一行代码前，先检查当前上下文是否涉及未出现的 API。如果是，向文档检索引擎发送关键词查询，返回最相关的 API 片段。检索结果被拼接到模型的下一轮提示中。  
   - **思维链提示**：在提示里加入 “思考步骤：① 初始化服务器 ② 注册路由 ③ 启动监听”。模型先输出这些步骤的文字描述，再依据描述写代码。  
   - **思维链微调**：使用包含思考步骤的标注数据（如 “Step1: … Step2: … Code: …”）继续训练模型，使其在生成时自动先输出步骤，再输出代码。

3. **代码生成**  
   - 模型接收统一格式的提示：`[Domain Knowledge] + [Previous Code] + [Thought Steps]`。  
   - 采用温度较低的采样策略，确保生成的 API 调用保持一致性。  
   - 生成完毕后，使用静态分析工具检查是否出现未导入的符号或类型不匹配，若发现错误则回到步骤 2 重新检索或补充思考步骤。

**巧妙之处**：外部查询器并不是一次性把所有文档塞进去，而是“按需拉取”。这类似于程序员写代码时只打开需要的手册章节，既节省了上下文空间，又保证了信息的时效性。思维链微调则把“写草稿”的习惯内化到模型权重里，让模型在没有外部帮助时也能自行组织调用顺序。

### 实验与效果
- **数据集与任务**：作者挑选了两个真实行业的代码生成任务——Go 语言的 Web 开发（使用 Gin、Prometheus、gRPC-go）和 C++ 的游戏开发（使用 Unreal Engine、Cocos2d-x、Bgfx）。每个任务提供若干功能需求描述，要求模型输出完整可编译的实现。  
- **对比基线**：包括原始 ChatGPT/GPT‑4（仅一次性提示）、基于同等规模的 CodeLlama、以及在公开 HumanEval 上微调的模型。  
- **主要结果**：论文声称，在所有六个库的测试用例中，DomCoder 的三种策略均比纯提示基线提升了约 10%–20% 的通过率，尤其在需要复杂初始化顺序的 Unreal 示例中，思维链微调的提升最高，超过 25%。  
- **消融实验**：分别去掉外部查询、思维链提示或微调，发现外部查询对 API 覆盖率贡献最大，思维链提示对错误率下降贡献显著，微调在资源受限（无检索）时提供了最稳健的备份。  
- **局限性**：作者指出，检索系统的质量直接影响外部查询的效果；在文档不完整或多语言混用的场景下仍会出现错误。此外，思维链微调需要额外的标注成本，当前只在少数领域完成。  

### 影响与延伸思考
这篇工作打开了“让大模型主动查文档、写思考步骤”的新视角，随后出现了多篇跟进研究，例如将向量检索与 LLM 结合的 **Retrieval‑Augmented Code Generation**，以及专注于 **API‑aware Prompt Engineering** 的工作。对想进一步探索的读者，可以关注以下方向：  
- **跨语言 API 对齐**：如何让模型在同一提示中同时使用 Go 与 C++ 的库？  
- **自动化思维链标注**：利用已有代码自动生成思考步骤，降低标注成本。  
- **实时文档更新**：把 CI/CD 中的库变更自动同步到检索库，保持模型的知识新鲜度。  

### 一句话记住它
让大语言模型在写专业代码时先“查手册、写草稿”，就能显著提升领域代码生成的准确率。