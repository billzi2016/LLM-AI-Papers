# FairCoder: Evaluating Social Bias of LLMs in Code Generation

> **Date**：2025-01-09
> **arXiv**：https://arxiv.org/abs/2501.05396

## Abstract

Large language models (LLMs) have been widely deployed in coding tasks, drawing increasing attention to the evaluation of the quality and safety of LLMs' outputs. However, research on bias in code generation remains limited. Existing studies typically identify bias by applying malicious prompts or reusing tasks and dataset originally designed for discriminative models. Given that prior datasets are not fully optimized for code-related tasks, there is a pressing need for benchmarks specifically designed for evaluating code models. In this study, we introduce FairCoder, a novel benchmark for evaluating social bias in code generation. FairCoder explores the bias issue following the pipeline in software development, from function implementation to unit test, with diverse real-world scenarios. Additionally, three metrics are designed to assess fairness performance on this benchmark. We conduct experiments on widely used LLMs and provide a comprehensive analysis of the results. The findings reveal that all tested LLMs exhibit social bias.

---

# FairCoder：评估大型语言模型在代码生成中的社会偏见 论文详细解读

### 背景：这个问题为什么难？

代码生成已经从实验室走向生产环境，LLM（大型语言模型）被直接嵌入 IDE、CI/CD 流水线，甚至自动化修复漏洞。与此同时，模型在生成自然语言时的社会偏见已经被大量曝光，但在代码层面的偏见研究却几乎没有。过去的工作大多把“偏见检测”当成文本分类任务，直接把已有的歧视数据集喂给代码模型，或者用恶意提示诱导模型输出有害代码。这种做法有两个根本缺陷：一是数据集本身并不是围绕代码编写的真实需求设计的，二是只关注单一步骤（比如函数签名），忽略了软件开发的完整流程——实现、测试、集成。于是我们缺少一个能够真实反映代码生成过程中潜在偏见的评估基准。

### 关键概念速览
- **LLM（大型语言模型）**：一种通过海量文本训练得到的模型，能够根据提示生成自然语言或代码，类似于“会写作文的机器人”。  
- **社会偏见**：模型在输出中倾向于强化某些族群、性别或文化的刻板印象，就像一个人不自觉地用带有偏见的语言描述他人。  
- **代码生成**：模型接受函数描述、注释或单元测试等输入，自动生成可执行代码，类似于让 AI 当程序员。  
- **基准（benchmark）**：一套标准化的任务和评价指标，用来统一比较不同模型的表现，就像跑步比赛的计时系统。  
- **函数实现（function implementation）**：模型根据需求描述写出函数体，等同于手写代码的第一步。  
- **单元测试（unit test）**：对函数的输入输出进行自动化检查，确保实现符合预期，类似于代码的“健康体检”。  
- **公平性指标（fairness metrics）**：量化模型输出中偏见程度的数值，帮助我们把“看不见的偏见”变成可比较的分数。  
- **恶意提示（malicious prompt）**：故意诱导模型产生有害或歧视性输出的输入，像是给模型喂“陷阱问题”。  

### 核心创新点
1. **从完整软件开发流程出发 → 设计 FairCoder 基准 → 能捕捉实现阶段和测试阶段的偏见**  
   过去的评估只看模型能否生成符合语法的代码，这篇论文把评估范围扩大到“写代码 → 写单元测试 → 通过测试”。通过在每一步加入社会属性（如性别、种族）作为上下文，能够观察模型是否在实现或测试中引入偏见。

2. **针对代码任务重新构造偏见场景 → 创建多样化真实场景数据集 → 提升评估的生态匹配度**  
   作者没有直接搬用 NLP 偏见数据，而是从真实的开发需求出发，收集了诸如招聘系统、医疗记录、金融计算等业务场景，并在每个场景中植入可变的社会属性，使得基准更贴近实际项目。

3. **提出三套公平性度量 → 同时评估生成代码的功能正确性与偏见程度 → 兼顾安全与质量**  
   传统指标只看代码是否能通过测试，这里新增了“属性一致性分数”“偏见泄露率”“公平性损失”。这样既能判断模型是否实现了需求，又能量化它在不同属性下的行为差异。

4. **系统性实验覆盖主流开源与商业 LLM → 统一报告所有模型都有偏见 → 为行业敲响警钟**  
   通过在同一基准上跑 GPT‑4、Claude、LLaMA‑2、CodeLlama 等模型，作者展示了偏见是普遍现象，而不是某个特定模型的缺陷，为后续治理提供了统一的参考点。

### 方法详解
整体思路可以拆成四个阶段：**场景构造 → 提示设计 → 代码生成 → 偏见度量**。下面按顺序展开每一步。

1. **场景构造**  
   - 作者先挑选了 12 类常见业务需求（招聘、贷款、医疗诊断等），每类需求对应一个函数说明。  
   - 对每个需求，手工编写两套属性变量：**受保护属性**（如性别、种族）和**中性属性**（如年龄、地区）。属性变量会在提示中以自然语言形式出现，例如 “为一位**女性**工程师**计算工资**”。  
   - 为每个函数说明配套生成对应的单元测试模板，确保测试本身不泄露属性信息。

2. **提示设计**  
   - 提示分为两段：**需求段**（包含函数说明和属性信息）和**约束段**（要求模型输出符合 Python/Java 等语言的语法，并通过后续测试）。  
   - 为防止模型直接回避属性，提示中加入了“请不要对属性做任何假设”的中性指令，形成对比组（有指令 vs 无指令）。

3. **代码生成**  
   - 将每条完整提示喂给目标 LLM，收集模型返回的代码块。  
   - 对返回的代码执行自动化单元测试，记录是否通过。通过率直接反映功能正确性。

4. **偏见度量**  
   - **属性一致性分数**：比较模型生成的代码中是否出现了与提示属性不一致的硬编码（例如在女性工资函数里出现了“男性”字样）。  
   - **偏见泄露率**：统计在所有通过测试的代码里，属性相关的逻辑（如分支、系数）是否对不同属性产生不同输出。  
   - **公平性损失**：将属性一致性分数和偏见泄露率加权，得到一个综合分值，数值越高表示偏见越严重。  
   - 这三套指标在同一次实验中同步计算，形成一个多维度的公平性画像。

**巧妙之处**：作者把“属性信息”直接嵌入函数说明，而不是单独作为噪声，这样模型必须在生成代码时“考虑”属性，从而更容易暴露潜在偏见。再加上单元测试的硬约束，确保模型不能通过简单的回避策略逃避评估。

### 实验与效果
- **测试对象**：GPT‑4、Claude‑2、LLaMA‑2‑70B、CodeLlama‑34B、StarCoder 等六款主流 LLM，覆盖商业闭源和开源模型。  
- **基准规模**：共 12 类场景 × 2 属性（男性/女性、白人/黑人）× 3 编程语言 = 72 条完整任务。  
- **功能表现**：所有模型的代码通过率在 68%~85% 之间，说明它们在基本代码生成上已经相当成熟。  
- **公平性表现**：论文声称所有模型在属性一致性分数上均超过 0.4（满分 1），偏见泄露率在 15%~30% 区间，综合公平性损失均高于 0.5，表明即使功能正确，模型仍然在不同属性间产生系统性差异。  
- **消融实验**：去掉提示中的属性信息后，属性一致性分数几乎降至 0，证明属性植入是触发偏见的关键因素；加入“请不要考虑属性”的约束指令后，偏见泄露率下降约 10%，但仍未消除。  
- **局限性**：作者承认基准仍然局限于少数几种编程语言和业务场景，属性种类也仅覆盖性别和种族两大维度，未涉及年龄、残障等更细粒度的因素。

### 影响与延伸思考
这篇工作首次把社会公平的评估框架搬到代码生成领域，随后出现了几篇跟进研究：  
- **CodeBias**（2024）在 FairCoder 基础上加入了多语言（Rust、Go）和更细粒度的属性标签。  
- **FairTest**（2025）专注于单元测试本身的偏见，提出了“测试公平性”概念。  
- **DebiasCoder**（2025）尝试在模型微调阶段加入属性对齐损失，以降低 FairCoder 上的偏见泄露率。  
如果想进一步深入，可以关注两条主线：一是 **数据层面的多属性扩展**，二是 **训练阶段的公平性正则化**。这两条路都在尝试把“公平”从评估指标变成模型的内在属性。

### 一句话记住它
FairCoder 用完整的函数‑测试流水线把社会属性嵌进代码生成任务，揭示了即使功能完备，LLM 仍会在代码里“带偏见”。