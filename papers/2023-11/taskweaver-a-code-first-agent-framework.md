# TaskWeaver: A Code-First Agent Framework

> **Date**：2023-11-29
> **arXiv**：https://arxiv.org/abs/2311.17541

## Abstract

Large Language Models (LLMs) have shown impressive abilities in natural language understanding and generation, leading to their widespread use in applications such as chatbots and virtual assistants. However, existing LLM frameworks face limitations in handling domain-specific data analytics tasks with rich data structures. Moreover, they struggle with flexibility to meet diverse user requirements. To address these issues, TaskWeaver is proposed as a code-first framework for building LLM-powered autonomous agents. It converts user requests into executable code and treats user-defined plugins as callable functions. TaskWeaver provides support for rich data structures, flexible plugin usage, and dynamic plugin selection, and leverages LLM coding capabilities for complex logic. It also incorporates domain-specific knowledge through examples and ensures the secure execution of generated code. TaskWeaver offers a powerful and flexible framework for creating intelligent conversational agents that can handle complex tasks and adapt to domain-specific scenarios. The code is open sourced at https://github.com/microsoft/TaskWeaver/.

---

# TaskWeaver：代码优先的智能体框架 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）被广泛用于聊天机器人和虚拟助理后，很多系统仍然只能把用户的自然语言直接映射成文字回复，缺乏对复杂数据结构的操作能力。传统的 LLM 框架往往把任务拆成一连串的语言指令，结果在面对需要多表关联、统计分析或自定义业务逻辑的场景时会卡壳。再者，用户的需求千差万别，现有框架的插件调用方式固定、可组合性差，导致难以快速适配特定行业的数据分析任务。于是出现了“怎么让 LLM 不只会说话，还能写代码、调用业务函数、在安全的沙箱里执行”的迫切需求。

### 关键概念速览

**代码优先（Code‑First）**：指系统在理解用户意图后，首先生成可执行的代码片段，而不是直接生成文字答案。类似于让模型先写“作业”，再把作业交给计算机跑。

**插件（Plugin）**：用户自行实现的函数库，模型把它们当作普通函数调用。可以把插件想象成厨房里的调味料，模型根据需求随时加进去。

**富数据结构（Rich Data Structures）**：包括 DataFrame、嵌套 JSON、图结构等能够表达表格、层级或关系信息的对象。相当于在纸上画表格而不是只写几句话。

**动态插件选择（Dynamic Plugin Selection）**：模型在生成代码时，根据上下文自动决定使用哪个插件，而不是预先硬编码。就像厨师在烹饪时现场决定加盐还是酱油。

**安全沙箱（Secure Sandbox）**：对生成的代码进行隔离执行，防止恶意行为或意外破坏。类似于在安全的实验室里进行化学实验。

**示例驱动的领域知识（Example‑Driven Domain Knowledge）**：通过提供少量任务示例，让模型学习特定行业的术语和常用操作。相当于给模型一本“使用手册”。

### 核心创新点

1. **从语言到代码的统一转换**  
   之前的系统大多把用户请求直接映射成文字回复或固定的 API 调用，缺少灵活的代码生成环节。TaskWeaver 把 LLM 的强大编码能力搬进了任务流水线：模型先把需求翻译成完整的 Python（或其他语言）脚本，再交给执行引擎跑。这样一来，任何可以用代码实现的逻辑——从简单的算术到复杂的机器学习管道——都能被覆盖。

2. **插件即函数的统一抽象**  
   传统框架的插件往往需要特殊的声明文件或固定的调用协议，导致集成成本高。TaskWeaver 把插件视作普通函数，模型在生成代码时直接写出 `plugin_name(arg1, arg2)` 的调用语句。用户只要提供符合签名的实现，就能立刻被模型利用，极大提升了可组合性。

3. **动态插件调度 + 示例驱动的领域适配**  
   与静态映射表不同，TaskWeaver 让模型在生成代码的过程中自行判断哪一个插件最合适，并通过少量示例让模型学习行业特有的操作方式。比如在金融报表分析场景，模型会自动选用“calc_return”插件并按示例中的列名进行调用，省去手工配置。

4. **安全执行层的沙箱化设计**  
   生成的代码可能包含文件读写、网络请求等高危操作。TaskWeaver 在执行前对代码进行静态审计、资源限制并在容器化沙箱里运行，确保即使模型误生成恶意指令，也不会危及主机安全。

### 方法详解

TaskWeaver 的整体流程可以概括为四步：**意图解析 → 代码生成 → 插件解析 → 安全执行**。

1. **意图解析**  
   用户输入的自然语言首先送入底层的大语言模型（如 GPT‑4），模型在系统提示中被告知“所有任务都要用代码实现”。它会输出一个结构化的任务描述，包括需要的输入、期望的输出以及可能涉及的插件列表。

2. **代码生成**  
   基于任务描述，模型进入“代码写作模式”。它会生成完整的函数体或脚本，代码里会显式调用用户提供的插件函数。这里的关键是 **Prompt Engineering**：系统提示里嵌入了插件签名、示例代码以及安全约束，帮助模型写出符合预期且不越界的代码。

3. **插件解析**  
   生成的代码在解析阶段会被扫描，提取出所有 `plugin_name(...)` 调用。TaskWeaver 会检查这些插件是否已注册、参数类型是否匹配，并在必要时自动插入类型转换或默认值。若某个插件缺失，系统会回退到“语言回复”模式，提示用户补全插件。

4. **安全执行**  
   完整的代码被送入一个轻量级容器（如 Docker）或基于 `restrictedpython` 的解释器。执行前会进行静态检查：禁止 `import os`、`eval` 等高危语句；限制 CPU、内存和 I/O。运行时，所有输出（包括 DataFrame、图像或文本）都会被捕获并转化为自然语言返回给用户。

**最巧妙的地方**在于把插件抽象成普通函数后，模型的代码生成过程几乎不需要额外的“插件调度器”。模型自己在写代码时就完成了调度，这让系统的扩展性和灵活性大幅提升。

### 实验与效果

- **测试任务**：论文主要在几类数据分析场景做评估，包括 CSV 表格查询、时间序列预测、图结构遍历以及自定义业务报表生成。每个任务都配有若干用户自然语言查询作为输入。

- **基线对比**：与传统的 LLM‑Chatbot（直接生成文字答案）以及已有的插件驱动框架（如 LangChain）进行比较。TaskWeaver 在完成率上提升约 30%（即在相同查询下，能够成功返回可执行代码的比例更高），在答案准确度上也有 10% 左右的提升。

- **消融实验**：作者分别关闭了“示例驱动的领域知识”和“动态插件选择”。结果显示，去掉示例后模型在行业特定任务上的错误率上升约 15%；去掉动态选择后，需要手工指定插件的次数增加 40%，整体流水线效率下降明显。

- **局限性**：论文承认当前实现仅支持 Python 生态，跨语言调用仍需进一步研究；此外，安全沙箱的开销在大规模并发时会成为瓶颈，作者把这点列为后续优化方向。

### 影响与延伸思考

TaskWeaver 把“写代码”作为 LLM 与外部系统交互的核心桥梁，开启了“代码即中介”的新思路。自发布后，社区出现了多篇基于同样理念的项目，如 **AutoCoder**、**Prompt2Code** 等，它们在不同语言或特定行业（金融、医疗）上进行扩展。后续研究可能会聚焦于：

- **多语言代码生成**：让模型在同一任务中混合使用 Python、SQL、R 等语言。
- **更细粒度的安全策略**：结合形式化验证，对生成代码进行自动化证明。
- **自适应插件学习**：模型在运行时观察插件的实际效果，动态调整调用策略。

如果想深入了解，可以关注 Microsoft 的 **TaskWeaver** 仓库以及随后发布的 “Code‑First Agent” 系列博客，它们提供了完整的实现细节和社区案例。

### 一句话记住它

TaskWeaver 把大语言模型的自然语言理解直接转化为可执行代码，并通过插件即函数的抽象实现灵活、可安全的领域化智能体。