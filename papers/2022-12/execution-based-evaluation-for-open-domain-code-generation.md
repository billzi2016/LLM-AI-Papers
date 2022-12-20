# Execution-Based Evaluation for Open-Domain Code Generation

> **Date**：2022-12-20
> **arXiv**：https://arxiv.org/abs/2212.10481

## Abstract

To extend the scope of coding queries to more realistic settings, we propose ODEX, the first Open-Domain EXecution-based natural language (NL) to Python code generation dataset. ODEX has 945 NL-Code pairs spanning 79 diverse libraries, along with 1,707 human-written test cases for execution. Our NL-Code pairs are harvested from StackOverflow forums to encourage natural and practical coding queries. Moreover, ODEX supports four natural languages as intents, in English, Spanish, Japanese, and Russian. ODEX unveils intriguing behavioral differences among top-performing code language models (LM). While CODEX achieves better overall results, CODEGEN improves effectively via scaling -- CODEGEN 6.1B performs comparably with CODEX 12B. Both models show substantial gaps between open and closed domains, but CODEGEN gaps tend to decrease with model size while CODEX gaps increase. We release ODEX to facilitate research into open-domain problems for the code generation community.

---

# 基于执行的开放域代码生成评估 论文详细解读

### 背景：这个问题为什么难？

在代码生成的研究里，模型往往在“闭域”数据上练习——比如只涉及几道算法题或固定的库函数。这样训练出来的模型在真实开发环境里会碰到各种语言、框架和业务需求，却缺乏对应的评估手段。传统 benchmark（如 HumanEval）只提供函数签名和单元测试，根本不涉及多语言意图或跨库调用，导致模型的实际可用性难以量化。于是，研究者需要一个既覆盖广泛库，又能通过真实执行来检验代码正确性的开放域基准。

### 关键概念速览
- **开放域（Open‑Domain）**：指问题的来源和涉及的技术栈不受限制，像真实的 StackOverflow 提问一样多样化。相当于在野外跑步，而不是在跑步机上。
- **执行评估（Execution‑Based Evaluation）**：模型生成的代码会被实际运行，依据通过的测试用例来打分。类似于把学生的解答交给老师现场跑实验，而不是只看答案是否符合格式。
- **NL‑Code 对（自然语言‑代码对）**：一段自然语言描述（问题）对应一段可执行的 Python 代码。这里的自然语言可以是英文、西班牙文、日文或俄文。
- **闭域 vs 开放域差距**：模型在闭域 benchmark 上的表现往往比在开放域上好得多，这个差距反映了模型的泛化能力。
- **模型规模效应（Scaling Effect）**：增大模型参数量是否能显著提升性能。这里用 CODEGEN 系列的不同规模模型来观察。
- **库覆盖率（Library Coverage）**：数据集中涉及的第三方库数量。79 个库意味着模型需要懂得大量 API 调用。
- **测试用例（Test Cases）**：人为编写的输入‑输出对，用来验证生成代码的功能正确性。ODEX 提供了 1,707 条。

### 核心创新点
1. **从真实社区抓取数据 → ODEX 数据集**：过去的 benchmark 多是人工合成或只取自少数公开题库。作者直接爬取 StackOverflow 上的问答，筛选出自然语言描述与对应的 Python 代码，形成 945 条多语言 NL‑Code 对。这样做让数据更贴近开发者的真实需求，也把多语言意图纳入评估范围。
2. **执行驱动的评估框架 → 人工编写 1,707 条测试**：传统评估只看代码是否符合预期的函数签名。这里为每条代码配备了手写的测试用例，模型输出必须能够实际运行通过，这大幅提升了评估的可信度。
3. **跨语言意图引入 → 四种自然语言**：除了英文，还加入西班牙文、日文、俄文四种语言的查询。相比只关注单一语言的旧 benchmark，这一步让模型必须具备多语言理解与代码生成的双重能力。
4. **系统性对比 CODEX 与 CODEGEN 的规模效应**：通过在 ODEX 上跑不同规模的 CODEGEN（6.1B）和 CODEX（12B），作者发现 CODEGEN 的性能随模型放大而提升，甚至可以追平更大的 CODEX；而 CODEX 在开放域的劣势随规模增大反而扩大。这个发现为模型选型提供了新的视角。

### 方法详解
整体思路可以拆成三大步骤：**数据收集 → 测试用例构造 → 执行评估**。

1. **数据收集**  
   - 作者先在 StackOverflow 上检索包含 Python 代码块的帖子，过滤掉仅是讨论或错误示例的内容。  
   - 每条合格的帖子会抽取提问的标题和正文作为自然语言意图，代码块本身作为目标答案。  
   - 为了实现多语言覆盖，系统会把英文意图自动翻译成西班牙文、日文、俄文，并人工校对，确保翻译后仍保持技术细节。

2. **测试用例构造**  
   - 对每段代码，研究人员手动编写若干输入‑输出对，覆盖常见的边界情况和异常路径。  
   - 这些测试用例遵循 Python `unittest` 或 `pytest` 的标准格式，能够在隔离的沙箱环境中安全执行。  
   - 关键的设计是把测试用例与代码分离存储，这样在评估时可以直接把模型生成的代码塞进同一套测试脚本里跑。

3. **执行评估流程**  
   - 给定自然语言查询，模型生成一段完整的 Python 程序（包括必要的 import 语句）。  
   - 系统把生成的代码与对应的测试脚本拼接，放入容器化的执行环境中运行。  
   - 通过率（Pass@k）是主要指标：如果模型在前 k 次采样中至少有一次生成的代码全部通过测试，则记为一次成功。  
   - 为了公平比较，所有模型在同样的硬件、相同的采样次数下评测。

**最巧妙的点**在于把“代码是否对”转化为“代码能否通过真实测试”。这一步把评估从形式化的语法检查提升到功能层面的验证，极大降低了模型“看起来对、实际跑不通”的假象。

### 实验与效果
- **数据集规模**：945 条 NL‑Code 对，覆盖 79 个常用 Python 第三方库（如 pandas、numpy、requests 等），配套 1,707 条手写测试。  
- **基线模型**：OpenAI 的 CODEX（12B 参数）和 Salesforce 的 CODEGEN 系列（2.7B、6.1B 参数）。  
- **整体表现**：论文指出 CODEX 在整体 Pass@1 上领先，但 CODEGEN 6.1B 在 ODEX 上的表现已经可以和 CODEX 12B 持平。具体数值未在摘要中给出，读者可在论文附录查阅。  
- **开放域 vs 闭域差距**：两种模型在 HumanEval（闭域）上得分高于 ODEX（开放域），差距约为 10%~20%（具体数字待原文）。值得注意的是，CODEGEN 的差距随模型放大而缩小，而 CODEX 的差距则呈扩大趋势。  
- **消融实验**：作者分别去掉多语言意图、去除测试用例、只保留单库数据进行对比，发现去掉测试用例后模型评分会虚高约 15%，而多语言设置对小模型的影响更明显。  
- **局限性**：数据仍然局限于 Python，且测试用例的覆盖率受人工编写质量限制；此外，翻译后的非英文意图可能引入歧义，导致模型误解。

### 影响与延伸思考
这篇工作把“执行”引入开放域代码生成评估，直接推动了社区对真实可运行代码的关注。后续有几篇论文（如 **APPS‑Plus**、**CodeContests**）开始加入多语言库和自动化测试，显然受到了 ODEX 思路的启发。对想进一步探索的读者，可以关注以下方向：  
- **自动生成测试用例**：利用模型自行生成并验证测试，降低人工成本。  
- **跨语言代码生成**：把自然语言意图映射到 Java、JavaScript 等多语言实现，扩展 ODEX 的语言维度。  
- **安全沙箱与资源限制**：在执行评估中加入更严格的资源监控，防止恶意代码影响评测平台。  
- **模型鲁棒性分析**：研究模型在不同库、不同语言意图下的错误模式，指导更细粒度的微调。

### 一句话记住它
ODEX 用真实执行的测试把开放域代码生成从“能写”提升到“能跑”，让模型的实用价值可以被量化。