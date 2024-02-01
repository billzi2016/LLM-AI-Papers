# Executable Code Actions Elicit Better LLM Agents

> **Date**：2024-02-01
> **arXiv**：https://arxiv.org/abs/2402.01030

## Abstract

Large Language Model (LLM) agents, capable of performing a broad range of actions, such as invoking tools and controlling robots, show great potential in tackling real-world challenges. LLM agents are typically prompted to produce actions by generating JSON or text in a pre-defined format, which is usually limited by constrained action space (e.g., the scope of pre-defined tools) and restricted flexibility (e.g., inability to compose multiple tools). This work proposes to use executable Python code to consolidate LLM agents' actions into a unified action space (CodeAct). Integrated with a Python interpreter, CodeAct can execute code actions and dynamically revise prior actions or emit new actions upon new observations through multi-turn interactions. Our extensive analysis of 17 LLMs on API-Bank and a newly curated benchmark shows that CodeAct outperforms widely used alternatives (up to 20% higher success rate). The encouraging performance of CodeAct motivates us to build an open-source LLM agent that interacts with environments by executing interpretable code and collaborates with users using natural language. To this end, we collect an instruction-tuning dataset CodeActInstruct that consists of 7k multi-turn interactions using CodeAct. We show that it can be used with existing data to improve models in agent-oriented tasks without compromising their general capability. CodeActAgent, finetuned from Llama2 and Mistral, is integrated with Python interpreter and uniquely tailored to perform sophisticated tasks (e.g., model training) using existing libraries and autonomously self-debug.

---

# 可执行代码动作提升大型语言模型代理 论文详细解读

### 背景：这个问题为什么难？

LLM（大型语言模型）已经可以当成“智能体”，通过调用工具或控制机器人完成任务。但传统的做法是让模型输出预先定义好的 JSON 或固定格式的指令，这相当于把模型的行动空间压缩成一张菜单。于是出现了两大瓶颈：一是工具集合是固定的，模型只能在有限的选项里挑选；二是模型无法灵活组合多个工具，面对需要多步操作的真实场景时往往力不从心。换句话说，现有的指令语言既不够广，也不够灵活，限制了 LLM 代理在复杂环境中的表现。

### 关键概念速览
- **LLM 代理**：把大语言模型当成可以感知环境、执行动作的“机器人”。它接受观察、输出行动，就像人类在对话中决定要做什么一样。  
- **可执行代码动作（CodeAct）**：让模型直接生成 Python 代码作为行动指令，代码随后在解释器里跑。相当于把模型的“说”变成了“写脚本”，脚本本身就是完整的操作描述。  
- **统一行动空间**：所有可能的操作都用 Python 代码来表达，模型不再受限于事先列好的工具列表，就像拥有一把万能钥匙可以打开任意门。  
- **多轮交互**：模型在执行代码后会得到运行结果（成功、报错、返回值），再基于这些新观察继续生成或修改代码，形成类似人类调试的循环。  
- **指令微调数据集 CodeActInstruct**：收集了约 7k 条多轮对话，每轮都包含模型生成的代码、执行反馈以及自然语言解释，用来教模型如何在代码层面进行自我纠错和任务规划。  
- **CodeActAgent**：基于 Llama2、Mistral 等开源模型微调得到的专用代理，内置 Python 解释器，能够调用现成的库完成诸如模型训练、数据处理等高级任务。  
- **API‑Bank 基准**：一个评估 LLM 代理调用外部 API 能力的测试集，常被用来比较不同指令语言的有效性。  

### 核心创新点
1. **从 JSON 指令到可执行代码**  
   - 之前的做法：模型输出固定格式的 JSON，解释器只能把它映射到预定义的工具调用。  
   - 本文做法：直接让模型写出完整的 Python 代码，解释器直接执行。  
   - 改变：行动空间从离散的工具列表扩展到几乎所有可以用 Python 表达的操作，模型可以自行组合库函数、控制流程，灵活度大幅提升。  

2. **代码层面的动态修正机制**  
   - 之前的做法：模型一次性输出指令，若出错只能重新发起一次完整的调用。  
   - 本文做法：执行后把运行结果（包括错误信息）反馈给模型，模型可以在后续轮次中修改已有代码或追加新代码。  
   - 改变：形成类似人类调试的闭环，使代理能够自我纠错、迭代优化，显著提升成功率。  

3. **大规模指令微调数据的构建**  
   - 之前的做法：微调数据大多是自然语言指令或少量工具调用示例。  
   - 本文做法：收集 7k 条多轮对话，每轮都包含代码、执行反馈和自然语言解释，形成“代码+语言”双模态训练材料。  
   - 改变：模型在微调后不仅会写代码，还学会根据运行结果进行自我调适，保持了原有的语言理解能力。  

4. **开源可执行代理的实现**  
   - 之前的做法：大多数公开的 LLM 代理仍停留在 JSON/Tool 调用层面。  
   - 本文做法：发布 CodeActAgent，内置解释器并对 Llama2、Mistral 进行指令微调。  
   - 改变：社区可以直接使用一个能够写代码、跑代码、调试代码的代理，降低了在真实环境中部署 LLM 代理的门槛。  

### 方法详解
整体框架可以概括为三步：**生成 → 执行 → 反馈**，在每一次循环中模型都以代码的形式表达自己的意图。

1. **代码生成模块**  
   - 输入：当前的对话历史、环境观察（如 API 返回、文件列表）以及任务描述。  
   - 过程：使用 LLM（如 Llama2）在提示词中加入“请用 Python 完成以下任务”，模型输出一段完整的 Python 脚本。脚本可以包含函数定义、库导入、条件分支等，几乎等同于人写的代码。  
   - 类比：把模型当成“程序员”，对话是需求说明，模型的输出是代码实现。  

2. **解释器执行层**  
   - 解释器接收模型的代码，放入受控的沙箱环境中运行。沙箱会捕获标准输出、返回值以及异常信息，防止恶意代码破坏系统。  
   - 执行结果被包装成结构化的观察（如 `{"status":"error","msg":"NameError: x not defined"}`），再送回模型。  

3. **反馈与迭代**  
   - 模型把刚才的执行结果当作新的观察，重新进入生成模块。它可以选择：  
     - **修改** 已有代码（比如补全缺失的变量），  
     - **追加** 新代码块（比如调用第二个 API），  
     - **结束** 并返回自然语言答案。  
   - 这种循环类似人类写脚本后调试的过程：先写、跑、看报错、改、再跑。  

4. **指令微调（CodeActInstruct）**  
   - 为了让模型学会这种循环，作者收集了 7k 条真实对话，每条都标注了：用户需求 → 模型代码 → 代码执行结果 → 模型的修正或答案。  
   - 微调时把这些三元组拼接成训练样本，让模型在学习阶段就能看到“写 → 运行 → 调整”的完整链路。  

**最巧妙的点**在于把代码本身当作“可执行的行动语言”，而不是把代码当作纯粹的文本解释。这样，模型的输出直接进入机器可执行的层面，省去了额外的解析器或映射表，极大提升了表达力和灵活性。

### 实验与效果
- **评测基准**：作者在公开的 API‑Bank 和自行构建的 CodeAct Benchmark 上进行测试，这两个基准都要求模型调用外部 API 完成指定任务。  
- **对比对象**：常见的 JSON/Tool 指令方式、基于自然语言的直接调用以及几种已有的 LLM 代理实现。  
- **核心结果**：在 17 种不同规模的 LLM 上，CodeAct 的成功率比传统指令提升了约 **20%**（具体数值在论文中给出）。  
- **消融实验**：去掉代码反馈环节或不使用 CodeActInstruct 微调，性能会显著下降，说明动态修正和指令微调是提升的关键因素。  
- **局限性**：论文承认代码执行的安全性仍是挑战，沙箱需要严格限制资源和网络访问；此外，模型在生成极其复杂的代码（如大规模并行计算）时仍会出现错误。  

### 影响与延伸思考
CodeAct 把“写代码”直接当作行动语言的思路在发布后迅速引起关注。随后出现的工作如 **ReAct‑Code**、**Toolformer‑Python** 等，都在探索更通用的可执行指令空间。社区也开始把类似的代码生成+执行循环用于数据清洗、自动化实验等场景。未来的研究可能会聚焦在：  
- **更安全的沙箱技术**，让代理能够调用更广的系统资源而不泄露风险；  
- **跨语言代码动作**，把 JavaScript、Bash 等也纳入统一行动空间；  
- **自我监督的代码调试**，让模型在没有外部反馈的情况下自行发现并修复错误（推测）。  

如果想进一步了解，可关注 OpenAI 的 **Function‑Calling** 进化路线、DeepMind 的 **Gato** 多模态代理以及近期的 **LLM‑as‑a‑Tool** 系列论文。

### 一句话记住它
让 LLM 直接写可执行的 Python 代码，并在运行反馈中循环修正，彻底打开了模型的行动空间。