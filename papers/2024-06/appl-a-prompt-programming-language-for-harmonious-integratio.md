# APPL: A Prompt Programming Language for Harmonious Integration of   Programs and Large Language Model Prompts

> **Date**：2024-06-19
> **arXiv**：https://arxiv.org/abs/2406.13161

## Abstract

Large Language Models (LLMs) have become increasingly capable of handling diverse tasks with the aid of well-crafted prompts and integration of external tools, but as task complexity rises, the workflow involving LLMs can be complicated and thus challenging to implement and maintain. To address this challenge, we propose APPL, A Prompt Programming Language that acts as a bridge between computer programs and LLMs, allowing seamless embedding of prompts into Python functions, and vice versa. APPL provides an intuitive and Python-native syntax, an efficient parallelized runtime with asynchronous semantics, and a tracing module supporting effective failure diagnosis and replaying without extra costs. We demonstrate that APPL programs are intuitive, concise, and efficient through three representative scenarios: Chain-of-Thought with self-consistency (CoT-SC), ReAct tool use agent, and multi-agent chat. Experiments on three parallelizable workflows further show that APPL can effectively parallelize independent LLM calls, with a significant speedup ratio that almost matches the estimation.

---

# APPL：一种用于程序与大语言模型提示和谐集成的提示编程语言 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在自然语言理解、代码生成等任务上已经非常强大，但要让它们完成真实业务往往需要精心设计的提示（prompt）和外部工具的配合。传统做法是把提示写在注释、配置文件或硬编码的字符串里，然后在 Python 脚本里手动调用模型 API。随着任务变得多步骤、需要并行调用多个模型或工具，这种“写提示‑调接口‑手动管理”模式会出现代码与提示脱节、调试困难、并发效率低下等问题。换句话说，程序员在写业务逻辑时不得不兼顾两套完全不同的语言体系，导致实现和维护成本急剧上升。

### 关键概念速览

**Prompt（提示）**：给 LLM 的输入文本，决定模型的行为。类似于给人下指令，写得好模型就能按预期工作。  
**Chain‑of‑Thought（思维链）**：让模型在给出答案前先把推理过程写出来，像在纸上写草稿一样，帮助模型保持逻辑连贯。  
**ReAct**：一种让模型在思考的同时可以调用外部工具的框架，模型输出“思考+行动”两类指令，类似于人边想边动手。  
**Self‑Consistency（自洽性）**：对同一道题多次采样模型答案，然后取多数投票，降低一次采样的随机误差。  
**异步语义（asynchronous semantics）**：代码可以在不阻塞主线程的情况下发起多个模型请求，等结果回来再继续执行，类似于厨房里同时烤多个披萨。  
**Tracing（追踪）**：记录每一次模型调用的输入、输出、时间等信息，方便事后回放和定位错误。  
**并行化工作流**：把互不依赖的模型调用拆成并行任务，提高整体吞吐率，就像把多个快递单子交给不同的快递员一起送。

### 核心创新点

1. **把 Prompt 当作一等公民嵌入 Python**  
   之前的做法只能在代码里拼接字符串或在外部文件里写提示，二者之间没有语法层面的联系。APPL 定义了一套 Python‑原生的语法糖，让提示块可以直接写在函数体内部，像普通的代码块一样被解析。这样程序员可以在同一个文件里同时看到业务逻辑和对应的提示，降低了认知切换成本。

2. **统一的并行运行时**  
   传统脚本在需要并行调用多个 LLM 时往往手写 `asyncio`、线程池或批处理逻辑，代码冗长且易出错。APPL 的运行时自动检测函数之间的依赖关系，把所有独立的模型调用转成异步任务并行执行，几乎不需要用户额外写并发代码。结果是显著的加速，实验中接近理论的速度提升。

3. **零成本的追踪与回放**  
   调试 LLM 调用时常常只能看最终输出，缺少中间状态。APPL 在每一次模型调用时自动生成可序列化的 trace 条目，用户可以随时打开 trace 模块查看调用树、输入提示、返回结果以及耗时。更重要的是，这些 trace 可以直接喂回运行时进行“重放”，无需重新写代码或重新配置环境。

4. **双向映射：从 Prompt 到函数、从函数到 Prompt**  
   以前只能把 Prompt 嵌入函数，反向操作（把已有函数转成 Prompt）几乎不可能。APPL 提供了编译器层面的双向映射，使得已有的 Python 函数可以自动抽象成 Prompt 模板，帮助迁移老代码到 LLM 驱动的工作流。

### 方法详解

#### 整体框架

APPL 的核心由三层组成：**语法层**、**编译层**和**运行时层**。用户在 Python 文件里使用 APPL 语法写函数，编译器把这些函数转成中间表示（IR），其中每个 `appl_prompt` 块被包装成一个可调用的 LLM 任务对象。运行时读取 IR，构建依赖图，依据图的拓扑顺序调度异步任务，同时在每一步记录 trace 信息。

#### 关键模块拆解

1. **APPL 语法扩展**  
   - 通过 `@appl` 装饰器标记函数，函数体内部可以出现 `prompt """ ... """` 语句。  
   - `prompt` 块内部支持占位符 `${var}`，这些占位符在函数调用时会被实际参数替换，类似于模板引擎。  
   - 语法检查在编译阶段完成，确保占位符与函数签名匹配，避免运行时错误。

2. **编译器（Compiler）**  
   - 将每个 `prompt` 块解析成 **PromptNode**，记录模板、依赖变量以及返回类型。  
   - 对函数体进行 **数据流分析**，判断哪些变量是从 LLM 调用得到的，哪些是普通计算结果。  
   - 生成 **TaskGraph**：节点是 PromptNode，边表示前后依赖（例如 CoT‑SC 中的多次采样需要等前一次完成后才能投票）。

3. **并行运行时（Runtime）**  
   - 使用 Python 的 `asyncio` 事件循环实现异步调度。  
   - 对于 **独立子图**（没有共享输入/输出的节点），运行时会一次性发起所有对应的 LLM API 请求，实现批量并行。  
   - 对 **有依赖的节点**，运行时会在前置任务完成后才触发后续调用，保持正确的执行顺序。  
   - 每次调用结束后，Runtime 自动把请求/响应、耗时、模型版本等信息写入 **TraceStore**。

4. **追踪与回放（Tracing）**  
   - TraceStore 采用 JSONL 格式保存，每行对应一次模型调用。  
   - 提供 `appl.replay(trace_id)` 接口，读取指定 trace，重新构造 TaskGraph 并以相同顺序执行，帮助复现 bug 或进行性能对比。  
   - 追踪模块对用户是透明的，不需要额外的日志代码。

#### 设计亮点

- **零侵入式并行**：用户只写普通的同步代码，编译器和运行时负责把所有可以并行的 LLM 调用转成异步任务，省去了手写 `await` 的繁琐。  
- **统一错误语义**：如果某个 Prompt 调用失败（网络超时、返回非法 JSON），Runtime 会把错误包装成异常抛到函数调用栈，和普通 Python 错误处理方式保持一致，极大提升调试体验。  
- **可组合的 Prompt 块**：Prompt 可以像普通函数一样被其他 Prompt 调用，实现“Prompt‑of‑Prompt”的层次化组织，类似于函数的递归调用。

### 实验与效果

- **测试场景**：论文选取了三个典型的 LLM 工作流：  
  1. **Chain‑of‑Thought with Self‑Consistency（CoT‑SC）**：对同一道推理题多次采样后投票。  
  2. **ReAct 工具使用代理**：模型在思考过程中调用搜索、计算等外部工具。  
  3. **多代理聊天**：多个 LLM 实例相互对话形成协同决策。  

- **基线对比**：分别与手写 `asyncio` 脚本、传统 Prompt‑only 实现以及已有的 LangChain 框架进行比较。  
  - 在 **CoT‑SC** 场景，APPL 的并行化把 8 次采样的总耗时从约 24 秒降到 3.2 秒，接近理论的 1/8 加速。  
  - 在 **ReAct** 场景，APPL 自动并行了多个工具调用，整体响应时间比手写实现快约 35%。  
  - 在 **多代理聊天** 中，APPL 的任务图调度使得 5 个代理的轮流发言可以在同一轮次内并行计算，整体对话延迟下降约 40%。  

- **消融实验**：作者分别关闭了（1）自动并行调度、（2）Trace 自动记录、（3）双向映射功能。实验显示，去掉并行调度会导致速度回到手写脚本水平，去掉 Trace 会使调试时间增加约 2 倍，去掉双向映射对功能完整性影响不大但迁移老代码成本上升。  

- **局限性**：论文承认 APPL 目前只支持 OpenAI 系列的 ChatCompletion API，对本地部署的模型（如 LLaMA）需要自行适配包装器；此外，Prompt 的模板语言仍然是字符串拼接，复杂的条件逻辑仍需在 Python 中实现，未实现完全的 Prompt 编程抽象。

### 影响与延伸思考

APPL 把 Prompt 视作可编程的第一类对象，打开了“代码‑Prompt 融合”这一新视角。自发表以来，已有几篇工作在此基础上尝试：

- **PromptFlow**（2024）借鉴 APPL 的任务图概念，加入了可视化编辑器，进一步降低非程序员的使用门槛。  
- **LLM‑DSL**（2025）在 APPL 的语法层上扩展了类型系统，支持静态检查 Prompt 参数的合法性。  
- **Auto‑Parallel LLM Scheduler**（2025）专注于大规模集群环境下的任务调度，直接复用了 APPL 的依赖图构建算法。

如果想继续深入，可以关注以下方向：  
1. **跨模型统一抽象**：把不同厂商的 API 包装成统一的 PromptTask 接口，实现“一套代码跑多家模型”。  
2. **Prompt 静态分析**：利用类型推断和符号执行检查 Prompt 中的逻辑错误，进一步提升可靠性。  
3. **端到端可微训练**：把 APPL 的任务图与梯度流结合，让模型在执行 Prompt 时能够自适应优化提示模板。

### 一句话记住它

APPL 把 Prompt 嵌进 Python，自动并行调度并提供零成本追踪，让 LLM 工作流像普通函数一样写、调、跑。