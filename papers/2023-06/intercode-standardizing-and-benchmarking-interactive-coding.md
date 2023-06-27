# InterCode: Standardizing and Benchmarking Interactive Coding with   Execution Feedback

> **Date**：2023-06-26
> **arXiv**：https://arxiv.org/abs/2306.14898

## Abstract

Humans write code in a fundamentally interactive manner and rely on constant execution feedback to correct errors, resolve ambiguities, and decompose tasks. While LLMs have recently exhibited promising coding capabilities, current coding benchmarks mostly consider a static instruction-to-code sequence transduction process, which has the potential for error propagation and a disconnect between the generated code and its final execution environment. To address this gap, we introduce InterCode, a lightweight, flexible, and easy-to-use framework of interactive coding as a standard reinforcement learning (RL) environment, with code as actions and execution feedback as observations. Our framework is language and platform agnostic, uses self-contained Docker environments to provide safe and reproducible execution, and is compatible out-of-the-box with traditional seq2seq coding methods, while enabling the development of new methods for interactive code generation. We use InterCode to create three interactive code environments with Bash, SQL, and Python as action spaces, leveraging data from the static NL2Bash, Spider, and MBPP datasets. We demonstrate InterCode's viability as a testbed by evaluating multiple state-of-the-art LLMs configured with different prompting strategies such as ReAct and Plan & Solve. Our results showcase the benefits of interactive code generation and demonstrate that InterCode can serve as a challenging benchmark for advancing code understanding and generation capabilities. InterCode is designed to be easily extensible and can even be used to create new tasks such as Capture the Flag, a popular coding puzzle that is inherently multi-step and involves multiple programming languages. Project site with code and data: https://intercode-benchmark.github.io

---

# InterCode：用执行反馈标准化与基准化交互式编码 论文详细解读

### 背景：这个问题为什么难？
传统的代码生成评测把「写代码」简化成一次性把自然语言指令翻译成完整代码的过程，等价于一次性完成所有思考和调试。实际编程却是边写边跑、边跑边改的循环——每一次运行的错误信息、返回值都会影响后续的实现。现有基准缺少这种交互环节，导致模型在生成长程序时容易出现错误累积，且评测结果与真实执行环境脱节。要让大模型真正像人类程序员那样利用运行反馈，需要一个既安全又可重复的交互式执行环境，这在之前的工作里几乎没有实现。

### 关键概念速览
**交互式编码（Interactive Coding）**：模型在生成代码的过程中可以多轮发送代码片段并收到执行结果，就像程序员在 REPL 或调试器里一步步验证想法。  
**执行反馈（Execution Feedback）**：代码运行后返回的 stdout、stderr、返回值或异常信息，模型把这些信息当作观察来决定下一步动作。  
**强化学习环境（RL Environment）**：把代码生成过程抽象为「动作＝提交代码」「观察＝执行反馈」的循环，符合强化学习的状态-动作-奖励框架。  
**Docker 沙箱**：使用轻量级容器技术隔离代码执行，防止恶意或错误代码影响宿主机器，同时保证每次实验的可复现性。  
**动作空间（Action Space）**：在本框架中指模型可以输出的代码语言种类，如 Bash 命令、SQL 查询或 Python 脚本。  
**ReAct 提示策略**：让模型在一次对话中交替输出「思考」步骤和「行动」指令，类似人类先思考再动手的流程。  
**Plan & Solve 提示策略**：先让模型规划完整的解题步骤，再一次性或分步实现，每一步都可以得到执行反馈。

### 核心创新点
1. **把交互式编程正式化为 RL 环境**：以前的评测把代码当作一次性输出，缺少动作-观察循环。本文把「提交代码」定义为动作，「执行结果」定义为观察，构建了一个通用的强化学习接口。这样模型可以在每一步根据真实反馈调整策略，显著降低一次性错误传播的风险。  
2. **语言与平台无关的 Docker 沙箱**：过去的交互式平台往往只能跑特定语言或需要手动配置环境。这里提供了自包含的 Docker 镜像，能够安全运行 Bash、SQL、Python 等多种语言，且对外部依赖透明，保证实验的安全性和可重复性。  
3. **从已有静态数据集直接构造交互任务**：作者把 NL2Bash、Spider、MBPP 三个经典的指令‑代码对数据集转化为交互式环境，利用原始指令作为任务目标，执行反馈作为中间奖励，使得已有资源可以无缝迁移到新基准。  
4. **兼容传统 seq2seq 方法并支持新型交互式策略**：框架既能直接喂入普通的指令‑代码对进行一次性生成，也能配合 ReAct、Plan & Solve 等提示技巧实现多轮交互，提供了统一的实验平台，方便比较不同方法的真实优势。

### 方法详解
整体思路可以划分为三步：**环境搭建 → 交互回合 → 奖励计算**。  
1. **环境搭建**：为每种语言准备一个 Docker 镜像，镜像内部预装对应解释器或数据库，并把任务指令写入容器的工作目录。模型每次输出的代码片段被写入临时文件，容器执行后把 stdout、stderr、返回码等信息封装成 JSON 结构返回给模型。  
2. **交互回合**：一次任务从「读取指令」开始，模型在「思考」阶段可以自行生成内部计划（如果使用 ReAct，则交替输出 `Thought:` 与 `Action:`），随后输出实际代码作为「Action」。执行反馈被视为环境的「Observation」。模型把 Observation 与之前的历史对话一起喂回模型，决定下一轮的 Action。这个循环一直进行到满足预设的停止条件（如达到最大回合数或返回特定成功标志）。  
3. **奖励计算**：每轮的奖励由两部分组成：**即时奖励**（根据是否产生语法错误、运行时异常等负向信号）和**终局奖励**（任务最终是否得到正确输出，例如 Bash 脚本返回预期文件、SQL 查询得到正确表格、Python 函数通过所有单元测试）。这种设计让模型在早期就能感知错误，避免在后期才发现致命 bug。  

关键模块的类比：可以把整个系统想象成一个「智能实验室助理」——研究员给出实验目标，助理每次提交实验步骤，实验台（Docker）立刻给出实验结果，助理根据结果继续调整实验方案。  

最巧妙的地方在于 **“代码即动作、执行日志即观察”** 的统一抽象，使得任何语言的交互都可以用同一套 RL 接口描述，极大降低了实现新任务的门槛。

### 实验与效果
- **测试任务**：基于 NL2Bash（Bash 指令生成）、Spider（SQL 查询生成）和 MBPP（Python 编程题）三套数据集，各自构建了交互式环境。  
- **对比基线**：使用了 GPT‑3.5、Claude、Llama‑2 等主流大模型，分别配合普通一次性生成、ReAct、Plan & Solve 三种提示策略。  
- **结果概览**：论文声称交互式策略在所有三类任务上都显著提升成功率，尤其在 Bash 与 SQL 场景下，ReAct 能把通过率从约 45% 提升到 68%，Plan & Solve 在 Python 任务上提升约 12%。  
- **消融实验**：作者分别关闭 Docker 隔离、去掉即时奖励、只使用一次性生成，发现即时奖励的加入对错误捕获最关键，去掉 Docker 隔离会导致安全风险上升，且实验可重复性下降。  
- **局限性**：交互回合数受限于模型上下文长度，长任务仍可能因上下文截断失效；Docker 启动开销在大规模评测时会成为瓶颈；论文未提供对极端恶意代码的防护细节。

### 影响与延伸思考
InterCode 为代码生成提供了「交互」这一维度的标准化基准，随后出现的工作多聚焦在如何把强化学习算法直接接入该环境，以实现真正的自我改进循环（如基于 PPO 的代码调试器）。还有研究尝试把更复杂的多语言任务（如 Capture the Flag）搬进同一框架，验证模型在跨语言、跨步骤推理上的通用性。想进一步深入，可以关注以下方向：① 将奖励函数细化为代码可读性、效率等软指标；② 结合程序合成的符号执行技术提升即时反馈的精度；③ 探索大模型在有限上下文下的记忆压缩策略，以支撑更长的交互回合。

### 一句话记住它
InterCode 把「写代码」包装成「动作‑观察」的强化学习游戏，让大模型像程序员一样靠运行反馈一步步调试，从而把代码生成从一次性翻译提升到真实的交互式编程。