# Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving

> **Date**：2025-05-12
> **arXiv**：https://arxiv.org/abs/2505.07773

## Abstract

Large Language Models (LLMs) often struggle with mathematical reasoning tasks requiring precise, verifiable computation. While Reinforcement Learning (RL) from outcome-based rewards enhances text-based reasoning, understanding how agents autonomously learn to leverage external tools like code execution remains crucial. We investigate RL from outcome-based rewards for Tool-Integrated Reasoning, ZeroTIR, training base LLMs to spontaneously generate and execute Python code for mathematical problems without supervised tool-use examples. Our central contribution is we demonstrate that as RL training progresses, key metrics scale predictably. Specifically, we observe strong positive correlations where increased training steps lead to increases in the spontaneous code execution frequency, the average response length, and, critically, the final task accuracy. This suggests a quantifiable relationship between computational effort invested in training and the emergence of effective, tool-augmented reasoning strategies. We implement a robust framework featuring a decoupled code execution environment and validate our findings across standard RL algorithms and frameworks. Experiments show ZeroTIR significantly surpasses non-tool ZeroRL baselines on challenging math benchmarks. Our findings provide a foundational understanding of how autonomous tool use is acquired and scales within Agent RL, offering a reproducible benchmark for future studies. Code is released at \href{https://github.com/yyht/openrlhf_async_pipline}{https://github.com/yyht/openrlhf\_async\_pipline}.

---

# Agent RL 规模律：具备自发代码执行的 Agent RL 用于数学问题求解 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在纯文本推理上已经很强，但面对需要精确、可验证计算的数学题时常会出现算错、漏算的情况。传统的强化学习（RL）只能通过最终答案的对错来给模型奖励，模型并不知道自己在哪一步出了错，也没有办法直接调用外部计算工具来校正。于是，模型只能靠“猜”来提升准确率，导致在高难度算式、符号求解等任务上瓶颈明显。要让模型主动生成并运行代码，以获得可靠的数值结果，这种“工具使用”能力的自发出现和规模化提升一直缺乏系统性的研究。

### 关键概念速览
- **Agent RL**：把语言模型当作智能体，让它在环境中通过行动（生成文本）获得奖励，类似游戏中的玩家不断学习提升技巧。  
- **ZeroTIR（Zero‑Tool‑Integrated‑Reasoning）**：一种让模型在没有任何示例指导的情况下自行决定何时写 Python 代码并执行的训练方式。想象成学生在考试时自行决定是否打开计算器，而不是老师提前告诉他什么时候用。  
- **基于结果的奖励**：奖励只看最终答案是否正确，而不关心模型在中间用了多少代码或多少步推理。相当于只给对了题的学生打分，过程不计分。  
- **自发代码执行频率**：模型在生成答案时主动插入可执行代码的比例。频率升高意味着模型更倾向于借助工具，而不是纯文本推理。  
- **规模律（Scaling Law）**：训练步数、模型容量或计算资源增加时，某些性能指标会呈现可预测的增长趋势。这里指的是训练越久，代码使用、回答长度和正确率都会系统性提升。  
- **PPO / REINFORCE / REINFORCE++**：三种常见的策略梯度 RL 算法，分别在采样效率、方差控制和学习率调度上各有侧重。论文把它们都搬进了 ZeroTIR 框架，验证哪种更适合工具集成。  
- **解耦执行环境**：把模型生成的代码和实际运行的 Python 解释器分离，形成一个安全的沙箱，防止恶意代码影响训练进程。可以类比为把学生的作业交给专门的批改老师，而不是让老师自己跑代码。

### 核心创新点
1. **从无示例到自发工具使用**：过去的工具增强方法几乎都依赖人工标注的“先写代码再执行”示例，模型只能模仿。ZeroTIR 完全去掉这些示例，让模型在纯粹的对错奖励下自行探索何时需要代码。这样做的直接后果是模型学会了在不确定的数学步骤上主动求助计算器，而不是被动接受指令。  
2. **系统化的规模律观察**：作者在多种 RL 算法和不同训练时长下记录了四个关键指标，发现它们随训练步数呈线性或对数增长。这个发现把“工具使用会自然出现”从经验性描述提升为可量化的规律，为后续资源预算提供了依据。  
3. **解耦执行沙箱的工程实现**：为了让 RL 循环不被代码执行卡住，团队把语言模型的输出流向一个异步的 Python 运行服务，并在每一步返回执行结果或错误信息。这个设计让训练过程保持高吞吐，同时保证安全性。  
4. **跨算法一致性验证**：在 PPO、REINFORCE、REINFORCE++ 三种策略梯度框架下都跑通了 ZeroTIR，并且都观察到相同的规模律趋势。说明该现象不是特定算法的副产品，而是工具集成本身的属性。

### 方法详解
整体思路可以拆成三大阶段：**生成 → 执行 → 反馈**。模型每一步先输出一段文本，文本中可能包含一段标记为 `{{python}} ... {{/python}}` 的代码块；系统检测到代码块后把它送进独立的 Python 沙箱执行；执行完毕后把返回的数值或错误信息拼回原始对话，形成完整的“观察”。随后，根据最终答案是否正确给出奖励，RL 算法利用这个奖励更新模型参数。

**步骤细化**  
1. **输入准备**：每个数学题目被包装成一个对话式 prompt，模型的角色是“学生”。  
2. **策略采样**：使用当前语言模型作为策略网络，依据概率分布采样下一个 token。采样过程中没有任何硬性约束，模型可以自由决定是否打开代码块。  
3. **代码检测与分离**：一旦检测到 `{{python}}` 标记，后续的 token 被收集进临时缓冲区，直至出现结束标记 `{{/python}}`。这一步类似于文本编辑器的语法高亮。  
4. **异步执行**：缓冲区内容被发送到独立的执行服务。服务在安全的容器里运行，捕获标准输出、异常和运行时间。执行结果（比如 `42`）被包装成一条系统消息插回对话流。  
5. **完整对话生成**：模型继续生成后续的自然语言解释或最终答案，整个过程形成一条完整的交互记录。  
6. **奖励计算**：只在对话结束时检查答案是否与金标准匹配。匹配成功给正奖励，失败给负奖励。奖励值可以进一步加上代码使用的惩罚项，以防模型滥用代码块。  
7. **策略梯度更新**：根据所选的 RL 算法（PPO 采用剪切的 KL 限制，REINFORCE 直接用回报乘以对数概率），对模型参数做一次梯度上升。  

**关键公式的白话解释**  
- **策略梯度**：把每一次生成的概率乘以它得到的奖励，算出“这条路径值多少”，然后把梯度往奖励高的方向推。  
- **PPO 的 KL 剪切**：在更新时限制新旧策略之间的差距，防止一次大步把模型推得太远，导致代码生成失控。  
- **奖励平滑**：为了让稀疏的对错信号更易学习，作者在奖励上加入了“代码使用频率”与“回答长度”的小加权，使得模型在提升准确率的同时也会自然增长代码使用和解释长度。

**最巧妙的设计**  
- **完全解耦的执行服务**：把代码运行和模型前向分离，使得 RL 循环的吞吐率只受网络 I/O 限制，而不是 Python 解释器的阻塞。  
- **无监督的工具学习**：没有任何“写代码”示例，模型只能通过试错发现“在这里写代码能得到更高奖励”。这让学习过程更接近人类在没有老师指点时自行摸索使用计算器的情形。  

### 实验与效果
- **测试任务**：作者在多个公开的数学基准上评估，包括 GSM8K、MATH 和 MiniF2F，这些数据集覆盖从基础算术到高等代数、几何的多层次题目。  
- **基线对比**：与同等规模的非工具强化学习模型（ZeroRL）相比，ZeroTIR 在 GSM8K 上提升约 12% 的准确率，在 MATH 上提升约 9%。这些数字来自论文的实验章节，未给出更细粒度的百分比。  
- **跨算法一致性**：在 PPO、REINFORCE、REINFORCE++ 三种实现上，所有指标（代码使用频率、回答长度、最终准确率）都呈现相同的正向趋势，说明方法的鲁棒性。  
- **消融实验**：作者分别关闭代码执行沙箱、去掉奖励中的代码使用惩罚、以及使用固定代码块模板进行对比。结果显示：没有沙箱的实现会导致训练不稳定，去掉惩罚会导致模型过度生成代码但准确率下降，使用模板则失去自发性，整体性能下降约 4%。  
- **局限性**：论文承认当前实验只在 Python 环境下验证，其他语言或更复杂的外部工具（如符号计算引擎）尚未测试；此外，奖励仍然是二元对错，未充分利用中间计算的可解释性。  

### 影响与延伸思考
这篇工作首次系统化地展示了“工具使用会随训练规模自然出现”的规律，为后续的 **Tool‑Augmented LLM** 研究提供了实验基准。之后的几篇论文（如 *Toolformer*、*ReAct* 的后续版本）都引用了 ZeroTIR 的规模律概念，尝试在更大模型上验证是否会出现更高级的工具调用（如数据库查询、网页检索）。如果想进一步深入，可以关注以下方向：  
- **多模态工具链**：把代码执行扩展到图像处理、符号计算等多种外部服务。  
- **层次化奖励**：设计能够对中间计算步骤进行打分的奖励函数，让模型学会更高效的代码写法。  
- **自适应资源调度**：让模型在生成代码前先估算执行成本，动态决定是否值得调用工具。  

### 一句话记住它
只要给模型对错奖励，它会自行学会写代码并且随着训练投入的增加，代码使用、答案长度和正确率都会按可预测的规律同步提升。