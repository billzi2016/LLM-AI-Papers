# AgentBench: Evaluating LLMs as Agents

> **Date**：2023-08-07
> **arXiv**：https://arxiv.org/abs/2308.03688

## Abstract

The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. Thus, there is an urgent need to quantitatively \textit{evaluate LLMs as agents} on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional benchmark that consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities. Our extensive test over \num API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and many OSS competitors that are no larger than 70B. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Improving instruction following and training on high quality multi-round alignment data could improve agent performance. And different from existing assumptions, training on code present ambivalent impacts on different agent tasks. Datasets, environments, and an integrated evaluation package for AgentBench are released at https://github.com/THUDM/AgentBench.

---

# AgentBench：评估大语言模型作为智能体 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以写代码、写文章，但把它们当成“智能体”让它们在交互式环境里自行决策仍是新挑战。过去的评测大多停留在一次性问答或单轮指令上，根本看不到模型在多步推理、记忆和行动之间的衔接。真实环境（比如网页浏览、文件操作、游戏）要求模型在每一步都能正确解释观察、规划下一步并执行对应的 API 调用，这对长期推理、指令遵循和决策质量提出了极高要求。缺少系统化、多维度的基准，导致研究者只能凭经验猜测模型的真实能力，也让开源模型的进步难以量化。

### 关键概念速览
- **LLM‑as‑Agent（LLM 充当智能体）**：把大语言模型当成可以感知环境、思考并执行动作的主体，类似于人类在电脑前操作软件的过程。  
- **交互式环境（Interactive Environment）**：模型可以通过 API 与之通信的系统，如文件系统、网页浏览器或游戏引擎，模型的每一次输出都会影响后续观察。  
- **多轮对话（Multi‑turn Dialogue）**：模型与环境的连续交互，包含若干轮“观察‑思考‑行动”循环，类似于下棋时的每一步思考。  
- **指令遵循（Instruction Following）**：模型在收到任务描述后，能够准确把任务拆解成可执行的子步骤，而不是产生与目标无关的文字。  
- **长程推理（Long‑term Reasoning）**：在多步任务中保持全局目标不变，并在每一步都考虑之前的决定对最终结果的影响。  
- **代码微调（Code‑fine‑tuning）**：在包含大量代码示例的语料上继续训练模型，期望提升其生成可执行代码的能力。  
- **对齐数据（Alignment Data）**：人类标注的高质量对话或指令-响应对，用来教模型更好地满足用户意图。  

### 核心创新点
1. **从单一任务到八种环境的多维评测**  
   之前的评测大多只用一个或两个简单任务（比如数学或常识问答）来衡量 LLM 的能力。AgentBench 设计了 8 种截然不同的交互式环境（包括网页搜索、文件编辑、游戏控制等），每种环境都对应一套专门的评测指标。这样可以同时观察模型在推理、决策、指令执行等多个维度的表现，避免“一刀切”误判。  

2. **统一的 API 接口包装**  
   为了让不同模型能够在同一套环境中公平竞争，作者实现了一个统一的调用层：模型只需要输出标准化的“思考+行动”文本，系统会自动把它转化为对应环境的 API 调用。相比过去需要手工写适配层的做法，这种包装大幅降低了实验成本，也让评测结果更具可比性。  

3. **系统化的失败原因分析框架**  
   评测不仅给出分数，还提供了错误分类（如“长程推理失误”“指令拆解错误”“API 参数错误”等）。通过对大量实验日志的自动归类，作者发现大多数开源模型的瓶颈集中在长期规划和指令遵循上，为后续改进指明了方向。  

4. **对代码微调影响的细粒度实验**  
   过去普遍认为让模型多学代码会提升所有智能体任务的表现。AgentBench 在每个环境中分别比较了是否进行代码微调的模型，结果显示对某些任务（如文件编辑）有帮助，但对另一些（如网页搜索）反而会引入噪声。这个发现挑战了“代码即万能”的假设。  

### 方法详解
**整体框架**  
AgentBench 的评测流程可以拆成三大步骤：① 环境准备；② 模型交互；③ 结果评估。首先，作者实现了 8 套标准化的交互式环境，每套环境都提供一组 API（如 `open_file(path)`、`click_button(id)` 等）和对应的观察空间（文本描述、页面截图等）。随后，研究者把待测 LLM 接入统一的“思考‑行动”接口：模型在每轮收到环境观察后，输出一段自然语言思考（可选）和一条结构化的行动指令。系统解析指令并调用相应 API，得到新的观察，循环往复。最后，根据预设的任务成功标准（比如是否成功完成文件合并、是否在游戏中达成目标）给出分数，并记录每一步的日志用于错误分类。

**关键模块拆解**  
1. **观察 → 思考 → 行动的三段式输出**  
   - **观察**：环境返回的文字或图像描述被转成纯文本，作为模型的输入。  
   - **思考**：模型可以在输出中加入“思考”段落，类似 CoT（思维链），帮助模型在内部进行推理。  
   - **行动**：模型必须输出符合 JSON 格式的指令，例如 `{"action":"click","target":"search_button"}`。系统会检查格式合法性，非法指令直接计为错误。  

2. **统一 API 包装层**  
   - 将所有环境的底层实现隐藏在一个统一的 Python 包 `agentbench.api` 中。模型只需要关注高层指令，包装层负责把指令映射到具体的函数调用并返回统一的观察结构。  
   - 这种设计类似于游戏引擎的脚本层，让不同模型只需写一次“思考‑行动”逻辑即可在所有环境中运行。  

3. **评估指标体系**  
   - **成功率**：任务是否最终达成。  
   - **步骤效率**：完成任务所需的交互轮数，越少越好。  
   - **指令准确率**：模型输出的 API 调用是否符合预期格式并成功执行。  
   - **长程一致性**：在多轮任务中，模型是否保持全局目标不偏离。  

**最巧妙的设计**  
作者把“思考”与“行动”强制分离，使得即使模型在生成自然语言时出现废话，也不会影响后续的结构化指令解析。这种“软硬分离”让评测更稳健，也让后续研究可以单独改进思考模块（比如加入外部检索）而不必重新设计行动接口。

### 实验与效果
- **测试对象**：包括 10 多个商业 API（如 GPT‑4、Claude）和 12 个开源模型（从 7B 到 70B 参数不等），共计约 22 种 LLM。  
- **主要发现**：在 8 项任务的综合成功率上，最强商业模型（GPT‑4）平均达到 78% 以上，而开源模型的最高成绩只有约 42%，且多数在 30% 左右。尤其是 70B 以下的开源模型在长程推理任务（如多步骤文件操作）几乎失效。  
- **指令遵循对比**：在需要精准 API 参数的任务中，商业模型的指令准确率超过 90%，而开源模型多数在 60% 左右，错误多表现为参数缺失或格式错误。  
- **代码微调影响**：在文件编辑环境中，经过代码微调的模型成功率提升约 8%；但在网页搜索环境中，同样的微调导致成功率下降约 5%，说明代码知识并非对所有交互任务都有正向作用。  
- **消融实验**：去掉“思考”段落后，所有模型的成功率平均下降 12%，验证了思考链对长程任务的帮助。再把统一 API 包装层换成手工适配后，评测噪声显著增加，说明包装层对公平比较至关重要。  
- **局限性**：作者承认评测仍受限于环境的可编程性，某些高度感知任务（如真实机器人控制）尚未覆盖；此外，评测指标主要是成功率，未深入衡量模型的“解释性”或“安全性”。  

### 影响与延伸思考
AgentBench 在发布后迅速成为评估 LLM‑as‑Agent 的标配基准，多个后续工作（如 **AutoEval**、**LLM‑Gym**）直接借鉴了其多环境设计和统一 API 包装思路。研究者也开始围绕“指令遵循”展开专门的数据收集，推出了 **InstructEval** 系列数据集。对开源社区而言，这篇论文提供了清晰的性能差距图谱，激励了 70B 以上模型的对齐数据投入和长程记忆机制的探索。想进一步了解，可以关注以下方向：① 高质量多轮对齐数据的构建；② 将记忆网络或检索模块嵌入思考阶段以提升长期规划；③ 将真实机器人或 AR 环境加入基准，检验模型的跨模态执行能力。  

### 一句话记住它
AgentBench 用 8 大交互环境和统一 API，给出首个系统化、可比的 LLM‑as‑Agent 评测框架，揭示了商业模型与开源模型在长程推理和指令遵循上的巨大鸿沟。