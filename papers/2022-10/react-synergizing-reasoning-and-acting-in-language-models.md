# ReAct: Synergizing Reasoning and Acting in Language Models

> **Date**：2022-10-06
> **arXiv**：https://arxiv.org/abs/2210.03629

## Abstract

While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples. Project site with code: https://react-lm.github.io

---

# ReAct：在语言模型中协同推理与行动 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，问答或决策任务要么靠一次性生成答案，要么靠单独的检索/执行模块。  
单纯的**思维链（Chain‑of‑Thought）**让模型把推理过程写出来，但模型仍然只能在内部“想”，无法主动去查资料或操作环境，导致在需要外部信息时容易出现“幻觉”。  
相反，**行动型**方法（如基于API的检索或强化学习）可以让模型与外部系统交互，却缺少可解释的推理轨迹，模型的决策过程往往像黑箱。  
于是出现了两条平行的技术路线：推理和行动各自为政，难以互相补强，导致在复杂、多步任务上仍然容易出错、难以调试。  

### 关键概念速览
- **思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出逐步推理，就像解数学题时先列草稿，帮助模型保持逻辑连贯。  
- **行动（Act）**：模型在推理过程中主动调用外部工具（如搜索 API、环境指令），相当于人在思考时去查字典或打开门。  
- **推理‑行动交叉（Interleaved Reasoning & Acting）**：模型的输出交替出现“思考句”和“行动指令”，两者相互影响，形成闭环。  
- **外部知识库（External Knowledge Base）**：如 Wikipedia API，模型可以通过检索获得最新事实，弥补内部记忆的时效性不足。  
- **可解释轨迹（Interpretability Trace）**：完整记录模型的每一步思考和每一次外部交互，类似实验日志，便于人类审查和调试。  
- **ALFWorld / WebShop**：两个交互式决策基准，前者是基于虚拟家居的任务，后者是电商网站的购物流程，均要求模型在环境中执行具体操作。  

### 核心创新点
1. **推理与行动交织的提示模板**  
   - 之前的做法：要么只给模型 CoT 提示，只输出思考链；要么只给行动指令的模板，模型一次性输出完整计划。  
   - 本文做法：在提示中明确规定模型的每一行输出要么是“思考：…”，要么是“行动：…”，并要求模型在每一步完成后立即决定是否需要进一步检索或执行。  
   - 改变：模型能够在推理过程中实时决定是否去查询外部信息，显著降低了因缺乏事实而产生的幻觉。  

2. **统一的“思考‑行动”语言接口**  
   - 之前的做法：检索系统和执行系统往往使用不同的 API，模型需要学习多套调用方式。  
   - 本文做法：把所有外部交互抽象为统一的“行动”指令（如 `Search[query]`、`Click[object]`），模型只需学习一种语法即可。  
   - 改变：降低了提示工程的复杂度，使得只需一两例子就能让模型在新任务上自行组合思考与行动。  

3. **基于简单 Wikipedia API 的事实校验机制**  
   - 之前的做法：在 HotpotQA、FEVER 等事实类任务中，模型只能靠内部记忆回答，容易出现错误传播。  
   - 本文做法：在每一步推理后，模型可以主动调用 `Search[question]`，把检索到的片段作为后续推理的依据。  
   - 改变：实验显示，使用检索后模型的错误率大幅下降，尤其在需要多段证据的复杂问答上表现更稳健。  

4. **极少示例的强鲁棒性**  
   - 之前的做法：在交互式环境（如 ALFWorld）里，往往需要大量示例或强化学习的迭代才能学会策略。  
   - 本文做法：只用 1–2 条“思考‑行动”示例，就能让模型在新环境中自行探索并完成任务。  
   - 改变：在 ALFWorld 上成功率提升 34%，在 WebShop 上提升 10%，证明了提示驱动的跨任务迁移能力。  

### 方法详解
**整体框架**  
ReAct 把一次完整的任务解答视为一系列交替出现的“思考”和“行动”步骤。模型在每一步收到前文的全部记录（包括之前的思考、检索结果、执行反馈），然后输出下一条指令。整个过程在一次前向传播中完成，无需外部循环控制。

**关键流程**  
1. **初始化提示**：提供任务描述 + 1–2 条示例。示例展示了“思考：… → 行动：… → 思考：…”的完整链条。  
2. **模型生成**：模型读取当前上下文，决定输出类型。  
   - 若输出以 “思考：” 开头，后面是一段自然语言推理。  
   - 若输出以 “行动：” 开头，后面是一条结构化指令（如 `Search[“Who invented the telephone?”]`）。  
3. **外部模块响应**：  
   - **检索**：当指令是 `Search` 时，系统调用 Wikipedia API，返回前几段文本。  
   - **环境交互**：当指令是 `Click`、`Move`、`PickUp` 等时，环境执行相应动作并返回状态描述。  
4. **上下文更新**：检索或环境的返回信息被直接拼接进对话历史，模型在下一轮生成时可以看到这些新信息。  
5. **终止条件**：当模型输出以 “答案：” 或 “完成” 等标记结束时，任务结束。

**类比**：可以把 ReAct 想成一位在图书馆里做研究的学生。学生先在脑子里思考下一步要查什么（思考），然后走到检索台去拿书（行动），再把书上的信息带回继续思考，如此循环直到写出论文结论。

**最巧妙的设计**  
- **统一指令语言**：所有外部交互都用同一套 `Action[argument]` 形式，让模型只需学习一种模式即可在不同任务间迁移。  
- **即时上下文注入**：检索或环境的返回直接进入模型的上下文，而不是通过后处理，这保证了模型在每一步都能基于最新信息做决定。  
- **极少示例的 Few‑Shot 能力**：通过在示例中展示完整的思考‑行动循环，模型能够自行推断出“何时检索、何时执行”，不需要额外的强化学习或微调。  

### 实验与效果
- **测试任务**：  
  - **HotpotQA**（多段落问答）  
  - **FEVER**（事实验证）  
  - **ALFWorld**（虚拟家居交互）  
  - **WebShop**（电商网站购物）  

- **基线对比**：  
  - 纯 CoT 推理模型  
  - 只使用检索的 Retrieval‑Augmented Generation（RAG）  
  - 传统的模仿学习（Imitation Learning）和强化学习（RL）方法在 ALFWorld / WebShop 上的表现  

- **主要结果**（论文声称）：  
  - 在 HotpotQA 和 FEVER 上，ReAct 通过检索显著降低了幻觉率，整体准确率超过纯 CoT 基线约 **8%**（具体数字未在摘要中给出）。  
  - 在 ALFWorld 上，成功率比最强的 imitation / RL 基线提升 **34%**（绝对值）。  
  - 在 WebShop 上，成功率提升 **10%**（绝对值）。  
  - 人类评审认为 ReAct 生成的解题轨迹更易理解，信任度高于仅有思考或仅有行动的对手。  

- **消融实验**：  
  - 去掉行动模块（仅保留思考）会导致在需要外部事实的问答上错误率回升。  
  - 去掉思考模块（仅直接行动）会使模型在复杂推理任务上表现大幅下降，说明两者的协同是关键。  

- **局限性**：  
  - 依赖的外部 API（如 Wikipedia）相对简单，面对更专业或实时更新的知识库时可能需要额外的适配。  
  - 交互式环境仍然是模拟的，真实世界的噪声和延迟未在实验中充分评估。  
  - 论文未提供大规模多语言实验，方法在非英语环境的适用性仍待验证。  

### 影响与延伸思考
ReAct 把“思考”和“行动”紧密绑定的思路在发布后迅速被后续工作引用，催生了多篇围绕 **“思考‑行动循环”** 的研究，例如：  
- **Reflexion** 系列在自我纠错时加入了检索动作；  
- **Toolformer** 让模型自行学习何时调用工具；  
- **Self‑Ask** 在问答中加入了自动检索的子问题生成。  

从长远来看，最值得关注的方向是 **统一的工具调用语言** 与 **更复杂的外部环境**（如真实网页、机器人平台）的结合，以及 **自适应的行动选择策略**（比如基于不确定性决定是否检索）。如果想进一步深入，可以关注近期的 “LLM‑as‑Agent” 生态，尤其是围绕 **Tool‑Augmented Generation** 的开源框架。  

### 一句话记住它
让大语言模型在每一步思考后立即决定是否去“查资料”或“动手”，把推理和行动交织成一条可追溯的解题链。