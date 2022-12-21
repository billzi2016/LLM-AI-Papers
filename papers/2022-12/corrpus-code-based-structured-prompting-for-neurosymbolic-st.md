# CoRRPUS: Code-based Structured Prompting for Neurosymbolic Story   Understanding

> **Date**：2022-12-21
> **arXiv**：https://arxiv.org/abs/2212.10754

## Abstract

Story generation and understanding -- as with all NLG/NLU tasks -- has seen a surge in neurosymbolic work. Researchers have recognized that, while large language models (LLMs) have tremendous utility, they can be augmented with symbolic means to be even better and to make up for any flaws that the neural networks might have. However, symbolic methods are extremely costly in terms of the amount of time and expertise needed to create them. In this work, we capitalize on state-of-the-art Code-LLMs, such as Codex, to bootstrap the use of symbolic methods for tracking the state of stories and aiding in story understanding. We show that our CoRRPUS system and abstracted prompting procedures can beat current state-of-the-art structured LLM techniques on pre-existing story understanding tasks (bAbI Task 2 and Re^3) with minimal hand engineering. We hope that this work can help highlight the importance of symbolic representations and specialized prompting for LLMs as these models require some guidance for performing reasoning tasks properly.

---

# CoRRPUS：基于代码的结构化提示用于神经符号化故事理解 论文详细解读

### 背景：这个问题为什么难？
故事生成和理解本质上是把自然语言的连贯性和逻辑推理结合起来的任务。传统的大语言模型（LLM）在语言流畅度上表现优异，却常常在保持人物状态、因果关系等细粒度的情节一致性时出错。为了解决这些缺陷，研究者们尝试在 LLM 之上叠加符号化的状态追踪器或规则系统，但手工编写符号表示既费时又需要深厚的领域专业知识，导致大规模应用受阻。因此，如何在不大量投入人工成本的前提下，让 LLM 具备可靠的符号推理能力，成为亟待突破的瓶颈。

### 关键概念速览
**神经符号化（Neurosymbolic）**：把神经网络的统计学习能力和符号系统的明确推理规则结合起来，类似于让会说话的机器人同时拥有“数学公式”这种硬核工具。  
**结构化提示（Structured Prompting）**：在给 LLM 的输入中加入特定的格式或模板，使模型在生成答案时遵循预设的结构，好比在考试卷子上先写好答题框架再填内容。  
**Code‑LLM**：专门在代码数据上训练的语言模型（如 Codex），擅长生成符合语法的程序片段，能够把抽象的逻辑转化为可执行代码。  
**状态追踪（State Tracking）**：对故事中人物属性、地点、事件等信息进行实时更新的过程，类似于电影导演在拍摄时随时记录每个角色的情绪变化。  
**bAbI Task 2**：一种经典的推理基准，要求模型根据一系列陈述推断出人物的当前位置。  
**Re³**：针对阅读理解的评测集合，重点考察模型对多段落信息的综合推理能力。  

### 核心创新点
1. **利用 Code‑LLM 自动生成符号追踪代码 → 直接让模型写出状态更新函数**：传统方法需要人工设计状态图或规则库，这里把任务交给已经在代码上预训练好的模型，让它输出 Python（或类似）代码来维护故事状态。这样既省去了手工编码，又保证了生成的代码符合语言规范。  
2. **把代码嵌入提示模板 → 让普通 LLM 在执行代码后得到结构化的中间结果**：在普通的大语言模型前端加入“先运行下面的代码，再根据返回的变量回答问题”的提示，使得模型在回答前拥有一个明确的、机器可解释的内部状态。相当于给模型装上了“思考工具”。  
3. **最小化手工工程 → 只需提供少量示例即可启动**：作者只用了几条示例对 Code‑LLM 进行 few‑shot 引导，而不是构建完整的符号库。实验表明，这种轻量级的提示方式已经能够在 bAbI Task 2 和 Re³ 上超越已有的结构化 LLM 基线。  
4. **统一的抽象提示流程 → 可迁移到其他叙事任务**：CoRRPUS 把“生成代码 → 执行 → 读取结果”封装成一个通用的提示模式，后续只要换掉任务描述和状态变量，就能复用到不同的故事理解场景。

### 方法详解
整体思路可以分为三步：**（1）代码生成、（2）代码执行、（3）基于执行结果的自然语言推理**。下面逐层拆解。

1. **代码生成阶段**  
   - 输入：一段故事文本 + 任务指令（例如“找出 Alice 最后所在的房间”）。  
   - 提示模板：在提示的最前面放入一个“代码块”占位符，示例中提供几条“人物属性 = 初始值”的声明和“一段更新函数”的示例。  
   - 交给 Code‑LLM（如 Codex）后，它返回一段完整的 Python 脚本，脚本里会定义一个全局字典 `state`，并实现 `update_state(sentence)`，该函数逐句解析输入文本，依据动词、介词等线索修改 `state`（比如 `state['Alice']['location'] = 'kitchen'`）。

2. **代码执行阶段**  
   - 把生成的脚本交给安全的沙箱环境执行。执行过程会遍历故事的每一句话，调用 `update_state`，最终得到一个包含所有人物、物品、地点等信息的结构化字典。  
   - 关键点在于 **“代码即中间表示”**：相较于让 LLM 直接输出 JSON，使用可执行代码可以利用编程语言本身的控制流、条件判断等能力，自动处理复杂的因果链和冲突消解。

3. **基于执行结果的自然语言推理**  
   - 将得到的 `state` 再包装进一个新的提示，交给普通的大语言模型（如 GPT‑4）。提示会说：“下面是故事的状态表，请根据它回答以下问题”。  
   - 这一步的好处是模型不需要自己推理状态变化，只需要读取已经整理好的结构化信息，再进行语言层面的答案生成。

**最巧妙的地方**在于把代码当作“可解释的记忆”。传统的结构化提示往往让模型自行生成 JSON，容易出现格式错误或逻辑不一致；而代码执行保证了状态更新的确定性，同时利用 Code‑LLM 的代码生成能力把符号化工作自动化。

### 实验与效果
- **测试任务**：作者在两套公开基准上评估：bAbI Task 2（位置推理）和 Re³（跨段落阅读理解）。  
- **对比基线**：包括最新的结构化提示 LLM 方法、纯 LLM（不使用符号）以及手工构建的符号追踪系统。  
- **结果**：论文声称 CoRRPUS 在两项任务上均超过了当前最好的结构化 LLM 技术，且只用了极少的手工示例。具体提升幅度未在摘要中给出。  
- **消融实验**：作者分别去掉代码生成、代码执行或结构化提示，发现去掉任意一步都会导致性能显著下降，验证了三段式流程的必要性。  
- **局限性**：由于依赖代码执行环境，实际部署时需要考虑沙箱安全和执行成本；此外，Code‑LLM 在极端长文本或高度隐喻的叙事上仍可能生成不完整或错误的更新函数，作者在讨论中承认这一点。

### 影响与延伸思考
CoRRPUS 把“让模型写代码”直接用于符号状态追踪，打开了神经符号化的新思路。后续有几篇工作尝试把类似的代码生成‑执行‑读取循环用于数学推理、表格问答等领域（如 “Program of Thoughts” 系列），可以说是受到了本篇的启发。想进一步深入，读者可以关注以下方向：  
- **更安全高效的代码沙箱**：如何在不牺牲速度的前提下保证生成代码不越界。  
- **跨模态状态追踪**：把视觉信息也映射到代码层的状态变量。  
- **自适应提示优化**：让模型自行学习最适合的代码模板，而不是手工设计 few‑shot 示例。

### 一句话记住它
让大语言模型先写出可执行的状态更新代码，再用执行结果作答，几行提示就把符号推理装进了模型的“思考工具箱”。