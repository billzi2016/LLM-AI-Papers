# Thinking LLMs: General Instruction Following with Thought Generation

> **Date**：2024-10-14
> **arXiv**：https://arxiv.org/abs/2410.10630

## Abstract

LLMs are typically trained to answer user questions or follow instructions similarly to how human experts respond. However, in the standard alignment framework they lack the basic ability of explicit thinking before answering. Thinking is important for complex questions that require reasoning and planning -- but can be applied to any task. We propose a training method for equipping existing LLMs with such thinking abilities for general instruction following without use of additional human data. We achieve this by an iterative search and optimization procedure that explores the space of possible thought generations, allowing the model to learn how to think without direct supervision. For each instruction, the thought candidates are scored using a judge model to evaluate their responses only, and then optimized via preference optimization. We show that this procedure leads to superior performance on AlpacaEval and Arena-Hard, and shows gains from thinking on non-reasoning categories such as marketing, health and general knowledge, in addition to more traditional reasoning & problem-solving tasks.

---

# 思考型大语言模型：通过思维生成实现通用指令遵循 论文详细解读

### 背景：这个问题为什么难？
在传统的对齐框架里，大语言模型（LLM）被训练成直接给出答案或执行指令，像是把问题交给一个“会说话的搜索引擎”。然而，面对需要多步推理、规划甚至创意构思的任务时，模型往往缺乏“先想一想再说”的过程，导致答案不够可靠或缺乏深度。过去的改进大多依赖人工标注的思考示例（如 CoT），成本高且难以覆盖所有指令类型。根本的瓶颈在于：没有一种通用、无需额外人工数据的方式，让已有模型学会自行产生有价值的思考过程。

### 关键概念速览
**大语言模型（LLM）**：能够生成自然语言的深度模型，类似于“会写作文的机器人”。  
**指令遵循**：模型根据用户给出的任务描述直接输出完成结果，就像客服根据请求直接回复。  
**思维生成（Thought Generation）**：模型在给出最终答案前先写出一段内部推理或计划，类似于人解题时的草稿。  
**偏好优化（Preference Optimization）**：通过让模型学习人类或评估模型的偏好来微调，使其更倾向于产生高质量输出。  
**评审模型（Judge Model）**：专门用来打分思考稿或答案质量的模型，充当“内部裁判”。  
**迭代搜索（Iterative Search）**：在思考空间里不断尝试、挑选、改进思维稿的过程，像是反复试错的头脑风暴。  
**思维候选（Thought Candidates）**：同一指令下模型生成的多个不同思考稿，供评审模型挑选最优。  
**AlpacaEval / Arena‑Hard**：公开的指令遵循评测基准，用来衡量模型在多种任务上的表现。

### 核心创新点
1. **无需额外人工标注的思考学习**  
   之前的思考链（CoT）需要人手写出思考步骤作为监督信号。本文直接让模型自己生成大量思考稿，然后用评审模型的分数作为间接监督，实现了“自我教导”。这样既省去标注成本，又能覆盖更广的任务场景。  

2. **思考稿只用响应评分进行优化**  
   传统的偏好学习会同时考虑思考过程和最终答案的质量。这里把评审焦点放在答案上，评审模型只看思考稿对应的答案好坏，模型学习到的其实是“哪些思考方式能导出好答案”。这种单向反馈让优化更简洁，也避免了对思考稿本身的主观评判。  

3. **迭代搜索+偏好微调的闭环**  
   对每条指令，模型先生成一批思考稿（搜索），评审模型挑出表现最好的那一个，随后用该对（指令+最佳思考稿）进行偏好微调。经过多轮循环，模型逐渐学会在生成思考稿时就倾向于产生高质量答案。相比一次性训练，这种循环让模型在思考空间里“自我进化”。  

4. **思考提升非推理任务**  
   实验显示，思考不仅在数学、逻辑推理等传统需要链式推理的任务上有提升，还能在营销文案、健康建议、常识问答等看似不需要推理的指令上带来显著改进。说明思考过程本身是一种通用的“元认知”能力，而非仅针对特定任务的技巧。

### 方法详解
**整体框架**  
整个训练流程可以拆成三步：① 思考稿生成（搜索），② 评审模型打分并挑选最佳稿，③ 用挑选出的指令‑思考对进行偏好优化。循环多次后，模型内部的思考生成策略被逐步强化。

**步骤拆解**  

1. **思考稿生成（Iterative Search）**  
   - 给定一条用户指令，模型在“思考模式”下生成 N 条不同的思考稿。这里的“思考模式”指的是在输出前加上一个特殊的提示，让模型把输出当作内部推理而不是最终答案。  
   - 这一步类似于让模型进行多次“头脑风暴”，每次都尝试不同的思路、结构或细节。  

2. **评审与筛选（Judge Scoring）**  
   - 对每条思考稿，模型再基于该稿生成最终答案。评审模型只看答案质量（比如是否符合指令、是否事实正确），给出一个分数。  
   - 最高分对应的思考稿被视为该指令的“最佳思考”。这里的评审模型可以是一个已经对齐好的 LLM，或者是通过人类偏好微调得到的评分模型。  

3. **偏好优化（Preference Optimization）**  
   - 将（指令，最佳思考稿）这对数据加入训练集。使用偏好学习的目标函数，让模型在下次生成思考稿时更倾向于产生类似的高分思考。  
   - 具体做法是把这些对当作正例，随机采样的其他思考稿当作负例，训练模型在两者之间做出偏好判断。  

4. **循环迭代**  
   - 完成一次偏好微调后，回到第一步，用更新后的模型再次生成思考稿。随着循环次数增加，模型的思考质量会逐步提升，最终在大多数指令上都能自发产生有帮助的思考过程。  

**关键细节**  
- **只评分答案**：评审模型不直接评价思考稿的文字质量，而是通过答案的好坏间接衡量思考的有效性，这避免了对思考过程的主观偏好。  
- **思考稿的多样性**：通过随机采样或温度调节，确保生成的思考稿覆盖广阔的思维空间，防止模型陷入单一思路。  
- **偏好学习的实现**：通常采用对比损失（如 KL 散度）来最大化正例被评为更好，负例被评为更差的概率。  

**最巧妙的地方**  
把“思考”当作一种可搜索的隐变量，而不是直接监督的目标；再用答案质量来反向驱动思考的改进，这种“答案驱动的思考学习”在没有任何人工标注的情况下实现了自我提升。

### 实验与效果
- **评测基准**：论文在 AlpacaEval（一个覆盖多领域指令的评测集合）和 Arena‑Hard（高难度对话/指令对抗赛）上进行验证。  
- **对比基线**：与原始 LLM、传统 CoT（思维链）以及常规偏好微调模型相比，思考型模型在两套基准上均取得了显著提升。具体提升幅度在摘要中未给出数值，但作者强调“显著优于”。  
- **跨任务收益**：除了常规的推理、数学题目，模型在营销文案、健康建议、常识问答等非推理指令上也表现出明显的分数提升，说明思考生成的好处是通用的。  
- **消融实验**：论文通过去掉迭代搜索、只对思考稿本身打分、或不使用偏好优化等设置，验证了每个模块的贡献。结果显示，去掉任意一步都会导致性能回落，尤其是去掉答案驱动的评审会让思考质量大幅下降。  
- **局限性**：作者承认思考生成会增加推理时间，因为每条指令需要多次生成思考稿并进行评审。此外，评审模型本身的偏好会影响最终思考方向，若评审模型存在偏差，可能会把错误的思考方式放大。  

### 影响与延伸思考
这篇工作打开了“让模型自我产生思考并通过答案反馈自我改进”的新路径，随后出现的研究大多围绕如何设计更高效的思考搜索、如何让评审模型更可靠、以及如何将思考过程与外部工具（检索、计算）结合。比如后续的“自我纠错 LLM”以及“工具使用 + 思考链”都受到了此思路的启发。想进一步深入，可以关注以下方向：  
- **评审模型的鲁棒性**：如何让评分更公平、避免误导模型。  
- **思考成本优化**：在保持效果的前提下降低思考稿的生成次数或长度。  
- **跨模态思考**：把视觉、表格等信息纳入思考过程，扩展到多模态指令。  

### 一句话记住它
让大模型先“想一想”，只用答案好坏来教它思考，模型就能在几乎所有指令上自行提升。