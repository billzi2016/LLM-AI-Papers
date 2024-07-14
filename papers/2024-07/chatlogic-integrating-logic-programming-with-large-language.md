# ChatLogic: Integrating Logic Programming with Large Language Models for   Multi-Step Reasoning

> **Date**：2024-07-14
> **arXiv**：https://arxiv.org/abs/2407.10162

## Abstract

Large language models (LLMs) such as ChatGPT and GPT-4 have demonstrated impressive capabilities in various generative tasks. However, their performance is often hampered by limitations in accessing and leveraging long-term memory, leading to specific vulnerabilities and biases, especially during long interactions. This paper introduces ChatLogic, an innovative framework specifically targeted at LLM reasoning tasks that can enhance the performance of LLMs in multi-step deductive reasoning tasks by integrating logic programming. In ChatLogic, the language model plays a central role, acting as a controller and participating in every system operation stage. We propose a novel method of converting logic problems into symbolic integration with an inference engine. This approach leverages large language models' situational understanding and imitation skills and uses symbolic memory to enhance multi-step deductive reasoning capabilities. Our results show that the ChatLogic framework significantly improves the multi-step reasoning capabilities of LLMs. The source code and data are available at \url{https://github.com/Strong-AI-Lab/ChatLogic}

---

# ChatLogic：将逻辑编程与大语言模型结合用于多步推理 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在对话、写作等生成任务上已经很强，但它们在需要多轮、跨时段的推理时常常“忘记”前面的信息。传统的提示工程（prompt engineering）只能让模型在一次前向传播里记住有限的上下文，长对话或复杂的演绎链会导致答案漂移、逻辑漏洞，甚至出现系统性偏见。换句话说，LLM缺少一种可靠的“长期记忆”和可验证的推理机制，这让它们在数学证明、程序验证等需要严格多步推理的场景里表现不佳。

### 关键概念速览
- **大语言模型（LLM）**：基于海量文本训练的神经网络，能够生成自然语言。把它想象成一个“会说话的统计机器”，擅长捕捉语言模式但不具备显式的推理规则。
- **逻辑编程**：用一组规则（如Prolog的Horn子句）描述事实与推理过程，机器可以通过搜索得到严格的结论。类似于在纸上写下“如果…则…”的推理链，然后让计算机自动演绎。
- **符号记忆（symbolic memory）**：把推理过程、事实或中间结论以结构化的符号形式保存，类似于在笔记本上记下每一步的草稿，后续可以随时查阅或修改。
- **控制器（controller）**：在ChatLogic里，LLM本身承担指挥官的角色，决定何时查询符号引擎、何时生成自然语言输出。可以把它比作一位“导演”，既写剧本又指挥演员（推理引擎）演出。
- **推理引擎（inference engine）**：执行逻辑程序的核心组件，接受符号化的查询并返回满足规则的答案。它像是一个“裁判”，只会给出符合逻辑的判决。
- **CoT（Chain‑of‑Thought）**：让模型在给出最终答案前先把思考过程写出来，类似于解题时的草稿纸。ChatLogic把CoT的思路延伸到符号层面，让每一步都可以被引擎验证。

### 核心创新点
1. **LLM 兼任控制器 → 将语言模型直接嵌入推理循环**  
   传统方案往往把 LLM 当作“前端”，把逻辑引擎当作“后端”，两者之间缺乏紧密交互。ChatLogic 让 LLM 在每一步都决定是否调用符号引擎、如何构造查询，并把引擎返回的符号结果再喂回模型。这样模型的语言理解与符号推理形成闭环，显著提升了长链推理的连贯性。

2. **逻辑问题的符号化转译 → 从自然语言到可执行的逻辑程序**  
   作者提出一种自动化的“翻译器”，把用户的自然语言描述（如“所有鸟都会飞，企鹅是鸟，企鹅会不会飞？”）转成 Prolog‑style 的事实与规则。相比手工编码，这一步降低了使用门槛，也保证了符号引擎能够直接操作问题的核心结构。

3. **符号记忆与 LLM 记忆的双层存储**  
   在推理过程中，所有中间结论都会写入符号记忆，而 LLM 只保留对话上下文。这样即使对话变长，模型仍然可以通过查询符号记忆快速恢复之前的推理状态，避免了纯神经网络的“遗忘”问题。

4. **统一的多步推理框架 → 兼容多种任务**  
   ChatLogic 的整体流程（语言理解 → 符号转译 → 引擎求解 → 结果回写 → 语言生成）可以直接套用到数学证明、程序分析、常识推理等不同领域。相比只针对单一任务的专用系统，这种通用框架更具扩展性。

### 方法详解
整体思路可以划分为四个阶段：**理解 → 符号化 → 推理 → 生成**。下面逐步拆解每个模块的工作方式。

1. **语言理解（LLM 作为控制器）**  
   - 输入是一段自然语言问题或对话。  
   - LLM 先生成一段“思考草稿”，包括可能的事实、规则以及需要查询的目标。  
   - 这一步类似 CoT：模型把自己的推理意图写出来，随后把这些意图转成结构化的指令。

2. **符号化转译器**  
   - 控制器输出的指令被送入一个轻量的语义解析器。  
   - 解析器把自然语言的事实和规则映射为逻辑编程语言（如 Prolog）的谓词形式。  
   - 例如，“所有鸟都会飞” → `fly(X) :- bird(X).`  
   - 关键在于保持语义完整性：解析器必须捕捉量词、条件等逻辑要素，否则后续引擎会产生错误。

3. **符号推理引擎**  
   - 生成的逻辑程序被加载进符号推理引擎。  
   - 引擎执行深度优先或宽度优先搜索，返回满足查询的最小证明或失败信息。  
   - 这里的“查询”是控制器在第一步里明确的目标，例如 `fly(penguin)?`。  
   - 引擎的输出是符号化的事实（如 `false`）或具体的证明路径。

4. **结果回写与自然语言生成**  
   - 符号引擎的答案被包装成一段结构化的“系统状态”，再喂回 LLM。  
   - LLM 根据这段状态继续生成自然语言答案，同时可以把证明过程以可读的形式呈现给用户。  
   - 如果用户继续追问，模型会把新的问题与已有的符号记忆结合，形成新的查询，循环往复。

**最巧妙的设计**在于把 LLM 的“情境理解”与符号引擎的“严格推理”放在同一个控制循环里。模型不再是一次性输出答案，而是像人类一样：先在脑中构思推理步骤（语言层），再把关键步骤写在纸上（符号层），最后检查纸上的结论是否符合逻辑。

### 实验与效果
- **测试任务**：论文在公开的多步推理基准上评估，包括逻辑谜题、数学证明和常识链式推理等。  
- **对比基线**：与纯 LLM（直接生成答案）以及传统 CoT 方法相比，ChatLogic 在所有任务上都取得了显著提升。论文声称在某些逻辑谜题上准确率提升了约 20% 以上。  
- **消融实验**：作者分别去掉符号记忆、去掉自动转译器或让 LLM 不再充当控制器，性能均出现明显下降，说明每个模块都是提升的关键因素。  
- **局限性**：转译器对自然语言的依赖仍然较强，复杂句式或歧义表达可能导致错误的逻辑程序；此外，符号引擎的搜索成本随规则规模指数增长，当前实现更适合中等规模的逻辑集合。

### 影响与延伸思考
ChatLogic 把“语言模型+符号推理”的组合推向了一个可操作的系统层面，随后的研究开始探索更高效的符号转译、与图数据库的结合以及在大规模知识图谱上的实时推理。推测未来会有工作尝试把 LLM 的自监督预训练目标直接加入符号一致性约束，让模型在训练阶段就学会生成可验证的逻辑程序。对想深入的读者，可以关注以下方向：① 可微分的逻辑编程框架（如 Neural Theorem Provers），② 大模型的长期记忆机制（如 Retrieval‑Augmented Generation），③ 人机协同推理的交互协议设计。

### 一句话记住它
ChatLogic 让大语言模型既会说话又会写逻辑程序，通过循环调用符号推理引擎，实现可靠的多步推理。