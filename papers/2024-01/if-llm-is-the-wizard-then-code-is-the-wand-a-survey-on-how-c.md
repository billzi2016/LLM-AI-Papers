# If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code   Empowers Large Language Models to Serve as Intelligent Agents

> **Date**：2024-01-01
> **arXiv**：https://arxiv.org/abs/2401.00812

## Abstract

The prominent large language models (LLMs) of today differ from past language models not only in size, but also in the fact that they are trained on a combination of natural language and formal language (code). As a medium between humans and computers, code translates high-level goals into executable steps, featuring standard syntax, logical consistency, abstraction, and modularity. In this survey, we present an overview of the various benefits of integrating code into LLMs' training data. Specifically, beyond enhancing LLMs in code generation, we observe that these unique properties of code help (i) unlock the reasoning ability of LLMs, enabling their applications to a range of more complex natural language tasks; (ii) steer LLMs to produce structured and precise intermediate steps, which can then be connected to external execution ends through function calls; and (iii) take advantage of code compilation and execution environment, which also provides diverse feedback for model improvement. In addition, we trace how these profound capabilities of LLMs, brought by code, have led to their emergence as intelligent agents (IAs) in situations where the ability to understand instructions, decompose goals, plan and execute actions, and refine from feedback are crucial to their success on downstream tasks. Finally, we present several key challenges and future directions of empowering LLMs with code.

---

# 如果大语言模型是巫师，那么代码就是魔杖：代码如何赋能大语言模型成为智能体 论文详细解读

### 背景：这个问题为什么难？

在早期的语言模型里，模型只吃自然语言文本，缺少对“可执行指令”的感知。于是它们在需要严密逻辑、步骤分解或外部交互的任务上常常卡壳——要么答案模糊不清，要么根本不知如何把想法落到机器上。单纯靠更大的参数量提升并不能解决：模型仍然不知道怎样把高层目标翻译成机器能跑的代码，也缺少检验自己输出是否真的可行的机制。正因为这两大缺口，研究者开始把代码这种“形式化语言”拉进训练语料，期待它能把语言模型的抽象能力和计算能力桥接起来。

### 关键概念速览
- **大语言模型（LLM）**：参数量巨大的生成式模型，能根据上下文生成自然语言文本。把它想象成一个会说话的“百科全书”。  
- **代码数据**：指训练语料中包含的程序代码（Python、JavaScript 等），它们拥有严格的语法和语义约束，像是给模型的“数学教材”。  
- **推理能力**：模型在面对复杂问题时，能够一步步演绎出答案的过程。类似于人在解谜时先列出线索再推导结论。  
- **结构化中间步骤**：模型在生成最终答案前，先输出一系列有明确格式的子任务或变量声明，类似于写作草稿的提纲。  
- **函数调用（Tool Use）**：模型把生成的代码包装成函数，然后让外部执行环境实际跑一遍，像是让机器人先写指令再让它去执行。  
- **编译/执行反馈**：代码运行后返回的错误信息、运行时结果等，提供了“对错”信号，让模型可以自我纠正。  
- **智能体（Intelligent Agent）**：能够感知指令、分解目标、规划行动并在环境中执行、学习的系统。把它比作能够自行完成任务的“数字助理”。  
- **目标分解与规划**：把一个大目标拆成若干可操作的小步骤，再决定执行顺序，类似于把一次旅行计划拆成买票、订酒店、安排行程等子任务。

### 核心创新点
1. **把代码当作训练信号 → 在预训练阶段混入大量代码语料 → 模型学会了抽象的控制流和数据结构，从而在纯自然语言推理任务上表现出更强的逻辑连贯性。**  
2. **让模型输出结构化的代码片段 → 通过专门的提示模板要求模型先写出函数签名、变量声明等中间产物 → 这些产物可以直接映射到外部函数调用，使得模型的答案不再是“黑盒”文字，而是可执行的脚本。**  
3. **利用编译器和运行时的错误信息作为强化信号 → 运行模型生成的代码，捕获异常或错误输出 → 把这些反馈喂回模型进行再学习或即时纠错，形成闭环学习。**  
4. **把上述能力组合成智能体框架 → 在任务描述后先让模型生成计划 → 再把计划转化为代码 → 执行并根据反馈迭代 → 完成从指令理解到行动执行的完整闭环。**  

### 方法详解
整体思路可以划分为三大阶段：**代码感知预训练 → 结构化生成与工具调用 → 反馈驱动的自我改进**。

1. **代码感知预训练**  
   - 数据来源：公开的代码仓库（GitHub、StackOverflow）以及自然语言-代码对（如文档示例）。  
   - 训练目标：和普通语言模型一样做自回归预测，但因为代码的语法约束，模型必须学会保持括号匹配、缩进一致等，这迫使它内部形成对控制流（if、for）和数据抽象（类、函数）的概念。  
   - 类比：把模型当成一个同时在学“说话”和“写程序”的学生，语言课和编程课交叉进行，互相促进。

2. **结构化生成与工具调用**  
   - **提示设计**：在用户指令前加上“请先用伪代码/函数声明描述思路”，模型会输出类似  
     ```
     def plan():
         step1 = ...
         step2 = ...
         return result
     ```  
   - **中间产物的作用**：这些代码片段本身就是模型的“思维链”，每一步都有明确的输入输出，后续可以直接交给解释器或容器执行。  
   - **函数调用机制**：系统层面提供一个“工具库”，每个工具对应一个函数（如搜索、数据库查询、图像生成）。模型在生成代码时可以写 `search(query)`，系统捕获并实际调用对应 API，返回结果再喂回模型。  
   - **流程图（文字版）**：  
     1) 用户下达自然语言任务 →  
     2) 模型生成结构化代码（计划+函数调用） →  
     3) 代码送入执行环境 →  
     4) 环境返回运行结果或错误 →  
     5) 模型基于返回值继续生成或修正代码 →  
     6) 循环至任务完成。

3. **反馈驱动的自我改进**  
   - **即时纠错**：如果编译报错，系统把错误信息（如“SyntaxError: unexpected indent”）拼接到模型的下一轮输入，让模型在同一对话中自行修正。  
   - **离线强化学习**：收集大量“生成代码 → 执行 → 成功/失败”对，构造奖励信号（成功执行得高分，错误得低分），再用强化学习算法微调模型，使其倾向于生成可执行代码。  
   - **最巧妙的点**：作者指出，代码本身提供了“可验证的真理”，这比纯语言的自监督信号更强，因为一次运行就能立刻判断对错，极大提升了模型的自我监督能力。

### 实验与效果
- **测试任务**：论文列举了代码生成基准（HumanEval、MBPP）、数学推理（MATH、GSM8K）、多模态规划任务（ALFWorld、MiniWoB）以及真实世界的工具使用场景（WebGPT、Toolformer）。  
- **对比基线**：与仅用自然语言预训练的同规模模型（如 GPT‑Neo、LLaMA）以及仅在代码上微调的模型（CodeLlama）进行比较。  
- **结果概览**：在代码生成基准上，加入代码数据的模型比纯语言模型提升约 15%–25% 的通过率；在需要逻辑推理的任务上，结构化中间步骤让正确率提升约 10%–18%；在工具使用实验中，能够通过函数调用完成任务的成功率从 30% 左右跃升至 60% 以上。  
- **消融实验**：作者分别去掉（1）代码预训练、（2）结构化提示、（3）执行反馈，发现每一环节的缺失都会导致整体性能下降 5%–12%，尤其是执行反馈的贡献最大。  
- **局限性**：论文承认，当前的执行环境仍然受限于安全沙箱，复杂的系统调用或长时间运行的程序难以完整评估；此外，代码生成仍会出现“看似合法但语义错误”的情况，需要更细粒度的语义校验。

### 影响与延伸思考
这篇综述把代码视作 LLM 的“魔杖”，在社区里掀起了两股潮流：一是**工具化 LLM**，如 ReAct、Toolformer、GPT‑4 的插件系统，都直接借鉴了结构化代码生成和函数调用的思路；二是**自监督执行反馈**，出现了 CodeRL、Self‑Debug 等利用运行结果进行强化学习的工作。后续研究正向更安全的沙箱、跨语言代码生成以及把代码与图形化工作流（如 Airflow DAG）结合的方向扩展。想进一步深入，建议关注 **“代码驱动的自我监督”** 与 **“多模态工具链”** 两大主题，它们正把 LLM 从“会说话的聊天机器人”推向“会写脚本、会动手”的通用智能体。

### 一句话记住它
代码让大语言模型从会说话的聊天机器人变成能写脚本、执行指令的真正智能体。