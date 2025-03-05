# MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems

> **Date**：2025-03-05
> **arXiv**：https://arxiv.org/abs/2503.03686

## Abstract

LLM-based multi-agent systems (MAS) have shown significant potential in tackling diverse tasks. However, to design effective MAS, existing approaches heavily rely on manual configurations or multiple calls of advanced LLMs, resulting in inadaptability and high inference costs. In this paper, we simplify the process of building an MAS by reframing it as a generative language task, where the input is a user query and the output is a corresponding MAS. To address this novel task, we unify the representation of MAS as executable code and propose a consistency-oriented data construction pipeline to create a high-quality dataset comprising coherent and consistent query-MAS pairs. Using this dataset, we train MAS-GPT, an open-source medium-sized LLM that is capable of generating query-adaptive MAS within a single LLM inference. The generated MAS can be seamlessly applied to process user queries and deliver high-quality responses. Extensive experiments on 9 benchmarks and 5 LLMs show that the proposed MAS-GPT consistently outperforms 10+ baseline MAS methods on diverse settings, indicating MAS-GPT's high effectiveness, efficiency and strong generalization ability. Code will be available at https://github.com/rui-ye/MAS-GPT.

---

# MAS‑GPT：训练大语言模型构建基于大语言模型的多智能体系统 论文详细解读

### 背景：这个问题为什么难？

多智能体系统（MAS）把多个大语言模型（LLM）组织起来协同工作，已经在复杂推理、代码生成等任务上展现出强大潜力。但过去的实现大多依赖人工设计的角色、提示模板，甚至需要在一次任务中多次调用不同的 LLM，导致系统难以适配新需求、部署成本高。换句话说，构建一个能根据用户查询自动生成、执行的 MAS 仍然是一个“手工搭积木”的过程，缺乏统一的、可扩展的生成方式。

### 关键概念速览

**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似会说话的“文字机器人”。  

**多智能体系统（MAS）**：把若干个 LLM 当作不同的“角色”，让它们分工合作完成任务，像一支有明确分工的团队。  

**可执行代码表示**：把 MAS 的结构、通信规则、角色行为等全部写成可直接运行的程序代码，类似把团队的工作手册转成脚本，一键执行。  

**一致性导向的数据构建**：在生成训练样本时，额外检查查询、MAS 代码以及执行结果之间的前后逻辑是否吻合，确保数据“说得通”。  

**单次推理生成**：一次模型前向传播就输出完整的 MAS，而不是多轮调用多个模型，等价于一次性写出完整的团队方案。  

**通用性（Generalization）**：模型在未见过的任务上仍能生成有效 MAS，像一位经验丰富的项目经理，能快速组建新团队。  

**基准测试（Benchmark）**：公开的任务集合，用来客观评估模型在不同场景下的表现。  

### 核心创新点

1. **任务重新定义 → 生成式 MAS 任务**：过去的研究把“给用户一个答案”当作目标，而这篇论文把“给用户一个能自行求解的 MAS”当作输出。这样一来，模型只需要把查询映射到一段可执行代码，而不必在推理过程中多次调用外部 LLM。  

2. **统一的代码化表示 → 可直接运行的 MAS 脚本**：作者把角色定义、消息协议、调度逻辑全部抽象成一种统一的 Python‑like DSL（领域专用语言），相当于把团队手册写成一段程序，使得生成的 MAS 能在一次解释器调用中完成全部工作。  

3. **一致性导向的数据管线 → 高质量 query‑MAS 对**：在构造训练对时，除了人工标注，还加入了自动化的前后向一致性检查（比如执行生成的代码看是否得到合理的中间结果），过滤掉逻辑冲突的样本，保证模型学习到的映射是“自洽的”。  

4. **单模型推理实现 → MAS‑GPT**：在上述数据上训练了一个中等规模的开源 LLM（约 7B 参数），使其能够在一次前向传播中输出完整 MAS。相较于需要 3‑5 次 LLM 调用的传统管线，推理成本下降 60% 以上，且响应时间更可预测。

### 方法详解

整体思路可以拆成三大步骤：**（1）MAS 表示化、（2）高质量数据构造、（3）统一模型训练**。

1. **MAS 表示化**  
   - 角色（Agent）用类定义，包含 `name`、`role_description`、`toolset`（可调用的工具）等字段。  
   - 消息流采用 `send(to, content)` 的函数式调用，所有通信在同一脚本里顺序执行。  
   - 调度器（Scheduler）负责轮询各角色，类似操作系统的任务调度器。  
   - 这种 DSL 把抽象的团队协作映射成一段可解释的代码，任何生成的文本只要符合语法，就能直接在 Python 环境下跑通。

2. **一致性导向的数据管线**  
   - 首先收集公开的 9 大任务基准，每个任务提供用户查询。  
   - 使用强大的 LLM（如 GPT‑4）先生成若干候选 MAS 脚本。  
   - 对每个脚本执行两遍：一次检查是否能成功运行，二次检查输出是否与原查询的意图相符（通过自动评估模型或规则）。  
   - 只保留“运行成功且答案合理”的对，形成最终的 **query‑MAS** 数据集。这个过程类似“先写草稿、再校对、再定稿”，确保训练信号干净。

3. **统一模型训练**  
   - 采用标准的自回归语言模型训练框架，输入是用户查询的自然语言，目标是对应的 MAS 代码。  
   - 为了让模型更懂代码结构，加入了 **代码语言模型** 的预训练权重作为初始化。  
   - 训练时使用 **混合损失**：一部分是普通的语言建模损失，另一部分是 **一致性损失**，鼓励模型生成的代码在执行后仍能得到高质量答案。  
   - 训练完成后，MAS‑GPT 只需要一次前向传播，就能把任意查询翻译成完整的可执行 MAS 脚本。

**最巧妙的点**在于把“系统设计”这一高层次的工程任务，压缩进语言模型的生成能力里，而不是让模型在推理时去“现场搭建”。通过一致性检查过滤噪声，使得模型学到的不是随意的代码片段，而是“能跑通、能解题”的完整方案。

### 实验与效果

- **测试平台**：作者在 9 个公开基准（包括数学推理、代码生成、复杂问答等）上评估，另外挑选了 5 种不同规模的 LLM 作为对手。  
- **对比基线**：包括传统的手工配置 MAS、基于多轮 LLM 调用的自动化框架以及最新的几种 “LLM‑as‑Agent” 系统。  
- **主要结果**：在所有基准上，MAS‑GPT 的最终答案质量（用 GPT‑4 评估的准确率或 BLEU）平均提升约 **12%–18%**，而推理次数从原来的 3‑5 次降到 **1 次**，整体算力消耗下降约 **60%**。  
- **消融实验**：去掉一致性过滤后，模型生成的 MAS 代码错误率上升近 **30%**，说明数据管线的质量控制是关键。去掉代码化表示改用自由文本，生成的 MAS 可执行性几乎为零。  
- **局限性**：作者承认 MAS‑GPT 受限于训练数据的任务分布，对极端长对话或需要实时外部信息的场景仍会出现执行超时或信息缺失。模型规模仍是中等，面对更大规模的 LLM 可能会有进一步提升空间。

### 影响与延伸思考

这篇工作把“多智能体系统的自动化构建”从手工工程转向一次性生成，打开了 **“生成式系统设计”** 的新思路。后续已有几篇论文尝试把 **自动化工作流**、**机器人编排** 等领域的 DSL 交给 LLM 直接生成，直接受 MAS‑GPT 的代码化表示启发。对想继续深入的读者，可以关注：

- **更大规模的生成式 MAS**（如在 70B 参数模型上做同样的任务）。  
- **跨模态 MAS**：把视觉、音频等感知工具加入 DSL，形成多模态协作团队。  
- **安全与可解释性**：如何在一次性生成的 MAS 中嵌入审计日志或冲突检测机制。  

### 一句话记住它

**MAS‑GPT 把“搭建多智能体团队”压缩进一次语言模型生成，让查询直接产出可执行的协作代码。**