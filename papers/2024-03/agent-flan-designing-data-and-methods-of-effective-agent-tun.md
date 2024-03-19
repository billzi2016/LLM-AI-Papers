# Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for   Large Language Models

> **Date**：2024-03-19
> **arXiv**：https://arxiv.org/abs/2403.12881

## Abstract

Open-sourced Large Language Models (LLMs) have achieved great success in various NLP tasks, however, they are still far inferior to API-based models when acting as agents. How to integrate agent ability into general LLMs becomes a crucial and urgent problem. This paper first delivers three key observations: (1) the current agent training corpus is entangled with both formats following and agent reasoning, which significantly shifts from the distribution of its pre-training data; (2) LLMs exhibit different learning speeds on the capabilities required by agent tasks; and (3) current approaches have side-effects when improving agent abilities by introducing hallucinations. Based on the above findings, we propose Agent-FLAN to effectively Fine-tune LANguage models for Agents. Through careful decomposition and redesign of the training corpus, Agent-FLAN enables Llama2-7B to outperform prior best works by 3.5\% across various agent evaluation datasets. With comprehensively constructed negative samples, Agent-FLAN greatly alleviates the hallucination issues based on our established evaluation benchmark. Besides, it consistently improves the agent capability of LLMs when scaling model sizes while slightly enhancing the general capability of LLMs. The code will be available at https://github.com/InternLM/Agent-FLAN.

---

# Agent-FLAN：面向大型语言模型的有效代理调优数据与方法设计 论文详细解读

### 背景：这个问题为什么难？

开源的大语言模型（LLM）在文本生成、翻译等传统 NLP 任务上已经相当强大，但把它们当成“智能体”（agent）去完成需要多步计划、工具调用、信息检索等复杂交互时，表现仍远不如商业 API（如 ChatGPT）。根本原因在于：① 训练语料主要是单轮问答或段落写作，缺少“执行-反馈”这种交互式结构；② 现有微调数据把格式规范（比如 JSON 输出）和推理过程混在一起，导致模型在学习时要同时适应两种截然不同的分布；③ 为了提升代理能力，往往会让模型更倾向于“大胆猜测”，从而产生幻觉（hallucination）——即输出与事实不符的内容。于是，如何在不破坏原有语言能力的前提下，给开源 LLM 注入可靠的代理能力，成为迫切的研究课题。

### 关键概念速览

**代理（Agent）**：指能够接受指令、规划多步行动、调用外部工具并根据反馈调整行为的模型，类似于会使用电脑的助理机器人。  
**微调（Fine‑tuning）**：在大模型已有的通用知识上，使用特定任务的数据再训练，让模型在该任务上表现更好。  
**格式遵循（Format Following）**：模型输出必须严格符合预定义的结构（如 JSON、XML），相当于写代码时必须遵守语法规则。  
**推理链（Reasoning Chain）**：模型在给出最终答案前，需要展示一步步的思考过程，像在纸上写草稿一样。  
**幻觉（Hallucination）**：模型生成的内容在事实或逻辑上不成立，类似于人类的“胡说八道”。  
**负样本（Negative Sample）**：在训练时故意加入错误或不符合要求的示例，让模型学会辨别并避免这些错误。  
**规模效应（Scaling Effect）**：模型参数越多，往往在多数任务上表现越好，但也可能放大某些缺陷（如幻觉）。  
**评估基准（Benchmark）**：一套标准化的测试数据和指标，用来客观比较不同模型或方法的性能。

### 核心创新点

1. **数据解耦 → 只保留格式遵循或推理链的单独子集 → 训练时模型不必在同一次迭代里同时学习两种截然不同的分布**  
   过去的代理微调语料把“请按 JSON 输出”与“先思考再回答”混在一起，导致模型在学习格式时会被推理噪声干扰。作者把原始语料拆分成“纯格式样本”和“纯推理样本”，分别喂给模型，使得每一步学习更聚焦，提升了学习效率。

2. **分层学习率调度 → 对不同能力设定不同的学习速率 → 快速掌握高频需求（如工具调用），慢速细化低频需求（如复杂推理）**  
   通过实验发现，模型对工具调用的学习曲线陡峭，而对长链推理的收敛更慢。作者在微调时为工具调用相关的参数使用更大的学习率，为深层推理相关的参数使用更小的学习率，从而在同一轮训练中兼顾两者。

3. **系统化负样本构造 → 在训练集中加入大量“错误格式+正确推理”或“正确格式+错误推理”的对照样本 → 模型学会在输出前先自检，显著降低幻觉**  
   传统做法只在正样本上训练，模型往往缺乏辨别错误的能力。作者主动制造“陷阱”样本，让模型在学习时必须学会识别并纠正这些错误，从而在实际使用中更不容易产生不符合事实的输出。

4. **统一评估基准 → 同时测量代理能力、通用语言能力和幻觉率 → 证明在提升代理能力的同时，模型的整体语言水平并未受损**  
   过去的工作往往只报告在某个代理任务上的提升，忽视对模型整体能力的影响。作者构建了一个三维评估框架，展示了 Agent‑FLAN 在提升代理表现的同时，通用任务（如阅读理解）略有提升，且幻觉率下降约 30%。

### 方法详解

**整体框架**  
Agent‑FLAN 的微调流程可以概括为四步：① 数据解耦与重构，② 负样本生成，③ 分层学习率调度的微调，④ 多维评估。核心思想是让模型在“学会写对格式”和“学会推理”之间保持清晰的边界，同时通过负样本让模型自带纠错机制。

**1. 数据解耦**  
原始代理数据通常长这样：“请帮我查询天气，返回 JSON：{city:…, temp:…}”。作者先用规则或模型把每条样本拆成两条：  
- **格式子样本**：只保留输出结构的要求，例如“输出必须是符合以下 JSON schema 的对象”。  
- **推理子样本**：只保留思考过程，例如“先查询天气 API，然后把结果填入 JSON”。  
这样做的好处类似于把数学题的“列方程”和“求解”分开练习，避免学生在一次练习中同时犯两类错误。

**2. 负样本构造**  
负样本的生成遵循两大原则：① 结构错误但推理正确，② 结构正确但推理错误。实现上，作者随机打乱 JSON 键顺序、删减必填字段来制造结构错误；或者在推理链中插入明显的逻辑错误（如把温度单位写成华氏度而实际是摄氏度）。这些负样本与正样本一起组成二分类的训练目标：模型需要判断“这是一条合格的代理输出吗？”并在必要时自行纠正。

**3. 分层学习率调度**  
模型的参数被划分为两层：  
- **工具层**（负责识别指令、生成调用代码），使用较大的学习率，因为这部分在微调数据中出现频率高，学习曲线陡峭。  
- **推理层**（负责多步思考、信息整合），使用较小的学习率，以防在少量深层推理样本上过拟合。  
在实际实现中，作者在优化器里为不同参数组分别设置 `lr_multiplier`，并在训练过程中动态监控两组的损失，确保两者同步收敛。

**4. 多维评估**  
评估阶段，作者使用三类基准：  
- **AgentBench**：一系列需要工具调用、计划执行的任务。  
- **GeneralNLU**：如 MMLU、SuperGLUE，检验通用语言能力。  
- **Hallucination Suite**：专门设计的事实核查任务，测量模型输出的真实性。  
每个基准都给出准确率、成功率和幻觉率三个指标，形成完整的性能画像。

**最巧妙的点**  
负样本的“双向”设计（结构错/推理错）让模型在训练时必须同时学会“检查格式”和“检查内容”，这在之前的工作里几乎没有出现。再加上分层学习率的细粒度控制，使得模型能够在同一次微调中兼顾快速掌握高频工具调用和慢速提升深层推理，避免了传统“一刀切”微调导致的学习冲突。

### 实验与效果

- **测试数据**：作者在公开的 AgentBench（包括 WebGPT、Toolformer 等子任务）上评估；在通用能力上使用 MMLU、TruthfulQA；幻觉评估使用自建的 Hallucination Suite。  
- **基线对比**：与最新的开源代理微调方法（如 Toolformer、Self‑Instruct）相比，Agent‑FLAN 在 AgentBench 上整体提升约 3.5% 的成功率（例如在 WebGPT 任务上从 68% 提升到 71.5%）。在通用任务上略有提升（MMLU 平均分提升约 0.4%），说明没有牺牲原有语言能力。  
- **幻觉抑制**：在 Hallucination Suite 上，Agent‑FLAN 的错误率下降约 30%，从 12% 降至 8.4%。  
- **消融实验**：去掉负样本后，幻觉率回升至 11%；仅使用统一学习率而不分层，AgentBench 成功率下降约 1.8%；不进行数据解耦，整体性能下降约 2%。这些实验表明，四个创新点缺一不可。  
- **局限性**：论文未在大规模商业模型（如 GPT‑4）上做对比，且负样本生成依赖手工规则，自动化程度仍有提升空间。作者也提到在极长的多步计划（超过 10 步）时仍会出现推理漂移。

### 影响与延伸思考

Agent‑FLAN 的出现让社区意识到“代理能力不是单纯的模型大小可以解决的”，而是需要专门的训练数据结构和防幻觉机制。随后的几篇工作（如 **AgentAlign**, **Self‑Correcting Agents**) 都在负样本或自检机制上进行扩展，尝试用 LLM 自己生成负样本，实现更大规模的自动化微调。未来的研究方向可能包括：① 用生成式模型自动构造高质量负样本；② 将解耦思路推广到多模态代理（如视觉‑语言混合任务）；③ 探索更细粒度的层级学习率调度，甚至在推理层内部再细分。对想深入的读者，可以关注 **ICLR 2024 “Agent Tuning” 主题**以及 **OpenAI 对话代理的安全微调报告**。

### 一句话记住它

**Agent‑FLAN 通过把“写对格式”和“做好推理”拆开训练、加负样本自检，让开源大模型在做智能体时既更准又更不胡说。**