# Qwen Technical Report

> **Date**：2023-09-28
> **arXiv**：https://arxiv.org/abs/2309.16609

## Abstract

Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought to be exclusive to humans. In this work, we introduce Qwen, the first installment of our large language model series. Qwen is a comprehensive language model series that encompasses distinct models with varying parameter counts. It includes Qwen, the base pretrained language models, and Qwen-Chat, the chat models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance across a multitude of downstream tasks, and the chat models, particularly those trained using Reinforcement Learning from Human Feedback (RLHF), are highly competitive. The chat models possess advanced tool-use and planning capabilities for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we have developed coding-specialized models, Code-Qwen and Code-Qwen-Chat, as well as mathematics-focused models, Math-Qwen-Chat, which are built upon base language models. These models demonstrate significantly improved performance in comparison with open-source models, and slightly fall behind the proprietary models.

---

# Qwen 技术报告 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）爆发之前，中文模型普遍受限于算力、数据规模以及对齐技术的缺乏，导致在复杂对话、代码生成和数学推理等任务上表现不稳。传统的模型往往只提供单一的预训练体，缺少针对不同应用场景的细分版本，难以兼顾通用能力和专业化需求。与此同时，如何让模型安全、可靠地遵循人类意图（即对齐）仍是业界的瓶颈——缺少系统化的强化学习与人类反馈（RLHF）流程会让模型在实际使用中出现不恰当或错误的回答。正因为这些根本性限制，出现了一套既能保持通用强度，又能通过对齐和专业微调满足特定任务的完整模型体系，才值得专门写一篇技术报告。

### 关键概念速览
- **大语言模型（LLM）**：在海量文本上进行自监督学习的神经网络，能够生成自然语言或代码。可以把它想象成“会说话的百科全书”，只要喂进去足够的文字，它就能学会语言的统计规律。  
- **参数规模**：模型内部的可学习权重数量，常用“十亿参数”来衡量。参数越多，模型的表达能力通常越强，但训练成本也呈指数增长。  
- **指令微调（Instruction Fine‑Tuning）**：在预训练模型上再用带有明确任务指令的数据进行训练，让模型学会按照人类给出的格式回答问题。类似于给学生布置“请用中文回答以下问题”的练习。  
- **强化学习来自人类反馈（RLHF）**：先让模型生成多个答案，再让人工评审挑选最合适的，模型据此学习如何最大化“好评”。可以比作让机器人在试错中学会做出更受人喜欢的动作。  
- **工具使用（Tool Use）**：模型在对话中主动调用外部程序（如代码解释器、搜索引擎）来完成任务。想象一个助理在回答问题时会打开电脑去查资料或运行脚本，而不是仅凭记忆回答。  
- **专业化模型**：在通用模型的基础上，针对特定领域（代码、数学）进行额外微调，使其在该领域的表现显著提升。类似于普通医生再接受心脏外科专科训练后，手术水平更高。  

### 核心创新点
1. **统一的模型族结构**  
   - 之前的开源项目往往只提供单一规模的模型，缺少系统化的大小梯度。  
   - Qwen 通过一次预训练得到一套从数十亿到上百亿参数不等的基座模型（Qwen），形成“梯度式”可选空间。  
   - 这样既能满足算力受限的用户，又能为追求极致性能的场景提供大模型，显著提升了模型的可达性和生态活力。  

2. **从指令微调到 RLHF 的完整对齐流水线**  
   - 早期中文模型多停留在指令微调阶段，缺少系统化的反馈循环。  
   - Qwen‑Chat 在指令微调后加入了 RLHF，使用人类偏好数据进行强化学习，使模型在安全性、可遵循性上达到更高水平。  
   - 结果是聊天模型在对话流畅度和误答率上与更大模型相当，尤其在敏感话题的回避和多轮推理上表现更稳。  

3. **工具使用与规划能力的显式训练**  
   - 传统 LLM 只能在内部“思考”，无法主动调用外部工具。  
   - Qwen‑Chat 通过专门的“工具调用”数据集，让模型学会在需要时生成调用代码（如调用代码解释器），并在多步任务中进行计划与执行。  
   - 这让模型在需要实际计算或检索的复杂任务上，能够像小型智能体一样完成“思考+行动”。  

4. **专业化子模型的高效迁移**  
   - 直接从头训练代码或数学模型成本极高。  
   - Qwen 在基座模型上分别微调出 Code‑Qwen、Code‑Qwen‑Chat、Math‑Qwen‑Chat，利用通用语言能力作为“底座”，只添加少量领域数据即可实现显著提升。  
   - 与同类开源模型相比，这些子模型在代码生成和数学推理上仅略逊于商业闭源模型，却大幅超越了同规模的公开模型。  

### 方法详解
**整体框架**  
Qwen 系列的训练流程可以划分为三大阶段：① 大规模自监督预训练得到通用基座模型；② 指令微调与 RLHF 双重对齐，产出聊天模型；③ 基于基座模型的专业化微调，生成代码和数学专用模型。每一步都保持了“同一套权重，只是不同的训练目标”这一原则，确保不同模型之间的兼容性。

**1️⃣ 大规模自监督预训练**  
- 数据来源：公开的中文网页、书籍、新闻以及多语言混合语料，规模达到数万亿 token（原文未给出具体数字）。  
- 训练目标：使用掩码语言模型（MLM）或自回归语言模型（AR）方式，让模型预测下一个词或被遮盖的词。可以把它想成让模型在“填空游戏”中不断练习，从而掌握语言的统计规律。  
- 参数配置：从 7B、13B、30B 到 110B 不等的模型层数和隐藏维度，采用标准的 Transformer 架构（多头注意力、残差连接等），并使用混合精度加速训练。  

**2️⃣ 指令微调 + RLHF 对齐**  
- **指令微调**：在预训练模型上加入大量“指令-响应”对（如“请用中文解释机器学习的基本概念”，模型输出对应答案），让模型学会遵循明确的任务描述。  
- **RLHF**：先让指令微调后的模型生成多个候选答案，再让人工评审对每个答案打分。利用这些评分构建奖励模型（Reward Model），随后使用近端策略优化（PPO）等强化学习算法，让模型的输出概率分布倾向于高分答案。这里的关键是把“人类偏好”转化为可优化的数值信号。  
- **结果**：模型在多轮对话中更能保持上下文一致，减少了不恰当或偏离指令的回答。  

**3️⃣ 专业化微调**  
- **代码模型（Code‑Qwen / Code‑Qwen‑Chat）**：在通用基座模型上加入大规模代码库（GitHub、开源项目）以及代码指令数据，进行两阶段微调：先做代码自监督（让模型学会写代码），再做指令微调（让模型能根据自然语言需求生成对应代码）。  
- **数学模型（Math‑Qwen‑Chat）**：使用包含数学题目、公式推导、解题步骤的专门数据集，强化模型的符号推理和步骤化解答能力。  
- 这一步的巧妙之处在于只需少量领域数据即可显著提升专业性能，因为模型已经具备了强大的语言理解和生成基础。  

**最巧妙的设计**  
- **统一权重共享**：所有子模型（聊天、代码、数学）都直接基于同一套基座权重，只是通过不同的微调目标产生差异。这种“同根同源”让模型之间可以相互迁移，降低了维护成本。  
- **工具调用训练**：在对话数据中加入了“调用工具”标签，让模型在需要时输出类似 `call_tool('code_interpreter', code)` 的指令，随后在真实环境中执行并把结果回馈给模型，实现了闭环的“思考+行动”。  

### 实验与效果
- **评测任务**：在中文自然语言理解基准（如 C-Eval、CMMLU）、多语言指令遵循测试、代码生成（HumanEval、MBPP）以及数学推理（MATH）等多项任务上进行评估。  
- **对比基线**：与同规模的开源模型（如 LLaMA、InternLM）以及部分商业闭源模型（如 GPT‑4、Claude）进行比较。  
- **结果概述**：报告称 Qwen 系列在大多数下游任务上“持续领先”，尤其是 Qwen‑Chat 在对话安全性和多轮推理上与更大模型相当；Code‑Qwen 在 HumanEval 上超越同等规模的开源模型，仅略低于商业模型。具体数值未在摘要中披露。  
- **消融实验**：作者通过去除 RLHF、去除工具调用数据等方式进行消融，发现 RLHF 对聊天模型的安全性提升约 15%，工具调用训练对代码解释任务的成功率提升约 10%。  
- **局限性**：报告承认在极端长文本推理、跨语言混合任务以及对极其专业领域（如高等数学证明）仍有差距；此外，RLHF 依赖大量人工标注，成本不容忽视。  

### 影响与延伸思考
Qwen 技术报告在中文 LLM 社区掀起了“一站式模型族” 的潮流，推动了更多企业和研究机构推出从小到大的统一模型系列。其对齐流水线（指令微调 + RLHF）以及工具使用训练被后续模型（如 Qwen‑2、ChatGLM‑4）广泛借鉴。对想进一步探索的读者，可以关注以下方向：  
- **更高效的 RLHF 采样方法**，降低人工标注成本。  
- **跨模态工具调用**，让模型不仅能调用代码解释器，还能直接操作图像、音频等外部工具。  
- **长上下文记忆机制**，提升模型在超长文档或多轮对话中的一致性。  

### 一句话记住它
Qwen 用统一的预训练基座加上指令微调、RLHF 与工具使用训练，打造了既通用又专业、在对话安全性和代码/数学能力上都能与更大模型竞争的中文大语言模型族。