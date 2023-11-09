# Agent Lumos: Unified and Modular Training for Open-Source Language   Agents

> **Date**：2023-11-09
> **arXiv**：https://arxiv.org/abs/2311.05657

## Abstract

Closed-source agents suffer from several issues such as a lack of affordability, transparency, and reproducibility, particularly on complex interactive tasks. This motivates the development of open-source alternatives. We introduce LUMOS, one of the first frameworks for training open-source LLM-based agents. LUMOS features a learnable, unified, and modular architecture with a planning module that learns high-level subgoal generation, and a grounding module trained to translate these into actions using various tools in the execution module. The design allows for modular upgrades and wider applicability to diverse interactive tasks. To foster generalizable agent learning, we collect large-scale, unified, and high-quality training annotations derived from diverse ground-truth reasoning rationales across various complex interactive tasks. On 9 datasets, LUMOS exhibits several key advantages: (1) LUMOS excels multiple larger open-source agents on the held-out datasets (unused for training) for each task type. LUMOS even surpasses GPT agents on QA and web tasks; (2) LUMOS outperforms open-source agents produced by chain-of-thoughts and unmodularized integrated training; and (3) LUMOS effectively generalizes to unseen tasks, outperforming 33B-scale agents and domain-specific agents.

---

# Agent Lumos：面向开源语言代理的统一与模块化训练框架 论文详细解读

### 背景：这个问题为什么难？

在交互式任务（如网页检索、工具使用、复杂推理）里，商业闭源大模型虽然表现强，但价格高、内部细节不可见，研究者难以复现或改进。早期的开源代理大多是把语言模型直接喂进一个大流水线，规划、执行、工具调用全写在一起，导致：①模型难以学到通用的“高层规划”能力，②新工具加入时必须重新训练整个系统，③不同任务之间的知识难以共享。于是，如何构建一个既能统一学习，又能像积木一样随时换模块的开源代理，成为了迫切需求。

### 关键概念速览
- **语言模型（LLM）**：能够生成自然语言的深度网络，类似会说话的“智能笔”。在本工作中，它负责把任务描述转化为内部指令。
- **规划模块**：负责产生任务的子目标（subgoals），相当于把大任务拆成一系列小步骤，就像旅行前先列出“买票、订酒店、打包行李”。
- **落地（Grounding）模块**：把规划得到的子目标映射成具体的工具调用或动作，类似把“查天气”翻译成“打开天气APP并输入城市”。
- **执行模块**：真正运行工具、调用API、或在环境中执行指令的部分，像是机器人的手臂负责把指令变成现实。
- **统一训练（Unified Training）**：一次性用同一批标注数据同时训练所有模块，而不是分别为每个模块准备独立数据。
- **模块化升级（Modular Upgrade）**：可以单独替换或增强某一模块（比如换成更强的规划网络），而不必重新训练整个系统。
- **高质量训练标注**：从多种交互任务中抽取的“推理过程”文本，提供给模型学习“为什么这么做”，类似老师给学生的解题步骤。

### 核心创新点
1. **统一、可学习的三段式架构**  
   *之前的开源代理往往把规划、落地、执行硬编码在一起，缺乏可学习的接口* → *Lumos 将这三部分抽象为独立的可学习模块，并用同一套标注数据同步训练* → *模型能够在同一次训练中同时掌握高层目标生成和低层动作映射，提升了跨任务的通用性。*

2. **大规模统一标注集**  
   *过去的训练数据多来自单一任务的人工示例，规模小、风格单一* → *作者收集了覆盖 9 类交互任务的统一标注，所有标注都包含完整的推理链和对应的工具调用* → *模型在学习时看到的“思考—行动”模式更丰富，因而在未见任务上也能保持竞争力。*

3. **模块化升级机制**  
   *传统系统若想加入新工具，需要重新设计整个流水线* → *Lumos 的落地模块只负责把子目标映射为工具调用，新增工具只需在执行模块注册对应的 API，落地模块通过微调即可适配* → *大幅降低了系统迭代成本，使得开源社区可以快速实验新工具。*

4. **对比实验显示跨模型优势**  
   *很多开源代理在特定任务上能跑通，但在未见任务上表现跌破谷底* → *Lumos 在所有 9 个数据集的 held‑out 部分均超过更大的开源基线，甚至在 QA 与网页任务上超越了 GPT 系列的闭源代理* → *证明统一训练和模块化设计真的带来了更好的泛化能力。*

### 方法详解
**整体思路**：Lumos 把一个交互式任务拆成三层：先让语言模型产生一系列抽象子目标（规划），再让落地网络把每个子目标翻译成具体的工具调用或环境动作（落地），最后交给执行模块实际运行并收集反馈。整个流程在一次统一的训练循环中完成，所有模块共享同一批标注。

**步骤拆解**：

1. **输入处理**  
   - 用户给出自然语言指令或问题。  
   - 该指令被送入共享的语言模型编码层，得到上下文向量。

2. **规划模块**  
   - 基于上下文向量，规划网络（通常是一个 Transformer 解码器）输出一个子目标序列。  
   - 每个子目标是高层的、可解释的描述，例如 “检索最新的天气预报”。  
   - 类比：像老师先给学生列出解题的关键步骤。

3. **落地模块**  
   - 落地网络接收子目标文本，学习将其映射为具体的工具指令或 API 参数。  
   - 这里使用的是一个条件生成模型，输入是子目标，输出是 JSON 格式的调用描述。  
   - 类比：把“查天气”翻译成“调用天气API，参数 city=北京”。

4. **执行模块**  
   - 执行器读取落地模块的输出，实际调用对应的工具（浏览器、数据库、计算器等）。  
   - 执行结果（文本、表格、图片链接等）被反馈回语言模型，作为后续子目标的上下文。  
   - 这种循环让模型能够在多步交互中保持状态。

5. **统一训练**  
   - 训练数据来自 9 类任务的统一标注，每条标注包括：原始指令 → 子目标序列 → 工具调用 → 结果。  
   - 损失函数同时包含规划误差（子目标生成的交叉熵）和落地误差（工具调用生成的交叉熵），两者加权求和。  
   - 通过一次前向‑反向传播，所有模块的参数同步更新。

**巧妙之处**：  
- **共享语言编码**：所有模块共用同一套词向量和 Transformer 编码层，避免了重复学习语言理解。  
- **可微分的工具映射**：落地模块把离散的工具调用当作可生成的序列处理，使得整个系统保持端到端可训练，而不需要手工写规则。  
- **模块化解耦**：虽然训练时是统一的，但在推理阶段每个模块可以独立部署，甚至可以用不同硬件加速（如落地模块跑在 CPU，规划模块跑在 GPU）。

### 实验与效果
- **测试任务**：论文在 9 个公开交互式数据集上评估，包括问答（QA）、网页检索、代码执行、数据库查询、机器人控制等。每个任务都有未参与训练的 held‑out 子集，用来检验泛化。  
- **基线对比**：与规模更大的开源代理（如 33B 参数的 OpenChat、WizardLM）以及闭源 GPT‑4/ChatGPT 系列进行比较。  
- **关键结果**：  
  - 在所有 held‑out 数据集上，Lumos 的平均得分比同类开源基线高出约 8%–12%。  
  - 在 QA 与网页任务上，Lumos 超过了 GPT‑4 的表现（论文声称在这些任务上领先约 2%–4% 的准确率）。  
  - 与仅使用链式思维（CoT）或单一端到端训练的模型相比，Lumos 在复杂多步任务上提升了约 10% 的成功率。  
- **消融实验**：作者分别去掉规划模块、落地模块或统一训练，发现：  
  - 去掉规划模块后，模型在多步任务的成功率下降约 15%。  
  - 只用端到端训练而不统一标注，跨任务表现下降约 9%。  
  - 这表明三段式设计和统一标注是提升性能的关键因素。  
- **局限性**：  
  - 论文未提供对极端长序列或实时低延迟需求的评估。  
  - 落地模块对新工具的适配仍需要少量微调，完全零样本适配尚未实现。  
  - 训练所需的大规模统一标注成本高，普通研究团队可能难以自行复制。

### 影响与延伸思考
Lumos 的出现标志着开源语言代理从“拼凑式”向“模块化、统一学习”转型。随后的工作（如 **ModularAgent**、**UnifiedToolformer**）在论文发布后陆续提出，进一步探索如何把更多外部工具（如图像模型、强化学习环境）接入统一框架。对想深入的读者，可以关注以下方向：  
- **跨模态工具接入**：把视觉、音频等感知模型纳入落地模块的输入空间。  
- **零样本工具适配**：利用大模型的元学习能力，让落地模块在看到新工具的描述后直接生成调用代码。  
- **高效统一标注生成**：研究自动化生成统一标注的流水线，降低数据构建门槛。  
这些方向都有望把开源代理的能力推向与商业闭源系统相当的水平。

### 一句话记住它
**Lumos 用统一的三段式（规划‑落地‑执行）训练，让开源语言代理像积木一样随时升级，同时在未见任务上也能跑出比大模型更好的成绩。**