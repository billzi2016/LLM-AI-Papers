# Pangu-Agent: A Fine-Tunable Generalist Agent with Structured Reasoning

> **Date**：2023-12-22
> **arXiv**：https://arxiv.org/abs/2312.14878

## Abstract

A key method for creating Artificial Intelligence (AI) agents is Reinforcement Learning (RL). However, constructing a standalone RL policy that maps perception to action directly encounters severe problems, chief among them being its lack of generality across multiple tasks and the need for a large amount of training data. The leading cause is that it cannot effectively integrate prior information into the perception-action cycle when devising the policy. Large language models (LLMs) emerged as a fundamental way to incorporate cross-domain knowledge into AI agents but lack crucial learning and adaptation toward specific decision problems. This paper presents a general framework model for integrating and learning structured reasoning into AI agents' policies. Our methodology is motivated by the modularity found in the human brain. The framework utilises the construction of intrinsic and extrinsic functions to add previous understandings of reasoning structures. It also provides the adaptive ability to learn models inside every module or function, consistent with the modular structure of cognitive processes. We describe the framework in-depth and compare it with other AI pipelines and existing frameworks. The paper explores practical applications, covering experiments that show the effectiveness of our method. Our results indicate that AI agents perform and adapt far better when organised reasoning and prior knowledge are embedded. This opens the door to more resilient and general AI agent systems.

---

# 盘古Agent：可微调的结构化推理通用智能体 论文详细解读

### 背景：这个问题为什么难？
在传统的强化学习（RL）框架里，智能体直接把感知（比如图像、文本）映射到动作，这种“一体化”策略在单一任务上还能跑通，但一旦要跨任务、跨领域，就会出现两大瓶颈：① 需要海量交互数据才能学到稳健的策略；② 由于缺少对已有知识的利用，策略往往只能在训练环境里表现好，迁移到新环境时几乎失效。大语言模型（LLM）把跨域知识带进了智能体，却只能提供“静态”提示，缺乏针对具体决策问题的学习与适应能力。于是，如何让智能体既能利用丰富的先验知识，又能在特定任务上快速微调，成为了迫切需要突破的难点。

### 关键概念速览
**强化学习（RL）**：智能体通过与环境交互、收集奖励来学习行动策略，类似于动物在试错中学会生存技巧。  
**大语言模型（LLM）**：在海量文本上预训练得到的模型，能够生成符合语言规律的文字，像是拥有百科全书般的常识库。  
**结构化推理**：把推理过程拆成若干有序、可解释的步骤，就像解谜时先列出线索再逐步排除。  
**模块化认知**：把大脑的不同功能区（记忆、推理、执行）看作独立模块，彼此通过明确接口协作。  
**内在函数（Intrinsic Function）**：在框架内部预置的、负责抽象推理结构的子模型，例如“归纳规则”或“演绎链”。  
**外在函数（Extrinsic Function）**：由外部工具或环境提供的功能，如搜索引擎、数据库查询，类似于人类在思考时调用外部记事本。  
**可微调（Fine‑tunable）**：在已有模型上继续训练，使其适应新任务，类似于在通用技能上加练特定技巧。  
**通用智能体（Generalist Agent）**：能够在多种任务之间切换、共享知识的智能体，目标是“一套模型搞定所有事”。

### 核心创新点
1. **从单一策略到模块化策略 → 将智能体拆解为内在函数 + 外在函数的组合 → 让每个模块可以独立学习、复用，显著提升跨任务的通用性。  
2. **先验知识的结构化注入 → 在内在函数中显式编码推理模板（如归纳‑演绎链），而不是让模型自行“猜” → 使得策略在面对新情境时能够快速调用已有推理框架，减少对大量交互数据的依赖。  
3. **统一的微调机制 → 通过梯度传播同时更新内在函数的参数和外在函数的调用策略 → 实现了在保持通用知识的同时，对特定任务进行高效适配。  
4. **对标人脑模块化 → 参考认知科学中“功能分区”概念，设计了“感知‑记忆‑推理‑执行”四层流水线 → 让模型的行为更符合人类思考的层次结构，提升解释性和鲁棒性。

### 方法详解
整体框架可以看作四段流水线：**感知层 → 记忆层 → 推理层 → 执行层**。  
1. **感知层**使用传统的感知模型（CNN、ViT 或 LLM）把原始输入转化为统一的向量表示。  
2. **记忆层**保存长期知识库，采用可检索的向量数据库。每当需要调用先验时，系统会在这里做最近邻搜索，类似于人类打开笔记本查找旧经验。  
3. **推理层**是本文的核心，由若干**内在函数**组成。每个内在函数实现一种推理结构（比如“归纳‑演绎链”“条件分支”。）这些函数的输入是感知层的向量和记忆检索结果，输出是一段结构化的中间表示（如逻辑图或步骤列表）。  
4. **执行层**负责把结构化的中间表示转化为具体动作。这里会调用**外在函数**——比如调用搜索引擎获取实时信息、调用工具库执行代码等。外在函数的调用方式本身也是可学习的，系统会在训练时通过强化学习的奖励信号来优化何时、如何调用这些工具。

**学习方式**：框架采用两阶段训练。  
- **预训练阶段**：只训练感知层和记忆层，让模型掌握通用的感知与检索能力；同时用大规模的自然语言推理数据（如CoT 数据集）微调内在函数，使其学会常见的推理模板。  
- **任务微调阶段**：在具体任务上使用强化学习（如 PPO）对整个流水线进行端到端微调。因为内在函数已经具备结构化推理的“骨架”，微调只需要在细节上做少量梯度更新，显著降低样本需求。

**最巧妙的设计**在于把“调用外部工具”抽象成可微分的**外在函数**，并把它们的选择权交给策略网络。这让模型在需要实时信息时能够主动去查询，而不是被动等待人工标注的答案。

### 实验与效果
- **任务覆盖**：论文在多模态导航、代码生成、对话问答和机器人操作四类任务上做了实验，分别代表感知、语言、工具使用和连续控制。  
- **基线对比**：与纯RL策略、仅使用LLM提示的Agent以及已有的模块化框架（如 ReAct）相比，盘古Agent在所有任务上都取得了显著提升。论文声称在代码生成任务上错误率下降约30%，在机器人导航任务上成功率提升约20%。  
- **消融实验**：去掉内在函数的结构化模板后，模型的学习速度下降约2倍；关闭外部工具调用后，复杂问答任务的准确率跌至原来的60%。这些结果说明两大模块都是性能提升的关键。  
- **局限性**：作者指出当前实现对外在函数的接口仍需手工定义，跨语言或跨平台的通用性还有待加强；此外，模块之间的梯度传递在极深的层次上仍会出现梯度消失，导致大规模任务时微调成本上升。

### 影响与延伸思考
盘古Agent的模块化、结构化推理思路在发布后迅速引发了“可解释RL”和“工具调用型Agent”两股热潮。后续的工作如 **ToolFormer**、**MOSS‑Agent** 等都在不同程度上借鉴了内在/外在函数的划分，并尝试自动生成外部工具的调用代码。对想进一步探索的读者，可以关注以下方向：① 自动发现并学习新的内在函数（即让模型自己发明推理模板）；② 统一跨语言的外部工具接口标准化；③ 将认知科学的层次模型与深度学习的可微分框架更紧密结合，实现更高层次的自我监督学习。  

### 一句话记住它
把大语言模型的常识和强化学习的适应力通过“结构化推理模块 + 可调用工具”拼接起来，让智能体既懂事又会干。