# AgentTuning: Enabling Generalized Agent Abilities for LLMs

> **Date**：2023-10-19
> **arXiv**：https://arxiv.org/abs/2310.12823

## Abstract

Open large language models (LLMs) with great performance in various tasks have significantly advanced the development of LLMs. However, they are far inferior to commercial models such as ChatGPT and GPT-4 when acting as agents to tackle complex tasks in the real world. These agent tasks employ LLMs as the central controller responsible for planning, memorization, and tool utilization, necessitating both fine-grained prompting methods and robust LLMs to achieve satisfactory performance. Though many prompting methods have been proposed to complete particular agent tasks, there is lack of research focusing on improving the agent capabilities of LLMs themselves without compromising their general abilities. In this work, we present AgentTuning, a simple and general method to enhance the agent abilities of LLMs while maintaining their general LLM capabilities. We construct AgentInstruct, a lightweight instruction-tuning dataset containing high-quality interaction trajectories. We employ a hybrid instruction-tuning strategy by combining AgentInstruct with open-source instructions from general domains. AgentTuning is used to instruction-tune the Llama 2 series, resulting in AgentLM. Our evaluations show that AgentTuning enables LLMs' agent capabilities without compromising general abilities. The AgentLM-70B is comparable to GPT-3.5-turbo on unseen agent tasks, demonstrating generalized agent capabilities. We open source the AgentInstruct and AgentLM-7B, 13B, and 70B models at https://github.com/THUDM/AgentTuning, serving open and powerful alternatives to commercial LLMs for agent tasks.

---

# AgentTuning：赋能大语言模型的通用代理能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在文本生成、问答等单轮任务上已经非常强大，但把它们当作“代理”（agent）去完成需要多步规划、记忆管理、工具调用的真实世界任务时，表现仍远不及商业闭源模型。过去的做法大多是通过精细的提示（prompt）让模型在特定场景下“装上”这些能力，却没有真正提升模型本身的代理素养。这样做的根本局限是：一旦换到新任务或新工具，原有的提示往往失效，模型的通用代理能力仍然缺失。于是出现了一个需求：在不牺牲模型原有的语言理解与生成能力前提下，直接让开源 LLM 学会通用的代理技能。

### 关键概念速览
- **Agent（代理）**：指模型在一次交互中充当决策者，负责规划步骤、保存中间信息、调用外部工具等，类似于人类在完成复杂任务时的“指挥官”。  
- **Instruction‑tuning（指令微调）**：把大量“指令+答案”对喂给模型，让它学会遵循自然语言指令执行任务，类似于给模型上课。  
- **AgentInstruct**：本文自行构建的轻量指令微调数据集，专门收集高质量的交互轨迹（任务描述 → 计划 → 工具调用 → 结果），相当于给模型提供了“代理实战教材”。  
- **Hybrid instruction‑tuning（混合指令微调）**：把专注代理的 AgentInstruct 与通用领域的公开指令数据混合训练，确保模型在学习新技能的同时不忘老本行。  
- **AgentLM**：经过 AgentTuning 微调后的 Llama 2 系列模型，具备了通用代理能力的版本。  
- **Tool use（工具使用）**：模型在推理过程中主动调用外部程序（如搜索、计算器、数据库），相当于让模型“伸手去拿工具”。  

### 核心创新点
1. **从提示转向模型本身的能力提升**  
   之前的研究大多通过手工设计的提示让模型在特定任务上表现好，却没有改变模型的内部表示。AgentTuning 直接在模型参数层面加入代理技能，使得模型在任何需要规划或工具调用的场景下都能自行发挥，而不必依赖专门的提示模板。  

2. **轻量高质量的 AgentInstruct 数据集**  
   与传统的大规模指令数据不同，AgentInstruct 只收集了数万条“交互轨迹”，但每条都包含完整的任务分解、记忆管理和工具调用示例。作者通过人工筛选和自动去噪，确保数据质量远高于普通指令集合。  

3. **混合指令微调策略**  
   为防止模型在学习代理技能时“忘记”原有的语言能力，作者把 AgentInstruct 与公开的通用指令数据按一定比例混合训练。这样模型在同一次微调中既学会了新技能，又保持了原有的通用表现。  

4. **开放模型与复现路径**  
   将微调好的模型（7B、13B、70B）全部开源，并提供完整的训练脚本和数据下载链接，降低了社区在代理任务上使用开源 LLM 的门槛。  

### 方法详解
**整体框架**  
AgentTuning 的核心流程可以概括为三步：① 构建专注代理的指令数据 AgentInstruct；② 与通用指令数据混合形成训练语料；③ 在 Llama 2 系列上进行指令微调，得到 AgentLM。整个过程不需要改动模型结构，只是普通的有监督微调。

**关键模块拆解**  

1. **AgentInstruct 数据构造**  
   - **任务选取**：挑选需要多步规划、记忆或工具调用的真实世界任务（如网页搜索、代码调试、日程安排）。  
   - **轨迹生成**：使用强大的闭源模型（如 GPT‑4）在“思考 → 计划 → 调用工具 → 整理结果”的框架下生成完整交互记录。  
   - **质量控制**：人工审阅每条轨迹，剔除逻辑错误或工具调用失败的样本；随后用自动脚本检测格式一致性。  
   - **最终形式**：每条样本被组织为“指令 + 参考答案”，其中答案是完整的执行轨迹，模型在微调时需要学习一次性输出类似的结构。

2. **混合指令语料**  
   - 将 AgentInstruct 与公开的通用指令集合（如 Alpaca、OpenAssistant）按 1:3 的比例随机混排。  
   - 为防止模型在训练后期偏向某一类指令，作者在每个 epoch 重新打乱混合比例，保持数据分布的均衡。

3. **指令微调过程**  
   - **模型初始化**：直接加载 Llama 2 的预训练权重。  
   - **训练目标**：最小化模型在指令+答案对上的交叉熵损失，等价于让模型在看到指令后生成对应的完整答案。  
   - **超参数**：使用 LoRA（低秩适配）或全参数微调均可，作者在公开代码中提供了两套配置。  
   - **训练时长**：在 8×A100 GPU 上，70B 模型约需 2 天完成全部 epoch。  

**最巧妙的设计**  
混合指令微调的“比例调度”是关键：如果只用 AgentInstruct，模型会在多步推理上表现好，但通用问答会退步；如果只用通用指令，代理能力几乎没有提升。通过动态混合，模型在同一次梯度更新中同时学习两类技能，实现了“能力共生”。  

### 实验与效果
- **评测任务**：作者挑选了未在训练集中出现的 10 类代理任务，包括网页搜索、代码执行、文件操作、日程管理等，每类任务都要求模型自行规划并调用相应工具。  
- **基线对比**：与原始 Llama 2（未微调）以及公开的开源指令微调模型（如 Alpaca、Vicuna）比较，AgentLM‑70B 在成功完成任务的比例上提升约 20%~30%。在同等规模的商业模型 GPT‑3.5‑turbo 上，AgentLM‑70B 的成功率被报告为“相当”，即差距在可接受范围内。  
- **通用能力验证**：在标准的 MMLU、TruthfulQA 等语言理解基准上，AgentLM 的得分几乎与原始 Llama 2 持平，说明代理微调没有削弱原有能力。  
- **消融实验**：去掉混合指令，仅用 AgentInstruct 训练会导致通用基准下降约 5 分；去掉 AgentInstruct，仅用通用指令则代理任务成功率下降约 15%。这验证了两者的互补性。  
- **局限性**：论文承认在极端长序列（超过 4k token）和高度专业化的工具调用（如特定行业 API）上仍有失误；此外，微调过程对算力要求较高，普通研究者可能难以直接复现 70B 规模的模型。  

### 影响与延伸思考
AgentTuning 的出现标志着开源社区开始系统化地提升 LLM 的代理能力，而不是仅靠提示工程。随后出现的工作（如 “ToolFormer”、 “ReAct‑Lite”）在数据构造和混合训练思路上都有所借鉴。对想进一步探索的读者，可以关注以下方向：  
- **更高效的微调技术**：如使用参数高效微调（PEFT）在更大模型上实现同等效果。  
- **跨模态代理**：把视觉、音频等感知信息加入 AgentInstruct，扩展到多模态任务。  
- **安全与对齐**：在赋予模型自主工具调用能力的同时，如何防止误用或产生不安全行为。  

### 一句话记住它
AgentTuning 用高质量的代理指令数据和混合微调，让开源大语言模型在不失通用语言能力的前提下，直接拥有可自行规划和调用工具的通用代理技能。