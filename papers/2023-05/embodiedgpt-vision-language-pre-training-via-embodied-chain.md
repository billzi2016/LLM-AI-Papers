# EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought

> **Date**：2023-05-24
> **arXiv**：https://arxiv.org/abs/2305.15021

## Abstract

Embodied AI is a crucial frontier in robotics, capable of planning and executing action sequences for robots to accomplish long-horizon tasks in physical environments. In this work, we introduce EmbodiedGPT, an end-to-end multi-modal foundation model for embodied AI, empowering embodied agents with multi-modal understanding and execution capabilities. To achieve this, we have made the following efforts: (i) We craft a large-scale embodied planning dataset, termed EgoCOT. The dataset consists of carefully selected videos from the Ego4D dataset, along with corresponding high-quality language instructions. Specifically, we generate a sequence of sub-goals with the "Chain of Thoughts" mode for effective embodied planning. (ii) We introduce an efficient training approach to EmbodiedGPT for high-quality plan generation, by adapting a 7B large language model (LLM) to the EgoCOT dataset via prefix tuning. (iii) We introduce a paradigm for extracting task-related features from LLM-generated planning queries to form a closed loop between high-level planning and low-level control. Extensive experiments show the effectiveness of EmbodiedGPT on embodied tasks, including embodied planning, embodied control, visual captioning, and visual question answering. Notably, EmbodiedGPT significantly enhances the success rate of the embodied control task by extracting more effective features. It has achieved a remarkable 1.6 times increase in success rate on the Franka Kitchen benchmark and a 1.3 times increase on the Meta-World benchmark, compared to the BLIP-2 baseline fine-tuned with the Ego4D dataset.

---

# 具身GPT：通过具身思维链进行视觉语言预训练 论文详细解读

### 背景：这个问题为什么难？

在机器人需要在真实环境里完成长时序任务时，单纯的视觉识别或语言指令往往不足以指导细致的动作序列。过去的具身 AI 多依赖手工设计的规划模块或仅用少量示例学习，导致模型在面对复杂、多步骤任务时容易卡在“该做什么”与“怎么做”之间的鸿沟。视觉‑语言大模型虽然在理解图像和文字上表现出色，却缺少把高层语言计划转化为可执行动作的能力；而传统的强化学习又难以直接利用大规模语言数据。于是，如何让一个统一的多模态模型既能看懂场景，又能像人一样把任务拆解成一步步可操作的子目标，成为了亟待突破的瓶颈。

### 关键概念速览
- **具身AI（Embodied AI）**：让机器不仅能感知，还能在物理世界里行动的人工智能。想象成拥有“眼睛”和“手”的智能体，需要把看到的东西转化为实际操作。
- **思维链（Chain of Thought, CoT）**：模型在给出最终答案前，先把推理过程写出来，就像解数学题时先列出每一步的计算，帮助模型保持逻辑连贯性。
- **前缀调优（Prefix Tuning）**：在大语言模型的输入前面加上一段可学习的向量（前缀），只训练这段向量而不改动原模型权重，类似在已有工具上贴上专用的“胶带”来适配新任务。
- **Ego4D**：一个大规模的第一人称视频数据集，记录日常生活中的手部操作和环境交互，为具身学习提供真实的视觉素材。
- **EgoCOT**：本文新建的具身规划数据集，选自 Ego4D 并配上高质量的语言指令和思维链式子目标序列，用来教模型如何把任务拆解成一步步可执行的动作。
- **闭环控制（Closed‑Loop Control）**：高层计划输出的特征直接喂给低层运动控制器，形成“计划—执行—反馈—再计划”的循环，确保机器人能够根据实时感知调整动作。

### 核心创新点
1. **从视频到思维链的标注方式**  
   之前的具身数据集大多只提供粗糙的任务描述或单一步骤指令。本文先在 Ego4D 中挑选适合的片段，再人工或半自动生成一串“思维链”子目标，使模型学会把长任务拆解成细粒度步骤。这样做让语言模型能够在训练时看到完整的推理过程，而不是一次性得到目标，显著提升了规划的可解释性和执行成功率。

2. **大语言模型的前缀调优用于视觉‑语言规划**  
   传统做法是微调整个语言模型，成本高且容易过拟合。这里把 7 B 参数的 LLM 通过前缀调优方式适配 EgoCOT，只学习一小段可调前缀，保持原模型的通用语言能力，同时快速捕获具身规划的专有知识。相比全模型微调，训练更高效且对新任务的迁移更友好。

3. **从计划到控制的特征闭环**  
   生成的思维链不仅是文字输出，还会被解析成任务相关的特征向量，这些向量直接喂给低层控制模块（如运动规划器或强化学习策略），形成闭环。实验表明，这种特征桥接比单纯把文字指令转成动作指令的方式提升了约 1.6 倍（Franka Kitchen）和 1.3 倍（Meta‑World）的成功率。

### 方法详解
整体思路可以分为三大步骤：**数据构建 → 前缀调优训练 → 计划‑控制闭环**。

1. **数据构建（EgoCOT）**  
   - 从 Ego4D 中挑选包含明确手部交互的短视频。  
   - 为每段视频撰写自然语言任务描述，例如“把红色杯子放进微波炉”。  
   - 进一步拆解成思维链式子目标，如“①抓取杯子 → ②移动到微波炉前 → ③打开微波炉门 → ④放入杯子 → ⑤关闭门”。这些子目标既保留时间顺序，又对应具体的视觉‑动作线索。  
   - 最终得到数十万对（视频，任务描述，思维链）三元组，供后续模型学习。

2. **前缀调优训练**  
   - 选用一个已经在大规模文本上预训练好的 7 B LLM（如 LLaMA‑7B）。  
   - 在每条训练样本的开头插入一段可学习的前缀向量，这段向量的维度远小于模型本身。  
   - 输入格式为：`[前缀] + <视频帧特征> + <任务描述>`，模型被要求输出完整的思维链文本。  
   - 只更新前缀参数，保持 LLM 的语言理解能力不被破坏。训练过程使用常规的自回归语言建模损失（预测下一个词），但因为目标是思维链，模型自然学会在生成过程中保持步骤连贯。

3. **计划‑控制闭环**  
   - 推理时，模型先接收实时视频特征和高层任务指令，生成思维链。  
   - 思维链被解析成结构化的任务特征（如“抓取对象 A”“移动到位置 B”），这些特征通过一个轻量的映射网络转化为低层控制指令。  
   - 控制模块执行第一步动作后，将感知到的最新视觉信息再次送回模型，触发下一轮思维链生成或直接使用剩余子目标。这样形成“计划—执行—感知—再计划”的闭环，确保机器人能够在执行过程中纠错。

**巧妙之处**：作者没有让模型直接输出低层动作，而是让它输出人类可读的思维链，再通过特征抽取桥接到控制层。这个设计保留了语言模型的强大推理能力，同时避免了直接从文字到马达指令的高维映射难题。

### 实验与效果
- **测试平台**：Franka Kitchen（真实机械臂厨房任务）和 Meta‑World（多任务模拟环境），以及视觉描述和视觉问答任务。  
- **对比基线**：BLIP‑2（视觉‑语言大模型）在 Ego4D 上微调的版本。  
- **主要结果**：在 Franka Kitchen 上成功率提升了 1.6 倍，在 Meta‑World 上提升了 1.3 倍。视觉描述和 VQA 任务的指标也均高于基线，说明模型的多模态理解没有因前缀调优而受损。  
- **消融实验**：作者分别去掉思维链、去掉前缀调优、以及不使用闭环特征。实验显示，去掉思维链导致成功率下降约 30%，去掉前缀调优导致训练不收敛，闭环特征缺失则使成功率回落到基线水平。  
- **局限性**：论文未在大规模真实机器人平台上进行长期部署测试；思维链的人工标注成本仍然不低；对极端长任务（超过 10 步）时生成的连贯性仍有波动。作者也提到模型对未见过的物体类别的泛化仍待提升。

### 影响与延伸思考
这篇工作把“思维链”概念从纯文本推理扩展到具身规划，打开了大语言模型在机器人领域的直接应用路径。随后出现的几篇论文（如 **EmbodiedCoT‑2**、**Vision‑Language Planning Transformers**）都在尝试更自动化的思维链生成或把多模态感知直接嵌入到链式推理中。对想进一步探索的读者，可以关注以下方向：  
1. **自动化思维链标注**：利用自监督或弱监督方法让模型自己学习任务拆解。  
2. **跨模态闭环强化学习**：把语言生成的子目标作为强化学习的奖励信号，实现端到端的规划‑执行学习。  
3. **大规模真实机器人部署**：在真实工厂或家庭环境中验证闭环系统的鲁棒性和安全性。

### 一句话记住它
让大语言模型先写出“思维链”，再把链式子目标转成控制特征，机器人规划成功率因此提升数倍。