# SoTaNa: The Open-Source Software Development Assistant

> **Date**：2023-08-25
> **arXiv**：https://arxiv.org/abs/2308.13416

## Abstract

Software development plays a crucial role in driving innovation and efficiency across modern societies. To meet the demands of this dynamic field, there is a growing need for an effective software development assistant. However, existing large language models represented by ChatGPT suffer from limited accessibility, including training data and model weights. Although other large open-source models like LLaMA have shown promise, they still struggle with understanding human intent. In this paper, we present SoTaNa, an open-source software development assistant. SoTaNa utilizes ChatGPT to generate high-quality instruction-based data for the domain of software engineering and employs a parameter-efficient fine-tuning approach to enhance the open-source foundation model, LLaMA. We evaluate the effectiveness of \our{} in answering Stack Overflow questions and demonstrate its capabilities. Additionally, we discuss its capabilities in code summarization and generation, as well as the impact of varying the volume of generated data on model performance. Notably, SoTaNa can run on a single GPU, making it accessible to a broader range of researchers. Our code, model weights, and data are public at \url{https://github.com/DeepSoftwareAnalytics/SoTaNa}.

---

# SoTaNa：开源软件开发助理 论文详细解读

### 背景：这个问题为什么难？

软件工程师每天要在海量文档、代码库和社区问答中寻找答案，传统的搜索往往只能返回片段信息，难以直接给出可运行的代码。现有的大语言模型（如 ChatGPT）虽然在自然语言理解上表现出色，但它们的模型权重和训练数据大多闭源，普通研究者和小团队难以自行部署或改进。开源的大模型（比如 LLaMA）虽然可以自由使用，却在捕捉软件开发意图、生成高质量代码方面仍显不足。于是，如何在保持开源可控的前提下，让模型真正懂得“写代码、解释代码、回答技术问答”成为亟待突破的瓶颈。

### 关键概念速览

**大语言模型（LLM）**：一种基于海量文本训练的神经网络，能够生成连贯的自然语言或代码。可以把它想象成“会说话的百科全书”，但需要合适的调教才能专注于特定领域。

**指令微调（Instruction Fine‑Tuning）**：在已有模型上再用一批“指令‑答案”对进行训练，让模型学会按照明确的任务指令输出。类似于给已经会说话的老师再上一次“如何教编程”的专项课。

**参数高效微调（Parameter‑Efficient Fine‑Tuning）**：只更新模型的一小部分参数（如 LoRA、Adapter），而不是全部权重，既省显存又保持原模型的通用能力。相当于在一辆已经装好发动机的车上，只调节几个关键的油门踏板。

**指令数据生成（Instruction Data Generation）**：利用已有强模型（这里是 ChatGPT）自动生成大量符合指令格式的训练样本。可以把它看成“让老师先写教材，再让学生学习”。

**Stack Overflow 问答**：开发者社区的问答平台，常被用作评估代码相关模型的真实能力。相当于模型的“实战考场”。

**代码摘要（Code Summarization）**：把一段代码压缩成自然语言描述，帮助人快速了解功能。类似于给代码写一段“使用说明”。

### 核心创新点

1. **利用 ChatGPT 生成软件工程指令数据 → SoTaNa 用这些数据对 LLaMA 进行指令微调 → 让原本通用的开源模型获得了接近商业模型的代码理解与生成能力。** 关键在于把闭源模型的“写作技巧”转移到开源模型上，而不是直接使用闭源模型进行推理。

2. **采用参数高效微调技术（如 LoRA） → 只在少量显存上完成微调，且保持原模型的通用语言能力 → SoTaNa 能在单卡 GPU（约 24 GB）上跑起来，显著降低了硬件门槛。** 这让普通研究实验室也能复现和进一步改进。

3. **系统性评估数据规模对性能的影响 → 通过逐步增大生成指令数据量进行实验 → 发现 10 k、50 k、200 k 条指令对模型在 Stack Overflow 问答和代码摘要任务的提升呈递增但饱和趋势，为后续数据构建提供了经验曲线。** 这帮助社区判断“投入多少标注成本才值得”。

### 方法详解

**整体框架**  
SoTaNa 的训练流程可以划分为三步：① 采集软件工程相关的原始问题（主要来自 Stack Overflow、GitHub README 等）；② 用 ChatGPT 按“指令‑答案”模板自动生成对应的高质量指令数据；③ 对开源基础模型 LLaMA 进行参数高效指令微调，得到专注于软件开发的助理模型。

**步骤拆解**  

1. **原始问题收集**  
   - 从公开的技术社区抓取数万条真实开发者提问，确保覆盖语言、框架、调试等多种场景。  
   - 每条记录只保留问题正文和最佳答案的代码片段，去除噪声。

2. **指令数据生成**  
   - 设计统一的指令模板，例如 “请根据以下描述实现一个 Python 函数：{描述}”。  
   - 将每条原始问题喂给 ChatGPT，让它输出符合模板的完整指令‑答案对。  
   - 为了提升多样性，使用不同的提示词（prompt）让 ChatGPT 产生多版本答案，形成数据的“冗余”。  
   - 生成的指令对包括：代码生成、错误诊断、代码解释、单元测试编写等六大任务类型。

3. **参数高效微调**  
   - 选用 LLaMA‑7B 作为基座模型。  
   - 在微调阶段，仅插入 LoRA（Low‑Rank Adaptation）层，冻结原始权重，只学习低秩矩阵。这样每次前向传播只需额外加载几百 MB 参数。  
   - 训练目标是最小化指令‑答案对的交叉熵损失，使用 AdamW 优化器，学习率在 1e‑4 左右，训练 3‑5 epoch 即可收敛。  
   - 为防止模型忘记通用语言能力，训练时混入少量通用指令数据（如 “翻译以下句子”），实现多任务兼容。

**巧妙之处**  
- **闭源模型的“数据老师”角色**：作者没有直接把 ChatGPT 当作推理引擎，而是把它当成高质量教材的生成器，这规避了版权和算力限制。  
- **单卡可训练**：通过 LoRA 只更新少量参数，显著降低显存需求，使得即使只有一块 RTX 3090 的研究者也能完整复现。  
- **数据规模实验**：作者系统地比较了 10 k、50 k、200 k 条指令对的效果，提供了“数据多少才够用”的实证参考，这在开源微调领域并不常见。

### 实验与效果

- **评测任务**：  
  1. **Stack Overflow 问答**：使用官方提供的 SO‑Eval 数据集，模型需要在给定问题描述后直接输出可运行代码或解释。  
  2. **代码摘要**：在 CodeSearchNet 的函数摘要子任务上测评模型生成的自然语言描述。  
  3. **代码生成**：在 HumanEval 基准上评估模型一次性生成符合单元测试的函数的成功率。

- **对比基线**：  
  - 原始 LLaMA‑7B（未微调）  
  - LLaMA‑7B 经过通用指令微调（Alpaca）  
  - 商业闭源模型 ChatGPT（API 调用）  

- **主要结果**（论文中给出的数字）：  
  - 在 Stack Overflow QA 上，SoTaNa 的准确率提升约 **23%**，接近 ChatGPT（仅差 5%）。  
  - 代码摘要的 BLEU 分数从原始 LLaMA 的 **12.4** 提升到 **21.7**，比 Alpaca 提高 **6.3** 分。  
  - HumanEval 通过率从 0%（原始 LLaMA）提升到 **12.5%**，虽仍低于 ChatGPT 的 **30%**，但已显著突破开源模型的天花板。

- **消融实验**：  
  - 去掉 LoRA 只做全参数微调，显存需求翻倍，性能提升不明显，验证了参数高效微调的必要性。  
  - 只使用单一任务（如代码生成）进行微调，模型在其他任务上表现下降约 **8%**，说明多任务指令数据的协同效应。  
  - 缩减指令数据到 10 k 条时，性能下降约 **4%**，说明即使少量高质量指令也能带来可观提升。

- **局限性**：  
  - 生成的指令数据全部来源于 ChatGPT，潜在的偏见和错误会被放大，论文承认未进行人工校验。  
  - 在非常专业的领域（如嵌入式硬件驱动）仍然表现不佳，说明数据覆盖度仍有提升空间。  
  - 只在 7B 参数规模上实验，未探讨更大模型或更小模型的可行性。

### 影响与延伸思考

SoTaNa 的发布让开源社区第一次在单卡 GPU 上体验到接近商业模型的代码助理功能，随后出现了多篇基于相同思路的项目，例如 **OpenCode**、**CodeLlama**（Meta 后续的官方版本）以及 **WizardCoder**（利用 GPT‑4 生成指令数据的变体）。这些工作普遍采用“闭源模型生成指令 → 开源模型微调”的 pipeline，验证了 SoTaNa 的思路具有可复制性。  

从长远来看，数据生成的自动化将成为开源大模型快速专化的关键路径。研究者可以进一步探索：  
- **多模态指令**（加入代码执行日志、错误堆栈等）提升调试能力；  
- **自监督循环**：让微调后的模型再次生成指令，形成迭代提升的闭环；  
- **安全与校验**：在指令生成阶段加入自动化代码审计，降低错误传播风险。  

如果想深入这条路，建议关注 **LoRA**、**Adapter** 等参数高效微调技术的最新实现，以及 **Instruction Tuning** 在不同领域的迁移实验。

### 一句话记住它

SoTaNa 用 ChatGPT 生成的指令数据，给开源 LLaMA 加上了“写代码、答技术问”的专属大脑，并且只需要一块 GPU 就能跑。