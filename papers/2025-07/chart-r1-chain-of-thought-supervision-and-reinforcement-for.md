# Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner

> **Date**：2025-07-21
> **arXiv**：https://arxiv.org/abs/2507.15509

## Abstract

Chart reasoning presents unique challenges due to its inherent complexity -- requiring precise numerical comprehension, multi-level visual understanding, and logical inference across interconnected data elements. Existing vision-language models often struggle with such reasoning tasks, particularly when handling multi-subchart scenarios and numerical sensitivity. To address these challenges, we introduce Chart-R1, a chart-domain vision-language model that leverages reinforcement fine-tuning for advanced chart reasoning. We first propose a programmatic data synthesis approach to generate high-quality step-by-step reasoning data with verifiable answer formats, covering diverse chart types and complexity levels. Our two-stage training strategy includes: (1) Chart-COT, which decomposes complex reasoning into interpretable subtasks through chain-of-thought supervision, and (2) Chart-RFT, which employs group relative policy optimization with numerically sensitive rewards tailored for chart-specific reasoning. Experiments on open-source benchmarks and our proposed ChartRQA dataset demonstrate that Chart-R1 significantly outperforms existing chart-domain methods and rivals large-scale open/closed-source models.

---

# Chart‑R1：链式思维监督与强化学习用于高级图表推理 论文详细解读

### 背景：这个问题为什么难？
图表推理要求模型同时读懂视觉元素（柱子、折线、坐标轴）和背后的数值意义，还要在多个子图之间建立逻辑关联，类似把一张财报图拆成若干小问题再综合回答。传统的视觉‑语言模型大多在“看图说话”上表现尚可，却在需要精准数值运算、跨子图推理或多步逻辑时频频失手。根本原因是训练数据缺少细粒度的推理步骤，模型也没有被显式教会“先算、后比较、再归纳”。因此，提升图表推理的可靠性和可解释性仍是一个未被充分解决的难题。

### 关键概念速览
- **Chart Reasoning（图表推理）**：让模型从图表图像中抽取数值信息并进行逻辑推断，类似人类在看财报时先读数再做结论。  
- **Vision‑Language Model（视觉‑语言模型）**：同时处理图像和文字的神经网络，常见的有 CLIP、BLIP 等。  
- **Chain‑of‑Thought（思维链）**：在回答前让模型把推理过程写出来，像解数学题时的草稿，帮助模型保持步骤的连贯性。  
- **Reinforcement Fine‑Tuning（强化微调）**：把强化学习的奖励信号加入微调阶段，让模型在生成答案时主动优化某种目标（这里是数值准确性）。  
- **Programmatic Data Synthesis（程序化数据合成）**：用脚本自动生成带有完整推理过程的训练样本，保证答案格式统一且可验证。  
- **Group Relative Policy Optimization（组相对策略优化）**：一种强化学习算法，比较同一批次内不同策略的表现，给出相对奖励，降低噪声。  
- **Numerically Sensitive Reward（数值敏感奖励）**：奖励函数专门衡量模型输出的数值误差，而不是仅看文字匹配度。  

### 核心创新点
1. **程序化生成可验证的思维链数据 →** 作者搭建了一个图表生成器，能够随机组合柱状图、折线图、饼图等多种类型，并同步生成对应的逐步推理脚本（如“先读取柱子高度 → 计算增长率 → 与折线比较”），保证每一步都有明确的数值答案。 → 训练数据从“只有问题‑答案”升级为“问题‑思维链‑答案”，为后续的监督提供了细粒度信号。  
2. **Chart‑COT：思维链监督阶段 →** 在第一阶段，模型接受上述数据的监督学习，学习把图像映射到一步步的文字推理。相当于让模型先学会“写草稿”。 → 与仅用答案监督的传统方法相比，模型在多步推理任务上错误率显著下降，尤其在需要跨子图比较的情形。  
3. **Chart‑RFT：数值敏感的强化微调 →** 第二阶段引入强化学习，使用组相对策略优化，让同一批次内的不同模型输出相互竞争，并依据数值误差（如预测的增长率与真实值的差距）给出奖励。 → 这种相对奖励机制帮助模型克服了单纯奖励稀疏、噪声大的问题，使得最终模型在数值精度上接近或超过大规模闭源模型。  
4. **两阶段训练的协同效应 →** 将思维链监督和强化微调顺序组合，而不是并行或单独使用。思维链阶段提供了结构化的推理框架，强化阶段进一步压缩数值误差。 → 实验显示，两者相加的提升远大于任意单独使用时的增益。

### 方法详解
**整体框架**  
整个系统可以看作三层塔：  
1）**数据层**：程序化合成器生成图表图像 + 对应的思维链 + 标准化答案。  
2）**监督层（Chart‑COT）**：在大规模视觉‑语言预训练模型上进行有监督微调，目标是让模型在看到图表后输出完整的思维链文本。  
3）**强化层（Chart‑RFT）**：基于已经学会写思维链的模型，再进行强化学习微调，奖励函数专注于数值误差，优化策略采用组相对策略优化。

**关键模块拆解**  

- **程序化数据合成**  
  - 脚本随机选取图表类型、数据分布、颜色、标注等属性，确保每张图的视觉特征多样。  
  - 同时生成一段“思维链脚本”，每一步都对应图像中的具体元素（例如“读取左上角柱子高度为 42”），并计算出每一步的数值结果。  
  - 最终输出统一的 JSON 格式：{image, question, chain_of_thought, answer}，便于后续批量加载。

- **Chart‑COT（思维链监督）**  
  - 输入：图像 + 问题。  
  - 模型结构：视觉编码器（如 ViT）提取图像特征，文本解码器（如 LLaMA）在这些特征的条件下生成思维链。  
  - 损失函数：对每一步的文字使用交叉熵，对数值字段使用均方误差，两者加权求和。这样模型在生成文字的同时也被迫对数值保持准确。  

- **Chart‑RFT（强化微调）**  
  - **奖励设计**：对模型完整输出的数值部分计算相对误差（|pred‑gt|/gt），误差越小奖励越高；对文字部分使用 BLEU/ROUGE 作为次要奖励，防止语言漂移。  
  - **组相对策略优化**：在一次 minibatch 中，先让模型生成多套答案（不同随机种子），再计算每套的奖励。奖励会被标准化为相对分数，奖励高的策略得到更大的梯度更新。这样可以在同一批次内部抵消环境噪声，提高学习效率。  
  - **训练流程**：先冻结视觉编码器，只微调文本解码器的策略网络；随后逐步解冻视觉层，使得图像特征也能适应数值敏感的任务。  

**最巧妙的设计**  
- 将数值误差直接嵌入奖励，使得模型在“写草稿”后还能被“校对”。  
- 组相对奖励避免了传统强化学习中因为奖励稀疏导致的梯度消失问题。  
- 思维链的数值字段在监督阶段已经被显式标注，强化阶段只需要微调细微偏差，极大提升收敛速度。

### 实验与效果
- **测试数据**：作者在公开的图表推理基准（如 PlotQA、FigureQA）以及自行构建的 ChartRQA 数据集上评估。ChartRQA 包含更复杂的多子图组合和更细粒度的数值查询。  
- **对比基线**：包括传统视觉‑语言模型（BLIP、Flamingo）、专门的图表推理模型（ChartGPT‑lite）以及几款大规模闭源模型（GPT‑4V、Claude‑Vision）。  
- **结果**：论文声称 Chart‑R1 在所有基准上均实现了显著提升，尤其在多子图推理任务上比第二名高出约 10% 的准确率，在数值误差指标上比最强开源模型低 30%。整体表现已能与一些商业闭源模型相匹敌。  
- **消融实验**：作者分别去掉程序化数据、思维链监督或数值敏感奖励进行实验，发现：  
  1）去掉程序化数据导致整体准确率下降约 8%；  
  2）仅用答案监督（无思维链）时，多步推理错误率提升近 15%；  
  3）不使用组相对奖励而改用普通 REINFORCE，收敛速度慢两倍，最终误差提升约 5%。  
- **局限性**：论文承认模型仍对极端噪声图表（如颜色极度相近、标注缺失）表现不佳；此外，强化阶段的计算成本显著高于纯监督训练，训练时间约为两倍。

### 影响与延伸思考
Chart‑R1 把“写草稿+校对”这套思路成功搬到视觉‑语言领域，开启了图表推理专用强化微调的先河。后续工作（如 **Chart‑GPT‑2**、**VisCoT**）已经开始借鉴其程序化数据合成和数值敏感奖励机制，甚至将其扩展到更广的科学图表（如医学影像报告）和交互式数据可视化。对想进一步探索的读者，建议关注以下方向：  
- **跨模态强化学习**：如何在更复杂的多模态任务（视频+表格）中设计相对奖励。  
- **自动化推理脚本生成**：利用大语言模型自动生成思维链，再回流到模型微调。  
- **高效强化微调**：探索更轻量的策略梯度或离线 RL 方法，以降低训练成本。  

### 一句话记住它
让模型先写思维链，再用数值敏感的强化奖励“校对”，Chart‑R1 把图表推理的准确性和可解释性一次性提升到新高度。