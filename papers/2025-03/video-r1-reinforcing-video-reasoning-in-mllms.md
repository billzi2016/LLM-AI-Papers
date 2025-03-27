# Video-R1: Reinforcing Video Reasoning in MLLMs

> **Date**：2025-03-27
> **arXiv**：https://arxiv.org/abs/2503.21776

## Abstract

Inspired by DeepSeek-R1's success in eliciting reasoning abilities through rule-based reinforcement learning (RL), we introduce Video-R1 as the first attempt to systematically explore the R1 paradigm for incentivizing video reasoning within multimodal large language models (MLLMs). However, directly applying RL training with the GRPO algorithm to video reasoning presents two primary challenges: (i) a lack of temporal modeling for video reasoning, and (ii) the scarcity of high-quality video-reasoning data. To address these issues, we first propose the T-GRPO algorithm, which encourages models to utilize temporal information in videos for reasoning. Additionally, instead of relying solely on video data, we incorporate high-quality image-reasoning data into the training process. We have constructed two datasets: Video-R1-CoT-165k for SFT cold start and Video-R1-260k for RL training, both comprising image and video data. Experimental results demonstrate that Video-R1 achieves significant improvements on video reasoning benchmarks such as VideoMMMU and VSI-Bench, as well as on general video benchmarks including MVBench and TempCompass, etc. Notably, Video-R1-7B attains a 37.1% accuracy on video spatial reasoning benchmark VSI-bench, surpassing the commercial proprietary model GPT-4o. All code, models, and data are released in: https://github.com/tulerfeng/Video-R1.

---

# Video‑R1：在多模态大语言模型中强化视频推理 论文详细解读

### 背景：这个问题为什么难？

视频是时间序列的视觉数据，光靠单帧图像的特征很难捕捉动作、因果和跨帧关系。早期的多模态大语言模型（MLLM）大多把视频当成一堆独立图片来处理，导致在需要“先看后推理”的任务上表现乏力。另一方面，强化学习（RL）已经在文本和图像推理上展示了提升模型思考深度的潜力，但直接搬到视频上会碰到两大壁垒：一是缺少专门的时序建模机制，二是高质量的视频推理标注极其稀缺。正是这些根本性短板，让视频推理成为亟待突破的难点。

### 关键概念速览
- **MLLM（多模态大语言模型）**：能够同时理解文字、图片甚至视频等多种输入，并用自然语言输出答案的模型。把它想象成一个会说话的全能助理。
- **R1范式**：一种通过规则驱动的强化学习让模型主动产生推理链的训练方式。类似于给模型设定“思考奖励”，鼓励它把思考过程写出来。
- **GRPO（Generalized Reward‑Prompted Optimization）**：在R1中使用的核心RL算法，模型在每一步生成的文本会根据预设规则得到奖励或惩罚，从而学习更有条理的推理。
- **T‑GRPO（Temporal‑GRPO）**：本文针对视频专门改进的GRPO版本，奖励模型利用时间信息进行推理。可以把它看作在原有“思考奖励”上加了一个“时间感知”滤镜。
- **CoT（Chain‑of‑Thought，思维链）**：让模型在给出最终答案前先写出逐步推理过程，像解数学题时先列步骤一样。
- **SFT（Supervised Fine‑Tuning）**：在大规模标注数据上进行的监督微调，用来让模型先学会基本的输入‑输出映射。
- **RL‑training（强化学习微调）**：在SFT之后，用奖励信号进一步引导模型的行为，使其在特定任务上更具策略性。

### 核心创新点
1. **时序感知的奖励机制**  
   - 之前的GRPO只关注文本的逻辑连贯性，对视频帧之间的因果关系没有专门奖励。  
   - 本文在奖励函数中加入了“时间一致性”项：模型若在推理中引用了前后帧的关系（如“先出现的物体随后被遮挡”），就会得到额外奖励。  
   - 这样模型被迫学会在思考链里显式使用时间线索，显著提升了对动作顺序和因果推断的准确率。

2. **跨模态数据混合训练**  
   - 直接收集大规模视频推理标注成本高，现有数据稀缺。  
   - 作者把高质量的图像推理数据也加入训练，利用图像‑CoT 数据帮助模型掌握基本推理技巧，再通过少量视频数据让模型迁移到时序场景。  
   - 这种“图像+视频”混合策略在保持推理能力的同时，缓解了数据匮乏的问题。

3. **两阶段数据构建**  
   - 为了让模型从零开始学习，构建了 Video‑R1‑CoT‑165k（仅用于 SFT）和 Video‑R1‑260k（用于 RL）。  
   - 前者提供了大量图像和少量视频的思维链示例，帮助模型建立语言‑视觉对应；后者在此基础上加入了时序奖励，完成强化学习微调。  
   - 这种分层数据设计让模型在每一步都能得到合适的学习信号，提升了整体收敛速度和最终表现。

### 方法详解
整体思路可以拆成三大块：**数据准备 → SFT 预热 → T‑GRPO 强化**。

1. **数据准备**  
   - **CoT 冷启动数据**：从公开的图像推理数据集抽取 150k 条高质量思维链，再加入 15k 条手工标注的视频推理示例，形成 Video‑R1‑CoT‑165k。每条样本的结构是“视频/图片 → 问题 → 思维链 → 答案”。  
   - **RL 训练数据**：在冷启动数据的基础上，进一步采集 260k 条包含时间标签的样本，标注中明确指出哪些帧提供了关键线索（如“第 3 帧出现的球被第 5 帧的手抓住”），为后续奖励函数提供依据。

2. **SFT 预热**  
   - 使用标准的跨模态对齐技术（如 Q‑Former）把视频帧特征映射到语言空间。  
   - 在 Video‑R1‑CoT‑165k 上进行监督微调，让模型学会把视觉输入转化为连贯的思维链。此阶段的目标是让模型能够“写草稿”，但不涉及时序奖励。

3. **T‑GRPO 强化学习**  
   - **奖励函数设计**：在原有的逻辑连贯奖励（如答案正确、思维链完整）之外，引入两类时序奖励：  
     a) **帧引用奖励**：若模型在思维链中提到具体帧编号或时间戳，并且该引用与标注的关键帧对应，就加分。  
     b) **因果一致奖励**：模型若正确描述了前后事件的因果关系（如“因为 A 发生，才导致 B”），也会得到奖励。  
   - **T‑GRPO 迭代**：模型在每一步生成思维链的 token 时，依据上述奖励进行策略梯度更新。因为奖励是基于规则而非人工评分，训练过程可以大规模自动化。  
   - **时间感知的特征融合**：在语言模型的自注意力层加入时间编码（类似 Transformer 中的相对位置编码），让模型在生成每个 token 时能够“看到”当前帧的时序上下文。

4. **最巧妙的设计**  
   - 将 **图像 CoT** 与 **视频时序奖励** 结合，使得模型先在“单帧推理”上打好基础，再在“跨帧推理”上加分。这个两步走的思路避免了直接在稀缺视频数据上训练导致的过拟合。  
   - 奖励函数完全基于 **规则**（帧编号匹配、因果词汇检测），不需要人工打分，极大降低了 RL 训练的成本。

### 实验与效果
- **评测数据集**：VideoMMMU、VSI‑Bench（空间推理）、MVBench、TempCompass 等，覆盖了视频问答、空间定位、时间顺序等多维度任务。  
- **主要结果**：在 VSI‑Bench 上，Video‑R1‑7B 达到 37.1% 的准确率，超过了商业闭源模型 GPT‑4o（官方未公布具体数值，但作者称其被超越）。在 VideoMMMU 上提升约 6%‑8% 的整体得分，TempCompass 上的时间推理准确率提升约 5%。  
- **Baseline 对比**：与未使用时序奖励的普通 GRPO、以及仅用图像 CoT 进行微调的模型相比，T‑GRPO 版在所有视频基准上都有两位数的提升。  
- **消融实验**：作者分别去掉（1）时间奖励、（2）图像 CoT 数据、（3）时间编码，发现去掉时间奖励导致 VSI‑Bench 准确率下降约 9%，去掉图像 CoT 数据导致整体收敛慢且最终得分下降约 4%，去掉时间编码则在因果推理任务上跌幅最高（约 7%）。这些实验表明三个设计缺一不可。  
- **局限性**：论文承认仍然依赖于规则式奖励，难以捕捉更细粒度的语义时序关系；此外，模型在长视频（超过 30 秒）上的推理仍显不足，可能需要更强的记忆机制。

### 影响与延伸思考
Video‑R1 把强化学习的“思考奖励”成功搬进了视频领域，打开了 **时序强化学习** 的新方向。后续工作已经开始探索更柔性的奖励信号（如利用预训练的时间关系检测器自动生成奖励），以及把 **记忆网络** 与 T‑GRPO 结合，以处理更长的影片。对想进一步研究的读者，可以关注以下两个方向：  
1. **自动化时序奖励生成**：利用大模型自行标注因果链，减少人工规则的依赖。  
2. **跨模态长期记忆**：在 Transformer 基础上加入可检索的时序记忆库，让模型在数分钟的视频中保持一致的推理。  

### 一句话记住它
**Video‑R1 用规则化的时序奖励让多模态大语言模型学会在视频里“写思维链”，把视频推理从单帧猜测提升到真正的时间因果理解。**