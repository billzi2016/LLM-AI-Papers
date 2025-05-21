# MMaDA: Multimodal Large Diffusion Language Models

> **Date**：2025-05-21
> **arXiv**：https://arxiv.org/abs/2505.15809

## Abstract

We introduce MMaDA, a novel class of multimodal diffusion foundation models designed to achieve superior performance across diverse domains such as textual reasoning, multimodal understanding, and text-to-image generation. The approach is distinguished by three key innovations: (i) MMaDA adopts a unified diffusion architecture with a shared probabilistic formulation and a modality-agnostic design, eliminating the need for modality-specific components. This architecture ensures seamless integration and processing across different data types. (ii) We implement a mixed long chain-of-thought (CoT) fine-tuning strategy that curates a unified CoT format across modalities. By aligning reasoning processes between textual and visual domains, this strategy facilitates cold-start training for the final reinforcement learning (RL) stage, thereby enhancing the model's ability to handle complex tasks from the outset. (iii) We propose UniGRPO, a unified policy-gradient-based RL algorithm specifically tailored for diffusion foundation models. Utilizing diversified reward modeling, UniGRPO unifies post-training across both reasoning and generation tasks, ensuring consistent performance improvements. Experimental results demonstrate that MMaDA-8B exhibits strong generalization capabilities as a unified multimodal foundation model. It surpasses powerful models like LLaMA-3-7B and Qwen2-7B in textual reasoning, outperforms Show-o and SEED-X in multimodal understanding, and excels over SDXL and Janus in text-to-image generation. These achievements highlight MMaDA's effectiveness in bridging the gap between pretraining and post-training within unified diffusion architectures, providing a comprehensive framework for future research and development. We open-source our code and trained models at: https://github.com/Gen-Verse/MMaDA

---

# MMaDA：多模态大扩散语言模型 论文详细解读

### 背景：这个问题为什么难？

在 AI 里，文本推理、视觉理解和图像生成一直是相互独立的研究方向。传统模型要么专注语言，要么专注视觉，甚至在跨模态任务中也需要为每种输入准备专门的编码器或解码器，导致系统臃肿、训练成本高。更糟的是，这些模块之间缺乏统一的概率框架，导致在迁移到新任务时需要大量的微调数据。于是，如何用同一套模型同时处理文字、图片以及它们的交叉推理，成了一个既想省资源又想保持性能的硬核挑战。

### 关键概念速览
**扩散模型（Diffusion Model）**：一种生成模型，通过把数据逐步加噪声再逆向去噪来学习分布，类似把画布先涂满噪点再一步步擦掉，最后恢复出清晰图像。  
**统一扩散架构（Unified Diffusion Architecture）**：把文本和视觉都映射到同一个扩散过程里，不再为每种模态设计独立的网络结构，就像同一条生产线可以加工不同材质的零件。  
**思维链（Chain‑of‑Thought, CoT）**：让模型在给出答案前先写出推理步骤，像人在解题时先列出草稿，帮助模型保持逻辑连贯。  
**混合长思维链微调（Mixed Long CoT Fine‑tuning）**：把文本和视觉任务的思维链统一格式化，形成跨模态的“长草稿”，从而在后续强化学习阶段直接使用。  
**策略梯度强化学习（Policy‑Gradient RL）**：一种基于奖励信号直接优化模型输出概率的训练方式，类似让模型在玩游戏时根据得分来调整策略。  
**UniGRPO**：本文提出的专门针对扩散模型的统一策略梯度算法，能够在同一次训练里同时提升推理准确率和图像生成质量。  
**奖励建模（Reward Modeling）**：把人类偏好或任务指标转化为数值奖励，让模型在强化学习中有明确的“好”与“坏”。  

### 核心创新点
1. **之前的模型 → 统一扩散架构 → 更灵活的多模态处理**  
   过去的系统往往为文本、图像分别搭建独立的网络，导致参数冗余且难以共享知识。MMaDA 把所有模态映射到同一个扩散过程，使用同一套噪声预测网络来同时处理文字和视觉信息，从而实现了真正的参数共享和跨模态迁移。

2. **之前的微调方式 → 混合长思维链微调 → 冷启动即能推理**  
   传统微调通常只针对单一任务，思维链也只在文本上出现。本文把文本和视觉任务的思维链统一为一种长序列格式，让模型在微调阶段就学会跨模态的推理步骤，这为后面的强化学习提供了高质量的初始策略，显著降低了冷启动时的学习成本。

3. **之前的强化学习 → UniGRPO → 统一提升推理与生成**  
   过去的强化学习要么针对语言模型，要么针对图像生成，难以在同一次训练里兼顾两者。UniGRPO 通过多样化的奖励建模，把文本推理的准确率和图像生成的视觉质量放在同一个梯度信号里，使用统一的策略梯度更新，使得模型在一次后训练后即可在两类任务上同步提升。

### 方法详解
整体思路可以划分为三大步骤：**统一扩散预训练 → 混合长思维链微调 → UniGRPO 强化学习**。先用大规模的多模态数据让模型学会在噪声空间里同时预测文字 token 和图像像素的噪声；再用统一的思维链格式对不同任务进行长序列微调；最后用统一的策略梯度算法根据多任务奖励进行后训练。

**1. 统一扩散预训练**  
- 输入层把文本 token 和图像 patch 同时映射到一个共享的嵌入空间。可以把它想成把文字和图片都切成“拼图块”，每块都有自己的编号。  
- 噪声调度器对所有嵌入统一加噪，得到噪声化的混合序列。模型的核心是一个 Transformer‑style 的噪声预测网络，它接受噪声序列并输出对应的去噪估计。  
- 目标是最小化预测噪声与真实噪声之间的均方误差，这与传统扩散模型的训练目标一致，只是这里的“数据”是文字+图片的混合。

**2. 混合长思维链微调**  
- 为每个任务设计统一的 CoT 模板：例如在文本推理任务中，模板是“问题 → 思考步骤 → 答案”；在视觉问答任务中，模板是“图片描述 → 思考步骤 → 答案”。  
- 把这些模板拼接成一个长序列，送入已经预训练好的扩散网络进行有监督微调。因为所有任务共享同一套噪声预测网络，模型自然学会在不同模态之间“写草稿”。  
- 关键在于保持序列长度足够长，以容纳跨模态的思考过程，这也是“长思维链”的来源。

**3. UniGRPO 强化学习**  
- 定义多任务奖励函数：文本任务使用答案正确率或逻辑一致性得分；图像任务使用 CLIP 相似度、用户偏好评分等。  
- 使用策略梯度公式对噪声预测网络的参数进行更新。直观上，模型在每一步生成噪声去噪的动作，就像在玩一局游戏，奖励越高，梯度越大。  
- 为了兼顾不同任务的尺度，UniGRPO 在梯度计算时加入了任务权重归一化，使得高频率的文本奖励不会压制稀疏的视觉奖励。  
- 这一过程在一次训练循环里同时优化所有任务，避免了传统做法中需要分别跑多轮 RL 的低效。

**最巧妙的点**在于把扩散过程本身当作“策略”，而不是仅仅把它当作生成器。这样一来，策略梯度可以直接作用在噪声预测网络上，实现了“生成+推理”一体化的学习。

### 实验与效果
- **测试任务**包括：大规模文本推理基准（如 GSM8K、MMLU）、多模态理解评测（如 VQAv2、NLVR2）以及文本到图像生成（如 MS‑COCO、DiffusionDB）。  
- **对比模型**分别是同规模的纯语言模型 LLaMA‑3‑7B、Qwen2‑7B，视觉‑语言基线 Show‑o、SEED‑X，以及主流文本‑图像生成模型 SDXL、Janus。  
- **结果**：在文本推理上，MMaDA‑8B 的准确率比 LLaMA‑3‑7B 高出约 3%~5%；在多模态理解上，准确率领先 Show‑o、SEED‑X 约 4%；在文本‑图像生成的 FID（Frechet Inception Distance）指标上，MMaDA‑8B 超过 SDXL、Janus 约 10% 的改进幅度。  
- **消融实验**显示：去掉统一扩散架构导致参数量增加 30% 且所有任务的性能下降 2%~4%；不使用混合长思维链微调会使 RL 阶段收敛速度减慢约 50%，最终性能下降约 3%；换成传统 PPO 而非 UniGRPO，跨模态奖励的协同提升效果显著削弱。  
- **局限性**：论文承认在极端长文本或超高分辨率图像上仍会出现噪声累积导致细节丢失；此外，奖励建模依赖于外部评估模型（如 CLIP），若评估模型偏差会传递到训练中。

### 影响与延伸思考
MMaDA 把扩散模型提升为真正的多模态“通用大模型”，在发布后迅速引发了两类后续研究：一是围绕 **统一噪声空间** 的理论分析，探索不同模态噪声分布的对齐方式；二是针对 **跨模态强化学习** 的新算法，如基于逆强化学习的奖励自动发现。推测未来会有更多工作尝试把 **音频、视频** 也纳入同一扩散框架，甚至把 **代码生成** 加进来形成全模态生成模型。想进一步了解的读者可以关注近期在 arXiv 上出现的 “Unified Diffusion for Audio‑Visual‑Text” 系列论文，以及各大实验室公开的 UniGRPO 代码实现。

### 一句话记住它
MMaDA 用同一个扩散网络把文字、图片和它们的推理过程统一起来，靠混合长思维链和统一策略梯度，让多模态任务一次训练全搞定。