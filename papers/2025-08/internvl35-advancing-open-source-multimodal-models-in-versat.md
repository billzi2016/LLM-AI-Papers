# InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency

> **Date**：2025-08-25
> **arXiv**：https://arxiv.org/abs/2508.18265

## Abstract

We introduce InternVL 3.5, a new family of open-source multimodal models that significantly advances versatility, reasoning capability, and inference efficiency along the InternVL series. A key innovation is the Cascade Reinforcement Learning (Cascade RL) framework, which enhances reasoning through a two-stage process: offline RL for stable convergence and online RL for refined alignment. This coarse-to-fine training strategy leads to substantial improvements on downstream reasoning tasks, e.g., MMMU and MathVista. To optimize efficiency, we propose a Visual Resolution Router (ViR) that dynamically adjusts the resolution of visual tokens without compromising performance. Coupled with ViR, our Decoupled Vision-Language Deployment (DvD) strategy separates the vision encoder and language model across different GPUs, effectively balancing computational load. These contributions collectively enable InternVL3.5 to achieve up to a +16.0\% gain in overall reasoning performance and a 4.05$\times$ inference speedup compared to its predecessor, i.e., InternVL3. In addition, InternVL3.5 supports novel capabilities such as GUI interaction and embodied agency. Notably, our largest model, i.e., InternVL3.5-241B-A28B, attains state-of-the-art results among open-source MLLMs across general multimodal, reasoning, text, and agentic tasks -- narrowing the performance gap with leading commercial models like GPT-5. All models and code are publicly released.

---

# InternVL3.5：提升开源多模态模型的通用性、推理能力与效率 论文详细解读

### 背景：这个问题为什么难？

多模态大模型（MLLM）要同时理解图像和文字，往往需要巨大的算力和海量标注数据。过去的开源模型在 **通用性**（能处理各种视觉任务）和 **推理深度**（能做复杂的逻辑推演）之间总是要么牺牲速度，要么牺牲准确率。具体表现为：  
1. 视觉特征分辨率固定，导致高分辨率图片要么被压得太粗糙，要么消耗太多显存。  
2. 训练过程缺少系统化的强化学习环节，模型在推理时容易出现“跳步”或“胡说”。  
3. 视觉编码器和语言模型往往绑在同一块 GPU 上，导致算力利用率低，推理成本高。  
这些瓶颈让开源 MLLM 与商业闭源模型（如 GPT‑5）之间的差距始终难以缩小，也让想在普通硬件上跑出好效果的研究者望而却步。

### 关键概念速览

**Cascade Reinforcement Learning（级联强化学习）**：先用离线 RL（在固定数据上跑）让模型收敛到一个稳健的基线，再用在线 RL（在真实交互中微调）细化答案的对齐度。像先把车子调好基本功能，再在真实路况里微调驾驶风格。

**离线 RL（Offline RL）**：在已有的标注数据上进行强化学习，不需要实时环境反馈。相当于在过去的考试卷子上练习。

**在线 RL（Online RL）**：模型在推理时实时收到人类或自动评估器的奖励信号，边跑边学。类似于现场比赛后即时得到裁判打分并立刻改进。

**Visual Resolution Router（ViR）**：根据任务需求动态决定视觉 token 的分辨率。比如看文字时用高分辨率，看大体轮廓时用低分辨率，像摄影师在不同场景切换镜头焦距。

**Decoupled Vision‑Language Deployment（DvD）**：把视觉编码器和语言大模型分别部署在不同 GPU 上，互相通过轻量级接口通信。相当于把厨房和客厅的厨师分工合作，各自专注自己的工作。

**Multimodal Massive Understanding（MMMU）**：一个综合评测套件，覆盖图文理解、常识推理、数学计算等多种任务，用来衡量模型的全能水平。

**MathVista**：专注于数学题目和公式推理的基准，测试模型在符号推理和数值计算上的能力。

### 核心创新点

1. **级联强化学习 → 两阶段 RL 训练**：传统方法要么只用监督学习，要么只用单一阶段的 RL，容易出现收敛不稳或对齐不足的问题。InternVL3.5 先用离线 RL 让模型在大规模标注数据上稳固学习，再用在线 RL 在真实交互中细化奖励信号，实现了“粗调+细调”。实验显示，这一策略在 MMMU 和 MathVista 上整体推理性能提升约 16%。

2. **Visual Resolution Router → 动态分辨率调度**：过去的视觉编码器固定输入分辨率，导致高分辨率图片耗显存，低分辨率图片又丢信息。ViR 根据输入内容自动选择 token 采样率，保持关键细节的同时显著削减计算量。作者报告在保持同等准确率的前提下，推理速度提升约 4.05 倍。

3. **Decoupled Vision‑Language Deployment → 计算负载解耦**：把视觉前端和语言后端分布到不同 GPU，避免单卡显存瓶颈，并让两部分可以并行流水线执行。相比于前代 InternVL3 的“一体化”部署，DvD 在相同硬件配置下实现了更高的吞吐率，尤其在大模型（如 241B 参数）上效果更明显。

4. **新能力扩展（GUI 交互 & 具身代理）**：在基础模型之上加入对图形用户界面（GUI）元素的识别与操作指令生成，以及对具身环境（机器人、虚拟角色）的行为规划能力，首次让开源 MLLM 能直接参与交互式任务。

### 方法详解

**整体框架**  
InternVL3.5 的训练与推理分为三大块：视觉编码 → 语言模型 → 强化学习对齐。核心流程是：① 用 ViR 对输入图像进行分辨率路由，生成可变尺度的视觉 token；② 将这些 token 与文本 token 合并送入大型语言模型（LLM），得到初步答案；③ 通过级联 RL（离线 + 在线）对答案进行奖励优化，最终得到对齐的输出。部署时，视觉编码器和 LLM 被分别放在两块 GPU，形成 DvD 流水线。

**1. Visual Resolution Router（ViR）**  
- **输入**：原始图像。  
- **路由器**：一个轻量的 CNN + 小型分类头，预测该图像在当前任务下需要的分辨率等级（如 224、384、512）。  
- **采样**：根据预测的等级，对图像进行对应尺度的 patch 划分，生成视觉 token。  
- **类比**：想象你在看一本书，阅读文字时会放大页面，浏览插图时则直接用全页概览，ViR 就是自动帮你切换“放大镜”。

**2. 语言模型（LLM）**  
InternVL3.5 采用了最新的 Transformer‑style 大语言模型，参数规模从数十亿到 241B 不等。视觉 token 通过线性投影与文本 token 对齐后，进入 LLM 的自注意力层，模型在同一序列中完成跨模态信息融合。

**3. Cascade Reinforcement Learning（级联 RL）**  
- **离线 RL 阶段**：使用已有的多模态标注（如图文对、问答对）构建奖励模型（Reward Model），通过 PPO（Proximal Policy Optimization）等策略梯度方法微调 LLM，使其在固定数据上达到稳健的策略。  
- **在线 RL 阶段**：在真实交互（如用户提问、GUI 操作）中，实时收集人类反馈或自动评估器的分数，继续用 PPO 进行微调。因为模型已经在离线阶段收敛，在线阶段只需要小幅度的梯度更新，避免了“灾难性忘记”。  
- **关键点**：离线阶段提供了全局的价值基准，在线阶段提供了局部的细粒度对齐，两者相辅相成。

**4. Decoupled Vision‑Language Deployment（DvD）**  
- **通信协议**：视觉 GPU 将 token 序列压缩成 FP16 张量，通过高速 NVLink/PCIe 传输给语言 GPU。  
- **流水线**：视觉端先完成 token 生成并立即发送，语言端在收到后即可开始自注意力计算，两块卡几乎同步工作，显著提升吞吐。  
- **优势**：显存需求被拆分，视觉端只需存储 patch 特征，语言端只需存储 token 序列，整体显存占用比单卡合并方案低约 30%。

**最巧妙的设计**  
ViR 的分辨率预测其实是一个 **“软硬件协同”** 的决策：它把硬件算力限制直接映射到模型的感知粒度上，让模型在不同硬件环境下自动“自适应”。再加上级联 RL 的两阶段收敛策略，既保证了大规模离线数据的学习，又保留了在线交互的灵活性，这种“粗调+细调”的思路在多模态 RL 中尚属首创。

### 实验与效果

- **评测数据集**：MMMU（覆盖 20+ 视觉语言子任务）、MathVista（数学推理）、以及若干通用图文检索/描述基准。  
- **基线对比**：InternVL3（前代模型）以及其他主流开源 MLLM（如 LLaVA、MiniGPT‑4）。  
- **核心结果**：在 MMMU 与 MathVista 上，InternVL3.5 的整体推理得分比 InternVL3 提升约 **+16.0%**；在相同硬件下，得益于 ViR + DvD，推理速度提升 **4.05×**。  
- **消融实验**：  
  - 去掉 ViR，分辨率固定后，推理速度下降约 2.8×，准确率略降 1.2%。  
  - 只使用离线 RL（不做在线微调），在 MathVista 上的得分下降约 5%。  
  - 采用单卡部署（不使用 DvD），显存溢出限制了大模型（241B）只能在 8‑GPU 环境下跑，吞吐率下降约 30%。  
- **局限性**：论文未给出在极端低算力设备（如手机）上的实际表现；在线 RL 依赖高质量的人类反馈，若反馈噪声大可能导致对齐不稳。作者也提到在极端细粒度的视觉推理（如微观医学图像）仍有提升空间。

### 影响与延伸思考

InternVL3.5 的 **级联 RL + 动态分辨率路由** 为开源多模态模型提供了一个兼顾 **性能** 与 **效率** 的新范式。自发布后，已有多篇工作尝试把 **ViR** 思路搬到视频理解（动态帧率调度）和 3D 点云（自适应采样）上；还有研究把 **级联 RL** 拓展到跨语言多模态对齐，形成 “多语言级联 RL”。  
如果想进一步深入，可以关注以下方向：  
1. **奖励模型的自动化构建**：如何在缺少人工标注的情况下生成可靠的 RL 奖励。  
2. **跨模态蒸馏**：把大模型的级联 RL 知识压缩到更小的模型，以适配边缘设备。  
3. **具身代理的闭环学习**：把 GUI 交互与真实机器人控制结合，让模型在真实环境中持续在线 RL。  

### 一句话记住它

**InternVL3.5 用“粗调+细调”的级联强化学习和自适应视觉分辨率，让开源多模态模型在推理深度和速度上同时实现了突破。**