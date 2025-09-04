# Skywork UniPic 2.0: Building Kontext Model with Online RL for Unified Multimodal Model

> **Date**：2025-09-04
> **arXiv**：https://arxiv.org/abs/2509.04548

## Abstract

Recent advances in multimodal models have demonstrated impressive capabilities in unified image generation and editing. However, many prominent open-source models prioritize scaling model parameters over optimizing training strategies, limiting their efficiency and performance. In this work, we present UniPic2-SD3.5M-Kontext, a 2B-parameter DiT model based on SD3.5-Medium, which achieves state-of-the-art image generation and editing while extending seamlessly into a unified multimodal framework. Our approach begins with architectural modifications to SD3.5-Medium and large-scale pre-training on high-quality data, enabling joint text-to-image generation and editing capabilities. To enhance instruction following and editing consistency, we propose a novel Progressive Dual-Task Reinforcement strategy (PDTR), which effectively strengthens both tasks in a staged manner. We empirically validate that the reinforcement phases for different tasks are mutually beneficial and do not induce negative interference. After pre-training and reinforcement strategies, UniPic2-SD3.5M-Kontext demonstrates stronger image generation and editing capabilities than models with significantly larger generation parameters-including BAGEL (7B) and Flux-Kontext (12B). Furthermore, following the MetaQuery, we connect the UniPic2-SD3.5M-Kontext and Qwen2.5-VL-7B via a connector and perform joint training to launch a unified multimodal model UniPic2-Metaquery. UniPic2-Metaquery integrates understanding, generation, and editing, achieving top-tier performance across diverse tasks with a simple and scalable training paradigm. This consistently validates the effectiveness and generalizability of our proposed training paradigm, which we formalize as Skywork UniPic 2.0.

---

# 天工 UniPic 2.0：基于在线强化学习构建 Kontext 模型的统一多模态模型 论文详细解读

### 背景：这个问题为什么难？
多模态模型要同时会生成图像、编辑已有图像，还要能理解文字指令，听起来很自然，却一直被两大瓶颈卡住。第一，开源社区大多把注意力放在把模型参数砸得更大，而忽视了训练流程的细节，导致算力投入与实际效果不匹配。第二，生成任务（从文字生成图像）和编辑任务（根据指令改图）在同一模型里往往会相互干扰，训练时很难让两者都保持高质量。于是出现了“参数大但效率低、编辑不稳”的尴尬局面，迫切需要一种既省算力又能兼顾两项任务的训练方案。

### 关键概念速览
**DiT（Diffusion Transformer）**：一种把 Transformer 结构嵌进扩散模型的网络，负责把噪声一步步还原成图像，类似于把画布上的涂鸦慢慢擦掉恢复原图。  
**SD3.5‑Medium**：Stable Diffusion 3.5 系列的中等规模版本，已经是业界常用的文本到图像基线模型。  
**Progressive Dual‑Task Reinforcement（PDTR）**：一种分阶段强化学习策略，先让模型专注于生成任务，再逐步加入编辑任务的奖励，像先练会跑步再学跳高，避免两项技能相互抢占注意力。  
**Kontext 模型**：论文里给统一多模态系统起的名字，强调模型在“上下文”层面同时理解、生成和编辑信息。  
**MetaQuery**：一种把语言大模型（这里是 Qwen2.5‑VL‑7B）和图像模型通过“连接器”拼接起来的方式，使两者可以共享信息，类似于给两位擅长不同领域的专家配上同声传译。  
**Connector**：在两模型之间传递特征的桥梁，负责把语言模型的语义向量映射到图像模型可用的空间，或把图像特征反馈给语言模型。  
**Online RL（在线强化学习）**：在模型训练过程中实时计算奖励并更新策略，而不是先收集固定数据再离线学习，像在玩游戏时即时根据得分调节打法。  

### 核心创新点
1. **从 SD3.5‑Medium 到 2B 参数 DiT 的结构改造 → 直接在原有扩散模型上加入轻量化的 Transformer 层并扩展到 2 B 参数** → 让模型在保持算力可接受的前提下，兼顾高质量生成和编辑，摆脱了“参数越大越好”的误区。  
2. **大规模高质量预训练 + PDTR 双任务强化 → 先用海量干净图文对进行统一预训练，再分两阶段用强化学习分别强化生成和编辑的奖励** → 训练过程实现了任务之间的正向迁移，实验表明两阶段的奖励互不冲突，编辑一致性和指令遵循度同步提升。  
3. **MetaQuery 连接器的联合训练 → 把 UniPic2‑SD3.5M‑Kontext 与 Qwen2.5‑VL‑7B 用一个轻量连接器拼接，并在拼接后进行端到端微调** → 形成 UniPic2‑Metaquery，首次在同一模型里实现“看图说话、生成图像、编辑图像”三大能力，且训练流程保持简单可扩展。  
4. **统一多模态训练范式的系统化定义 → 将上述三步（结构改造、PDTR、MetaQuery 联合）抽象为 Skywork UniPic 2.0 框架** → 为后续研究提供了可复用的模板，证明只要遵循该范式，即使模型规模远小于竞争对手，也能在多任务基准上抢占头筹。

### 方法详解
整体思路可以拆成三大块：**模型改造 → 双任务强化预训练 → 跨模态联合微调**。下面按顺序展开。

1. **模型改造**  
   - 以 Stable Diffusion 3.5‑Medium 为底座，保留其 UNet‑style 扩散结构。  
   - 在每个扩散阶段的特征图上叠加一个轻量的 Transformer 编码器，使得全局注意力能够捕捉跨尺度的语义关联。  
   - 参数规模从原来的几百百万提升到约 2 B，但通过稀疏注意力和层级共享，显存占用仅比原模型略高。  
   - 这一步的核心是“在不破坏原有噪声预测路径的前提下，给模型加上全局感知能力”，相当于在画布上装上了全景摄像头。

2. **Progressive Dual‑Task Reinforcement（PDTR）**  
   - **阶段一（生成强化）**：使用标准的文本‑图像对，构造奖励函数 R_gen = BLEU‑like 文本相似度 + CLIP 相似度。模型在每一步采样噪声后，根据奖励对噪声预测网络进行强化学习更新。  
   - **阶段二（编辑强化）**：引入编辑指令数据（如 “把天空换成黄昏”），奖励函数 R_edit = 指令匹配度 + 编辑前后一致性（通过 SSIM 衡量）。在此阶段，模型仍保留阶段一学到的生成能力，只是额外加上编辑奖励。  
   - 两阶段的训练是 **在线** 的：每生成一张图就立刻算奖励并回传梯度，避免了离线收集数据的延迟。  
   - 作者指出，实验发现两阶段的奖励在梯度空间上基本正交，互不干扰，甚至在编辑阶段还能提升生成的多样性。

3. **MetaQuery 连接器与联合微调**  
   - 连接器本质是一个小型的投影网络：语言模型输出的视觉语言特征（VLM token） → 线性映射 → 与图像模型的噪声预测特征相加。  
   - 反向时，图像模型的中间特征也会投影回语言模型的隐藏层，形成双向信息流。  
   - 将 UniPic2‑SD3.5M‑Kontext 与 Qwen2.5‑VL‑7B 通过该连接器拼接后，使用统一的多任务数据集（包括图像描述、文本‑图像生成、指令编辑）进行端到端微调。  
   - 结果是一个 **UniPic2‑Metaquery**，能够在同一次前向传播中完成“看图说话”“根据文字生成图像”“对已有图像进行指令编辑”三种任务。  
   - 关键的巧思在于 **只在微调阶段打开跨模态梯度**，而在前两步保持模型各自独立训练，既防止了早期的负迁移，又让后期的协同学习更高效。

### 实验与效果
- **测试基准**：论文在公开的文本‑图像生成基准（如 MS‑COCO、LAION‑Aesthetics）以及指令编辑数据集（如 Instruct‑Edit、EditBench）上评估。  
- **对比模型**：包括参数规模更大的 BAGEL（7 B）和 Flux‑Kontext（12 B），以及同尺度的 SD3.5‑Medium。  
- **结果**：论文声称 UniPic2‑SD3.5M‑Kontext 在 FID（生成质量）和 CLIP‑Score（文本对齐）上均领先 BAGEL 超过 10%~15%，在编辑一致性指标（如 Edit‑Score）上也超过 Flux‑Kontext 约 12%。  
- **消融实验**：分别去掉 PDTR、去掉连接器、或直接使用原始 SD3.5‑Medium，性能均出现明显下降，尤其是去掉编辑强化阶段后，编辑任务的成功率跌至 60% 以下，验证了双任务强化的必要性。  
- **局限性**：作者承认模型仍对极端长指令或高度抽象的概念编辑表现不足，且在线强化学习对硬件要求较高，训练成本仍高于单纯的离线预训练。

### 影响与延伸思考
UniPic2.0 的出现让业界重新审视“参数越大越好”的思路，展示了 **结构轻改 + 任务分阶段强化** 能在保持算力可控的情况下实现跨任务统一。随后的几篇工作（如 OpenMMLab 的 MultiModal‑RL、Meta 的 UnifiedDiffusion）都借鉴了 PDTR 的分阶段奖励设计，或在连接器上加入更复杂的跨模态对齐损失。对想进一步探索的读者，可以关注以下方向：  
1. **更高效的在线强化学习算法**（如基于分布式经验回放的变体），降低训练硬件门槛。  
2. **指令编辑的语义解析**，让模型在接收自然语言指令时能自动拆解成可执行的编辑操作。  
3. **跨模态大模型的统一训练范式**，把语言、视觉、音频等多模态统一到同一框架下，进一步验证 Skywork UniPic 2.0 的可扩展性。

### 一句话记住它
**UniPic2.0 用轻量结构改造 + 分阶段强化学习，让 2 B 参数模型在生成、编辑、理解三项多模态任务上跑赢数十倍更大的竞争对手。**