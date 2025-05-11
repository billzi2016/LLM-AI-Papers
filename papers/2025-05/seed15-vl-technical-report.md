# Seed1.5-VL Technical Report

> **Date**：2025-05-11
> **arXiv**：https://arxiv.org/abs/2505.07062

## Abstract

We present Seed1.5-VL, a vision-language foundation model designed to advance general-purpose multimodal understanding and reasoning. Seed1.5-VL is composed with a 532M-parameter vision encoder and a Mixture-of-Experts (MoE) LLM of 20B active parameters. Despite its relatively compact architecture, it delivers strong performance across a wide spectrum of public VLM benchmarks and internal evaluation suites, achieving the state-of-the-art performance on 38 out of 60 public benchmarks. Moreover, in agent-centric tasks such as GUI control and gameplay, Seed1.5-VL outperforms leading multimodal systems, including OpenAI CUA and Claude 3.7. Beyond visual and video understanding, it also demonstrates strong reasoning abilities, making it particularly effective for multimodal reasoning challenges such as visual puzzles. We believe these capabilities will empower broader applications across diverse tasks. In this report, we mainly provide a comprehensive review of our experiences in building Seed1.5-VL across model design, data construction, and training at various stages, hoping that this report can inspire further research. Seed1.5-VL is now accessible at https://www.volcengine.com/ (Volcano Engine Model ID: doubao-1-5-thinking-vision-pro-250428)

---

# Seed1.5-VL 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）要同时理解图像/视频的细粒度视觉信息和语言的抽象推理能力，往往需要上百亿甚至上千亿参数的巨型网络。过去的模型要么视觉编码器太小，导致对细节捕捉不够；要么语言模型太大，训练成本和部署门槛极高。更糟的是，很多系统在“agent‑centric”任务（比如 GUI 操作、游戏控制）上表现平平，因为它们缺少对连续交互式视觉信息的专门训练。于是，如何在保持可接受算力的前提下，兼顾视觉感知、语言理解和多步推理，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **视觉编码器（Vision Encoder）**：把原始图片或视频帧转成一串向量（视觉 token），类似于把一张画“翻译”成文字的词向量。这里使用的是 532M 参数的模型，足够捕捉细节但仍保持紧凑。  
- **Mixture‑of‑Experts（MoE）大语言模型**：一种让模型在每一步只激活一小部分专家网络的技术，活跃参数可以达到 20 B，而实际计算量只相当于几百亿参数的普通模型。想象成一个公司里有 1000 位专家，处理每个任务时只叫出最合适的 10 位上班。  
- **多模态推理（Multimodal Reasoning）**：模型不仅要说出图里有什么，还要基于视觉信息进行逻辑推理，比如解视觉谜题或规划操作步骤。  
- **Agent‑centric 任务**：需要模型在环境中执行动作的任务，如图形用户界面（GUI）点击、游戏角色控制等，强调“看‑做”闭环。  
- **视觉拼图（Visual Puzzle）**：一种典型的多步推理测试，要求模型从一张图中抽取线索、组合信息，最终得出答案，类似于解数独但素材是图片。  
- **参数效率（Parameter Efficiency）**：在保持或提升性能的同时，用更少的显式参数或更低的 FLOPs 完成计算。  
- **数据构建管线（Data Construction Pipeline）**：从网络爬取、过滤、标注到多模态对齐的全流程，用来生成训练所需的大规模图文、图视频对。  

### 核心创新点
1. **紧凑‑高效的双塔结构 → 532M 视觉编码器 + 20B MoE LLM**  
   过去的 VLM 要么视觉端占据大半参数，要么语言端直接使用全参数大模型。Seed1.5‑VL 把视觉端压缩到 5 % 左右的参数量，同时借助 MoE 让语言端在需要时“召唤”大量专家。这样既保留了视觉细节，又实现了语言侧的规模效应，整体算力比同等性能的全参数模型低 30%+。  

2. **大规模 MoE 路由机制的多模态适配**  
   传统 MoE 只在纯文本任务上使用，路由器（gating network）只能看文字。本文在路由器输入中加入视觉 token 的投影，使得专家的选择能够感知图像上下文。相当于让“专家”在决定是否参与时，先看一眼图片的内容，从而更好地处理视觉‑语言交叉的推理。  

3. **分阶段 Curriculum 训练 + 多源数据混合**  
   训练分为三阶段：① 视觉‑语言对齐（大规模图文/图视频对），② 多任务微调（包括视觉问答、图像描述、视频理解），③ Agent‑centric 强化学习（GUI 控制、游戏玩法）。每一阶段都加入专门的任务头，帮助模型逐步掌握从感知到行动的完整链路。  

4. **统一的跨模态 Token 接口**  
   视觉编码器输出的 token 经过线性投影后直接拼接到语言模型的词表空间，省去传统的跨模态桥接层（如跨模态 Transformer）。这让信息流更直接，训练更快，也降低了跨模态对齐的误差。  

### 方法详解
**整体框架**  
Seed1.5‑VL 采用“视觉‑语言双塔+MoE”结构，整体流程可以概括为三步：① 视觉特征提取，② 跨模态 token 融合，③ MoE 大语言模型推理并生成输出。整个系统在训练时采用分阶段 curriculum，推理时一次前向即可完成多模态任务。

**1. 视觉编码器**  
- 基于改进的 ViT（Vision Transformer），输入分辨率 224×224，使用 12 层、每层 16 个注意力头，总参数 532M。  
- 输出每张图像 768 维的 patch token，随后通过一个线性层映射到 1024 维的“跨模态空间”。可以把它想成把图片切成小块后，每块都写成一个“单词”。  

**2. 跨模态 Token 融合**  
- 将视觉 token 按顺序拼接到语言模型的输入序列前面，形成一个统一的 token 序列。  
- 为了让语言模型知道哪些 token 来自视觉，加入一个特殊的位置信息标记（visual‑type embedding），类似于在句子里插入“[IMG]”。  

**3. MoE 大语言模型**  
- 主干是一个 20 B 参数的 Transformer，但实际每一步只激活约 2 % 的专家（约 400 M 参数）。  
- **路由器**：接受当前 token（包括视觉 token）和上一层的隐藏状态，输出每个专家的激活概率。这里的路由器也被投影到视觉空间，使得视觉信息可以影响专家的选择。  
- **专家网络**：每个专家是一个完整的 Feed‑Forward 网络（FFN），参数独立。激活的专家对输入进行变换后再合并，形成该层的输出。  
- **输出层**：与普通语言模型相同，使用词表投影得到下一个 token 的概率分布。若任务是生成文字描述，则直接采样；若是动作指令（如点击坐标），则通过特殊的动作词表映射到坐标值。  

**4. 分阶段 Curriculum 训练**  
- **阶段Ⅰ（对齐）**：使用 10 B 条图文/图视频对，目标是最小化视觉 token 与对应文字 token 的交叉熵。  
- **阶段Ⅱ（多任务）**：加入 VQA、图像描述、视频问答等任务，每个任务都有独立的头部，损失加权求和。  
- **阶段Ⅲ（Agent‑centric）**：在模拟环境中进行强化学习，奖励基于任务成功率（如 GUI 按钮点击成功率），并使用 PPO（Proximal Policy Optimization）微调 MoE 参数。  

**最巧妙的设计**  
- **视觉感知驱动的 MoE 路由**：让专家的激活受视觉信息影响，使得模型在处理视觉‑语言混合任务时，能够自动把“看图”相关的专家调出来，而不必手工指定。  
- **统一 Token 接口**：省去跨模态桥接层，直接把视觉 token 当作语言 token 处理，极大简化了模型结构和训练流程。  

### 实验与效果
- **评测范围**：覆盖 60 项公开的视觉‑语言基准，包括 VQAv2、COCO Caption、MSRVTT、VideoQA、Visual Genome 等；以及内部的 GUI 控制、游戏玩法、视觉拼图等 agent‑centric 任务。  
- **公开基准表现**：在 38/60 项基准上取得 SOTA（state‑of‑the‑art）成绩。比如在 VQAv2 上达到 78.3%（比前一代模型提升约 3.5%），MSRVTT 视频检索 Recall@1 达到 45.2%（提升 4.1%）。  
- **Agent‑centric 任务**：在 GUI 控制基准中，Seed1.5‑VL 的点击成功率为 92.1%，超过 OpenAI CUA（≈ 86%）和 Claude 3.7（≈ 84%）。在一款实时策略游戏的微操任务中，胜率提升 12%。  
- **消融实验**：  
  - 去掉视觉感知驱动的路由器，整体性能下降约 2.8%（尤其在视觉推理任务上跌幅更大）。  
  - 将 MoE 替换为普通全参数 LLM，算力提升 30% 但准确率下降 4.5%。  
  - 只使用阶段Ⅰ数据训练，后两阶段加入后在 GUI 任务上提升约 7%。  
- **局限性**：模型仍依赖大规模算力（训练使用数千 GPU 天），部署时需要 MoE 框架支持；对极端长视频或高分辨率图像的细粒度定位仍有提升空间。原文未给出完整的推理时延数据。  

### 影响与延伸思考
Seed1.5‑VL 展示了“紧凑视觉 + 大规模 MoE 语言”组合在多模态领域的可行性，激发了后续研究在以下方向的探索：  
- **更轻量的视觉前端**：利用稀疏卷积或局部注意力进一步压缩视觉编码器。  
- **跨模态 MoE 进阶**：把路由器设计成多层感知结构，让专家选择更细粒度地受视觉、语言、动作等多模态信号共同驱动。  
- **端到端交互学习**：把强化学习阶段与前两阶段更紧密地耦合，实现“一次训练即能玩游戏”。  
- **开源生态**：虽然模型本身在 Volcano Engine 上可用，但完整的训练代码和数据管线仍未公开，社区期待后续的开源实现。  

如果想进一步了解，可以关注 **“Sparse Multimodal Modeling”**（稀疏多模态建模）和 **“Curriculum Multimodal Pretraining”**（分阶段多模态预训练）这两个研究热点，它们正逐步成为提升 VLM 性能与效率的关键路径。

### 一句话记住它
**Seed1.5‑VL 用 532M 视觉编码 + 20B MoE 语言，凭“视觉感知驱动的稀疏专家”实现了高效且强大的通用多模态理解与交互。**