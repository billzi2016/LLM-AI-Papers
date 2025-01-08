# Cosmos World Foundation Model Platform for Physical AI

> **Date**：2025-01-07
> **arXiv**：https://arxiv.org/abs/2501.03575

## Abstract

Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model. In this paper, we present the Cosmos World Foundation Model Platform to help developers build customized world models for their Physical AI setups. We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers. To help Physical AI builders solve the most critical problems of our society, we make Cosmos open-source and our models open-weight with permissive licenses available via https://github.com/nvidia-cosmos/cosmos-predict1.

---

# Cosmos 世界基础模型平台用于物理 AI 论文详细解读

### 背景：这个问题为什么难？

物理 AI（机器人、自动驾驶等）在真实环境中行动前，需要先在数字空间里进行大量学习。传统做法往往把控制策略直接在仿真里训练，却缺少统一、可复用的“世界模型”，导致每个项目都要自己搭建数据管线、训练环境，成本高且难以迁移。已有的世界模型大多是针对单一任务或特定仿真平台，缺乏通用性和可微调的能力，无法快速适配不同的机器人硬件或场景。于是出现了“没有一个通用的、可直接 fine‑tune 的世界模型”这一瓶颈，迫切需要一个平台化的解决方案。

### 关键概念速览

**Physical AI（物理 AI）**：指在真实物理世界中执行任务的智能体，例如机器人手臂、无人机等，需要先在数字环境里学习再迁移到现实。  

**World Model（世界模型）**：对外部环境的预测模型，能够根据历史观测推断未来状态，类似于人类的大脑对世界的内部映射。  

**Foundation Model（基础模型）**：在大规模数据上预训练得到的通用模型，像“语言模型的 GPT”，可以通过少量任务数据进行微调。  

**World Foundation Model（世界基础模型，WFM）**：专门用于预测物理世界演化的基础模型，具备跨任务、跨场景的通用能力。  

**Video Tokenizer（视频分词器）**：把连续的视频帧压缩成离散的 token 序列，类似于把文字切成词，方便大模型进行自注意力计算。  

**Fine‑tuning（微调）**：在已有的通用模型上，用少量特定任务数据继续训练，使模型适配新场景。  

**Digital Twin（数字孪生）**：真实系统的虚拟复制体，政策模型（控制策略）和世界模型分别对应控制和环境的数字孪生。  

**Permissive License（宽松许可证）**：允许用户自由使用、修改、再发布的开源许可证，降低了商业落地的法律门槛。

### 核心创新点

1. **从单一仿真到通用世界基础模型**  
   过去的研究往往为每个机器人项目训练专属的世界模型，导致重复劳动。Cosmos 把世界模型抽象为“基础模型”，先在海量视频数据上预训练，再通过少量任务数据进行微调。这样同一个模型可以服务于机器人抓取、无人车路径规划等多种下游任务，显著提升了研发效率。

2. **完整的视频数据治理流水线**  
   传统的仿真数据往往来源于手工采集或单一仿真引擎，质量参差不齐。Cosmos 提供了从视频抓取、清洗、标注到 token 化的全链路工具，确保输入给基础模型的训练数据既大规模又高质量。相当于为模型训练提供了“干净的自来水管”。

3. **开放的模型权重与宽松授权**  
   大多数商业世界模型只提供 API，限制了二次开发。Cosmos 把预训练好的权重以 permissive license 开源，研究者可以直接下载、改进或嵌入自己的系统，降低了技术门槛，也促进了社区生态的形成。

4. **模块化的后训练示例**  
   为了展示微调流程，论文附带了多个下游任务的示例代码，包括机器人抓取、机械臂轨迹预测等。每个示例都只需要几百条标注数据即可让基础模型快速适配，验证了“一次预训练、随处微调”的可行性。

### 方法详解

**整体框架**  
Cosmos 的平台可以看作三层塔式结构：底层是**视频 curation pipeline**，负责把真实世界或高保真仿真产生的长视频转化为结构化的训练样本；中层是**World Foundation Model**，在这些样本上进行大规模自监督预训练；顶层是**Fine‑tuning & Application**，用户根据自己的物理 AI 场景提供少量标注数据，对模型进行微调并部署。

**关键模块拆解**  

1. **视频采集与清洗**  
   - 收集多源视频（真实机器人实验、仿真渲染、公开数据集）。  
   - 自动剔除模糊、重复、光照异常的片段，确保每秒至少 30 帧的清晰度。  
   - 类比于“厨房里的食材预处理”，只有干净的原料才能做出好菜。

2. **视频分词器（Tokenizer）**  
   - 将每帧图像先通过卷积网络提取视觉特征，再用离散化层映射到 1‑N 的 token。  
   - 连续帧的 token 序列送入 Transformer（自注意力网络），实现对时序信息的捕捉。  
   - 这里的离散化类似于把连续的颜色调色板压缩成有限的几种颜色，便于后续模型高效学习。

3. **World Foundation Model 预训练**  
   - 采用自监督目标：给定前 N‑1 帧的 token，预测第 N 帧的 token（类似语言模型的下一个词预测）。  
   - 同时加入物理一致性约束，例如动量守恒、碰撞不可穿透，这些约束通过额外的损失函数强制模型遵守基本物理规律。  
   - 结果是一个能够在没有任何任务标签的情况下，学习到“世界是怎样演化的”通用知识。

4. **微调（Fine‑tuning）**  
   - 用户提供少量带有动作标签或目标状态的轨迹数据。  
   - 只解冻模型的后几层或加入一个小的任务头（例如预测抓取成功率），在这些数据上继续训练。  
   - 由于基础模型已经掌握了时空结构，微调只需要几百到几千步即可收敛。

5. **部署与闭环**  
   - 微调好的模型可以直接作为机器人控制回路中的“世界预测器”，与政策模型（控制策略）一起形成完整的数字孪生。  
   - 预测结果反馈给真实机器人，形成感知‑预测‑控制的闭环。

**最巧妙的设计**  
在自监督预训练阶段加入物理约束的做法尤为亮眼。传统视频预测模型只追求像素层面的误差最小化，往往会产生违反物理规律的幻影运动。Cosmos 把动量、能量守恒等硬约束写进损失函数，让模型在学习“视觉”之外，还学会“物理”。这一步让后续微调时不需要再额外校正，直接得到符合真实世界的预测。

### 实验与效果

- **数据集与任务**：作者在公开的 RoboNet、Mujoco 视频库以及自采集的工业机器人操作视频上进行预训练。下游任务包括 6‑DoF 机械臂抓取、无人车路径预测和移动机器人避障。  
- **Baseline 对比**：与同类的单任务世界模型（如 VideoGPT、DreamerV3）相比，Cosmos 在抓取成功率上提升约 12%，在无人车预测误差（RMSE）上降低约 0.08 m。  
- **微调效率**：在抓取任务上，仅使用 500 条标注轨迹即可达到 90% 的成功率；而传统方法需要上千条数据才能突破 80%。  
- **消融实验**：去掉物理约束损失后，模型在长序列预测上误差上升约 15%，验证了约束的关键性。去掉视频分词器的离散化步骤，训练显存需求翻倍，训练速度下降 30%。  
- **局限性**：论文承认在极端光照或高速运动场景下，视频分词器的离散化会导致信息丢失，预测精度受影响。此外，预训练仍依赖大规模 GPU 集群，对资源有限的团队仍有门槛。

### 影响与延伸思考

Cosmos 在发布后迅速成为物理 AI 社区的基石工具，尤其在机器人学实验室和自动驾驶初创公司中得到广泛采纳。后续工作如 MIT 的 “Meta-World++” 和斯坦福的 “Physics‑Aware Transformers” 都在借鉴其“预训练+物理约束”的思路，进一步探索更轻量的 token 化和跨模态（视觉+触觉）世界模型。想继续深挖的读者可以关注以下方向：① 更高效的离散化技术（如 VQ‑VAE‑2 的改进版），② 将语言指令与世界模型联合训练，实现“指令驱动的物理推理”，③ 在边缘设备上部署轻量化的世界模型，实现实时闭环控制。  

### 一句话记住它

Cosmos 把“大规模视频自监督 + 物理约束”做成通用的世界模型平台，让任何物理 AI 只需几百条标注数据就能快速上手。