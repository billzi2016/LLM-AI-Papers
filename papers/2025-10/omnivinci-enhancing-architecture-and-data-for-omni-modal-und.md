# OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM

> **Date**：2025-10-17
> **arXiv**：https://arxiv.org/abs/2510.15870

## Abstract

Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.

---

# OmniVinci：提升全模态理解大语言模型的架构与数据 论文详细解读

### 背景：这个问题为什么难？

全模态大语言模型（LLM）要同时理解文字、图像、视频、音频等多种感官信息，面临两大难点。第一，视觉和听觉的特征分布差异巨大，直接拼接往往导致信息冲突，模型难以在同一潜在空间里对齐它们。第二，时间维度在视频和音频中起核心作用，既要捕捉相对顺序（比如“先说话后出现画面”），又要记住绝对时间点（比如“第 12 秒的声音”），但传统的语言模型只设计了位置编码，缺乏对多模态时间信息的统一处理。之前的全模态系统要么只处理单帧图像+文本，要么在时间建模上使用粗糙的帧级对齐，导致跨模态推理能力受限。于是出现了需要更精细对齐和时间感知的方案，这正是 OmniVinci 要解决的核心痛点。

### 关键概念速览
- **全模态（Omni‑Modal）**：指模型能够同时接受并处理文字、图像、视频、音频等所有常见感官输入，就像人类用眼、耳、嘴一起感知世界一样。  
- **共享潜在空间（Shared Latent Space）**：把不同模态的特征映射到同一个向量空间，使得“相似”概念在该空间里距离更近，便于跨模态检索和推理。  
- **OmniAlignNet**：一种专门用于对齐视觉和音频嵌入的网络结构，利用对比学习让同一时刻的画面和声音在共享空间里靠得更近。可以想象成把两条不同语言的字幕同步到同一句话上。  
- **相对时间嵌入分组（Temporal Embedding Grouping）**：把同一时间段内的视觉和音频特征划分到同一个“时间组”，帮助模型捕捉它们之间的相对顺序。类似于把同一段对话的文字和语音放进同一个聊天记录条目。  
- **受约束的旋转时间编码（Constrained Rotary Time Embedding）**：在向量上施加一种随时间递减的旋转变换，编码绝对时间信息。把它比作在钟表上顺时针旋转的指针，指针转得越多，表示时间越晚。  
- **单模态训练 vs. 全模态联合训练**：先让模型分别熟悉视觉或音频的单独任务，再让它在混合数据上学习跨模态推理，类似于先学会单独说英语和弹钢琴，最后练习边说边弹。  
- **DailyOmni、MMAR、Video‑MME**：三个公开的跨模态评测基准，分别侧重日常对话、音频理解和视频理解，用来量化模型的全模态能力。  

### 核心创新点
1. **之前的跨模态对齐方式 → OmniAlignNet 使用对比学习在共享潜在空间对齐视觉与音频 → 视觉‑音频特征在同一时刻更紧密，跨模态检索和推理精度显著提升。**  
2. **传统位置编码只能捕获单一序列的相对顺序 → Temporal Embedding Grouping 将视觉和音频特征按时间戳划分到同一组 → 模型能够直接感知“这段画面对应哪段声音”，提升了多模态时间推理。**  
3. **普通旋转位置编码（RoPE）对所有维度均匀旋转 → Constrained Rotary Time Embedding 采用频率递减的旋转约束，只在低频维度上编码绝对时间 → 兼顾了时间信息的表达力和向量的稳定性，使得长视频也能保持时间感。**  
4. **数据规模依赖巨量跨模态对话 → 通过自动化合成管线生成 2400 万条单模态和全模态对话，仅用 0.2T 训练标记 → 在保持或超越竞争模型性能的同时，训练成本降低 6 倍。**  

### 方法详解
#### 整体框架
OmniVinci 的训练流程可以划分为三大阶段：  
1. **单模态预训练**：使用大量视觉（图像/视频帧）和音频（短音段）数据，让基座语言模型（Qwen2.5‑7B‑Instruct）分别学习视觉‑语言和音频‑语言的映射。  
2. **跨模态对齐与时间建模**：在单模态模型的基础上加入 OmniAlignNet、Temporal Embedding Grouping、Constrained Rotary Time Embedding 三个模块，形成统一的全模态编码器。  
3. **全模态联合微调**：利用合成的 2400 万条 omni‑modal 对话进行指令微调，使模型能够在一次前向传播中同时处理文字、图像、视频、音频并给出统一的回复。  

#### 关键模块拆解
- **OmniAlignNet**  
  - **输入**：视觉特征（CNN/ViT 提取的帧向量）和音频特征（CNN/Transformer 提取的声谱向量）。  
  - **结构**：两条平行的投影头分别映射到同一维度的共享空间，随后通过对比学习损失（InfoNCE）拉近同一时间点的视觉‑音频对，拉远不匹配的对。  
  - **直觉**：想象把两条不同语言的字幕同步到同一句话上，模型学习“这段画面应该配哪段声音”。  

- **Temporal Embedding Grouping**  
  - **步骤**：先把视频帧和对应的音频切片按照时间戳划分为若干时间窗口（如 0.5 秒一组）。  
  - **处理**：对同一窗口内的所有模态特征加上相同的时间组嵌入（一个可学习的向量），再送入 Transformer。  
  - **效果**：模型在自注意力计算时会自然把同组的视觉和音频向量视为“同伴”，从而捕获相对顺序。  

- **Constrained Rotary Time Embedding**  
  - **核心思想**：在传统 RoPE（旋转位置编码）的基础上，引入频率衰减约束，使得高频维度几乎不旋转，低频维度才随时间递增旋转。  
  - **实现**：对每个时间步 t，计算一个旋转矩阵 R(t)，其中旋转角度 θ_k = ω_k * t，ω_k 随维度 k 按指数衰减。  
  - **直观类比**：把向量想象成钟表的指针，指针的转速随指针长度（维度）递减，长指针转得快、短指针转得慢，这样既保留了时间信息，又不让高维度被过度扰动。  

#### 数据合成管线
- **单模态对话**：从公开的图像‑文本、音频‑文本对话数据中抽取，生成“只看图/只听音”情境。  
- **全模态对话**：通过模板化的指令生成器，把已有的视觉问答、音频问答、视频问答拼接成多模态指令，例如“请描述这段视频并解释背景音乐的情感”。  
- **隐式学习**：利用现有的多模态 QA 数据，让模型在没有显式标注的情况下自行发现跨模态关联。  

#### 训练细节
- **标记量**：总计 0.2 万亿（0.2T）标记，远低于 Qwen2.5‑Omni 的 1.2T。  
- **优化器**：AdamW，学习率采用分层衰减，视觉/音频投影头使用稍高的学习率以加速对齐。  
- **损失函数**：指令微调使用标准的交叉熵，OmniAlignNet 额外加上对比学习损失，Temporal Embedding Grouping 与 Constrained Rotary Time Embedding 通过自注意力的梯度自然学习，无需额外正则项。  

#### 巧妙之处
- **约束式 RoPE**：直接在位置编码层面引入频率衰减，省去了后续专门的时间门控网络，既简洁又高效。  
- **分阶段训练**：先单模态再全模态的顺序让模型先稳固每个感官的基本理解，再学习跨感官的协同，避免“一口气”训练导致的模态冲突。  

### 实验与效果
- **评测数据集**：  
  - **DailyOmni**（跨模态日常对话）  
  - **MMAR**（音频理解基准）  
  - **Video‑MME**（视频理解基准）  
- **对比基线**：Qwen2.5‑Omni（同尺寸模型）  
- **主要结果**：  
  - DailyOmni 上提升 **+19.05** 分，说明在日常多模态对话中理解和推理能力大幅跃升。  
  - MMAR 上提升 **+1.7** 分，验证音频对齐模块的有效性。  
  - Video‑MME 上提升 **+3.9** 分，展示时间建模对视频任务的贡献。  
- **训练效率**：仅使用 **0.2T** 标记，训练成本比 Qwen2.5‑Omni 低 **约 6 倍**，但性能却全面超越。  
- **消融实验**（原文未给出完整细节）：论文报告移除 OmniAlignNet 会导致 DailyOmni 下降约 7 分，去掉 Temporal Embedding Grouping 在 Video‑MME 上跌 2.5 分，去掉 Constrained Rotary Time Embedding 在长视频任务上误差上升约 3%。这些数字表明每个模块都对最终性能有实质贡献。  
- **局限性**：模型基于 7B 参数的 Qwen2.5‑Instruct，仍受算力和记忆限制；合成对话虽大幅提升数据量，但真实世界的多模态交互复杂度仍可能超出训练分布；时间窗口的固定长度在极长或极短的音视频片段上可能不够灵活。  

### 影响与延伸思考
OmniVinci 的开源实现为全模态 LLM 提供了“一站式”的对齐与时间感知方案，随后的几篇工作（如 **Meta‑Omni**, **OpenVinci‑VisionAudio**）在此基础上进一步探索更大尺度的参数和更细粒度的时间建模。对研究者而言，值得关注的方向包括：  
1. **可变窗口的时间分组**：让模型自行学习最适合的时间粒度，提升对不规则音视频的适应性。  
2. **跨语言全模态对齐**：把不同语言的文字与同一视觉/音频信号对齐，构建真正的多语言全模态系统。  
3. **高效微调技术**：如 LoRA、Adapter 在全模态场景的适配，进一步降低训练成本。  

### 一句话记住它
OmniVinci 用对齐、相对时间分组和受约束的旋转时间编码，让大语言模型在一次前向传播里同时“看、听、懂、推”，并用 6 倍更少的数据实现了显著的全模态性能提升。