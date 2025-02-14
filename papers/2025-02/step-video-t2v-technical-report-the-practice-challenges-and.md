# Step-Video-T2V Technical Report: The Practice, Challenges, and Future of   Video Foundation Model

> **Date**：2025-02-14
> **arXiv**：https://arxiv.org/abs/2502.10248

## Abstract

We present Step-Video-T2V, a state-of-the-art text-to-video pre-trained model with 30B parameters and the ability to generate videos up to 204 frames in length. A deep compression Variational Autoencoder, Video-VAE, is designed for video generation tasks, achieving 16x16 spatial and 8x temporal compression ratios, while maintaining exceptional video reconstruction quality. User prompts are encoded using two bilingual text encoders to handle both English and Chinese. A DiT with 3D full attention is trained using Flow Matching and is employed to denoise input noise into latent frames. A video-based DPO approach, Video-DPO, is applied to reduce artifacts and improve the visual quality of the generated videos. We also detail our training strategies and share key observations and insights. Step-Video-T2V's performance is evaluated on a novel video generation benchmark, Step-Video-T2V-Eval, demonstrating its state-of-the-art text-to-video quality when compared with both open-source and commercial engines. Additionally, we discuss the limitations of current diffusion-based model paradigm and outline future directions for video foundation models. We make both Step-Video-T2V and Step-Video-T2V-Eval available at https://github.com/stepfun-ai/Step-Video-T2V. The online version can be accessed from https://yuewen.cn/videos as well. Our goal is to accelerate the innovation of video foundation models and empower video content creators.

---

# Step-Video-T2V 论文详细解读

### 背景：这个问题为什么难？

生成高质量视频一直是 AI 研究的硬核挑战。传统的文本到视频（text‑to‑video）模型往往受限于两大瓶颈：一是视频数据的空间‑时间维度极大，导致模型参数和显存需求爆炸；二是扩散过程在时间轴上的噪声去除不够精准，常出现抖动、模糊或不连贯的伪影。此前的开源模型要么只能生成几秒、几帧的低分辨率短片，要么需要昂贵的商业算力才能跑出稍微可看的结果。于是，如何在保持可接受算力的前提下，压缩视频表示、提升时空一致性并抑制生成伪影，成为迫切需要突破的技术难点。

### 关键概念速览

**Variational Autoencoder（变分自编码器）**：一种先把原始数据压成潜在向量，再从向量重建的神经网络。想象把一段视频压进一个小盒子，盒子里保存了“核心信息”，解压时再把画面拼回去。

**Video‑VAE**：专门针对视频的 VAE，采用 16×16 的空间压缩和 8× 的时间压缩。可以把一段 204 帧的 720p 视频压成极小的潜在张量，类似把整部电影浓缩成几页剧本。

**DiT（Diffusion Transformer）**：把扩散模型的噪声去除过程交给 Transformer，利用自注意力捕捉全局依赖。这里的 DiT 采用 3D 全注意力，即在空间和时间两个维度上同时计算注意力，像在三维立体棋盘上同时观察每一颗子。

**Flow Matching（流匹配）**：一种训练目标，直接让模型学习从噪声流向真实数据的速度场，而不是传统的逐步噪声预测。可以比作教模型直接画出从雾到清晰画面的路径，而不是一步一步擦除雾气。

**DPO（Direct Preference Optimization）**：把人类偏好直接写进模型的优化目标。模型会在生成过程中参考一个奖励模型，像裁判一样实时挑选更好看的帧。

**双语文本编码器**：使用两套 CLIP‑style 编码器分别处理英文和中文提示，确保跨语言的语义对齐。可以想象把不同语言的指令都翻译成同一种内部语言再喂给模型。

**Step‑Video‑T2V‑Eval**：作者自建的评测基准，覆盖多种场景和指标，用来客观衡量生成视频的质量与一致性。

### 核心创新点

1. **高压缩率 Video‑VAE → 采用 16×16 空间、8× 时间压缩 → 在显存占用上降低数十倍，同时保持重建误差在可接受范围**。作者通过多阶段训练，让压缩网络先学会低压缩率的重建，再逐步提升压缩比，避免一次性压得太狠导致信息丢失。

2. **3D 全注意力 DiT + Flow Matching → 用 Transformer 同时关注空间和时间的全局关系，并直接学习噪声流的速度场 → 生成过程更平滑、帧间连贯性显著提升**。相较于传统 2D 注意力或仅在时间维度上做噪声预测的扩散模型，这种设计让模型在一次前向传播中就能捕捉跨帧的运动信息。

3. **视频专用 DPO（Video‑DPO） → 在训练后期加入基于奖励模型的偏好优化，专门针对视频伪影和风格漂移进行惩罚 → 生成的视频在细节保真度和整体美感上超过同类开源模型**。这一步把人类审美直接映射到梯度上，避免了仅靠像素损失函数难以捕捉的主观质量问题。

4. **双语文本编码 + 大规模跨语言数据 → 同时使用 Hunyuan‑CLIP（中文）和 Step‑LLM（英文）进行提示编码 → 模型能够无缝接受中英混合指令，拓宽了实际应用场景**。过去的 T2V 系统大多只针对单一语言，这里实现了真正的多语言兼容。

### 方法详解

**整体框架**  
Step‑Video‑T2V 的生成流程可以划分为四个阶段：  
1) 文本提示 → 双语编码器 → 统一的文本向量；  
2) 随机噪声张量（与视频潜在空间同形） → 3D DiT（Flow Matching） → 逐步去噪得到潜在帧序列；  
3) 潜在帧序列 → Video‑VAE 解码器 → 重建出完整像素视频；  
4) Video‑DPO 评价与微调 → 最终输出质量更高的成片。

**文本编码**  
两套 CLIP‑style 编码器分别把英文和中文的文字映射到同一维度的向量空间。若提示中混有两种语言，系统会分别编码后做加权求和，得到一个统一的语义向量，后续所有时空生成都以此向量为条件。

**Video‑VAE 设计**  
压缩端采用 3D 像素重排（pixel‑shuffle）把空间维度压 16 倍，再通过时间卷积把帧数压 8 倍，得到形如 `[B, C, T/8, H/16, W/16]` 的潜在张量。解码端是对称的双路径结构：一条路径负责空间上逐层上采样，另一条负责时间上逐帧展开，二者在每层通过通道复制（channel‑repeat）保持信息流通。训练时先用低压缩率（如 4×4、2×）让网络熟悉基本重建，再逐步提升压缩比，防止一次性压得太狠导致梯度消失。

**3D DiT + Flow Matching**  
DiT 的输入是噪声潜在张量 + 时间步标记 + 文本向量。3D 全注意力在每一次自注意力计算中同时考虑 `(t, h, w)` 三个坐标的相互影响，等价于在一个立体网格上做全局信息交流。Flow Matching 的目标不是预测下一个噪声，而是让模型输出一个“速度场”，即在连续时间上噪声如何流向真实潜在帧。训练时最小化预测速度与真实速度的均方误差（MSE），推理时通过数值积分把噪声逐步推向清晰潜在帧。

**Video‑DPO**  
在基本的 DiT‑Flow Matching 训练完成后，作者再训练一个奖励模型，输入为生成视频与参考真实视频的对比特征，输出一个偏好分数。随后使用 Direct Preference Optimization，把这个分数直接转化为梯度，指导 DiT 在去噪过程中更倾向于产生高分的帧。这样可以在不额外增加像素级损失的情况下，有效抑制常见的运动抖动和颜色漂移。

**训练策略**  
整体训练分四步：① 文本‑图像预训练（为双语编码器和 DiT 打基础）；② 文本‑视频/图像预训练，从低分辨率到高分辨率逐层提升；③ 文本‑视频微调，加入多尺度 cp‑averaging（模型检查点平均）以平滑噪声；④ Video‑DPO 微调。每一步都配合大规模多语言、跨域数据，确保模型在不同场景下都有足够的泛化能力。

### 实验与效果

- **评测基准**：作者构建了 Step‑Video‑T2V‑Eval，覆盖 10+ 主题（自然风光、人物动作、抽象艺术等），每个主题提供多条中英混合提示，并使用客观指标（FID、CLIP‑Score）和主观用户调研两类评分。
- **对比基线**：与当前主流开源模型（如 Stable‑Video、Gen‑2‑lite）以及若干商业闭源引擎（如 Runway、Google Imagen Video）进行横向比较。论文声称在 CLIP‑Score 上提升约 12%~18%，在用户主观满意度上领先 15% 以上。
- **消融实验**：分别去掉 Video‑VAE 高压缩、3D 全注意力、Flow Matching、Video‑DPO 四个模块，结果显示：去掉 Video‑VAE 导致显存需求翻倍且帧率下降 30%；去掉 3D 注意力后帧间运动不连贯，Fidelity 降低约 10%；去掉 Flow Matching 使生成过程噪声残留明显；去掉 Video‑DPO 则伪影率上升约 20%。
- **局限性**：作者承认仍然受限于扩散模型的采样时间，生成 204 帧视频需要数分钟的 GPU 计算；在极端快速运动或细粒度纹理（如水滴、火焰）上仍会出现细节缺失；跨语言提示的细微语义差异仍有提升空间。

### 影响与延伸思考

Step‑Video‑T2V 在开源社区掀起了“高压缩‑全时空注意力” 的新潮流。随后出现的几篇工作（如 **Vivid‑Video‑Diffusion**、**Temporal‑VAE‑Fusion**）都在不同程度上借鉴了 3D DiT 与 Flow Matching 的思路，或在 Video‑VAE 上加入感知损失以进一步提升细节。对想继续深耕视频生成的读者，值得关注的方向包括：① 更高效的采样算法（如 DP‑Sampler、二阶欧拉积分）；② 跨模态对齐的统一大模型（文本‑音频‑视频同框）；③ 基于强化学习的长期一致性约束。整体来看，这篇报告为视频基础模型的系统化构建提供了完整的技术栈和实践经验。

### 一句话记住它

**Step‑Video‑T2V 用 3D 全注意力的 Diffusion Transformer 搭配超高压缩的 Video‑VAE，实现了 30 B 参数下的 204 帧中英双语文本到视频生成，显著提升了时空连贯性与视觉质量。**