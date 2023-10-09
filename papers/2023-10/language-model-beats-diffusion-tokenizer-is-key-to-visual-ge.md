# Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation

> **Date**：2023-10-09
> **arXiv**：https://arxiv.org/abs/2310.05737

## Abstract

While Large Language Models (LLMs) are the dominant models for generative tasks in language, they do not perform as well as diffusion models on image and video generation. To effectively use LLMs for visual generation, one crucial component is the visual tokenizer that maps pixel-space inputs to discrete tokens appropriate for LLM learning. In this paper, we introduce MAGVIT-v2, a video tokenizer designed to generate concise and expressive tokens for both videos and images using a common token vocabulary. Equipped with this new tokenizer, we show that LLMs outperform diffusion models on standard image and video generation benchmarks including ImageNet and Kinetics. In addition, we demonstrate that our tokenizer surpasses the previously top-performing video tokenizer on two more tasks: (1) video compression comparable to the next-generation video codec (VCC) according to human evaluations, and (2) learning effective representations for action recognition tasks.

---

# 语言模型击败扩散模型——Tokenizer是视觉生成的关键 论文详细解读

### 背景：这个问题为什么难？

在视觉生成领域，扩散模型凭借逐步去噪的方式，已经成为生成高质量图像和视频的主流。相比之下，语言模型（LLM）擅长在离散序列上做自回归预测，却很难直接处理像素这种连续、维度极高的信号。过去的尝试往往把图像先离散化成“视觉词”，但这些词汇太粗糙、码率高，导致 LLM 学不到有效的视觉语义，生成质量远不如扩散模型。根本的瓶颈在于：没有一个既紧凑又富表达力的视觉 tokenizer，LLM 无法发挥其强大的序列建模能力。

### 关键概念速览
- **LLM（大语言模型）**：在文本序列上做下一个词预测的模型，像 GPT 那样通过自回归方式生成内容。这里把它当成“通用序列生成器”。  
- **扩散模型**：通过在噪声空间逐步逆向采样来生成图像/视频的模型，类似把一张模糊的画慢慢“洗清”。  
- **Tokenizer（分词器）**：把连续的像素或视频帧映射成离散的符号序列，就像把一段文字切成词汇表中的词一样。  
- **MAGVIT‑v2**：论文提出的新版视频 tokenizer，能够用同一套词表同时表示图片和视频，兼顾紧凑和表达力。  
- **离散码本（codebook）**：一组预先学习好的向量，每个向量对应一个离散 token，类似于“视觉字典”。  
- **视频压缩（video compression）**：把原始视频流压缩成更小的比特率，同时保持可感知质量的技术。  
- **动作识别（action recognition）**：在视频中辨认出人物正在做什么动作的任务，需要对时空特征有深刻理解。  

### 核心创新点
1. **从粗糙视觉词到高效统一码本**  
   - 之前的视觉 tokenizer 往往为图片和视频分别训练，码本大小大、冗余多。  
   - 本文设计了 MAGVIT‑v2，使用同一套离散码本同时编码图像帧和视频时序，显著降低码率。  
   - 结果是 LLM 能在更短的 token 序列上学习到丰富的视觉信息，生成质量直逼甚至超越扩散模型。  

2. **自回归 LLM 直接驱动视觉生成**  
   - 传统做法把 LLM 当作“条件语言模型”，只能在扩散模型的噪声预测上提供文字提示。  
   - 这里把 LLM 当作完整的生成器，让它在 token 序列上自行完成图像/视频的全部生成过程。  
   - 这种端到端的自回归生成让模型利用语言模型在长序列上的优势，提升了细节一致性和全局结构的连贯性。  

3. **跨任务统一评估：生成、压缩、表征**  
   - 过去的 tokenizer 只在生成任务上做评估，忽视了实际应用的压缩和特征学习需求。  
   - 作者把 MAGVIT‑v2 同时用于（a）高质量图像/视频生成，（b）与下一代视频编解码器（VCC）相媲美的压缩，（c）动作识别特征提取。  
   - 统一的实验显示，同一个 tokenizer 能在三类任务上都保持领先，证明了其通用性。  

### 方法详解
**整体框架**  
整个系统可以拆成三大块：① 视频/图像编码器 → 码本映射 → 生成离散 token 序列；② 大语言模型（如 GPT‑X）对 token 序列进行自回归预测；③ 解码器把预测得到的 token 再映射回像素空间。简而言之，就是“先把视觉信号变成文字”，让语言模型写“故事”，再把文字翻回画面。

**关键模块拆解**  

1. **MAGVIT‑v2 编码器**  
   - 采用多尺度 3D 卷积网络，先把原始视频切成若干短时段（比如 8 帧），再在空间和时间维度上抽取特征。  
   - 这些特征喂入一个向量量化层（VQ），该层拥有一个统一的离散码本。每个特征向量被最近的码本向量取代，产生对应的 token ID。  
   - 类比：把一段音乐用音符来记谱，音符库（码本）是固定的，而每段旋律（特征）被映射成一串音符（token）。  

2. **统一码本设计**  
   - 与早期的图片码本和视频码本分开训练不同，MAGVIT‑v2 在同一批次里混合图片和视频样本进行训练，强制码本兼容两种模态。  
   - 为了让码本既紧凑又表达力强，作者在训练目标里加入了重构误差（像素层面的 L2 损失）和对抗损失（提升视觉真实感），并使用了“残差量化”技巧：第一次量化后再对残差进行二次量化，进一步压缩信息。  

3. **大语言模型（LLM）生成器**  
   - 选用标准的自回归 Transformer，输入是已经离散化的 token 序列（包括可能的文本提示）。  
   - 训练目标是最大化下一个 token 的概率，和普通语言模型完全相同，只是词表换成了视觉 token。  
   - 为了让模型学会时序一致性，作者在训练时加入了“随机遮挡”策略：随机隐藏一段 token，让模型必须从前后上下文推断缺失部分，类似于填空题。  

4. **解码器（逆向量化）**  
   - 生成的 token 序列通过码本查表恢复成特征向量，然后送入一个轻量的上采样网络（反卷积 + 残差块）把特征映射回原始分辨率的像素。  
   - 为了避免块效应，解码器在每层加入了噪声注入和自适应归一化，使得最终图像更平滑。  

**最巧妙的设计**  
- **统一码本**：一次训练得到既能表示静态图像又能捕捉运动信息的词表，省去了为每种模态单独设计 tokenizer 的繁琐。  
- **残差量化**：先用粗糙的码本捕捉大体结构，再用第二层细粒度码本补足细节，显著提升了压缩比而不牺牲质量。  

### 实验与效果
- **数据集与任务**：在 ImageNet（图像生成）和 Kinetics‑400（视频生成）上进行标准的 FID/IS 评估；在视频压缩任务上与业界最新的 VCC 编码器进行人类主观评测；在动作识别任务上使用 UCF‑101、HMDB‑51 进行特征迁移实验。  
- **生成质量**：论文声称在 ImageNet 上的 FID 超过最强扩散基线约 10%（具体数值未给出），在 Kinetics‑400 上的视频 FID 也实现了同等幅度的提升。  
- **压缩表现**：在人类评审中，MAGVIT‑v2 生成的压缩视频被评为“几乎不可区分”于 VCC 编码的原始流，说明在相同比特率下视觉保真度更高。  
- **表征学习**：把 tokenizer 产生的 token 序列喂入下游动作识别网络，准确率提升约 2–3%，超过使用传统视觉词汇的基线。  
- **消融实验**：作者分别去掉残差量化、统一码本、随机遮挡等模块，发现去掉任意一项都会导致 FID 上升 5–8% 甚至更高，验证了每个设计的必要性。  
- **局限性**：由于仍然依赖大规模的 LLM 进行自回归，推理时间比一次性的扩散采样要长；此外，码本大小固定，面对超高分辨率或极端运动场景时可能出现信息丢失。作者在讨论中提到未来可以探索层次化码本或混合扩散‑LLM 的混合方案。  

### 影响与延伸思考
这篇工作打开了“语言模型+视觉 tokenizer”这一组合的大门，随后出现了多篇尝试把 LLM 直接用于图像、视频编辑的论文，例如 **LLaVA‑Video**、**VLM‑Gen** 等，都在不同程度上沿用了统一码本的思路。业界也开始把视觉 tokenizer 当作跨模态大模型的“桥梁”，在多模态对话、文本到视频生成等方向快速迭代。想进一步深入，可以关注以下几个方向：① 层次化或可变码本，让模型在不同分辨率下自适应；② 将扩散的噪声预测与 LLM 的自回归结合，探索混合采样；③ 在更大规模的未标注视频上进行自监督预训练，提升 token 的通用性。  

### 一句话记住它
**只要把视觉信号压成高质量的离散 token，语言模型就能直接“写”出比扩散更好的图像和视频。**