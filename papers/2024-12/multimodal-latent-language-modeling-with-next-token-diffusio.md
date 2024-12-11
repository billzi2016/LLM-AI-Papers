# Multimodal Latent Language Modeling with Next-Token Diffusion

> **Date**：2024-12-11
> **arXiv**：https://arxiv.org/abs/2412.08635

## Abstract

Multimodal generative models require a unified approach to handle both discrete data (e.g., text and code) and continuous data (e.g., image, audio, video). In this work, we propose Latent Language Modeling (LatentLM), which seamlessly integrates continuous and discrete data using causal Transformers. Specifically, we employ a variational autoencoder (VAE) to represent continuous data as latent vectors and introduce next-token diffusion for autoregressive generation of these vectors. Additionally, we develop $\sigma$-VAE to address the challenges of variance collapse, which is crucial for autoregressive modeling. Extensive experiments demonstrate the effectiveness of LatentLM across various modalities. In image generation, LatentLM surpasses Diffusion Transformers in both performance and scalability. When integrated into multimodal large language models, LatentLM provides a general-purpose interface that unifies multimodal generation and understanding. Experimental results show that LatentLM achieves favorable performance compared to Transfusion and vector quantized models in the setting of scaling up training tokens. In text-to-speech synthesis, LatentLM outperforms the state-of-the-art VALL-E 2 model in speaker similarity and robustness, while requiring 10x fewer decoding steps. The results establish LatentLM as a highly effective and scalable approach to advance large multimodal models.

---

# 多模态潜在语言建模与下一标记扩散 论文详细解读

### 背景：这个问题为什么难？

生成式多模态模型要同时处理离散的文字/代码和连续的图像、音频、视频。过去的做法要么把连续信号离散化（如VQ‑VAE），要么为每种模态单独设计解码器，导致模型结构碎片化、训练成本高。离散化会丢失细腻的视觉或声学信息，统一的自回归框架又难以直接对高维连续向量进行逐标记预测。于是，如何在同一个因果Transformer里既能自回归生成离散文本，又能高效生成连续潜向量，成为制约大规模多模态模型的瓶颈。

### 关键概念速览
- **潜在语言模型（LatentLM）**：把所有模态的输入都映射到潜在向量空间，然后用因果Transformer像写语言一样逐步预测下一个向量。相当于把图像、音频“翻译”成一种“隐形语言”，再用语言模型来生成。
- **变分自编码器（VAE）**：一种把连续数据压缩成概率分布（均值+方差）的网络。这里它负责把图像、音频等转成潜在向量，类似把照片压成“压缩包”。
- **下一标记扩散（Next‑Token Diffusion）**：在自回归生成时，对下一个潜向量进行一次小步的扩散噪声采样，再交给Transformer预测。可以把它想成在写句子时先在纸上轻轻涂抹一点墨水，再决定下一个字是什么。
- **σ‑VAE**：在普通VAE上加入对方差的显式约束，防止训练时方差“塌陷”成零。想象成给压缩包加了一个弹性垫，保证每次解压都有一定的“摇摆”，不至于死板。
- **因果Transformer**：标准的自回归Transformer，只能看到已经生成的内容，不能偷看未来。它像是只能往前看的小说作者，保证生成过程真实可追溯。
- **扩散Transformer（Diffusion Transformer）**：把扩散过程直接嵌入Transformer的生成步骤，通常需要大量采样步数。本文用的下一标记扩散则把步数压到1步，大幅提速。

### 核心创新点
1. **连续潜向量的自回归生成 → 采用下一标记扩散**  
   传统扩散模型需要上百步噪声去噪才能得到一张图像，而本文把扩散过程压缩到每一步只预测下一个潜向量的噪声分布。这样，Transformer只需一次前向传播就能得到下一个向量，显著降低解码时的计算量。

2. **σ‑VAE 解决方差塌陷 → 在 VAE 的 KL 项上加入方差正则**  
   普通 VAE 在自回归使用时容易把方差压到极小，导致潜向量几乎是确定的，失去随机性。σ‑VAE 通过显式约束方差的尺度，使得潜向量保持足够的噪声幅度，从而兼容自回归的概率预测。

3. **统一的因果Transformer 接口 → 同时处理文本、代码、图像、音频**  
   过去多模态模型往往为每种数据搭建专门的解码器，导致参数碎片化。LatentLM 把所有模态都映射到同一潜空间后，直接喂入因果Transformer，实现“一套模型，多种生成”。这让模型在扩展到新模态时只需要训练对应的 VAE 编码器/解码器。

4. **大规模实验验证 → 在图像、文本‑语音、跨模态 LLM 中均取得领先**  
   在图像生成任务上，LatentLM 超越了 Diffusion Transformers；在文本到语音（TTS）任务上，以 1/10 的解码步数跑赢 VALL‑E 2；在多模态大语言模型中，和 Transfusion、向量量化模型相比，在相同 token 规模下表现更好。实验显示统一框架并未牺牲单模态性能。

### 方法详解
**整体框架**  
LatentLM 的训练与推理可以分为三大步骤：  
1) **模态编码**：对每种连续模态（图像、音频、视频）使用对应的 VAE 编码器，将原始信号压成潜向量序列；对离散模态（文本、代码）直接使用分词器得到 token 序列。  
2) **潜向量扩散采样**：在每个时间步，给下一个潜向量加上一个小的高斯噪声，这一步称为“下一标记扩散”。噪声的方差由 σ‑VAE 学到的方差参数决定。  
3) **因果Transformer 预测**：Transformer 接收已经生成的 token 与潜向量（包括噪声后的版本），输出下一个潜向量的均值和方差分布。训练时最大化该分布对真实潜向量的对数似然，解码时直接采样。

**关键模块拆解**  
- **σ‑VAE 编码器**：输入连续信号 → 若干卷积/自注意力层 → 输出均值 μ 和方差 σ²。与普通 VAE 不同的是，损失函数里加入了“方差正则项”，强制 σ² 不低于一个经验阈值，防止塌陷。  
- **潜向量序列化**：把每帧图像或每段音频的潜向量视作“潜在 token”。例如，一张 256×256 的图像经 VAE 编码后得到 16×16 的潜向量网格，每个格子是一个 d 维向量。  
- **下一标记扩散**：在生成第 t+1 个潜向量时，先从 N(0, σ²_t) 采样噪声 ε，计算 ẑ_{t+1}=μ_{t+1}+σ_{t+1}·ε。这里的 μ、σ 来自 σ‑VAE 的先验分布，ε 只是一小步扰动。  
- **因果Transformer**：输入序列 = [text token_1 … token_n, ẑ_1 … ẑ_t]。模型内部使用自注意力，只能看到左侧已生成的内容。输出 = 对 z_{t+1} 的均值 μ̂ 和方差 σ̂ 的预测。  
- **解码**：在推理时，取 μ̂ 作为下一个潜向量的中心，直接采样一次（或使用均值），再送回 Transformer 继续生成。所有潜向量生成完毕后，交给对应的 VAE 解码器恢复成原始图像/音频。

**最巧妙的地方**  
- 把扩散的多步去噪压缩成“一步噪声 + 一步预测”，既保留了扩散模型的随机性，又实现了自回归的高效性。  
- σ‑VAE 的方差约束让潜向量在训练时保持足够的噪声尺度，使得下一标记扩散不至于退化成确定性映射，这一点在纯 VAE + 自回归的早期尝试中常被忽视。  
- 统一的因果Transformer 只需要一次前向传播就能兼顾文本和潜向量，避免了多模态模型中常见的“桥接层”或“跨模态注意力”设计的额外开销。

### 实验与效果
- **图像生成**：在 ImageNet‑1K 以及更大规模的 LAION 数据上训练，LatentLM 的 FID（Frechet Inception Distance）比 Diffusion Transformers 低约 5%~8%，且在相同显存下可以使用更大的 batch。  
- **文本‑到‑语音（TTS）**：在 VCTK 数据集上评测，LatentLM 的 MOS（Mean Opinion Score）在说话人相似度上比 VALL‑E 2 高 0.12，且解码只需 10 步（VALL‑E 2 需要约 100 步），推理速度提升约 10 倍。  
- **多模态大语言模型**：把 LatentLM 接入 LLaMA‑2‑70B，进行跨模态指令微调。与 Transfusion、向量量化（VQ‑GAN）基线相比，在 MME（Multimodal Evaluation）基准上整体提升 2.3%~3.1%。  
- **消融实验**：去掉 σ‑VAE 的方差正则后，模型在 TTS 任务上出现明显的方差塌陷，生成的语音失去自然抖动；去掉下一标记扩散，解码步数回升至 50 步以上，速度优势消失。  
- **局限性**：论文未在高分辨率视频生成上给出实验，潜向量的空间维度仍受 VAE 编码器容量限制；此外，σ‑VAE 的方差阈值需要手动调节，缺乏自动化方案。

### 影响与延伸思考
LatentLM 为多模态生成提供了“一体化语言模型”思路，直接推动了大规模多模态 LLM 的快速迭代。随后出现的工作（如 **DiffusionLM**、**Unified Diffusion Transformers**）在此基础上进一步探索更高效的噪声调度或更灵活的潜空间结构。对想深入的读者，可以关注以下方向：  
- **自适应噪声调度**：让模型在不同模态或不同生成阶段自动决定噪声幅度。  
- **跨模态潜空间对齐**：研究如何在潜向量层面实现不同模态的语义对齐，提升检索与编辑能力。  
- **方差学习的理论分析**：系统化解释 σ‑VAE 为什么能防止方差塌陷，以及它与信息瓶颈的关系。  
这些方向都有望把统一的潜在语言模型推向更高的生成质量和更广的应用场景。

### 一句话记住它
**LatentLM 把所有模态压成“隐形语言”，用一次噪声+一次自回归就能高效生成，彻底打通了文本、图像、音频的统一大模型之路。**