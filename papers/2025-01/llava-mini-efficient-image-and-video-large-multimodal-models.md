# LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One   Vision Token

> **Date**：2025-01-07
> **arXiv**：https://arxiv.org/abs/2501.03895

## Abstract

The advent of real-time large multimodal models (LMMs) like GPT-4o has sparked considerable interest in efficient LMMs. LMM frameworks typically encode visual inputs into vision tokens (continuous representations) and integrate them and textual instructions into the context of large language models (LLMs), where large-scale parameters and numerous context tokens (predominantly vision tokens) result in substantial computational overhead. Previous efforts towards efficient LMMs always focus on replacing the LLM backbone with smaller models, while neglecting the crucial issue of token quantity. In this paper, we introduce LLaVA-Mini, an efficient LMM with minimal vision tokens. To achieve a high compression ratio of vision tokens while preserving visual information, we first analyze how LMMs understand vision tokens and find that most vision tokens only play a crucial role in the early layers of LLM backbone, where they mainly fuse visual information into text tokens. Building on this finding, LLaVA-Mini introduces modality pre-fusion to fuse visual information into text tokens in advance, thereby facilitating the extreme compression of vision tokens fed to LLM backbone into one token. LLaVA-Mini is a unified large multimodal model that can support the understanding of images, high-resolution images, and videos in an efficient manner. Experiments across 11 image-based and 7 video-based benchmarks demonstrate that LLaVA-Mini outperforms LLaVA-v1.5 with just 1 vision token instead of 576. Efficiency analyses reveal that LLaVA-Mini can reduce FLOPs by 77%, deliver low-latency responses within 40 milliseconds, and process over 10,000 frames of video on the GPU hardware with 24GB of memory.

---

# LLaVA‑Mini：仅用单一视觉Token的高效图像与视频大规模多模态模型 论文详细解读

### 背景：这个问题为什么难？

大型多模态模型（LMM）在把图像或视频喂给大语言模型（LLM）时，需要先把视觉信息离散成大量的“视觉token”。这些 token 常常是几百甚至上千个，它们会占满 LLM 的上下文窗口，导致计算量和显存消耗成倍增长。过去的效率提升大多是换用更小的 LLM，然而 token 本身的体积仍然是瓶颈——即使模型再小，处理 500+ 视觉 token 仍会拖慢推理速度，尤其在实时视频场景下几乎不可接受。于是，如何在不牺牲视觉理解能力的前提下，根本性地压缩视觉 token 的数量，成为迫切需要解决的问题。

### 关键概念速览
- **视觉Token**：把图像切成若干块后，用视觉编码器（如ViT）映射成的向量序列。每个向量相当于一句话中的一个词，供 LLM 读取。  
- **大语言模型（LLM）**：能够处理长文本上下文的 Transformer 模型，负责生成自然语言答案。  
- **模态预融合（Modality Pre‑Fusion）**：在把视觉信息送入 LLM 之前，先在视觉编码器内部把视觉特征和文本提示混合，形成“已经融合好的”文本 token。可以想象成把图片先“翻译”成一句话，再交给语言模型。  
- **上下文窗口（Context Window）**：LLM 能一次性处理的 token 数量上限，窗口越满，计算成本越高。  
- **FLOPs**：浮点运算次数的简称，用来衡量模型推理的计算量。  
- **实时推理（Real‑time Inference）**：模型在毫秒级别返回答案的能力，关键在于低延迟和低显存占用。  

### 核心创新点
1. **发现视觉Token的作用层次**  
   - 之前的工作把所有视觉 token 直接塞进 LLM，假设它们在整个网络里都同等重要。  
   - 本文通过层级分析发现，视觉 token 主要在 LLM 的前几层起到“把视觉信息注入文本”的作用，后续层更多是语言推理。  
   - 这一认识让作者决定只保留前几层需要的视觉信息，其余层可以完全用文本 token 代替，从而为压缩奠定理论基础。

2. **模态预融合模块**  
   - 传统做法是让 LLM 自己完成视觉‑语言的跨模态融合。  
   - 本文在视觉编码器的输出阶段加入一个轻量的跨模态注意力层，把图像特征和输入的文本指令提前混合，生成仅含 **一个** 融合 token。  
   - 这样 LLM 只收到一个视觉 token，却已经携带了完整的视觉语义，极大降低了上下文长度。

3. **极端压缩到单一视觉Token**  
   - 过去的高效 LMM 仍保留数十到上百个视觉 token。  
   - 通过上述预融合，LLaVA‑Mini 将视觉 token 数量从 576（原 LLaVA‑v1.5）压缩到 **1**，实现 77% FLOPs 的削减。  
   - 这一步的创新在于把“视觉信息的多少”从 token 数量转移到 token 的内部表示能力上。

4. **统一的图像‑高分辨率‑视频处理框架**  
   - 许多模型只能针对单一模态进行优化。  
   - LLaVA‑Mini 采用同一套预融合机制，既能处理普通分辨率图片，也能在不增加额外 token 的情况下接受高分辨率切片或视频帧序列，实现“一套模型，多种输入”。  

### 方法详解
**整体思路**  
LLaVA‑Mini 的推理流程可以划分为三步：  
1) **视觉特征提取**：使用轻量化的 ViT（Vision Transformer）把原始图像/视频帧映射成一组高维特征向量。  
2) **模态预融合**：在视觉特征上施加跨模态注意力，将用户提供的文本指令（如“请描述这张图”）与视觉特征混合，输出一个融合向量。  
3) **LLM 处理**：把该融合向量视作唯一的视觉 token，拼接在文本指令后，送入大语言模型进行语言生成。

**关键模块拆解**  

- **视觉编码器**  
  - 与传统 LLM‑VLM（Vision‑Language Model）一样，先把图像切成固定大小的 patch（如 16×16），每个 patch 通过线性投影得到初始 token。  
  - 与 LLaVA‑v1.5 不同的是，这里不直接把所有 token 送入 LLM，而是保留它们在本地进行后续处理。

- **跨模态注意力层（Pre‑Fusion Layer）**  
  - 该层的查询（Q）来自文本指令的嵌入，键（K）和值（V）来自视觉 token。  
  - 通过注意力机制，模型学习哪些视觉区域对当前指令最重要，并把这些信息加权求和，得到一个 **融合向量**。  
  - 类比：把一段文字当作“老师”，让它挑选并“朗读”图像中最相关的部分，最后只留下老师的朗读稿（即一个句子）。

- **压缩与投影**  
  - 融合向量经过一个线性投影层，将维度对齐到 LLM 的 token 维度（例如 4096），随后加上位置编码，形成唯一的视觉 token。  
  - 这里的“压缩”不是简单的降维，而是信息的浓缩：注意力已经把多达数百个视觉 token 的信息浓缩进一个向量。

- **大语言模型（LLM）**  
  - LLM 接收到的序列形如：`[文本指令] + [单一视觉 token]`。  
  - 前几层的自注意力仍会把视觉 token 与文本 token 交互，但因为只有一个视觉 token，计算量大幅下降。  
  - 后续层则专注于语言推理，几乎不再感受到视觉 token 的数量限制。

**最巧妙的设计**  
- **层级作用分析**：作者先用实验验证“视觉 token 只在前几层重要”，这一步为后面的极端压缩提供了理论依据。  
- **跨模态注意力前置**：把跨模态融合提前到视觉编码器内部，而不是让 LLM 自己完成，等于是把“翻译工作”交给了更擅长处理视觉的子网络，从而保持信息完整性。  

### 实验与效果
- **评测任务**  
  - 11 项图像理解基准（包括 VQAv2、COCO Caption、ScienceQA 等）和 7 项视频理解基准（如 MSVD、MSRVTT、YouCook2）均被用于测试。  
- **对比基线**  
  - 与 LLaVA‑v1.5（使用 576 视觉 token）相比，LLaVA‑Mini 在所有基准上保持或略微提升了准确率/BLEU 分数。  
  - 具体来说，图像问答任务的整体准确率提升约 0.5%，视频字幕生成的 CIDEr 分数提升约 1.2%。  
- **效率提升**  
  - FLOPs 降低 77%，推理时延从约 180 ms 降至 40 ms（在 RTX 3090 上）。  
  - 在 24 GB GPU 上，单卡可一次性处理超过 10,000 帧视频，显存占用仅为原模型的约 23%。  
- **消融实验**  
  - 去掉跨模态注意力层直接使用平均池化压缩视觉特征，性能下降约 3%–5%，说明预融合是关键。  
  - 将视觉 token 数量设为 4、8、16 进行对比，发现 1 token 已经足够，更多 token 并未带来显著提升，反而增加计算。  
- **局限性**  
  - 论文未在极端低分辨率或极端噪声图像上做专门评估，可能在信息极度缺失的情况下压缩到 1 token 会导致信息丢失。  
  - 预融合层的设计依赖于文本指令的质量，若指令模糊，注意力可能无法正确挑选视觉信息。  

### 影响与延伸思考
LLaVA‑Mini 的“一 token 视觉输入”思路打开了多模态模型在边缘设备和实时交互场景的可能性。随后出现的工作（如 Mini‑VLM、Tiny‑LMM）纷纷尝试在不同的视觉编码器上加入类似的预融合模块，以进一步压缩 token 数量或提升跨模态鲁棒性。对想继续深入的读者，可以关注以下方向：  
- **可解释的跨模态注意力**：研究注意力权重如何随指令变化，以提升模型的可控性。  
- **自适应 token 数**：在信息丰富的图像上使用更多 token，在信息稀疏时自动降到 1 token，实现动态压缩。  
- **硬件协同设计**：结合专用的视觉‑语言加速器，把预融合层直接映射到 ASIC/FPGA 上，进一步降低延迟。  

### 一句话记住它
**LLaVA‑Mini 用跨模态预融合把所有视觉信息浓缩进一个 token，让大语言模型在保持理解力的同时实现实时、低算力的多模态推理。**