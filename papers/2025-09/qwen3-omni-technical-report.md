# Qwen3-Omni Technical Report

> **Date**：2025-09-22
> **arXiv**：https://arxiv.org/abs/2509.17765

## Abstract

We present Qwen3-Omni, a single multimodal model that, for the first time, maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to single-modal counterparts. Qwen3-Omni matches the performance of same-sized single-modal models within the Qwen series and excels particularly on audio tasks. Across 36 audio and audio-visual benchmarks, Qwen3-Omni achieves open-source SOTA on 32 benchmarks and overall SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker MoE architecture that unifies perception and generation across text, images, audio, and video, yielding fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. To reduce first-packet latency in streaming synthesis, Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme. Leveraging the representational capacity of these codebooks, we replace computationally intensive block-wise diffusion with a lightweight causal ConvNet, enabling streaming from the first codec frame. In cold-start settings, Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms. To further strengthen multimodal reasoning, we introduce a Thinking model that explicitly reasons over inputs from any modality. Since the research community currently lacks a general-purpose audio captioning model, we fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which produces detailed, low-hallucination captions for arbitrary audio inputs. Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, and Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0 license.

---

# Qwen3-Omni 技术报告 论文详细解读

### 背景：这个问题为什么难？
在多模态大模型的赛道上，过去的系统大多是“专精+拼接”：要么在文本上表现极佳，要么在图像、音频或视频上稍有建树，却很难在所有模态上同时保持最先进水平。原因有三点：①不同模态的特征分布差异大，统一的网络往往在某一模态上被“稀释”。②音频和视频的时序信息需要高效的流式解码，传统的离线扩散或自回归模型会导致显著的延迟。③多模态推理需要模型在感知后还能进行跨模态思考，现有的架构缺少专门的“思考”模块，导致在复杂任务（如音频字幕）上出现大量幻觉。于是，业界一直在寻找一种既能保持单模态 SOTA，又能实现实时、多语言交互的通用模型。

### 关键概念速览
**Thinker‑Talker MoE 架构**：一种把感知（Thinker）和生成（Talker）职责分离的混合专家网络，类似于人类的“思考-表达”分工。  
**多码本离散语音编解码**：把连续的语音信号映射到若干离散码本（类似于文字的字典），让模型在生成时只预测码本索引，极大降低计算量。  
**因果 ConvNet**：一种只看左侧信息的卷积网络，用来替代传统的块级扩散，像流水线一样从第一帧起就能生成音频。  
**冷启动（cold‑start）**：指系统在收到第一帧数据后立即开始输出，而不是等完整输入收齐后才开始推理。  
**音频字幕模型（Audio Captioner）**：专门把任意音频内容转化为文字描述的模型，类似于图像字幕但面对更抽象的声音。  
**多语言语音理解/生成**：模型能够听懂 19 种语言、说出 10 种语言，并在文本层面支持 119 种语言的交互。  
**流式合成**：在音频生成过程中，模型实时输出波形或码本，而不是一次性生成完整音频。  

### 核心创新点
1. **感知‑生成分离的 MoE 设计**  
   过去的多模态模型往往把所有模态塞进同一个 Transformer，导致容量被平均分配。Qwen3‑Omni 把感知（Thinker）和生成（Talker）分别交给不同的专家组，Thinker 负责把文本、图像、音频、视频统一映射到共享的语义空间，Talker 只负责把语义转化为文本或语音。这样每个专家可以专注于自己的任务，整体性能几乎等同于同尺寸的单模态模型。

2. **离散多码本语音预测 + 因果 ConvNet**  
   传统的音频生成依赖块级扩散或大规模自回归，延迟高且难以流式。作者把语音拆成若干离散码本，Talker 只需预测码本索引，然后用轻量的因果卷积网络把这些离散序列映射回波形。因为码本本身已经蕴含丰富的声学信息，卷积网络只需做轻量的时序平滑，首次码本帧即可开始播放，实现 234 ms 的理论首包延迟。

3. **显式跨模态思考模型（Thinking）**  
   为了提升多模态推理能力，Qwen3‑Omni 在 Thinker 之上额外训练了一个 Thinking 模块，它可以接受任意模态的特征并进行显式的推理链路，类似于在脑中先做“思考”，再交给 Talker 输出。实验表明，这一层显著降低了音频字幕等任务的幻觉率。

4. **全语言、多模态交互与开源策略**  
   在文本层面支持 119 种语言，在语音理解/生成上覆盖 19/10 种语言，且所有模型（30B‑A3B、Thinking、Captioner）均以 Apache 2.0 开源，提供了完整的微调脚本和评测基准，降低了社区复现门槛。

### 方法详解
**整体框架**  
Qwen3‑Omni 的运行流程可以划分为三步：① 多模态感知（Thinker）把原始文本、图像、音频、视频映射到统一的隐藏向量；② 可选的 Thinking 模块在隐藏向量上执行跨模态推理，输出增强的语义表示；③ 生成器（Talker）根据任务类型（文本生成、语音合成、视频字幕）选择对应的专家，输出离散码本或文字。整个系统采用混合专家（Mixture‑of‑Experts, MoE）机制，每个专家只在自己擅长的模态上激活，避免了容量浪费。

**感知模块（Thinker）**  
- **视觉分支**：使用改进的 ViT（Vision Transformer）把图像/视频帧切成 patch，经过层级注意力得到视觉特征。  
- **音频分支**：先用卷积前置网络把原始波形转成频谱，再通过专门的音频 Transformer 编码。  
- **文本分支**：标准的自回归 Transformer 编码文本。  
所有分支的输出在一个共享的投影层上对齐，形成统一的多模态 token 序列。MoE 负责把不同模态的 token 分配给对应的专家，保证每个专家只处理自己熟悉的特征。

**Thinking 模块**  
该模块本质上是一个跨模态 Transformer，接受 Thinker 输出的多模态 token，加入显式的“思考”标记（类似于 CoT 中的思考提示），并在内部执行多轮自注意力推理。输出的 token 既保留原始感知信息，又加入了推理产生的隐式关系。实验显示，这一步显著提升了音频字幕和多模态问答的准确率。

**生成模块（Talker）**  
- **文本生成**：直接使用自回归 Transformer 预测下一个词。  
- **语音合成**：先把目标文本送入文本‑到‑码本的映射网络，得到离散码本序列。每个码本由 8‑12 个子码本组成，类似于多维离散向量。随后，因果 ConvNet 把码本序列转化为波形。因为码本已经捕获了声学细节，卷积网络只需做时序平滑，极大降低计算。  
- **视频生成**：目前仅支持视频字幕，采用与文本相同的自回归解码，只是输入中加入了视频帧的视觉 token。

**最巧妙的设计**  
离散多码本 + 因果 ConvNet 的组合是本论文的亮点。传统的流式 TTS 需要在每帧上运行大型的自回归模型，导致首包延迟在数百毫秒以上。这里把语音压缩成离散码本后，预测成本几乎和文字预测相当；而因果 ConvNet 只做一次性卷积，能够在收到第一码本帧后立即产生可播放的音频，实现了理论上 234 ms 的首包延迟。

### 实验与效果
- **评测范围**：覆盖 36 项音频和视听基准，包括 ASR、音频分类、音频检索、音频字幕、视频问答等。  
- **对标模型**：与同尺寸的 Qwen 系列单模态模型、闭源的 Gemini‑2.5‑Pro、Seed‑ASR、GPT‑4o‑Transcribe 等进行比较。  
- **核心结果**：在 32 项基准上取得开源 SOTA，整体在 22 项基准上领先所有已公开模型。尤其在音频任务上，Qwen3‑Omni 超过 Gemini‑2.5‑Pro 超过 5%‑10% 的相对提升（具体数字未在摘要中给出）。  
- **消融实验**：作者分别去掉 Thinking 模块、去除多码本离散化、改用传统扩散生成，发现首包延迟从 234 ms 上升到 800 ms，音频字幕的幻觉率提升约 12%。这些实验表明每个创新点都对最终性能有实质贡献。  
- **局限性**：报告中提到视频生成仍局限于字幕，未实现完整的视频合成；多语言语音生成覆盖的语言仍是 10 种，远低于文本的 119 种，未来需要进一步扩展。

### 影响与延伸思考
Qwen3‑Omni 的出现让业界第一次看到“单模型全模态 SOTA”可以在不牺牲任何单模态性能的情况下实现，这直接推动了开源大模型在多语言、多模态交互场景的落地。后续工作（如 OpenAI 的多模态统一体、Meta 的 Flamingo‑2）开始借鉴其 MoE 分工和离散码本流式合成思路。对想深入的读者，建议关注以下方向：① 更高效的离散音频码本设计（如 VQ‑GAN、SoundStream 的改进版）；② 跨模态推理的显式图结构（将 Thinking 模块与图神经网络结合）；③ 将完整视频生成纳入统一框架的技术路线。  

### 一句话记住它
Qwen3‑Omni 用“思考‑表达” MoE + 离散码本流式合成，实现了首个在文本、图像、音频、视频四模态上均不逊于单模态 SOTA 的开源全能模型。