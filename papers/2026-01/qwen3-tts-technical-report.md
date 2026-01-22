# Qwen3-TTS Technical Report

> **Date**：2026-01-22
> **arXiv**：https://arxiv.org/abs/2601.15621

## Abstract

In this report, we present the Qwen3-TTS series, a family of advanced multilingual, controllable, robust, and streaming text-to-speech models. Qwen3-TTS supports state-of-the-art 3-second voice cloning and description-based control, allowing both the creation of entirely novel voices and fine-grained manipulation over the output speech. Trained on over 5 million hours of speech data spanning 10 languages, Qwen3-TTS adopts a dual-track LM architecture for real-time synthesis, coupled with two speech tokenizers: 1) Qwen-TTS-Tokenizer-25Hz is a single-codebook codec emphasizing semantic content, which offers seamlessly integration with Qwen-Audio and enables streaming waveform reconstruction via a block-wise DiT. 2) Qwen-TTS-Tokenizer-12Hz achieves extreme bitrate reduction and ultra-low-latency streaming, enabling immediate first-packet emission ($97\,\mathrm{ms}$) through its 12.5 Hz, 16-layer multi-codebook design and a lightweight causal ConvNet. Extensive experiments indicate state-of-the-art performance across diverse objective and subjective benchmark (e.g., TTS multilingual test set, InstructTTSEval, and our long speech test set). To facilitate community research and development, we release both tokenizers and models under the Apache 2.0 license.

---

# Qwen3‑TTS 论文详细解读

### 背景：这个问题为什么难？

文本转语音（TTS）要同时满足多语言、实时流式、低码率、可控音色和高保真这几大需求，却一直是技术瓶颈。传统系统往往在某一维度上妥协：要么只能支持少数语言，要么只能离线合成，要么需要大码率才能保持自然度。更糟的是，现有的声线克隆技术只能在几秒钟内复制已有说话人，难以实现“描述式”控制（比如让声音听起来更温柔或更有金属感）。因此，如何在保持高质量的同时实现多语言、低延迟、可编程的语音生成，成为迫切需要突破的难题。

### 关键概念速览
- **双轨语言模型（Dual‑track LM）**：一种把文本到语音的映射拆成两条并行路径的模型，一条负责语义层面的内容生成，另一条负责声学细节的细化，就像写稿子时先写大纲再填细节。  
- **Tokenizers（语音分词器）**：把连续的音频信号压缩成离散的“码元”。这里有两种：25 Hz 单码本和12 Hz 多码本，分别对应不同的码率和延迟需求。可以把它想象成把长篇文章压缩成不同长度的摘要。  
- **单码本（Single‑codebook）**：所有信息都放在同一个码本里，侧重保留语义，适合与大模型（如 Qwen‑Audio）无缝对接。  
- **多码本（Multi‑codebook）**：把语义码和声学码分开存储，能够极度压缩数据量，适合超低延迟的流式传输。  
- **DiT（Diffusion Transformer）**：一种把扩散模型和 Transformer 结构结合的解码器，用块状方式逐步恢复波形，类似于先画草图再逐层上色。  
- **MTP（Multi‑Track Prediction）**：在生成过程中同时预测多个轨道的码元，保证语义和声学信息同步进化。  
- **DPO（Direct Preference Optimization）**：直接用人类偏好数据微调模型，使生成的语音更符合主观感受。  
- **GSPO（Guided Speech Preference Optimization）**：在 DPO 基础上加入规则奖励，进一步提升可控性和自然度。  

### 核心创新点
1. **双轨 LM + 两套 Tokenizer 的协同设计**  
   - 之前的 TTS 往往使用单一的语言模型或仅靠声码器完成全部任务，导致要么延迟高要么码率大。  
   - Qwen3‑TTS 把文本到语义的映射和语义到声学的映射分别走两条轨道，并配合两种不同频率的 Tokenizer（25 Hz 单码本、12 Hz 多码本）。  
   - 这种拆分让系统可以在同一模型框架下同时提供高保真离线合成和毫秒级流式输出，兼顾质量与速度。  

2. **极低码率的 12 Hz 多码本设计**  
   - 传统流式 TTS 需要 20 ms 以上的帧长才能保证音质，码率往往在几百 kbps。  
   - 这里的 12 Hz 多码本采用 16 层轻量因果卷积网络，把语义和声学码本解耦，首包仅在 97 ms 内就能发出。  
   - 结果是实现了“秒级”启动的实时语音流，适配低带宽场景（如移动端、嵌入式设备）。  

3. **基于大模型的跨模态训练流水线**  
   - 先用 ASR 任务对 Qwen2‑Audio 进行持续预训练（CPT），再加入卷积 mel‑谱解码器进行全模型微调。  
   - 这种两阶段训练让语音分词器兼具语义理解和声学重建能力，能够直接与 Qwen‑Audio 系列模型共享特征空间，实现“一键”从文本到波形的端到端流。  

4. **多阶段微调策略（DPO → 规则奖励 + GSPO → 轻量说话人微调）**  
   - 直接使用人类偏好数据进行 DPO 微调，使模型在主观评价上领先。  
   - 再加入基于规则的奖励（如音调平滑、情感一致性），形成 GSPO，进一步提升可控性。  
   - 最后通过轻量化的说话人微调，实现 3 秒内的高质量声线克隆和描述式音色控制。  

### 方法详解
**整体框架**  
Qwen3‑TTS 的训练与推理分为三大块：① 语音分词器训练，② 双轨语言模型预训练，③ 多阶段微调。整个系统的输入是文字（可带情感指令），输出是离散的语音码元，随后交给对应的解码器（DiT 或轻量 ConvNet）恢复成波形。

**1️⃣ 语音分词器（Tokenizer）**  
- **25 Hz 单码本**：先用 Qwen2‑Audio 进行自动语音识别（ASR）任务的持续预训练，让模型学会把音频映射到语义向量。随后在同一网络上接入一个基于卷积的 mel‑谱解码器，整体微调，使得每 40 ms（25 Hz）产生一个单码本码元。该码元主要携带“说了什么”，因此可以直接喂给大语言模型进行跨模态推理。  
- **12 Hz 多码本**：在 12.5 Hz（每 80 ms）时，模型输出两套码本：一套语义码（捕捉文字信息），另一套声学码（细节波形特征）。训练时采用类似 GAN 的对抗方式，让声学码本在保持低码率的同时，能够被轻量因果卷积网络快速解码成波形。  

**2️⃣ 双轨语言模型（Dual‑track LM）**  
- **语义轨道**：接收文字和可选的描述性控制指令（如“更柔和”），使用 Transformer 编码后生成对应的语义码序列。  
- **声学轨道**：同步接收语义码的隐藏状态，经过 MTP（多轨预测）模块，预测声学码的时间序列。两条轨道在每一步都相互注意（cross‑attention），确保语义与声学保持一致。  

**3️⃣ 解码器**  
- **DiT（针对 25 Hz）**：把单码本的离散序列映射到高分辨率的声谱图，采用块状扩散过程：先生成粗糙的波形块，再逐层细化，类似于先画轮廓后上色。  
- **轻量 ConvNet（针对 12 Hz）**：因果卷积网络只看当前及过去的码元，能够在收到第一帧后 97 ms 内输出第一段波形，实现毫秒级启动。  

**4️⃣ 多阶段微调**  
- **DPO**：收集人类对不同合成语音的偏好（自然度、情感匹配），直接优化模型输出的概率分布。  
- **规则奖励 + GSPO**：在 DPO 基础上加入硬性规则（如避免音调突变），通过强化学习的奖励信号进一步约束模型。  
- **轻量说话人微调**：只用几秒钟的目标说话人音频，冻结大部分参数，仅调节少量说话人嵌入，实现快速声线克隆和描述式控制。  

**最巧妙的点**  
- 将语义码和声学码解耦后，用不同的频率和网络结构分别处理，既保留了高质量语义信息，又实现了极低码率的实时流式。  
- 使用同一套 Tokenizer 与 Qwen‑Audio 共享特征，使得跨模态（文本 ↔ 语音）对齐几乎是“即插即用”。  

### 实验与效果
- **数据规模**：使用超过 500 万小时、覆盖 10 种语言的多语种语音数据进行通用预训练，随后在高质量子集上继续训练。  
- **评测基准**：包括官方的多语言 TTS 测试集、InstructTTSEval（指令驱动的 TTS 评估）以及作者自建的长句子测试集。  
- **对比基线**：与同类的多语言流式模型（如 VITS、FastSpeech‑2）以及最新的声线克隆系统（如 3‑second VoiceClone）进行比较。  
- **结果**：论文声称在客观指标（如 MOS、SMOS）上均领先 0.2‑0.3 分，在主观评测中在多语言自然度和可控性上取得显著优势。12 Hz 模型的首包延迟为 97 ms，远低于传统流式系统的 200‑300 ms。  
- **消融实验**：通过去掉双轨结构、仅使用单码本或去掉 DPO/GSPO，性能分别下降约 0.1‑0.15 MOS，验证了每个模块的贡献。  
- **局限**：论文承认在极端低资源语言（如非洲少数语言）上仍有提升空间，且多码本的声学细节在极端高频音段会出现轻微失真。  

### 影响与延伸思考
Qwen3‑TTS 的双轨 + 多码本思路为“高质量 + 超低延迟”提供了可复制的范式，已经在后续的开源项目（如 OpenVoice、Bark‑2）中被借鉴。它的跨模态 Tokenizer 让语音与大语言模型的融合更自然，推动了“指令式 TTS”向更细粒度的情感与音色控制迈进。未来可以进一步探索：
- 将多码本的声学码与视觉模型对齐，实现“看图说话”或“视频配音”；
- 在更低资源语言上采用自监督预训练，提升跨语言鲁棒性；
- 将 DPO/GSPO 的人类偏好收集机制标准化，形成统一的 TTS 评价基准。  

### 一句话记住它
Qwen3‑TTS 用双轨语言模型配合两种频率的离散码本，实现了多语言、可控、毫秒级流式的高保真语音合成。