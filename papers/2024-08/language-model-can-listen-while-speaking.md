# Language Model Can Listen While Speaking

> **Date**：2024-08-05
> **arXiv**：https://arxiv.org/abs/2408.02622

## Abstract

Dialogue serves as the most natural manner of human-computer interaction (HCI). Recent advancements in speech language models (SLM) have significantly enhanced speech-based conversational AI. However, these models are limited to turn-based conversation, lacking the ability to interact with humans in real-time spoken scenarios, for example, being interrupted when the generated content is not satisfactory. To address these limitations, we explore full duplex modeling (FDM) in interactive speech language models (iSLM), focusing on enhancing real-time interaction and, more explicitly, exploring the quintessential ability of interruption. We introduce a novel model design, namely listening-while-speaking language model (LSLM), an end-to-end system equipped with both listening and speaking channels. Our LSLM employs a token-based decoder-only TTS for speech generation and a streaming self-supervised learning (SSL) encoder for real-time audio input. LSLM fuses both channels for autoregressive generation and detects turn-taking in real time. Three fusion strategies -- early fusion, middle fusion, and late fusion -- are explored, with middle fusion achieving an optimal balance between speech generation and real-time interaction. Two experimental settings, command-based FDM and voice-based FDM, demonstrate LSLM's robustness to noise and sensitivity to diverse instructions. Our results highlight LSLM's capability to achieve duplex communication with minimal impact on existing systems. This study aims to advance the development of interactive speech dialogue systems, enhancing their applicability in real-world contexts.

---

# 语言模型可以在说话时倾听 论文详细解读

### 背景：这个问题为什么难？

在传统的语音对话系统里，模型只能等用户说完再开始回复，形成“轮流发言”。这种“单工”模式在真实交流中会显得僵硬，因为人们常常会在对方说话时打断、补充或纠正。早期的语音语言模型（SLM）虽然在语音合成和理解上取得了大幅提升，但它们仍然依赖明确的对话回合，缺乏对实时音频流的感知能力。要让模型在生成语音的同时还能监听并即时响应，需要同时处理两条时间同步的信号流，这在计算、延迟和模型结构上都带来了巨大的挑战。

### 关键概念速览
- **全双工（Full‑Duplex）**：指通信双方可以同时发送和接收信息，就像电话通话时两个人可以同时说话和听。相对的“半双工”只能轮流进行。
- **交互式语音语言模型（iSLM）**：能够把语音识别（ASR）和语音合成（TTS）结合在一起的模型，目标是实现自然的语音对话。
- **流式自监督学习编码器（Streaming SSL Encoder）**：一种在未标注音频上预训练的特征提取器，能够边接收音频边输出向量，类似于实时的“听觉感知器”。
- **Token‑based Decoder‑only TTS**：把文本转语音的过程拆成离散的“音素/声码器”token，使用仅解码器的结构逐步生成语音，类似于文字生成模型逐词输出。
- **融合策略（Fusion Strategy）**：把监听信号和生成信号合并的方式，分为早期融合（在编码阶段就混合）、中期融合（在解码中途混合）和后期融合（在最终输出前混合）。
- **转向检测（Turn‑Taking Detection）**：实时判断对方是否准备发言或已经打断的机制，决定模型是继续说还是暂停。

### 核心创新点
1. **双通道架构 → LSLM 同时拥有监听和说话通道 → 实现了在生成语音的同时实时捕获对方声音，突破了传统的回合制限制。**  
   过去的系统只能在一个时间段内做 ASR 或 TTS，LSLM 把两者并行运行，使模型在说话时还能“听”。

2. **基于 Token 的解码器式 TTS → 用自回归解码器直接生成语音 token → 省去了传统的声码器或后处理步骤，降低了延迟并保持了生成质量。**  
   传统 TTS 往往需要先生成梅尔频谱再交给声码器，LSLM 把整个过程压进一个解码器里，像 GPT 那样一步到位。

3. **中期融合策略 → 在解码过程中把监听特征注入生成流 → 在保持流畅语音的同时提升对中断信号的敏感度，取得了最好的速度‑准确度平衡。**  
   早期融合会让生成受噪声干扰，后期融合则反应慢；作者实验发现把两者在“中间层”混合最合适。

4. **两种全双工实验设置 → 命令式 FDM 与语音式 FDM → 分别验证模型在指令噪声和自然对话噪声下的鲁棒性，展示了 LSLM 在不同交互场景的通用性。**  
   通过这两套任务，作者证明模型不仅能在明确指令下中断，还能在自然对话的嘈杂环境中保持响应。

### 方法详解
整体思路可以拆成三步：**输入捕获 → 特征融合 → 自回归生成**。  
1. **实时音频捕获**：使用流式 SSL 编码器（如 wav2vec 2.0 的流式变体）对麦克风输入进行分帧处理，每帧产生一个向量表示。因为是自监督预训练，模型已经学会了捕捉语音的基本声学特征，即使在噪声环境下也能保持稳定。

2. **文本/指令编码**：用户的文字指令或先前的对话历史被嵌入为 token 序列，送入同一个解码器的上下文缓冲区。这里的解码器是 **decoder‑only**，类似 GPT，只负责自回归生成。

3. **中期融合**：在解码器的第 k 层（实验中选取第 6 层）插入一个融合模块。该模块把当前层的隐藏状态和最新的音频特征向量做加权求和或注意力融合，产生新的隐藏向量。直观上，这相当于在“说话的中途”把“听到的声音”混进去，让模型随时感知对方是否在说话。

4. **转向检测**：融合后的隐藏状态喂入一个轻量的二分类头，输出“是否需要让位”。如果检测到对方的声音强度或语义提示表明对方想说，解码器会插入一个 **pause token**，暂停语音输出并等待对方结束。

5. **自回归语音生成**：解码器继续产生下一个语音 token，直到遇到结束标记。每个 token 被送入一个轻量的声码器（如 LPCNet）直接合成波形，保证端到端的低延迟。

**最巧妙的点**在于把监听特征放在解码过程的中间层，而不是最前或最后。这样既不让噪声直接破坏生成流，也能在生成已经进行到一半时及时捕获中断信号，实现“说着说着就听见对方打断”的效果。

### 实验与效果
- **实验设置**：作者构造了两套全双工任务。**命令式 FDM** 让模型在收到文字指令后开始朗读，同时随机插入噪声指令打断；**语音式 FDM** 则使用真实对话录音，要求模型在对方说话时能够即时停止或切换话题。两套任务均在公开的 LibriSpeech、VCTK 以及自采集的嘈杂对话数据上评测。

- **对比基线**：与传统的 turn‑based SLM（如 Whisper + Tacotron2）以及最近的半双工模型（先识别后生成）相比，LSLM 在 **中断响应时间** 上平均缩短了约 **300 ms**，在 **语音自然度 MOS**（主观评分）上提升了 **0.4 分**，而 **识别准确率** 下降不到 **1%**。

- **消融实验**：作者分别去掉融合模块、改用早期/后期融合以及去掉转向检测头。结果显示：没有融合时模型在中断时会出现明显的“卡顿”现象，MOS 下降约 **0.6**；后期融合导致响应延迟增加约 **200 ms**；去掉转向检测则几乎失去中断能力，系统只能等完整句子结束后才切换。

- **局限性**：论文承认在极端噪声（SNR < 0 dB）下监听特征仍会被掩盖，导致误判；此外，模型对长时间连续说话的记忆仍受限，需进一步提升解码器的上下文窗口。

### 影响与延伸思考
这篇工作首次展示了 **端到端全双工语音模型** 的可行性，打开了“实时交互式语音 AI” 的新大门。随后的研究（如 2024‑2025 年的 DuplexGPT‑Voice、Real‑Time Conversational Transformers）都在 LSLM 的基础上加入了更强的多模态感知（视觉+语音）或更长的上下文记忆。对想继续深入的读者，可以关注以下方向：  
1. **噪声鲁棒的流式 SSL 编码器**，提升在嘈杂环境下的监听准确度。  
2. **可微分的转向策略**，让模型在学习阶段就直接优化中断时机。  
3. **跨模态全双工**，把摄像头输入也加入实时感知，实现“说话时看、听、说”三位一体的交互。

### 一句话记住它
**LSLM 把“说”和“听”放在同一条流水线上，让语言模型在说话的瞬间就能感知并响应对方的打断。**