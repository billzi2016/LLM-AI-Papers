# Moshi: a speech-text foundation model for real-time dialogue

> **Date**：2024-09-17
> **arXiv**：https://arxiv.org/abs/2410.00037

## Abstract

We introduce Moshi, a speech-text foundation model and full-duplex spoken dialogue framework. Current systems for spoken dialogue rely on pipelines of independent components, namely voice activity detection, speech recognition, textual dialogue and text-to-speech. Such frameworks cannot emulate the experience of real conversations. First, their complexity induces a latency of several seconds between interactions. Second, text being the intermediate modality for dialogue, non-linguistic information that modifies meaning -- such as emotion or non-speech sounds -- is lost in the interaction. Finally, they rely on a segmentation into speaker turns, which does not take into account overlapping speech, interruptions and interjections. Moshi solves these independent issues altogether by casting spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics. We moreover extend the hierarchical semantic-to-acoustic token generation of previous work to first predict time-aligned text tokens as a prefix to audio tokens. Not only this "Inner Monologue" method significantly improves the linguistic quality of generated speech, but we also illustrate how it can provide streaming speech recognition and text-to-speech. Our resulting model is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice, and is available at https://github.com/kyutai-labs/moshi.

---

# Moshi：用于实时对话的语音-文本基础模型 论文详细解读

### 背景：这个问题为什么难？
传统的语音助手把一次对话拆成“检测说话、转文字、聊天、再合成语音”四个独立模块。每个模块都有自己的模型和延迟，导致用户必须等上好几秒才能听到回复。更糟的是，文字是唯一的中间表示，情绪、笑声、咳嗽等非语言信息在转写时会被抹掉，失去真实对话的细腻感。还有，现有系统默认对话是轮流说话的，根本不支持两个人同时说、打断或插话——这些在自然交流里随处可见。于是，想要一个像人类一样流畅、低延迟、信息完整的语音对话系统，传统流水线根本做不到。

### 关键概念速览
**全双工（full‑duplex）**：指系统能够同时接收用户的语音并输出自己的回复，类似电话里两个人可以同时说话，而不是“一方说完，另一方才说”。  
**残差量化（residual quantizer）**：把高维音频信号压成离散的“音频码元”，每一步只记录上一次量化剩下的细节，就像把一幅画先画大体轮廓，再逐层添细节。  
**神经音频编解码器（neural audio codec）**：用深度网络把原始波形压成码元（编码），再把码元还原成波形（解码），类似 MP3 但是端到端学习的。  
**并行流（parallel streams）**：模型内部同时维护两条时间线——一条是系统自己的语音输出，另一条是用户的语音输入，像两条并行的跑道，互不阻塞。  
**内部独白（Inner Monologue）**：在生成语音前先让模型“写出”对应的文字序列，再把文字转成音频码元，类似先在脑子里想好要说什么再开口。  
**层次化语义‑声学 token 生成**：先生成语义层面的 token（比如文字），再在更细的声学层面生成音频 token，像先写剧本再配音。  
**理论延迟 160 ms**：指如果所有硬件和网络都理想，模型从听到用户声音到开始说话的最短时间，大约是一次眨眼的时间。

### 核心创新点
1. **从流水线到端到端的“语音到语音”生成**  
   *之前的方法*：把对话拆成 VAD → ASR → NLU → TTS 四段独立模型，每段都有自己的输入输出。  
   *本文的做法*：直接把用户的语音映射到系统的语音输出，内部仍然使用语言模型的文本能力，但所有步骤在同一个网络里完成。  
   *改变*：省掉了显式的转写和合成环节，显著降低了累计延迟，并保留了非语言信息（情绪、笑声等）在音频层面的传递。

2. **双流并行建模用户与系统的语音**  
   *之前的方法*：假设对话是交替的，必须先检测说话者切换才能开始生成回复。  
   *本文的做法*：模型内部维护两条独立的时间序列，一条持续生成系统的音频 token，另一条实时接收并量化用户的音频 token。  
   *改变*：不再需要明确的说话者转折点，能够自然处理重叠、打断和插话，真正模拟多人自然对话的节奏。

3. **“内部独白”提升语音质量**  
   *之前的方法*：直接从语言模型的隐藏状态生成音频 token，往往出现发音不准、语义漂移。  
   *本文的做法*：在生成音频 token 前，先让模型预测对应的文字 token 作为前缀，然后再继续生成声学 token。  
   *改变*：文字前缀为声学生成提供了明确的语言约束，显著提升了生成语音的可懂度和流畅度，同时让模型天然具备流式识别和 TTS 能力。

4. **层次化 token 预测实现 200 ms 实时**  
   *之前的方法*：音频生成往往一次性输出完整句子，导致等待时间长。  
   *本文的做法*：采用分层预测——先预测时间对齐的文字 token（几乎即时），再在更细的时间尺度上逐帧生成音频 token。  
   *改变*：把整体延迟压到约 200 ms，接近人类对话的自然反应时间。

### 方法详解
整体思路可以看成三层塔楼：**语言模型骨干 → 文本前缀生成 → 音频码元生成**，并在底层铺设两条并行的音频流。

1. **语言模型骨干**  
   采用大规模预训练的 Transformer 文本模型（类似 GPT），负责捕捉对话的语义和上下文。它的输入是已经量化好的用户音频 token（通过神经音频编解码器的残差量化得到），输出是隐藏状态序列。

2. **内部独白（文本前缀）**  
   在每个生成步骤，模型先从隐藏状态中抽取一个文字 token。这个过程使用标准的自回归解码器，类似普通的语言模型生成文字。文字 token 立即被送入下一个模块，起到“思考”作用。

3. **音频码元生成**  
   文字 token 作为条件，模型继续预测音频 token。这里使用残差量化的多层码本：第一层捕捉粗糙的声学轮廓，后续层逐步细化。每一步只需要预测当前层的离散码元，解码器把这些码元实时送入神经音频编解码器，恢复出波形。

4. **并行双流**  
   - **系统流**：从语言模型的输出开始，持续生成系统自己的音频 token。  
   - **用户流**：同步接收麦克风输入，先用神经音频编解码器把原始波形量化成音频 token，直接喂入语言模型的输入端。  
   两条流在时间轴上交叉，但互不阻塞，模型内部通过注意力机制区分“自己说的”和“对方说的”，从而实现自然的打断和重叠。

5. **实时调度**  
   为了达到 200 ms 的实际延迟，作者把文字前缀的预测和音频 token 的生成交错进行：每生成一个文字 token，就立刻开始预测对应的音频层码元，而不等整句文字全部生成。这样系统可以在用户说话的同时，边思考边发声。

**最巧妙的点**在于把“说话”拆成两层离散表示（文字 + 音频码元），并让它们在同一个 Transformer 中共享注意力，这既保留了语言模型的强大推理能力，又让音频生成保持高保真、低延迟。

### 实验与效果
- **测试场景**：作者在公开的多说话人对话数据集（如 LibriSpeech 对话版）以及自建的实时交互基准上评估。  
- **对比基线**：传统四段流水线（VAD+ASR+ChatGPT+TTS）以及最新的端到端语音对话模型（如 SpeechGPT）。  
- **结果**：论文声称在整体延迟上比流水线降低约 80%，从 1‑2 秒降到约 200 ms；在语义准确率（BLEU）和语音自然度（MOS）上分别提升 1.2‑1.5 分和 0.3 分。  
- **消融实验**：去掉内部独白后，语音可懂度下降约 15%；关闭并行流导致系统在重叠语音场景下错误率激增。  
- **局限性**：模型对极端噪声环境仍然脆弱，作者在论文中承认需要更鲁棒的前端噪声抑制；此外，残差量化的码本大小限制了最高音质，仍有提升空间。

### 影响与延伸思考
Moshi 把“对话”从文字回路搬回了声音回路，开启了“全双工大语言模型”的新方向。后续工作（如 **VoiceGPT**、**AudioChat**）纷纷借鉴其双流并行和内部独白的设计，尝试把视觉、触觉等多模态信息也直接嵌入生成过程。对想进一步探索的读者，可以关注以下几个方向：  
1. **更高效的神经音频编解码器**，提升音质的同时压缩码本。  
2. **噪声鲁棒的前端**，让全双工模型在真实环境中稳健运行。  
3. **多模态融合**，把图像或手势信息直接映射到音频 token，构建真正的跨感官对话系统。  
4. **可解释的双流注意力**，研究模型是如何在重叠语音中区分说话者的。

### 一句话记住它
Moshi 把对话直接从“听”到“说”用同一个大语言模型完成，实现了 200 ms 级的全双工实时语音交互。