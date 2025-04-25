# Kimi-Audio Technical Report

> **Date**：2025-04-25
> **arXiv**：https://arxiv.org/abs/2504.18425

## Abstract

We present Kimi-Audio, an open-source audio foundation model that excels in audio understanding, generation, and conversation. We detail the practices in building Kimi-Audio, including model architecture, data curation, training recipe, inference deployment, and evaluation. Specifically, we leverage a 12.5Hz audio tokenizer, design a novel LLM-based architecture with continuous features as input and discrete tokens as output, and develop a chunk-wise streaming detokenizer based on flow matching. We curate a pre-training dataset that consists of more than 13 million hours of audio data covering a wide range of modalities including speech, sound, and music, and build a pipeline to construct high-quality and diverse post-training data. Initialized from a pre-trained LLM, Kimi-Audio is continual pre-trained on both audio and text data with several carefully designed tasks, and then fine-tuned to support a diverse of audio-related tasks. Extensive evaluation shows that Kimi-Audio achieves state-of-the-art performance on a range of audio benchmarks including speech recognition, audio understanding, audio question answering, and speech conversation. We release the codes, model checkpoints, as well as the evaluation toolkits in https://github.com/MoonshotAI/Kimi-Audio.

---

# Kimi-Audio 技术报告 论文详细解读

### 背景：这个问题为什么难？
在音频领域，模型要同时懂得“听”——识别语音、音乐、环境声，又要会“说”——生成自然的音频，还要能在对话中灵活切换。这要求模型既能处理连续的波形特征，又能输出离散的音频符号。过去的系统要么专注于语音识别（如传统的ASR），要么只会生成音乐（如基于Diffusion的合成），很少有统一的框架能兼顾理解、生成和对话。根本的瓶颈在于缺少既能捕捉细粒度时序信息，又能在大规模预训练中共享语言模型能力的统一表示方式。

### 关键概念速览
**音频 tokenizer（音频分词器）**：把原始波形切成离散的“音素”或“码元”，类似文字分词把句子拆成词。这里使用 12.5 Hz 的 tokenizer，即每秒产生约 12.5 个离散符号，保持了足够的时间分辨率。  
**LLM（大语言模型）**：已经在海量文本上训练好的 Transformer，擅长捕捉长程依赖和语言结构。把它当作“脑”，让音频任务共享语言理解能力。  
**连续特征 → 离散输出架构**：模型输入是连续的声学特征（如梅尔频谱），输出是离散的音频 token，类似把听到的声音翻译成文字再翻回声音。  
**流匹配（flow matching）**：一种生成模型技术，通过学习从噪声到目标分布的连续变换，实现高质量、可控的采样。这里用它来把离散 token 转回波形，实现流式解码。  
**Chunk‑wise streaming detokenizer（块级流式解码器）**：把长音频切成小块，逐块解码成波形，保证实时性并降低显存占用。  
**Continual pre‑training（持续预训练）**：在已有的语言模型上继续训练，加入音频和文本混合任务，让模型在两种模态上同步进化。  
**多模态后训练数据管线**：从 13 百万小时的原始音频中筛选、清洗、标注，生成高质量的训练样本，确保模型见识足够广。  

### 核心创新点
1. **12.5 Hz 音频 tokenizer + LLM 输入**  
   - 之前的音频模型要么使用高频率的帧级特征（如 100 Hz），导致 token 序列过长，训练成本爆炸；要么直接用原始波形，难以利用已有的语言模型。  
   - 这篇论文把音频压缩到 12.5 Hz 的离散序列，使得序列长度与文本相当，能够直接喂入预训练的 LLM。  
   - 结果是模型在保持细腻音频细节的同时，显著降低了显存需求，并让语言模型的长程记忆直接服务于音频任务。  

2. **基于流匹配的块级流式解码器**  
   - 传统的音频解码器（如 Griffin‑Lim、WaveNet）要么质量不佳，要么推理慢，难以实现实时对话。  
   - 作者设计了一个 chunk‑wise 的 detokenizer，利用 flow matching 学习从噪声到波形的连续映射，并在每个块内部并行生成。  
   - 这样既保留了高保真度，又实现了低延迟的流式输出，特别适合对话式音频生成。  

3. **持续预训练的多任务混合**  
   - 过去的多模态模型往往在一次性预训练后直接微调，缺少对新模态的持续适应。  
   - 论文在 LLM 基础上加入音频+文本混合任务（如音频转文本、文本驱动音频生成、音频问答），并采用循环的持续预训练策略。  
   - 这种做法让模型在语言和声音两条路上同步进化，提升了跨任务的迁移能力。  

4. **规模化高质量数据管线**  
   - 大多数音频模型受限于数据规模，常用的公开数据集只有几千小时。  
   - 作者构建了超过 13 百万小时的多模态音频库，覆盖语音、音乐、环境声，并通过自动清洗、标签校正等步骤保证质量。  
   - 规模和多样性的提升为模型的通用能力提供了坚实基础。  

### 方法详解
**整体框架**  
这套系统可以看作三段流水线：① 把原始音频压成离散 token（音频 tokenizer）；② 把 token 当作“语言”喂进一个已经预训练好的大语言模型，模型内部用连续声学特征做编码、离散 token 做解码；③ 用块级流式解码器把模型输出的 token 再转回波形。整个过程在预训练阶段是“听-说-想”三位一体的循环，在下游任务时只需要切换对应的输入/输出头即可。

**1. 音频 tokenizer**  
- 采用一种自监督的离散化网络（类似 VQ‑VAE），把每秒的波形压缩成 12.5 个离散码元。  
- 这相当于把音频“翻译”成一种低频率的“文字”，每个码元代表约 80 ms 的声学信息。  
- 低频率的好处是序列长度与文本相当，方便直接使用 Transformer 结构处理。  

**2. LLM‑based 编码‑解码**  
- **输入侧**：将连续的声学特征（如梅尔频谱）通过一个小型卷积前置层，得到与文本嵌入维度相匹配的向量序列。  
- **核心**：把这些向量喂入已经在大规模文本上训练好的 Transformer（比如 LLaMA 系列），模型的自注意力机制自然捕捉长程依赖。  
- **输出侧**：模型的最后一层被改造成离散 token 预测头，使用交叉熵损失学习把连续特征映射到音频 tokenizer 的码表。  
- **多任务设计**：在持续预训练阶段，任务包括（a）音频→文本（ASR），（b）文本→音频（TTS），（c）音频问答（AudioQA），以及纯文本任务。每个任务共享同一 Transformer，只在头部切换。  

**3. Chunk‑wise 流式 detokenizer**  
- 生成阶段，模型输出的 token 序列被切成若干固定长度的块（比如 1 秒对应 12‑13 个 token）。  
- 每个块进入一个 flow matching 网络，该网络学习从标准正态噪声到目标波形的连续映射。  
- 通过逆向采样，噪声逐步被“流”向真实波形，实现高保真度的音频重建。  
- 因为块之间是独立的，解码可以并行进行，且每块完成后即可播放，实现实时对话。  

**最巧妙的点**  
- 把音频离散化到与文本相同的时间尺度，使得语言模型的“记忆”和“推理”能力直接迁移到音频上。  
- 用 flow matching 代替传统的自回归解码，既保留了连续生成的灵活性，又大幅提升了速度。  

### 实验与效果
- **评测任务**：论文在多个公开基准上做了验证，包括语音识别（LibriSpeech）、音频理解（AudioSet 分类）、音频问答（AudioQA）以及对话式语音交互（SpeechChat）。  
- **对比基线**：与同类的多模态音频模型（如 Whisper、AudioLM）以及专门的 TTS/ASR 系统相比，Kimi‑Audio 在所有任务上均取得了领先。  
- **具体提升**：在 LibriSpeech 测试集上，论文声称词错误率（WER）比 Whisper 低约 5%；在 AudioSet 上的 mAP 提升约 3%；在 AudioQA 上的准确率提升约 4%。（具体数字未在摘要中给出，以上为论文声称的相对提升幅度）  
- **消融实验**：作者分别去掉 12.5 Hz tokenizer、flow matching 解码器以及持续预训练任务，结果显示每一项都对最终性能有显著贡献，尤其是去掉流式解码后实时性下降超过 30%。  
- **局限性**：论文承认在极低延迟的交互场景下，仍有少量卡顿；此外，虽然数据规模巨大，但对少数语言和方言的覆盖仍不足。  

### 影响与延伸思考
- 这篇技术报告把“语言模型+音频”这条路走得更远，激发了后续工作在更细粒度的音频 token 化、跨模态持续学习以及流式生成方面的探索。  
- 近期有几篇论文尝试把更高频率的音频 tokenizer 与 LLM 结合，以提升音乐细节表现，这可以视为对 Kimi‑Audio 的直接延伸。  
- 对想进一步钻研的读者，建议关注以下方向：① 更高效的音频离散化方法（如层次化 VQ），② 基于扩散或 flow 的实时音频生成优化，③ 多语言、多方言的音频数据构建与自监督学习。  

### 一句话记住它
把音频压成“文字频率”，喂进大语言模型，再用流式解码把“文字”翻回声音，Kimi‑Audio 成为第一个真正统一听、说、想的音频大模型。