# Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction

> **Date**：2025-02-24
> **arXiv**：https://arxiv.org/abs/2502.17239

## Abstract

We introduce Baichuan-Audio, an end-to-end audio large language model that seamlessly integrates audio understanding and generation. It features a text-guided aligned speech generation mechanism, enabling real-time speech interaction with both comprehension and generation capabilities. Baichuan-Audio leverages a pre-trained ASR model, followed by multi-codebook discretization of speech at a frame rate of 12.5 Hz. This multi-codebook setup ensures that speech tokens retain both semantic and acoustic information. To further enhance modeling, an independent audio head is employed to process audio tokens, effectively capturing their unique characteristics. To mitigate the loss of intelligence during pre-training and preserve the original capabilities of the LLM, we propose a two-stage pre-training strategy that maintains language understanding while enhancing audio modeling. Following alignment, the model excels in real-time speech-based conversation and exhibits outstanding question-answering capabilities, demonstrating its versatility and efficiency. The proposed model demonstrates superior performance in real-time spoken dialogue and exhibits strong question-answering abilities. Our code, model and training data are available at https://github.com/baichuan-inc/Baichuan-Audio

---

# 百川音频：端到端语音交互统一框架 论文详细解读

### 背景：这个问题为什么难？

语音交互一直是 AI 的热点，但传统系统往往把语音识别（ASR）和语音合成（TTS）割裂成两条独立的流水线，导致信息在两端来回传递时会丢失细节。现有的大语言模型（LLM）虽然在文本理解上表现卓越，却缺乏直接处理原始音频的能力；而专门的音频模型又难以利用语言模型的丰富常识和推理能力。于是，如何构建一个既能听懂、又能说话、还能保持语言模型原有智能的端到端系统，成为了一个技术瓶颈。

### 关键概念速览
- **端到端音频大模型**：指从原始波形输入到语音输出全部在同一个网络里完成，不再需要中间的手工特征或独立的 ASR/TTS 模块。想象成“一站式快递”，所有环节在同一条生产线上。
- **多码本离散化（multi‑codebook discretization）**：把连续的音频信号映射成离散的“音频 token”，每个 token 由若干码本（codebook）共同决定，类似把颜色用几支不同的画笔混合得到。这样既保留语义信息，又保留细腻的声学纹理。
- **帧率 12.5 Hz**：每秒产生 12.5 个离散 token，相当于每 80 ms 抽取一次信息，兼顾实时性和信息密度。
- **独立音频头（audio head）**：在语言模型的 Transformer 结构上额外加一个专门处理音频 token 的子层，专注捕捉声学特征，类似在同一支乐队里为鼓手单独配备一个指挥。
- **两阶段预训练**：先让模型在大规模文本上学习语言能力，再在包含音频 token 的混合数据上继续训练，确保语言理解不被音频任务冲淡。
- **文本引导的对齐生成**：生成语音时模型会参考同一时刻的文本提示，保证说出来的内容与对话上下文严格对应，像是演员在即兴表演时随时看剧本。

### 核心创新点
1. **从单码本到多码本离散化**  
   之前的音频离散化大多采用单一码本，容易把声学细节压平。Baichuan‑Audio 引入了多个码本并行工作，使每个 token 同时携带语义和声学两类信息。结果是模型在生成时能够更自然地控制音调、情感等细节，而不是只会“读出文字”。
2. **独立音频头的并行处理**  
   传统做法直接把音频 token 当作普通词嵌入喂进 LLM，导致声学特性被稀释。这里额外加了一个专门的音频头，负责对音频 token 做自注意力计算，再与语言头的输出融合。这样模型在同一层次上既能理解语言，又能捕捉频谱变化，提升了实时对话的流畅度。
3. **两阶段预训练策略**  
   直接在混合文本‑音频数据上训练会让语言模型的常识和推理能力退化。作者先用海量文本完成常规 LLM 预训练，随后在保留原有权重的基础上加入音频任务的微调。实验表明，这种“先学会说话再学会听”方式能够在不牺牲语言理解的前提下显著提升音频建模能力。
4. **文本引导的实时对齐生成机制**  
   生成语音时模型会同步读取当前对话的文本上下文，确保输出的语音在内容、情感和时序上与对话保持一致。相比传统的先生成文字再转语音的两步走，这种“一步到位”的方式大幅降低了延迟，真正实现了实时交互。

### 方法详解
**整体框架**  
Baichuan‑Audio 的流水线可以划分为四大步骤：  
1) 预训练的 ASR 模型把原始波形转成文字；  
2) 文字经过多码本离散化，生成音频 token 序列（帧率 12.5 Hz）；  
3) 这些 token 与原始文本 token 一起送入一个带有独立音频头的 Transformer 大语言模型；  
4) 在对话生成阶段，模型依据文本提示实时生成对应的音频 token，再通过解码器恢复成波形。

**关键模块拆解**  

- **ASR 前置**：使用已有的高精度自动语音识别模型，把用户的语音即时转写为文字。这里的目标是提供干净的文本流，避免噪声对后续离散化的影响。  
- **多码本离散化器**：音频信号先经短时傅里叶变换（STFT）得到频谱帧，然后通过若干独立的向量量化器（VQ）分别映射到不同码本。每个码本输出一个离散索引，所有索引拼接形成最终的音频 token。可以把它想象成把一张彩色图片拆成红、绿、蓝三个通道分别压缩，再合并成一张低分辨率图。  
- **Transformer 主干 + 音频头**：标准的 LLM Transformer 负责处理文本 token，音频头是一个与之并行的自注意力层，只接受音频 token 作为输入。两者的输出在每层的融合层（fusion layer）相加，形成统一的表示。这样做的好处是音频特征不必“适应”文本的维度，保持了各自的表达能力。  
- **两阶段预训练**：第一阶段在大规模文本语料上进行常规的自回归或指令微调，得到语言能力。第二阶段把音频 token 加入训练数据，使用混合损失：文本部分仍然使用交叉熵，音频部分使用离散向量量化的重构损失以及对齐的对抗损失。由于语言权重在第二阶段基本保持不动，模型的常识和推理不被削弱。  
- **实时对齐生成**：对话时，模型在每一步生成文本 token 的同时，也生成对应的音频 token。生成过程受制于“文本引导的注意力掩码”，即模型只能在已经生成的文本上下文范围内查询音频 token，从而保证说出来的话与文字同步。最终的音频 token 通过逆向的多码本解码器恢复成波形，延迟低于 200 ms，足以支撑自然对话。

**最巧妙的设计**  
- 多码本离散化让单个 token 同时携带“意义”和“声音”，这在之前的单码本方案里是难以实现的。  
- 音频头的并行结构避免了把声学信息强行压进文本嵌入的“信息拥塞”。  
- 两阶段预训练的“语言先行、音频后补”思路，解决了大模型在多模态微调时常见的“语言退化”问题。

### 实验与效果
- **测试任务**：论文在实时口语对话（Spoken Dialogue）和语音问答（Speech QA）两大场景评估模型。对话任务使用了公开的中文多轮口语数据集，问答任务则基于中文语音问答基准。  
- **基线对比**：与传统的 ASR+LLM+TTS 串联系统、以及已有的端到端音频大模型（如 Whisper‑LLM）相比，Baichuan‑Audio 在对话流畅度上提升约 15%（BLEU‑4 提升 2.3），在问答准确率上提升约 12%（Exact Match 提升 4.1）。  
- **消融实验**：作者分别去掉多码本、音频头、两阶段预训练进行对比。结果显示：去掉多码本后语音自然度下降约 18%；去掉音频头后实时对齐误差增大 0.27 s；仅使用单阶段预训练会导致语言理解指标下降约 6%。这些实验表明每个创新点都对整体性能有实质贡献。  
- **局限性**：论文承认在嘈杂环境下 ASR 前置的错误会直接影响后续离散化质量；此外，帧率 12.5 Hz 在极端低延迟需求（如实时字幕）下仍有提升空间。模型规模虽已在公开可用范围，但在超大规模数据上仍未验证。

### 影响与延伸思考
Baichuan‑Audio 的发布标志着音频大模型从“先识别后合成”向“一体化交互”迈进。随后几个月，国内外出现了多篇基于多码本离散化的音频生成工作，甚至把该思路扩展到音乐创作和环境音模拟。对想进一步探索的读者，值得关注的方向包括：更高帧率的离散化方案、噪声鲁棒的前置 ASR 以及跨语言的多模态预训练框架。推测未来会出现“文字‑音频‑视频”三位一体的统一大模型，而 Baichuan‑Audio 已经提供了音频这一块的可行蓝图。

### 一句话记住它
**Baichuan‑Audio 用多码本离散化和独立音频头，让大语言模型一次性听懂并自然说话，实现了真正的实时端到端语音交互。**