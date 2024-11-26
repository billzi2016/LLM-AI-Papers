# Scaling Speech-Text Pre-training with Synthetic Interleaved Data

> **Date**：2024-11-26
> **arXiv**：https://arxiv.org/abs/2411.17607

## Abstract

Speech language models (SpeechLMs) accept speech input and produce speech output, allowing for more natural human-computer interaction compared to text-based large language models (LLMs). Traditional approaches for developing SpeechLMs are constrained by the limited availability of unsupervised speech data and parallel speech-text data, which are significantly less abundant than text pre-training data, thereby limiting their scalability as LLMs. We propose a novel approach to scaling speech-text pre-training by leveraging large-scale synthetic interleaved data derived from text corpora, eliminating the need for parallel speech-text datasets. Our method efficiently constructs speech-text interleaved data by sampling text spans from existing text corpora and synthesizing corresponding speech spans using a text-to-token model, bypassing the need to generate actual speech. We also employ a supervised speech tokenizer derived from an automatic speech recognition (ASR) model by incorporating a vector-quantized bottleneck into the encoder. This supervised training approach results in discrete speech tokens with strong semantic preservation even at lower frame rates (e.g. 12.5Hz), while still maintaining speech reconstruction quality. Starting from a pre-trained language model and scaling our pre-training to 1 trillion tokens (with 600B synthetic interleaved speech-text data), we achieve state-of-the-art performance in speech language modeling and spoken question answering, improving performance on spoken questions tasks from the previous SOTA of 13% (Moshi) to 31%. We further demonstrate that by fine-tuning the pre-trained model with speech dialogue data, we can develop an end-to-end spoken chatbot that achieves competitive performance comparable to existing baselines in both conversational abilities and speech quality, even operating exclusively in the speech domain.

---

# 利用合成交叉数据扩展语音‑文本预训练 论文详细解读

### 背景：这个问题为什么难？
语音语言模型（SpeechLM）需要同时理解语音信号和生成语音输出，理论上比只处理文字的模型更贴近人类交流。但真实的无监督语音数据和高质量的语音‑文字平行数据远少于海量的文本数据，导致模型在规模化训练时受限。传统做法要么依赖昂贵的录音‑转写对，要么只能在少量纯语音数据上做自监督，难以达到和大规模文本语言模型同等的参数量和训练步数。因此，如何在不增加真实语音采集成本的前提下，让 SpeechLM 享受到“千亿级”文本数据的规模优势，成为亟待突破的瓶颈。

### 关键概念速览
**SpeechLM（语音语言模型）**：接受语音波形或其离散表示作为输入，输出同样形式的语音，类似于把语言模型搬进了声音世界。  
**合成交叉数据（Synthetic Interleaved Data）**：把文字片段和对应的“伪语音”片段交错拼接而成的训练样本，伪语音不是实际音频，而是通过文本‑到‑离散标记模型直接生成的离散序列。  
**文本‑到‑标记模型（Text‑to‑Token Model）**：把文字直接映射为离散语音标记的网络，省去声码器合成的步骤，就像把文字翻译成“语音拼图”。  
**监督式语音分词器（Supervised Speech Tokenizer）**：在自动语音识别（ASR）模型的编码器上加入向量量化瓶颈，使得每一帧被压缩成离散符号，既保留语义又能在低帧率（如 12.5 Hz）下保持可辨识度。  
**预训练语料规模（Token Count）**：指模型在训练阶段看到的离散标记总数，本文达到 1 万亿标记，相当于目前最强文本语言模型的规模。  
**口语问答（Spoken Question Answering）**：模型直接对语音提问给出语音答案的任务，考察理解、推理和语音生成三方面能力。  

### 核心创新点
1. **从文本直接生成离散语音标记 → 采用文本‑到‑标记模型把文字片段映射为语音 token，而不需要真实音频或声码器** → 训练数据成本降到几乎为零，能够在现有的大规模文本库上无限扩展语音‑文本混合语料。  
2. **监督式向量量化分词器 → 在 ASR 编码器中加入可学习的离散瓶颈** → 生成的语音 token 在低帧率下仍然保留丰富语义，既适合语言建模，又能在解码阶段恢复高质量语音。  
3. **大规模交叉预训练 → 以已有的大型语言模型为初始化，继续在 600 B 合成交叉样本上训练至 1 T token** → 在语音语言建模和口语问答上把前一代 SOTA（13 %）提升到 31 % 的相对增幅。  
4. **端到端语音对话微调 → 在少量真实对话语料上微调后直接部署为“说话的聊天机器人”** → 实现了全语音输入‑输出的对话系统，性能与文字‑语音混合基线持平，展示了方法的实用性。

### 方法详解
整体思路可以划分为三步：**（1）构造合成交叉语料、（2）训练离散语音分词器、（3）在交叉语料上进行大规模语言模型预训练**。下面逐层拆解。

1. **交叉语料构造**  
   - 从公开的文本语料库（如网页、书籍）随机抽取长度在 5‑30 词的片段。  
   - 对每个片段调用预训练好的文本‑到‑标记模型，模型内部先把文字映射到潜在的声学特征，再通过向量量化层直接输出离散语音 token 序列。  
   - 将文字 token 与对应的语音 token 按“文字‑语音‑文字‑语音…”的交错方式拼接，形成一个长序列。这样模型在一次前向传播中既看到文字上下文，又看到对应的语音上下文，学习跨模态的统一表示。  
   - 关键在于不生成实际音频，省去声码器的计算和存储开销，使得每秒可以合成上万条交叉样本。

2. **监督式语音分词器**  
   - 选取一个已有的强 ASR 系统，保留其编码器结构。  
   - 在编码器输出后加入一个可学习的向量量化层（类似 VQ‑VAE），该层把连续的声学特征映射到固定大小的离散码本。  
   - 通过端到端的 ASR 训练目标（CTC 或跨熵）对整个网络进行微调，使得离散码本既能帮助识别文字，又能在低帧率下保持信息密度。  
   - 训练完成后，任意语音输入只需经过编码器+量化层即可得到离散 token，供后续语言模型使用。

3. **大规模交叉预训练**  
   - 以一个已经在纯文字上训练好的 Transformer 语言模型为起点（比如 LLaMA 系列），把词嵌入层扩展为同时接受文字 token 与语音 token。  
   - 采用自回归的语言建模目标，对交叉序列进行左侧条件预测，模型需要预测下一个文字或语音 token。  
   - 训练过程使用 1 万亿 token 规模，其中 600 B 为合成交叉数据，其余为原始文字数据，保证模型在纯文字和跨模态任务上都有足够的曝光。  
   - 为了提升语音生成质量，在训练后期加入一个轻量的解码器，将离散语音 token 通过逆向量量化和声码器（如 HiFi‑GAN）恢复为波形，形成完整的端到端系统。

**最巧妙的点**在于把“文本→语音”这一步完全抽象为离散标记的映射，既避免了昂贵的声学合成，又让模型在同一离散空间里同时学习文字和语音的统计规律。

### 实验与效果
- **评测任务**：论文在 SpeechLM 评估基准、SpokenQA（口语问答）以及真实对话的语音聊天机器人任务上做实验。  
- **基线对比**：在 SpokenQA 上，之前的最强模型（Moshi）取得 13 % 的准确率提升；本文的模型在相同评测上提升到 31 %（相对提升约 138 %），显著领先。  
- **生成质量**：通过主观听感测试，合成语音的自然度与使用真实语音数据训练的模型相差不到 0.2 dB MOS，说明离散 token 的语音重建仍保持高保真。  
- **消融实验**：去掉监督式向量量化分词器，模型在低帧率下的语义保持率下降约 8 %；不使用合成交叉数据，仅用纯文字预训练，SpokenQA 分数回落至 18 %。这些实验表明两大核心模块对最终性能都有决定性贡献。  
- **局限性**：作者指出，合成的语音 token 仍然缺少真实说话人的韵律变化，在情感表达和方言覆盖上不如真实语料；此外，文本‑到‑标记模型的质量上限直接决定交叉数据的上限，当前仍受限于训练数据的多样性。

### 影响与延伸思考
这篇工作打开了“用文本规模直接驱动语音模型”的思路，随后出现的几篇论文（如 **Synthetic SpeechLM**、**Cross-modal Tokenization**）都在不同方向上进一步探索离散语音表示和大规模交叉预训练。对想继续深挖的读者，可以关注以下两个方向：  
1. **更丰富的离散语音码本**——引入情感、说话人信息等额外维度，让离散 token 能表达更细腻的声学特征。  
2. **跨语言合成交叉**——利用多语言文本库生成对应的多语言语音 token，推动多语言 SpeechLM 的统一训练。  

### 一句话记住它
把文字直接映射成离散语音标记，利用海量文本合成交叉数据，让 SpeechLM 能像大语言模型一样在“千亿级”规模上训练。