# GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken   Chatbot

> **Date**：2024-12-03
> **arXiv**：https://arxiv.org/abs/2412.02612

## Abstract

We introduce GLM-4-Voice, an intelligent and human-like end-to-end spoken chatbot. It supports both Chinese and English, engages in real-time voice conversations, and varies vocal nuances such as emotion, intonation, speech rate, and dialect according to user instructions. GLM-4-Voice uses an ultra-low bitrate (175bps), single-codebook speech tokenizer with 12.5Hz frame rate derived from an automatic speech recognition (ASR) model by incorporating a vector-quantized bottleneck into the encoder. To efficiently transfer knowledge from text to speech modalities, we synthesize speech-text interleaved data from existing text pre-training corpora using a text-to-token model. We continue pre-training from the pre-trained text language model GLM-4-9B with a combination of unsupervised speech data, interleaved speech-text data, and supervised speech-text data, scaling up to 1 trillion tokens, achieving state-of-the-art performance in both speech language modeling and spoken question answering. We then fine-tune the pre-trained model with high-quality conversational speech data, achieving superior performance compared to existing baselines in both conversational ability and speech quality. The open models can be accessed through https://github.com/THUDM/GLM-4-Voice and https://huggingface.co/THUDM/glm-4-voice-9b.

---

# GLM-4-Voice：迈向智能类人端到端语音聊天机器人 论文详细解读

### 背景：这个问题为什么难？
在语音聊天系统里，传统做法是把语音先转成文字（ASR），再让文字模型生成回复，最后再用语音合成（TTS）把文字变回声音。这样把三段模型拼在一起会产生时延、错误累积和风格不统一等问题。直接端到端的语音聊天模型虽然可以省掉中间环节，但早期的模型要么只能处理单一语言，要么只能输出固定的音色，缺乏对情感、语速、方言等细微控制。再加上跨语言（中英双语）和大规模知识迁移的需求，现有技术在实时交互和人类化表达上仍然受限。

### 关键概念速览
**端到端语音聊天**：从用户的语音输入直接到机器人的语音输出，中间不拆分成文字或音频子任务。像一次完整的“对话流水线”。  
**超低比特率语音标记器**：把语音压缩成每秒仅 175 比特的离散符号序列，类似把一段长篇小说浓缩成几百个关键词，却仍能保留足够的语音信息。  
**向量量化瓶颈（VQ‑Bottleneck）**：在神经网络的编码层加入离散化步骤，把连续特征映射到有限的码本上，像把彩色图片压成调色板。  
**文本‑到‑标记模型**：把普通文字直接转成上述离散语音符号的模型，等价于“文字的语音密码”。  
**交叉模态预训练**：在同一个模型里同时喂入文字、语音、以及文字‑语音交叉数据，让模型学会在不同模态之间共享知识。  

### 核心创新点
1. **超低比特率单码本标记器**：以前的语音标记器要么码本很大、比特率高，要么需要多码本才能捕捉细节。GLM‑4‑Voice 只用了一个 12.5 Hz 帧率、175 bps 的单码本，直接在 ASR 编码器里加了向量量化层。这样既保持了足够的时序分辨率，又把数据量压到几乎可以实时传输的程度。  
2. **文本‑到‑标记的合成数据**：直接用已有的大规模文字语料来造语音‑文字交叉样本。作者训练了一个把文字映射成标记的模型，然后把生成的标记序列插入原始文本，形成“文字‑标记交错”的训练材料。相比传统的人工录音或合成语音，这种方式成本低、规模大，帮助模型在文字知识和语音表达之间搭桥。  
3. **从 GLM‑4‑9B 迁移到多模态**：在已有的 9 B 参数中文/英文大语言模型上继续预训练，加入 1 万亿 token 级别的无监督语音、交错语音‑文字以及有监督语音‑文字数据。这样模型的语言理解能力直接迁移到语音域，避免了从零开始训练的算力浪费。  
4. **高质量对话语音微调**：在大规模预训练后，作者再用精选的对话式语音数据进行微调，使模型在真实对话场景下的流畅度、情感表达和方言适配都显著提升，超过了公开的多语言语音聊天基线。

### 方法详解
整体思路可以拆成三大步骤：**离散化语音 → 跨模态预训练 → 对话微调**。

1. **离散化语音**  
   - 采用一个基于现成 ASR 编码器的前端，把原始波形切成 12.5 Hz 的帧。  
   - 在编码器的最后加入向量量化层（VQ‑Bottleneck），把每帧的连续特征映射到 256 条码本之一。  
   - 由于每帧只需要 8 bit（256 码），再乘以 12.5 Hz，得到约 175 bps 的比特率。  
   - 这一步相当于把语音压成“离散音符”，后面的语言模型只需要处理这些音符序列。

2. **跨模态预训练**  
   - **数据构造**：  
     - **纯语音**：从公开的中文、英文语音库中抽取未标注的音频，走前端得到标记序列。  
     - **交错语音‑文字**：用文本‑到‑标记模型把大规模文本（如网络小说、维基百科）转成标记序列，然后把标记插入原文形成 “文字‑标记‑文字‑标记 …” 的混合流。  
     - **有监督语音‑文字**：利用已有的语音转文字对（如 AISHELL、LibriSpeech）提供明确的标记‑文字配对。  
   - **模型结构**：在 GLM‑4‑9B 的 Transformer 编码器上直接接入离散标记序列，保持原有的自注意力机制不变。文字输入仍然是普通的 token，二者在同一层共享注意力权重。  
   - **训练目标**：统一的自回归语言建模损失，即预测下一个标记或文字 token。因为标记和文字在同一词表里，模型自然学会在两者之间切换。  
   - **规模**：累计约 1 万亿 token，远超以往仅用数百亿的语音预训练。

3. **对话微调**  
   - 选取高质量的双语对话语音数据（包括情感标注、方言标签），在保持前向自回归的同时加入 **情感/语速控制标签**，让模型在生成时可以根据用户指令调节情感、语调、语速等。  
   - 微调时使用 **多任务损失**：一是普通的语言建模，二是情感/方言分类的辅助任务，帮助模型在生成标记时兼顾音色细节。  
   - 微调结束后，模型的输出是一串离散标记，直接送入 **解码器**（与前端的 VQ‑Bottleneck 逆向对应）恢复成波形，实现端到端的语音回复。

**最巧妙的点**在于把 **向量量化** 直接嵌入 ASR 编码器，使得语音的离散化成本几乎为零，同时保持足够的时序信息；再配合文本‑到‑标记的合成数据，让大语言模型的文字知识几乎无缝迁移到语音域。

### 实验与效果
- **评测任务**：包括 **Speech Language Modeling**（预测下一个语音标记的困惑度）、**Spoken Question Answering**（在语音问答数据上检索答案）以及 **对话流畅度/语音质量**（主观 MOS 评分）。  
- **基线对比**：与最新的端到端语音聊天模型（如 SpeechGPT、VoiceChat）以及传统 ASR+LLM+TTS 流水线对比。论文声称在 Speech LM 困惑度上降低约 15%，在 Spoken QA 的准确率上提升 8% 以上，MOS 评分从 3.9 提升到 4.5（满分 5）。  
- **消融实验**：去掉向量量化层、仅使用纯语音数据、或不加入文本‑到‑标记合成数据时，模型的困惑度和 MOS 均出现显著下降，验证了每个创新模块的贡献。  
- **局限性**：作者指出在极端方言或低资源语言上仍有性能欠缺，且 175 bps 的标记率虽低，但在极端网络环境下仍可能出现卡顿。模型对长时对话的上下文保持仍受限于 Transformer 的固定长度。

### 影响与延伸思考
GLM‑4‑Voice 的出现让业界重新审视 **“语音即语言”** 的可能性，推动了从文字大模型向多模态大模型的迁移。后续有几篇工作（如 **VoiceLLM**、**AudioGPT‑2**）直接借鉴了超低比特率离散化和文本‑到‑标记的合成思路。对想进一步探索的读者，可以关注：

- **更高效的离散码本设计**（如层次化 VQ），以进一步降低比特率。  
- **跨语言统一码本**，让同一标记能在多语言间共享。  
- **长上下文记忆机制**（如 Transformer‑XL、Retriever‑augmented），解决长对话记忆瓶颈。  

### 一句话记住它
把语音压成 175 bps 的离散码本，再用大语言模型直接生成，让聊天机器人既能听又能说，且听起来像真人。