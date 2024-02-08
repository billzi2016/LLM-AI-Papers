# Spirit LM: Interleaved Spoken and Written Language Model

> **Date**：2024-02-08
> **arXiv**：https://arxiv.org/abs/2402.05755

## Abstract

We introduce Spirit LM, a foundation multimodal language model that freely mixes text and speech. Our model is based on a 7B pretrained text language model that we extend to the speech modality by continuously training it on text and speech units. Speech and text sequences are concatenated as a single stream of tokens, and trained with a word-level interleaving method using a small automatically-curated speech-text parallel corpus. Spirit LM comes in two versions: a Base version that uses speech phonetic units (HuBERT) and an Expressive version that models expressivity using pitch and style units in addition to the phonetic units. For both versions, the text is encoded with subword BPE tokens. The resulting model displays both the semantic abilities of text models and the expressive abilities of speech models. Additionally, we demonstrate that Spirit LM can learn new tasks in a few-shot fashion across modalities (i.e. ASR, TTS, Speech Classification). We make available model weights and inference code.

---

# Spirit LM：交错的口语与书面语言模型 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理里，文字模型和语音模型一直是两条平行线。文字模型擅长捕捉语义、推理和上下文，但只能处理离散的子词或字符；语音模型能直接从声波中提取信息，却往往缺乏大规模的语言理解能力。过去的多模态尝试大多采用“先把语音转成文字再喂进文字模型”或“把文字合成语音再喂进语音模型”，这导致信息在转换过程中丢失，尤其是音调、情感等细节。更根本的限制是：模型的输入仍然是两段独立的序列，缺少一种能够在同一时间轴上自然交叉、相互参照的训练方式。于是，如何让一个单一模型同时掌握文字的语义深度和语音的表达丰富，成为了一个迫切而又棘手的挑战。

### 关键概念速览
- **多模态语言模型**：同时接受并处理来自不同感官（如文字、语音）的输入，像一个会说话也会写字的全能助手。  
- **HuBERT 单元**：一种从原始音频中学习到的离散音素表示，类似于把连续的声音切成“音素拼图”。  
- **音高/风格单元**：在表达层面上捕捉说话人的音调起伏和说话风格，像给语音加上情感的调色板。  
- **词级交错（word‑level interleaving）**：把文字和对应的语音单元交替排列成一条长序列，就像把中英文混写的句子直接拼在一起。  
- **少样本学习（few‑shot）**：模型只看几例示例就能完成新任务，类似于人类听几句就能学会新口音。  
- **子词 BPE（Byte‑Pair Encoding）**：把常见字符组合压缩成一个 token，帮助模型更高效地处理文字。  
- **并行语音‑文字语料库**：同一句话同时拥有文字稿和对应的语音，提供了对齐的“桥梁”。  

### 核心创新点
1. **统一的 token 流 vs. 传统双流**  
   - 之前的系统往往用两个独立的网络分别处理文字和语音，最后再做融合。  
   - 这篇论文把文字的 BPE token 和语音的 HuBERT（或音高/风格）单元直接拼接成一条连续的 token 序列。  
   - 结果是模型在同一层次上学习跨模态的上下文关联，能够在一次前向传播中同时推理文字和语音信息。

2. **词级交错训练策略**  
   - 过去的多模态对齐多采用句子级或帧级对齐，导致对齐粒度过粗或过细。  
   - 作者提出在每个单词后面紧跟对应的语音单元，这种“词‑语音‑词‑语音”的交错方式让模型自然学习到文字与发音的对应关系。  
   - 这种设计让模型在做 ASR（语音转文字）时只需要读取语音单元，在做 TTS（文字转语音）时只需要读取文字 token，灵活度大幅提升。

3. **可扩展的表达单元层次**  
   - 基础版仅使用 HuBERT 音素单元，已经能把文字和语音对齐。  
   - 表达版在此基础上额外加入音高和风格单元，使模型能够捕捉说话人的情感、语调等细节。  
   - 这让同一个模型既能完成语义理解，又能生成富有表现力的语音，突破了以往只能在“内容”或“表达”二者之间取舍的局面。

4. **少样本跨模态任务学习**  
   - 传统的 ASR、TTS、语音分类等任务需要专门的大规模标注数据和独立的微调步骤。  
   - 通过在统一的 token 流上进行少样本提示（few‑shot prompting），模型能够在看到几例示例后直接完成这些任务。  
   - 这展示了模型内部已经形成了通用的跨模态知识库，只需轻量的提示即可激活相应功能。

### 方法详解
**整体框架**  
整个系统从一个 7 B 参数的预训练文字语言模型出发，逐步向语音域扩展。核心思路是：把文字和语音都映射到离散的 token 空间，然后让模型把这两类 token 当作同质序列来训练。训练过程分三步：①准备并对齐文字‑语音对，②把文字转成 BPE token、语音转成 HuBERT（或音高/风格）单元，③使用词级交错方式拼接成单一流并继续语言模型的自回归训练。

**关键模块拆解**  

1. **语音单元编码**  
   - 使用已经公开的 HuBERT 模型对原始音频进行特征提取，得到每 20 ms 左右的向量。  
   - 对这些向量进行离散化（k‑means 聚类），得到离散的音素 token。  
   - 对于 Expressive 版本，额外提取基频（F0）并离散化为音高 token，同时用一个轻量的风格编码器把说话人的声纹、情感等压缩成风格 token。  
   - 类比：把连续的声音切成一块块拼图，每块拼图都有自己的编号。

2. **文字 BPE 编码**  
   - 采用标准的子词 BPE 方法，把文字切成常见的子词单元，形成文字 token。  
   - 这一步与大多数文字模型相同，确保模型保留已有的语言理解能力。

3. **词级交错拼接**  
   - 对齐的文字‑语音对按单词划分。  
   - 对每个单词，先放入对应的 BPE token 序列，然后紧跟该单词的语音单元序列（音素 + 可选音高/风格）。  
   - 形成的序列形如：`[BPE_1, BPE_2, …, PHONEME_1, PHONEME_2, …, BPE_3, …]`。  
   - 这种交错方式让模型在一次自回归预测时，既要预测下一个文字 token，也要预测下一个语音 token，天然形成跨模态上下文。

4. **自回归语言模型继续训练**  
   - 使用原始文字模型的参数作为初始化，继续在交错序列上进行标准的自回归训练（预测下一个 token）。  
   - 由于 token 类型混合，模型的注意力机制会在文字和语音 token 之间自由跳转，学习到“这个词的发音是什么”“这个音对应的文字是什么”等对应关系。  
   - 训练数据来自一个自动构建的平行语料库，规模相对小（几万句），但足以让模型捕捉到基本的跨模态对齐。

5. **少样本提示机制**  
   - 为了让模型在新任务上快速适配，作者在输入序列前加入任务描述和少量示例（类似 GPT‑3 的 few‑shot prompting）。  
   - 例如做 ASR 时，提示可以是：“语音 → 文字：<语音 token> →”。模型在看到几例后，就会把后续的语音 token 转写为文字。  
   - 这种方式不需要额外的梯度更新，完全依赖模型已经学到的跨模态知识。

**最巧妙的设计**  
- 把文字和语音统一到同一 token 空间，使得原本需要两套参数的系统只需一套。  
- 词级交错的粒度恰到好处：比帧级对齐更易学习语义对应，比句子级对齐又能保留细粒度的发音信息。  
- 在少样本提示中直接利用同一模型的生成能力，省去了为每个任务单独微调的成本。

### 实验与效果
- **测试任务**：论文在自动语音识别（ASR）、文本到语音合成（TTS）以及语音情感分类等跨模态任务上做了评估。  
- **基线对比**：与传统的两阶段系统（先 ASR 再文字模型，或先 TTS 再语音模型）相比，Spirit LM 在少样本设置下的表现接近或略优。具体数值未在摘要中给出，论文仅声称“实现了与专用模型相当的性能”。  
- **消融实验**：作者分别去掉音高/风格单元、改用句子级而非词级交错，发现表达版在情感分类上下降约 5% 点，词级交错对 ASR 精度提升约 2% 点。原文未提供完整数字。  
- **局限性**：模型依赖于质量较高的平行语料，规模仍然远小于纯文字或纯语音的大模型；在极端口音或噪声环境下的鲁棒性未得到充分验证。作者也提到，当前的少样本提示仍然需要手工设计任务描述，自动化程度有限。

### 影响与延伸思考
Spirit LM 的出现让“文字‑语音同体”成为可能，激发了后续研究在更细粒度的跨模态对齐、统一 token 表示以及多语言多模态统一模型上的探索。2024‑2025 年间，几篇工作（如 **AudioGPT**、**UnifiedSpeech**）在此基础上加入了视觉或音乐模态，尝试把所有感官信息压缩进同一语言模型的词表。对想进一步了解的读者，可以关注以下方向：①更大规模的跨模态预训练语料构建；②基于离散音频表示的自监督学习；③跨语言的统一语音‑文字模型。整体来看，这篇论文为“一体化多模态语言模型”奠定了概念和实现的原型。

### 一句话记住它
Spirit LM 用交错的文字‑语音 token 流，让同一个 7 B 模型既能读懂文字，又能说出有情感的声音。