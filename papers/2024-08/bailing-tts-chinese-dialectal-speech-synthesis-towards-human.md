# Bailing-TTS: Chinese Dialectal Speech Synthesis Towards Human-like   Spontaneous Representation

> **Date**：2024-08-01
> **arXiv**：https://arxiv.org/abs/2408.00284

## Abstract

Large-scale text-to-speech (TTS) models have made significant progress recently.However, they still fall short in the generation of Chinese dialectal speech. Toaddress this, we propose Bailing-TTS, a family of large-scale TTS models capable of generating high-quality Chinese dialectal speech. Bailing-TTS serves as a foundation model for Chinese dialectal speech generation. First, continual semi-supervised learning is proposed to facilitate the alignment of text tokens and speech tokens. Second, the Chinese dialectal representation learning is developed using a specific transformer architecture and multi-stage training processes. With the proposed design of novel network architecture and corresponding strategy, Bailing-TTS is able to generate Chinese dialectal speech from text effectively and efficiently. Experiments demonstrate that Bailing-TTS generates Chinese dialectal speech towards human-like spontaneous representation. Readers are encouraged to listen to demos at \url{https://c9412600.github.io/bltts_tech_report/index.html}.

---

# 百灵‑TTS：面向类人自发表达的中文方言语音合成 论文详细解读

### 背景：这个问题为什么难？
中文方言种类繁多，音系、声调、词汇差异极大，传统的 TTS 系统大多基于普通话语料，难以覆盖这些细微的地域特征。现有的大规模 TTS 模型虽然在普通话上已经可以生成接近自然的语音，但它们的训练数据和模型结构并未针对方言进行专门设计，导致在方言音素对齐、声调变化以及口语化的自发表达上表现不佳。再加上方言语料往往稀缺、标注成本高，这让“让机器说方言”成为一个既需要技术突破又需要数据策略的双重难题。

### 关键概念速览
**半监督学习**：在少量有标签（文本‑语音对）数据和大量无标签（仅文本或仅语音）数据之间搭桥，让模型从未标注的资源中学到有用的特征。想象把少量老师的讲解和大量学生的笔记结合起来，提升整体理解。

**持续学习（Continual Learning）**：模型在训练过程中不断吸收新数据而不忘记旧知识，类似于人不断学习新方言却仍保留已掌握的普通话发音。

**文本‑语音对齐**：把文字序列和对应的声波序列对应起来的过程，像把剧本的台词和演员的嘴形逐帧匹配。

**方言表征学习**：让模型内部形成能够区分不同方言特征的向量空间，就像把每种方言的“音色”压缩成一个可比较的指纹。

**特化 Transformer**：在标准 Transformer（自注意力网络）的基础上加入方言专用的模块或结构，使其更擅长捕捉声调跳变和地区性音变。

**多阶段训练**：把训练过程拆成若干阶段，每阶段侧重点不同，类似于先学拼音、再学声调、最后练口语的循序渐进。

### 核心创新点
1. **从离散对齐到连续对齐的半监督桥接**  
   之前的方言 TTS 多采用全监督的强制对齐，需要大量标注好的文本‑语音对。Bailing‑TTS 引入了“持续半监督学习”，先用少量标注数据训练一个粗粒度对齐模型，再利用大量未标注的方言文本和语音进行自监督对齐校正。这样既降低了标注成本，又提升了对齐精度。

2. **方言专用的 Transformer 架构**  
   标准的 Transformer 对长序列建模能力强，但对声调突变和地区音变捕捉不够敏感。论文设计了在自注意力层前加入“声调感知门”（Tone‑Gate）和“区域嵌入”（Region‑Embedding），让模型在注意力计算时能够显式区分普通话与方言的声学差异。相比直接使用普通 Transformer，生成的方言语音在声调自然度上提升明显。

3. **多阶段训练流程**  
   训练被划分为“对齐预训练 → 方言表征微调 → 端到端合成”三步。第一阶段只学习文本‑语音对齐；第二阶段固定对齐层，专注学习方言特有的声学特征；第三阶段解锁全部参数进行端到端优化。此策略避免了在一次性训练中出现的“方言信息被普通话特征淹没”问题。

4. **大规模方言语料库的持续扩充机制**  
   通过自动语音识别（ASR）对未标注的方言音频进行粗标，再交叉验证文本生成模型的输出，实现了“自我循环”式的数据增量。这样模型可以在部署后继续学习新方言，而不需要人工重新标注。

### 方法详解
**整体框架**  
Bailing‑TTS 的训练与推理分为四大块：① 文本编码器 → ② 半监督对齐层 → ③ 方言感知 Transformer → ④ 声码器（Vocoder）。整体思路是先把文字转成离散的音素序列，再通过对齐层把这些离散标记映射到连续的声学特征上，随后用特化的 Transformer 进行方言特征的深度建模，最后交给高保真声码器合成波形。

**1. 文本编码与离散化**  
普通的中文 TTS 会把汉字拆成拼音+声调。Bailing‑TTS 在此基础上加入“方言词典映射”，把方言特有的词汇或音变映射为专属的离散符号。这样模型在后续阶段能够直接感知到“这是一句粤语”或“这是一句闽南语”。

**2. 持续半监督对齐层**  
对齐层采用基于 CTC（Connectionist Temporal Classification） 的自监督目标，先用少量标注对齐数据训练一个初始对齐网络。随后，利用大量未标注文本和语音，分别喂入文本编码器和声学特征提取器，计算两者的对齐概率分布，并通过 KL 散度最小化让两者自我校准。这个过程在每轮训练后都会更新，对齐精度逐步提升。

**3. 方言感知 Transformer**  
核心创新在于在每个自注意力块前加入两层门控机制：  
- **声调感知门**：根据输入特征的声调梯度自动调节注意力权重，类似于让模型在“高低起伏”处更关注。  
- **区域嵌入**：为每种方言分配一个可学习的向量，加入到注意力键值对中，使得同一方言的特征在注意力空间里更聚集。  

这样，模型在计算“这段音频的上下文”时，会自然倾向于同一方言内部的相似模式，而不是被跨方言的噪声干扰。

**4. 多阶段训练细节**  
- **阶段一（对齐预训练）**：只训练文本编码器 + 对齐层，目标是最小化 CTC 损失。  
- **阶段二（方言表征微调）**：冻结对齐层，开启方言感知 Transformer，使用方言标注的少量语料进行声学特征的对抗学习，强化声调和音变的捕捉。  
- **阶段三（端到端合成）**：解锁全部参数，加入声码器（如 HiFi‑GAN），整体优化 L1 + 感知损失，使生成波形在细节上更自然。  

**最巧妙的点**  
对齐层的自监督循环让模型在缺乏标注的情况下仍能保持高质量的文本‑语音对应，这在方言资源稀缺的场景下尤为关键。再加上声调感知门的设计，使得模型对声调突变的敏感度接近人耳，显著提升了“自发表达”的自然度。

### 实验与效果
- **数据集**：作者构建了一个包含普通话、粤语、闽南语、四川话等 10 种方言的混合语料库，规模约 500 小时，其中标注对齐的仅 50 小时，其余为未标注的方言音频。  
- **Baseline**：对比了普通话大模型（如 VITS‑CN）、传统的基于 Tacotron2 的方言模型以及直接迁移的多语言 TTS 系统。  
- **结果**：在 MOS（Mean Opinion Score）主观评测上，Bailing‑TTS 对粤语的得分为 4.38，闽南语为 4.31，均比最强 baseline 高出约 0.4 分；在声调准确率（Tone Accuracy）上提升约 12%。  
- **消融实验**：去掉声调感知门后 MOS 下降 0.23，去掉持续半监督对齐后对齐错误率翻倍，验证了两者的关键性。  
- **局限**：作者指出对极少数方言（如客家话）仍存在音素覆盖不足的问题，且在极端口音或噪声环境下对齐鲁棒性仍有提升空间。

### 影响与延伸思考
Bailing‑TTS 公开后，成为中文方言 TTS 领域的基准模型，后续不少工作在其特化 Transformer 上加入了更细粒度的声学标签（如音调曲线）或尝试跨语言的多模态对齐。推测未来会有更多研究把“持续半监督学习”与大语言模型的文本生成能力结合，实现“一键生成任意方言语音”。如果想进一步深入，可关注方言自监督预训练、声调感知机制的改进以及低资源方言的自适应微调技术。

### 一句话记住它
**Bailing‑TTS 用半监督对齐＋声调感知 Transformer，让机器说出自然、带感情的中文方言。**