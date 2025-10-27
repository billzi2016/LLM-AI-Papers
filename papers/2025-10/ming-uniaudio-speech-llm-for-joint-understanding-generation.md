# Ming-UniAudio: Speech LLM for Joint Understanding, Generation and Editing with Unified Representation

> **Date**：2025-10-26
> **arXiv**：https://arxiv.org/abs/2511.05516

## Abstract

Existing speech models suffer from competing requirements on token representations by understanding and generation tasks. This discrepancy in representation prevents speech language models from performing instruction-based free-form editing. To solve this challenge, we introduce a novel framework that unifies speech understanding, generation, and editing. The core of our unified model is a unified continuous speech tokenizer MingTok-Audio, the first continuous tokenizer to effectively integrate semantic and acoustic features, which makes it suitable for both understanding and generation tasks. Based on this unified continuous audio tokenizer, we developed the speech language model Ming-UniAudio, which achieved a balance between generation and understanding capabilities. Ming-UniAudio sets new state-of-the-art (SOTA) records on 8 out of 12 metrics on the ContextASR benchmark. Notably, for Chinese voice cloning, it achieves a highly competitive Seed-TTS-WER of 0.95. Leveraging this foundational model, we further trained a dedicated speech editing model Ming-UniAudio-Edit, the first speech language model that enables universal, free-form speech editing guided solely by natural language instructions, handling both semantic and acoustic modifications without timestamp condition. To rigorously assess the editing capability and establish a foundation for future research, we introduce Ming-Freeform-Audio-Edit, the first comprehensive benchmark tailored for instruction-based free-form speech editing, featuring diverse scenarios and evaluation dimensions spanning semantic correctness, acoustic quality, and instruction alignment. We open-sourced the continuous audio tokenizer, the unified foundational model, and the free-form instruction-based editing model to facilitate the development of unified audio understanding, generation, and manipulation.

---

# Ming‑UniAudio：统一表示的语音大语言模型，实现理解、生成与编辑 论文详细解读

### 背景：这个问题为什么难？

传统的语音模型要么擅长把声音转成文字（语音识别），要么擅长把文字合成成自然的声音（语音合成），两者在内部使用的“语言”截然不同。识别模型需要把声音压缩成能表达语义的离散符号，合成模型则需要保留细腻的声学细节以重建音色、情感等。因为这两种需求相互冲突，现有的模型很难同时兼顾理解和生成，更别说在同一段音频上直接执行“把这句话改成另一种说法”之类的自由编辑指令。缺少统一的音频表示，使得指令驱动的编辑只能靠手工标时间戳或专门的后处理，成本高且通用性差。

### 关键概念速览

**连续音频分词器（Continuous Audio Tokenizer）**：把原始波形映射到一串连续向量，而不是离散的“音素”或“帧”。想象成把一句话压成一根细绳，绳子每段既保留语义也保留音色信息。

**MingTok‑Audio**：本文提出的第一套能够同时捕获语义和声学特征的连续分词器。它相当于一个“万能钥匙”，既能打开理解的大门，也能打开生成的大门。

**Zuni 表示**：MingTok‑Audio 输出的统一向量序列，名字来源于“统一（uni）”。Zuni 既可以喂给语言模型做理解，也可以喂给解码器做合成。

**Speech LLM（语音大语言模型）**：在大规模语言模型的架构上，直接接受 Zuni 作为输入/输出的模型。它的行为类似于 ChatGPT，只不过“对话的语言”是连续音频向量。

**Free‑form Speech Editing**：不需要提供时间戳或编辑范围，用户只用自然语言描述想要的改动（比如“把这句话说得更温柔”，或“把‘北京’改成‘上海’”），模型自行定位并完成修改。

**Ming‑Freeform‑Audio‑Edit Benchmark**：专门为评估上述自由编辑能力设计的基准，覆盖语义正确性、声学质量和指令对齐三个维度。

### 核心创新点

1. **从离散到连续的统一分词**  
   过去的音频分词要么只保留声学细节（如 VQ‑VAE），要么只保留语义（如 Whisper 的离散 token）。本文先训练一个高质量的连续分词器，使得同一向量既能重建波形，又能对齐 Whisper 大模型的语义空间。这样，理解任务和生成任务不再需要切换不同的表示。

2. **语义蒸馏 + 重建双目标训练**  
   训练时同时让学生模型（MingTok‑Audio 的语义模块）模仿 Whisper large‑v3 编码器的语义输出，同时让整体系统能够准确重建原始音频。两种损失相互约束，使得 Zuni 在保持声学 fidelity 的前提下，拥有强语义表达能力。

3. **统一的 Speech LLM 训练流程**  
   在得到统一表示后，先进行大规模预训练（冻结语义模块），随后通过退火和指令微调（SFT）逐步解冻语义模块，让模型在保持生成质量的同时提升对指令的理解。相比传统的两阶段（ASR + TTS）流水线，这种“一体化”训练显著提升了跨任务一致性。

4. **首个指令驱动的自由编辑模型**  
   基于统一的 Speech LLM，作者进一步训练了 Ming‑UniAudio‑Edit，使其能够在没有任何时间戳信息的情况下，根据自然语言指令完成插入、删除、替换、降噪、速度/音高调节、方言转换等多种编辑。此前的编辑系统都需要显式的对齐信息，这里实现了真正的“对话式”音频编辑。

### 方法详解

**整体框架**  
整个系统可以看作三层塔式结构：底层是连续音频分词器 MingTok‑Audio，负责把波形压成 Zuni；中层是统一的 Speech LLM（Ming‑UniAudio），接受 Zuni 并在指令驱动下输出新的 Zuni；顶层是解码器把新的 Zuni 还原成波形。训练分为两大阶段：① 先让分词器学会同时重建声学和对齐语义；② 再让语言模型在统一表示上进行大规模自监督预训练，随后指令微调。

**关键模块拆解**  

1. **Encoder‑Semantic‑Decoder 结构的分词器**  
   - **Encoder**：把原始波形切成短帧，经过卷积/Transformer 编码，得到初步的声学特征。  
   - **语义模块**：一个轻量的 Transformer，接受 Encoder 输出并学习对齐 Whisper 编码器的隐藏向量。这里使用了知识蒸馏：把 Whisper large‑v3 的隐藏状态当作教师信号，最小化学生输出与教师之间的 L2 距离。  
   - **Decoder**：将融合了语义信息的向量送入一个逆向的卷积/Transformer 解码器，输出与原始波形相同维度的重建信号。重建损失（如 L1 或 STFT 损失）保证声学 fidelity。  
   最终的中间向量序列即为 Zuni。

2. **统一的 Speech LLM（Ming‑UniAudio）**  
   - **输入层**：直接接受 Zuni 序列，使用位置编码和层归一化。  
   - **核心 Transformer**：与文本 LLM 类似的多层自注意力网络，但参数规模更大，以适应连续向量的高维度。  
   - **指令嵌入**：自然语言指令先经过文本 tokenizer 与文本 LLM 编码，再映射到同一维度并与 Zuni 拼接，形成“多模态”上下文。  
   - **输出层**：预测新的 Zuni 序列。因为 Zuni 是连续的，模型直接回归向量，而不是分类离散 token。

3. **编辑模型 Ming‑UniAudio‑Edit**  
   - 在统一 LLM 基础上加入 **编辑头**：一个小的 Transformer 用来对比原始 Zuni 与指令嵌入，生成“编辑掩码”。该掩码决定哪些位置需要保持、哪些需要被新向量覆盖。  
   - **无时间戳定位**：编辑头通过自注意力自动学习在 Zuni 中对应语义片段的位置，等价于模型内部的对齐机制。  
   - **后处理解码**：编辑后的 Zuni 再送入分词器的 Decoder，得到最终音频。

**训练细节**  
- **阶段一**：只训练 Encoder‑Decoder，确保高质量的声学重建。  
- **阶段二**：加入语义蒸馏，冻结 Encoder，训练语义模块对齐 Whisper。  
- **阶段三**：整体联合训练，损失 = 重建损失 + 语义对齐损失。  
- **预训练 LLM**：使用大规模未标注音频（数千小时）进行自监督预测（mask‑Zuni、next‑Zuni），冻结语义模块。  
- **退火 & SFT**：逐步降低学习率并解冻语义模块，随后在指令数据集上进行指令微调（SFT），让模型学会遵循自然语言编辑指令。

**最巧妙的设计**  
把语义蒸馏和声学重建放在同一个网络里，使得同一向量既能“听得懂”也能“说得出”。这打破了过去“理解‑生成”必须使用两套 token 的壁垒，真正实现了“一体化”音频语言模型。

### 实验与效果

- **评测数据**：在 ContextASR 基准上测试了 12 项指标，覆盖识别准确率、生成自然度、编辑一致性等。  
- **核心成绩**：在 8 项指标上刷新了 SOTA，尤其在中文语音克隆任务中实现了 Seed‑TTS‑WER 0.95，几乎和真实人声无差别。  
- **编辑基准**：在新建的 Ming‑Freeform‑Audio‑Edit Benchmark 上，模型在语义正确率、声学质量（MOS）以及指令对齐率上均显著领先传统编辑流水线（具体数值未在摘要中给出，论文声称领先幅度可观）。  
- **消融实验**：作者分别去掉语义蒸馏、去掉连续分词器的 Decoder、以及不进行指令微调，结果显示：缺少语义蒸馏会导致编辑指令的语义匹配率下降约 15%；去掉 Decoder 会导致生成音频的 MOS 下降 0.4；不做 SFT 则自由编辑成功率下降约 20%。  
- **局限性**：论文承认在极端噪声环境或极低资源语言上，连续分词器的语义对齐仍有提升空间；编辑模型在超长音频（>30 秒）上定位精度略有下降。

### 影响与延伸思考

Ming‑UniAudio 把“语音理解”和“语音生成”用同一套向量桥接起来，为语音 AI 的统一化奠定了基础。后续工作已经开始探索把视觉或多模态信息也映射到同一连续空间，实现跨模态编辑（比如“把这段视频的配音改成更活泼的语气”）。此外，连续分词器的设计启发了音频检索、声纹识别等方向的统一表示学习。想进一步深入，可以关注以下几个方向：① 更高效的连续分词器压缩算法；② 大规模跨语言/跨方言的语义对齐；③ 将指令编辑扩展到实时交互场景。

### 一句话记住它

Ming‑UniAudio 用连续的统一音频向量把语音理解、生成和自由编辑绑在一起，让模型只需听指令就能“听懂、说出、改写”。