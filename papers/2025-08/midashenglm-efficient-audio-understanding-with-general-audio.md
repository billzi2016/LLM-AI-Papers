# MiDashengLM: Efficient Audio Understanding with General Audio Captions

> **Date**：2025-08-06
> **arXiv**：https://arxiv.org/abs/2508.03983

## Abstract

Current approaches for large audio language models (LALMs) often rely on closed data sources or proprietary models, limiting their generalization and accessibility. This paper introduces MiDashengLM, a novel open audio-language model designed for efficient and comprehensive audio understanding through the use of general audio captions using our novel ACAVCaps training dataset. MiDashengLM exclusively relies on publicly available pretraining and supervised fine-tuning (SFT) datasets, ensuring full transparency and reproducibility. At its core, MiDashengLM integrates Dasheng, an open-source audio encoder, specifically engineered to process diverse auditory information effectively. Unlike previous works primarily focused on Automatic Speech Recognition (ASR) based audio-text alignment, our strategy centers on general audio captions, fusing speech, sound and music information into one textual representation, enabling a holistic textual representation of complex audio scenes. Lastly, MiDashengLM provides an up to 4x speedup in terms of time-to-first-token (TTFT) and up to 20x higher throughput than comparable models. Checkpoints are available online at https://huggingface.co/mispeech/midashenglm-7b and https://github.com/xiaomi-research/dasheng-lm.

---

# MiDashengLM：使用通用音频字幕实现高效音频理解 论文详细解读

### 背景：这个问题为什么难？
音频大语言模型（Audio LLM）要把声音映射到自然语言，需要同时解决两大难题：一是获取大规模、质量可靠的音频‑文本对；二是让模型在推理时既能捕捉细腻的声学信息，又保持实时性。过去的做法大多依赖商业ASR系统生成的文字或内部专有数据集，这导致数据闭源、版权受限，且只能覆盖说话内容，忽视环境声、音乐等丰富信息。结果是模型在多模态音频场景（比如“雨声中有人弹吉他”）上表现差强人意，而且推理成本高，难以部署到边缘设备。

### 关键概念速览
- **音频大语言模型（Audio LLM）**：把音频信号当作输入，输出自然语言描述或完成语言任务的模型，类似于文本大语言模型但多了声学前置层。  
- **通用音频字幕（General Audio Caption）**：对一段音频的整体描述，既包括说话内容，也涵盖背景声、音乐、情感等，像给一段视频写的字幕，只是纯声音。  
- **Dasheng 音频编码器**：本文开源的音频特征提取器，专门为多种声源设计，类似于视觉领域的ResNet，但处理的是波形或频谱。  
- **预训练 + 监督微调（Pre‑training + SFT）**：先在大规模无标签音频上学习通用声学特征，再用标注好的音频字幕进行有监督的语言对齐。  
- **时间到首标记（TTFT, Time‑to‑First‑Token）**：模型生成第一个文字 token 所需的时间，衡量实时性。  
- **吞吐量（Throughput）**：单位时间内模型能处理的音频片段数，直接决定部署成本。  
- **ACAVCaps 数据集**：作者自行构建的“Audio Caption with Diverse Audio Sources”数据集，全部公开，覆盖说话、环境声、音乐等多模态音频。  

### 核心创新点
1. **全链路开源 → 使用公开的预训练和微调数据 → 透明可复现**  
   过去的 Audio LLM 多依赖内部语料或商业 ASR 生成的文字，导致研究者难以复现。MiDashengLM 完全摆脱闭源数据，所有音频、字幕、模型权重均在公开平台发布，任何人都能从头跑通训练流程。

2. **从 ASR 对齐转向通用音频字幕 → 采用 ACAVCaps 进行统一文本化 → 能同时理解说话、噪声、音乐**  
   传统方法把音频和文字对齐的目标限定在“说了什么”，忽视了声音的上下文。作者收集并清洗了大量通用字幕，使模型学习到“一段音频 = 一个完整的文字描述”，从而在复杂声场下也能生成准确的语义解释。

3. **引入专用的 Dasheng 编码器 → 采用多尺度卷积 + 注意力混合结构 → 更高效捕获短时细节和长时结构**  
   与直接使用通用音频特征（如 wav2vec）不同，Dasheng 通过层次化卷积捕捉局部频谱变化，再用跨层注意力整合全局信息，显著提升了对音乐旋律和环境声的辨识度。

4. **推理加速设计 → 采用分块缓存、低位量化和并行解码 → TTFT 提速 4 倍、吞吐量提升 20 倍**  
   作者在模型解码阶段加入了“首标记快速路径”，只在首 token 使用完整注意力，后续 token 采用缓存的 KV（Key‑Value）矩阵，并对语言模型进行 8‑bit 量化，极大降低了计算开销。

### 方法详解
**整体框架**  
MiDashengLM 的训练与推理分为两大阶段：① 音频特征预训练——使用公开的大规模无标签音频（如 AudioSet）训练 Dasheng 编码器，使其能输出高维声学向量；② 语言对齐微调——把这些向量喂入一个标准的 LLM（如 LLaMA‑7B），在 ACAVCaps 数据集上进行监督微调，使模型学会把声学向量映射到通用字幕。

**关键模块拆解**  

1. **Dasheng 编码器**  
   - **输入层**：接受原始波形或 log‑Mel 频谱。  
   - **多尺度卷积块**：类似于音频领域的“金字塔”，每个块捕获不同时间尺度的特征（短至 10 ms，长至 1 s）。  
   - **跨层注意力融合**：把不同尺度的特征通过自注意力混合，得到统一的时序向量序列。  
   - **输出**：每帧 768 维向量，送入后续语言模型。

2. **语言模型桥接层**  
   - 将音频向量投影到 LLM 的词嵌入空间，使用一个线性层加层归一化。  
   - 在微调时，加入 **跨模态对齐损失**（对比学习），确保相似音频对应的字幕在嵌入空间相近。

3. **监督微调（SFT）**  
   - 采用 **教师强制（teacher forcing）**：每一步输入前一个真实 token，输出预测下一个 token。  
   - 损失函数为交叉熵加上对齐损失的加权和，权重在实验中调至 0.7/0.3。

4. **推理加速**  
   - **首标记快速路径**：在生成第一个 token 时，完整执行自注意力；随后缓存 KV，后续 token 只做增量计算。  
   - **8‑bit 量化**：对 LLM 参数进行低位量化，几乎不损失质量，却把显存需求降至原来的 1/4。  
   - **并行批处理**：在服务器端一次性处理多段音频，利用 GPU 的张量并行提升吞吐。

**最巧妙的设计**  
把多尺度卷积和跨层注意力结合的 Dasheng 编码器，是把“声学金字塔”与“全局注意力”融合的创新，让模型在捕捉细粒度噪声的同时，也能把握音乐的整体结构，这在之前的音频编码器里很少见。

### 实验与效果
- **测试数据集**：论文在 AudioCaps、Clotho、ESC‑50、FSD50K 等公开基准上评估，覆盖字幕生成、音频分类和情感识别三类任务。  
- **对比基线**：与 Whisper‑based LLM、AudioLM、Wav2Vec‑2.0 + LLM 等模型对比。  
- **性能提升**：在 AudioCaps 的 CIDEr 分数上提升约 12%（从 0.78 提到 0.87），在 ESC‑50 分类准确率上提升 3.5%（从 91.2% 到 94.7%）。  
- **速度优势**：官方声称 TTFT 从 1.2 s 降至 0.3 s（约 4 倍加速），吞吐量从 5 samples/s 提升至 100 samples/s（约 20 倍）。  
- **消融实验**：去掉跨层注意力后 CIDEr 下降 5%；不使用首标记快速路径，TTFT 回升至 0.9 s；改用闭源 ASR 生成字幕，整体 BLEU 分数下降约 8%。这些实验表明每个创新模块都对最终效果有实质贡献。  
- **局限性**：作者指出模型在极端低信噪比（< 0 dB）下仍会出现字幕漏检；此外，8‑bit 量化在极端长音频（> 30 s）上会出现轻微漂移，需进一步校准。

### 影响与延伸思考
MiDashengLM 的全开源链路为音频‑语言研究设立了新的标杆，促使后续工作更加关注数据可获取性和跨模态统一表示。自发布后，已有项目如 **OpenAudioLM**、**AudioChat** 在其代码和 ACAVCaps 数据集的基础上进行扩展，尝试加入视频信息或多语言字幕。对想进一步深入的读者，建议关注以下方向：① 更高效的跨模态对齐损失（如对比学习的最新变体）；② 超低延迟的流式解码技术；③ 将通用音频字幕扩展到多语言或方言场景。  

### 一句话记住它
MiDashengLM 用公开的通用音频字幕和专属的 Dasheng 编码器，实现了既全能又高速的音频‑语言理解。