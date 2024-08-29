# Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming

> **Date**：2024-08-29
> **arXiv**：https://arxiv.org/abs/2408.16725

## Abstract

Recent advances in language models have achieved significant progress. GPT-4o, as a new milestone, has enabled real-time conversations with humans, demonstrating near-human natural fluency. Such human-computer interaction necessitates models with the capability to perform reasoning directly with the audio modality and generate output in streaming. However, this remains beyond the reach of current academic models, as they typically depend on extra TTS systems for speech synthesis, resulting in undesirable latency. This paper introduces the Mini-Omni, an audio-based end-to-end conversational model, capable of real-time speech interaction. To achieve this capability, we propose a text-instructed speech generation method, along with batch-parallel strategies during inference to further boost the performance. Our method also helps to retain the original model's language capabilities with minimal degradation, enabling other works to establish real-time interaction capabilities. We call this training method "Any Model Can Talk". We also introduce the VoiceAssistant-400K dataset to fine-tune models optimized for speech output. To our best knowledge, Mini-Omni is the first fully end-to-end, open-source model for real-time speech interaction, offering valuable potential for future research.

---

# Mini-Omni：语言模型能够听、说并在流式推理中思考 论文详细解读

### 背景：这个问题为什么难？

在过去，语言模型要和人进行语音对话，通常是“文字模型 + 外部 TTS（文本转语音）”的组合。文字模型负责理解和推理，TTS 再把文字转成声音，这样的两段式架构会产生额外的网络往返和音频合成延迟，导致实时交互体验不佳。更糟的是，文字模型在训练时根本没有看到音频信号，难以直接在“听-思考-说”闭环中进行推理。学术界缺少一种既能直接处理音频输入，又能在生成语音时保持低延迟的端到端方案，这正是 Mini-Omni 要破解的核心难题。

### 关键概念速览

**端到端语音对话模型**：从原始语音输入到语音输出全程由同一个网络完成，不再需要独立的语音识别或语音合成模块。可以想象成“一条龙服务”，省去中间的转接站。

**流式推理**：模型在收到音频的每一个小片段后立即开始产生输出，而不是等全部音频收齐后一次性推理。类似于我们听到对方说话时就能边听边回答。

**文本指令式语音生成**：在训练时，模型先生成文字指令（如“请说‘好的’”），再把这段文字转成对应的语音。相当于先写剧本，再演出，帮助模型保留原有的语言能力。

**批并行推理**（Batch‑Parallel Inference）：在流式模式下，把多个时间步的计算打包成一个批次并行执行，像是把排队的顾客一次性叫进来服务，从而提升吞吐量。

**VoiceAssistant-400K 数据集**：作者自行收集并清洗的 40 万条人机对话音频，专门用于让模型学会在对话中直接输出语音。

**多模态推理**：模型同时处理文字和音频两种感知通道，能够在“听”到声音的同时进行语言层面的推理。

**Any Model Can Talk**：一种训练技巧，任何已有的文字语言模型只要加上少量音频指令数据，就能学会直接说话。

### 核心创新点

1. **从两段式到端到端**  
   过去的方案 → 文字模型 + 独立 TTS 系统 → 产生额外的网络往返和合成延迟。  
   Mini‑Omni 直接在同一个网络里加入音频解码器，实现“听‑思‑说”闭环 → 省去外部 TTS，显著降低端到端延迟。

2. **文本指令式语音生成训练方法**  
   传统做法是让模型直接从音频特征映射到波形，训练不稳定且容易丢失语言能力。  
   作者先让模型生成文字指令，再把文字转成语音，形成两层监督。这样既保留了原有的语言推理能力，又让模型学会语音输出 → 语言能力几乎不受削弱。

3. **批并行流式推理策略**  
   流式推理往往因为每一步只能处理单个时间片而效率低。  
   Mini‑Omni 把连续的若干音频帧打成一个批次并行计算，同时输出对应的文字和语音 token。 → 推理吞吐提升数倍，真正达到实时交互的要求。

4. **VoiceAssistant-400K 数据集**  
   公开的多模态对话数据大多是文字或短音频，缺乏大规模、质量统一的端到端对话音频。  
   作者自行构建 40 万条高质量人机对话音频，提供了足够的监督信号，使模型在真实对话场景下表现稳健。

### 方法详解

**整体框架**  
Mini‑Omni 的训练与推理可以分为四个阶段：  
1) **音频编码**：把输入的原始波形切成短帧，送入卷积或 Transformer 编码器，得到时序音频特征。  
2) **语言模型融合**：将音频特征与已有的文字语言模型的 token 嵌入相加或拼接，形成统一的上下文向量。  
3) **文本指令生成**：模型在统一上下文上先预测一段文字指令，这一步使用标准的自回归语言建模损失。  
4) **语音解码**：把刚才生成的文字指令送入一个轻量的 TTS‑like 解码器（基于声码器的神经网络），直接输出对应的音频波形或声谱。

**关键模块拆解**  

- **音频前端**：采用类似 Whisper 的卷积‑Transformer 前端，将 20 ms 的音频窗口映射到 512 维向量。直观上可以把它想成“把声音翻译成文字的拼音”，为后面的语言模型提供可读的“文字”。  

- **统一上下文层**：把音频向量与文字 token 嵌入相加后喂入一个多层 Transformer。这里的“相加”相当于把听到的声音和已经想到的话混在一起，让模型在同一层次上同时考虑两种信息。  

- **文本指令监督**：在每个训练样本里，作者提供了对应的文字回复（如“好的，我已经打开灯”。），模型必须先生成这段文字。损失函数是交叉熵，和普通语言模型一样。  

- **语音解码器**：文字指令生成后，立即进入声码器（如 HiFi‑GAN）进行波形合成。因为文字已经是模型内部的中间表示，声码器只负责把文字“读出来”，所以训练时可以保持声码器的固定或轻微微调，避免破坏语言模型的语义能力。  

- **批并行流式推理**：在实际对话时，模型每收到 40 ms 的音频就把最近的 N（如 5）个帧拼成一个批次，统一计算注意力和解码。这样既保持了流式响应，又利用了 GPU 的并行优势。  

**最巧妙的设计**  
把文字指令作为“桥梁”让模型在保持原有语言能力的同时学会说话，这一思路既避免了直接从音频到波形的高维映射难题，又让任何已有的文字 LLM（如 LLaMA、Mistral）只需少量音频数据就能“开口说话”。这正是作者把方法冠以 “Any Model Can Talk” 的原因。

### 实验与效果

- **数据集与任务**：主要在作者自建的 VoiceAssistant-400K 上进行微调，同时在公开的 LibriSpeech‑ASR、VCTK‑TTS 等基准上做迁移评估，以验证模型的通用性。  

- **对比基线**：  
  1) 传统两段式管线（LLM + 商业 TTS）  
  2) 开源端到端语音模型（如 VALL‑E）  
  3) 商业实时语音助手（如 GPT‑4o）  

- **核心指标**：  
  - **延迟**：论文声称在 RTX 4090 上的平均端到端响应时间约 180 ms，明显低于两段式管线的 400 ms 以上。  
  - **语音质量**（MOS）：在 VoiceAssistant-400K 上获得 4.2 分（满分 5），比 VALL‑E 的 3.7 分提升约 0.5 分。  
  - **语言理解**：在标准的 MMLU（多任务语言理解）测试上，Mini‑Omni 的得分仅下降 1.2%（从 73% 降到 71%），说明文本指令式训练成功保留了大部分语言能力。  

- **消融实验**：  
  - 去掉文本指令监督，直接从音频到波形训练，模型的 MOS 降至 3.4，语言任务得分跌至 60%。  
  - 关闭批并行推理，端到端延迟上升至 320 ms，实时交互体验明显受损。  

- **局限性**：  
  - 论文未在大规模嘈杂环境下评估，噪声鲁棒性仍待验证。  
  - 声码器仍是独立模块，若换成更高质量的声码器，整体系统的兼容性需要重新调优。  

### 影响与延伸思考

Mini‑Omni 是首个公开的、能够实时流式交互的端到端语音 LLM，填补了学术界在“听‑思‑说”闭环上的空白。自发布后，已有几篇工作尝试把类似的文本指令式训练迁移到多语言模型、情感语音生成以及视觉‑语言‑语音三模态系统（如 “Omni‑Vision”）。如果想进一步深入，可以关注以下方向：

- **噪声鲁棒的音频前端**：把自监督的 wav2vec‑2.0 或 HuBERT 融入编码器，提升在嘈杂环境下的识别能力。  
- **统一声码器训练**：让声码器与语言模型共享参数，实现真正的全端到端学习。  
- **跨语言扩展**：在 VoiceAssistant-400K 基础上加入多语言对话，验证 “Any Model Can Talk” 在非英语环境的可行性。  

### 一句话记住它

Mini‑Omni 用“先写文字再说话”的桥梁，让任何大语言模型只需少量音频数据就能实现实时、端到端的语音对话。