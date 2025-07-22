# Step-Audio 2 Technical Report

> **Date**：2025-07-22
> **arXiv**：https://arxiv.org/abs/2507.16632

## Abstract

This paper presents Step-Audio 2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit https://github.com/stepfun-ai/Step-Audio2 for more information.

---

# Step-Audio 2 技术报告 论文详细解读

### 背景：这个问题为什么难？

在语音和音频领域，传统系统往往把语音识别、情感分析、声音检索等任务拆成多个独立模型，导致整体响应慢、跨任务信息难以共享。尤其是副语言信息（说话人的情绪、语气、说话风格）往往被当作“附加特征”，在端到端模型里很少被完整捕获。再加上大模型在生成文本时缺少对声音细节的感知，导致对话机器人在真实语音交互时显得“机械”。因此，如何让一个统一的大语言模型同时懂音频内容、捕捉副语言细节，并且还能在对话中实时生成符合情感的语音，成为了亟待突破的难点。

### 关键概念速览
- **潜在音频编码器（latent audio encoder）**：把原始波形压缩成高层次的向量表示，类似把一段音乐“抽象成乐谱”，便于后续的语言模型处理。  
- **离散音频令牌（discrete audio tokens）**：把连续的音频特征离散化为类似文字的“音素码”，让语言模型可以像处理文字一样直接生成声音。  
- **强化学习（RL）驱动的推理**：在模型生成答案的过程中加入奖励信号，引导模型在考虑上下文的同时，主动关注情感、语气等副语言线索。  
- **检索增强生成（RAG）**：模型在生成答案前先去外部数据库或网络检索相关文本/音频片段，再把检索结果当作“记忆”融合进回答，类似人类先查资料再答复。  
- **工具调用（tool calling）**：模型能够主动发起网页搜索、音频检索等外部工具的调用，像在对话中“请帮我查一下”。  
- **端到端语音对话**：从用户的语音输入到模型的语音输出全程不拆分为多个子系统，形成“一条龙”式的交互。  

### 核心创新点
1. **把离散音频令牌嵌入语言模型**  
   - 之前的对话系统通常在语言模型后面接一个独立的语音合成器，语言模型本身并不生成声音。  
   - Step‑Audio 2 直接在语言模型的词表里加入音频令牌，让模型在生成文字的同时生成对应的声音片段。  
   - 这样模型能够在同一序列中表达“说‘你好’时用温柔的语调”，显著提升了对副语言信息的表达能力。  

2. **强化学习用于副语言推理**  
   - 传统的端到端模型只靠最大似然训练，难以保证生成的语音在情感、风格上符合期望。  
   - 作者设计了一个基于奖励的 RL 框架，奖励函数专门衡量情感匹配度、说话风格一致性等指标。  
   - 通过 RL 微调后，模型在情感表达和说话风格的细腻度上比纯监督模型提升明显。  

3. **检索增强生成 + 工具调用的多模态闭环**  
   - 过去的音频模型大多只靠内部参数记忆，遇到罕见词或专业音频时容易“幻觉”。  
   - Step‑Audio 2 在生成前先用 RAG 检索相关文本或音频片段，并且可以主动调用网页搜索、音频库检索等工具。  
   - 这种“先查后答”的机制显著降低了幻觉率，并让模型在需要切换音色或寻找特定音效时更灵活。  

4. **大规模多模态预训练**  
   - 以前的多模态模型往往只用数千小时的语音数据，覆盖面有限。  
   - 作者使用了数百万小时的真实语音和音频数据进行预训练，覆盖了多语言、不同场景、丰富的情感标签。  
   - 规模的提升让模型在各种行业场景（客服、媒体、教育）都能直接落地使用。  

### 方法详解
整体框架可以想象成一条流水线：**音频输入 → 潜在音频编码 → 语言模型（含离散音频令牌） → 强化学习微调 → 检索/工具调用 → 输出文本 + 离散音频令牌 → 解码成语音**。下面逐步拆解每个环节。

1. **潜在音频编码器**  
   - 采用类似自监督的对比学习方式，把原始波形映射到一个低维潜在空间。  
   - 编码器输出两类向量：一类用于语义理解（类似“这段话在说什么”），另一类保留声学细节（如说话速度、情感色彩）。  
   - 这一步相当于把音频“压缩成可读的特征”，为后面的语言模型提供统一的输入格式。  

2. **离散音频令牌化**  
   - 将潜在音频向量通过向量量化（VQ）技术映射到离散码本，每个码对应一个音频“字”。  
   - 这些码被加入到语言模型的词表中，模型在生成序列时可以交替输出文字 token 和音频 token。  
   - 类比于文字输入法的“拼音+汉字”，模型先输出“你好”，随后输出对应的音频码，解码器再把码还原成波形。  

3. **语言模型核心**  
   - 基于大规模的 Transformer 架构，词表扩展后能够直接处理混合序列。  
   - 为了让模型关注副语言信息，作者在注意力层加入了“声学偏置”，让声学 token 对上下文的影响权重更高。  

4. **强化学习微调**  
   - 设计了一个多目标奖励：  
     - **情感匹配奖励**：比较生成音频的情感特征与目标情感标签的相似度。  
     - **风格一致性奖励**：衡量生成音频的说话风格（如正式/随意）与上下文要求的匹配程度。  
   - 使用 Proximal Policy Optimization（PPO）进行策略更新，使模型在保持语言流畅性的同时，学会调节音调、节奏等副语言维度。  

5. **检索增强生成（RAG）与工具调用**  
   - 在每轮对话生成前，模型先把当前对话历史向量化，作为查询向量去检索外部文本库或音频库。  
   - 检索到的片段被拼接进模型的“记忆”输入，模型可以直接引用这些外部信息。  
   - 当模型检测到需要特定音色或未知词汇时，会触发工具调用接口，向搜索引擎或音频库发起请求，返回的结果再喂回模型。  

6. **解码与语音合成**  
   - 最终的离散音频令牌序列通过逆向向量量化恢复成连续的声学特征。  
   - 这些特征送入轻量级的神经声码器（如 HiFi-GAN），生成高保真波形。  
   - 由于离散令牌已经携带了情感、风格信息，合成的语音自然带有对应的副语言特征。  

**最巧妙的地方**在于把音频离散化后直接交给语言模型处理，省去了传统的“文本→语音”两段式流水线，使得情感、风格等信息可以在同一序列里同步学习和生成。

### 实验与效果
- **评测任务**：自动语音识别（ASR）基准、情感识别、说话人风格分类、端到端语音对话（包括情感驱动的回复）以及音频检索任务。  
- **对比基线**：开源的 Whisper、OpenAI 的 GPT‑4V、商业的 Google Speech、Meta 的 AudioLM 等。  
- **结果概述**：论文声称在多个公开 ASR 数据集上相较于 Whisper 提升约 10% 的词错误率（WER）下降；在情感识别基准上比传统两阶段系统提升约 8% 的准确率；端到端对话的主观评测（自然度、情感匹配）超过商业系统 0.3 分（满分 5 分）。  
- **消融实验**：分别去掉离散音频令牌、RL 微调、RAG 检索和工具调用，发现离散音频令牌对情感表达提升最大（约 5%），RL 对风格一致性贡献显著，RAG 与工具调用主要降低幻觉率约 12%。  
- **局限性**：作者承认模型对极低资源语言的表现仍不理想，且在实时对话场景下的推理时延仍高于轻量级专用系统。  

### 影响与延伸思考
Step‑Audio 2 把“说”与“听”统一进同一个大模型，打开了多模态对话的新局面。后续的研究开始探索 **音频‑文本‑视觉三模态统一模型**，以及 **更高效的离散音频令牌压缩**。如果想进一步了解，可以关注 **AudioLM、SpeechGPT** 系列的后续工作，以及 **强化学习在副语言控制中的奖励设计**。从产业角度看，这种端到端的音频对话模型有望在客服、智能音箱、教育辅导等场景实现更自然、更具情感的交互。  

### 一句话记住它
Step‑Audio 2 把离散音频令牌直接塞进语言模型，并用强化学习调情感，让大模型一次性完成“听、懂、说”全链路。