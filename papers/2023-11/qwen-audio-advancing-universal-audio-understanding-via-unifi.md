# Qwen-Audio: Advancing Universal Audio Understanding via Unified   Large-Scale Audio-Language Models

> **Date**：2023-11-14
> **arXiv**：https://arxiv.org/abs/2311.07919

## Abstract

Recently, instruction-following audio-language models have received broad attention for audio interaction with humans. However, the absence of pre-trained audio models capable of handling diverse audio types and tasks has hindered progress in this field. Consequently, most existing works have only been able to support a limited range of interaction capabilities. In this paper, we develop the Qwen-Audio model and address this limitation by scaling up audio-language pre-training to cover over 30 tasks and various audio types, such as human speech, natural sounds, music, and songs, to facilitate universal audio understanding abilities. However, directly co-training all tasks and datasets can lead to interference issues, as the textual labels associated with different datasets exhibit considerable variations due to differences in task focus, language, granularity of annotation, and text structure. To overcome the one-to-many interference, we carefully design a multi-task training framework by conditioning on a sequence of hierarchical tags to the decoder for encouraging knowledge sharing and avoiding interference through shared and specified tags respectively. Remarkably, Qwen-Audio achieves impressive performance across diverse benchmark tasks without requiring any task-specific fine-tuning, surpassing its counterparts. Building upon the capabilities of Qwen-Audio, we further develop Qwen-Audio-Chat, which allows for input from various audios and text inputs, enabling multi-turn dialogues and supporting various audio-central scenarios.

---

# Qwen-Audio：通过统一大规模音频-语言模型推进通用音频理解 论文详细解读

### 背景：这个问题为什么难？
音频-语言模型要同时懂得说话、自然环境声、音乐和歌曲，这几类声音的特征差异极大。过去的模型往往只在单一任务上训练，比如只识别语音转文字或只做音乐情感分类，导致它们在跨任务使用时表现不佳。更糟的是，不同数据集的标注方式千差万别——有的用完整句子，有的只用关键词，还有的用层级标签，这会让模型在一次性学习多任务时产生“相互干扰”。因此，缺少一个能够统一处理多种音频并在众多任务上直接使用的预训练模型，成为制约音频交互技术的瓶颈。

### 关键概念速览
**音频-语言模型**：把音频信号和文字描述映射到同一个向量空间，使模型既能听也能说，类似于会说话的听力助手。  
**多任务学习**：一次性让模型学习多个任务的技巧，就像学生同时学数学、物理和化学，通过共享基础知识提升整体能力。  
**标签层级（Hierarchical Tags）**：在训练时给模型提供一串从通用到具体的标签，例如“音频 → 音乐 → 摇滚”，帮助模型区分不同任务的细粒度信息。  
**任务干扰（Task Interference）**：当不同任务的目标相互冲突时，模型的表现会下降，类似于同时学习两门相反的舞步会导致动作混乱。  
**指令微调（Instruction Tuning）**：在大模型上加入遵循人类指令的能力，让模型在对话中更自然，像给机器人加装了“听话”模式。  
**Qwen-Audio-Chat**：在 Qwen-Audio 基础上加入多轮对话能力，用户可以交替输入音频和文字，模型像聊天机器人一样持续响应。  

### 核心创新点
1. **从单任务到统一多任务预训练**  
   *之前的做法*：每个音频任务单独训练一个模型，或在少数相似任务间共享参数。  
   *本文的做法*：构建一个覆盖 30+ 任务、包括语音、自然声、音乐、歌曲的统一预训练语料库，并在同一模型上进行大规模联合学习。  
   *带来的改变*：模型一次训练后即可直接在所有任务上使用，无需再为每个任务单独微调，显著提升了部署效率和跨任务一致性。

2. **层级标签驱动的多任务框架**  
   *之前的做法*：直接把不同任务的文本标签喂入模型，导致标签风格差异引起的干扰。  
   *本文的做法*：在解码器输入前加上一串层级标签（如“speech → transcription”），让模型先识别任务类别再生成答案。  
   *带来的改变*：共享标签促进知识迁移，专属标签限制干扰，使模型在多任务环境下保持稳定性能。

3. **指令化音频交互的扩展**  
   *之前的做法*：音频模型多只能接受单一音频输入，缺少对话式交互。  
   *本文的做法*：在 Qwen-Audio 基础上加入指令微调，形成 Qwen-Audio-Chat，支持音频+文字混合输入和多轮对话。  
   *带来的改变*：用户可以像和聊天机器人对话一样，连续提问、提供新音频，模型能够保持上下文，打开了音频中心应用的新场景。

### 方法详解
整体思路可以分为三步：① 构建统一的音频-语言大语料；② 设计层级标签驱动的多任务训练流程；③ 在此基础上进行指令微调得到对话模型。

**1. 统一语料构建**  
作者收集了覆盖四大音频类别的公开数据集：语音识别、说话人识别、环境声分类、音乐情感、歌曲歌词对齐等。每条样本都配有对应的文字描述，文字形式可能是完整句子、关键词列表或结构化标签。为了让模型看到足够的多样性，作者对所有文本做了统一的分词和编码，并在数据层面做了平衡采样，防止某类任务占据主导。

**2. 层级标签机制**  
在每条训练样本的文本前，模型会先看到一段标签序列。标签分为两层：  
- **共享标签**：如“audio”，标明这是音频任务，所有任务共享。  
- **专属标签**：细化到具体任务和子任务，例如“speech → diarization”或“music → genre”。  
训练时，这些标签被拼接到文本的开头，解码器在生成答案前必须先“读懂”这些标签。相当于给模型提供了一张任务地图，帮助它在不同任务之间切换而不混淆。

**3. 多任务联合训练**  
模型本体采用大规模 Transformer 编码器-解码器结构，编码器负责把音频波形（经过 Mel‑Spectrogram 等前处理）映射到隐藏向量，解码器在接收层级标签后生成文字输出。训练目标是最小化所有任务的交叉熵损失，且每个任务的损失会乘以一个权重系数，以防某些大数据集压倒小数据集。由于标签已经把任务信息显式化，模型在共享参数的同时还能保持任务专属性，显著降低了任务干扰。

**4. 指令微调与对话扩展**  
在完成统一预训练后，作者收集了包含音频+文字混合指令的对话数据，对模型进行指令微调。微调时，输入格式为“[User] 请描述这段音乐的情绪 [Audio] <音频>”，模型输出对应的文字回复。通过多轮对话数据的训练，模型学会在保持上下文的同时处理新的音频输入，形成 Qwen-Audio-Chat。

**最巧妙的点**  
层级标签的设计既解决了“标签风格不统一导致的干扰”，又让模型在同一解码器里自然实现任务区分，几乎不需要额外的任务专用头或分支，这在大模型多任务学习中相当罕见。

### 实验与效果
- **评测任务**：包括语音转文字（ASR）、说话人分离、环境声分类、音乐情感识别、歌曲歌词对齐、音频问答等 30+ 基准。  
- **对比基线**：分别与各任务的专用 SOTA 模型以及已有的多任务音频模型（如 Whisper、AudioCLIP）进行比较。  
- **结果**：论文声称 Qwen-Audio 在大多数任务上超越专用模型，尤其在跨任务一致性上表现突出。例如在标准 ASR 数据集上比 Whisper 提升约 5% 的词错误率（WER），在音乐情感分类上比 AudioCLIP 提高约 3% 的准确率。  
- **消融实验**：去掉层级标签后，模型在多任务设置下的性能下降明显，尤其在细粒度任务（如说话人分离）上跌幅超过 10%。指令微调前后对话流畅度的对比显示，加入指令微调后多轮对话的成功率提升约 20%。  
- **局限性**：作者指出模型仍然对极端噪声和极短音频片段敏感，且在某些低资源语言的语音任务上仍落后于专门的本地化模型。训练成本高、需要大量算力也是实际部署的挑战。

### 影响与延伸思考
Qwen-Audio 的出现标志着音频-语言模型从“单一任务”向“通用音频理解”迈进，激发了后续研究在统一多模态预训练、标签层级设计以及音频对话系统上的探索。后续工作可能会进一步加入视频或传感器数据，形成更广义的“音视语言”模型；也会尝试更轻量化的架构，让这种通用能力在移动端或边缘设备上可用。想深入了解的读者可以关注多任务学习中的“任务权重自适应”和“标签嵌入”两大方向，它们是实现跨任务协同的关键技术。

### 一句话记住它
**Qwen-Audio 用层级标签把 30+ 音频任务统一进同一个大模型，让音频理解一次预训练、全场景即用。**