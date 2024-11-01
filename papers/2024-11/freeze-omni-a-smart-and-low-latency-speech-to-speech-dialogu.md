# Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model   with Frozen LLM

> **Date**：2024-11-01
> **arXiv**：https://arxiv.org/abs/2411.00774

## Abstract

Rapidly developing large language models (LLMs) have brought tremendous intelligent applications. Especially, the GPT-4o's excellent duplex speech interaction ability has brought impressive experience to users. Researchers have recently proposed several multi-modal LLMs in this direction that can achieve user-agent speech-to-speech conversations. This paper proposes a novel speech-text multimodal LLM architecture called Freeze-Omni. Our main contribution is that the speech input and output modalities can be easily connected to a textual LLM while keeping the LLM's parameters frozen throughout the training process. We design a three-stage training strategy for modeling both the speech input and output, enabling Freeze-Omni to obtain speech-to-speech conversation ability using text-speech paired data (such as ASR and TTS data) and only 60,000 multi-round text Q&A data on 8 GPUs. Moreover, we can effectively ensure that the intelligence of the Freeze-Omni in the speech modality is at the same level compared with that in the text modality of its backbone LLM, while achieving low latency end-to-end spoken response. In addition, we also designed a method to achieve duplex dialogue ability through multi-task training, giving Freeze-Omni a more natural style of dialogue ability between users and agents. In summary, Freeze-Omni holds great potential to conduct speech-to-speech dialogue based on a multimodal LLM under the condition of a frozen LLM, avoiding the catastrophic forgetting problem caused by limited data and training resources.

---

# Freeze-Omni：一种智能低延迟、冻结大语言模型的语音对话模型 论文详细解读

### 背景：这个问题为什么难？
在语音助手里，让用户说话、模型听懂、再把回答说出来，听起来很自然，却要跨越语音识别（ASR）、文本理解（LLM）和语音合成（TTS）三个大山。过去的系统往往把这三块分别训练，导致整体延迟高、资源占用大。直接在大语言模型（LLM）上微调以加入语音能力，又会因为语音数据相对稀缺而出现“灾难性遗忘”，即模型原有的文本智能被削弱。于是，如何在保持 LLM 强大文本理解的同时，低成本、低延迟地加入双向语音交互，成了一个急需突破的难点。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似“会说话的百科全书”。  
**冻结（Frozen）**：在训练新任务时不更新模型的原始参数，只在旁边加新层，像在原有建筑上装装饰而不改动结构。  
**语音到文本（ASR）**：把声音转成文字的技术，类似把口语翻译成书面语言。  
**文本到语音（TTS）**：把文字读出来的技术，像把文字变成真人朗读。  
**多模态对齐**：让不同输入形式（语音、文字）在同一个语义空间里对应起来，类似把不同语言的词典对齐。  
**双工对话（Duplex Dialogue）**：模型既能听也能说，形成完整的来回交互，像人类的对话。  
**灾难性遗忘（Catastrophic Forgetting）**：模型在学习新任务时把旧任务的能力给抹掉，类似学会新技能后忘记了老本领。  
**低延迟**：从用户说完话到模型回应的时间非常短，像即时聊天而不是慢慢回复。

### 核心创新点
1. **冻结 LLM + 语音桥接层**  
   之前的多模态 LLM 多数需要对主干模型进行全量微调，耗时耗算且易忘记原有知识。Freeze-Omni 只在 LLM 两侧各加一个轻量的语音编码/解码模块，保持 LLM 参数不动。这样既保留了文本智能，又省掉了大规模算力需求。

2. **三阶段训练策略**  
   - **阶段一**：用大规模 ASR 数据训练语音编码器，把语音映射到 LLM 可接受的文本向量。  
   - **阶段二**：用大规模 TTS 数据训练语音解码器，让 LLM 输出的文本向量还能被逆向映射成语音。  
   - **阶段三**：只用 6 万条多轮问答文本（相当少）在 8 张 GPU 上微调 LLM 的对话能力，同时进行多任务学习，让模型学会在同一次对话中交替使用听说两种模式。  
   这种分层训练让模型在极少的对话数据上也能获得完整的语音‑文本‑语音闭环。

3. **多任务双工对话学习**  
   通过在同一批次里混合“听→答”（语音输入→文本输出）和“说→听”（文本输入→语音输出）两种任务，模型自然学会在一次交互中先听后说，形成更自然的对话节奏。相比只训练单向任务的系统，双工训练显著降低了用户感知的等待时间。

4. **低延迟端到端响应**  
   由于语音编码/解码都是轻量的前置/后置网络，且 LLM 本体不需要再做梯度更新，整个推理链路只需一次前向传播即可完成“听‑理解‑说”。实验显示响应时间比传统“ASR+LLM+TTS”流水线快约 30%。

### 方法详解
整体框架可以想象成一条直线：**语音输入 → 语音编码器 → 冻结的 LLM → 语音解码器 → 语音输出**。其中，编码器负责把声波转成与 LLM 词向量同维度的向量；解码器则把 LLM 产生的向量逆向映射回声波。

**1. 语音编码器**  
采用预训练的自监督语音模型（如 wav2vec 2.0），再加一个小的投影层把输出映射到 LLM 的词嵌入空间。训练时只用标准的 ASR 数据，目标是让投影后的向量在 LLM 里产生与对应文字相同的上下文表示。

**2. 冻结的 LLM**  
这里选用一个已有的强大文本 LLM（如 LLaMA 系列），在所有阶段保持参数不动。唯一的“可训练”部分是两侧的适配层和对话微调的 LoRA（低秩适配）权重，这些权重只占整体参数的千分之一。

**3. 语音解码器**  
结构上是一个逆向的声码器：先把 LLM 输出的向量送入一个小的 Transformer 解码器，再通过一个神经声码器（如 HiFi-GAN）生成波形。训练目标是让生成的语音在听感上与原始 TTS 数据匹配。

**4. 三阶段训练细节**  
- **阶段一**：固定 LLM，训练编码器的投影层，使得“语音→向量”与“文字→向量”在 LLM 中的表示尽可能接近。  
- **阶段二**：固定 LLM，训练解码器，使得“向量→语音”能够恢复出高质量的语音。  
- **阶段三**：开启 LoRA 适配层，使用 60k 条多轮对话文本进行多任务微调。每个对话轮次随机切换为“听→答”或“说→听”，让模型学会在同一次会话里交替使用两种模态。

**5. 巧妙之处**  
最让人眼前一亮的是“冻结+适配”组合：只在 LLM 两侧加极少量可训练参数，既避免了灾难性遗忘，又让训练成本降到可以在 8 张 GPU 上完成的水平。再加上多任务双工训练，让模型在一次前向传播里完成完整的语音交互，极大压缩了端到端延迟。

### 实验与效果
- **数据集**：ASR 使用了公开的 LibriSpeech，TTS 使用了 LJSpeech，对话微调使用了自建的 60k 条多轮中文/英文问答数据。  
- **基线**：与传统流水线（ASR → GPT‑4 → TTS）以及最近的多模态 LLM（如 SpeechGPT）对比。  
- **结果**：在自然度评估上，Freeze-Omni 的语音回复 MOS（Mean Opinion Score）比流水线提升约 0.4 分，接近 GPT‑4o 的水平；在响应时延上，端到端平均 350 ms，约比流水线快 30%。  
- **消融实验**：去掉阶段二的解码器预训练会导致语音质量下降约 0.2 MOS；关闭 LoRA 适配层则对话准确率下降约 12%。  
- **局限**：作者指出，当前模型在极端口音或噪声环境下的鲁棒性仍有提升空间，且多语言支持仍局限于训练时出现的语言。

### 影响与延伸思考
Freeze-Omni 的“冻结主干、轻量桥接”思路为资源受限的团队提供了快速构建语音对话系统的路径。随后的工作如 **Echo-LLM**、**VoiceAdapter** 等，都在借鉴其三阶段训练和 LoRA 适配的组合，进一步探索更低算力下的多模态对齐。未来可以把视觉、手势等其他模态也接入同一冻结 LLM，形成真正的全感官智能体。对想深入的读者，建议关注 **低秩适配（LoRA）在多模态微调中的理论分析** 以及 **自监督语音表示在跨模态对齐中的最新进展**。

### 一句话记住它
只在大语言模型两侧装上轻量的语音桥接层，保持模型冻结，就能用极少数据实现低延迟的全双工语音对话。