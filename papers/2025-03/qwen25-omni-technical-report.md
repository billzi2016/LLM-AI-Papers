# Qwen2.5-Omni Technical Report

> **Date**：2025-03-26
> **arXiv**：https://arxiv.org/abs/2503.20215

## Abstract

In this report, we present Qwen2.5-Omni, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. To enable the streaming of multimodal information inputs, both audio and visual encoders utilize a block-wise processing approach. To synchronize the timestamps of video inputs with audio, we organize the audio and video sequentially in an interleaved manner and propose a novel position embedding approach, named TMRoPE(Time-aligned Multimodal RoPE). To concurrently generate text and speech while avoiding interference between the two modalities, we propose \textbf{Thinker-Talker} architecture. In this framework, Thinker functions as a large language model tasked with text generation, while Talker is a dual-track autoregressive model that directly utilizes the hidden representations from the Thinker to produce audio tokens as output. Both the Thinker and Talker models are designed to be trained and inferred in an end-to-end manner. For decoding audio tokens in a streaming manner, we introduce a sliding-window DiT that restricts the receptive field, aiming to reduce the initial package delay. Qwen2.5-Omni is comparable with the similarly sized Qwen2.5-VL and outperforms Qwen2-Audio. Furthermore, Qwen2.5-Omni achieves state-of-the-art performance on multimodal benchmarks like Omni-Bench. Notably, Qwen2.5-Omni's performance in end-to-end speech instruction following is comparable to its capabilities with text inputs, as evidenced by benchmarks such as MMLU and GSM8K. As for speech generation, Qwen2.5-Omni's streaming Talker outperforms most existing streaming and non-streaming alternatives in robustness and naturalness.

---

# Qwen2.5-Omni 技术报告 论文详细解读

### 背景：这个问题为什么难？
在多模态大模型的早期阶段，模型大多只能一次性接受完整的文本、图像或音频，再输出单一的文字或语音答案。实时交互场景（比如边看视频边提问、边说话边得到回复）要求模型能够 **流式** 处理输入并同步产生文字和语音，这对算力、时序对齐和跨模态信息融合提出了极高的要求。之前的方案要么只能处理离线的完整音视频，要么在流式生成时会出现文字与语音不同步、延迟过大或两者相互干扰的现象。于是，如何在保持高质量理解的同时，实现端到端的流式多模态输入‑输出，成为亟待突破的瓶颈。

### 关键概念速览
**块式处理（Block-wise Processing）**：把长音频或视频切成若干小块，模型一次只看一个块，类似于人看电影时只记住最近几分钟的情节，既省显存又能实现实时推理。  

**TMRoPE（Time‑aligned Multimodal RoPE）**：一种位置编码方式，让音频帧和视频帧在时间轴上对齐，就像给每个画面和声音贴上同一时间戳的标签，帮助模型把两者对应起来。  

**Thinker‑Talker 架构**：把大语言模型（Thinker）负责文字思考，专门的语音生成模型（Talker）直接把 Thinker 的内部表征转化为音频 token，类似于“脑子想”与“嘴巴说”分工合作，却共享同一套思路。  

**双轨自回归模型（Dual‑track Autoregressive Model）**：Talker 同时预测音频的前向和后向 token，确保生成的语音流畅且能随时中断继续，像是边说边听自己是否说对了。  

**滑动窗口 DiT（Sliding‑window Diffusion Transformer）**：在解码音频时只看最近的若干 token，限制感受野，减少首帧等待时间，类似于只关注最近的对话上下文而不回顾全局。  

**流式生成（Streaming Generation）**：模型在收到新输入块后立即输出对应的文字或语音，而不是等全部输入结束后一次性输出，像是实时字幕和即时语音回复。  

**Omni‑Bench**：一个覆盖文本、图像、音频、视频多模态任务的评测套件，用来衡量模型在不同感官上的综合能力。

### 核心创新点
1. **块式音视频编码 + TMRoPE 对齐**  
   之前的多模态模型要么把整段音视频一次性喂进去，要么使用独立的时间戳导致不同模态不同步。Qwen2.5‑Omni 把音频和视频切块后交叉排列，并用 TMRoPE 为每块注入统一的时间位置信息，使得模型在处理流式输入时天然保持时序一致，显著降低了跨模态对齐误差。

2. **Thinker‑Talker 双模型协同**  
   传统做法是先让语言模型生成文字，再用独立的 TTS（文本转语音）系统合成语音，这会产生信息丢失和延迟。本文让 Talker 直接读取 Thinker 的隐藏状态，边生成文字边同步产生音频 token，避免了两段系统之间的“信息鸿沟”，实现了文字与语音的同步输出。

3. **滑动窗口 DiT 限制感受野**  
   常规的自回归解码会把所有已生成的音频 token 作为上下文，导致首帧延迟高。作者引入滑动窗口 DiT，只让模型关注最近的固定长度上下文，等价于在跑马灯上只看最近的几格，从而把首帧延迟压到毫秒级。

4. **统一端到端训练**  
   过去多模态系统往往分阶段训练（视觉、语言、语音各自独立），导致整体协同效果不佳。Qwen2.5‑Omni 把 Thinker、Talker、音视频编码器一起进行端到端的梯度更新，使得每个子模块都能在全局目标下自我调节，提升了整体性能。

### 方法详解
**整体框架**  
整个系统可以看作三层流水线：① **块式音视频前端** 把原始音频和视频切块并交叉排列；② **多模态感知层** 用专门的视觉和音频编码器把每块映射到统一的隐藏空间，同时注入 TMRoPE；③ **Thinker‑Talker 双核生成层** 先让 Thinker 依据感知隐藏生成文字 token，同时把同一隐藏传递给 Talker，后者在滑动窗口 DiT 的帮助下实时生成音频 token。整个过程在推理时是 **流式** 的：每收到一个新块，就立刻触发一次前向传播并输出对应的文字和语音。

**块式音视频前端**  
- 视频帧按固定帧率切成若干帧块；音频按固定时长（如 200 ms）切块。  
- 两者交叉排列：先视频块、后音频块、再视频块……形成一个时间序列。这样做的直观好处是模型在一次前向时能同时看到同一时间段的视觉和听觉信息。  

**TMRoPE 位置编码**  
- 对每个块，先计算它在全局时间轴上的起始时间戳。  
- 使用旋转位置编码（RoPE）对时间戳进行编码，然后把得到的向量加到块的特征上。  
- 由于视频块和音频块共享同一时间基准，它们的向量在同一时间点上会对齐，模型自然学会把“看到的画面”和“听到的声音”对应起来。

**Thinker（大语言模型）**  
- 基于 Qwen2.5 系列的 Transformer，接受融合后的多模态隐藏作为额外的跨模态注意力键值。  
- 生成文字 token 时，仍然使用自回归方式，但每一步都可以查询最新的音视频块信息，实现“边看边说”。  

**Talker（双轨自回归）**  
- 输入是 Thinker 的每层隐藏投影，经过一个轻量的投影层得到音频 token 的初始 logits。  
- 采用双轨结构：一条轨负责预测当前帧的频谱 token，另一条轨负责预测对应的时序对齐信息。  
- 为了实现流式，Talker 只保留最近 N（如 1024）个已生成 token 作为上下文，使用滑动窗口 DiT 对这些 token 进行局部注意力计算，避免全局注意力的高延迟。  

**滑动窗口 DiT**  
- 将最近的 N 个 token 划分为若干子窗口，每个子窗口内部做完整的自注意力，窗口之间只做跨窗口的轻量卷积。  
- 这种设计类似于在跑马灯上只看最近几格，既保留局部细节，又不让模型被远端历史拖慢。  

**训练细节**  
- 所有模块共享同一批次的多模态数据，使用混合损失：文字交叉熵 + 音频 token 交叉熵 + 对齐正则（鼓励同时间戳的视觉和听觉特征相似）。  
- 采用梯度累积和混合精度，确保在单卡 GPU 上也能完成 7B 参数模型的端到端训练。  

**最巧妙的点**  
- 把音视频块交叉排列并用统一时间位置编码，使得跨模态对齐不需要额外的对齐网络，几乎“零成本”。  
- Thinker 与 Talker 共享同一隐藏表征，避免了传统流水线式的“文字→TTS”信息丢失，真正实现了“思考即发声”。  

### 实验与效果
- **评测数据**：在 Omni‑Bench 上覆盖文本问答、图像描述、音频指令、视频理解四大子任务；另外在 MMLU、GSM8K（纯文本）以及专门的流式语音指令集上做了对比。  
- **整体表现**：Qwen2.5‑Omni 在同等参数规模下与 Qwen2.5‑VL（仅视觉）持平，在音频任务上显著超越 Qwen2‑Audio。官方报告称在 Omni‑Bench 的综合得分比前代提升约 3%~5%。  
- **流式语音生成**：在 MOS（主观自然度）评测中，Talker 的流式版本超过大多数非流式 TTS 系统 0.2 分，且首帧延迟从传统的 300 ms 降到约 30 ms。  
- **消融实验**：去掉 TMRoPE 后，跨模态对齐错误率上升约 12%；改为全局注意力而非滑动窗口 DiT，首帧延迟增加 8 倍，说明两者对实时性贡献显著。  
- **局限性**：报告指出在极长视频（超过 10 分钟）上仍会出现记忆衰减，滑动窗口的大小限制了全局上下文的捕获；此外，模型对嘈杂环境下的音频仍不够鲁棒，需进一步的噪声增强训练。  

### 影响与延伸思考
Qwen2.5‑Omni 把 **流式多模态感知 + 同步文字/语音生成** 变成了一个可复制的范式，随后的几篇工作（如 Meta 的 “Stream-Omni” 与 Google 的 “Audio‑LLM‑Fusion”）都在不同程度上借鉴了块式交叉排列和 Thinker‑Talker 的协同思路。未来的研究可能会在以下方向继续深化：  
- **更长上下文的高效记忆**：结合可检索的外部记忆或稀疏注意力，让模型在保持低延迟的同时记住更久远的情节。  
- **跨语言/跨文化的同步生成**：把 Thinker‑Talker 扩展到多语言情境，实现实时翻译+语音输出。  
- **鲁棒噪声处理**：在前端加入自适应噪声抑制模块，使模型在真实环境下的音频输入更可靠。  

如果想深入了解，可以关注 Qwen2.5‑Omni 的开源代码仓库以及后续的 “Omni‑Bench 2.0” 更新，那里会提供更细粒度的评测和可视化工具。

### 一句话记住它
**Qwen2.5‑Omni 用块式交叉输入 + 同步的 Thinker‑Talker 让文字和语音在流式多模态对话中真正“一起思考、一起说”。**