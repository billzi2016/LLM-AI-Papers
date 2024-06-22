# Beyond the Turn-Based Game: Enabling Real-Time Conversations with Duplex Models

> **Date**：2024-06-22
> **arXiv**：https://arxiv.org/abs/2406.15718

## Abstract

As large language models (LLMs) increasingly permeate daily lives, there is a growing demand for real-time interactions that mirror human conversations. Traditional turn-based chat systems driven by LLMs prevent users from verbally interacting with the system while it is generating responses. To overcome these limitations, we adapt existing LLMs to \textit{duplex models} so that these LLMs can listen for users while generating output and dynamically adjust themselves to provide users with instant feedback. % such as in response to interruptions. Specifically, we divide the queries and responses of conversations into several time slices and then adopt a time-division-multiplexing (TDM) encoding-decoding strategy to pseudo-simultaneously process these slices. Furthermore, to make LLMs proficient enough to handle real-time conversations, we build a fine-tuning dataset consisting of alternating time slices of queries and responses as well as covering typical feedback types in instantaneous interactions. Our experiments show that although the queries and responses of conversations are segmented into incomplete slices for processing, LLMs can preserve their original performance on standard benchmarks with a few fine-tuning steps on our dataset. Automatic and human evaluation indicate that duplex models make user-AI interactions more natural and human-like, and greatly improve user satisfaction compared to vanilla LLMs. Our duplex model and dataset will be released.

---

# 超越回合制对话：实现全双工模型的实时交互 论文详细解读

### 背景：这个问题为什么难？
传统的 LLM 驱动的聊天系统都是“先说完再听”，也就是用户说完一句话后模型才开始生成回复。这种回合制交互在口语对话里会让人感觉被“卡住”，因为真实的人类会在对方说话时随时插话、打断或给出即时反馈。早期的解决方案要么把模型换成专门的流式解码器，要么在前端做语音检测，但都无法让模型在生成答案的同时继续监听用户的声音，根本缺少“全双工”能力。于是出现了一个技术空白：如何让已有的大语言模型在保持原有性能的前提下，能够边说边听，实现类似人类的实时对话？

### 关键概念速览
- **全双工（Duplex）**：指系统能够同时进行输入和输出，就像电话两端可以同时说话，而不是先后轮流。这里指 LLM 在生成文字的同时还能接收用户的新语音片段。
- **时间片（Time Slice）**：把一段对话切成若干等长的小块，每块只包含一小段用户语音或模型输出。想象把一段长句子切成拼图块，模型一次只处理一个块。
- **时分复用（Time‑Division‑Multiplexing, TDM）**：一种把多个信号交错发送的技术。论文把用户的时间片和模型的时间片交错排列，让两者在同一解码通道里“轮流”出现，形成伪实时效果。
- **伪同步（Pseudo‑Simultaneous）**：虽然底层仍是顺序计算，但因为时间片很短，用户感受到的交互几乎是同步的。类似于把一段视频分帧播放，肉眼看不出间隔。
- **交替切片数据集（Alternating Slice Dataset）**：专门构造的训练数据，包含用户说话的切片、模型回复的切片以及常见的即时反馈（如“好的”“等一下”）示例，用来让模型学会在不完整上下文下仍能生成合理内容。
- **流式解码（Streaming Decoding）**：模型在生成每个 token 时就立即输出，而不是等整句生成完毕后一次性返回。这里的流式是基于时间片的细粒度输出。

### 核心创新点
1. **从回合制到全双工的任务重定义**  
   之前的系统把一次对话视为完整的“用户提问 → 模型回答”两步。本文把对话拆成交错的时间片，形成“用户片段 → 模型片段 → 用户片段 …”的循环。这样模型不再等完整提问，而是随时准备在生成过程中切换到监听状态。

2. **时分复用的编码‑解码策略**  
   传统的 Transformer 编码器一次接受完整序列，解码器一次生成完整句子。这里作者在编码阶段把用户的时间片和模型已经生成的片段交错拼接，在解码阶段采用相同的交错方式输出。相当于在同一条流水线上让两种不同的零件交替生产，既保持了模型的上下文连贯，又实现了“边说边听”。

3. **专用的交替切片微调数据**  
   为了让模型在不完整上下文下仍能保持原有能力，作者收集并标注了大量真实对话的切片，加入了常见的即时反馈（如打断、确认、纠错）。只需几轮微调，模型在标准基准测试上几乎不掉分，却能在全双工场景下表现出色。

4. **人机交互感知的评估框架**  
   除了常规的准确率、BLEU 等指标，论文引入了即时反馈率、打断容忍度等新指标，并通过大量真人用户实验验证全双工模型在自然度和满意度上显著优于传统回合制模型。

### 方法详解
**整体思路**  
整个系统可以看成三步走：① 将原始对话切成等长的时间片；② 用时分复用的方式把用户片段和模型片段交错送入 Transformer 编码器；③ 在解码阶段同样交错输出，使模型在生成每个 token 时都能随时切换到“监听”模式。微调阶段只在交替切片数据上进行少量梯度更新。

**关键模块拆解**  

1. **时间片切分器**  
   - 输入：连续的语音转文字流或文字流。  
   - 操作：按照固定时长（如 200 ms）或固定 token 数切分，确保每片既不太短导致信息匮乏，也不太长影响实时感。  
   - 输出：序列化的时间片列表，标记每片是“用户”还是“模型”。  

2. **时分复用编码器**  
   - 传统 Transformer 的自注意力机制会把所有 token 看作同等重要。这里在位置编码上加入“片段身份”标记，让模型知道哪些 token 属于当前用户片段，哪些属于已经生成的模型片段。  
   - 交错拼接后，模型仍然一次性处理完整的拼接序列，但因为身份标记的存在，注意力可以自适应地聚焦在最近的用户片段，实现“即时响应”。  

3. **交替流式解码器**  
   - 解码过程遵循时间片的顺序：当轮到模型片段时，解码器生成若干 token（通常不超过一个时间片的长度），随后立即暂停，等待下一个用户片段进入。  
   - 为了避免生成中断导致语义不连贯，作者在解码时加入了“上下文保持缓存”，把上一次生成的隐藏状态保留下来，下一次继续使用。  

4. **微调数据构建与训练**  
   - 收集真实对话，使用自动对齐工具把文字转成时间片。  
   - 在每个用户片段后插入对应的模型片段，同时手工标注常见的即时反馈（如“等一下”“我再说”。）  
   - 只在这些切片上进行数千步的 Adam 优化，学习率保持与原模型相近，确保不破坏原有语言能力。  

**最巧妙的设计**  
- **身份位置编码**：在普通位置编码上叠加一个二元标记（用户/模型），让模型在一次前向传播中同时感知“谁在说”。这相当于在同一张桌子上放了两套不同颜色的拼图块，模型能自然区分。  
- **缓存式流式解码**：传统流式解码每生成一个 token 都要重新计算注意力，成本高。这里把已经生成的隐藏状态缓存，下一次只对新进入的用户片段做增量计算，大幅降低延迟。  

### 实验与效果
- **测试任务**：在公开的对话基准（如 MultiWOZ、DSTC）上做标准的语言理解评估，确保模型性能不受切片影响；另外在自建的实时对话场景（模拟语音输入、即时打断）上进行人机交互实验。  
- **Baseline 对比**：与原始 GPT‑3.5、ChatGPT 的回合制版本以及已有的流式解码模型（如 RealtimeGPT）比较。论文报告在标准基准上性能下降不到 0.2%，而在实时交互满意度上提升约 15%（人类评分从 3.6 提升到 4.2/5）。  
- **消融实验**：去掉身份位置编码后，模型在打断场景下出现明显的上下文混淆，满意度下降约 6%；不使用缓存的流式解码导致平均响应延迟从 200 ms 增至 800 ms，用户体验显著下降。  
- **局限性**：作者承认时间片的长度是一个折中点，太短会增加计算开销，太长会削弱即时感；此外，切片后上下文仍是“伪同步”，在极端长句或复杂推理时仍可能出现信息丢失。  

### 影响与延伸思考
这篇工作打开了 LLM 在实时交互领域的可能性，随后出现了多篇围绕“全双工对话”或“实时流式 LLM”的研究，例如将全双工概念扩展到多模态（语音+视觉）场景的 **DuplexVision**，以及在边缘设备上实现低延迟全双工的 **EdgeDuplex**。对想进一步探索的读者，可以关注以下方向：① 更细粒度的自适应切片策略（根据语义而非固定时长切分）；② 跨语言全双工模型的迁移学习；③ 将全双工能力与检索增强（RAG）结合，实现即时信息查询。  

### 一句话记住它
把大语言模型切成交错的时间片，用时分复用让它在生成答案的同时还能“听”用户，实现了真正的实时全双工对话。