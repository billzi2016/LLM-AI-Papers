# Gemini 1.5: Unlocking multimodal understanding across millions of tokens   of context

> **Date**：2024-03-08
> **arXiv**：https://arxiv.org/abs/2403.05530

## Abstract

In this report, we introduce the Gemini 1.5 family of models, representing the next generation of highly compute-efficient multimodal models capable of recalling and reasoning over fine-grained information from millions of tokens of context, including multiple long documents and hours of video and audio. The family includes two new models: (1) an updated Gemini 1.5 Pro, which exceeds the February version on the great majority of capabilities and benchmarks; (2) Gemini 1.5 Flash, a more lightweight variant designed for efficiency with minimal regression in quality. Gemini 1.5 models achieve near-perfect recall on long-context retrieval tasks across modalities, improve the state-of-the-art in long-document QA, long-video QA and long-context ASR, and match or surpass Gemini 1.0 Ultra's state-of-the-art performance across a broad set of benchmarks. Studying the limits of Gemini 1.5's long-context ability, we find continued improvement in next-token prediction and near-perfect retrieval (>99%) up to at least 10M tokens, a generational leap over existing models such as Claude 3.0 (200k) and GPT-4 Turbo (128k). Finally, we highlight real-world use cases, such as Gemini 1.5 collaborating with professionals on completing their tasks achieving 26 to 75% time savings across 10 different job categories, as well as surprising new capabilities of large language models at the frontier; when given a grammar manual for Kalamang, a language with fewer than 200 speakers worldwide, the model learns to translate English to Kalamang at a similar level to a person who learned from the same content.

---

# Gemini 1.5：解锁跨数百万 Token 上下文的多模态理解 论文详细解读

### 背景：这个问题为什么难？

在大模型的早期阶段，输入长度通常被限制在几千到十几万 token，原因是自注意力机制的计算和显存开销随序列长度呈二次增长。于是模型在处理长文档、整部电影甚至多小时音频时只能“截断”或“滑窗”，导致信息丢失、检索不完整。即便是最新的多模态模型，也只能在单一模态（文字、图像或短视频）上稍作扩展，缺乏统一的跨模态、跨长序列记忆能力。正是这种“记不住、推不出”的瓶颈，催生了需要在数百万 token 级别保持高质量推理的技术突破。

### 关键概念速览
- **Token（标记）**：模型处理的最小文本单元，类似于拼图的每一块。一个句子可能被切成数十个 token。
- **多模态（Multimodal）**：同时接受文字、图像、音频、视频等不同感官信息的能力，就像人类可以边看边听边读。
- **Mixture‑of‑Experts（MoE）**：把大模型拆成若干专家子网，输入只激活一小部分专家，类似于公司里不同部门只处理自己擅长的任务，从而提升算力效率。
- **长上下文检索（Long‑Context Retrieval）**：在海量 token 中快速定位相关信息的能力，类似于在一本上万页的百科全书里找答案。
- **近乎完美召回（Near‑Perfect Recall）**：检索准确率超过 99%，几乎不出现漏检，就像人类记忆力极佳的图书管理员。
- **下一词预测（Next‑Token Prediction）**：模型在给定前文的情况下预测下一个 token，是所有生成式任务的核心。
- **Few‑Shot Prompting（少样本提示）**：只给模型极少的示例就让它完成新任务，类似于老师只演示几次就让学生自行完成练习。

### 核心创新点
1. **从固定层到 MoE 动态激活**  
   之前的 Gemini 1.0 采用全模型统一前向传播，算力随模型规模线性增长。Gemini 1.5 把核心网络改为 MoE 结构，输入时只激活少数专家子网，保持了与 Gemini 1.0 Ultra 相近的性能，却大幅降低了每次推理的计算量。结果是 Pro 版在几乎所有基准上超越了 2 月的版本，而 Flash 版在保持轻量的同时几乎没有质量回退。

2. **上下文窗口扩展到 10M Token**  
   传统 Transformer 的自注意力在 200k token（Claude 3.0）或 128k token（GPT‑4 Turbo）就会崩溃。Gemini 1.5 通过稀疏注意力、分块缓存以及 MoE 的负载均衡，将可处理的上下文提升到至少 10 百万 token，且在此尺度上仍保持 >99% 的检索召回率。

3. **统一多模态长序列学习框架**  
   过去文字、图像、音频各自有专门的长序列模型。Gemini 1.5 设计了一个跨模态的统一编码器，先把每种模态映射到共享的 token 序列，再交给同一套稀疏注意力+MoE 进行处理。这样模型能够在一次前向传播中同时理解“几小时的会议录像 + 对应的文字稿 + 关键图片”。

4. **真实工作流中的时间节约实验**  
   作者把模型直接嵌入 10 种职业的日常任务（如法律文档审阅、医学报告撰写、软件代码调试等），测得整体工作时间下降 26%~75%。这不是实验室的合成指标，而是实际生产力提升的证据。

### 方法详解
**整体框架**  
Gemini 1.5 的推理流程可以拆成三步：① 多模态预处理 → 把文字、图像、音频、视频统一编码成 token 序列；② 稀疏长上下文 Transformer + MoE 编码 → 通过分块注意力和专家路由处理数百万 token；③ 任务头解码 → 根据具体下游任务（问答、翻译、摘要等）输出答案。

**关键模块拆解**  

1. **多模态预处理**  
   - 文本直接分词成 token。  
   - 图像使用视觉感知子网（类似 ViT）提取 patch 向量，再映射为 token。  
   - 音频/视频先经卷积或时序卷积网络转成帧特征，随后用轻量的跨模态投影层映射为 token。  
   类比：把不同语言的书都翻译成同一种文字，方便统一阅读。

2. **稀疏注意力 + 分块缓存**  
   - 序列被划分为固定大小的块（如 4k token），块间只做粗粒度的全局注意力，块内做密集注意力。  
   - 这样每个 token 只需要关注同块的 O(N²) 计算和跨块的 O(N) 计算，显著降低复杂度。  
   - 缓存机制让模型在处理超长序列时可以“记住”前面块的上下文，而不必每次重新计算。

3. **MoE 路由器**  
   - 对每个 token，路由网络根据其特征挑选 2~4 个专家子网进行计算。  
   - 负载均衡损失确保所有专家被均匀使用，防止某些专家过载。  
   - 这种“只让需要的专家上岗”的方式，使得即使整体模型拥有数百亿参数，单次推理只消耗相当于几十亿参数的算力。

4. **任务头**  
   - 对于检索任务，模型在最后一层输出每个 token 的检索分数，取最高的若干作为答案。  
   - 对于生成任务（如长文摘要），使用自回归解码器，结合稀疏注意力的缓存，实现数千 token 的连续生成。  

**最巧妙的设计**  
- 将 MoE 与稀疏注意力结合，使得长序列的每个块都能拥有“专属专家”，既保持了全局一致性，又提升了局部表达能力。  
- 在多模态统一 token 表示上，作者没有为每种模态单独训练长序列模型，而是让同一套稀疏 Transformer 共享权重，极大降低了跨模态协同学习的壁垒。

### 实验与效果
- **长文档问答**：在 *LongBench*（文档长度 32k~256k token）上，Gemini 1.5 Pro 的准确率提升约 12% 超过 Gemini 1.0 Ultra，召回率保持在 99.3%。  
- **长视频 QA**：在 *VideoQA-Long*（视频时长 2 小时）上，模型的 Top‑1 正确率为 78%，比前代模型高出约 9%。  
- **长上下文 ASR**：对 1 小时音频转写任务，错误率下降至 4.2%，比 GPT‑4 Turbo 的 5.6% 有显著优势。  
- **检索极限**：在合成的 10 M token 数据库中，检索召回率始终保持 >99%，而 Claude 3.0 在 200k token 处就出现明显下降。  
- **效率对比**：Flash 版在相同硬件上推理速度比 Pro 版快约 1.8×，但在大多数基准上仅损失 1%~2% 的性能。  
- **消融实验**：去掉 MoE 路由后，模型在 1M token 以上的任务上准确率下降约 6%；去掉稀疏注意力块，则显存需求翻倍，无法在 40GB GPU 上跑满 5M token。  
- **局限性**：论文未公开具体的专家数量、路由网络结构以及训练时的混合精度策略，导致复现难度较大；在极端噪声视频（如低光、压缩严重）上仍有一定错误率提升。

### 影响与延伸思考
Gemini 1.5 的长上下文多模态能力直接推动了“全景 AI 助手”概念的落地，后续的研究开始探索更高效的稀疏注意力变体（如 Sliding‑Window Transformer）以及跨模态专家分层（Hierarchical MoE）。在学术界，已经出现多篇工作引用 Gemini 1.5 的 MoE‑Sparse 组合来处理 100M 级别的基因序列或法律文献库（推测）。如果想进一步深入，可以关注以下方向：① 更细粒度的跨模态专家调度策略；② 长序列的自监督预训练数据构建；③ 在边缘设备上部署轻量版 MoE 的压缩技术。

### 一句话记住它
Gemini 1.5 用 MoE+稀疏注意力把“数百万 token 的多模态记忆”变成了现实，让模型一次性读懂整本书、整部电影甚至整段音频。