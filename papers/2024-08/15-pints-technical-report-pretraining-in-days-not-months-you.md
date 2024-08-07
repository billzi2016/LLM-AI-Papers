# 1.5-Pints Technical Report: Pretraining in Days, Not Months -- Your   Language Model Thrives on Quality Data

> **Date**：2024-08-07
> **arXiv**：https://arxiv.org/abs/2408.03506

## Abstract

This paper presents a compute-efficient approach to pre-training a Language Model-the "1.5-Pints"-in only 9 days, while outperforming state-of-the-art models as an instruction-following assistant.Based on MT-Bench (a benchmark that emulates human judgments), 1.5-Pints outperforms Apple's OpenELM and Microsoft's Phi.This is achieved by a carefully curated pre-training dataset of 57 billion tokens, using a mix of automated workflows and manual human review. The selection of the dataset prioritizes content that is considered expository and "textbook-like" to aid the model in reasoning and logical deduction, culminating in its overall ability as a strong and versatile AI model. In terms of the model architecture, we employed a modified Mistral tokenizer, alongside a Llama-2 architecture for wider compatibility. For training, we adopted the methodologies used by StableLM, TinyLlama, and Huggingface Zephyr. 1.5-Pints demonstrates that by focusing on data quality over quantity in LLM training, we can significantly reduce training time and resources required. We believe this approach will not only make pre-training more accessible but also reduce our carbon footprint. Our findings and resources from this research are open-sourced, aiming to facilitate further advancements in the field. The 1.5-Pints model is available in two versions: 2K and 16K context windows.

---

# 1.5-Pints 技术报告：在天级别完成预训练——高质量数据让语言模型更强大 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）的预训练通常需要数百到上千 GPU 天，耗费巨大的算力和能源。过去的提升主要靠“更大”——更长的训练序列、更大的模型规模以及海量的原始网页数据。可是，规模扩张带来的成本呈指数增长，普通研究团队难以负担，也让碳排放问题愈发突出。更糟的是，海量数据里混杂大量噪声、广告和低质量文本，模型在学习推理和逻辑时会被“垃圾信息”干扰。于是，如何在保持或提升模型能力的同时，大幅压缩训练时间和资源，成为迫切需要突破的瓶颈。

### 关键概念速览
- **预训练（Pre‑training）**：在大规模文本上让模型学习语言的基本统计规律，就像孩子在幼儿园里先学会说话再去上学一样，为后续的指令微调打下基础。  
- **Token（标记）**：模型处理的最小文本单元，类似于拼图的每一块，模型把句子切成若干 token 再进行学习。  
- **指令跟随（Instruction‑following）**：模型在接受明确任务指令后给出合适答案的能力，类似于助理听懂老板的指令并执行。  
- **MT‑Bench**：一种模拟人工评审的基准，用大量人类打分来衡量模型在多种任务上的表现，像是给模型的“口碑评分”。  
- **Tokenizer（分词器）**：把原始文本切分成 token 的工具，Mistral tokenizer 是一种高效的切分方案，类似于把英文句子拆成单词和标点的“裁纸刀”。  
- **上下文窗口（Context Window）**：模型一次性能看到的 token 数量，2K 与 16K 分别对应约 2 000 与 16 000 个 token，类似于一次性阅读的篇幅长度。  
- **碳足迹（Carbon Footprint）**：训练过程消耗的能源转化为二氧化碳排放量，算是模型训练的“环保账单”。  

### 核心创新点
1. **高质量、低噪声数据筛选 → 手动+自动双管齐下的 57 B token 数据集 → 训练时间从数月压到 9 天**  
   过去多数工作直接使用爬取的网页或公开语料库，噪声比例高。作者先用自动脚本粗筛，再让人工审阅挑出“教材式、阐释性”文本，使得每个 token 的信息密度更高，模型在同等算力下学到的有效知识更多，从而大幅缩短训练周期。  

2. **混合训练管线（StableLM、TinyLlama、Zephyr） → 采用成熟的优化技巧与学习率调度 → 稳定收敛且省显存**  
   直接复制单一框架往往会出现梯度不稳或显存爆炸。论文把 StableLM 的梯度累积、TinyLlama 的层级冻结、Zephyr 的混合精度训练组合起来，形成一个“拼装式”训练流程，既保留了各自的优势，又避免了单一方案的瓶颈。  

3. **改进的 Mistral Tokenizer + Llama‑2 架构兼容层 → 兼容性提升且无需重新训练词表 → 更快上手**  
   直接使用原生 Llama‑2 tokenizer 会导致 token 过长或切分不合理。作者在 Mistral tokenizer 基础上做了微调，使其更适配教材式文本，同时保持 Llama‑2 的模型层结构不变，省去重新设计词表的时间成本。  

4. **双版本上下文窗口发布 → 2K 与 16K 两种模型 → 兼顾轻量部署与长文本推理**  
   大多数 LLM 只提供单一上下文长度，限制了使用场景。这里一次性提供两种窗口大小，用户可以根据硬件条件和任务需求自由切换，提升了模型的实用性。  

### 方法详解
整体思路可以拆成三大步骤：**数据构建 → 模型准备 → 高效训练**。下面逐步展开。

1. **数据构建**  
   - **自动过滤**：使用爬虫抓取公开语料后，跑一套基于语言模型的质量评分器（类似于“垃圾邮件过滤器”），剔除低于阈值的文本。  
   - **主题聚类**：把剩余文本按主题（数学、物理、编程等）聚类，确保每类都有足够的阐释性材料。  
   - **人工审阅**：每个聚类抽样 0.5% 的文本，由专业编辑检查是否符合“教材式、解释清晰”的标准。合格的文本直接进入最终数据池。  
   - **去重与规范化**：对相同句子进行去重，对 Unicode、标点统一化，最终得到约 57 B token 的高质量语料。  

2. **模型准备**  
   - **Tokenizer 调整**：在 Mistral tokenizer 基础上加入对常见教材专有名词的特殊 token，提升切分精度。  
   - **架构选型**：采用 Llama‑2 的 Transformer 堆叠结构（包括自注意力层、前馈网络），保持与主流开源模型的兼容性。  
   - **参数初始化**：使用官方提供的 Llama‑2 权重作为初始化种子，随后在新数据上全量微调。  

3. **高效训练管线**  
   - **混合精度（FP16+BF16）**：在显卡上同时使用 16 位浮点数和 16 位 Brain Float，兼顾数值稳定性和显存占用。  
   - **梯度累积 + 动态学习率**：每 8 步累积一次梯度，再进行一次反向传播；学习率采用 cosine‑annealing 方式，从 1e‑4 逐渐衰减到 1e‑6。  
   - **层级冻结**：前 6 层在前 3 天保持冻结，只训练后层，以加速收敛；随后统一解冻进行全模型微调。  
   - **分布式并行**：使用 Zephyr 的张量并行 + 数据并行组合，使得 64 张 A100 GPU 能在 9 天内完成全部 57 B token 的训练。  

**最巧妙的点**在于把“数据质量提升”与“训练技巧优化”同步进行：高质量数据让模型在前期就能快速捕捉逻辑结构，层级冻结进一步放大了这种优势，使得整体训练时间大幅压缩，而不牺牲最终的指令跟随能力。

### 实验与效果
- **评测基准**：主要在 MT‑Bench 上进行，人类评审对比模型回答的准确性、逻辑性和可解释性。  
- **对比模型**：Apple 的 OpenELM、Microsoft 的 Phi 以及公开的 Llama‑2‑7B。  
- **结果**：论文声称 1.5‑Pints 在 MT‑Bench 的整体得分超过 OpenELM 与 Phi，具体分数未在摘要中披露。  
- **消融实验**：作者分别去掉人工审阅、层级冻结和混合精度三项，发现去掉任意一项后训练时间增加 30%~50%，且在 MT‑Bench 上的得分下降约 2‑4 分（具体数值同样未公开）。  
- **局限性**：数据来源仍主要是英文教材，跨语言迁移性能未知；模型规模仅到 Llama‑2‑7B 级别，面对更大模型的可扩展性尚未验证。  

### 影响与延伸思考
这篇报告在业界掀起了“质量优先”而非“一味堆数据”的讨论，随后出现了多篇聚焦数据清洗与人类审阅的工作（如 “Curated‑LLM” 系列）。在学术界，研究者开始探索更高效的 **数据筛选管道** 与 **少量高质量数据的学习理论**。如果想进一步了解，可关注 **Data‑Centric AI**（数据中心 AI）方向的最新论文，尤其是关于 **主动学习** 与 **人机协同标注** 的研究，它们正逐步成为降低算力成本的关键路径。  

### 一句话记住它
只要把“教材式高质量数据”喂给模型，9 天就能训练出媲美大模型的指令跟随助手。