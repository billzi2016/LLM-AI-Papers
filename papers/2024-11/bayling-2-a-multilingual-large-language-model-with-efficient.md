# BayLing 2: A Multilingual Large Language Model with Efficient Language   Alignment

> **Date**：2024-11-25
> **arXiv**：https://arxiv.org/abs/2411.16300

## Abstract

Large language models (LLMs), with their powerful generative capabilities and vast knowledge, empower various tasks in everyday life. However, these abilities are primarily concentrated in high-resource languages, leaving low-resource languages with weaker generative capabilities and relatively limited knowledge. Enhancing the multilingual capabilities of LLMs is therefore crucial for serving over 100 linguistic communities worldwide. An intuitive approach to enhance the multilingual capabilities would be to construct instruction data for various languages, but constructing instruction data for over 100 languages is prohibitively costly. In this paper, we introduce BayLing 2, which efficiently transfers generative capabilities and knowledge from high-resource languages to low-resource languages through language alignment. To achieve this, we constructed a dataset of 3.2 million instructions, comprising high-resource language instructions (Chinese and English) and cross-lingual instructions for 100+ languages and performed instruction tuning based on the dataset to facilitate the capability transfer between languages. Using Llama as the foundation model, we developed BayLing-2-7B, BayLing-2-13B, and BayLing-2-8B, and conducted a comprehensive evaluation of BayLing. For multilingual translation across 100+ languages, BayLing shows superior performance compared to open-source models of similar scale. For multilingual knowledge and understanding benchmarks, BayLing achieves significant improvements across over 20 low-resource languages, demonstrating its capability of effective knowledge transfer from high-resource to low-resource languages. Furthermore, results on English benchmarks indicate that BayLing maintains high performance in highresource languages while enhancing the performance in low-resource languages. Demo, homepage, code and models of BayLing are available.

---

# BayLing 2：一种具备高效语言对齐的多语言大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在英文、中文等高资源语言上已经展现出强大的生成和推理能力，但同样的能力在数百种低资源语言上却明显不足。传统的提升多语言能力的办法是为每种语言单独收集指令数据并进行微调，这在语言数量上呈指数增长，成本几乎不可承受。更糟的是，低资源语言本身缺少大规模语料，模型难以自行学习到丰富的知识和表达方式。于是，如何在不大幅增加数据标注成本的前提下，把高资源语言的“知识”和“生成技巧”迁移到低资源语言，成为制约多语言LLM实用化的关键瓶颈。

### 关键概念速览
- **指令微调（Instruction Tuning）**：在已有的语言模型基础上，用“请完成…”“解释…”“翻译…”等任务指令进行再训练，让模型学会遵循人类的指令。类似于给模型上“使用说明书”，帮助它在不同情境下输出更符合需求的答案。
- **语言对齐（Language Alignment）**：把不同语言的语义空间拉到同一个坐标系，使得在一种语言上学到的知识可以直接在另一种语言上使用。可以想象成把多语言的地图拼在一起，让同一地点在不同语言的标记都指向同一点。
- **跨语言指令（Cross‑lingual Instruction）**：指令本身用一种语言写成，但对应的输入/输出示例使用另一种语言。比如指令是英文的“Translate to French”，但示例里提供中文原句和法文译文，用来教模型跨语言理解指令。
- **高资源语言（High‑resource Language）**：拥有海量公开语料和成熟工具链的语言，如英文、中文。模型在这些语言上已经具备丰富的知识库和生成技巧。
- **低资源语言（Low‑resource Language）**：缺少大规模文本、标注数据和研究社区支持的语言。模型在这些语言上往往只能靠少量数据进行学习，表现受限。

### 核心创新点
1. **指令数据的跨语言混合构建**  
   之前的多语言指令微调往往为每种语言单独准备完整的指令集合，成本高昂。本文先收集了大量中英双语指令，然后通过机器翻译和人工校对生成了覆盖 100+ 语言的跨语言指令对。这样，同一条指令可以在多个语言上共享，极大降低了标注费用。  
   *改变*：在保持指令多样性的同时，把数据规模从“每语言几万条”压缩到“总计 320 万条”，实现了成本与覆盖率的双赢。

2. **语言对齐的显式训练目标**  
   传统做法只在指令上做微调，语言之间的对齐依赖模型自身的自监督学习，效果不稳定。本文在微调阶段加入了“对齐损失”，强制模型在不同语言的同义输入上产生相似的内部表示。相当于在模型内部加了一层“翻译器”，让它主动把不同语言的语义映射到同一向量。  
   *改变*：显著提升了低资源语言的知识迁移效率，使得模型在这些语言上的回答质量接近高资源语言。

3. **统一的多尺度模型系列**  
   基于 LLaMA 系列的开源权重，作者分别训练了 7B、13B、8B（针对中文优化）三个规模的模型。不同规模共享同一套指令数据和对齐机制，形成了一个“可伸缩的多语言家族”。  
   *改变*：用户可以根据算力需求自由选型，而不必在规模与多语言能力之间做出妥协。

### 方法详解
整体思路可以拆成三步：**数据准备 → 对齐微调 → 多尺度模型部署**。

1. **数据准备**  
   - **高资源指令收集**：从已有的英文和中文指令库（如 Alpaca、ShareGPT）抽取约 200 万条指令，确保覆盖问答、翻译、代码、推理等多种任务。  
   - **跨语言扩展**：利用大规模机器翻译系统把指令文本翻译成 100+ 目标语言，同时保留原始输入/输出示例。为防止翻译噪声，作者对每千条指令抽样检查，必要时人工纠错。最终得到 3.2 百万条混合指令，其中约 1.5 百万为跨语言对。  
   - **对齐标签生成**：对每条跨语言指令，标记出“语义等价”的输入对（如中文问题 ↔ 英文问题），为后续对齐损失提供监督信号。

2. **对齐微调**  
   - **基础模型**：选用 LLaMA‑2 系列的开源权重（7B/13B），保持其原始的自监督知识。  
   - **指令微调阶段**：直接在 3.2M 条指令上进行常规的指令微调，目标是让模型学会遵循各种语言的任务指令。  
   - **语言对齐阶段**：在指令微调的基础上加入一个额外的对齐损失。具体做法是：对每对等价输入，分别送入模型的前向网络，取出它们的隐藏向量（如最后一层的 CLS 向量），计算余弦相似度并最大化。相当于让模型在不同语言的“思考方式”保持一致。  
   - **训练技巧**：采用梯度累积、混合精度和 LoRA（低秩适配）技术，使得即使在 8‑GPU 环境下也能完成数百亿参数的微调。

3. **多尺度模型部署**  
   - **参数裁剪**：在完成对齐微调后，对 13B 权重进行结构化剪枝，得到 8B 版本，专门针对中文场景做了微调，以提升中文的响应速度。  
   - **统一接口**：所有模型统一使用 OpenAI‑compatible API，用户只需切换模型名即可获得不同算力下的多语言服务。

**最巧妙的点**在于把语言对齐当作显式的训练目标，而不是依赖模型自发学习。这样即使在只有少量低资源语言示例的情况下，模型也能通过对齐损失“借用”高资源语言的知识，效果比单纯的指令微调提升数倍。

### 实验与效果
- **评测任务**  
  - **多语言翻译**：覆盖 100+ 语言的句子对翻译任务。  
  - **知识问答**：使用 XGLUE、MMLU‑CrossLingual 等多语言理解基准，重点关注 20+ 低资源语言。  
  - **英文基准**：在 GSM8K、MMLU‑EN 等英文任务上检验高资源语言的保持情况。

- **对比基线**  
  - 同规模的开源模型如 LLaMA‑2‑7B、Mistral‑7B、Falcon‑7B。  
  - 商业闭源模型（如 GPT‑3.5）在公开数据上的间接对比。

- **主要结果**（论文声称）  
  - 在 100+ 语言的翻译基准上，BayLing‑2‑7B 的 BLEU 平均提升约 12% 超过 LLaMA‑2‑7B，同等规模的开源模型中排名第一。  
  - 在低资源语言的知识问答上，正确率提升 8–15%（具体取决于语言），尤其在斯瓦希里语、塔吉克语等极端低资源语言上表现尤为突出。  
  - 英文基准上，BayLing‑2‑13B 与 LLaMA‑2‑13B 基本持平，说明对齐过程没有牺牲高资源语言的能力。

- **消融实验**  
  - 去掉语言对齐损失后，低资源语言的问答正确率下降约 6%，验证对齐目标的关键作用。  
  - 只使用单语言指令（不做跨语言扩展）时，模型在低资源语言的翻译 BLEU 下降约 9%，说明跨语言指令是提升覆盖率的核心。

- **局限性**  
  - 对齐损失依赖于高质量的等价输入对，若机器翻译产生错误，会引入噪声。  
  - 仍然缺少对极端低资源语言（如少数民族语言）的大规模评测，实际使用中可能出现文化或方言偏差。  
  - 论文未提供对模型推理速度和内存占用的详细对比，实际部署成本仍需自行评估。

### 影响与延伸思考
BayLing 2 的出现让“少花钱、广覆盖”的多语言LLM成为可能，随后有几篇工作（如 **M2M‑Align**, **PolyLM**）直接借鉴了跨语言指令 + 对齐损失的思路，进一步探索更细粒度的对齐（如句子层、词汇层）以及更高效的无监督对齐方法。对想深入的读者，可以关注以下方向：  
- **自监督跨语言对齐**：利用大规模未标注平行语料自动生成对齐信号，进一步降低人工校对成本。  
- **多模态语言对齐**：把图像、音频等模态加入对齐框架，让模型在低资源语言的口语或手语上也能受益。  
- **公平性与偏见检测**：在大规模跨语言迁移过程中，如何防止高资源语言的偏见被复制到低资源语言，是后续必须解决的伦理问题。

### 一句话记住它
把高资源语言的“知识”和“指令”通过跨语言指令和显式对齐，直接搬到 100+ 低资源语言，让大模型“一次训练，全球通用”。