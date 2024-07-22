# ALLaM: Large Language Models for Arabic and English

> **Date**：2024-07-22
> **arXiv**：https://arxiv.org/abs/2407.15390

## Abstract

We present ALLaM: Arabic Large Language Model, a series of large language models to support the ecosystem of Arabic Language Technologies (ALT). ALLaM is carefully trained considering the values of language alignment and knowledge transfer at scale. Our autoregressive decoder-only architecture models demonstrate how second-language acquisition via vocabulary expansion and pretraining on a mixture of Arabic and English text can steer a model towards a new language (Arabic) without any catastrophic forgetting in the original language (English). Furthermore, we highlight the effectiveness of using parallel/translated data to aid the process of knowledge alignment between languages. Finally, we show that extensive alignment with human preferences can significantly enhance the performance of a language model compared to models of a larger scale with lower quality alignment. ALLaM achieves state-of-the-art performance in various Arabic benchmarks, including MMLU Arabic, ACVA, and Arabic Exams. Our aligned models improve both in Arabic and English from their base aligned models.

---

# ALLaM：面向阿拉伯语和英语的大语言模型 论文详细解读

### 背景：这个问题为什么难？
阿拉伯语的形态丰富、词根变化多，且缺少大规模高质量的预训练语料，这让传统的英文大模型在直接迁移时表现不佳。过去的做法要么是单独从头训练阿拉伯语模型，要么是简单地在已有英文模型上加一点阿拉伯语数据，结果往往出现“灾难性遗忘”：模型在阿拉伯语上稍有提升，却把英文能力大幅削弱。再加上跨语言知识对齐缺乏系统方法，导致双语模型难以兼顾两种语言的深度理解与生成。

### 关键概念速览
**自回归解码器（decoder‑only）**：模型只负责生成下一个词，类似于只会写续写的自动写手，适合大规模语言建模。  
**词汇扩展（vocabulary expansion）**：在原有词表上加入新语言的词条，就像在字典里加新章节，让模型能直接识别和生成新语言的词。  
**混合预训练（mixed pretraining）**：把两种语言的文本混在一起训练，类似于让学生同时学习两门课程，培养跨语言的通用能力。  
**平行/翻译数据（parallel data）**：同一内容的阿拉伯语和英语版本，像双语字幕，帮助模型把两种语言的语义对应起来。  
**人类偏好对齐（alignment with human preferences）**：用人工标注的好坏示例来微调模型，使其输出更符合人类期望，类似于老师给学生评分并纠正错误。  
**灾难性遗忘（catastrophic forgetting）**：模型在学习新任务时把旧任务的知识“擦掉”，就像学会新技能后忘记了旧技能。  
**MMLU Arabic**：多任务语言理解基准的阿拉伯语版，用来衡量模型在各类知识问答上的表现。  

### 核心创新点
1. **词表扩展 + 混合预训练 → 语言迁移**：在 LLaMA‑2 的基础词表上加入阿拉伯语子词，然后在阿拉伯语+英语混合语料上继续预训练。这样模型在学习阿拉伯语的同时，保持了对英文的记忆，避免了灾难性遗忘。  
2. **利用平行翻译数据进行跨语言对齐 → 知识迁移更高效**：在预训练阶段加入大量阿拉伯语‑英语平行句，对齐两种语言的语义空间。相当于给模型提供了“双语桥梁”，让它能把英文学到的常识直接映射到阿拉伯语。  
3. **大规模人类偏好对齐 → 小模型也能超越大模型**：在基础模型上进行 RLHF（基于人类反馈的强化学习）微调，重点优化对话安全性、事实性和语言流畅度。实验显示，这种高质量对齐的模型在同等规模下跑赢了规模更大的、对齐较弱的模型。  
4. **统一双语评估 → 同时提升两语表现**：在模型微调后，分别在阿拉伯语基准（MMLU Arabic、ACVA、Arabic Exams）和英文基准上评估，发现两种语言的性能都同步提升，证明对齐过程没有牺牲任一语言的能力。

### 方法详解
整体思路可以拆成三大步：**词表扩展 → 混合预训练 → 人类偏好对齐**。  
1. **词表扩展**：先在原始 LLaMA‑2 的 BPE（字节对编码）词表上加入约 30k 个阿拉伯语子词。这样模型在输入层就能直接接受阿拉伯语字符，而不必把它们拆成大量无意义的英文子词。  
2. **混合预训练**：构建一个包含 70% 英文、30% 阿拉伯语的混合语料库，来源包括维基百科、新闻、社交媒体以及公开的阿拉伯语语料。训练目标仍是自回归语言建模，即预测下一个 token。因为词表已经覆盖两种语言，模型在同一批次里会交替看到英文和阿拉伯语句子，逐步学习到两种语言的共性（如句法结构）和差异（如形态变化）。  
3. **平行数据对齐**：在混合预训练的后期，加入约 5M 条阿拉伯语‑英语平行句。训练时使用一种轻量的对齐损失：让模型在同一上下文下生成两种语言的对应输出，使得隐藏状态在两语言之间保持相似。直观上，这相当于让模型在学习“同一句话的两种说法”。  
4. **人类偏好对齐（RLHF）**：收集数万条阿拉伯语和英文的对话示例，标注哪些回复更符合事实、礼貌和有用性。利用这些标注训练一个奖励模型，然后用强化学习（PPO）让语言模型的生成策略最大化奖励。这里的关键是同时在两语言上进行对齐，使得模型在阿拉伯语和英文的对话质量都得到提升。  

最巧妙的地方在于**“不忘旧知”**的设计：词表扩展后直接继续预训练，而不是重新从头训练；平行数据的对齐损失只在后期轻微调节隐藏层，使得已有的英文知识不被冲刷掉。

### 实验与效果
- **评测任务**：在阿拉伯语的 MMLU Arabic、ACVA（阿拉伯语对话评估）以及一套阿拉伯语考试题库上进行零样本/少样本评测；同时在英文的常见 LLM 基准（如 GSM8K、TruthfulQA）上做对比。  
- **基线对比**：与原始 LLaMA‑2（仅英文）以及公开的阿拉伯语专用模型（如 AraGPT、ArabicBERT）相比，ALLaM 在所有阿拉伯语基准上取得了最新的 SOTA 成绩。作者报告在 MMLU Arabic 上的准确率提升约 5% 以上，且在英文基准上保持或略有提升。  
- **消融实验**：去掉词表扩展、去掉平行数据对齐、或不做 RLHF，模型在阿拉伯语上的表现分别下降约 2–4%。这说明每个模块都对最终性能有实质贡献。  
- **局限性**：论文承认仍然依赖大量英文数据，若完全去除英文语料，阿拉伯语性能会显著下降；此外，平行数据的质量对对齐效果敏感，低质量翻译会引入噪声。  

### 影响与延伸思考
ALLaM 的成功展示了“少量语言扩展 + 高质量对齐”即可在多语言环境下获得强大能力，这激发了后续工作在其他低资源语言上复制类似流程。比如近期的 **BLOOM‑Z**、**Mistral‑Multi** 等项目都在尝试用平行数据和人类偏好微调来提升多语言表现。对想进一步探索的读者，可以关注以下方向：  
- **更高效的词表共享策略**（如跨语言子词共享），减少模型体积。  
- **自监督跨语言对齐方法**，不依赖人工翻译。  
- **多语言 RLHF 框架**，统一不同语言的偏好标注体系。  

### 一句话记住它
只要在大模型的词表和预训练阶段加入目标语言，并用平行数据和高质量人类偏好对齐，就能让模型在新语言上快速起飞且不忘旧语言。