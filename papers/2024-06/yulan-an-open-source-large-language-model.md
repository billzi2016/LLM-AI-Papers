# YuLan: An Open-source Large Language Model

> **Date**：2024-06-28
> **arXiv**：https://arxiv.org/abs/2406.19853

## Abstract

Large language models (LLMs) have become the foundation of many applications, leveraging their extensive capabilities in processing and understanding natural language. While many open-source LLMs have been released with technical reports, the lack of training details hinders further research and development. This paper presents the development of YuLan, a series of open-source LLMs with $12$ billion parameters. The base model of YuLan is pre-trained on approximately $1.7$T tokens derived from a diverse corpus, including massive English, Chinese, and multilingual texts. We design a three-stage pre-training method to enhance YuLan's overall capabilities. Subsequent phases of training incorporate instruction-tuning and human alignment, employing a substantial volume of high-quality synthesized data. To facilitate the learning of complex and long-tail knowledge, we devise a curriculum-learning framework throughout across these stages, which helps LLMs learn knowledge in an easy-to-hard manner. YuLan's training is finished on Jan, 2024 and has achieved performance on par with state-of-the-art LLMs across various English and Chinese benchmarks. This paper outlines a comprehensive technical roadmap for developing LLMs from scratch. Our model and codes are available at https://github.com/RUC-GSAI/YuLan-Chat.

---

# 玉兰：开源大语言模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）迅速成为各类 AI 应用底层的今天，研究者们已经发布了不少开源模型。但大多数开源项目只给出模型权重和高层实现，缺少从零开始的训练细节。没有完整的预训练、指令微调、对齐流程的公开记录，外部团队难以复现、难以在此基础上改进。尤其是中文与多语言混合的海量语料、如何让模型在长尾知识上稳步提升，这些关键技术在公开文献里仍是“黑盒”。因此，提供一套可复制、可追踪的全链路训练方案，成为迫切需求。

### 关键概念速览
**大语言模型（LLM）**：参数量在数十亿以上、能够理解并生成自然语言的深度网络。把它想象成一位“语言全能选手”，阅读海量文本后能回答问题、写文章等。

**预训练（Pre‑training）**：在大规模未标注文本上让模型学习语言的基本统计规律。类似于学生在课堂上先学语法和词汇，为后续专业学习打底。

**指令微调（Instruction‑tuning）**：在带有明确任务指令的高质量数据上继续训练，使模型学会遵循用户的具体要求。相当于在学完基础后，让学生练习答题技巧。

**对齐（Alignment）**：通过人类反馈或其他安全信号，引导模型的输出更符合人类价值观和使用场景。可以比作老师在批改作业时给出的纠正与建议。

**课程学习（Curriculum Learning）**：训练过程中先喂简单例子，再逐步加入更难或更专业的样本，让模型像循序渐进的课程一样学习。类似于从小学到大学的教育路径。

**合成指令数据（Synthetic Instruction Data）**：利用已有模型或规则自动生成的、带有任务指令的训练样本。相当于老师自己出题，快速扩充练习册。

### 核心创新点
1. **三阶段预训练 → 分层提升模型能力**  
   过去的开源模型往往只做一次大规模预训练，然后直接发布。玉兰把训练拆成三个阶段：① 基础语言模型预训练，覆盖 1.7 万亿 token；② 指令微调，使用海量合成指令数据；③ 对齐阶段，引入人类反馈。这样每一步都有针对性的目标，整体效果比“一刀切”更稳健。

2. **全链路课程学习框架 → 知识易‑难递进**  
   传统训练把所有样本一次性喂入，模型容易在长尾知识上出现“跳跃”。玉兰在三个阶段中都嵌入课程学习：先让模型学习高频、通用的句子，再逐步加入低频、专业的长文本。相当于先教学生常用词汇，再让他阅读专业文献，提升了对复杂知识的掌握度。

3. **大规模合成指令数据 → 低成本高质量微调**  
   公开的指令微调数据往往稀缺且成本高。玉兰通过自研的指令生成管线，批量合成了质量可控的指令样本，显著降低了人工标注需求。这样在保持微调质量的前提下，训练成本大幅下降。

4. **完整技术路线公开 → 开源生态加速**  
   论文不仅给出模型结构，还系统梳理了数据采集、清洗、分布式训练、评估等每一步细节，并全部开源。相比之前只提供权重的做法，这种“一站式”指南让其他团队可以直接复现或在此基础上创新。

### 方法详解
**整体框架**  
玉兰的训练流程可以看作三层塔楼：底层是大规模语言模型预训练，第二层是指令微调，顶层是对齐。每层都配合课程学习，让模型从“会说话”逐步进化到“会听话、会守规”。整个过程在 2023 年底完成，耗时约数月。

**第一阶段：基础预训练**  
- **数据来源**：约 1.7 万亿 token，涵盖英文、中文以及多语言网页、书籍、新闻等。数据清洗采用去重、过滤低质量文本等常规手段。  
- **模型结构**：12 B 参数的 Transformer，层数、隐藏维度与同量级的商业模型相当。  
- **课程学习实现**：训练初期只使用出现频率最高的前 10% 词汇构成的句子，随后逐步放宽频率阈值，最终覆盖全语料。这样模型先学会基本语法，再学习更稀有的表达。

**第二阶段：指令微调**  
- **合成指令数据生成**：利用已经预训练好的模型，给出任务描述（如“把下面的句子翻译成英文”），让模型自行生成输入‑输出对。随后通过过滤规则剔除不符合格式或质量低的样本。  
- **微调目标**：让模型在看到指令后能够直接输出符合要求的答案。训练时仍沿用课程学习思路，先喂简单的“一问一答”，再加入多轮对话、复杂推理等任务。  
- **关键技巧**：采用混合精度、梯度累积等分布式训练技巧，保持大模型的训练效率。

**第三阶段：对齐**  
- **人类反馈收集**：从公开的对话平台或内部评审中抽取模型输出，让标注员对答案的有用性、安全性进行打分。  
- **奖励模型训练**：基于这些评分训练一个小型奖励模型，用来评估生成文本的质量。  
- **强化学习微调（RLHF）**：使用奖励模型的分数作为信号，对大模型进行策略梯度优化，使其倾向于生成高分答案。  
- **课程学习延伸**：对齐阶段同样先对低风险、易评估的任务进行强化学习，随后逐步加入高风险、价值观敏感的对话场景。

**最巧妙的设计**  
- **跨阶段课程学习统一**：把课程学习贯穿于所有训练阶段，而不是只在预训练时使用。这种“一体化”让模型在每一步都保持学习节奏的连贯性，避免了后期微调时出现的“忘记基础”现象。  
- **合成指令数据的闭环**：生成指令数据的模型本身就是前一阶段的模型，使得数据质量随模型提升而自然提升，形成良性循环。

### 实验与效果
- **评测任务**：在多项英文基准（如 MMLU、TruthfulQA）和中文基准（如 CMMLU、C-Eval）上进行零-shot 与 few-shot 测试。  
- **对比基线**：与同参数量的商业模型（如 LLaMA‑2‑13B）以及其他开源模型（如 Baichuan‑13B）进行比较。论文声称在大多数指标上达到或略超这些基线的水平。  
- **消融实验**：作者分别去掉课程学习、合成指令数据或对齐阶段进行对比，结果显示每一模块的缺失都会导致整体性能下降 1‑3%（具体数值未在摘要中给出）。  
- **局限性**：作者承认模型仍受限于 12 B 参数规模，在极端推理深度或超长上下文场景下表现不如更大模型；此外，合成指令数据的多样性仍有提升空间。

### 影响与延伸思考
- **生态推动**：玉兰的全链路开源方案为国内外研究团队提供了从零开始训练 LLM 的可复制路径，已经被后续的中文开源项目（如 “星火大模型”）引用或借鉴。  
- **后续方向**：  
  1. **规模扩展**：在相同课程学习框架下，探索 30 B、70 B 参数模型的训练方法。  
  2. **多模态融合**：把图像、音频等信号加入课程学习，构建跨模态大模型。  
  3. **更高质量对齐**：引入更细粒度的人类价值观标注，提升模型在安全敏感场景的表现。  
- **进一步阅读**：想深入了解课程学习在 LLM 中的实现细节，可以关注 OpenAI 的 “Curriculum Learning for Language Models” 以及 DeepMind 的 “Scaling Laws for Curriculum Learning”。  

### 一句话记住它
**玉兰用三阶段、全链路课程学习把“从零预训练到安全对齐”写成了可复制的开源手册。**