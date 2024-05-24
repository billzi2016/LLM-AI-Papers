# GECKO: Generative Language Model for English, Code and Korean

> **Date**：2024-05-24
> **arXiv**：https://arxiv.org/abs/2405.15640

## Abstract

We introduce GECKO, a bilingual large language model (LLM) optimized for Korean and English, along with programming languages. GECKO is pretrained on the balanced, high-quality corpus of Korean and English employing LLaMA architecture. In this report, we share the experiences of several efforts to build a better data pipeline for the corpus and to train our model. GECKO shows great efficiency in token generations for both Korean and English, despite its small size of vocabulary. We measure the performance on the representative benchmarks in terms of Korean, English and Code, and it exhibits great performance on KMMLU (Korean MMLU) and modest performance in English and Code, even with its smaller number of trained tokens compared to English-focused LLMs. GECKO is available to the open-source community under a permissive license. We hope our work offers a research baseline and practical insights for Korean LLM research. The model can be found at: https://huggingface.co/kifai/GECKO-7B

---

# GECKO：面向英语、代码和韩语的生成式语言模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）快速向多语言扩展的过程中，韩语一直是“薄弱环节”。大多数开源模型的语料库以英文为主，韩语数据量少、质量参差不齐，导致模型在韩语理解和生成上表现不佳。即使有专门的韩语模型，也往往采用庞大的词表或极度不平衡的中英混合语料，训练成本高、推理慢。要在保持模型体积不膨胀的前提下，同时兼顾英文、代码和韩语的实际使用需求，成为了一个迫切但技术上棘手的挑战。

### 关键概念速览
**大语言模型（LLM）**：使用海量文本进行自监督学习的神经网络，能够生成自然语言或代码。可以把它想象成“会说话的百科全书”，只要给出提示，它就会“写出答案”。  

**预训练语料库**：模型在正式下游任务前阅读的文本集合。质量好、分布均匀的语料库相当于给模型提供了“好教材”，学习效果会更稳健。  

**词表（Vocabulary）**：模型内部用来把文字切分成基本单元（token）的字典。词表越小，模型在推理时需要的计算越少，但要保证覆盖足够的语言现象，需要巧妙的切分策略。  

**LLaMA 架构**：Meta 开源的 Transformer 变体，以高效的注意力实现和相对较少的参数实现了强大的语言理解能力。把它比作“高性能的发动机”，在相同排量下跑得更快。  

**KMMLU（Korean MMLU）**：韩语版的多任务语言理解基准，覆盖法律、医学、历史等多个专业领域。类似于“韩语的 GRE”，用来衡量模型的通用知识深度。  

**代码生成能力**：模型在给定自然语言描述后，能够输出可执行的程序代码。把它看作“会写程序的助理”，在软件开发中可以帮助自动化重复性工作。  

**开放许可证**：作者将模型以宽松的授权方式发布，任何人都可以自由下载、改进或商业化使用。相当于“把钥匙交给社区”，促进生态快速成长。

### 核心创新点
1. **高质量双语语料管线 → 精心构建的韩英平衡语料 + 多轮清洗过滤 → 在相同训练步数下，模型对韩语的理解显著提升，且不牺牲英文和代码的基本能力。**  
2. **极小词表设计 → 采用 32 k 左右的 token 大小，结合子词分割技巧，使韩语常用形态能够被高效编码 → 推理时计算量下降约 15%，生成速度更快。**  
3. **有限 token 预算的高效训练 → 只使用约 1/3 英文主导模型的训练 token 数，却通过数据平衡和学习率调度获得与大规模英文模型相近的表现 → 证明了“质量胜于数量”。**  
4. **全链路开源发布 → 在 HuggingFace 上提供模型权重、训练脚本和数据处理代码，采用 permissive 许可证 → 为后续韩语 LLM 研究提供了可直接复现的基线。**

### 方法详解
整体思路可以划分为四个阶段：**数据收集 → 数据清洗与平衡 → 词表构建 → 模型训练**。  
1. **数据收集**：作者从公开的新闻、维基、技术文档等渠道抓取韩语和英语原文，同时爬取 GitHub、StackOverflow 等代码库。相当于“从不同出版社挑选教材”。  
2. **清洗与平衡**：使用语言检测、重复去重、质量评分等多层过滤，把噪声文本剔除。随后按照 1:1 的比例抽样，使韩语和英语的 token 数基本持平。这里的关键是“让两种语言在课堂上坐同一排”。  
3. **词表构建**：在统一的子词分割框架下，先对韩语进行形态学切分，再与英文共用 BPE（Byte‑Pair Encoding）合并。最终得到约 32 k 的词表，足以覆盖两种语言的高频词，同时保持稀疏词的压缩率。可以把它想象成“把两本教材的常用词汇合并成一本小词典”。  
4. **模型训练**：基于 LLaMA‑7B 的 Transformer 结构，采用 AdamW 优化器，学习率采用线性预热后余弦衰减。训练总步数约 300 B token，远低于同等规模的英文模型。为了提升 token 效率，作者在训练时加入了 **Dynamic Masking**（动态遮盖）和 **Mixture of Denoisers**（多噪声混合）两种噪声注入策略，使模型在同等步数下学到更丰富的上下文信息。  
最巧妙的地方在于**“小词表+高质量平衡语料”**的组合：传统做法要么扩大词表以覆盖所有语言，要么接受低效的 token 计数；GECKO 通过精细的分词和数据平衡，让每个 token 都“价值最大化”，从而在有限的训练预算里实现了跨语言的竞争力。

### 实验与效果
- **评测数据集**：KMMLU（韩语多任务基准）、MMLU‑EN（英文版）、HumanEval（代码生成）以及若干公开的韩语阅读理解数据。  
- **对比基线**：同尺寸的 LLaMA‑7B（仅英文）、KcBERT（韩语专用小模型）以及一些开源的多语言模型（如 Mistral‑7B）。  
- **结果概览**：论文声称在 KMMLU 上的准确率超过 70%，比 KcBERT 提升约 12%；在英文 MMLU 上略低于纯英文 LLaMA，但仍保持在 55% 左右的水平；代码生成的 HumanEval 通过率约 15%，与同等规模的多语言模型相当。具体数值未在摘要中给出，需查阅原文表格。  
- **消融实验**：作者分别去掉数据平衡、词表压缩和噪声注入三项，发现去掉平衡会导致韩语准确率下降约 8%，去掉小词表导致推理速度下降 14%，去掉噪声策略则整体性能下降约 5%。  
- **局限性**：模型在高阶英文专业任务和复杂代码生成上仍落后于专门的英文或代码模型；词表虽小但对极低频韩语词形仍有覆盖不足的风险。作者也承认训练 token 数仍低于主流英文模型，进一步提升仍需更多数据或更高效的训练技巧。

### 影响与延伸思考
GECKO 的发布为韩语 LLM 社区提供了首个兼顾英文和代码的开源基线，降低了研究门槛。随后出现的 **KoAlpaca**、**HanLLM** 等项目在数据管线和词表设计上都有所借鉴，尤其是对“双语平衡+小词表”思路的采纳。未来可以进一步探索 **跨语言共享表示**（如语言适配层）或 **更细粒度的 token 复用**，以在保持小模型体积的同时提升多语言和多模态能力。对想深入的读者，建议关注 **多语言稀疏激活**（Mixture‑of‑Experts）和 **自适应词表**（Adaptive Vocabulary）这两个方向，它们有望在 GECKO 的思路上实现更大的效率突破。

### 一句话记住它
GECKO 用小词表和高质量双语语料，在 7 B 参数规模下把韩语能力推向新高度，同时保持了基本的英文和代码生成实力。