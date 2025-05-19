# Krikri: Advancing Open Large Language Models for Greek

> **Date**：2025-05-19
> **arXiv**：https://arxiv.org/abs/2505.13772

## Abstract

We introduce Llama-Krikri-8B, a cutting-edge Large Language Model tailored for the Greek language, built on Meta's Llama 3.1-8B. Llama-Krikri-8B has been extensively trained on high-quality Greek data to ensure superior adaptation to linguistic nuances. With 8 billion parameters, it offers advanced capabilities while maintaining efficient computational performance. Llama-Krikri-8B supports both Modern Greek and English, and is also equipped to handle polytonic text and Ancient Greek. The chat version of Llama-Krikri-8B features a multi-stage post-training pipeline, utilizing both human and synthetic instruction and preference data, by applying techniques such as MAGPIE. In addition, for evaluation, we propose three novel public benchmarks for Greek. Our evaluation on existing as well as the proposed benchmarks shows notable improvements over comparable Greek and multilingual LLMs in both natural language understanding and generation as well as code generation.

---

# Krikri：推动面向希腊语的开源大语言模型 论文详细解读

### 背景：这个问题为什么难？
希腊语的形态变化极其丰富，词尾屈折、重音符号以及古希腊的多音调文字让通用多语言模型难以捕捉细微差别。此前的开源模型大多基于英文或西欧语言，希腊语数据量少、质量参差不齐，导致在阅读理解、生成和代码任务上表现不佳。更糟的是，现有模型几乎不支持古希腊的多音调（polytonic）文本，限制了学术和文化应用的可能性。于是，需要一种专门为希腊语“量身定制”的大语言模型来突破这些瓶颈。

### 关键概念速览
**大语言模型（LLM）**：拥有数十亿参数、通过海量文本自监督学习的模型，类似于“会说话的百科全书”。  
**参数**：模型内部的可调数字，参数越多模型的表达能力越强，但计算成本也随之上升。  
**多语言模型**：在同一个模型里同时学习多种语言的能力，就像一个会说多国语言的翻译官。  
**Polytonic（多音调）**：古希腊文字中使用的多种重音和呼吸符号，类似于英文的重音标记，但更复杂。  
**指令微调（Instruction Tuning）**：让模型学习在特定指令下产生期望输出的过程，好比给模型上了一堂“如何回答问题”的课。  
**偏好学习（Preference Modeling）**：利用人类或合成的偏好数据教模型判断哪种回答更好，类似于让模型学会“挑选最佳答案”。  
**MAGPIE**：一种在指令微调后进一步提升模型对指令理解的技术，名字来源于“Multi‑Aspect Guided Preference‑based Instruction Enhancement”。  
**基准测试（Benchmark）**：统一的评测集合，用来比较不同模型在同一任务上的表现，像是模型之间的体育比赛成绩单。

### 核心创新点
1. **高质量希腊语语料 → 大规模专属预训练 → 更贴合希腊语语法与词汇**  
   过去的模型大多使用通用的多语言语料，希腊语占比极低。Krikri 通过爬取并清洗新闻、文学、学术和代码等多源希腊语文本，构建了一个专门的希腊语语料库，然后在 Llama 3.1‑8B 的基础上进行全量预训练，使模型对希腊语的屈折变化和重音规则有更深的感知。

2. **多阶段后训练管线 → 人类+合成指令与偏好数据 → 生成更自然、更符合指令的回复**  
   传统的指令微调只用少量人工标注。Krikri 先用人工编写的指令数据进行微调，再引入大规模合成指令（模型自生成的任务）和偏好对齐数据，使用 MAGPIE 技术在每一阶段都进行偏好学习，最终得到的聊天模型在回答准确性和流畅度上都有显著提升。

3. **支持现代、古典及多音调希腊语 → 统一模型处理三种文字形态**  
   大多数模型只能处理现代希腊语的标准正字法。Krikri 在词表和分词阶段加入了 polytonic 符号的编码，并在训练数据中混入古希腊文本，使模型能够在同一模型里同时理解现代希腊语、古希腊语以及带重音的多音调文本，打开了学术研究和文化遗产数字化的新大门。

4. **公开三套希腊语基准 → 系统评估模型实力**  
   为了解决希腊语评测资源匮乏的问题，作者自行构建了阅读理解、文本生成和代码生成三个公开基准，并在这些基准上与现有希腊语专属模型和多语言模型进行对比，展示了 Krikri 在各项任务上的领先优势。

### 方法详解
整体思路可以拆成四个阶段：**数据准备 → 基础预训练 → 指令微调 + 偏好对齐 → 多阶段 MAGPIE 强化**。先把每一步说清楚。

1. **数据准备**  
   - **语料收集**：从希腊主流新闻站点、文学作品、维基百科、学术论文以及开源代码库抓取文本。  
   - **质量过滤**：使用语言检测、重复去除和噪声清洗脚本，确保每条数据都是完整、自然的希腊语句子。  
   - **多音调与古希腊混入**：专门爬取古典文献库，保留原始的 polytonic 符号，并在词表中加入对应的 Unicode 编码，使模型能够直接看到这些符号。

2. **基础预训练**  
   - 以 Meta 的 Llama 3.1‑8B 为骨架，保持原始的 8 B 参数规模。  
   - 将准备好的希腊语语料与原始的多语言语料混合，比例倾向希腊语（约 70%），在自回归的语言建模任务上继续训练。这样模型的内部表示会更偏向希腊语的形态特征。

3. **指令微调 + 偏好对齐**  
   - **人工指令集**：作者团队手工编写了约 10 k 条希腊语指令，覆盖问答、翻译、代码解释等常见场景。  
   - **合成指令生成**：利用已经微调好的模型自生成指令-答案对，规模扩大到数十万条，形成“自教”数据。  
   - **偏好数据**：对每一对答案，收集人类或模型生成的偏好标签（哪一个更好），并用这些标签训练一个奖励模型，让主模型学习在生成时最大化奖励。  
   - **MAGPIE**：在每一次微调后，使用奖励模型对生成的答案进行再筛选，挑选出最符合指令意图的样本继续训练，形成闭环提升。

4. **多阶段 MAGPIE 强化**  
   - 该管线共进行三轮：第一轮基于人工指令，第二轮加入合成指令，第三轮使用偏好模型进行细粒度筛选。每轮结束后都会评估在内部验证集上的表现，确保每一步都在提升而不是过拟合。

**最巧妙的点**在于把“合成指令”与“偏好学习”结合起来，用 MAGPIE 把模型自己生成的答案再喂回模型自身，形成类似人类自我纠错的闭环，这在开源希腊语模型里是首次出现。

### 实验与效果
- **评测基准**：作者公开的三个希腊语基准分别是阅读理解（类似 SQuAD）、文本生成（摘要与对话）和代码生成（从自然语言描述生成 Python 代码）。  
- **对比对象**：包括开源的 GreekBERT、mBERT（多语言 BERT）以及最新的多语言 LLaMA‑2‑13B。  
- **结果**：论文声称 Krikri 在阅读理解准确率上领先 5–7%（相对提升），在生成流畅度和代码正确率上也有可观的提升。  
- **消融实验**：通过去掉合成指令或不使用 MAGPIE，模型在所有基准上都有明显下降，说明这两个模块是性能提升的关键。  
- **局限性**：作者承认模型仍然在极端长文本推理和少数方言上表现一般，且 8 B 参数规模在资源受限的环境下仍有部署挑战。

### 影响与延伸思考
Krikri 的发布填补了希腊语开源大模型的空白，随后出现了几篇基于其模型进行领域适配的工作，例如医学希腊语问答系统和古典文学自动注释工具。它的多阶段 MAGPIE 思路也被其他语言社区借鉴，用于提升低资源语言的指令微调效果。想进一步探索的读者可以关注以下方向：① 更高效的参数压缩（如 LoRA、QLoRA）在希腊语模型上的实践；② 将 polytonic 处理扩展到其他古典语言；③ 跨语言指令迁移，利用希腊语模型的指令能力帮助相邻语言提升。  

### 一句话记住它
Krikri 用专属希腊语语料和三阶段指令‑偏好管线，让 8 B 参数的开源模型在现代、古典和多音调希腊语上都能说得更好。