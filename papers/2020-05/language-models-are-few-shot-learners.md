# Language Models are Few-Shot Learners

> **Date**：2020-05-28
> **arXiv**：https://arxiv.org/abs/2005.14165

## Abstract

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

---

# 语言模型是少样本学习者 论文详细解读

### 背景：这个问题为什么难？

在 GPT‑3 之前，NLP 里最常见的提升方式是先在海量文本上预训练，再在每个下游任务上用几千甚至上万标注样本进行微调。虽然这种“预训练 + 微调”框架在翻译、阅读理解等任务上取得了惊人的成绩，但它把每个任务都当成了独立的学习目标，需要专门收集、清洗、标注大量数据。人类却往往只看几例说明或一句指令就能完成新任务，这种“一看就会”的能力在机器上几乎没有实现。于是，如何让同一个模型在没有梯度更新、只靠极少示例的情况下完成多种任务，成为了当时的瓶颈。

### 关键概念速览

**Few‑Shot Learning（少样本学习）**：模型只看到几条输入‑输出对（通常不超过 10 条）就要完成任务，类似于老师只给几道例题，学生就能自行解答。

**Zero‑Shot Learning（零样本学习）**：模型在没有任何示例的情况下，仅凭任务描述或提示完成任务，像是直接让模型猜测答案。

**In‑Context Learning（情境学习）**：把示例直接写进模型的输入序列里，让模型在“阅读”这些上下文后自行推断该怎么做，等同于在对话中给模型演示几次操作。

**Autoregressive Language Model（自回归语言模型）**：模型一次预测下一个词，然后把这个词接到已有文本后继续预测，像是写故事时一步步往下写。

**Meta‑Learning（元学习）**：模型在训练阶段学习“学习的技巧”，所以在新任务上只需要少量信息就能快速适应。

**Prompt（提示）**：给模型的文字指令或示例，决定模型把输入当成哪种任务来处理。

### 核心创新点

1. **规模化 → 175 B 参数的 GPT‑3**  
   之前的最大非稀疏语言模型最多约 10 B 参数。作者把模型放大到 175 B，约是前代的 10 倍。更大的模型在预训练阶段捕获了更丰富的语言规律和世界知识，为后续的少样本推理提供了“底层材料”。

2. **纯文本情境学习 → 不再需要梯度更新**  
   传统微调需要把任务数据喂进模型、计算梯度、更新参数。GPT‑3 直接把少量示例和任务描述拼接成一段文字，交给模型一次前向计算。这样模型在推理时不做任何参数改动，完全依赖上下文信息来决定行为。

3. **统一评估框架 → 同一模型覆盖多种任务**  
   作者在翻译、问答、填空、算术、词语重排、造新词等十余个基准上使用相同的提示方式进行评测。相比过去每个任务都要单独设计微调头，GPT‑3 展示了“一套模型、一次调用”即可应对多样任务的可能性。

4. **系统性错误分析 → 揭示模型的局限**  
   通过对比 few‑shot、one‑shot、zero‑shot 表现，作者指出在某些数据集（如需要严格逻辑推理或受训练语料偏差影响的任务）仍然表现不佳。这种自我审视帮助后续研究聚焦改进点。

### 方法详解

**整体思路**  
GPT‑3 的核心流程只有两步：① 用海量网页文本进行自回归预训练；② 在下游任务时，把任务描述和少量示例写进同一段输入（prompt），让模型直接生成答案。没有任何梯度传播，也不需要额外的任务专用层。

**步骤拆解**  

1. **大规模自回归预训练**  
   - 数据来源：公开的网络爬取语料，覆盖新闻、维基、代码等多种体裁。  
   - 目标：预测每个位置的下一个词，模型通过最大化预测概率学习语言统计。  
   - 规模：175 B 参数，使用数千 GPU 天的算力，训练步数达到数十万。

2. **构造 Prompt**  
   - **任务说明**：用自然语言简短描述要做的事，例如 “Translate English to French”。  
   - **示例对**：提供 1‑5 条输入‑输出对，格式与任务说明保持一致。  
   - **待预测输入**：在示例后紧跟需要模型回答的句子。  
   - 整体看起来像一段对话或练习册，模型在阅读完所有前文后，继续生成下一个词，这个词就是答案。

3. **模型推理**  
   - 采用束搜索或温度采样等解码技巧，控制生成的多样性与确定性。  
   - 由于模型已经在预训练阶段学到了大量知识，prompt 中的少量示例足以激活相应的内部“子网络”，从而完成任务。

**最巧妙的地方**  
把任务信息塞进文字序列，而不是通过显式的参数调节，是一种“软调优”。这让模型在一次前向传播里完成“学习‑推理”两个过程，类似于人类在阅读说明后立刻动手，而不需要额外的练习。

### 实验与效果

- **测试任务**：包括英法/英德翻译、SQuAD 问答、LAMBADA 填空、Winograd 语义消歧、3‑位数加减、单词重排、使用新造词造句等。  
- **对比基线**：传统 fine‑tune 的 BERT、RoBERTa、T5 等模型，以及之前的 GPT‑2 few‑shot 实验。  
- **表现**：在多数任务上，GPT‑3 的 few‑shot 结果接近或超过了同任务的 fine‑tune SOTA。例如在翻译任务上，few‑shot BLEU 分数仅比专门微调的模型低约 1‑2 分；在 Winograd 任务上，few‑shot 正确率突破 70%，显著高于零样本的 55%。  
- **消融实验**：作者分别去掉示例、改写任务描述、调低模型规模，发现示例数量和模型大小是提升的关键因素。小于 1 B 参数的模型几乎没有 few‑shot 能力。  
- **局限**：在需要严格数学推理或对训练语料高度偏倚的领域（如专业医学问答），GPT‑3 仍会产生错误或幻觉式答案。作者也指出，模型在大规模网页语料上训练导致某些社会偏见的迁移。

### 影响与延伸思考

这篇论文让整个社区重新审视“模型大小”与“任务适配”之间的关系，催生了大量围绕 **Prompt Engineering（提示工程）** 的研究——如何设计更高效的文字提示、如何自动生成提示、以及如何在少样本情境下提升鲁棒性。随后出现的 **In‑Context Learning**、**Chain‑of‑Thought**、**Instruction‑Tuned** 系列模型（如 FLAN‑T5、ChatGPT）都直接或间接受 GPT‑3 思路启发。对想进一步探索的读者，可以关注：

- **提示优化算法**（如 AutoPrompt、Prompt Tuning）  
- **更高效的少样本学习框架**（如 Meta‑ICL）  
- **大模型安全与偏见治理**（因为规模放大也放大了潜在风险）

### 一句话记住它

**把任务写进文字，让超大语言模型只靠一次前向传播就能“现场学习”。**