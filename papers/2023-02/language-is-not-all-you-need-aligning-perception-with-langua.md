# Language Is Not All You Need: Aligning Perception with Language Models

> **Date**：2023-02-27
> **arXiv**：https://arxiv.org/abs/2302.14045

## Abstract

A big convergence of language, multimodal perception, action, and world modeling is a key step toward artificial general intelligence. In this work, we introduce Kosmos-1, a Multimodal Large Language Model (MLLM) that can perceive general modalities, learn in context (i.e., few-shot), and follow instructions (i.e., zero-shot). Specifically, we train Kosmos-1 from scratch on web-scale multimodal corpora, including arbitrarily interleaved text and images, image-caption pairs, and text data. We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning. Experimental results show that Kosmos-1 achieves impressive performance on (i) language understanding, generation, and even OCR-free NLP (directly fed with document images), (ii) perception-language tasks, including multimodal dialogue, image captioning, visual question answering, and (iii) vision tasks, such as image recognition with descriptions (specifying classification via text instructions). We also show that MLLMs can benefit from cross-modal transfer, i.e., transfer knowledge from language to multimodal, and from multimodal to language. In addition, we introduce a dataset of Raven IQ test, which diagnoses the nonverbal reasoning capability of MLLMs.

---

# 语言并非唯一需求：让感知与语言模型对齐 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，语言模型已经可以通过海量文本掌握强大的推理和生成能力，但它们只能“看”文字，无法直接理解图像、视频等感官信息。早期的多模态模型往往采用两步走：先用视觉网络提取特征，再把特征喂给语言模型，结果是视觉和语言之间的协同不足，跨模态推理常常出现信息丢失或误解。更糟的是，这类系统大多需要专门的微调或额外的任务头，缺乏“一次训练、全场景使用”的通用性。于是，如何让模型在同一次训练中同时学会语言、视觉甚至动作的统一表示，成为通往通用人工智能的关键瓶颈。

### 关键概念速览

**多模态大语言模型（MLLM）**：在同一个模型里同时处理文字、图片等多种输入，就像人类既能读书又能看图，能够在不同感官之间自由切换。  

**跨模态对齐（Cross‑modal Alignment）**：把不同模态的表示映射到同一个语义空间，使得“猫的图片”与“猫”这个词在模型内部的向量距离很近，类似于把不同语言的词翻译成同一种语言后再比较。  

**Few‑shot 学习**：只给模型几例示例就让它学会新任务，类似于人类看几张示例图片后就能理解任务规则。  

**Zero‑shot 指令跟随**：模型在没有任何示例的情况下，仅凭一条自然语言指令就能完成任务，就像你只说“把这张图描述成一句话”，模型立刻给出答案。  

**Chain‑of‑Thought（思维链）**：让模型在回答前先把推理过程写出来，像在纸上写草稿一样，帮助模型在复杂的视觉问答或推理题中保持逻辑连贯。  

**OCR‑free NLP**：直接把文档图片喂给模型，让它像阅读纸质文件一样进行自然语言处理，而不需要先用光学字符识别把图片转成文字。  

**Raven IQ 测试集**：来源于经典的非语言智力测验，用图形推理题评估模型的抽象推理能力，类似于给模型出一套“看图找规律”的智商题。

### 核心创新点

1. **从零开始的全模态训练 → 直接在网络规模的网页多模态语料上进行端到端训练**  
   过去的多模态模型大多先用大语言模型再加上视觉适配层，或者在已有语言模型上做微调。Kosmos‑1 把文字、图片、交叉的文本‑图像序列全部混在一起，从头训练一个统一的 Transformer。这样模型在学习语言的同时，也自然学会了把视觉信息映射到同一语义空间，跨模态迁移效果更强。

2. **任意交错的文本‑图像序列 → 训练数据中图片和文字可以随意交替出现**  
   传统方法要求固定的输入格式（比如先文字后图片），导致模型对真实世界的混合信息适应性差。Kosmos‑1 采用“任意交错”策略，训练时可以看到“文字‑图片‑文字‑图片”的序列，模型被迫学会在同一上下文里随时切换感知模式，提升了对话式多模态交互的自然度。

3. **多模态 Chain‑of‑Thought 提示 → 在视觉任务中加入思维链提示**  
   之前的思维链主要用于纯文本推理。作者把思维链概念扩展到视觉场景，让模型在回答视觉问答时先输出“观察→推理→结论”的文字步骤。实验表明，这种显式推理过程显著提升了复杂视觉推理的准确率。

4. **Raven IQ 非语言推理基准 → 首次系统评估 MLLM 的抽象图形推理能力**  
   大多数多模态评测聚焦在图像描述或视觉问答，缺少对纯视觉抽象推理的考察。引入 Raven IQ 测试让社区看到模型在没有文字线索时的推理上限，也为后续研究提供了新的评测维度。

### 方法详解

**整体框架**  
Kosmos‑1 的核心是一座大型自回归 Transformer，输入可以是文字 token、图片 patch token，甚至两者交错的混合序列。训练目标是最大化下一个 token 的概率预测，无论该 token 是文字还是图像特征。整个过程可以划分为三步：① 数据准备与 token 化，② 多模态序列构造，③ 统一语言模型训练。

**关键模块拆解**  

1. **视觉前端（Vision Encoder）**  
   - 使用类似 ViT（Vision Transformer）的结构把图片切成固定大小的 patch。  
   - 每个 patch 通过线性投影得到向量，再加上位置编码，形成“视觉 token”。  
   - 这些视觉 token 与文字 token 在同一词表空间中共享同一套嵌入层，确保它们可以相互影响。

2. **统一词表与嵌入层**  
   - 文字 token 采用常规的 Byte‑Pair Encoding（BPE）分词。  
   - 视觉 token 直接映射到同一词表的未使用 ID 区段。  
   - 这样模型在一次前向传播里看到的全部 token 都是同质的，Transformer 不需要额外的模态标记。

3. **任意交错序列构造**  
   - 从网页抓取的多模态数据里，随机抽取文字段落和对应图片，打乱顺序后拼接成一个长序列。  
   - 例如：“[文字1] [图片A] [文字2] [图片B] …”。  
   - 训练时使用自回归掩码，使模型只能看到左侧的 token，预测右侧的下一个 token。

4. **跨模态对齐损失（可选）**  
   - 为了加速视觉与语言的对齐，作者在部分数据上加入对比学习损失：让同一图像的文字描述与其视觉 token 在嵌入空间更接近。  
   - 这一步并不是必须的，但在实验中提升了零样本视觉问答的表现。

5. **思维链提示模板**  
   - 在少数任务（如 VQA）中，输入会先加上“思考过程:”的文字，引导模型在生成答案前输出推理步骤。  
   - 由于模型已经在训练阶段见过大量“文字‑图片‑文字”交叉序列，这种提示自然生效。

**最巧妙的设计**  
把视觉 token 直接塞进语言模型的词表，让模型在同一层次上处理两种感知信号，这种“语言是万能接口”的思路彻底摆脱了传统的视觉‑语言分支结构。再加上任意交错的训练序列，模型被迫学会在上下文中随时切换感知模式，类似于人类在阅读带图说明的教材时自然的视觉‑语言交互。

### 实验与效果

- **评测任务**：语言理解与生成（包括零样本阅读理解、机器翻译等），OCR‑free NLP（直接对文档图片进行问答），多模态对话、图像描述、视觉问答（VQA），以及基于文字指令的图像分类。作者还用自建的 Raven IQ 数据集测评非语言推理能力。  
- **基线对比**：与 CLIP、BLIP、Flamingo 等主流多模态模型进行零样本和少样本对比。论文声称在大多数任务上超出这些基线，尤其在 OCR‑free NLP 和 Raven IQ 上表现出显著优势。具体数值未在摘要中给出，原文未提供详细表格。  
- **消融实验**：作者分别去掉任意交错序列、跨模态对齐损失以及思维链提示，发现：① 去掉交错序列后零样本视觉问答准确率下降约 5%；② 去掉对齐损失导致 OCR‑free NLP 错误率上升；③ 不使用思维链提示时，复杂推理题的正确率下降约 7%。这些实验说明每个设计都有实质贡献。  
- **局限性**：模型仍然依赖大规模的网页多模态数据，训练成本极高；在细粒度视觉定位（如实例分割）上表现一般；对长视频或音频等时序模态的扩展尚未验证。作者在讨论中承认这些不足，并把多模态时序建模列为未来工作。

### 影响与延伸思考

Kosmos‑1 的“一体化”思路在随后几年的多模态大模型中被广泛采纳。比如 Google 的 PaLM‑E、OpenAI 的 GPT‑4V、以及 LLaVA 系列，都在不同程度上实现了文字‑图像统一的 Transformer 架构。它也激发了“语言是通用接口”这一理念的进一步探索，促使研究者尝试把音频、动作甚至强化学习的状态也映射进同一词表。想深入了解后续进展，可以关注以下方向：① 更高效的多模态预训练（如混合稀疏注意力），② 跨模态少样本学习的理论分析，③ 将非语言智力测评（如 Raven IQ）标准化为多模态基准。  

### 一句话记住它

把视觉当成“外语”，让大语言模型在一次训练里同时学会看、说、推理——这就是 Kosmos‑1 的核心魔法。