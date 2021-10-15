# Multitask Prompted Training Enables Zero-Shot Task Generalization

> **Date**：2021-10-15
> **arXiv**：https://arxiv.org/abs/2110.08207

## Abstract

Large language models have recently been shown to attain reasonable zero-shot generalization on a diverse set of tasks (Brown et al., 2020). It has been hypothesized that this is a consequence of implicit multitask learning in language models' pretraining (Radford et al., 2019). Can zero-shot generalization instead be directly induced by explicit multitask learning? To test this question at scale, we develop a system for easily mapping any natural language tasks into a human-readable prompted form. We convert a large set of supervised datasets, each with multiple prompts with diverse wording. These prompted datasets allow for benchmarking the ability of a model to perform completely held-out tasks. We fine-tune a pretrained encoder-decoder model (Raffel et al., 2020; Lester et al., 2021) on this multitask mixture covering a wide variety of tasks. The model attains strong zero-shot performance on several standard datasets, often outperforming models up to 16x its size. Further, our approach attains strong performance on a subset of tasks from the BIG-bench benchmark, outperforming models up to 6x its size. All trained models are available at https://github.com/bigscience-workshop/t-zero and all prompts are available at https://github.com/bigscience-workshop/promptsource.

---

# 多任务提示训练实现零样本任务泛化 论文详细解读

### 背景：这个问题为什么难？
大语言模型在预训练阶段已经展示出一定的零样本能力，但这种能力是“隐式的”，也就是说模型并没有被明确教会如何把一种任务的经验迁移到全新任务上。传统的微调方式往往只针对单一数据集进行训练，缺乏跨任务的通用知识，导致模型在遇到未见任务时表现急剧下滑。更糟的是，构造大规模的多任务数据集并让模型同时学习，往往会因为任务描述不统一、格式差异大而让训练过程变得混乱。于是，如何用一种统一、可读的方式把各种自然语言任务包装起来，让模型在一次训练中真正学会“任务通用的技巧”，成为了迫切需要解决的难题。

### 关键概念速览
- **Prompt（提示）**：把任务描述成一段自然语言文字，让模型把输入当成“问题”，输出当成“答案”。类似老师在课堂上给出的题目说明。  
- **Promptsource**：一个开源库，收集并自动生成每个任务的多种提示模板，确保同一任务可以用不同的说法表达，增加模型对提示多样性的适应能力。  
- **Zero‑Shot（零样本）**：模型在没有看到目标任务的训练数据的情况下直接完成该任务，就像人第一次玩新游戏，只靠说明书就能上手。  
- **Encoder‑Decoder（编码器‑解码器）**：一种模型结构，先把输入文字编码成向量，再把向量解码成输出文字，类似翻译系统的工作流程。  
- **Multitask Mixture（多任务混合）**：把来自上百个不同任务的提示数据混在一起训练，让模型在一次迭代中看到各种任务的例子。  
- **T0（Task‑Zero）**：本文训练得到的模型名称，意指它在零样本设置下能够完成多种任务。  
- **BIG‑bench**：一个包含上千种挑战性任务的基准套件，用来检验模型的通用智能水平。  

### 核心创新点
1. **显式多任务提示 → 统一的 Prompt‑Source 体系 → 让模型在训练阶段直接学习任务描述的通用模式**  
   过去的零样本能力主要归功于大规模语言模型的隐式学习。本文把每个监督任务都转化为可读的提示，并用 Promptsource 自动生成多样化的模板，使得模型在训练时就能看到“任务+答案”的完整配对，而不是仅仅靠语言本身的统计规律。

2. **大规模多任务混合 → 同时微调上百任务 → 零样本泛化显著提升**  
   传统微调一次只针对单一任务，模型容易过拟合。这里把数百个任务的提示数据混合在一起，形成一个巨大的多任务数据池，模型在一次训练中学习到跨任务的共享表示，从而在全新任务上也能快速推理。

3. **多提示多措辞 → 同一任务的不同表述 → 增强模型对提示语言的鲁棒性**  
   对每个任务提供多个不同 wording 的提示，类似给学生同一道题的不同表述方式，帮助模型不把答案绑定到特定的词句上，而是学会抽象的任务结构。

4. **使用 Encoder‑Decoder 结构进行指令微调 → 直接生成答案而非分类标签 → 更自然的零样本推理**  
   与只输出类别的分类头不同，Encoder‑Decoder 能把提示和答案统一成文本序列，省去任务特定的输出层设计，使得模型在面对新任务时只需“读懂提示”，即可生成合适的答案。

### 方法详解
整体思路可以拆成三步：**任务收集 → 提示生成 → 多任务微调**。

1. **任务收集**  
   作者挑选了数百个公开的监督数据集，覆盖文本分类、情感分析、问答、摘要、翻译等多种语言任务。每个数据集都提供了标准的输入‑输出对。

2. **提示生成（Promptsource）**  
   对每个任务，Promptsource 自动生成多条自然语言提示模板。例如，对情感分析任务，可能会有“这句话的情感是正面还是负面？”、“请判断下面这段文字的情绪倾向”。每条模板都对应一个 **input‑template**（把原始输入填进去）和 **target‑template**（把期望输出填进去）。这样，同一任务会产生 5‑10 种不同的提示，使得训练数据在语言表述上更加丰富。

3. **多任务混合与微调**  
   所有任务的提示对（input‑target）被统一拼接成一个巨大的训练集。模型采用 T5/FLAN 系列的 Encoder‑Decoder 结构（预训练权重来自 T5），在此数据上进行指令微调。训练目标是让模型在看到 **prompt + input** 后，直接生成 **target**。因为所有任务都用同一种“读‑写”方式，模型不需要额外的任务标识，只靠提示文字来区分任务。

   **关键细节**  
   - **随机抽样**：每个训练步骤从所有任务中均匀抽取一个 batch，防止大数据集压倒小数据集。  
   - **长度归一化**：对不同任务的输入/输出进行统一的最大长度截断，保证 GPU 利用率。  
   - **学习率调度**：使用 AdamW 优化器，先用较高学习率预热，再逐步衰减，帮助模型在多任务之间平滑迁移。  
   - **提示多样性**：在同一次训练中，同一任务的不同提示会交叉出现，迫使模型学会抽象的任务结构，而不是记住固定的词序列。

   **最巧妙的地方**  
   作者没有在模型内部加入任何任务编码或额外的元信息，完全依赖自然语言提示本身来传递任务指令。这种“纯语言驱动”的设计让模型在真正的零样本情境下，只需要给出合适的提示，就能激活对应的能力。

### 实验与效果
- **测试数据**：作者在多个公开的零样本基准上评估，包括 SuperGLUE、OpenAI‑Evals、以及 BIG‑bench 中挑选的子集。  
- **对比基线**：与同规模的原始 T5、GPT‑Neo、以及更大的 GPT‑3（约 175B）等模型进行比较。  
- **结果**：论文声称 T0 在多数标准任务上能够匹配甚至超越体积大 6‑16 倍的模型。例如，在 SuperGLUE 的 RTE 子任务上，T0 达到与 6 倍参数模型相当的准确率；在 BIG‑bench 的若干挑战上，T0 的表现超过了 6 倍参数的基线。  
- **消融实验**：作者分别去掉多提示、多任务混合、以及 Encoder‑Decoder 微调三个组件，发现每去掉一项，零样本性能都会出现显著下降，尤其是去掉多提示后，模型对新提示的适应能力大幅削弱。  
- **局限性**：论文承认 T0 仍然在高度专业化或需要深层推理的任务上表现不佳；此外，提示的质量对最终效果有较大影响，自动生成的提示有时会产生歧义。

### 影响与延伸思考
这篇工作开启了“指令微调”在大模型上的系统化探索，直接催生了后续的 FLAN、InstructGPT、ChatGPT 等系列模型。研究者们纷纷围绕 **提示工程**、**多任务指令学习**、以及 **更高效的提示生成** 开展新工作。未来的方向可能包括：自动化搜索最优提示、把提示学习与模型内部表示对齐、以及在更大规模的多语言、多模态任务上验证指令微调的通用性。如果想进一步了解，可以关注 **Promptsource** 的最新更新以及 **OpenAI 的 InstructGPT** 系列论文，它们在方法细节和规模上都有进一步的突破。

### 一句话记住它
只要把各种任务包装成自然语言提示并一起微调，模型就能在零样本情况下直接“读懂指令”，实现跨任务的通用推理。