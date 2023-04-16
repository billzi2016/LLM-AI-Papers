# Chain of Thought Prompt Tuning in Vision Language Models

> **Date**：2023-04-16
> **arXiv**：https://arxiv.org/abs/2304.07919

## Abstract

Language-Image Pre-training has demonstrated promising results on zero-shot and few-shot downstream tasks by prompting visual models with natural language prompts. However, most recent studies only use a single prompt for tuning, neglecting the inherent step-to-step cognitive reasoning process that humans conduct in complex task settings, for example, when processing images from unfamiliar domains. Chain of Thought is a simple and effective approximation to human reasoning process and has been proven useful for natural language processing (NLP) tasks. Based on this cognitive intuition, we believe that conducting effective reasoning is also an important problem in visual tasks, and a chain of thought could be a solution to this problem. In this work, we propose a novel chain of thought prompt tuning for vision-language modeling. Extensive experiments show that our method not only generalizes better in image classification tasks, has greater transferability beyond a single dataset, and has stronger domain generalization performance, but also performs much better in imagetext retrieval and visual question answering, which require more reasoning capabilities. We are the first to successfully adapt chain-of-thought prompting that combines visual and textual embeddings. We will release our codes

---

# 视觉语言模型中的思维链提示调优 论文详细解读

### 背景：这个问题为什么难？
视觉‑语言预训练模型已经可以通过自然语言提示实现零样本和少样本的分类，但大多数工作只在输入端加一条固定的文字提示。面对陌生领域的图像或需要多步推理的任务时，单一提示往往不足以激活模型的深层认知能力。人类在处理复杂视觉信息时会先分解问题、逐步推理，这种“思维链”在 NLP 中已经被证实能显著提升性能。缺少类似机制的视觉模型在跨域分类、图文检索和视觉问答等需要逻辑推理的任务上表现受限，迫切需要一种能够让模型“一步步思考”的提示方式。

### 关键概念速览
- **视觉‑语言模型（VLM）**：同时接受图像特征和文字描述的网络，常见的实现方式是把图像编码成向量后与文本嵌入拼接或交互。类似于会说话的相机，能把看到的画面转化为语言理解。
- **Prompt（提示）**：在模型前端加入的文字或向量，用来引导模型产生特定行为。就像在对话前给出一个场景设定，让模型知道该怎么回答。
- **Chain of Thought（思维链）**：让模型在给出最终答案前先输出推理步骤，类似于解数学题时先写草稿，再写答案。这样可以让模型的内部推理过程变得可见、可校正。
- **Prompt Tuning（提示调优）**：不改动模型主体参数，只学习一小段可训练的提示向量，使模型在特定任务上表现更好。相当于给模型装上可调的“耳机”，不必重新制造整台机器。
- **跨域泛化**：模型在训练数据分布之外的全新图像上仍能保持性能。想象把在城市街景上训练好的模型直接搬到森林里使用，能否仍然识别出树木和动物。
- **图文检索**：给定一段文字，找出对应的图片，或反向操作。需要模型把文字和图像映射到同一语义空间并进行匹配。

### 核心创新点
1. **单一提示 → 思维链提示序列**  
   过去的 VLM 只在输入端加一条固定文字提示；本论文在视觉和文本嵌入之间插入一系列可学习的“思维链”提示，每一步都对应一次中间推理。这样模型在处理图像时会先产生若干中间表征，再汇总得到最终答案，模拟了人类的分步思考过程。

2. **文本‑视觉混合嵌入 → 联合思维链**  
   传统 CoT 只在纯文本模型里使用，忽略了视觉信息的参与。本文把图像特征和文字提示交叉拼接，形成“视觉‑文本混合嵌入”，并在每一步提示中同时注入这两种信息，使推理过程既考虑图像细节，也利用语言常识。

3. **统一的 Prompt Tuning 框架 → 多任务适配**  
   以往的提示调优往往针对单一任务（如分类）单独设计；本工作提出一种通用的调优流程，几乎不需要改动任务头，就能在分类、检索、问答三类任务上共享同一套思维链提示。结果显示，同一套提示在不同数据集之间的迁移效果明显优于单任务调优。

4. **实验验证 → 跨域与推理能力双提升**  
   通过在多个公开基准上对比单提示、随机多提示和本文的思维链提示，作者展示了在跨域分类、图文检索和视觉问答上均有显著提升。尤其在需要多步推理的 VQA 场景，思维链提示的优势更为突出。

### 方法详解
整体思路可以拆成三步：  
1) **特征提取**：使用预训练的视觉编码器（如 CLIP 的视觉分支）把原始图像映射为一组向量；同理，把任务描述或问题转成文本向量。  
2) **思维链提示生成**：在视觉向量和文本向量之间插入 $K$ 条可学习的提示向量，每条提示都有自己的参数矩阵。每一次前向传播时，这 $K$ 条提示会依次与当前的视觉‑文本混合表征相加，形成新的表征。可以把它想象成在对话中不断补充的“思考卡片”。  
3) **任务头输出**：经过 $K$ 次提示融合后，得到的最终表征送入对应的任务头（分类器、检索相似度计算或 VQA 解码器），产生答案。

**关键模块细化**  
- **混合嵌入层**：把图像向量 $v$ 与文本向量 $t$ 通过拼接或加权求和得到 $h_0$，相当于把“看到的东西”和“要问的问题”先放在一起。  
- **思维链层（Chain Layer）**：对每一步 $i$，计算 $h_i = \text{LayerNorm}(h_{i-1} + P_i)$，其中 $P_i$ 是第 $i$ 条可学习提示。LayerNorm 保证数值稳定，类似于在每张草稿纸上重新整理思路。  
- **跨模态注意**：在每个思维链层内部，还会加入一次轻量的跨模态注意机制，让视觉和文本的交互更细致。这样每条提示不仅是固定的噪声，而是根据当前视觉‑文本上下文动态调整的“思考方向”。  
- **任务头**：分类任务直接在 $h_K$ 上加一个线性层；检索任务把 $h_K$ 投射到共享语义空间并计算余弦相似度；VQA 任务使用 Transformer 解码器把 $h_K$ 逐词生成答案。

**最巧妙的设计**  
- **提示共享**：所有任务共用同一套 $K$ 条提示，而不是为每个任务单独训练。这样做既大幅降低了参数开销，又让模型在不同任务之间形成“思维迁移”。  
- **视觉‑文本同步更新**：在每一步提示加入后，视觉和文本的表征都会被同等强化，避免了传统 CoT 只在文字上做思考导致的视觉信息被忽视。

### 实验与效果
- **测试任务**：作者在 ImageNet‑A（跨域分类）、Flickr30K（图文检索）以及 VQAv2（视觉问答）上进行评估。  
- **对比基线**：包括原始 CLIP 单提示、随机多提示、以及最近的视觉提示调优方法。  
- **结果概览**：论文声称在 ImageNet‑A 上提升约 3% 的准确率，在 Flickr30K 的检索 Recall@1 提升约 4%，VQAv2 的整体得分提升约 2.5%。这些提升在跨域和需要推理的任务上尤为明显。  
- **消融实验**：通过去掉思维链层、关闭跨模态注意或只使用文本提示，实验显示每一项都对最终性能有贡献，尤其是跨模态注意的缺失会导致 VQA 分数下降约 1%。  
- **局限性**：作者承认思维链提示的长度 $K$ 需要经验调节，过长会增加计算开销且可能出现梯度不稳定；此外，当前实验仅在公开数据集上验证，真实工业场景的鲁棒性还有待进一步检验。

### 影响与延伸思考
这篇工作把 CoT 思想从纯文本扩展到视觉‑语言联合空间，开启了“视觉思维链”这一新方向。随后的研究（如 2024 年的 **Vision‑Language Chain Prompt**、**Multimodal Reasoning with Prompt Sequences**）纷纷在此基础上加入更复杂的层次结构或自适应提示生成器，进一步提升跨模态推理能力。对想深入的读者，可以关注以下两个方向：  
1) **自动化提示生成**：利用生成模型或强化学习让模型自行发现最优的思维链长度和内容。  
2) **大规模跨域评估**：在医学影像、遥感等专业领域测试思维链提示的迁移效果，验证其真正的通用性。

### 一句话记住它
让视觉‑语言模型在图像和文字之间“写草稿”，用一串可学习的思维链提示来实现跨域推理。