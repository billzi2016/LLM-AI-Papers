# Deductive Closure Training of Language Models for Coherence, Accuracy,   and Updatability

> **Date**：2024-01-16
> **arXiv**：https://arxiv.org/abs/2401.08574

## Abstract

While language models (LMs) can sometimes generate factually correct text and estimate truth values of individual claims, these generally do not reflect a globally coherent, manipulable model of the world. As a consequence, current LMs also generate incorrect or nonsensical content, and are difficult to edit and bring up to date. We present a method called Deductive Closure Training (DCT) that uses LMs themselves to identify implications of (and contradictions within) the text that they generate, yielding an efficient self-supervised procedure for improving LM factuality. Given a collection of seed documents, DCT prompts LMs to generate additional text implied by these documents, reason globally about the correctness of this generated text, and finally fine-tune on text inferred to be correct. Given seed documents from a trusted source, DCT provides a tool for supervised model updating; if seed documents are sampled from the LM itself, DCT enables fully unsupervised fine-tuning for improved coherence and accuracy. Across the CREAK, MQUaKE, and Reversal Curse datasets, supervised DCT improves LM fact verification and text generation accuracy by 3-26%; on CREAK fully unsupervised DCT improves verification accuracy by 12%. These results show that LMs' reasoning capabilities during inference can be leveraged during training to improve their reliability.

---

# 演绎闭包训练：提升语言模型的一致性、准确性与可更新性 论文详细解读

### 背景：这个问题为什么难？

现有的大语言模型虽然能在单句层面给出看似可信的答案，却缺乏全局一致的世界观。它们往往在长篇生成时自相矛盾，或者把自己“记住”的事实和新信息混在一起，导致生成的文本出现错误或荒唐的内容。更糟的是，模型一旦训练完毕，想要纠正其中的错误或加入最新知识，需要重新大规模微调，成本高且不够灵活。换句话说，模型既不够“可靠”，也不够“可编辑”。这正是本文要破解的核心难题。

### 关键概念速览
- **演绎闭包（Deductive Closure）**：指在一组已知事实上，利用逻辑推理生成所有可以必然推出的结论，就像把已知的拼图全部拼完，看到完整的图案。  
- **自监督（Self‑Supervised）**：模型自己产生训练信号，而不依赖外部标注。这里指模型用自己的推理能力来判断哪些生成的句子可信。  
- **种子文档（Seed Documents）**：作为起点的高质量文本，可能来自可信来源，也可能是模型自己生成的。  
- **全局一致性（Global Coherence）**：文本内部的前后关系保持逻辑连贯，不出现自相矛盾或前后信息冲突。  
- **可更新性（Updatability）**：模型在保持已有知识的同时，能够快速吸收新信息或纠正错误。  
- **事实验证（Fact Verification）**：判断一条陈述是否与已知事实相符的任务，常用来评估模型的真实性。  
- **逆转诅咒（Reversal Curse）**：一种测试模型是否会在否定句中产生错误推理的基准。  

### 核心创新点
1. **利用模型自身进行闭环推理 → 让模型在生成文本后再一次“审稿”，检查自己写的内容是否能从种子文档中演绎出来 → 过滤掉不符合逻辑的句子，提升整体事实准确率。**  
2. **把推理过程嵌入微调流水线 → 先让模型生成“隐含句子”，再用同一个模型评估这些句子的真伪，最后只把被判为正确的句子加入训练集 → 形成一种自监督的细化循环，省去人工标注成本。**  
3. **兼容监督与完全无监督两种场景 → 当种子文档来自可信来源时，整个流程相当于有标签的增量学习；当种子文档直接抽取自模型自身时，则实现了完全自我提升 → 同时解决了可编辑性和可扩展性的问题。**  
4. **在多个事实验证基准上实现显著提升 → 在CREAK、MQUaKE、Reversal Curse等数据集上，监督式DCT提升3%~26%的验证准确率，完全无监督模式在CREAK上提升12% → 证明了推理能力可以在训练阶段被有效利用。**  

### 方法详解
**整体框架**  
DCT 把训练过程拆成三步：① 生成闭包文本，② 全局一致性评估，③ 过滤后微调。整个循环可以在有标签的种子文档上跑，也可以在模型自生的文档上跑，核心是让模型自己当“审稿人”。  

**步骤拆解**  

1. **种子文档准备**  
   - **监督模式**：从可信来源（如百科、官方文档）抽取若干段落，确保这些文本本身是高质量的事实集合。  
   - **无监督模式**：直接让语言模型在给定主题下生成若干段落，作为临时的种子文档。  

2. **闭包生成（Deductive Closure Generation）**  
   - 给模型一个提示，要求它基于种子文档“写出所有可以推导出的新陈述”。提示示例类似：“已知以下事实，请列出它们蕴含的其他事实”。  
   - 模型输出的每条陈述被视为潜在的“闭包句”。这一步相当于让模型做一次链式推理，把已有信息扩展成更丰富的知识图。  

3. **全局一致性评估（Global Consistency Check）**  
   - 再次调用模型，给出种子文档和闭包句，让模型判断闭包句是否与种子文档**不矛盾**且**可以被演绎**。常用的提示是：“下面这句话是否可以从前面的事实推出？如果可以，请回答‘Yes’，否则‘No’”。  
   - 这里的判断是二分类的自监督信号，模型的输出决定该句是否进入下一步。  

4. **过滤与微调（Filtering & Fine‑tuning）**  
   - 只保留被判为“正确”的闭包句，拼接回原始种子文档形成扩展文本。  
   - 使用标准的语言模型微调流程（如全参数或LoRA）在这些扩展文本上继续训练。因为训练数据已经经过模型自己的逻辑审查，模型在学习时会倾向于强化一致且可演绎的表述。  

**关键细节**  
- **提示工程**：作者发现只要把“演绎”或“蕴含”这类关键词放进提示，模型的推理质量会显著提升，这是一种低成本的技巧。  
- **循环迭代**：DCT 可以多轮执行，每轮产生的新闭包会成为下一轮的种子文档，形成逐步扩张的知识闭包。  
- **不需要外部标注**：整个过程只依赖模型自身的输出和判断，省去了人工标注的瓶颈。  

### 实验与效果
- **数据集**：CREAK（事实验证）、MQUaKE（多选问答）和 Reversal Curse（否定推理）三个公开基准。  
- **对比基线**：原始未微调的语言模型、传统监督微调、以及最近的自监督事实增强方法。  
- **结果**：在监督 DCT 场景下，验证准确率提升范围为 3%~26%；在完全无监督模式下，仅在 CREAK 上就提升了 12%。这些数字表明即使不引入额外标注，模型的事实可靠性也能得到可观提升。  
- **消融实验**：作者分别去掉闭包生成、全局检查或过滤步骤，发现去掉任何一步都会导致性能回落到基线水平，说明三者缺一不可。  
- **局限性**：论文承认 DCT 对模型的推理能力有一定依赖，如果底层模型本身推理不佳，闭包生成和检查的质量会下降；此外，循环迭代的计算成本随轮数线性增长。  

### 影响与延伸思考
DCT 把“推理在推理时”这一想法落地，开启了“训练期间利用模型自检”的新思路。随后的工作开始探索更复杂的自监督逻辑校正，如利用图结构进行闭环验证、或把外部符号推理器嵌入微调流程。对想进一步研究的读者，可以关注以下方向：  
- **可解释的自监督校正**：把模型的判断过程可视化，帮助人类审查。  
- **跨模态闭包**：把文本闭包扩展到图像、表格等多模态信息。  
- **高效迭代策略**：设计更轻量的闭包生成与检查，以降低计算开销。  

### 一句话记住它
让语言模型自己写出所有能推导的事实，再让它自行筛选真假，形成闭环微调，从而在训练时就把“推理能力”变成“可靠性”。