# Think Before You Speak: Explicitly Generating Implicit Commonsense   Knowledge for Response Generation

> **Date**：2021-10-16
> **arXiv**：https://arxiv.org/abs/2110.08501

## Abstract

Implicit knowledge, such as common sense, is key to fluid human conversations. Current neural response generation (RG) models are trained to generate responses directly, omitting unstated implicit knowledge. In this paper, we present Think-Before-Speaking (TBS), a generative approach to first externalize implicit commonsense knowledge (think) and use this knowledge to generate responses (speak). We expect that externalizing implicit knowledge allows more efficient learning, produces more informative responses, and enables more explainable models. We analyze different choices to collect knowledge-aligned dialogues, represent implicit knowledge, and transition between knowledge and dialogues. Empirical results show TBS models outperform end-to-end and knowledge-augmented RG baselines on most automatic metrics and generate more informative, specific, and commonsense-following responses, as evaluated by human annotators. TBS also generates knowledge that makes sense and is relevant to the dialogue around 85\% of the time.

---

# 先思考后发言：显式生成隐式常识用于对话生成 论文详细解读

### 背景：这个问题为什么难？
在日常聊天里，人们常常依赖未说出口的常识来让对话顺畅，例如“下雨了，记得带伞”。传统的神经对话生成模型直接从上下文映射到回复，忽略了这些隐含的常识信息，导致生成的句子往往平淡、缺乏信息或出现常识错误。根本原因在于模型没有显式的“思考”步骤，所有隐含知识只能在大规模数据中被动学习，学习效率低且难以解释。于是，如何让模型先把隐式常识外化，再基于这些知识生成回答，成为提升对话质量的关键瓶颈。

### 关键概念速览
- **隐式常识（Implicit Commonsense）**：人们在对话中默认的、未明确说出的背景知识，例如“咖啡会让人提神”。它像对话的潜在润滑剂，缺失时对话会显得生硬。  
- **显式化（Externalization）**：把隐式常识转化为可见的文字或结构化信息，就像把脑中的想法写在纸上，便于后续使用和检查。  
- **Think‑Before‑Speaking（TBS）**：本文提出的两阶段生成框架，先“思考”（生成常识）再“发言”（生成回复），类似先写提纲后写正文的写作流程。  
- **知识对齐对话（Knowledge‑Aligned Dialogue）**：在训练数据中，给每轮对话配上一段对应的常识描述，使模型能够学习“对话 ↔ 常识”的映射关系。  
- **知识转移模块（Knowledge‑to‑Utterance Module）**：把已经生成的常识转化为回复的具体语言表达，类似把提纲中的要点展开成完整句子。  
- **可解释性（Explainability）**：模型的输出不仅有回复，还能提供生成该回复的常识依据，便于人类审查和调试。  

### 核心创新点
1. **从端到端到两阶段**：传统对话模型直接从上下文生成回复 → TBS 先让模型生成一段与当前对话相关的常识文本 → 再把这段常识作为条件生成最终回复。这样把隐式知识显式化，使模型在学习时可以专注于常识的抽取与使用两个子任务，提升了信息密度和常识一致性。  
2. **构建知识对齐对话数据**：以往的对话数据缺少对应的常识标注 → 作者通过人工或半自动方式为每轮对话配上显式常识句子，形成“对话‑常识”配对。此举为模型提供了明确的学习信号，使得常识生成不再是模糊的副产物，而是可评估的子任务。  
3. **知识表示与转移的双向桥接**：直接把生成的常识喂入回复生成器往往会出现信息丢失 → TBS 设计了一个专门的知识转移模块，使用注意力机制让回复生成器在每一步都能检索到常识中的关键片段，确保常识被完整、精准地融入回复。  
4. **统一评估常识质量与回复质量**：大多数对话研究只看回复的流畅度或BLEU分数 → 本文同时评估生成的常识是否合理、是否与对话相关，并报告约85% 的常识被人类评审认为是“有意义且相关”。这种双重评估让模型的解释性得到量化验证。  

### 方法详解
**整体框架**  
TBS 把对话生成拆成两步：  
1. **Think（思考）**：给定对话历史，模型生成一段描述当前情境所需的隐式常识。  
2. **Speak（发言）**：把第一步得到的常识作为额外条件，生成最终的回复。  

**步骤拆解**  

1. **输入编码**  
   - 对话历史（多轮发言）先经过预训练的语言模型编码，得到上下文向量。  
   - 同时，若已有知识对齐对话对，常识文本也会被编码，用于监督学习。  

2. **常识生成器（Think 模块）**  
   - 采用标准的自回归生成网络（如 Transformer 解码器），在“上下文向量 + 特殊触发标记”条件下生成常识句子。  
   - 训练目标是最小化生成常识与人工标注常识之间的交叉熵损失。  
   - 为防止模型直接复制对话内容，作者在训练时加入了遮盖（mask）机制，只允许模型使用对话中未出现的词汇来表达常识。  

3. **知识转移层**  
   - 常识生成完毕后，得到一系列 token 向量。  
   - 在回复生成阶段，这些向量与对话上下文向量一起进入多头注意力层。  
   - 注意力机制让每个回复 token 在生成时都可以“查询”常识向量，类似于写作时随时翻看提纲。  

4. **回复生成器（Speak 模块）**  
   - 同样使用 Transformer 解码器，但其输入是对话上下文向量、常识向量以及已经生成的回复 token。  
   - 目标是最大化真实回复的似然，同时通过 KL 散度约束让生成的回复在语义上与常识保持一致。  

5. **训练与推理**  
   - 训练时两阶段是串联的：先优化 Think 模块，再在固定的 Think 输出上优化 Speak 模块，或者采用端到端的联合训练（交叉熵加权）。  
   - 推理时，模型先生成常识（可设定采样温度），随后立即使用该常识生成回复，整个过程只需一次前向传播，效率与传统端到端模型相当。  

**巧妙之处**  
- **显式常识作为可解释中间产物**：常识文本本身可以直接展示给用户或审查员，提供了模型决策的“理由”。  
- **遮盖策略防止信息泄露**：通过限制模型在 Think 阶段使用对话中已有的词，迫使模型真正“推理”出隐含信息，而不是简单复制。  
- **双向注意力桥接**：常识向量不只是一次性拼接，而是通过注意力在每一步动态检索，保证常识在回复中的使用是细粒度、上下文相关的。  

### 实验与效果
- **数据集与任务**：作者在公开的对话数据集（如 Persona‑Chat、DailyDialog）上构造了知识对齐版本，每轮对话配有人工标注的常识句子。任务是给定对话历史生成自然、符合常识的回复。  
- **对比基线**：包括传统的端到端生成模型（如 Seq2Seq、Transformer）、以及已有的知识增强模型（如检索式常识注入、知识图谱辅助）。  
- **主要结果**：在大多数自动评测指标（BLEU、ROUGE、Distinct‑n）上，TBS 超过所有基线；在人类评审中，TBS 的回复被评为更信息丰富、更具体且更符合常识。常识生成的质量约有 85% 被标注者认为是“有意义且与对话相关”。  
- **消融实验**：去掉遮盖策略、去掉知识转移层或直接使用端到端训练，模型的常识质量和回复质量均出现明显下降，说明每个设计都有实质贡献。  
- **局限性**：论文承认常识生成仍受限于训练数据的覆盖范围，面对极其专业或罕见情境时常识可能不完整或错误；此外，显式生成常识会增加推理时的计算开销。  

### 影响与延伸思考
这篇工作把“先思考后发言”理念系统化，为对话系统的可解释性和常识性提供了新路径。随后出现的研究（如 CoT‑style 对话、显式推理链的对话模型）都在不同程度上借鉴了显式中间产物的思路。未来可以探索：  
- 用更大规模的自动化常识标注方法降低人工成本。  
- 将结构化常识（如知识图谱三元组）与自然语言常识结合，提升常识的覆盖度和可操作性。  
- 在多模态对话（图像+语言）中让模型先生成视觉常识，再生成语言回复，进一步拓展“思考”层次。  

### 一句话记住它
先让模型把隐含常识写出来，再用这段常识生成回复，既提升了对话质量，又让模型的思考过程可见。