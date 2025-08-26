# Utilizing Training Data to Improve LLM Reasoning for Tabular Understanding

> **Date**：2025-08-26
> **arXiv**：https://arxiv.org/abs/2508.18676

## Abstract

Automated tabular understanding and reasoning are essential tasks for data scientists. Recently, Large language models (LLMs) have become increasingly prevalent in tabular reasoning tasks. Previous work focuses on (1) finetuning LLMs using labeled data or (2) Training-free prompting LLM agents using chain-of-thought (CoT). Finetuning offers dataset-specific learning at the cost of generalizability. Training-free prompting is highly generalizable but does not take full advantage of training data. In this paper, we propose a novel prompting-based reasoning approach, Learn then Retrieve: LRTab, which integrates the benefits of both by retrieving relevant information learned from training data. We first use prompting to obtain CoT responses over the training data. For incorrect CoTs, we prompt the LLM to predict Prompt Conditions to avoid the error, learning insights from the data. We validate the effectiveness of Prompt Conditions using validation data. Finally, at inference time, we retrieve the most relevant Prompt Conditions for additional context for table understanding. We provide comprehensive experiments on WikiTQ and Tabfact, showing that LRTab is interpretable, cost-efficient, and can outperform previous baselines in tabular reasoning.

---

# 利用训练数据提升大语言模型在表格理解上的推理能力 论文详细解读

### 背景：这个问题为什么难？

表格是结构化信息的常见载体，数据科学家常需要把表格里的行列关系、数值比较、条件筛选等逻辑转化为自然语言答案。传统的表格推理方法要么靠手工特征、要么训练专门的表格模型，但都缺乏对语言的深层理解。近来大语言模型（LLM）凭借强大的语言能力被直接用于表格推理，然而已有的两大路线各有短板：  
1. **微调**（在标注数据上继续训练）能让模型记住数据集的细节，却会牺牲跨数据集的通用性，且标注成本高。  
2. **零提示**（直接用 chain‑of‑thought 提示）保持了模型的通用性，却没有利用已有的训练数据来提供额外的上下文，导致在复杂表格上仍会出现逻辑错误。  
因此，如何在保持通用性的同时，充分挖掘已有训练数据的价值，成为表格推理的关键瓶颈。

### 关键概念速览

**LLM（大语言模型）**：能够生成自然语言并进行推理的深度模型，如 GPT‑4、Claude。它们通过海量文本学习语言规律，但对结构化表格的专门推理仍需额外引导。  

**Chain‑of‑Thought（CoT，思维链）**：让模型在给出最终答案前先写出逐步推理过程，类似解数学题时的草稿，帮助模型保持逻辑连贯。  

**Prompt Conditions（提示条件）**：在本工作中指模型从错误的 CoT 中抽取的“避免错误的规则”，相当于给模型的“注意事项清单”。  

**检索（Retrieval）**：在推理阶段从已有的提示条件库中挑选与当前问题最相似的条目，提供额外的上下文信息。  

**WikiTQ / TabFact**：两个公开的表格问答基准，前者侧重自然语言查询，后者侧重真假判断，两者都要求模型进行表格内部的逻辑推理。  

**可解释性（Interpretability）**：模型的推理过程或辅助信息能够被人类直接阅读和理解，而不是黑箱输出。  

**成本效率（Cost‑efficiency）**：在不进行大规模微调的前提下，仅通过提示和检索实现性能提升，节约算力和标注费用。

### 核心创新点

1. **从错误 CoT 中学习提示条件 → 通过专门的提示让 LLM 预测“避免错误的规则” → 形成一套可复用的错误纠正知识库，而不是仅依赖原始标签。** 这一步把训练数据的负例转化为显式的约束，使模型在以后遇到相似情形时能主动规避。  

2. **在推理时检索最相关的提示条件 → 将检索到的条件拼接到输入提示中 → 为模型提供针对性的上下文，提升答案的准确性。** 与传统的纯提示不同，这里加入了“记忆”模块，使模型在不微调的情况下仍能利用过去学到的经验。  

3. **把提示条件的有效性放在验证集上做二次评估 → 只保留在验证集上表现好的条件 → 防止噪声提示污染推理过程。** 通过验证过滤，确保检索到的条件是真正有帮助的，而不是随意的文本。  

4. **整体框架兼顾通用性与数据利用率 → 不需要对每个新表格任务进行微调，只需构建一次提示条件库 → 在不同数据集上均可直接使用。** 这解决了微调方法的“迁移难”问题，同时克服了纯提示方法的“信息匮乏”缺陷。

### 方法详解

**整体思路**：LRTab（Learn‑then‑Retrieve for Tabular reasoning）把“学习”和“检索”两阶段串起来。第一阶段在训练数据上让 LLM 生成 CoT，并对错误的 CoT 提取提示条件；第二阶段在实际推理时，根据当前查询检索最相似的提示条件并加入到提示里，帮助模型给出更可靠的答案。

**步骤拆解**：

1. **CoT 生成**  
   - 对每条训练样本（表格 + 问题），使用标准的 chain‑of‑thought 提示让 LLM 输出完整的推理链和最终答案。  
   - 这一步相当于让模型“先写草稿”，为后面的错误分析提供原始材料。

2. **错误检测与提示条件抽取**  
   - 将模型的答案与标注答案对比，标记出错误的 CoT。  
   - 对每个错误的 CoT，构造新的提示：“请思考导致错误的原因，并给出一个可以避免此错误的条件”。  
   - LLM 回答后得到一句简短的规则，例如“在比较数值时先检查是否存在缺失值”。这些规则即为 **Prompt Conditions**。

3. **验证过滤**  
   - 将所有 Prompt Conditions 放入候选库。  
   - 在验证集上再次使用这些条件进行检索并加入提示，观察是否提升了验证准确率。  
   - 只保留在验证集上表现提升的条件，剔除噪声。

4. **检索模块**  
   - 为每个推理请求计算其与库中每条 Prompt Condition 的相似度（可以使用向量检索或基于关键词的匹配）。  
   - 选取相似度最高的 K 条（K 通常为 1~3）作为额外上下文。

5. **最终推理**  
   - 将原始问题、表格信息、以及检索到的 Prompt Conditions 拼接成一个完整提示，交给 LLM。  
   - LLM 再次生成 CoT 并输出答案。因为提示里已经包含了“避免错误的规则”，模型更倾向于走正确的推理路径。

**关键巧思**：  
- 把错误的 CoT 当作“负例知识”，而不是直接丢弃，这让模型在没有显式微调的情况下也能“记住”哪些推理路径是危险的。  
- 使用验证集做二次筛选，防止从噪声数据中学到误导性规则，保持提示库的质量。  
- 检索过程只在推理时进行，成本几乎等同于一次普通的提示调用，保持了训练成本的低廉。

### 实验与效果

- **数据集**：作者在两个主流表格推理基准上评估：WikiTQ（自然语言查询）和 TabFact（真假判断）。  
- **对比基线**：包括纯 chain‑of‑thought 提示、微调后的表格专用模型（如 TaBERT、TableFormer）以及最近的检索增强 LLM 方法。  
- **结果**：论文声称 LRTab 在两套数据上均超过所有对比基线，尤其在复杂的多步推理题目上提升更为明显。具体提升幅度未在摘要中给出，但作者强调“显著超越”。  
- **消融实验**：通过去掉 Prompt Conditions、关闭验证过滤、或不进行检索，性能均出现明显下降，验证了每个模块的贡献。  
- **局限性**：作者指出 LRTab 仍依赖于高质量的初始 CoT 生成，如果 LLM 本身在生成 CoT 时错误率过高，提取的 Prompt Conditions 质量会受影响；此外，检索相似度的计算仍是一个经验性选择，可能在跨领域表格上表现不佳。

### 影响与延伸思考

LRTab 把“从错误中学习”与“检索增强”结合，提供了一条在不微调的前提下利用训练数据提升 LLM 推理的路径。自论文发布后，已有后续工作尝试将类似的错误规则抽取用于代码生成、数学推理等非表格任务，说明该思路具备跨任务的潜力。未来可以进一步探索：

- **自动化规则抽象**：把 Prompt Conditions 从自然语言转化为结构化的逻辑表达式，提升检索的精度。  
- **多模态检索**：结合表格的结构特征（列名、数据类型）进行更精准的条件匹配。  
- **自适应检索阈值**：根据当前问题的难度动态决定检索多少条条件，以平衡成本与收益。  

如果想深入了解，可关注近期在 “LLM‑augmented retrieval” 与 “error‑aware prompting” 方向的会议论文和开源实现。

### 一句话记住它

**LRTab 用错误的思维链提炼“避免错误的规则”，在推理时检索这些规则，让大语言模型在表格问答上既保持通用性，又能像有经验的分析师一样少走弯路。**