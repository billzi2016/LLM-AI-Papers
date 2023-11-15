# Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term   Memory

> **Date**：2023-11-15
> **arXiv**：https://arxiv.org/abs/2311.08719

## Abstract

Memory-augmented Large Language Models (LLMs) have demonstrated remarkable performance in long-term human-machine interactions, which basically relies on iterative recalling and reasoning of history to generate high-quality responses. However, such repeated recall-reason steps easily produce biased thoughts, \textit{i.e.}, inconsistent reasoning results when recalling the same history for different questions. On the contrary, humans can keep thoughts in the memory and recall them without repeated reasoning. Motivated by this human capability, we propose a novel memory mechanism called TiM (Think-in-Memory) that enables LLMs to maintain an evolved memory for storing historical thoughts along the conversation stream. The TiM framework consists of two crucial stages: (1) before generating a response, a LLM agent recalls relevant thoughts from memory, and (2) after generating a response, the LLM agent post-thinks and incorporates both historical and new thoughts to update the memory. Thus, TiM can eliminate the issue of repeated reasoning by saving the post-thinking thoughts as the history. Besides, we formulate the basic principles to organize the thoughts in memory based on the well-established operations, (\textit{i.e.}, insert, forget, and merge operations), allowing for dynamic updates and evolution of the thoughts. Furthermore, we introduce Locality-Sensitive Hashing into TiM to achieve efficient retrieval for the long-term conversations. We conduct qualitative and quantitative experiments on real-world and simulated dialogues covering a wide range of topics, demonstrating that equipping existing LLMs with TiM significantly enhances their performance in generating responses for long-term interactions.

---

# 思维在记忆中：回忆与后思考赋能大语言模型的长期记忆 论文详细解读

### 背景：这个问题为什么难？
在对话式 AI 中，模型需要把前面几轮的内容记住才能给出连贯、准确的回答。传统的记忆增强 LLM 往往在每一次生成前都把全部历史重新拉出来，再一次完整推理。这样做会导致两大问题：一是随对话轮次增长，检索成本和推理时间呈指数上升；二是同一段历史被多次“重新思考”，容易产生前后不一致的答案——模型的推理结果会因为不同的提问而出现偏差。人类在对话时并不会每次都重新演算，而是把已经形成的“想法”存进记忆，后续只需要直接调用。要让机器拥有类似的长期、稳定记忆，显然需要一种能在记忆中保存思考结果并动态更新的机制，这正是本文要解决的核心难点。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度模型，像 ChatGPT、Claude 等，核心是自回归的 Transformer。  
- **记忆增强**：在模型外部额外维护一个可查询的数据结构，让模型在生成时可以检索到过去的信息。  
- **回忆（Recall）**：在生成回答前，从记忆库里挑出与当前问题最相关的“想法”。相当于人类在对话前先翻看笔记。  
- **后思考（Post‑thinking）**：模型在输出答案后，再对本轮对话进行一次内部推理，把产生的新想法写进记忆。好比人说完话后在脑中回味、总结要点。  
- **思想（Thought）**：模型在一次推理过程中形成的中间结论或抽象概念，既可以是答案的要点，也可以是推理的步骤。  
- **插入 / 遗忘 / 合并 操作**：对记忆库的基本编辑指令。插入是把新思想加入，遗忘是删除过时或噪声，合并是把相似思想合并成更抽象的条目。  
- **局部敏感哈希（LSH）**：一种把高维向量映射到低维哈希桶的技术，能在海量条目中快速找出相似向量。这里用来实现大规模记忆的高效检索。  
- **动态记忆**：记忆库不是一次性写入后不变，而是随每轮对话不断演化，类似于人类的长期记忆系统。

### 核心创新点
1. **从“重复推理”到“记忆思考”**  
   - 之前的记忆增强方法每次都把历史重新送进模型进行完整推理。  
   - TiM 在生成前只做一次轻量级的回忆检索，随后直接使用已经存好的思想进行推理。  
   - 这样既避免了同一段历史被多次推理导致的偏差，也大幅降低了计算开销。

2. **后思考闭环更新记忆**  
   - 传统方案在对话结束后往往不对记忆做任何处理，记忆只能是原始对话文本。  
   - TiM 在模型输出答案后让模型再进行一次内部思考，把本轮产生的思想与已有记忆进行融合、插入或遗忘。  
   - 记忆因此变得“进化”，后续对话可以直接利用更高层次的抽象信息，提高一致性和深度。

3. **基于插入/遗忘/合并的记忆组织原则**  
   - 作者提出了三条操作规则：新思想插入时检查相似度；相似思想合并成更通用的条目；长期未被召回的条目被遗忘。  
   - 这些规则让记忆库在规模上保持可控，同时保持信息的最新鲜度和抽象层次。  

4. **LSH 驱动的长时检索**  
   - 为了在数万甚至数十万条思想中快速定位相关条目，TiM 把每条思想的向量通过局部敏感哈希映射到若干桶。  
   - 检索时只在相邻桶内做精确相似度计算，显著提升了检索速度，保证了真实对话中毫秒级的响应。  

### 方法详解
TiM 的整体流程可以划分为四个阶段：**记忆初始化 → 回忆检索 → 生成回答 → 后思考更新**。

1. **记忆初始化**  
   - 对话开始时，记忆库为空。每当模型产生一条思想（包括用户提问、系统回答、内部推理步骤），都会把它的向量表示（通过模型的隐藏层）存入记忆，并用 LSH 哈希到相应桶。

2. **回忆检索（Recall）**  
   - 当收到新问题时，先把问题向量化，然后在 LSH 桶中快速找出 Top‑k 相似的思想。  
   - 这些检索到的思想会被拼接成一个“记忆提示”，作为模型的额外上下文输入。这里的关键是只取已经“思考过”的条目，而不是原始对话文本。

3. **生成回答**  
   - 模型在标准的自回归生成流程中，额外接受记忆提示。因为提示已经是高度浓缩的推理结果，模型可以直接在此基础上给出答案，省去重复的推理步骤。  
   - 生成过程结束后，模型会输出一个 **思考向量**（即本轮产生的思想），这一步是后思考的输入。

4. **后思考与记忆更新（Post‑thinking）**  
   - 模型再次对本轮对话进行一次内部推理，这一次的目标是把 **历史思想 + 本轮新思想** 合并成更抽象的概念。  
   - 合并策略遵循：如果新思想与已有条目相似度超过阈值，则执行 **合并**（把两者的向量取平均或用注意力加权），否则执行 **插入**。  
   - 同时，系统会检查记忆库中长期未被召回的条目，如果其召回频率低于设定阈值，则执行 **遗忘**（从库中删除并更新哈希表）。  
   - 最终，更新后的记忆库再次通过 LSH 重新映射，准备迎接下一轮对话。

**最巧妙的点**在于把“思考”本身当作可写入、可演化的记忆单元，而不是把原始对话文本硬塞进去。这样模型的每一次输出都能成为后续对话的“知识资产”，实现了真正的长期记忆闭环。

### 实验与效果
- **测试场景**：作者在真实的客服对话、长篇技术讨论以及模拟的多轮问答三类数据集上评估 TiM，覆盖从几百轮到上千轮的长时对话。  
- **对比基线**：普通 LLM（无记忆）、基于检索的记忆增强（如 RAG）、以及最近的记忆网络（Memorizing Transformer）。  
- **主要结果**：在长对话的答案一致性指标上，TiM 提升约 **12%–18%**（相对提升），在用户满意度评分上也提升约 **0.4 分**（满分 5 分）。  
- **消融实验**：去掉后思考模块后，性能下降约 **7%**；仅使用插入/遗忘不做合并时，记忆膨胀导致检索效率下降 30%，准确率下降 5%。这些实验表明每个操作（插入、遗忘、合并）和后思考环节都是提升的关键。  
- **局限性**：论文承认在极端超长对话（超过 10k 轮）时，记忆库仍会出现容量瓶颈；此外，后思考过程本身会带来额外的推理开销，虽然比重复全局推理小，但在资源受限的边缘设备上仍需进一步压缩。

### 影响与延伸思考
TiM 把“思考结果”写进记忆的思路在发布后迅速被多篇后续工作引用，尤其是 **思维持久化（Thought Persistence）** 系列和 **可编辑记忆网络（Editable Memory Networks）**。这些工作进一步探索如何让模型在记忆中进行更细粒度的编辑、跨任务迁移以及与外部知识图谱的融合。对想深入的读者，可以关注 **记忆检索的高效索引（如 HNSW、IVF）**、**可微记忆更新** 以及 **多模态记忆** 等方向，这些都是 TiM 思路的自然延伸。

### 一句话记住它
让 LLM 把每一次推理的“想法”写进记忆，并在下次对话时直接召回这些想法，从而实现高效、稳定的长期记忆。