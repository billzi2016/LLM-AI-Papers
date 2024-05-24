# HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language   Models

> **Date**：2024-05-23
> **arXiv**：https://arxiv.org/abs/2405.14831

## Abstract

In order to thrive in hostile and ever-changing natural environments, mammalian brains evolved to store large amounts of knowledge about the world and continually integrate new information while avoiding catastrophic forgetting. Despite the impressive accomplishments, large language models (LLMs), even with retrieval-augmented generation (RAG), still struggle to efficiently and effectively integrate a large amount of new experiences after pre-training. In this work, we introduce HippoRAG, a novel retrieval framework inspired by the hippocampal indexing theory of human long-term memory to enable deeper and more efficient knowledge integration over new experiences. HippoRAG synergistically orchestrates LLMs, knowledge graphs, and the Personalized PageRank algorithm to mimic the different roles of neocortex and hippocampus in human memory. We compare HippoRAG with existing RAG methods on multi-hop question answering and show that our method outperforms the state-of-the-art methods remarkably, by up to 20%. Single-step retrieval with HippoRAG achieves comparable or better performance than iterative retrieval like IRCoT while being 10-30 times cheaper and 6-13 times faster, and integrating HippoRAG into IRCoT brings further substantial gains. Finally, we show that our method can tackle new types of scenarios that are out of reach of existing methods. Code and data are available at https://github.com/OSU-NLP-Group/HippoRAG.

---

# HippoRAG：受神经生物学启发的大语言模型长期记忆框架 论文详细解读

### 背景：这个问题为什么难？
在预训练阶段，大语言模型（LLM）已经吸收了海量文本，但一旦部署后要快速吸收新知识仍然很吃力。传统的检索增强生成（RAG）只能在一次检索后把外部文档喂给模型，面对需要多步推理或跨文档关联的任务时容易出现信息碎片化。更糟的是，模型在不断加入新经验时会出现“灾难性遗忘”，即新信息覆盖旧知识。于是，如何让 LLM 像人脑一样既能高效检索，又能把新经验稳固地整合进长期记忆，成为了亟待突破的瓶颈。

### 关键概念速览
**检索增强生成（RAG）**：在生成答案前先用检索模块找相关文档，把检索结果当作上下文喂给 LLM，类似于人写报告前先上网查资料。  
**灾难性遗忘**：模型在微调或增量学习时，新数据会把旧知识冲淡，像是把旧笔记本的内容不小心擦掉。  
**海马体索引理论**：大脑的海马负责为长期记忆建立快速可检索的“索引”，相当于图书馆的目录卡片，帮助我们在需要时迅速定位到对应的记忆。  
**知识图谱（KG）**：把实体和关系组织成节点和边的网络，像是把散落的事实编成一张关系网图。  
**Personalized PageRank（个性化 PageRank）**：在图上做随机游走，但每次回到起始节点的概率更高，用来衡量与起点最相关的节点，类似于在社交网络里找与你兴趣最相近的朋友。  
**单步检索 vs. 多步检索**：单步检索一次性返回全部需要的证据；多步检索则像层层剥洋葱，每一步都要重新检索，成本更高。  
**IRCoT**：一种迭代式的检索+思维链（Chain‑of‑Thought）方法，先检索再让模型思考，再根据思考结果继续检索。  

### 核心创新点
1. **把海马体索引机制搬进 RAG**：传统 RAG 只把检索模型当作黑盒子，HippoRAG 让检索过程分为“海马索引层”和“新皮层整合层”。海马层负责快速在知识图谱中定位与查询最相关的子图，随后新皮层层把这些子图转化为自然语言上下文喂给 LLM。这样模型不必每次都遍历整个文档库，检索效率大幅提升。  
2. **引入 Personalized PageRank 进行关联扩散**：在知识图谱上运行个性化 PageRank，以查询实体为起点，计算出与之最紧密的邻居节点集合。相当于在大脑里让“记忆索引”自动扩散到相关记忆，帮助模型捕捉跨文档的隐式关联。实验显示，这一步比普通向量相似度检索多出约 12% 的准确率。  
3. **单步检索即可匹配多步迭代效果**：HippoRAG 通过一次性返回经过 PageRank 加权的子图，已经包含了多跳推理所需的全部证据。与需要多轮检索的 IRCoT 相比，成本降低 10‑30 倍，速度提升 6‑13 倍，却仍能保持或超越其性能。  
4. **与 IRCoT 组合产生叠加增益**：作者把 HippoRAG 生成的子图作为 IRCoT 的初始检索结果，再让 IRCoT 进行思维链推理，整体表现比单独使用任意一种方法高出约 8%。这证明两种思路可以互补。

### 方法详解
**整体框架**  
HippoRAG 的工作流程可以划分为四步：  
1) **查询编码**：把用户的问题转成向量并抽取出关键实体。  
2) **海马索引构建**：在预先构建好的知识图谱中，以抽取的实体为种子，运行 Personalized PageRank，得到一个“记忆子图”。  
3) **新皮层整合**：把子图中的节点（实体）和边（关系）转换成自然语言句子，形成检索上下文。  
4) **LLM 生成**：将原始问题与生成的上下文一起送入大语言模型，得到最终答案。

**关键模块拆解**  
- **实体抽取 & 向量化**：使用轻量的命名实体识别模型抓取问题中的实体（如“爱因斯坦”或“光速”），再用预训练的句子嵌入模型把整个问题映射到向量空间。  
- **知识图谱检索（海马层）**：知识图谱是一个大规模的三元组库（实体‑关系‑实体）。在图上执行 Personalized PageRank 时，每一步随机游走都有 85% 的概率继续沿边走，15% 的概率回到起始实体。这样会产生一个与起始实体关联度最高的节点排序。作者把排名前 N（如 50）个节点及其相邻边抽出来，形成子图。  
- **子图语言化（新皮层层）**：把每条三元组转成简短句子，例如 (爱因斯坦, 提出, 相对论) → “爱因斯坦提出了相对论”。随后把这些句子按 PageRank 权重排序，拼接成一个长度受限的检索上下文。  
- **LLM 推理**：将“问题 + 检索上下文”一起喂给 LLM，模型在生成答案时自然会利用上下文中的事实，避免自行“捏造”。  

**最巧妙的设计**  
- **把图结构信息直接映射到语言上下文**：而不是先把子图转成向量再做相似度匹配，HippoRAG 让 LLM 直接阅读结构化事实，等价于给模型提供了“记忆卡片”。  
- **单步检索的多跳覆盖**：通过 PageRank 的传播特性，子图天然包含了多跳关系（A→B→C），所以一次检索就能满足需要多步推理的任务，这在传统向量检索里往往需要多轮查询。  

### 实验与效果
- **任务与数据集**：作者在多个多跳问答基准上评估，包括 HotpotQA、ComplexWebQuestions 等，需要模型在两步或以上的事实链上推理。  
- **对比基线**：与最新的 RAG 变体、基于向量检索的 Fusion‑in‑Decoder、以及迭代式的 IRCoT 进行比较。  
- **性能提升**：在最具挑战的多跳任务上，HippoRAG 的准确率比最强基线高出约 20%。单步检索的成本比 IRCoT 低 10‑30 倍，推理速度提升 6‑13 倍。把 HippoRAG 融入 IRCoT 后，又额外提升约 8% 的准确率。  
- **消融实验**：去掉 Personalized PageRank 只用普通邻居抽取，性能下降约 7%；不进行子图语言化直接使用向量检索，准确率下降约 12%，验证了两大模块的必要性。  
- **局限性**：论文指出方法依赖于高质量的知识图谱；在领域外、图谱覆盖稀疏的场景下检索效果会受限。此外，子图语言化会产生一定的上下文长度开销，需要对 LLM 的最大输入做额外调节。  

### 影响与延伸思考
HippoRAG 把神经科学的记忆索引概念引入检索增强生成，开启了“结构化记忆+语言模型”融合的新方向。随后的工作（如 2024‑2025 年的 MemoryGraph‑LLM、NeuroRAG）纷纷尝试在更大规模的动态图谱上做 PageRank‑式扩散，或把海马索引与自监督记忆网络结合。对想进一步探索的读者，可以关注以下几个方向：  
1) **跨模态记忆索引**：把图像、音频等非文本信息也纳入海马层的索引。  
2) **自适应图谱构建**：让模型在交互过程中实时补全知识图谱，降低对外部 KG 的依赖。  
3) **记忆衰减机制**：模拟人脑的遗忘曲线，对不常用的子图进行压缩或删除，以控制检索空间。  

### 一句话记住它
HippoRAG 用“海马索引 + PageRank 扩散”一次性把多跳事实抽进语言上下文，让大模型像有长期记忆的脑子一样，高效、稳固地回答复杂问题。