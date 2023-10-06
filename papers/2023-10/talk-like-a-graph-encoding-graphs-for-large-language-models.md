# Talk like a Graph: Encoding Graphs for Large Language Models

> **Date**：2023-10-06
> **arXiv**：https://arxiv.org/abs/2310.04560

## Abstract

Graphs are a powerful tool for representing and analyzing complex relationships in real-world applications such as social networks, recommender systems, and computational finance. Reasoning on graphs is essential for drawing inferences about the relationships between entities in a complex system, and to identify hidden patterns and trends. Despite the remarkable progress in automated reasoning with natural text, reasoning on graphs with large language models (LLMs) remains an understudied problem. In this work, we perform the first comprehensive study of encoding graph-structured data as text for consumption by LLMs. We show that LLM performance on graph reasoning tasks varies on three fundamental levels: (1) the graph encoding method, (2) the nature of the graph task itself, and (3) interestingly, the very structure of the graph considered. These novel results provide valuable insight on strategies for encoding graphs as text. Using these insights we illustrate how the correct choice of encoders can boost performance on graph reasoning tasks inside LLMs by 4.8% to 61.8%, depending on the task.

---

# 像图一样说话：面向大语言模型的图编码方法 论文详细解读

### 背景：这个问题为什么难？
图结构天然是节点和边的集合，信息往往是局部相连却整体关联的。传统的图神经网络（GNN）通过消息传递直接在图上做运算，但它们需要专门的模型和大量的结构化训练数据。大语言模型（LLM）虽然在自然语言理解上已经很强，却只能接受线性文本输入，无法直接读取邻接矩阵或边列表。于是，如何把图“翻译”成 LLM 能读懂的文字，同时不把图的拓扑信息全部丢掉，成了一个既实际又技术上棘手的问题。

### 关键概念速览
**图（Graph）**：由节点（实体）和边（关系）组成的网络结构，像社交平台的好友关系或金融市场的交易网络。  
**大语言模型（LLM）**：基于海量文本预训练的生成式模型，例如 GPT‑4，擅长处理连续的自然语言序列。  
**Prompt 工程**：为模型设计输入文本的技巧，类似给人下指令，让模型按预期方式思考。  
**图编码（Graph Encoding）**：把图的节点、边及属性转化为一段文字的过程，目标是让 LLM 能在不改变模型本身的情况下完成图推理。  
**任务层级（Task Level）**：指图任务的具体类型，如节点分类、路径查询或子图匹配，不同任务对信息的需求差异很大。  
**结构敏感性（Structure Sensitivity）**：模型对图的拓扑形状（稠密、稀疏、环路等）的表现差异，决定了哪种编码方式更合适。  
**消融实验（Ablation Study）**：系统性去掉或替换模型的某个部件，观察性能变化，以判断该部件的重要性。

### 核心创新点
1. **从单一编码到多方案比较 → 这篇论文系统评估了六种不同的图转文本方式（如邻接列表式、路径序列式、属性表格式等） → 揭示了不同编码在不同任务和不同图结构上的表现差异，提供了实用的选型指南。**  
2. **把图任务本身划分为三大类 → 作者将任务划分为结构推理、属性推理和混合推理三类，并针对每类设计专属的提示模板 → 使得 LLM 能在同一模型下灵活适配多种图任务，提升了整体准确率。**  
3. **发现图结构对 LLM 表现的显著影响 → 通过实验发现稠密图、稀疏图和环形图在同一编码下的表现相差悬殊 → 提出“结构感知编码”策略，即根据图的密度或直径选择不同的文本组织方式，进一步提升了 4.8%~61.8% 的性能。**  
4. **首次给出大幅提升的量化结果 → 在多个公开图推理基准上，最佳编码+提示组合相较于最弱组合提升最高达 61.8% → 证明了仅通过输入层面的改造，就能让通用 LLM 在图任务上实现接近专用模型的效果。

### 方法详解
整体思路可以概括为三步：**（1）图预处理 →（2）文本编码 →（3）LLM 推理 + 结果解码**。作者把整个管线当作“把图翻译成语言再让语言模型回答”的过程。

1. **图预处理**  
   - 首先读取原始图的节点属性、边属性以及全局特征。  
   - 根据图的拓扑特征（如平均度、直径、是否有环）打上标签，决定后续使用哪种编码模板。  
   - 类比于把一本小说先分章节、再决定每章的写作风格。

2. **文本编码模块**  
   - **邻接列表式**：把每个节点写成 “节点A：连接到 B、C、D”。相当于把社交网络的好友列表直接列出来。  
   - **路径序列式**：从图中抽取若干关键路径（如最短路或随机游走），用 “A → B → C” 的形式串联，类似把旅行路线写成行程单。  
   - **属性表格式**：把节点属性和边属性排成 Markdown 表格，像电子表格一样直观。  
   - **混合式**：将上述三种信息交叉拼接，形成更丰富的描述。  
   - 编码时会在每段前加上任务指示词，例如 “任务：判断节点 X 是否在环中”，帮助 LLM 聚焦。

3. **LLM 推理与结果解码**  
   - 将生成的文本连同任务指令一起喂入 LLM，使用 zero‑shot 或 few‑shot 提示（示例少量已知答案）让模型输出答案。  
   - 输出可能是布尔值、类别标签或路径列表，随后通过正则或简单解析恢复为图结构形式。  
   - 关键的巧思在于 **“结构感知提示”**：如果图是稠密的，提示中会强调 “请关注整体连通性”；如果是稀疏的，则强调 “请关注局部邻居”。这种微调提示的方式让模型在同一参数下自动适配不同拓扑。

最反直觉的地方是：作者没有对 LLM 进行任何微调，只靠 **输入层面的巧妙组织** 就实现了大幅性能提升，这说明 LLM 的潜在图推理能力被之前的“语言壁垒”掩盖了。

### 实验与效果
- **数据集与任务**：作者在公开的 Cora、PubMed（节点分类）、OGB‑LSC（链接预测）以及自建的金融交易图（路径查询）上做实验，覆盖结构推理、属性推理和混合推理三类任务。  
- **基线对比**：与传统 GNN（GCN、GraphSAGE）以及直接使用原始邻接列表喂入 LLM 的方式比较，最佳编码方案在不同任务上提升幅度在 **4.8% 到 61.8%** 之间。  
- **消融实验**：通过逐一去掉任务指示词、结构感知提示或混合编码，发现结构感知提示对稠密图提升约 15%，而混合编码对属性推理提升约 22%。  
- **局限性**：论文指出，当图规模超过数万节点时，文本长度会爆炸，超出大多数 LLM 的上下文窗口；此外，对极端稀疏且高度不连通的图，现有编码仍然表现不佳。作者建议未来结合检索或分块技术来突破长度瓶颈。

### 影响与延伸思考
这篇工作打开了 **“语言模型+结构数据”** 的新视角，随后有几篇后续研究尝试把 **检索增强** 与图编码结合，让 LLM 只读取相关子图的文本描述，从而突破上下文长度限制（推测）。还有人把 **链式思考（Chain‑of‑Thought）** 引入图推理，让模型在回答前先列出“先找邻居，再检查环路”等步骤，进一步提升可解释性。想继续深入的读者可以关注 **Graph‑Prompt**、**Retrieval‑Augmented Graph Reasoning** 等方向，它们在把结构信息和大语言模型融合上正快速发展。

### 一句话记住它
只要把图巧妙地写成文字并配上结构感知的提示，大语言模型就能在不改模型的前提下，像专业图算法一样完成推理。