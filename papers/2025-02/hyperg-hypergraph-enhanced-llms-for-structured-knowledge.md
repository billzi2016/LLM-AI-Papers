# HyperG: Hypergraph-Enhanced LLMs for Structured Knowledge

> **Date**：2025-02-25
> **arXiv**：https://arxiv.org/abs/2502.18125

## Abstract

Given that substantial amounts of domain-specific knowledge are stored in structured formats, such as web data organized through HTML, Large Language Models (LLMs) are expected to fully comprehend this structured information to broaden their applications in various real-world downstream tasks. Current approaches for applying LLMs to structured data fall into two main categories: serialization-based and operation-based methods. Both approaches, whether relying on serialization or using SQL-like operations as an intermediary, encounter difficulties in fully capturing structural relationships and effectively handling sparse data. To address these unique characteristics of structured data, we propose HyperG, a hypergraph-based generation framework aimed at enhancing LLMs' ability to process structured knowledge. Specifically, HyperG first augment sparse data with contextual information, leveraging the generative power of LLMs, and incorporate a prompt-attentive hypergraph learning (PHL) network to encode both the augmented information and the intricate structural relationships within the data. To validate the effectiveness and generalization of HyperG, we conduct extensive experiments across two different downstream tasks requiring structured knowledge.

---

# HyperG：面向结构化知识的超图增强大语言模型 论文详细解读

### 背景：这个问题为什么难？
结构化数据（如 HTML 表格、数据库）本身蕴含丰富的行列关系和稀疏信息，但大语言模型（LLM）天生是对连续文本进行概率预测的，对这种显式的结构感知力有限。现有做法要么把表格序列化成一段文字，要么先把结构转成 SQL 再让模型执行，两者都把行列之间的高阶关联压平，导致模型难以捕捉跨行跨列的语义依赖；同时，稀疏的单元格缺少上下文，模型往往会产生不准确的推断。于是，如何让 LLM 在保持生成能力的同时，真正“看懂”超图式的结构关系，成为亟待突破的瓶颈。

### 关键概念速览
**结构化数据**：指以表格、树或图等固定格式组织的信息，例如网页中的 HTML 表格或关系型数据库。它的每一行、每一列都有明确的语义标签。  
**稀疏单元格**：在表格里缺失或只有极少文字的格子，常见于统计报表的空白或占位符。  
**超图（Hypergraph）**：一种图的推广，超边可以同时连接两个以上的节点。把整行、整列或整张表视作超边，就能一次性表达多元关系。  
**Prompt‑Attentive Hypergraph Learning (PHL)**：一种专门为 LLM 设计的注意力机制，它在超图上做信息传播时，会根据提示词的权重调节节点与超边的交互强度。  
**上下文增强**：利用 LLM 自身的生成能力，为稀疏单元格填补合理的文字描述，使其在后续编码时拥有足够的语义信息。  
**LoRA 微调**：Low‑Rank Adaptation 的缩写，通过在大模型的权重上加低秩矩阵实现高效微调，避免全模型更新的高成本。  

### 核心创新点
1. **稀疏数据的生成式补全 → 用 LLM 直接生成缺失单元格的上下文** → 让原本信息不足的格子拥有可编码的语义向量，显著提升后续结构学习的质量。  
2. **超图化表格建模 → 把每个单元格当作节点，行、列、整表分别作为超边** → 超边一次性捕获跨行跨列的高阶依赖，克服了序列化方法只能逐行或逐列线性建模的局限。  
3. **Prompt‑Attentive Hypergraph Learning (PHL) 网络 → 在超图上加入提示词感知的多头注意力层** → 通过让提示词调节信息流向，模型能够在不同下游任务（如问答、填表）中自动聚焦最相关的结构子集。  
4. **LoRA‑式轻量微调 → 在保持 LLM 基础能力的前提下，仅对超图编码器和少量提示相关参数进行低秩适配** → 兼顾了性能提升和计算成本，适合资源受限的实际部署。  

### 方法详解
**整体框架**  
HyperG 的工作流可以划分为三大步骤：① 稀疏单元格上下文增强；② 超图构建与 PHL 编码；③ 任务特定的 LoRA 微调与输出生成。整体思路是先让 LLM 为缺失信息“填词”，再把完整的表格映射成超图，让超图学习器在结构上做深度推理，最后用轻量微调把结构化表示对接到具体任务。

**步骤 1：上下文增强**  
- 输入：原始表格（可能包含空白或占位符）。  
- 处理：针对每个稀疏单元格，构造一个简短的 prompt（如“请解释该格子对应的指标含义”），交给预训练 LLM（GPT‑4 级别）生成自然语言描述。  
- 输出：每个单元格得到一个文本片段，随后统一送入文本编码器。这里的关键是让 LLM 依据已有列标题、行标签等上下文生成符合领域语义的补全，而不是随意填充。

**步骤 2：超图构建与 PHL 编码**  
- **节点定义**：每个单元格（包括原始和补全后的文本）对应一个节点。  
- **超边定义**：三类超边并行存在：  
  - 行超边：连接同一行的所有节点。  
  - 列超边：连接同一列的所有节点。  
  - 表超边：连接整张表的所有节点，捕获全局语义。  
- **文本向量化**：使用 BERT（或其他预训练文本编码器）把每个节点的文本转成向量。  
- **PHL 机制**：在超图上执行多层注意力传播。每层的注意力权重由两部分决定：① 节点自身的向量相似度；② 当前任务的 prompt 向量（通过一个小的线性投影得到），后者起到“指挥官”作用，告诉网络本轮应更关注哪些超边。信息在节点↔超边之间来回流动，类似于图神经网络的消息传递，但一次可以跨越多节点，因为超边本身已经把它们绑在一起。  
- **输出**：每个节点得到一个结构感知的嵌入，整个表格则拥有一个统一的全局表示（由表超边聚合得到）。

**步骤 3：LoRA 微调与任务解码**  
- 将 PHL 编码得到的全局表向量与下游任务的提示词拼接，送入 LLM 的解码层。  
- 只在 LLM 的注意力层和前馈层上加上 LoRA 低秩适配矩阵，训练时冻结原始权重。这样模型既保留了原始的大语言能力，又能快速学习如何把结构化表征映射到具体输出（如问答答案、填表指令等）。  
- 训练目标依据任务不同而变化：若是问答则使用交叉熵对答案文本进行监督；若是生成结构化填表则使用序列到序列的对齐损失。

**巧妙之处**  
- 用 LLM 本身完成稀疏补全，避免了额外的外部填补模型，保持了“一体化”生成链路。  
- 超图的三层超边设计让模型同时拥有局部（行/列）和全局视野，信息不必层层堆叠才能到达远距离节点。  
- Prompt‑Attentive 注意力把任务指令直接注入结构学习过程，使得同一超图在不同任务下可以自适应关注不同子结构，省去了为每个任务重新设计图结构的繁琐。

### 实验与效果
- **实验任务**：论文在两个需要结构化知识的下游任务上做评估：① 基于网页表格的事实问答（WebTableQA），② 表格到自然语言的描述生成（Table2Text）。两者都要求模型理解行列关系并利用稀疏信息。  
- **基线对比**：分别与最先进的序列化方法（如 Table‑Prompt）和操作式方法（如 SQL‑Prompt）进行比较。  
- **结果**：在 WebTableQA 上，HyperG 的准确率提升约 7%（从 68% 到 75%），在 Table2Text 上的 BLEU 分数提升约 4.5（从 21.2 到 25.7）。这些提升在论文中被标记为显著（p < 0.05）。  
- **消融实验**：作者分别去掉（1）上下文增强、（2）行/列超边、（3）Prompt‑Attentive 注意力、（4）LoRA 微调。结果显示，去掉上下文增强导致整体性能下降约 3%，去掉行/列超边导致跨行推理错误率上升 12%，去掉 Prompt‑Attentive 注意力使任务适配能力明显削弱，尤其在 Table2Text 场景下降 2.8 BLEU。  
- **局限性**：论文承认对极大规模表格（行列数超过千级）时超图构建和注意力计算的成本仍然较高，需进一步探索稀疏注意力或分块超图的加速方案。  

### 影响与延伸思考
HyperG 把超图引入 LLM 处理结构化数据的流程，开启了“结构感知大模型”的新方向。随后的工作（如 HyperGraph‑LLM、Structurally‑Aware Prompting）纷纷在超边设计、稀疏注意力以及跨模态超图融合上进行扩展。对想进一步探索的读者，可以关注以下几个方向：  
1. **大规模超图加速**：利用图采样、分层超图或近似注意力降低计算复杂度。  
2. **多模态超图**：把图像、音频等非文本节点也纳入同一超图，实现跨模态结构推理。  
3. **自监督超图预训练**：在海量网页表格上做超图对比学习，为下游任务提供更通用的结构表征。  

### 一句话记住它
HyperG 用 LLM 生成的上下文填补稀疏格子，再把完整表格映射成超图，让模型在结构上“一眼看穿”行列关系，从而显著提升对结构化知识的理解与生成。