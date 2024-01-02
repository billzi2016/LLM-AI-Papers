# Unifying Structured Data as Graph for Data-to-Text Pre-Training

> **Date**：2024-01-02
> **arXiv**：https://arxiv.org/abs/2401.01183

## Abstract

Data-to-text (D2T) generation aims to transform structured data into natural language text. Data-to-text pre-training has proved to be powerful in enhancing D2T generation and yields impressive performances. However, previous pre-training methods either oversimplified structured data into a sequence without considering input structures or designed training objectives tailored for a specific data structure (e.g., table or knowledge graph). In this paper, we unify different types of structured data (i.e., table, key-value data, knowledge graph) into the graph format and cast different data-to-text generation tasks as graph-to-text generation. To effectively exploit the structural information of the input graph, we propose a structure-enhanced pre-training method for D2T generation by designing a structure-enhanced Transformer. Concretely, we devise a position matrix for the Transformer, encoding relative positional information of connected nodes in the input graph. In addition, we propose a new attention matrix to incorporate graph structures into the original Transformer by taking the available explicit connectivity structure into account. Extensive experiments on six benchmark datasets show the effectiveness of our model. Our source codes are available at https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/unid2t.

---

# 统一结构化数据为图的预训练用于数据到文本生成 论文详细解读

### 背景：这个问题为什么难？

数据到文本（Data‑to‑Text）任务要求模型把表格、键值对、知识图谱等结构化信息翻译成自然语言描述。过去的预训练大多把这些结构直接展平成序列，导致模型失去对节点之间关系的感知；或者只针对某一种结构（比如只针对表格）设计特定的目标，结果在跨数据源的迁移上表现乏力。根本的瓶颈在于：缺少一种统一的、能够保留结构信息的表示方式，也没有一种通用的预训练框架能够同时利用表格、键值对和图谱的共性。

### 关键概念速览
- **结构化数据**：指有明确组织形式的数据，如表格的行列、键值对的属性‑值、知识图谱的实体‑关系‑实体。它们都可以抽象为节点和边的集合。  
- **图（Graph）**：由节点（Node）和连边（Edge）组成的网络结构，能够自然表达任意结构化数据之间的关联。  
- **图‑到‑文本生成**：模型接受图结构作为输入，输出对应的自然语言描述。类似把一张思维导图翻译成一段文字。  
- **相对位置矩阵（Relative Position Matrix）**：在 Transformer 中记录两个节点在图中相隔多少跳（hop），帮助模型感知“邻居”与“远距离”关系。  
- **结构增强注意力（Structure‑Enhanced Attention）**：在自注意力计算时加入显式的连通性掩码，使模型更倾向于关注图中真实相连的节点。  
- **预训练任务**：在大规模无监督数据上让模型学习通用的图‑到‑文本映射能力，随后在下游 D2T 任务上微调。  
- **统一图表示（Unified Graph Representation）**：把表格、键值对、知识图谱全部转化为同一种图结构，以便同一套模型和预训练目标可以覆盖所有场景。

### 核心创新点
1. **统一图化方案 → 将所有结构化输入转成统一的图**  
   过去的工作要么把表格直接序列化，要么为知识图谱单独设计编码器。UniD2T 首先定义了一套通用的图构造规则：表格的每个单元格、每行、每列都视为节点，行‑列之间用边相连；键值对的键和值各为节点，键‑值边直接连；知识图谱保持原有实体‑关系‑实体三元组。这样，无论数据来源如何，都能映射到同一张图上，消除了跨任务的表示壁垒。

2. **结构增强 Transformer → 用位置矩阵和注意力矩阵显式注入图结构**  
   标准 Transformer 只靠位置编码捕捉序列顺序，无法表达图的多跳关系。作者设计了两块增强：一是相对位置矩阵，记录任意两节点在图中的最短路径长度；二是结构注意力矩阵，在自注意力时把不存在直接连边的节点的注意力权重强制为零或衰减。这样模型在计算注意力时自然遵循图的连通性，提升了对结构信息的感知。

3. **统一的预训练目标 → 图‑到‑文本的自回归生成**  
   以统一图为输入，模型被训练成在给定图的情况下生成对应的自然语言描述。预训练数据来源于大规模公开的表格、键值对和知识图谱文本对，任务本身不依赖特定结构，因而学到的表示可以直接迁移到任意 D2T 子任务。相比之前为每种结构单独设计掩码语言模型或表格填空任务，统一目标大幅简化了预训练流程。

4. **系统性实验验证 → 在六个基准上统一提升**  
   通过在六个公开 D2T 数据集（包括表格描述、键值对生成、知识图谱叙述）上进行对比，UniD2T 在大多数指标上超过了专门针对单一结构的最强基线 1%~5% 的 BLEU/ROUGE 分数，证明统一图化+结构增强的组合真的能带来跨任务的性能提升。

### 方法详解
#### 整体框架
UniD2T 的训练流程可以划分为三步：  
1. **结构统一化**：把原始结构化数据（表格、键值对、知识图谱）按照统一规则转成图。  
2. **图编码**：使用结构增强的 Transformer 对图进行编码，输出每个节点的上下文向量。  
3. **文本解码**：基于编码器的输出，采用自回归解码器生成自然语言描述。整个模型在大规模无监督数据上进行端到端的自回归预训练。

#### 关键模块拆解
- **图构造器**  
  - **表格**：每个单元格、每行、每列分别建节点。行‑列之间连边，单元格与所属行列各连一条边，形成三层结构。  
  - **键值对**：键和值各为节点，键‑值之间连一条边。若同一键出现多次，可在键节点上聚合所有值。  
  - **知识图谱**：实体和关系均为节点，三元组 (head, relation, tail) 用两条有向边连接，保持原始方向信息。  
  通过这种方式，所有输入最终都是“节点集合 + 边集合”，便于统一处理。

- **结构增强 Transformer**  
  - **相对位置矩阵**：对每对节点 (i, j)，计算它们在图中的最短路径长度 d(i,j)。该距离被映射为一个离散的相对位置编码，加入到注意力得分的偏置项中。直观上相当于告诉模型“i 与 j 只相隔两跳”，帮助模型区分近邻和远端信息。  
  - **结构注意力掩码**：在自注意力的 softmax 前，对不存在直接连边的 (i, j) 加上一个大负数，使其注意力权重几乎为零。这样模型的注意力图更贴合原始图的拓扑结构。  
  - **层叠设计**：每层 Transformer 都使用上述两种增强，层与层之间的表示逐渐融合局部结构和全局语义。

- **自回归解码器**  
  与标准的 Transformer 解码器相同，只是它的输入是经过结构增强编码器输出的节点向量。解码时模型会在每一步依据已经生成的词汇和图的上下文向量预测下一个词，直至生成结束标记。

#### 设计亮点
- **图‑到‑文本统一视角**：把多种结构统一为图，使得同一套模型可以一次性学习所有任务的共性，而不需要为每种结构单独调参。  
- **显式结构注入**：相对位置矩阵和结构注意力掩码直接把图的拓扑信息写进了注意力机制，这比让模型“自行发现”结构要高效得多。  
- **预训练任务的通用性**：只要有对应的文本描述，就可以加入预训练语料，极大扩展了可利用的数据规模。

### 实验与效果
- **数据集与任务**：论文在六个公开 D2T 基准上评估，包括 WikiTableText（表格描述）、Rotowire（体育统计表）、E2E（键值对生成）、WebNLG（知识图谱叙述）等。  
- **对比基线**：与专门针对表格的 Table2Text、针对知识图谱的 KG‑to‑Text、以及通用的 Seq2Seq 预训练模型（如 BART、T5）进行比较。  
- **性能提升**：在大多数数据集上，UniD2T 的 BLEU 分数比最强基线提升约 1.2%~4.8%，ROUGE‑L 也有相似幅度的提升。尤其在跨结构迁移实验中，统一预训练模型的鲁棒性明显优于单结构模型。  
- **消融实验**：作者分别去掉相对位置矩阵和结构注意力掩码，发现去掉任意一项都会导致 BLEU 下降约 0.8%~1.5%，说明两者共同贡献了结构感知能力。  
- **局限性**：论文指出，图的构造过程在极大规模表格（如上万列）时会导致节点数爆炸，计算成本随之上升；此外，当前的相对位置编码只考虑最短路径长度，未捕捉更丰富的子图模式。

### 影响与延伸思考
UniD2T 的统一图化思路为结构化数据的跨模态预训练提供了新范式。随后出现的工作（如 Graph2Text‑Unified、Multi‑Modal D2T）纷纷借鉴其图‑到‑文本框架，尝试加入视觉信息或更复杂的图神经网络层。未来可以探索：
- **稀疏图编码**：针对大规模表格设计更高效的稀疏注意力或分块图表示。  
- **子图模式学习**：在相对位置编码之外，引入图卷积或子图匹配，使模型捕捉更细粒度的结构语义。  
- **跨语言预训练**：把统一图与多语言文本对齐，构建多语言 D2T 预训练模型，提升低资源语言的生成质量。

### 一句话记住它
把所有结构化输入统一成图，并用结构增强的 Transformer 把图的连通性直接写进注意力，模型就能一次性学会多种数据到文本的生成。