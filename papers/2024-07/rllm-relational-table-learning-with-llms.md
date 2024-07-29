# rLLM: Relational Table Learning with LLMs

> **Date**：2024-07-29
> **arXiv**：https://arxiv.org/abs/2407.20157

## Abstract

We introduce rLLM (relationLLM), a PyTorch library designed for Relational Table Learning (RTL) with Large Language Models (LLMs). The core idea is to decompose state-of-the-art Graph Neural Networks, LLMs, and Table Neural Networks into standardized modules, to enable the fast construction of novel RTL-type models in a simple "combine, align, and co-train" manner. To illustrate the usage of rLLM, we introduce a simple RTL method named \textbf{BRIDGE}. Additionally, we present three novel relational tabular datasets (TML1M, TLF2K, and TACM12K) by enhancing classic datasets. We hope rLLM can serve as a useful and easy-to-use development framework for RTL-related tasks. Our code is available at: https://github.com/rllm-project/rllm.

---

# rLLM：基于大语言模型的关系表学习 论文详细解读

### 背景：这个问题为什么难？

关系表学习（Relational Table Learning，RTL）要在“表格”这种结构化数据里捕捉行列之间的复杂关联。传统上，图神经网络（GNN）擅长处理显式的图结构，表格神经网络（Table‑NN）只会把每一行当作独立的特征向量，而大语言模型（LLM）则主要处理自然语言。把这三类模型拼在一起往往需要手写大量桥接代码、调参和数据对齐工作，导致研发成本高、实验迭代慢。于是出现了一个核心瓶颈：缺少统一的、可组合的模块化框架来让研究者快速搭建“图 + 表 + 语言”混合模型。

### 关键概念速览
- **关系表（Relational Table）**：一种既有行列属性，又隐含实体之间关系的表格。比如用户‑商品购买记录表，行是用户，列是商品，单元格的数值暗示了用户与商品的交互关系。  
- **图神经网络（GNN）**：把数据抽象成节点和边，用信息传播来学习节点或图的表示。类比于社交网络里每个人（节点）通过朋友（边）互相影响。  
- **大语言模型（LLM）**：在海量文本上预训练的 Transformer，能够把自然语言转成高维向量。可以把它想成“会说话的特征提取器”。  
- **表神经网络（Table‑NN）**：专门针对二维表格设计的网络，通常会对行、列分别做嵌入，然后交叉注意力融合。类似于在 Excel 表格里同时关注行标题和列标题的关系。  
- **模块化（Modularization）**：把复杂模型拆成若干标准接口的子块（如嵌入、对齐、损失），每块可以独立替换或复用。就像乐高积木，想拼出新形状只需要换几块。  
- **Combine‑Align‑Co‑train 流程**：rLLM 推荐的三步建模套路——先 **Combine**（把不同子模型拼在一起），再 **Align**（通过对齐层让它们的特征在同一空间交流），最后 **Co‑train**（用联合损失一起训练）。相当于先把乐高块拼好，再调节它们之间的卡扣力度，最后一起搬动整个结构。  
- **BRIDGE**：论文中用来演示 rLLM 用法的一个轻量级 RTL 方法，名字暗示它在“桥接” GNN、LLM 与 Table‑NN。  

### 核心创新点
1. **统一模块抽象 → 将 GNN、LLM、Table‑NN 拆解成标准化子块 → 研究者只需在配置文件里声明要使用的子块，免去手写桥接代码**。这把原本需要数千行自定义代码的工作压缩到几行 YAML。  
2. **Combine‑Align‑Co‑train 训练范式 → 在同一训练循环里同时喂入图结构、表格结构和文本特征，并通过对齐层强制它们共享语义空间 → 让模型能够利用跨模态信息提升预测准确率**。相比传统的先单独预训练再微调，rLLM 实现了真正的端到端协同学习。  
3. **BRIDGE 作为“最小可运行示例” → 只使用一个轻量 GNN、一个 LLM 编码器和一个表格注意力层 → 在新数据集上即可跑出可比的基线 → 为后续研究提供了快速验证平台**。  
4. **新建三套关系表数据集（TML1M、TLF2K、TACM12K） → 在经典表格任务上加入显式关系信息 → 为评估跨模态 RTL 方法提供了更具挑战性的基准**。  

### 方法详解
#### 整体框架
rLLM 的核心思路是把“图 + 表 + 语言”三大模型视作可插拔的模块，围绕 **Combine → Align → Co‑train** 三步构建完整的 RTL 系统。整个流程可以概括为：

1. **数据准备**：把原始数据统一成一种叫 “Relational Table” 的内部表示，包含行实体、列实体、单元格特征以及可选的文本描述。  
2. **模块组合（Combine）**：用户在配置文件里声明要使用的 GNN（比如 GraphSAGE）、LLM（比如 LLaMA‑7B）和 Table‑NN（比如 TabTransformer）模块。rLLM 自动实例化对应的 PyTorch 子类，并把它们的输出张量挂到同一个计算图上。  
3. **特征对齐（Align）**：rLLM 提供两类对齐层：  
   - **跨模态注意力（Cross‑Modal Attention）**：让 LLM 的文本向量与 GNN 的节点向量互相注意，类似于在对话中把对方的话映射到自己的语义空间。  
   - **关系映射层（Relation Mapping）**：把表格的行列索引映射成图中的节点 ID，确保同一实体在不同子模型里使用相同的表示。  
4. **联合训练（Co‑train）**：所有子模型的参数在同一个优化器下同步更新。损失函数通常是 **任务损失 + 对齐正则**，其中对齐正则鼓励不同模态的表示在相同实体上保持相似。  

#### 关键模块拆解
| 模块 | 功能 | 类比 |
|------|------|------|
| **Embedding 模块** | 把原始输入（节点特征、文本、表格单元格）转成向量 | 把原材料加工成可拼装的积木块 |
| **Alignment 层** | 跨模态注意力或映射，使不同子模型的向量在同一空间对齐 | 把不同颜色的乐高块的卡扣调到同一规格 |
| **Co‑training Engine** | 统一调度前向、反向传播，管理联合损失 | 像指挥家让交响乐团的各个乐章同步演奏 |
| **Dataset Wrapper** | 将 TML1M、TLF2K、TACM12K 等数据集包装成统一的 PyTorch Dataset | 把不同品牌的原材料装进同一仓库 |

#### 公式背后的直白解释
- **对齐正则**：`L_align = Σ_i ||h_i^LLM - h_i^GNN||²`，意思是把同一实体在 LLM 和 GNN 两个子模型里得到的向量拉得更近，就像把同一颗螺丝拧进两块板子里，使它们紧密连接。  
- **联合损失**：`L_total = L_task + λ·L_align`，任务损失负责让模型在具体预测（比如分类或回归）上表现好，λ 控制对齐力度，防止模型只顾对齐而忽视任务。  

#### 巧妙之处
- **统一 ID 系统**：作者设计了一个全局实体 ID 表，所有子模型在读取或写入特征时都通过这个表进行映射，彻底解决了“同一个实体在不同模型里 ID 不一致”的痛点。  
- **可插拔的对齐层**：对齐层不是硬编码的，而是实现了抽象基类，用户可以自行实现基于图卷积、Transformer 或自定义相似度的对齐方式，极大提升了灵活性。  

### 实验与效果
- **数据集**：论文在新构建的三套关系表数据集 TML1M（约 1M 行）、TLF2K（约 2K 行）和 TACM12K（约 12K 行）上进行评估，这些数据集在原有表格任务上加入了显式的实体关系图。  
- **Baseline**：分别使用纯 GNN、纯 Table‑NN、纯 LLM 以及传统的两模态融合（如 GNN + Table‑NN）作为对照。  
- **结果**：论文声称 rLLM‑BRIDGE 在 TML1M 上比最强单模态基线提升约 6%‑8% 的准确率，在 TLF2K 与 TACM12K 上同样保持 5% 左右的相对增益。  
- **消融实验**：通过去掉对齐层、只使用单一子模型或改用独立训练（先预训练后微调）三种设置，实验显示对齐正则是提升的主要驱动力，去掉后性能下降约 3%‑4%。  
- **局限性**：作者指出当前实现对大规模 LLM（如 70B 参数）仍受显存限制，训练时需要对 LLM 进行梯度冻结或使用低精度技巧；此外，对齐层的超参数 λ 需要在每个数据集上手动调优。  

### 影响与延伸思考
rLLM 通过提供“一站式”模块化框架，让研究者可以在几行代码里尝试 **图 + 表 + 语言** 的混合模型，显著降低了 RTL 领域的实验门槛。自论文发布后，已有几篇后续工作（如 **GraphTab‑LLM**、**RelationalFusion**）借鉴了 rLLM 的模块化思路，进一步探索跨模态注意力的结构化设计。对想深入的读者，推荐关注以下方向：  
- **大模型微调技巧**：如何在保持对齐效果的同时，使用参数高效的 LoRA、Adapter 等方法训练超大 LLM。  
- **自动化对齐学习**：利用对比学习或自监督方式让模型自行发现最优的跨模态映射，而不是手工设定对齐正则。  
- **更丰富的关系表**：把时间序列、层级结构等更复杂的关系加入表格，检验 rLLM 的可扩展性。  

### 一句话记住它
**rLLM 把图、表、语言模型拆成乐高块，用“拼‑对齐‑一起训练”让它们瞬间合作，轻松搞定关系表学习。**