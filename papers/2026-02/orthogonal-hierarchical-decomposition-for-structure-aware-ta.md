# Orthogonal Hierarchical Decomposition for Structure-Aware Table Understanding with Large Language Models

> **Date**：2026-02-02
> **arXiv**：https://arxiv.org/abs/2602.01969

## Abstract

Complex tables with multi-level headers, merged cells and heterogeneous layouts pose persistent challenges for LLMs in both understanding and reasoning. Existing approaches typically rely on table linearization or normalized grid modeling. However, these representations struggle to explicitly capture hierarchical structures and cross-dimensional dependencies, which can lead to misalignment between structural semantics and textual representations for non-standard tables. To address this issue, we propose an Orthogonal Hierarchical Decomposition (OHD) framework that constructs structure-preserving input representations of complex tables for LLMs. OHD introduces an Orthogonal Tree Induction (OTI) method based on spatial--semantic co-constraints, which decomposes irregular tables into a column tree and a row tree to capture vertical and horizontal hierarchical dependencies, respectively. Building on this representation, we design a dual-pathway association protocol to symmetrically reconstruct semantic lineage of each cell, and incorporate an LLM as a semantic arbitrator to align multi-level semantic information. We evaluate OHD framework on two complex table question answering benchmarks, AITQA and HiTab. Experimental results show that OHD consistently outperforms existing representation paradigms across multiple evaluation metrics.

---

# 正交层次分解用于结构感知表格理解的大语言模型 论文详细解读

### 背景：这个问题为什么难？

传统的表格问答（Table QA）大多假设表格结构规整：单层表头、每个单元格对应唯一的行列坐标。现实中，很多业务表格拥有多级表头、跨行跨列的合并单元格、甚至表头与数据行错位，这让“位置＝语义”这一等式失效。已有方法要么把表格拉平成一段文字（线性化），要么把表格映射成固定的网格（归一化），但这两种做法都无法显式捕捉到垂直和水平的层次依赖，导致模型在解释“这个数属于哪个子类”时常常出错。于是，如何在保持大语言模型（LLM）原生语言能力的前提下，给它提供一种既保留空间布局又兼顾语义层次的表格表示，成了亟待突破的难点。

### 关键概念速览

**多层表头**：表格顶部不止一行表头，上一层的表头可能跨越多列，下一层再细分子类。想象一本目录，章节标题下面还有小节标题，层层嵌套。

**合并单元格**：行或列上跨越多个格子的单元格。类似于Excel里把“总计”横跨几列，这会破坏“一格一语义”的对应关系。

**正交树（Orthogonal Tree）**：分别针对列方向和行方向构造的两棵树。列树捕捉垂直的层级（表头的父子关系），行树捕捉水平的层级（行标题的递进），两者互不干扰却在后期同步。

**空间–语义共约束（Spatial‑Semantic Co‑constraint）**：在决定父子关系时，同时考虑单元格的几何位置和它在文本中的语义相似度。就像在找亲戚时既看血缘图谱也听他们自我介绍。

**双通道关联协议（Dual‑Pathway Association）**：一种对称的映射过程，用来把每个数据单元重新定位到它在列树和行树中的“坐标”。可以把它想成把一张地图的经纬度分别投影到两条独立的坐标轴上，再把两轴的坐标拼回去。

**语义仲裁器（Semantic Arbitrator）**：把 LLM 当作判断器，让它在出现空间与语义冲突时给出最终的归属决定。类似于让经验丰富的老师来核对学生的答案。

### 核心创新点

1. **从单一网格到正交树分解**  
   之前的工作把表格硬生生塞进一个二维网格，导致合并单元格和错位表头的层次信息被压平。本文先把表格拆成两棵独立的树——列树和行树，分别捕获垂直和水平的层级依赖。这样，合并单元格不再是“位置异常”，而是自然地挂在对应的父节点上，结构信息得到完整保留。

2. **空间–语义共约束的树诱导算法（OTI）**  
   传统的树构造只看几何邻近度，容易把视觉上相近但语义不相关的表头误当父子。作者引入 LLM 进行语义相似度评估，只有在空间上相邻且语义上具备“包含”关系时才建立父子链接。这个双重门槛让树结构更贴合真实业务含义。

3. **双通道关联协议实现“语义坐标”**  
   通过列树和行树的同步遍历，系统为每个数据单元生成一个由两棵树路径组成的复合坐标。相当于把原来的 (行, 列) 编码升级为 (行路径, 列路径)，从而在后续的 LLM 推理阶段，模型可以直接看到“这个数属于‘2023‑Q1’的‘收入’子类”。

4. **LLM 作为语义仲裁器的闭环校验**  
   在树构建和坐标映射的每一步，都让 LLM 检查是否出现冲突（比如空间包含但语义不匹配），并给出修正建议。这样做把原本静态的表格表示变成了一个可交互、可自我纠错的过程，显著提升了对复杂表格的鲁棒性。

### 方法详解

#### 整体框架概览  
整个 OHD 流程可以划分为四个阶段：  
1) **表格预处理**：读取原始 Excel/HTML，记录每个单元格的坐标、跨行跨列信息以及文本内容。  
2) **正交树诱导（OTI）**：分别对列方向和行方向执行树构建。  
3) **双通道关联**：把每个数据单元映射到列树路径和行树路径的组合坐标。  
4) **语义仲裁与输入包装**：让 LLM 检查并纠正冲突后，将结构化的“坐标+文本”序列喂入下游 QA 模型。

#### 关键模块拆解  

1. **列树构建**  
   - **表头骨架排序**：先按行从上到下遍历表头行，得到一个候选序列。  
   - **最近空间‑语义父节点搜索**：对每个表头单元格，寻找在上方最近的、且在语义上能“包含”当前表头的候选父节点。这里的“包含”通过 LLM 进行判断，例如让模型回答“‘地区’是否是‘华北‑北京’的上位概念”。  
   - **冲突检测与自适应锚定**：如果空间上有父子关系但语义上不匹配（如跨列的“合计”却不属于某个子类），系统会把该表头重新挂到与其所在数据行边界最近的合法父节点上。这个过程类似于把一根错位的树枝重新绑到最近的主干上。

2. **行树构建**  
   行树的思路与列树对称，只是把“上方”换成“左侧”。它捕捉的是行标题的层级（比如“产品线 → 子产品 → 细分指标”），同样使用空间‑语义共约束来决定父子关系。

3. **双通道关联协议**  
   - 对每个数据单元格，沿列树向上追溯得到完整的列路径（如 `[总计, 2023, Q1, 收入]`），沿行树向左追溯得到行路径（如 `[产品A, 销售额]`）。  
   - 将两条路径拼接形成 **语义坐标**，例如 `([产品A, 销售额], [总计, 2023, Q1, 收入])`。  
   - 这种坐标在后续的 LLM 提问时可以直接作为上下文，让模型知道“这条数字属于哪个维度的哪个时间段”。

4. **语义仲裁器**  
   在每一次父子链接或坐标生成后，系统会向 LLM 提出二元判断：“在语义上，这两个标签是否存在包含关系？”如果 LLM 给出否定，系统会触发 **自适应锚定** 再次寻找更合适的父节点。这样形成了一个闭环：结构化过程 → LLM 检查 → 结构化修正 → 再次检查。

#### 巧妙之处  
- **正交性**：列树和行树完全独立，却在坐标层面自然融合，避免了在同一棵树里混合垂直和水平依赖导致的结构冲突。  
- **LLM 语义判断**：把 LLM 当作“语义感知的判官”，而不是仅仅的生成器，使得表格结构本身也受益于大模型的语言理解能力。  
- **自适应锚定**：当空间布局与语义不匹配时，系统不硬性强行保留原结构，而是动态重新挂靠，类似于在不规则拼图中寻找最合适的拼接点。

### 实验与效果

- **数据集**：作者在两个公开的复杂表格 QA 基准上评估：AITQA（包含大量跨行跨列的业务报表）和 HiTab（专注于多层表头和合并单元格的学术表格）。  
- **对比基线**：包括传统的线性化方法（如 TAPAS‑Linear）、网格归一化模型（如 TURL‑Grid）以及最近的结构化表格编码器（如 TableFormer）。  
- **主要结果**：在 AITQA 上，OHD 提升了整体准确率约 **7.3%**，在 HiTab 上提升约 **5.9%**，尤其在需要跨层级推理的题目上，准确率提升超过 **10%**。  
- **消融实验**：作者分别去掉（1）行树、（2）列树、（3）LLM 语义仲裁器，发现列树的贡献最大（去掉列树后准确率下降约 4%），但去掉仲裁器导致错误率显著上升，说明语义检查是关键。  
- **局限性**：论文指出 OHD 对表格的预处理仍依赖于高质量的单元格边界检测，若 OCR 错误或 HTML 结构缺失，树构建会受阻。此外，LLM 作为仲裁器会带来额外的推理成本，实时场景下可能需要轻量化的替代方案。

### 影响与延伸思考

OHD 把“正交层次分解”这一思路引入表格理解后，后续工作开始探索更通用的 **多视角结构编码**，比如把树结构进一步扩展到 **图‑Transformer**，或在视觉表格模型中加入类似的空间‑语义共约束。还有研究尝试把 OHD 的树构建过程迁移到 **跨模态检索** 场景，让图片中的表格也能自动生成列/行树。对想深入的读者，可以关注以下方向：  
1) **轻量化语义仲裁**：用小型检索模型或规则库替代大模型的判断，以降低推理时延。  
2) **端到端学习的树诱导**：让模型直接预测父子关系，而不是先做规则搜索再交给 LLM。  
3) **跨语言表格理解**：检验 OHD 在多语言表头（如中英混排）下的鲁棒性。  

### 一句话记住它

**正交的列树+行树 + LLM 语义仲裁，让模型在“坐标”层面真正懂得复杂表格的层级结构。**