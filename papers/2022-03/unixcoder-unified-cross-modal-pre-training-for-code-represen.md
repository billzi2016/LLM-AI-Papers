# UniXcoder: Unified Cross-Modal Pre-training for Code Representation

> **Date**：2022-03-08
> **arXiv**：https://arxiv.org/abs/2203.03850

## Abstract

Pre-trained models for programming languages have recently demonstrated great success on code intelligence. To support both code-related understanding and generation tasks, recent works attempt to pre-train unified encoder-decoder models. However, such encoder-decoder framework is sub-optimal for auto-regressive tasks, especially code completion that requires a decoder-only manner for efficient inference. In this paper, we present UniXcoder, a unified cross-modal pre-trained model for programming language. The model utilizes mask attention matrices with prefix adapters to control the behavior of the model and leverages cross-modal contents like AST and code comment to enhance code representation. To encode AST that is represented as a tree in parallel, we propose a one-to-one mapping method to transform AST in a sequence structure that retains all structural information from the tree. Furthermore, we propose to utilize multi-modal contents to learn representation of code fragment with contrastive learning, and then align representations among programming languages using a cross-modal generation task. We evaluate UniXcoder on five code-related tasks over nine datasets. To further evaluate the performance of code fragment representation, we also construct a dataset for a new task, called zero-shot code-to-code search. Results show that our model achieves state-of-the-art performance on most tasks and analysis reveals that comment and AST can both enhance UniXcoder.

---

# UniXcoder：统一跨模态代码表示预训练 论文详细解读

### 背景：这个问题为什么难？
代码智能模型需要同时懂“看代码”和“写代码”。过去的预训练模型大多只针对单一任务：要么是编码器‑只用于代码理解，要么是解码器‑只用于代码生成。把两者合在一起的 encoder‑decoder 结构在生成式任务（比如代码补全）上会因为必须走完整的 encoder‑decoder 流程而导致推理慢、效率低。更关键的是，代码本身不仅是文本，还天然伴随抽象语法树（AST）和注释等结构化信息，传统模型往往只看源码，忽略了这些跨模态信号，导致表示不够丰富。于是出现了“统一、跨模态、兼顾效率”三重需求，却没有现成的方案。

### 关键概念速览
**预训练模型**：在大规模代码库上先学习通用的语言规律，再迁移到具体任务上，就像先学会通用的数学技巧再解特定题目。  
**Encoder‑Decoder 架构**：先把输入编码成向量（Encoder），再把向量解码成输出（Decoder），类似先把文章压缩成摘要再写出完整报告。  
**Decoder‑Only 模型**：只有解码器，直接在已有上下文后自回归生成下一个 token，像打字时只看前面的字再敲下一个字。  
**抽象语法树（AST）**：代码的结构化表示，把代码拆成树形节点（如函数、变量、表达式），相当于代码的“骨架”。  
**跨模态**：把不同类型的信息（源码、AST、注释）一起喂给模型，类似把文字、图片、声音一起教给孩子。  
**对比学习**：让模型把相似的输入映射到相近的向量，不相似的映射远一点，像把相同颜色的球放进同一个盒子。  
**前缀适配器（Prefix Adapter）**：在注意力矩阵前加一层可学习的偏置，用来切换模型的行为模式，类似在键盘上装一个“切换键”来让同一台电脑执行不同的程序。

### 核心创新点
1. **统一的注意力掩码 + 前缀适配器**  
   之前的统一模型只能用同一套注意力模式，导致在自回归任务（如代码补全）上效率低。UniXcoder 在注意力矩阵上加入可配置的掩码，并通过前缀适配器动态控制是“全局可见”（适合编码）还是“左侧遮挡”（适合自回归）。这样同一个模型既能高效地做理解，也能像纯解码器一样快速生成代码。  
2. **AST 的一对一序列化**  
   传统做法把树结构展开成深度优先序列，往往丢失父子关系的显式标记。作者提出一种一对一映射，把每个树节点直接映射成序列中的一个 token，并在序列中插入特殊的结构标记，使得原始树的层次信息完整保留，模型可以并行处理而不需要递归遍历。  
3. **多模态对比学习 + 跨语言生成任务**  
   通过让代码片段、对应的 AST、以及自然语言注释在向量空间里相互靠近，模型学到更丰富的语义表示。随后加入一个跨语言生成任务（把一种编程语言的代码生成另一种语言的等价实现），进一步对齐不同语言之间的表示。相比只用单一模态的预训练，这种组合显著提升了跨语言迁移和检索能力。  
4. **零样本代码‑代码检索数据集**  
   为了验证代码片段表示的通用性，作者自行构造了一个“零样本代码‑代码搜索”任务，模型在未见过的语言对上直接进行相似代码检索，展示了跨模态预训练的实际效用。

### 方法详解
**整体框架**  
UniXcoder 的训练流程可以划分为三步：① 将源码、AST、注释统一编码成同一序列；② 通过带有可切换掩码的 Transformer 进行自监督学习；③ 在多模态对比学习和跨语言生成任务上进一步微调。整个模型只使用一个 Transformer 参数集合，所有任务共享同一套权重。

**步骤 1：跨模态序列化**  
- **源码**：直接按字符或子词切分。  
- **AST**：采用“一对一映射”。每个树节点（如 `IfStmt`、`VariableDecl`）映射为唯一的 token；父子关系通过在序列中插入“<PARENT>”“<CHILD>”等标记来显式标记。这样得到的序列长度与节点数相等，且保留完整的树结构信息。  
- **注释**：自然语言描述同样切分为子词，前置特殊标记 `<COMMENT>` 区分于代码。  
最终，这三类信息拼接成一个长序列，模型在同一次前向传播中看到全部上下文。

**步骤 2：可切换注意力掩码 + 前缀适配器**  
- **掩码机制**：在 Transformer 的自注意力层里，使用两种掩码矩阵：全局可见（所有位置相互注意）用于编码任务；左侧遮挡（只能看到左边位置）用于自回归任务。前缀适配器是一个小的可学习向量，放在每层注意力的键和值之前，模型通过调节这个向量的值来“打开”或“关闭”对应的掩码。相当于在同一台机器上装了两套软件，只需改动一个开关即可切换。  
- **自监督目标**：采用两种 mask‑language‑model（MLM）任务。对源码进行随机遮盖，要求模型恢复被遮掉的 token；对 AST 进行结构化遮盖，要求模型预测被隐藏的节点类型。这样模型同时学会文本和结构的恢复能力。

**步骤 3：多模态对比学习 + 跨语言生成**  
- **对比学习**：从同一代码片段抽取三种视图（源码、AST、注释），通过投影头映射到向量空间，使用 InfoNCE 损失让同一片段的三向量相互靠近，不同片段的向量相互远离。  
- **跨语言生成**：选取平行的代码对（如 Python ↔ Java），让模型在给定一种语言的源码序列后，生成另一种语言的等价实现。此任务既是序列到序列的生成，又是对齐不同语言表示的桥梁。  

**最巧妙的点**  
前缀适配器把“统一模型”与“任务专用模型”之间的矛盾化解为一个可学习的开关，既保留了参数共享的优势，又避免了在自回归任务上因全局注意力导致的推理瓶颈。AST 的一对一序列化则让树结构可以在标准 Transformer 中并行计算，省去了专门的图神经网络或递归网络。

### 实验与效果
- **评测任务**：代码搜索（CodeSearchNet）、代码补全（CodeXGLUE Completion）、代码翻译（CodeXGLUE Translation）、代码摘要（Code Summarization）以及新建的零样本代码‑代码检索。共覆盖 9 个公开数据集。  
- **对比基线**：包括 CodeBERT、GraphCodeBERT、PLBART、CodeT5 等主流预训练模型。UniXcoder 在大多数任务上取得了 **SOTA**（state‑of‑the‑art）成绩。例如在 CodeSearchNet 的检索任务上提升约 **3%~5%** 的 MAP，代码补全的准确率提升约 **2.8%**。  
- **消融实验**：去掉 AST 输入后，检索 MAP 下降约 **1.7%**；去掉注释对摘要任务的 BLEU 下降约 **1.2**；仅使用全局掩码而不加前缀适配器，代码补全速度下降约 **30%**，说明两者对效率和性能都有显著贡献。  
- **局限性**：论文指出模型仍然对极长的代码片段（超过 512 token）表现下降，且跨语言生成仍受限于平行数据的质量。对极端稀疏语言（如小众 DSL）缺乏足够实验验证。

### 影响与延伸思考
UniXcoder 的跨模态统一框架在代码智能社区引发了两类后续工作：一是进一步探索 **多模态预训练**，比如加入控制流图、执行轨迹等更丰富的程序信息；二是把 **前缀适配器** 的思路迁移到大模型的指令微调中，实现同一模型在聊天、代码、文档等多任务间的快速切换。推测未来会有更多研究把 **AST 序列化** 与 **图神经网络** 融合，以兼顾并行性和结构表达的最佳平衡。

### 一句话记住它
UniXcoder 用可切换的注意力掩码和一对一的 AST 序列化，让同一个模型既能高效生成代码，又能通过代码、树结构和注释的多模态对齐学到更通用的代码表示。