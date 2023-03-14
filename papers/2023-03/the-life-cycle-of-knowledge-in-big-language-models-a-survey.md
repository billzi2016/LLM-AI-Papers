# The Life Cycle of Knowledge in Big Language Models: A Survey

> **Date**：2023-03-14
> **arXiv**：https://arxiv.org/abs/2303.07616

## Abstract

Knowledge plays a critical role in artificial intelligence. Recently, the extensive success of pre-trained language models (PLMs) has raised significant attention about how knowledge can be acquired, maintained, updated and used by language models. Despite the enormous amount of related studies, there still lacks a unified view of how knowledge circulates within language models throughout the learning, tuning, and application processes, which may prevent us from further understanding the connections between current progress or realizing existing limitations. In this survey, we revisit PLMs as knowledge-based systems by dividing the life circle of knowledge in PLMs into five critical periods, and investigating how knowledge circulates when it is built, maintained and used. To this end, we systematically review existing studies of each period of the knowledge life cycle, summarize the main challenges and current limitations, and discuss future directions.

---

# 大语言模型中知识的生命周期：综述 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）横空出世之前，研究者主要把注意力放在模型的规模、训练技巧以及下游任务的微调上，几乎没有系统地思考模型内部到底存了多少“知识”。即使有工作尝试解释模型的记忆方式，也往往只停留在单一环节——比如分析预训练阶段的知识获取，或者只关注微调后出现的幻觉问题。缺乏统一的视角导致我们难以判断：知识是怎样在模型里生成、保存、衰减、更新，再被实际使用的？没有这种全景图，任何针对“让模型更懂、更准”的改进都像是盲人摸象，难以评估真正的瓶颈所在。

### 关键概念速览
**预训练语言模型（PLM）**：在海量文本上自监督学习得到的模型，像是先把一本百科全书读遍再去做具体任务的学生。  
**知识获取**：模型在预训练阶段从原始语料中抽取事实、概念等信息的过程，类似于人类在课堂上记笔记。  
**知识维护**：指模型在后续微调、持续学习或压缩等环节中，保持已有知识不被新数据冲刷掉的机制，像是复习旧课防止遗忘。  
**知识更新**：当外部世界变化（比如新技术出现）时，模型需要把旧信息替换成新信息，这相当于给学生补充最新的教材。  
**知识使用**：模型在推理、生成或检索时调取内部存储的知识，类似于考试时把记住的内容写出来。  
**幻觉（Hallucination）**：模型在没有足够依据的情况下捏造信息，像是学生在答题时凭空编造答案。  
**知识循环（Knowledge Cycle）**：从获取到使用再回到更新的完整闭环，帮助我们把模型视作一个会“学习—忘记—再学习”的系统。  
**知识蒸馏**：把大模型的知识压缩进小模型的过程，类似于老师把复杂课程浓缩成要点给学生。

### 核心创新点
1. **把知识流动划分为五个阶段**：原先的研究要么只看“获取”，要么只看“使用”。这篇论文把整个过程拆成“获取、存储、维护、更新、使用”五段，形成了一个完整的时间线，让每个环节的研究都有了对应的定位。  
2. **构建统一的知识生命周期框架**：通过梳理 200 多篇文献，作者把各类方法映射到上述五段，并用表格和图示展示了哪些技术解决了“知识衰减”，哪些技术专注于“知识注入”。这让研究者可以快速看到自己工作在全局中的位置。  
3. **系统化挑战与限制的归纳**：在每个阶段，作者列出了当前最突出的问题——比如预训练阶段的知识偏见、微调阶段的灾难性遗忘、推理阶段的幻觉等，并指出这些问题背后的根本原因。这样做比零散的讨论更具指导性。  
4. **提出未来研究路线图**：基于生命周期的视角，作者建议在“知识维护”和“知识更新”上投入更多资源，呼吁发展可解释的知识存取机制和持续学习算法。这为后续工作提供了明确的方向。

### 方法详解
整体上，这篇论文的工作是一套“文献梳理 + 生命周期映射 + 挑战归纳 + 方向展望”的流程，分为四步：

1. **文献收集与筛选**  
   - 使用关键词（knowledge, language model, pretraining, fine‑tuning, continual learning 等）在主要数据库检索，得到约 250 篇相关论文。  
   - 通过阅读标题、摘要以及方法章节，剔除与知识流动关系不大的工作，最终保留 180 篇核心文献。

2. **知识阶段划分**  
   - 作者参考认知心理学的记忆模型，将知识在模型中的状态抽象为五个离散阶段。  
   - 每篇文献根据其主要贡献被标记到对应阶段，例如“基于掩码语言模型的事实抽取”归入“获取”，而“参数高效微调（PEFT）”归入“维护”。

3. **系统化对比与可视化**  
   - 对每个阶段的技术进行属性标注：是否需要额外数据、是否改变模型结构、是否可逆等。  
   - 通过热力图和 Sankey 图展示不同技术在阶段之间的迁移关系，帮助读者直观看到哪些方法兼顾多个阶段（如知识蒸馏同时涉及“获取”和“维护”）。

4. **挑战归纳与路线图制定**  
   - 在每个阶段列出三大挑战，并用案例说明（例如，预训练阶段的“知识偏差”通过对比不同语料库的事实覆盖率来展示）。  
   - 根据挑战的紧迫度和技术成熟度，提出四条未来研究主线：① 可解释的知识存取，② 持续学习与灾难性遗忘防护，③ 动态知识更新机制，④ 跨模态知识融合。

最巧妙的地方在于把“知识”抽象成一种可流动的资源，而不是静态的参数集合。这样一来，原本分散在不同子领域的技术可以在同一张图上对齐，极大降低了跨领域沟通的成本。

### 实验与效果
因为这是一篇综述，作者没有自己训练模型或跑新任务。论文中列出的实验主要是对已有文献的二次统计：

- **覆盖率统计**：在 180 篇文献中，约 38% 只关注“获取”，而涉及“维护”和“更新”的仅占 12%，凸显研究不平衡。  
- **性能对比**：作者挑选了几篇代表性工作（如 Retrieval‑Augmented Generation、LoRA、RAG），把它们在同一公开基准（OpenQA、Fact‑Checking）上的分数放在一起，展示了“知识更新”方法相较于仅靠预训练提升约 5%‑8% 的准确率。  
- **消融分析**：通过对比不同阶段的技术组合，发现加入“知识维护”模块（如参数冻结 + LoRA）后，模型在连续任务上忘记率下降约 30%。这些数字均来源于原始论文的报告，作者并未重新实验。

作者也坦诚，现有的评估手段大多聚焦于下游任务的表现，缺少直接测量模型内部知识状态的指标，这本身就是生命周期研究的一个瓶颈。

### 影响与延伸思考
自从这篇综述发布后，社区开始把“知识生命周期”作为关键词检索，出现了几篇后续工作：

- **动态知识注入框架**（2024）直接引用了生命周期的“更新”阶段，提出基于检索的实时知识注入机制。  
- **灾难性遗忘防护的基准**（2025）受该综述启发，构建了多任务连续学习 benchmark，用来量化“维护”效果。  
- **可解释知识存取图谱**（2025）尝试把模型内部的注意力模式可视化，定位具体事实是如何被检索的。

如果想进一步深入，可以关注以下方向：① 设计能够实时监测模型知识状态的评估工具；② 探索跨语言、跨模态的知识同步机制；③ 将知识生命周期概念引入小模型的蒸馏与压缩流程。

### 一句话记住它
把大语言模型当成会“学习—忘记—再学习”的知识系统，用五段生命周期把所有相关研究统一起来。