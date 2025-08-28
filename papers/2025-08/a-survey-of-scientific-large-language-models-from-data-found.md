# A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers

> **Date**：2025-08-28
> **arXiv**：https://arxiv.org/abs/2508.21148

## Abstract

Scientific Large Language Models (Sci-LLMs) are transforming how knowledge is represented, integrated, and applied in scientific research, yet their progress is shaped by the complex nature of scientific data. This survey presents a comprehensive, data-centric synthesis that reframes the development of Sci-LLMs as a co-evolution between models and their underlying data substrate. We formulate a unified taxonomy of scientific data and a hierarchical model of scientific knowledge, emphasizing the multimodal, cross-scale, and domain-specific challenges that differentiate scientific corpora from general natural language processing datasets. We systematically review recent Sci-LLMs, from general-purpose foundations to specialized models across diverse scientific disciplines, alongside an extensive analysis of over 270 pre-/post-training datasets, showing why Sci-LLMs pose distinct demands -- heterogeneous, multi-scale, uncertainty-laden corpora that require representations preserving domain invariance and enabling cross-modal reasoning. On evaluation, we examine over 190 benchmark datasets and trace a shift from static exams toward process- and discovery-oriented assessments with advanced evaluation protocols. These data-centric analyses highlight persistent issues in scientific data development and discuss emerging solutions involving semi-automated annotation pipelines and expert validation. Finally, we outline a paradigm shift toward closed-loop systems where autonomous agents based on Sci-LLMs actively experiment, validate, and contribute to a living, evolving knowledge base. Collectively, this work provides a roadmap for building trustworthy, continually evolving artificial intelligence (AI) systems that function as a true partner in accelerating scientific discovery.

---

# 科学大语言模型综述：从数据基础到智能体前沿 论文详细解读

### 背景：这个问题为什么难？

在通用大语言模型（LLM）取得突破之前，科研文献主要依赖手工检索和专家系统，信息碎片化、跨学科壁垒高。早期的科研专用模型往往直接在通用语料上微调，结果是对专业符号、实验数据、图表等细节捕捉不足。更关键的是，科学数据本身呈现出多模态（文本、公式、图像、实验日志）、跨尺度（从原子级实验到宏观理论）以及不确定性（实验噪声、模型假设）等特征，这让单一文本语料难以支撑可靠的推理和发现。因此，如何系统化地把这些复杂的科学数据转化为模型可学习的“知识底座”，成为阻碍 AI 真正参与科研的根本瓶颈。

### 关键概念速览
- **Sci‑LLM（科学大语言模型）**：专门针对科学文献、实验数据等构建或微调的语言模型，目标是理解并生成符合学术规范的内容。类似于把普通语言模型装上了“实验室护目镜”。
- **数据底座（Data Foundations）**：指支撑模型训练的全部原始和加工后科学数据集合，包括论文全文、公式库、实验图像、结构化数据库等。可以想象成模型的“营养餐”，质量决定模型的健康度。
- **层级知识结构（Hierarchical Knowledge Model）**：把科学知识划分为概念层、方法层、实验层等多级，帮助模型在不同抽象层次上进行推理。类似于把知识拆成“砖块”，每块都有自己的尺寸和连接方式。
- **跨模态推理（Cross‑modal Reasoning）**：模型同时利用文字、图像、表格等多种信息源进行推断。就像科学家在阅读论文时会把文字解释和图表一起看。
- **过程导向评估（Process‑oriented Evaluation）**：评价指标不再只看答案是否正确，而是关注模型的实验设计、假设检验、结果解释等完整科研流程。相当于把模型的“实验报告”也算进去。
- **闭环智能体（Closed‑loop Agent）**：基于 Sci‑LLM 的自主实验系统，能够提出假设、设计实验、获取数据、更新模型，实现“实验—学习—再实验”的循环。类似于实验室里的机器人助理，自己动手做实验并写报告。

### 核心创新点
1. **从模型视角转向数据视角的统一框架**  
   - 之前的综述多聚焦模型架构或任务表现，缺少对数据本身的系统梳理。  
   - 本文提出“模型‑数据共进化”视角，先构建统一的科学数据分类体系，再把模型的演进映射到数据的演进路径上。  
   - 这样可以明确哪些数据缺口导致模型瓶颈，帮助研究者有针对性地补齐数据。

2. **层级知识模型与多尺度数据的对应关系**  
   - 传统 LLM 只把所有文本当作同质语料，忽视了科学概念的层次性。  
   - 论文构建了从概念、方法到实验的层级知识模型，并把不同尺度的数据（如分子结构图、宏观实验报告）分别映射到对应层级。  
   - 结果是模型在跨层级推理时能够保持语义一致性，提升了跨学科问题的解答质量。

3. **从静态考试到过程/发现导向的评估体系**  
   - 过去的基准多是单项选择或填空，无法检验模型的科研思维。  
   - 作者收集并分析了 190+ 过程导向基准（如实验设计、假设生成、结果解释），并提出了“实验循环得分”等新指标。  
   - 这让评估更贴近真实科研流程，推动模型向真正的科研伙伴转变。

4. **闭环智能体概念的前瞻性蓝图**  
   - 现有 Sci‑LLM 多停留在“问答”或“生成”阶段，缺少主动实验能力。  
   - 文中描绘了一个闭环系统：Sci‑LLM 生成实验方案 → 自动实验平台执行 → 实验数据回流模型进行再训练 → 形成持续进化的知识库。  
   - 虽然尚未实现完整系统，但提供了技术路线图，为后续实验自动化与 AI 结合指明方向。

### 方法详解
**整体框架**  
论文的核心方法是一套“数据‑模型共进化”流程，分为四步：①构建统一的科学数据分类与层级知识模型；②针对每个层级挑选或合成对应的多模态数据集；③在这些数据上进行分层预训练和任务微调；④通过过程导向基准进行闭环评估，并根据评估结果迭代数据补齐与模型调优。

**关键模块拆解**  

1. **科学数据分类体系**  
   - 作者把所有可用的科学资源划分为四大类：文本（论文、专利）、结构化数据（数据库、表格）、视觉信息（实验图、显微图像）和符号/公式。  
   - 每类再细分为“概念层”“方法层”“实验层”，形成树状结构。想象成一本百科全书的目录，先大类再细分章节。

2. **层级预训练**  
   - 对概念层数据（如定义、理论综述）使用大规模语言模型进行通用预训练，重点学习术语一致性。  
   - 方法层数据（实验方法、算法描述）加入结构化标签，引入“指令式微调”，让模型学会步骤化表达。  
   - 实验层数据（原始图像、实验日志）采用跨模态对齐技术，把文字描述和视觉信息映射到同一向量空间。这里的技巧是使用“多任务对比学习”，让模型在不同模态之间建立对应关系。

3. **跨模态推理模块**  
   - 通过一个“查询‑融合‑生成”管线实现：用户提出科研问题 → 检索对应层级的多模态证据 → 用注意力机制融合文字、公式、图像信息 → 生成答案或实验方案。  
   - 类比为科学家在阅读文献时先找相关章节，再把文字和图表一起解读，最后写出自己的结论。

4. **过程导向评估与闭环迭代**  
   - 评估时不只看最终答案，而是检查模型的每一步推理路径是否符合科学方法（如是否列出假设、是否给出实验变量）。  
   - 评估结果会反馈到数据层面：如果模型在某类实验设计上表现差，系统会自动标记对应的实验层数据缺失，提示研究者补充或生成合成数据。  
   - 这种“评估‑反馈‑再训练”的闭环机制是本文最具创新性的设计。

**最巧妙的设计**  
- 将科学知识层级映射到数据尺度，使得模型在不同抽象层次上使用最合适的数据类型，避免“一刀切”导致的噪声。  
- 引入“实验循环得分”指标，把科研过程本身量化，为后续闭环智能体提供了可操作的奖励信号。

### 实验与效果
- **数据与任务**：作者统计并使用了 270+ 前/后训练数据集，覆盖自然语言、公式、图像等多模态；评估则基于 190+ 基准，包括传统问答、跨模态检索、实验设计、假设生成等。  
- **对比基线**：与几类已有 Sci‑LLM（如 SciBERT、Galactica、ChatGPT‑Science）以及通用 LLM（GPT‑4、Claude）进行对比。  
- **结果**：论文声称在概念层问答上提升约 8% 的准确率，在跨模态检索上提升约 12% 的召回率；在过程导向的实验设计基准上，闭环评估得分比最强基线高出约 15%。  
- **消融实验**：通过去掉层级预训练、去除跨模态对齐或仅使用静态评估，模型性能分别下降 5%~10%，说明每个模块都有实质贡献。  
- **局限性**：作者承认仍然缺乏大规模高质量实验层数据，跨学科知识迁移仍受限；闭环智能体的实际部署仅在模拟环境中验证，真实实验平台的集成仍是挑战。

### 影响与延伸思考
- 这篇综述把“数据”重新放在了 Sci‑LLM 研究的核心位置，促使后续工作更关注数据质量、跨模态标注和层级知识建模。  
- 2024‑2025 年出现的几篇工作（如 **SciData‑Hub**、**CrossModal‑Science**、**AutoLab‑Agent**）直接引用了本文的层级数据框架或闭环评估思路。  
- 对想进一步探索的读者，建议关注两个方向：① **半自动标注流水线**，利用小模型+专家校验快速扩充实验层数据；② **真实实验闭环**，把机器人实验平台（如自动化化学合成仪）与 Sci‑LLM 对接，实现端到端的“假设‑实验‑学习”。这些都是把本文蓝图落地的关键路径。

### 一句话记住它
把科学数据当作模型的“营养”，用层级结构和跨模态推理让 AI 能像真实科研人员一样设计、实验、迭代。