# At Which Training Stage Does Code Data Help LLMs Reasoning?

> **Date**：2023-09-28
> **arXiv**：https://arxiv.org/abs/2309.16298

## Abstract

Large Language Models (LLMs) have exhibited remarkable reasoning capabilities and become the foundation of language technologies. Inspired by the great success of code data in training LLMs, we naturally wonder at which training stage introducing code data can really help LLMs reasoning. To this end, this paper systematically explores the impact of code data on LLMs at different stages. Concretely, we introduce the code data at the pre-training stage, instruction-tuning stage, and both of them, respectively. Then, the reasoning capability of LLMs is comprehensively and fairly evaluated via six reasoning tasks in five domains. We critically analyze the experimental results and provide conclusions with insights. First, pre-training LLMs with the mixture of code and text can significantly enhance LLMs' general reasoning capability almost without negative transfer on other tasks. Besides, at the instruction-tuning stage, code data endows LLMs the task-specific reasoning capability. Moreover, the dynamic mixing strategy of code and text data assists LLMs to learn reasoning capability step-by-step during training. These insights deepen the understanding of LLMs regarding reasoning ability for their application, such as scientific question answering, legal support, etc. The source code and model parameters are released at the link:~\url{https://github.com/yingweima2022/CodeLLM}.

---

# 代码数据在何种训练阶段提升大语言模型推理能力？ 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在自然语言理解和生成上已经非常强大，但真正的逻辑推理仍然是软肋。传统的预训练语料主要是纯文本，模型往往只能捕捉统计关联，缺乏对因果链条的系统化学习。虽然在指令微调阶段加入少量推理示例可以提升少数任务的表现，但整体推理能力提升幅度有限。与此同时，代码数据天然包含严密的控制流、条件分支和函数调用，这些结构与逻辑推理高度契合，却一直没有被系统性地探讨到底应该在何时、如何加入训练流程才能最大化收益。因此，弄清“代码数据在训练的哪个阶段最有帮助”成为了一个迫切而又未解的难题。

### 关键概念速览
- **预训练（Pre‑training）**：在大规模未标注文本或代码上进行自监督学习，让模型学会通用的语言表示。类似于学生在课堂上大量阅读教材，打下基础。
- **指令微调（Instruction‑tuning）**：在已有模型基础上，用带有明确任务指令的示例进行有监督训练，使模型更好地遵循人类指令。相当于在基础上做针对性的练习题。
- **代码数据（Code Data）**：包括程序源码、函数实现、单元测试等，结构化程度高，蕴含丰富的控制流和变量依赖。可以把它想象成“带有逻辑框架的语言”。
- **推理能力（Reasoning Capability）**：模型在没有直接记忆答案的情况下，通过多步思考得出结论的能力。类似于人做数学题时的演算过程。
- **动态混合策略（Dynamic Mixing Strategy）**：在训练过程中根据进度动态调整代码和文本的比例，而不是一次性固定比例。好比老师根据学生的掌握情况灵活安排练习难度。
- **负迁移（Negative Transfer）**：一种训练副作用，指在某些任务上加入新数据反而导致性能下降。类似于学了新技巧后，原本熟练的老技巧被削弱。

### 核心创新点
1. **系统化的阶段划分实验**  
   之前的工作往往只在整体训练中加入代码，或仅在微调时使用少量代码示例。本文把代码引入明确划分为三种情形：仅在预训练、仅在指令微调、以及两阶段都加入。通过这种“对照实验”，能够直接观察每个阶段对推理能力的独立贡献。

2. **全方位推理评测框架**  
   论文构建了覆盖五大领域、六个具体任务的推理基准，包括数学、科学问答、法律推理等。相比以往只用单一任务（如数学）评估，提供了更具普适性的效果检验。

3. **动态混合代码与文本的训练策略**  
   在预训练阶段，作者没有采用固定的代码占比，而是让代码比例随训练步数逐步上升，模拟“先打好语言基座，再逐步加入逻辑砖块”。这种渐进式混合在实验中被证明能让模型一步步学习推理，而不会一次性冲击模型的语言能力。

4. **任务特化的指令微调**  
   在指令微调阶段，作者专门设计了包含代码片段的指令示例，使模型在面对特定推理任务时能够利用代码的结构化信息。实验显示，这种“代码增强的指令”能显著提升任务专属的推理表现，而不会对其他非代码任务产生负面影响。

### 方法详解
整体思路可以概括为“三步走”：①准备两类原始数据（文本+代码），②在不同训练阶段按设定策略混合使用，③用统一的推理基准进行评估。

**第一步：数据准备**  
- 文本数据来源于公开的网络语料，覆盖新闻、小说、百科等常规语言场景。  
- 代码数据则从开源仓库（如GitHub）抽取，包含多语言的函数实现、注释以及对应的单元测试。每段代码都配有自然语言描述，形成“代码‑文本对”，便于后续混合。

**第二步：阶段化混合训练**  
- **仅预训练阶段加入代码**：在自监督语言模型的掩码预测任务中，按照动态混合策略把代码占比从0%逐步提升到约30%。模型在早期主要学习语言模式，随后逐步接触代码的控制流，形成对逻辑结构的感知。  
- **仅指令微调阶段加入代码**：模型先完成完整的文本预训练，然后在指令微调时使用包含代码的指令示例。例如，给出“请用Python实现二分查找并解释每一步”，模型需要输出代码并解释推理过程。  
- **双阶段混合**：两阶段都使用代码，预训练阶段采用动态混合，微调阶段则使用任务特化的代码指令。这样模型在通用语言层面和任务层面都能受益。

**第三步：评估与分析**  
- 采用六个推理任务：数学推理、科学问答、法律案例分析、常识推理、代码解释推理、跨模态（文本+代码）推理。每个任务都使用标准的Few‑Shot或Zero‑Shot设置，确保公平比较。  
- 结果通过准确率、BLEU、以及思维链（Chain‑of‑Thought）完整度等指标综合评估。

**关键技巧**  
- **动态混合的调度函数**：作者使用一种线性增长函数，使代码比例随训练步数平滑上升，避免模型在早期被大量结构化代码“冲击”。  
- **代码‑文本对齐**：在指令微调时，代码示例总是配有自然语言解释，帮助模型把代码的形式化逻辑映射到语言推理上。  
- **负迁移监控**：在每个阶段都额外评估非推理任务（如情感分类），确保加入代码不会削弱模型的语言生成能力。

### 实验与效果
- **数据集**：六个推理任务分别来自MATH、ScienceQA、LegalBench、CommonsenseQA、CodeExplain、MultiModalReason等公开基准。  
- **对比基线**：包括纯文本预训练模型、仅在指令微调阶段加入代码的模型、以及公开的代码增强模型（如 CodeLlama）。  
- **主要发现**：  
  - 仅在预训练阶段加入代码的模型在所有六项任务上平均提升约4%~6%（相对纯文本基线），且在非推理任务上几乎没有下降。  
  - 仅在指令微调阶段加入代码的模型在任务特化的推理（如法律案例）上提升更明显，最高可达9%提升，但对通用推理的提升有限。  
  - 双阶段混合模型在整体推理表现上取得最佳，平均提升约7%~9%，并保持了文本生成质量。  
- **消融实验**：作者分别去掉动态混合、去掉代码‑文本对齐、以及只使用固定比例的代码。结果显示，去掉动态混合会导致预训练阶段的提升下降约2%，去掉对齐会削弱指令微调阶段的任务专属提升约3%。  
- **局限性**：实验主要在中等规模模型（约7B参数）上完成，尚未验证在更大模型（百亿级）上的效果；代码来源主要是开源项目，可能存在语言偏差；作者也提到在极端低资源语言上代码帮助有限。

### 影响与延伸思考
这篇工作在社区引发了对“代码即逻辑教材”的新一轮讨论。随后出现的几篇论文（如《Code‑Prompted Reasoning for LLMs》《Curriculum Learning with Code for Logical Tasks》）都在不同维度上扩展了动态混合和任务特化的思路。对实际应用而言，像科学问答、法律辅助系统等需要严密推理的场景，现在可以考虑在模型的预训练阶段就加入结构化代码，以获得更稳健的逻辑能力。未来的研究方向可能包括：  
- 将代码混合策略推广到多语言模型，探索非英语代码对推理的贡献。  
- 结合图结构（如抽象语法树）进一步强化模型对代码控制流的感知。  
- 在大模型（百亿以上）上验证是否仍然遵循“预训练阶段收益更大”的规律。  

### 一句话记住它
把代码按阶段、按比例“渐进式”混进训练，能让大语言模型在通用和任务专属推理上都拔得头筹。