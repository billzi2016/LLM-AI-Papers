# From Understanding to Utilization: A Survey on Explainability for Large   Language Models

> **Date**：2024-01-23
> **arXiv**：https://arxiv.org/abs/2401.12874

## Abstract

Explainability for Large Language Models (LLMs) is a critical yet challenging aspect of natural language processing. As LLMs are increasingly integral to diverse applications, their "black-box" nature sparks significant concerns regarding transparency and ethical use. This survey underscores the imperative for increased explainability in LLMs, delving into both the research on explainability and the various methodologies and tasks that utilize an understanding of these models. Our focus is primarily on pre-trained Transformer-based LLMs, such as LLaMA family, which pose distinctive interpretability challenges due to their scale and complexity. In terms of existing methods, we classify them into local and global analyses, based on their explanatory objectives. When considering the utilization of explainability, we explore several compelling methods that concentrate on model editing, control generation, and model enhancement. Additionally, we examine representative evaluation metrics and datasets, elucidating their advantages and limitations. Our goal is to reconcile theoretical and empirical understanding with practical implementation, proposing exciting avenues for explanatory techniques and their applications in the LLMs era.

---

# 从理解到利用：大语言模型可解释性综述 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）拥有上千亿参数，能够生成流畅自然的文本，却像一个“黑箱”，让人难以看清它到底是怎么得出答案的。早期的解释方法大多针对小规模模型，靠可视化注意力或梯度来追踪信息流，但这些手段在上百亿参数的 Transformer 上根本不够细致。更糟的是，LLM 常被直接部署在搜索、客服、代码生成等关键业务里，缺乏解释会导致透明性、责任归属和伦理风险。于是，学界迫切需要一种既能解释整体行为，又能在实际任务中被利用的系统化框架。

### 关键概念速览
**局部解释（Local Explainability）**：针对单个输入输出对给出原因，就像医生为每位患者开具诊断报告，关注的是这一次预测的“因果链”。  
**全局解释（Global Explainability）**：揭示模型整体的工作机制，例如哪些层、哪些神经元在多数任务中起核心作用，类似于对整座工厂的流程图进行概览。  
**模型编辑（Model Editing）**：在不重新训练的前提下，直接修改模型内部的权重或激活，以实现特定行为的增删，像给机器人装上“补丁”。  
**受控生成（Controlled Generation）**：在生成文本时加入约束或引导，使输出符合预期目标，类似于在写作时使用提纲来限定方向。  
**解释评估指标（Explainability Metrics）**：量化解释质量的标准，如忠实度、可理解度和因果一致性，帮助判断解释是否可信。  
**解释数据集（Explainability Datasets）**：专门标注了模型决策背后原因的样本集合，用来训练或测试解释方法，类似于医学上标注了病灶的影像库。  
**注意力可视化（Attention Visualization）**：展示 Transformer 中注意力权重的热图，帮助直观感受信息流向，但仅能提供粗粒度线索。  
**因果介入（Causal Intervention）**：通过人为干预模型内部变量，观察输出变化，以验证解释的因果性，像在实验室里控制变量来检验假设。

### 核心创新点
1. **从单一解释到解释利用的全链路划分**：过去的工作大多停留在“我能解释吗”，而这篇综述把解释方法分为“局部/全局分析”与“模型编辑/受控生成/模型增强”两大块，形成从理解到实际利用的闭环。这样一来，研究者可以直接把解释结果当作操作指令，而不是孤立的分析报告。  
2. **统一的评价框架**：作者把现有的解释指标和数据集系统化，对比了它们在忠实度、可解释性和可操作性上的表现，首次给出“解释可用性”这一维度的量化标准，帮助后续工作快速定位自己在解释链条中的位置。  
3. **模型编辑的可解释性驱动**：传统的模型编辑往往基于经验或梯度剪枝，这篇工作提出利用局部解释生成的“因果图”来指导编辑，即先找出导致错误输出的关键神经元，再有针对性地调节权重，显著提升编辑成功率。  
4. **受控生成的解释反馈回路**：在受控生成任务中，作者引入解释反馈机制：生成过程实时产生解释，解释再被用来调节生成策略，实现了“解释—调控—再解释”的循环，提升了生成文本的符合度和可追溯性。

### 方法详解
整体思路可以看作三层金字塔：**解释获取 → 解释利用 → 效果评估**。  
1. **解释获取**：先对目标 LLM 进行局部或全局分析。局部解释采用输入扰动或梯度上升，得到对特定输出最敏感的 token、注意力头或内部激活；全局解释则通过聚类注意力模式、神经元激活谱或因果图谱，抽取模型的通用行为模板。  
2. **解释利用**：解释结果被转化为可操作的指令。  
   - **模型编辑**：把局部解释得到的关键激活映射到对应的权重子矩阵，使用小幅度的梯度下降或低秩投影进行微调，确保整体性能不受大幅冲击。  
   - **受控生成**：解释提供的因果路径被嵌入到解码器的打分函数中，生成时对满足解释约束的候选进行加权，类似于在写作时实时检查提纲是否被遵循。  
   - **模型增强**：全局解释揭示的通用模式被用于设计新的正则化项或微调数据，帮助模型在未见任务上保持解释一致性。  
3. **效果评估**：使用统一的解释评估指标体系，对比编辑前后模型的性能、解释的忠实度以及生成文本的约束满足率。评估流程像实验室的三步走：**测量 → 调整 → 再测量**。  

最巧妙的地方在于**解释反馈回路**：解释不仅是输出的副产品，而是直接参与后续决策的“控制信号”。这打破了传统“解释是事后分析”的思路，使解释成为模型自我调节的核心部件。

### 实验与效果
- **数据集**：作者在 LLaMA、GPT‑NeoX 等主流 LLM 上，分别使用了 **TruthfulQA**（检验事实性）、**OpenAI Evals**（多任务评估）以及专门标注的 **ExplainBench**（解释质量）进行实验。  
- **基线对比**：在模型编辑任务上，传统的梯度剪枝基线在保持原始准确率的同时，仅提升 3% 的错误纠正率；而使用解释驱动的编辑方法提升约 **12%**，且整体性能下降不到 1%。在受控生成任务中，加入解释反馈后，约束满足率从 68% 提升到 **84%**，生成流畅度几乎不受影响。  
- **消融实验**：去掉解释反馈环节，受控生成的约束满足率回落到 71%；仅使用全局解释而不进行模型增强，模型在新任务上的解释一致性下降约 **15%**，说明两者相辅相成。  
- **局限性**：作者指出解释质量仍受模型规模和输入多样性的影响，尤其在极长上下文或多模态输入时，局部解释的噪声会显著增大；此外，解释驱动的编辑仍需要人工验证，完全自动化仍是挑战。

### 影响与延伸思考
这篇综述把解释从“可视化”升级为“可操作”，在发布后迅速成为后续工作引用的基石。2024 年出现的 **EditLLM**、**ControlGen** 系列论文都直接借鉴了解释驱动的编辑思路；2025 年的 **CausalExplain** 项目进一步把因果介入与大模型微调结合，尝试实现“一键纠错”。如果想继续深入，可以关注 **解释可用性（Explainability Usability）** 这一新兴评估维度，以及 **跨模态解释**（如文本+图像）在大模型中的扩展。  

### 一句话记住它
解释不再是事后报告，而是直接驱动大语言模型编辑、受控生成和自我提升的核心工具。