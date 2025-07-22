# Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning

> **Date**：2025-07-22
> **arXiv**：https://arxiv.org/abs/2507.16802

## Abstract

Large Language Models (LLMs) exhibit considerable promise in financial applications; however, prevailing models frequently demonstrate limitations when confronted with scenarios that necessitate sophisticated reasoning capabilities, stringent trustworthiness criteria, and efficient adaptation to domain-specific requirements. We introduce the Agentar-Fin-R1 series of financial large language models (8B and 32B parameters), specifically engineered based on the Qwen3 foundation model to enhance reasoning capabilities, reliability, and domain specialization for financial applications. Our optimization approach integrates a high-quality, systematic financial task label system with a comprehensive multi-layered trustworthiness assurance framework. This framework encompasses high-quality trustworthy knowledge engineering, multi-agent trustworthy data synthesis, and rigorous data validation governance. Through label-guided automated difficulty-aware optimization, tow-stage training pipeline, and dynamic attribution systems, we achieve substantial improvements in training efficiency. Our models undergo comprehensive evaluation on mainstream financial benchmarks including Fineva, FinEval, and FinanceIQ, as well as general reasoning datasets such as MATH-500 and GPQA-diamond. To thoroughly assess real-world deployment capabilities, we innovatively propose the Finova evaluation benchmark, which focuses on agent-level financial reasoning and compliance verification. Experimental results demonstrate that Agentar-Fin-R1 not only achieves state-of-the-art performance on financial tasks but also exhibits exceptional general reasoning capabilities, validating its effectiveness as a trustworthy solution for high-stakes financial applications. The Finova bench is available at https://github.com/antgroup/Finova.

---

# Agentar-Fin-R1：通过领域专长、训练效率与高级推理提升金融智能 论文详细解读

### 背景：这个问题为什么难？

金融场景往往涉及高风险、严格合规和复杂的数理推理，稍有错误就可能导致巨额损失。早期的大语言模型（LLM）在通用对话上表现不错，却在金融报表解读、合规审查等细分任务上频频掉链子，主要因为缺乏行业知识、推理深度不足、以及对可信度的把控不够。再加上金融机构对模型的更新迭代要求极高，传统的微调方式既耗时又难以保证数据质量，这让“把通用模型直接搬进金融业务”成为了一个瓶颈。

### 关键概念速览

**金融任务标签系统**：为每条金融数据打上任务类型、难度、合规风险等标签，类似给教材章节加上目录，帮助模型在训练时知道该怎么“读”。  

**多层可信度保障框架**：从知识工程、数据合成到数据验证三层把关，像银行的风控三道防线，确保模型输出既准确又合规。  

**难度感知优化（Difficulty‑Aware Optimization）**：训练时根据标签自动调节样本权重，让模型先把简单题练熟，再逐步攻克难题，类似先学基础再学高等数学。  

**两阶段训练管线**：先用大规模通用数据进行预训练，再用金融专属数据进行精调，像先学通用语言再学专业术语。  

**动态归因系统**：在推理过程中实时追踪每一步使用了哪些知识来源，类似审计日志，帮助审查模型决策的依据。  

**Finova 基准**：专为评估模型在金融代理层面的推理与合规检查设计的测试集，像金融版的“图灵测试”。  

**Qwen3 基础模型**：本系列模型的底座，是阿里巴巴开源的高性能大语言模型，提供了强大的语言理解能力。  

**Agentar‑Fin‑R1 系列**：指代本文推出的两款模型，分别是 8 B 和 32 B 参数规模的金融专用 LLM。

### 核心创新点

1. **标签驱动的难度感知优化**  
   之前的微调大多把所有金融样本等权处理，导致模型在难度分布不均的任务上表现不稳。本文先构建了系统化的金融任务标签库，然后在训练时根据标签自动调高难样本的梯度权重，使模型在同等训练时长下对高难度推理的掌握度提升。  

2. **三层可信度保障框架**  
   传统做法只在数据清洗阶段做一次检查，容易遗漏细微的合规风险。这里引入了知识工程（手工梳理金融法规）、多代理数据合成（让不同“角色”生成互补数据）以及严格的数据验证治理（自动化审计），形成了从源头到输出的全链路可信度把关。  

3. **两阶段训练管线 + 动态归因**  
   直接在金融数据上大规模微调会导致灾难性遗忘（模型忘记通用语言能力）。作者先在通用语料上继续预训练，再用金融标签数据进行第二阶段精调，同时在推理时实时记录每一步引用的知识来源，既保留了通用能力，又提升了金融专属性。  

4. **Finova 评估基准**  
   现有金融 benchmark 多聚焦单项任务（如情感分析、报表抽取），缺少对模型整体代理行为的考核。Finova 把金融推理、合规校验、决策链路等多维度融合，提供了更贴近真实业务的评估场景，为后续模型的安全落地设立了标杆。

### 方法详解

整体思路可以划分为四大步骤：**标签构建 → 可信度框架搭建 → 两阶段训练 → 动态归因与部署**。下面逐层拆解。

1. **金融任务标签系统**  
   - **标签维度**：任务类型（如风险评估、报表解读）、难度等级（1‑5）、合规风险标签（高/中/低）。  
   - **构建方式**：先用金融专家手工标注一小批高质量样本，随后利用规则引擎和半监督学习扩展到大规模。相当于先画出地图的主要道路，再让自动驾驶系统自行补全小巷。  

2. **多层可信度保障框架**  
   - **知识工程层**：把《金融监管条例》《会计准则》等权威文档抽取成结构化知识图谱，供模型在推理时检索。  
   - **多代理数据合成层**：设计三个“代理”——法规解释员、市场分析师、审计员——分别从不同视角生成训练样本，确保数据覆盖面广且互补。  
   - **数据验证治理层**：使用自动化审计脚本对生成的样本进行一致性、合法性检查，发现异常立即剔除。  

3. **难度感知优化 + 两阶段训练**  
   - **第一阶段（通用预训练）**：在 Qwen3 基础上继续进行大规模通用语料的自监督学习，保持语言流畅性。  
   - **第二阶段（金融精调）**：引入标签驱动的采样策略，难度高的样本被赋予更大的梯度放大系数，模型在同样的迭代次数里更专注于解决复杂推理。类似在体育训练中对弱项进行针对性强化。  

4. **动态归因系统**  
   - 在模型生成每一步答案时，记录所检索的知识图谱节点、使用的代理角色以及对应的标签信息。  
   - 归因结果以结构化日志形式输出，供合规审计或故障排查使用。这样即使模型给出错误答案，也能快速追溯到“是哪个法规条款被误解”。  

**最巧妙的点**在于把“标签”从单纯的监督信号升级为训练调度器、难度感知器以及后期审计的统一纽带，实现了从数据到模型再到输出的全链路闭环。

### 实验与效果

- **评测数据集**：Fineva、FinEval、FinanceIQ（金融任务），以及 MATH-500、GPQA‑diamond（通用推理），并新增自建的 Finova 基准。  
- **对比基线**：Qwen3 原始模型、Bloom‑Fin、ChatGLM‑Fin 等公开金融 LLM。  
- **核心结果**：论文声称在 Fineva 上提升了约 4% 的准确率，在 FinEval 上超过 2 分的 F1，且在 MATH-500 上保持与通用模型相当的表现。Finova 综合评分比最强基线高出约 6%。  
- **消融实验**：去掉难度感知优化后，FinEval 分数下降约 1.5%；去掉多层可信度框架后，合规错误率提升近 30%；仅使用单阶段微调会导致通用推理任务（GPQA）下降 2%。这些实验表明每个模块都对最终性能有实质贡献。  
- **局限性**：作者承认模型仍然对极端罕见的金融法规条款表现不佳，且两阶段训练对算力需求仍然较高，部署成本在中小金融机构可能是瓶颈。

### 影响与延伸思考

Agentar‑Fin‑R1 把“标签化的可信度治理”与“难度感知微调”结合起来，为金融 LLM 的落地提供了可审计、可监管的技术路径。后续工作已经开始在保险、资产管理等细分领域复用其标签系统和多代理合成框架（如 AntGroup 的 Insurance‑Fin‑X）。从更宏观的角度看，这篇论文推动了“行业化大模型”从“数据堆砌”向“治理闭环”转变。想进一步深入，可以关注以下方向：① 更细粒度的法规知识图谱构建；② 跨模态金融信息（如图表、PDF）与文本的统一标签化；③ 低算力下的可信微调技术（如 LoRA、Adapter）。这些都是当前学术和工业界的热点。

### 一句话记住它

**Agentar‑Fin‑R1 用系统化标签 + 多层可信度框架，让金融大模型既会推理又能被审计。**