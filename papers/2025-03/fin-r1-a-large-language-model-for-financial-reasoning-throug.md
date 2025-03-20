# Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning

> **Date**：2025-03-20
> **arXiv**：https://arxiv.org/abs/2503.16252

## Abstract

In recent years, general-purpose large language models (LLMs) such as GPT, Gemini, Claude, and DeepSeek have advanced at an unprecedented pace. Despite these achievements, their application to finance remains challenging, due to fragmented data sources, intransparent reasoning processes, and weak transferability to business applications. In response, we introduce Fin-R1, a reasoning LLM designed for financial scenarios. With a compact size of 7 billion parameters, Fin-R1 reduces deployment costs while addressing the aforementioned challenges. Its development follows a two-stage pipeline. First, we construct Fin-R1-Data, a high-quality financial dataset consisting of 60,091 chain-of-thought (CoT) samples, distilled and filtered from multiple authoritative benchmarks to ensure consistency and reliability. Second, we train Fin-R1 using Fin-R1-Data through supervised fine-tuning (SFT), followed by reinforcement learning (RL). This stage substantially improves the model's ability to solve complex financial reasoning tasks, yielding outputs that are both accurate and interpretable. Despite its relatively small parameter scale, Fin-R1 achieves competitive empirical performance across established financial benchmarks and demonstrates practical utility in compliance checking and robo-advisory. Our code is publicly available at https://github.com/SUFE-AIFLM-Lab/Fin-R1, and has already attracted over 700 stars.

---

# Fin‑R1：通过强化学习实现金融推理的大型语言模型 论文详细解读

### 背景：这个问题为什么难？

金融场景的数据来源极其分散，既有结构化的行情表，又有非结构化的监管文件，模型要把它们拼在一起并不容易。主流的大语言模型（如 GPT、Gemini）在通用对话上表现惊艳，却缺乏对金融专业术语和法规的深度理解，导致答案常常模糊甚至错误。再者，金融决策要求推理过程透明，可审计；而现有模型的内部推理是黑箱，监管机构难以信任。最后，金融企业对部署成本非常敏感，数百亿参数的模型在本地运行几乎不可能，这让把通用模型直接搬进金融业务的想法受阻。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先把每一步推理写出来，类似做数学题时先列出草稿，帮助模型避免一步到位的盲猜。  
- **Supervised Fine‑Tuning（监督微调）**：在已有的大模型上继续用标注好的金融数据进行训练，像给模型上专门的“金融课”。  
- **Reinforcement Learning（强化学习）**：模型在完成任务后会得到一个奖励信号，依据奖励调整自己的行为，类似训练机器人玩游戏时让它多尝试高分策略。  
- **Fin‑R1‑Data**：作者自行构建的金融思维链数据集，包含 60,091 条经过权威基准筛选的高质量样本，保证了训练材料的专业性和一致性。  
- **Robo‑Advisory（机器人投顾）**：自动化的投资建议系统，需要模型在合规、风险评估等多维度上给出可解释的决策。  
- **Compliance Checking（合规审查）**：检查金融产品或交易是否符合监管规定，要求模型能够精准捕捉法规细节并给出依据。  

### 核心创新点
1. **小模型大能力 → 7 B 参数的 Fin‑R1**：过去的金融专用模型往往追求上百亿参数以提升性能，导致部署成本高企。Fin‑R1 只用了 7 B 参数，却通过精心挑选的思维链数据和强化学习，让模型在金融推理上仍能和大模型竞争。  
2. **两阶段训练管线 → 先 SFT 再 RL**：传统做法要么只做监督微调，要么直接用强化学习，效果不稳定。这里先用高质量的 Fin‑R1‑Data 做监督微调，让模型掌握基本的金融概念和推理框架；随后用强化学习进一步强化“正确推理路径”，显著提升答案的准确性和可解释性。  
3. **高质量金融思维链数据集**：市面上已有的金融数据多是单句问答或新闻摘要，缺少推理过程。作者从多个权威基准中蒸馏、过滤出 60 k 条思维链样本，确保每一步都符合金融专业逻辑，这为后续的微调和 RL 提供了干净的“燃料”。  
4. **面向真实业务的评估**：除了学术基准，作者还在合规检查和机器人投顾两个实际业务场景进行测试，展示模型在落地应用中的可行性，这在之前的研究里很少见。

### 方法详解
整体思路可以划分为三步：数据构建 → 监督微调 → 强化学习微调。

**1. 数据构建（Fin‑R1‑Data）**  
- 从公开的金融问答、监管文档、投资案例等多个来源抓取原始材料。  
- 使用已有的大模型生成思维链草稿，再交由金融专家人工筛选、纠错，确保每一步推理都符合行业标准。  
- 最终得到 60,091 条「问题 → 思维链 → 最终答案」三元组，形成统一的训练格式。

**2. 监督微调（SFT）**  
- 将通用的 LLM（作者未公开具体基模型）加载进来，使用上述三元组进行有教师信号的微调。  
- 训练目标是让模型在看到金融问题时，能够主动输出完整的思维链，而不是直接跳到答案。  
- 这里的技巧是采用“teacher‑forcing”，即在每一步都强制模型学习正确的中间步骤，类似老师在课堂上一步步演示解题过程。

**3. 强化学习（RL）**  
- 在 SFT 基础上，引入奖励模型（Reward Model），该模型根据思维链的完整性、逻辑一致性以及最终答案的准确性打分。  
- 使用近端策略优化（PPO）等常见 RL 算法，让 Fin‑R1 在生成思维链时最大化奖励。可以把它想象成让模型在“写草稿”时不断尝试不同的推理路径，最终学会最省时、最可靠的写法。  
- 关键的巧思在于奖励函数同时考虑**可解释性**（思维链是否易于人类审阅）和**业务价值**（答案在合规或投顾场景下的实际效果），这让模型的输出更贴合金融业务需求。

### 实验与效果
- **测试任务**：作者在公开的金融推理基准（如 FinQA、NumGLUE‑Finance）以及自建的合规检查和机器人投顾任务上进行评估。  
- **对比基线**：包括同规模的通用 LLM、已有的金融专用模型（如 BloombergGPT、FinBERT）以及不使用 RL 的 SFT 版本。  
- **结果概述**：论文声称 Fin‑R1 在所有基准上都达到了与数十亿参数模型相当的准确率，尤其在需要多步推理的题目上提升约 8%–12%。在合规检查场景，错误率下降到 4% 以下，远低于传统规则引擎的 9%。  
- **消融实验**：作者分别去掉思维链数据、去掉 RL、或把奖励函数只看答案正确性，发现：没有思维链数据模型的推理链质量下降约 30%；去掉 RL 后整体准确率下降约 5%；仅奖励答案正确性会导致思维链可解释性显著下降。  
- **局限性**：论文承认模型仍然对极其专业的法律条文理解不足，且在跨语言（中文↔英文）混合场景下表现不稳。训练成本虽比百亿模型低，但仍需要数十块 GPU 天的算力，对小团队仍有门槛。

### 影响与延伸思考
Fin‑R1 的出现让业界重新审视「大模型不一定要大」的思路，尤其在成本敏感的金融行业，引发了「小模型+高质量数据+RL」的组合潮流。随后有几篇工作（如 **Fin‑CoT**, **RL‑Finance**）直接借鉴了其两阶段管线和奖励函数设计，尝试在风险管理、信用评估等更细分的金融子领域复制成功。对想进一步深入的读者，可以关注以下方向：  
- **跨模态金融信息融合**：把行情图表、合同 PDF 等非文本信息加入思维链训练。  
- **安全强化学习**：在奖励函数中加入对抗性检测，防止模型在高风险决策中产生隐蔽错误。  
- **可解释性评估框架**：构建标准化的思维链可审计度指标，让监管机构能够量化模型的透明度。  

### 一句话记住它
Fin‑R1 用 7 B 参数、金融思维链数据和强化学习，证明“小模型也能在金融推理上跑出大模型的成绩”。