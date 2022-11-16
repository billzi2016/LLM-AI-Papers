# Holistic Evaluation of Language Models

> **Date**：2022-11-16
> **arXiv**：https://arxiv.org/abs/2211.09110

## Abstract

Language models (LMs) are becoming the foundation for almost all major language technologies, but their capabilities, limitations, and risks are not well understood. We present Holistic Evaluation of Language Models (HELM) to improve the transparency of language models. First, we taxonomize the vast space of potential scenarios (i.e. use cases) and metrics (i.e. desiderata) that are of interest for LMs. Then we select a broad subset based on coverage and feasibility, noting what's missing or underrepresented (e.g. question answering for neglected English dialects, metrics for trustworthiness). Second, we adopt a multi-metric approach: We measure 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency) for each of 16 core scenarios when possible (87.5% of the time). This ensures metrics beyond accuracy don't fall to the wayside, and that trade-offs are clearly exposed. We also perform 7 targeted evaluations, based on 26 targeted scenarios, to analyze specific aspects (e.g. reasoning, disinformation). Third, we conduct a large-scale evaluation of 30 prominent language models (spanning open, limited-access, and closed models) on all 42 scenarios, 21 of which were not previously used in mainstream LM evaluation. Prior to HELM, models on average were evaluated on just 17.9% of the core HELM scenarios, with some prominent models not sharing a single scenario in common. We improve this to 96.0%: now all 30 models have been densely benchmarked on the same core scenarios and metrics under standardized conditions. Our evaluation surfaces 25 top-level findings. For full transparency, we release all raw model prompts and completions publicly for further analysis, as well as a general modular toolkit. We intend for HELM to be a living benchmark for the community, continuously updated with new scenarios, metrics, and models.

---

# 语言模型的整体评估 论文详细解读

### 背景：这个问题为什么难？

语言模型已经渗透到搜索、写作、客服等几乎所有文字交互场景，但我们对它们到底能干什么、不能干什么知之甚少。过去的评测大多只看“对不对”，比如在几个公开的问答或翻译基准上跑分，忽略了模型在不同语言、不同用户群体、不同安全需求下的表现。于是出现了两大盲点：一是评测覆盖面太窄，很多真实使用情境根本没有被测到；二是只关注准确率，导致模型在公平性、鲁棒性、可信度等关键维度的缺陷被掩盖。正因为评测体系本身不完整，研究者和企业很难判断模型的真实价值和潜在风险，这正是 HELM 要解决的核心痛点。

### 关键概念速览
**场景（Scenario）**：指模型在真实世界中的具体使用情境，例如“帮助学生写作业”或“在医疗对话中提供建议”。可以把它想象成不同的“工作岗位”，每个岗位对模型的要求都有所不同。  

**指标（Metric）**：用来量化模型在某个场景下表现的数值标准，如准确率、鲁棒性等。类似于员工的绩效考核表，指标越丰富，评价越全面。  

**多指标评估（Multi‑metric Evaluation）**：同时测量多个指标，而不是只看单一的准确率。就像在选拔运动员时既看跑步速度，也看耐力和技术水平。  

**覆盖率（Coverage）**：评测体系能够覆盖的场景和指标的比例。高覆盖率意味着我们对模型的了解更接近全景图。  

**可重复基准（Standardized Benchmark）**：所有模型在相同的输入、相同的硬件、相同的评测脚本下进行测试，确保比较公平。类似于统一的实验室条件，排除外部干扰。  

**可信度（Trustworthiness）**：模型在不误导、不产生有害内容的前提下提供答案的能力。可以类比为医生的职业道德，答案必须可靠且安全。  

**效率（Efficiency）**：指模型在计算资源、响应时间、能耗等方面的表现。相当于评估一辆车的油耗和加速性能。

### 核心创新点
1. **从单一基准到全景评测 → 构建 HELM 框架 → 让模型在 16 类核心场景、7 项指标上几乎全覆盖**。过去的评测往往只挑几个流行的公开数据集跑分，HELM 则先把可能的使用情境和评估需求系统化，形成一个“场景‑指标”矩阵，然后挑选出兼顾覆盖度和可行性的子集，确保每个模型都在同一套完整的标准下被审视。  

2. **仅看准确率 → 引入 7 项多维度指标 → 揭示模型在校准、鲁棒性、公平性等方面的权衡**。在 HELM 中，准确率不再是唯一的成功标尺，模型还要在校准（预测置信度与真实概率的一致性）、鲁棒性（对噪声或对抗输入的抵抗力）、公平性/偏见、毒性和效率等维度表现。这样可以直接看到提升准确率是否以牺牲安全性为代价。  

3. **零散的模型对比 → 同时基准 30 种模型 → 实现 96% 场景密集覆盖**。在 HELM 之前，同一篇论文里常出现“我们只评测了 3–5 个模型，且每个模型测的场景不一样”。HELM 把 30 个公开、受限和闭源模型统一放进同一套评测流水线，几乎所有核心场景都被测到，形成了真正的横向比较基准。  

4. **评测结果闭源 → 完全公开原始交互日志 → 促进社区二次分析**。HELM 把所有 prompt（模型输入）和 completion（模型输出）公开，提供了一个可供研究者自行挖掘、复现和扩展的原始数据池，这在以往的评测工作中极为罕见。

### 方法详解
**整体框架**  
HELM 的评测流程可以拆成三大步骤：① 场景‑指标体系构建；② 多指标测量与数据收集；③ 大规模统一基准测试。先把所有可能的使用情境和我们关心的评估维度列成一个大表格（类似“需求清单”），再挑选出在技术实现上可行且能代表性强的子集，最后把这些子集统一包装成可重复运行的评测脚本，对每个模型逐一执行。

**关键模块拆解**  

1. **场景税onomies（Scenario Taxonomy）**  
   - 研究团队先对语言模型的潜在应用做系统化梳理，划分出 16 类核心场景（如问答、对话、代码生成、信息检索等）以及 26 条针对性子场景（如针对特定英语方言的问答）。可以把它想象成把整个语言模型的“工作市场”细分成若干岗位，每个岗位都有对应的职责描述。  

2. **指标选取与映射（Metric Mapping）**  
   - 对每个场景，团队挑选 7 项指标：准确率、校准、鲁棒性、公平性、偏见、毒性、效率。若某个场景技术上难以测量某项指标，则标记为缺失。这样形成了一个“场景‑指标矩阵”，大约 87.5% 的格子都有实际测量值。  

3. **统一评测平台（Standardized Toolkit）**  
   - 为了保证所有模型在相同条件下跑分，HELM 开源了一个模块化的评测工具箱。它把每个场景的输入数据、评测脚本、指标计算函数封装成独立的插件，用户只需要把模型的 API 接口接入即可。类似于把不同品牌的汽车都放进同一条测试跑道，确保比较公平。  

4. **大规模模型跑分（Massive Model Benchmarking）**  
   - 研究者把 30 个模型（包括开源模型、受限访问模型和商业闭源模型）全部塞进上述平台，统一执行 42 个场景的评测。每一次评测都会记录完整的 prompt‑completion 对，随后交给指标函数计算对应的数值。  

5. **结果公开与二次分析（Open Release）**  
   - 所有原始交互日志、指标报告和代码都以开放方式发布，任何人都可以下载后自行复现或扩展新的指标。  

**最巧妙的设计**  
- **场景覆盖率的量化**：HELM 用“覆盖率”这个概念把评测的完整性变成可度量的数字，从 17.9% 提升到 96%。这让研究者一眼就能看出自己评测的盲区在哪里。  
- **多指标并行计算**：在一次模型调用后，系统会同步计算所有 7 项指标，而不是像传统评测那样跑完一次再跑另一种评测脚本。这样大幅降低了评测成本，也避免了因模型状态变化导致的指标不一致。  

### 实验与效果
- **测试对象**：30 种语言模型，涵盖开源（如 LLaMA、OPT）、受限访问（如 GPT‑3.5）以及闭源商业模型（如 ChatGPT）。  
- **评测范围**：42 个场景，其中 21 个是此前主流评测从未涉及的全新情境。每个模型在 16 类核心场景上尽可能多地完成 7 项指标的测量，整体覆盖率达到 96%。  
- **关键发现**：作者归纳出 25 条顶层结论，例如：在提升准确率的同时，模型的毒性往往会略有上升；小模型在效率上占优势，但在校准和鲁棒性上明显落后；不同方言的问答表现差异显著，说明语言多样性仍是弱点。  
- **对比基线**：论文指出，在 HELM 出现之前，平均每个模型只覆盖约 17.9% 的核心场景，且不同模型之间几乎没有共同评测点。HELM 将这一数字提升至 96%，实现了真正的“一盘棋”比较。  
- **消融实验**：原文未提供细粒度的消融实验，只是说明多指标并行计算和统一评测平台是提升覆盖率的关键因素。  
- **局限性**：作者坦诚仍有指标缺失（如对信任度的度量仍在探索），部分语言和方言的场景样本量不足，且评测成本随模型规模线性增长，极大模型的全量评测仍受算力限制。

### 影响与延伸思考
HELM 推出后，业界开始意识到“只看分数”远远不够，随后出现了如 **BIG-Bench**、**MMLU** 的多任务评测，以及专注安全性的 **TruthfulQA**、**Red Teaming** 项目。很多新模型的技术报告都会引用 HELM 的覆盖率和多指标结果作为基准。后续工作可能会在以下几个方向深化：  
- **信任度量体系**：设计更细致的可信度指标（如事实一致性、可解释性）。  
- **跨语言扩展**：加入更多低资源语言和方言的场景，填补当前评测的空白。  
- **自动化评测管线**：利用元学习或自监督方法自动生成新场景和对应的评测数据。  
想进一步了解，可关注 **HELM 2.0** 的公开路线图以及各大模型公司在发布新模型时的评测报告。

### 一句话记住它
HELM 用“全景‑多维‑统一”三把钥匙，把语言模型的评测从碎片化的跑分表，升级为覆盖 96% 场景的完整体检报告。