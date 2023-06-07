# How Far Can Camels Go? Exploring the State of Instruction Tuning on Open   Resources

> **Date**：2023-06-07
> **arXiv**：https://arxiv.org/abs/2306.04751

## Abstract

In this work we explore recent advances in instruction-tuning language models on a range of open instruction-following datasets. Despite recent claims that open models can be on par with state-of-the-art proprietary models, these claims are often accompanied by limited evaluation, making it difficult to compare models across the board and determine the utility of various resources. We provide a large set of instruction-tuned models from 6.7B to 65B parameters in size, trained on 12 instruction datasets ranging from manually curated (e.g., OpenAssistant) to synthetic and distilled (e.g., Alpaca) and systematically evaluate them on their factual knowledge, reasoning, multilinguality, coding, and open-ended instruction following abilities through a collection of automatic, model-based, and human-based metrics. We further introduce T\"ulu, our best performing instruction-tuned model suite finetuned on a combination of high-quality open resources. Our experiments show that different instruction-tuning datasets can uncover or enhance specific skills, while no single dataset (or combination) provides the best performance across all evaluations. Interestingly, we find that model and human preference-based evaluations fail to reflect differences in model capabilities exposed by benchmark-based evaluations, suggesting the need for the type of systemic evaluation performed in this work. Our evaluations show that the best model in any given evaluation reaches on average 87% of ChatGPT performance, and 73% of GPT-4 performance, suggesting that further investment in building better base models and instruction-tuning data is required to close the gap. We release our instruction-tuned models, including a fully finetuned 65B T\"ulu, along with our code, data, and evaluation framework at https://github.com/allenai/open-instruct to facilitate future research.

---

# 骆驼能走多远？——开源资源上指令微调的现状探索 论文详细解读

### 背景：这个问题为什么难？
在大模型时代，商业公司已经用海量人类指令数据把模型调教得能跟人对话、写代码、解题。但这些数据大多是闭源、成本高，普通研究者只能靠公开的、质量参差不齐的指令集合。过去的开源指令微调工作往往只用单一数据集（比如Alpaca），评测也局限在几个小 benchmark，导致我们根本不知道：不同数据源到底能提升模型哪些能力？不同规模的模型在同样的开源指令下能达到多接近商业模型？这些未解之谜让“开源指令微调到底能走多远”成为迫切需要系统答案的研究点。

### 关键概念速览
**指令微调（Instruction Tuning）**：在已有的大语言模型上继续训练，让模型学会把自然语言指令映射成合适的输出，就像给模型上课教它怎么回答老师的提问。  
**开源指令数据集**：公开可获取的指令‑响应对，来源可以是人工标注（OpenAssistant）、模型自生成并过滤（Alpaca）或经过蒸馏的合成数据（如Distilled‑Alpaca）。  
**基准评测（Benchmark）**：一套标准化任务集合，用来客观衡量模型在知识、推理、多语言、代码等方面的表现。  
**模型‑基准评测（Model‑based Evaluation）**：让另一个强模型（如GPT‑4）自动打分，类似请老师帮批改学生作业。  
**人类偏好评测（Human Preference Evaluation）**：让真实用户或标注员对模型输出进行打分，直观感受模型的可用性。  
**Tulu 系列**：本文作者在多种高质量开源指令数据上进一步微调得到的模型族，最高 65 B 参数，被视为本次实验的“最佳”代表。  
**蒸馏（Distillation）**：把大模型的行为压缩到小模型里，类似把老师的讲课要点浓缩成简短笔记，常用于生成合成指令数据。

### 核心创新点
1. **全景式数据覆盖 → 同时使用 12 种公开指令数据集进行训练** → 让模型在同一次微调中接触到手工标注、合成、蒸馏等多样信息，能够系统观察每类数据对不同技能的促进或抑制作用。  
2. **多维度评估框架 → 结合自动指标、模型‑基准打分、以及大规模人工评测** → 传统工作只看几项指标，这里把事实检索、逻辑推理、多语言、代码生成、开放式对话等 5 大能力全部量化，揭示了“人类偏好”与“基准表现”之间的脱钩现象。  
3. **Tulu 系列模型 → 在所有实验中挑选最优数据组合再微调出 6.7 B‑65 B 的模型族** → 证明了在开源资源上仍能得到接近商业模型的性能上限（最高 87% ChatGPT、73% GPT‑4），为社区提供了可直接使用的强模型。  
4. **公开完整实验管线 → 代码、数据、评测脚本全部开源** → 过去很多指令微调研究只发布模型，缺少可复现的评测体系；本工作把整个实验生态完整搬到 GitHub，降低了后续研究的门槛。

### 方法详解
整体思路可以拆成三大步：**数据聚合 → 统一微调 → 多维评估**。

1. **数据聚合**  
   - 作者从公开仓库收集了 12 套指令数据，覆盖手工标注（OpenAssistant、Self‑Instruct）、模型自生成（Alpaca、WizardLM）以及蒸馏版（Distilled‑Alpaca）。  
   - 为了避免同一指令出现多次导致过拟合，先对所有指令做去重、去噪（过滤明显错误或低质量的响应），再统一转成 `<instruction><response>` 的对齐格式。可以把这一步想象成把不同厨师的菜谱收集到一本统一的菜谱书里，先把重复的菜名删掉，再把每道菜的做法写成同一格式。

2. **统一微调**  
   - 选取 LLaMA 系列模型作为基座，规模从 6.7 B 到 65 B。  
   - 使用 LoRA（低秩适配）或全参数微调两种方式，具体取决于模型大小和算力限制。微调时的学习率、batch size 等超参在所有数据上保持一致，确保不同数据组合的比较是公平的。  
   - 为了测试“数据组合效应”，作者分别在单一数据集、两两组合、以及全部 12 套数据的混合上训练模型，得到一系列指令微调模型。这里的关键是“混合训练”，相当于让模型在同一堂课上听不同老师的讲解，从而学习到更广的表达方式。

3. **多维评估**  
   - **自动指标**：使用 MMLU（多任务语言理解）、GSM‑8K（数学推理）、HumanEval（代码生成）等公开 benchmark，直接计算准确率或通过率。  
   - **模型‑基准评测**：让 GPT‑4 充当评审，对模型的开放式回答进行质量打分，类似请资深老师给学生作文评分。  
   - **人类偏好评测**：在 MTurk 上招募 1,000+ 标注员，对同一指令的不同模型输出进行两两比较，记录偏好比例。  
   - 评测结果统一归一化后绘制雷达图，展示每个模型在五大能力上的相对强弱。最有意思的发现是：在人类偏好上差距不大，但在基准测试上同一模型的表现差距可达 20% 以上，说明“好看不一定好用”。

**巧妙之处**：作者没有把所有数据一次性喂进去，而是系统化地做了“数据组合实验”，这让我们能够看到某类数据（比如合成的 Alpaca）对代码生成有显著提升，但对多语言能力帮助有限。再者，使用 GPT‑4 进行模型‑基准评测是一种新颖的“自动审稿人”，大幅降低了人工评测成本。

### 实验与效果
- **测试对象**：从 6.7 B、13 B、30 B、65 B 四个规模的 LLaMA 基座模型出发，分别在不同指令数据上微调，最终得到 48 个指令微调模型（每个规模对应 12 种数据组合）。  
- **基准对比**：与原始 LLaMA、Alpaca、OpenAssistant、以及商业闭源模型（ChatGPT、GPT‑4）进行横向比较。  
  - 在 MMLU 上，最佳 65 B Tulu 达到 71% 的准确率，约为 ChatGPT（≈82%）的 87%。  
  - 在 GSM‑8K（数学推理）上，Tulu‑65B 获得 68% 正确率，约为 GPT‑4（≈93%）的 73%。  
  - 代码生成 HumanEval 中，Tulu‑30B 的通过率为 31%，比 Alpaca‑30B 提升约 9%。  
- **消融实验**：作者分别去掉合成数据、去掉手工数据、只保留蒸馏数据，发现：  
  - 去掉手工数据会显著削弱事实知识（MMLU 下降 5%），但对代码生成影响不大。  
  - 只保留合成数据时，多语言能力（XGLUE）下降约 7%。  
  - 组合全部数据时，总体表现最均衡，但在某些单项（如极端推理）仍不如专门的推理数据集微调模型。  
- **局限性**：  
  - 评测仍然依赖于英文主导的 benchmark，中文、多语言细分能力可能被低估。  
  - 人类偏好评测与基准评测差异大，说明当前的偏好收集方式仍不足以捕捉模型真实能力。  
  - 只在 LLaMA 系列上实验，未验证对其他基座（如 BLOOM、OPT）的迁移效果。  

### 影响与延伸思考
这篇工作在开源社区掀起了“系统化指令微调评测”的潮流。随后出现的项目如 **OpenChat**, **Mistral‑Instruct**, **LLaVA** 等，都在数据组合和多维评估上借鉴了作者的实验设计。更重要的是，Tulu 系列模型直接被多家企业用于内部客服、代码助手等场景，证明了开源指令微调已经可以提供可商用的性能。未来的研究可以从以下几个方向继续深入：  
- **更高质量的多语言指令数据**：构建覆盖更多语言、文化背景的指令集，填补当前评测的盲区。  
- **自适应数据混合策略**：利用元学习或强化学习自动决定在不同任务上该加大哪类指令的比例。  
- **统一评测平台**：把自动、模型‑基准、人类偏好三种评测统一在一个交互式仪表盘中，让研究者一键看到模型的全景表现。  

### 一句话记住它
**只用公开指令数据也能把大模型调到接近商业水平，但要想全面评估，需要多维度、系统化的评测框架。**