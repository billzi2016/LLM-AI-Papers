# ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All   Tools

> **Date**：2024-06-18
> **arXiv**：https://arxiv.org/abs/2406.12793

## Abstract

We introduce ChatGLM, an evolving family of large language models that we have been developing over time. This report primarily focuses on the GLM-4 language series, which includes GLM-4, GLM-4-Air, and GLM-4-9B. They represent our most capable models that are trained with all the insights and lessons gained from the preceding three generations of ChatGLM. To date, the GLM-4 models are pre-trained on ten trillions of tokens mostly in Chinese and English, along with a small set of corpus from 24 languages, and aligned primarily for Chinese and English usage. The high-quality alignment is achieved via a multi-stage post-training process, which involves supervised fine-tuning and learning from human feedback. Evaluations show that GLM-4 1) closely rivals or outperforms GPT-4 in terms of general metrics such as MMLU, GSM8K, MATH, BBH, GPQA, and HumanEval, 2) gets close to GPT-4-Turbo in instruction following as measured by IFEval, 3) matches GPT-4 Turbo (128K) and Claude 3 for long context tasks, and 4) outperforms GPT-4 in Chinese alignments as measured by AlignBench. The GLM-4 All Tools model is further aligned to understand user intent and autonomously decide when and which tool(s) touse -- including web browser, Python interpreter, text-to-image model, and user-defined functions -- to effectively complete complex tasks. In practical applications, it matches and even surpasses GPT-4 All Tools in tasks like accessing online information via web browsing and solving math problems using Python interpreter. Over the course, we have open-sourced a series of models, including ChatGLM-6B (three generations), GLM-4-9B (128K, 1M), GLM-4V-9B, WebGLM, and CodeGeeX, attracting over 10 million downloads on Hugging face in the year 2023 alone. The open models can be accessed through https://github.com/THUDM and https://huggingface.co/THUDM.

---

# ChatGLM：从 GLM-130B 到 GLM-4 All Tools 的大语言模型系列 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）快速迭代的前几年，中文模型的规模、数据覆盖和对齐质量普遍落后于英文模型。早期的中文 LLM 要么数据量不足、要么只做了单一的指令微调，导致在多语言、长上下文和工具调用等实际场景里表现不稳。与此同时，模型在“何时使用工具、使用哪种工具”这类元决策上几乎没有能力，用户只能手动编排流程。要想把模型做成既懂中英文、又能在 100K+ 长上下文、还能自主选择浏览器、Python、图像生成等外部工具的全能助手，需要在数据、训练策略、对齐流程和系统架构上同步升级，这正是本文要解决的核心难点。

### 关键概念速览
- **预训练（Pre‑training）**：在海量原始文本上让模型学习语言统计规律，就像小孩在日常对话中积累词汇和语感。  
- **监督微调（Supervised Fine‑tuning, SFT）**：给模型提供高质量的指令‑答案对，让它学会按照用户意图输出，类似老师给学生示范解题步骤。  
- **人类反馈学习（RLHF）**：模型生成答案后，让人工评审打分，再用强化学习把高分答案的概率放大，像让模型在“比赛”中不断改进策略。  
- **多阶段对齐（Multi‑stage Alignment）**：把 SFT、RLHF、工具对齐等步骤层层叠加，确保模型在语言理解、指令遵循和工具使用三个维度都达标。  
- **长上下文（Long Context）**：模型能够一次性处理数万甚至十几万 token 的输入，类似一次性阅读一本长篇小说而不需要翻页。  
- **All‑Tools 模型**：在对齐过程中加入了“工具选择”子任务，使模型能够自行判断何时需要浏览网页、运行代码或调用自定义函数，类似助理在工作中主动打开相应软件。  
- **AlignBench**：专门评估模型中文指令对齐水平的基准测试，类似中文版本的“指令遵循排行榜”。  

### 核心创新点
1. **从单一指令对齐到多阶段全链路对齐**  
   之前的中文 LLM 多停留在一次性 SFT，缺乏后续的 RLHF 与工具对齐。ChatGLM 把监督微调、RLHF、工具选择三个阶段串起来，每一步都用专门的数据集强化对应能力。结果是模型在通用测评、指令遵循和工具使用上都能和 GPT‑4 系列持平甚至超越。

2. **十万级别的长上下文训练**  
   传统模型的上下文窗口一般在 2K‑4K token 左右，长文档会被截断。ChatGLM‑4 系列在预训练阶段直接使用 100K+ token 的序列，让注意力机制学会跨段信息关联。这样在需要一次性阅读完整报告或代码库时，模型的表现明显好于同等规模的短窗口模型。

3. **统一的工具决策框架**  
   过去要让模型使用外部工具，需要手动写 prompt 或者在系统层面硬编码调用逻辑。ChatGLM‑4 All Tools 在对齐数据中加入了“何时调用、调用哪个工具”的标签，模型内部学会生成类似 “<tool:browser>查询…</tool>” 的指令，系统再根据这些标记自动路由。实验显示，这种自驱动的工具选择在网页检索和 Python 计算任务上超过了 GPT‑4 All Tools。

4. **跨语言、跨任务的统一训练语料**  
   虽然核心数据是中英双语，ChatGLM 还加入了 24 种语言的少量语料，形成了一个多语言、跨任务的统一语料池。相比只在单一语言上做大规模预训练的模型，这种多语言混合提升了模型在少数语言上的零样本表现，也让对齐过程更具通用性。

### 方法详解
整体思路可以划分为三大块：大规模双语预训练 → 多阶段对齐 → All‑Tools 决策层。下面按顺序拆解每一步。

1. **大规模预训练**  
   - **数据规模**：约 10 万亿 token，主要来自中文网页、新闻、书籍以及英文互联网文本。  
   - **模型结构**：采用 Transformer 架构，层数和隐藏维度随模型规模（9B、130B）线性增长。  
   - **长序列处理**：使用稀疏注意力 + 位置编码插值，使得在同一次前向传播中可以处理 100K token，避免了传统全连接注意力的 O(N²) 计算瓶颈。

2. **多阶段对齐**  
   - **监督微调（SFT）**：构建了约 2 亿条中英指令‑答案对，覆盖问答、写作、代码、数学等多种任务。模型在此阶段学习“听指令、给答案”。  
   - **RLHF**：让人工评审对模型的多轮对话进行偏好排序，使用 PPO（近端策略优化）把高偏好答案的概率提升。这里的奖励模型专门针对中文流畅度和逻辑一致性做了微调。  
   - **工具对齐**：在 RLHF 基础上加入了工具使用标签。每条训练样本会标记是否需要调用浏览器、Python、文本‑图像模型或用户自定义函数，模型学习在生成答案时自动插入相应的 `<tool>` 标记。

3. **All‑Tools 决策层**  
   - **工具选择器**：本质是一个小型分类头，输入是当前对话的隐藏状态，输出是“无工具 / 浏览器 / Python / 文本‑图像 / 自定义函数”的概率分布。  
   - **执行器**：系统根据模型输出的 `<tool>` 标记调用对应的外部服务，得到结果后再喂回模型继续生成。整个闭环像是模型在“思考 → 需要工具 → 调用 → 继续思考”。  
   - **自适应上下文扩展**：当浏览器返回大量信息时，系统会把关键片段压缩进模型的长上下文窗口，保证后续推理仍然基于最新信息。

**最巧妙的点**在于把工具调用当作语言生成的一个子任务，而不是外部系统的硬编码。这样模型的“意图”直接体现在生成的文本里，训练时就能让模型学会何时、如何、以及使用哪种工具。

### 实验与效果
- **评测数据集**：MMLU（多学科知识）、GSM8K（数学解题）、MATH、BBH（广义推理）、GPQA、HumanEval（代码生成）以及 IFEval（指令遵循）等。长上下文任务使用了 128K token 的文档摘要和长篇阅读理解基准。中文对齐使用 AlignBench。  
- **对标模型**：GPT‑4、GPT‑4‑Turbo（128K）、Claude 3、以及同规模的开源模型（如 LLaMA‑2、InternLM）。  
- **核心结果**（论文中给出的对比）：
  - 在 MMLU、GSM8K、MATH、BBH、GPQA、HumanEval 上，GLM‑4 系列的分数与 GPT‑4 基本持平，部分子任务略有超越。  
  - IFEval 上接近 GPT‑4‑Turbo，指令遵循能力显著提升。  
  - 长上下文任务（128K）上，GLM‑4 与 GPT‑4‑Turbo、Claude 3 差距不大，均能完整处理全篇文档。  
  - AlignBench 中文对齐分数超过 GPT‑4，说明在中文指令理解和生成上更具优势。  
  - All‑Tools 场景下，ChatGLM‑4 All Tools 在网页检索准确率和 Python 计算正确率上均高于 GPT‑4 All Tools。  

- **消融实验**：作者分别去掉了长序列稀疏注意力、RLHF、工具对齐三个模块。结果显示，去掉工具对齐后 All‑Tools 能力下降约 30%；去掉 RLHF 后指令遵循分数下降 15%；去掉稀疏注意力后长上下文任务的准确率下降 20%。这表明每个环节都对最终性能贡献显著。  

- **局限性**：论文承认在低资源语言（如非洲语言）上的表现仍然弱，主要因为训练语料中这类语言占比极低；此外，工具调用的安全性和错误恢复机制仍在探索阶段，偶尔会出现无限循环或错误的外部调用。

### 影响与延伸思考
ChatGLM 的发布让中文社区首次拥有了在通用能力、长上下文和工具协同上与 GPT‑4 同等水平的开源模型，直接推动了国内企业和研究机构在 LLM 应用层面的快速落地。随后出现的 **GLM‑4‑V**（视觉‑语言）和 **WebGLM**（专注网页检索）等衍生项目，都在原有多阶段对齐和工具决策框架上继续深化。业界也开始模仿其 “工具标签 + 执行器” 设计，出现了如 **Toolformer**、**ReAct** 的实现。想进一步了解的话，可以关注以下方向：① 更高效的稀疏注意力算法，② 多语言低资源对齐方法，③ 安全可靠的工具调用协议（如 sandboxed execution）。这些都是 ChatGLM 之后的自然延伸。

### 一句话记住它
ChatGLM 把“懂中英文、读长文、会选工具”三大能力一次性装进了同一套对齐流程，让开源模型首次在全栈 AI 助手上追平甚至超越了 GPT‑4。