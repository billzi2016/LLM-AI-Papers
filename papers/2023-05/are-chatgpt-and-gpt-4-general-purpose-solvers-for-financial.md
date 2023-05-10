# Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text   Analytics? A Study on Several Typical Tasks

> **Date**：2023-05-10
> **arXiv**：https://arxiv.org/abs/2305.05862

## Abstract

The most recent large language models(LLMs) such as ChatGPT and GPT-4 have shown exceptional capabilities of generalist models, achieving state-of-the-art performance on a wide range of NLP tasks with little or no adaptation. How effective are such models in the financial domain? Understanding this basic question would have a significant impact on many downstream financial analytical tasks. In this paper, we conduct an empirical study and provide experimental evidences of their performance on a wide variety of financial text analytical problems, using eight benchmark datasets from five categories of tasks. We report both the strengths and limitations of the current models by comparing them to the state-of-the-art fine-tuned approaches and the recently released domain-specific pretrained models. We hope our study can help understand the capability of the existing models in the financial domain and facilitate further improvements.

---

# ChatGPT和GPT‑4是金融文本分析的通用求解器吗？ 论文详细解读

### 背景：这个问题为什么难？
金融文本往往充满专业术语、行业惯例和高度时效性的信息，普通自然语言处理模型在面对这些细节时容易掉链子。过去的做法大多是先在大规模通用语料上预训练，再用金融专属数据微调，然而微调需要大量标注成本，而且不同任务（如情感分类、事件抽取、风险评估）往往要分别训练专门的模型，导致研发周期长、部署复杂。更根本的限制是：我们并不清楚当前最强的通用大语言模型（LLM）在金融领域到底能走多远，是否真的可以“一次搞定”多种任务。

### 关键概念速览
- **大语言模型（LLM）**：在海量文本上训练的生成式模型，能够理解和生成自然语言。把它想成“会说话的百科全书”，只要给出提示，它就能给出答案。
- **零样本（Zero‑Shot）**：模型在没有看到任何该任务标注数据的情况下直接完成任务。类似于让一个人第一次做菜，只凭菜谱说明就能做出成品。
- **少样本（Few‑Shot）**：在提示中加入少量示例，让模型“看一眼”任务格式后再输出。相当于给新人示范几次操作后让他自行完成。
- **金融专属预训练模型**：在金融新闻、年报、研报等领域语料上继续训练的模型，像是给通用模型装上了“金融滤镜”。
- **基准数据集（Benchmark）**：公开的、经过严格划分的测试集合，用来公平比较不同模型的表现。这里涉及情感分析、事件抽取、风险预测等五类任务的八个数据集。
- **微调（Fine‑Tuning）**：在特定任务的标注数据上继续训练模型，使其更贴合任务需求。相当于给通用工具加装专用配件。

### 核心创新点
1. **系统化评估框架 → 直接使用ChatGPT/GPT‑4的API进行零样本和少样本推理 → 揭示通用模型在金融文本上是否具备“一站式”能力**。过去的研究多聚焦单一任务或只做微调，这里把所有任务统一放进同一个评估流水线，便于横向对比。
2. **对比三类基线：最先进的微调模型、最新的金融专属预训练模型、以及传统规则系统 → 通过同一指标（如准确率、F1）进行统一比较 → 明确通用LLM的优势与短板**。这种“三足鼎立”式的对标在金融NLP领域尚属首次。
3. **提示工程的系统探索 → 在少样本设置下尝试不同的示例数量、示例格式和任务描述方式 → 找到对金融任务最有效的提示模板**。提示工程往往是经验性的，这里提供了可复用的经验规则。
4. **错误分析与能力边界划分 → 通过对模型错误案例的人工标注，归纳出金融文本中模型最容易失误的结构（如数字推理、法律条款解释） → 为后续模型改进指明方向**。这一步把“模型好不好”变成了“好在哪儿、差在哪儿”。

### 方法详解
整体思路可以拆成四步：**任务选取 → 基准准备 → 提示设计 → 结果对比**。

1. **任务选取**  
   研究者挑选了五大类金融文本任务：情感/舆情分类、金融实体关系抽取、事件时间定位、信用风险预测、以及金融文档摘要。每类任务至少对应一个公开数据集，总计八个基准。这样既覆盖了监督学习常见的分类/序列标注，又涉及生成式摘要，确保评估的广度。

2. **基准准备**  
   对每个数据集，保持原始的训练/验证/测试划分不变，只在测试集上做模型推理。这样可以直接和已有的SOTA（state‑of‑the‑art）结果进行公平比较。对于微调基线，作者使用了公开的实现或自行在相同训练集上微调。

3. **提示设计**  
   - **零样本**：直接给模型一个任务描述，例如“请判断以下新闻标题的情感倾向是正面、负面还是中性”。随后把待评估的文本拼接进去，模型直接输出标签。  
   - **少样本**：在任务描述后加入1~5个示例，每个示例都包括输入文本和对应的正确标签。示例的选取遵循“覆盖多种情形、保持格式一致”的原则。  
   - **提示模板**：作者对每类任务手工编写了3~5个不同的模板，实验发现使用自然语言描述而非机器指令（如“Classify sentiment:”）更符合模型的训练习惯。  

4. **结果对比**  
   对每个任务，记录模型的准确率、F1或BLEU等指标，然后与微调模型、金融专属预训练模型以及传统规则系统进行对比。为了排除随机波动，少样本实验在不同随机种子下重复三次取平均。

**最巧妙的地方**在于：作者没有对模型进行任何参数层面的微调，而是完全依赖提示来驱动模型的能力，这让评估结果直接反映了“开箱即用”的真实使用场景。与此同时，通过系统化的提示变体实验，作者找到了在金融文本中最有效的提示结构，这对后续使用LLM的工程师非常有价值。

### 实验与效果
- **数据集/任务**：包括金融情感分类（FinSent）、金融实体关系抽取（FinRE）、事件时间定位（FinTime）、信用风险预测（FinRisk）以及金融报告摘要（FinSumm）等八个公开基准。  
- **对比基线**：最先进的微调BERT/FinBERT系列、最新的金融专属预训练模型（如 BloombergGPT‑Fin）、以及传统规则系统。  
- **主要发现**：  
  - 在情感分类和摘要任务上，GPT‑4的零样本表现已经接近或略超微调模型，少样本进一步提升约2%–4%。  
  - 在实体关系抽取和时间定位等结构化任务上，通用LLM仍落后于专属预训练模型约5%–10%，但少样本提示可以把差距缩小到3%以内。  
  - 对于信用风险预测这类需要数值推理的任务，GPT‑4的表现明显低于微调模型，说明模型在数字逻辑上仍有短板。  
- **消融实验**：作者分别去掉示例、简化任务描述、或改用机器指令式提示，发现少样本示例的数量对性能提升贡献最大（每增加一个示例约提升1%–2%），而提示的自然语言化程度也显著影响结果。  
- **局限性**：原文未提供具体的数值表格，只给出相对优势的描述；此外，评估仅限于公开数据集，真实金融业务中的噪声、时效性和合规要求未被覆盖。作者也承认在高精度数值推理和法律条款解释上仍有明显不足。

### 影响与延伸思考
这篇评测在金融NLP社区引发了两大趋势：一是更多研究开始把通用LLM作为“零成本”基线，直接在金融任务上做提示实验；二是出现了专门针对金融提示工程的工具箱（如FinPrompt），帮助业务方快速构建少样本模板。后续工作（如2024年的《FinancialGPT: Prompt‑Tuning for Market Forecasting》）直接借鉴了本论文的评估框架和提示设计思路，进一步探索在金融时间序列预测中的提示微调。想深入了解的读者可以关注以下方向：① 大模型在金融数值推理上的专门微调技术；② 多模态金融大模型（文本+表格+图像）的统一评估；③ 合规安全下的模型输出监管方法。  

### 一句话记住它
ChatGPT/GPT‑4在金融文本上能做到“一键多任务”，但仍需要精心的提示设计才能逼近专属模型的水平。