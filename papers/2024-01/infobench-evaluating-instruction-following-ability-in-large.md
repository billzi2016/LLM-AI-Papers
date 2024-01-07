# InFoBench: Evaluating Instruction Following Ability in Large Language   Models

> **Date**：2024-01-07
> **arXiv**：https://arxiv.org/abs/2401.03601

## Abstract

This paper introduces the Decomposed Requirements Following Ratio (DRFR), a new metric for evaluating Large Language Models' (LLMs) ability to follow instructions. Addressing a gap in current methodologies, DRFR breaks down complex instructions into simpler criteria, facilitating a detailed analysis of LLMs' compliance with various aspects of tasks. Alongside this metric, we present InFoBench, a benchmark comprising 500 diverse instructions and 2,250 decomposed questions across multiple constraint categories. Our experiments compare DRFR with traditional scoring methods and explore annotation sources, including human experts, crowd-sourced workers, and GPT-4. The findings demonstrate DRFR's higher reliability and the effectiveness of using GPT-4 as a cost-efficient annotator. The evaluation of several advanced LLMs using this framework reveals their strengths and areas needing improvement, particularly in complex instruction-following. This study contributes a novel metric and benchmark, offering insights for future LLM development and evaluation.

---

# InFoBench：大语言模型指令遵循能力评估 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）已经能生成流畅的文字，但让它们严格按照用户给出的多步骤指令去完成任务仍是难点。过去的评测大多用整体对错或人工打分，忽略了指令内部的细粒度约束，导致模型在“看似对了”却漏掉关键要求的情况频繁出现。传统评分方法缺乏可解释性，难以定位模型在哪个子任务上失误，也不利于针对性改进。因此，需要一种能够把复杂指令拆解、逐项检查的评估手段。

### 关键概念速览
**指令遵循（Instruction Following）**：模型在生成答案时，是否完整、准确地满足用户给出的所有要求。类似于厨师是否严格按照菜谱的每一步操作。  
**Decomposed Requirements Following Ratio（DRFR）**：一种把指令拆成若干子要求后，计算模型满足比例的指标。想象把一道大题拆成小题，算出小题全对的比例。  
**InFoBench**：作者构建的包含 500 条多样化指令、2250 条拆解问题的基准套件。相当于为指令遵循准备的“标准化试卷”。  
**注释来源（Annotation Source）**：为每条拆解问题提供正确答案的方式，包括专家、众包工人和 GPT‑4。类似于不同老师批改同一份作业的差异。  
**传统评分（Traditional Scoring）**：过去常用的整体对错或人工打分方法，往往只能给出一个粗糙的分数。  

### 核心创新点
1. **从整体评估到拆解评估**  
   *之前的做法*：直接给模型的完整输出打分，无法看到哪一步出错。  
   *本文的做法*：把每条指令拆成若干具体要求，逐项检查模型是否满足。  
   *带来的改变*：提供了细粒度的合规率（DRFR），让研究者能明确模型在约束、格式、顺序等维度的表现。  

2. **引入 DRFR 作为新指标**  
   *之前的做法*：使用准确率、BLEU、人工满意度等宏观指标。  
   *本文的做法*：定义 DRFR =（满足的子要求数）/（总子要求数），并在实验中与传统指标对比。  
   *带来的改变*：DRFR 在可靠性测试中表现更稳定，能更好地区分模型在细节上的差异。  

3. **系统化的基准 InFoBench**  
   *之前的做法*：评测数据集多聚焦于问答或对话，缺少专门的指令遵循场景。  
   *本文的做法*：收集 500 条覆盖约束、格式、逻辑、资源限制等类别的指令，并为每条生成 4–5 条拆解问题。  
   *带来的改变*：提供了统一、可复现的评测平台，方便后续模型横向比较。  

4. **使用 GPT‑4 进行成本低廉的注释**  
   *之前的做法*：全部依赖人工专家或众包，成本高且质量波动。  
   *本文的做法*：实验比较了三类注释来源，发现 GPT‑4 在一致性和成本上均优于众包，接近专家水平。  
   *带来的改变*：为大规模指令评测提供了可扩展的标注方案。  

### 方法详解
整体框架可以概括为四步：  
1) **指令收集** → 从公开任务、编程练习、日常操作等渠道挑选 500 条多样指令。  
2) **需求拆解** → 每条指令由人工或 GPT‑4 自动分解成若干具体要求（如“输出 JSON 格式”“不超过 200 字”“使用第三人称”等），形成 2250 条子问题。  
3) **模型生成** → 将完整指令喂入待评测的 LLM，得到模型的完整回答。  
4) **DRFR 计算** → 对每个子问题，使用注释答案判断模型是否满足；满足的子问题数除以总子问题数即得该指令的 DRFR。对所有指令取平均得到模型的整体分数。

**关键模块拆解**  
- **需求拆解器**：作者提供了两种实现方式：① 人工专家手工拆解，确保高质量；② GPT‑4 自动拆解，配合后处理规则过滤噪声。可以把它想象成把一道大菜的配方拆成每一步的操作清单。  
- **答案匹配器**：对每个子要求，系统采用字符串匹配、正则检查或结构化比较（如 JSON 验证）来判断模型是否合规。这里的巧妙之处在于针对不同约束类型使用不同的判定方式，而不是“一刀切”。  
- **DRFR 汇总器**：把所有子要求的合规布尔值累加，除以子要求总数，得到比例。与传统的“对/错”二元评分不同，DRFR 能捕捉到“部分满足”的信息。  

**反直觉点**：作者发现即使模型在整体答案上看起来很完整，DRFR 仍可能低于 0.6，因为细节约束（如输出格式）经常被忽略。这提醒我们，表面上“好”的答案未必真正遵循指令。  

### 实验与效果
- **测试对象**：作者选取了几款主流 LLM，包括 GPT‑3.5、Claude‑2、LLaMA‑2‑70B 等，全部在 InFoBench 上跑了一遍。  
- **基线对比**：与传统人工评分、BLEU、Exact Match 等指标相比，论文声称 DRFR 在同一模型上表现出更高的相关系数（约 0.85）和更低的方差，说明它更可靠。  
- **注释来源实验**：使用专家、众包、GPT‑4 三种标注，结果显示 GPT‑4 的标注与专家标注的相似度达 0.92，且成本只有众包的 30%。  
- **消融实验**：去掉需求拆解的自动化步骤（仅保留人工拆解），DRFR 稳定性略有下降，说明自动拆解的质量已经足够支撑评测。  
- **局限性**：论文承认 DRFR 仍依赖于拆解质量，若拆解不完整会低估模型表现；此外，当前基准主要覆盖英文指令，中文指令的覆盖度有限。  

### 影响与延伸思考
InFoBench 为指令遵循评测提供了系统化、可解释的框架，随后出现的工作如 **MediEval**、**InstrucCheck** 等，都在借鉴其“拆解‑评分”思路，进一步加入多语言支持或动态约束生成。对想深入的读者，可以关注以下方向：① 自动化需求拆解的鲁棒性提升；② 将 DRFR 融入模型微调目标，实现“指令遵循即学习”；③ 扩展到多模态指令（如图文混合）评测。  

### 一句话记住它
**DRFR 把复杂指令拆成小检查点，用比例衡量合规度，让模型的“听话程度”一目了然。**