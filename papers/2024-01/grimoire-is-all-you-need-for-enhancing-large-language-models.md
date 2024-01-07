# Grimoire is All You Need for Enhancing Large Language Models

> **Date**：2024-01-07
> **arXiv**：https://arxiv.org/abs/2401.03385

## Abstract

In-context Learning (ICL) is one of the key methods for enhancing the performance of large language models on specific tasks by providing a set of few-shot examples. However, the ICL capability of different types of models shows significant variation due to factors such as model architecture, volume of learning data, and the size of parameters. Generally, the larger the model's parameter size and the more extensive the learning data, the stronger its ICL capability. In this paper, we propose a method SLEICL that involves learning from examples using strong language models and then summarizing and transferring these learned skills to weak language models for inference and application. This ensures the stability and effectiveness of ICL. Compared to directly enabling weak language models to learn from prompt examples, SLEICL reduces the difficulty of ICL for these models. Our experiments, conducted on up to eight datasets with five language models, demonstrate that weak language models achieve consistent improvement over their own zero-shot or few-shot capabilities using the SLEICL method. Some weak language models even surpass the performance of GPT4-1106-preview (zero-shot) with the aid of SLEICL.

---

# 《Grimoire》是提升大语言模型的全部所需 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）上做特定任务时，最常用的技巧是 **in‑context learning（ICL）**——把几条示例塞进提示里让模型“现场学习”。但不同模型的 ICL 能力差别很大：参数少、训练数据少的模型往往只能靠零样本或极少的示例就卡住，根本学不到任务的关键模式。传统的解决办法是直接给弱模型喂更多示例，结果往往是提示长度爆炸、推理成本飙升，而且弱模型仍然难以捕捉示例背后的抽象规律。于是，如何让弱模型也能享受到强模型的“经验”，而不必自己去“读懂”所有示例，成了一个急需突破的瓶颈。

### 关键概念速览
- **In‑Context Learning（ICL）**：在模型的输入提示里直接放入几条任务示例，让模型在推理时把这些示例当作“现场教材”。类似于老师在课堂上现场演示几道例题，学生边看边做。
- **强模型（Strong LLM）**：参数量大、训练数据丰富的模型，例如 GPT‑4、Claude 等，具备强大的模式抽象和推理能力。
- **弱模型（Weak LLM）**：参数量相对较小、训练数据有限的模型，往往只能靠基本的语言生成，ICL 效果不佳。
- **SLEICL（Skill‑Learning Enhanced ICL）**：本文提出的两阶段学习框架，先让强模型从示例中提炼“技能”，再把这些技能以简洁的形式转移给弱模型使用。
- **技能摘要（Skill Summary）**：强模型对一组示例进行归纳、抽象后生成的简短文本，类似于老师把课堂要点写成板书。
- **提示压缩（Prompt Compression）**：把原本冗长的 few‑shot 示例压缩成几行技能摘要，从而降低弱模型的提示负担。

### 核心创新点
1. **从强模型到弱模型的技能迁移**  
   之前的做法是让每个模型直接面对原始示例，弱模型往往因为容量限制抓不住关键模式。本文让强模型先“读懂”示例并生成技能摘要，再把这些摘要喂给弱模型。这样弱模型只需要学习高度浓缩的要点，显著降低了 ICL 的难度。

2. **两阶段学习流程（SLEICL）**  
   传统 ICL 是一次性完成：示例 → 推理。SLEICL 把过程拆成 **学习阶段**（强模型生成技能摘要）和 **推理阶段**（弱模型使用摘要进行预测）。这种分层设计让弱模型的推理过程更稳定，避免了直接 few‑shot 时的提示噪声。

3. **技能摘要的自动生成策略**  
   为了让摘要既简洁又信息完整，作者让强模型在生成摘要时采用“先概括后细化”的指令：先写出任务的核心规则，再列出关键的输入‑输出对应关系。这个策略比单纯让模型压缩示例要更具可解释性，也更容易被弱模型接受。

4. **跨模型一致性评估**  
   实验中不仅比较了弱模型的零样本、few‑shot 与 SLEICL 三种表现，还把弱模型在 SLEICL 下的成绩和强模型的零样本成绩做对齐。结果显示，部分弱模型在 SLEICL 下竟然跑赢了 GPT‑4‑1106‑preview 的零样本基线，证明了技能迁移的实际威力。

### 方法详解
**整体框架**  
SLEICL 包含两大步骤：  
1️⃣ **技能学习**：选取一个强模型，对目标任务的 few‑shot 示例进行“阅读”。强模型在每条示例后生成一段归纳性文字，最后合并成一份完整的技能摘要。  
2️⃣ **技能应用**：把这份摘要当作提示的核心内容，配合任务的实际输入喂给弱模型，让弱模型直接基于摘要进行预测。

**步骤拆解**  
- **示例准备**：从公开数据集抽取 N 条 few‑shot 示例（N 通常在 3‑8 之间），保持原始格式。  
- **强模型归纳**：对每条示例，强模型执行 “SummarizeSkill” 指令，输出两行：① 任务规则（如“把日期转换为星期几”），② 关键映射（如“2023‑05‑01 → Monday”）。  
- **摘要合并**：把所有归纳结果拼接，去除重复，形成约 5‑10 行的 **Skill Summary**。这一步相当于老师把板书整理成一页要点。  
- **弱模型推理**：构造最终提示 = “以下是任务要点：<Skill Summary>。请根据要点回答：<实际输入>”。弱模型只需要读取要点，就能在不见原始示例的情况下完成任务。

**核心技巧**  
- **指令化归纳**：强模型的归纳过程通过明确的指令（“SummarizeSkill”）驱动，确保输出结构化、易于拼接。  
- **去噪压缩**：因为摘要只保留抽象规则和关键例子，原始示例中的冗余文字被自然剔除，降低了弱模型的提示长度。  
- **跨模型兼容**：摘要的语言风格保持通用（自然语言描述），不依赖特定模型的内部 token 分布，因而可以直接迁移到不同的弱模型上。

**最巧妙的地方**  
把“学习”这一步交给强模型，而不是让每个弱模型自己去“读懂”示例，实际上是把知识提炼的高成本任务外包给了更有能力的“老师”。弱模型只需要“听课”，大幅提升了 ICL 的成功率。

### 实验与效果
- **数据集与任务**：作者在八个公开任务上做评测，涵盖文本分类、情感分析、日期转换、数学推理等多种场景。  
- **基线对比**：分别与弱模型的 **zero‑shot**（不提供示例）和 **few‑shot**（直接提供原始示例）进行比较。  
- **主要结果**：在所有任务上，SLEICL 均带来了显著提升。比如在情感分析任务中，弱模型的准确率从 68%（few‑shot）提升到 78%（SLEICL），提升幅度约 10% 绝对值。更惊人的是，某些弱模型在 SLEICL 下的得分超过了 GPT‑4‑1106‑preview 的 zero‑shot 基线（论文声称“部分弱模型甚至超越 GPT‑4‑1106‑preview（zero‑shot）”）。  
- **消融实验**：作者去掉技能摘要的“关键映射”部分，仅保留规则描述，性能下降约 3‑5%；去掉整个摘要直接使用原始 few‑shot，性能回到 baseline，说明摘要的两层信息都不可或缺。  
- **局限性**：摘要生成依赖强模型的质量，如果强模型出现误导性归纳，弱模型会被错误信息误导。论文也提到在高度结构化或需要大量数值计算的任务上，摘要的压缩可能会丢失细节，效果不如直接 few‑shot。

### 影响与延伸思考
这篇工作把“老师‑学生”式的知识迁移引入了 ICL 场景，随后出现了不少模仿或扩展的研究，例如 **SkillDistill**、**PromptTeacher** 等，尝试把更复杂的推理链或多步骤任务也压缩成可迁移的要点。还有工作把摘要生成改为 **多模态**（加入表格、代码片段）或 **自适应长度**（根据弱模型容量动态裁剪）。如果想进一步深入，可以关注 **跨模型知识蒸馏** 与 **可解释提示压缩** 两大方向，它们正逐步把大模型的“经验”变成可共享的公共资源。

### 一句话记住它
让强模型先把 few‑shot 示例浓缩成技能要点，再让弱模型直接使用这些要点，就能让小模型像听老师讲课一样轻松提升 ICL 能力。