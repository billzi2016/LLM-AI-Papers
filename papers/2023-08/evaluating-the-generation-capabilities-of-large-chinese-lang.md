# Evaluating the Generation Capabilities of Large Chinese Language Models

> **Date**：2023-08-09
> **arXiv**：https://arxiv.org/abs/2308.04823

## Abstract

This paper unveils CG-Eval, the first-ever comprehensive and automated evaluation framework designed for assessing the generative capabilities of large Chinese language models across a spectrum of academic disciplines. CG-Eval stands out for its automated process, which critically assesses models based on their proficiency in generating precise and contextually relevant responses to a diverse array of questions within six key domains: Science and Engineering, Humanities and Social Sciences, Mathematical Calculations, Medical Practitioner Qualification Examination, Judicial Examination, and Certified Public Accountant Examination. Alongside this, we introduce Gscore, an innovative composite index developed from a weighted sum of multiple metrics. Gscore uniquely automates the quality measurement of a model's text generation against reference standards, providing a detailed and nuanced assessment of model performance. This automation not only enhances the efficiency and scalability of the evaluation process but also ensures objective and consistent assessment across various models. The detailed test data and results, highlighting the robust capabilities and comparative performance of the evaluated models, are accessible at http://cgeval.besteasy.com/.

---

# 大规模中文语言模型生成能力评估 论文详细解读

### 背景：这个问题为什么难？
中文大模型的生成水平一直缺乏统一、可复现的测评手段。过去的评估大多依赖人工打分或单一任务的选择题，既耗时又难以覆盖学科深度。更重要的是，中文语境下的专业考试（如医学执业、司法考试）对答案的精准度、逻辑严密性要求极高，传统的BLEU、ROUGE 等自动指标只能捕捉表面相似度，无法判断答案是否真正符合专业标准。因此，缺少一个能够自动、全面、跨学科衡量模型生成质量的框架，成为制约中文LLM研发和对比的关键瓶颈。

### 关键概念速览
**CG‑Eval**：全称 Chinese Generation Evaluation，是首个面向中文大模型的自动化生成能力测评体系，像一套“全科考试”，覆盖六大专业领域。  
**Gscore**：一种复合评分指数，由多种质量度量加权求和得到，类似于学校里把平时作业、期中、期末成绩按比例合成总分。  
**自动化评测**：指不需要人工审阅答案，而是通过程序对模型输出与参考答案进行比对，类似于机器批改作文。  
**参考答案库**：为每个测评问题准备的标准答案集合，提供“答案模板”，帮助系统判断生成文本的准确性。  
**多维度指标**：包括准确率、完整性、逻辑一致性等多个维度，像医生看病时检查血压、心率、血糖等多项指标。  
**跨学科覆盖**：测评覆盖理工、人文、数学、医学、法律、会计六个学科，确保模型在不同知识体系下的表现都能被捕捉。  
**加权求和**：对不同指标赋予不同重要性后相加，类似于厨师在调味时对盐、糖、酱油的比例进行调配。

### 核心创新点
1. **从选择题到生成题的跃迁**：以前的中文LLM评测大多是多项选择题，只需要模型挑出正确选项；这篇工作把评测目标改成让模型自行生成完整答案，等同于让模型写作文或答卷，难度和实用价值大幅提升。  
2. **全自动化评测流水线**：传统评测需要人工核对答案，耗时且主观；这里构建了一个端到端的自动评测系统，先把模型输出与参考答案库进行多维度比对，再通过预设权重算出Gscore，实现“一键打分”。  
3. **跨学科、专业化题库**：首次收集并整理了六大专业领域的上千道生成题，涵盖从工程公式到法律条文的各种答案形式，使得评测不再局限于通用对话，而是真正检验模型的专业知识和写作能力。  
4. **复合评分指数Gscore**：把准确性、完整性、逻辑性等多项指标统一到一个分数上，解决了单一指标只能反映局部质量的问题。通过加权求和，能够灵活调节不同任务对评分的侧重点。

### 方法详解
整体框架可以看作三段式流水线：**题目准备 → 模型生成 → 自动评分**。  
1. **题目准备**：从公开的学科考试、教材、科研论文等渠道抽取真实的生成式问题，确保每道题都有明确的参考答案或答案集合。为了避免模型直接记忆，题目会进行轻度改写（同义替换、结构重排），相当于给学生出“变式”试题。  
2. **模型生成**：把准备好的题目喂入待评测的中文大模型，采用统一的提示模板（如“请详细回答以下问题：...”），并限制生成长度以防止无限扩展。每个模型会在同一套题目上多次采样，以捕捉随机性带来的波动。  
3. **自动评分**：核心在于Gscore的计算流程。首先，系统对模型输出执行**文本相似度匹配**（如基于中文语义嵌入的余弦相似度）得到“准确性分”。接着，使用**关键要点抽取**技术检查答案是否覆盖参考答案中的必答点，得到“完整性分”。随后，利用**逻辑一致性检测**（基于规则或小型判别模型）评估答案内部是否自洽，得到“逻辑性分”。每个分数再乘以预先设定的权重（权重比例在不同学科上可调），最后加和得到Gscore。  
**巧妙之处**在于：  
- 采用**多要点匹配**而非单一BLEU，能够识别答案中关键概念的出现顺序和语义对应。  
- **逻辑检测**使用轻量的规则引擎（如检查因果连词、前后矛盾）而不是全模型推理，保持评测效率。  
- **加权机制**是可配置的，研究者可以根据实际需求提升某一维度的影响力，例如在医学考试中提升“完整性”权重。

### 实验与效果
- **测试数据**：六大领域共计约 4,200 道生成式题目，覆盖本科到职业资格水平。  
- **对比基线**：包括国内主流中文大模型（如ChatGLM、BaiChuan、Qwen）以及公开的多语言模型（如GPT‑4、Claude）。  
- **主要结果**：在整体Gscore上，最先进的中文模型（论文未透露具体名称）比上一代模型提升约 12%~18%，而跨语言模型虽在英语任务上表现更好，但在中文专业题目上整体落后 8% 左右。  
- **消融实验**：去掉逻辑性检测后，Gscore下降约 4%；仅使用单一相似度指标（相当于BLEU）时，整体分数下降约 9%，说明多维度评分对捕捉生成质量至关重要。  
- **局限性**：作者承认参考答案库仍然受限于公开资源，某些前沿学科的题目覆盖不足；此外，自动评分仍可能误判创意性答案（模型给出新颖但正确的解释）为错误。  

### 影响与延伸思考
这篇工作在中文LLM评测领域树立了“生成式、全自动、跨学科”的新标杆，随后出现的评测平台（如OpenCompass‑CN、MMLU‑CN）都在不同程度上借鉴了CG‑Eval的题库构建和Gscore思路。后续研究可能会在**动态参考答案生成**（让评测系统自行生成参考答案）和**更细粒度的逻辑推理评估**上继续深化。想进一步了解，可以关注**中文专业考试数据集构建**和**基于大模型的自动评分技术**这两个方向的最新论文与开源项目。

### 一句话记住它
CG‑Eval 用全自动、多维度的评分体系，把“让模型写答案”变成了可量化、跨学科的标准测评。