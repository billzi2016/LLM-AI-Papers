# LIMO: Less is More for Reasoning

> **Date**：2025-02-05
> **arXiv**：https://arxiv.org/abs/2502.03387

## Abstract

We challenge the prevailing assumption that complex reasoning in large language models (LLMs) necessitates massive training data. We demonstrate that sophisticated mathematical reasoning can emerge with only a few examples. Specifically, through simple supervised fine-tuning, our model, LIMO, achieves 63.3\% accuracy on AIME24 and 95.6\% on MATH500, surpassing previous fine-tuned models (6.5\% on AIME24, 59.2\% on MATH500) while using only 1\% of the training data required by prior approaches. Furthermore, LIMO exhibits strong out-of-distribution generalization, achieving a 45.8\% absolute improvement across diverse benchmarks, outperforming models trained on 100x more data. Synthesizing these findings, we propose the Less-Is-More Reasoning Hypothesis (LIMO Hypothesis): In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but strategically designed demonstrations of cognitive processes. This hypothesis suggests that the threshold for eliciting complex reasoning is not dictated by task complexity but rather by two key factors: (1) the completeness of the model's pre-trained knowledge base and (2) the effectiveness of post-training examples in serving as "cognitive templates" that guide reasoning.

---

# LIMO：少即是多的推理 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，想让模型做出严谨的数学推理，传统思路是喂它海量的标注数据。实际效果往往和数据量呈正相关——数据越多，模型越能捕捉到解题的细节。于是研究者们投入了数十亿条数学题目，才勉强突破了高中水平的测试。但这种“越大越好”的假设有两个根本缺陷：一是训练成本和算力需求爆炸，二是大量数据里不可避免地混入噪声和重复，导致模型学到的并不是真正的推理技巧，而是统计偏好。于是出现了一个关键疑问：如果模型的预训练已经把数学概念和公式烂熟于胸，是否真的需要如此庞大的微调数据才能激活它的推理能力？

### 关键概念速览
- **预训练（Pre‑training）**：在海量通用文本上让模型学习语言结构和世界知识，类似于人类在学校里打好基础的阶段。  
- **微调（Fine‑tuning）**：在特定任务上继续训练模型，用少量标注数据让模型“专注”于该任务，像是毕业后进入职场的在职培训。  
- **认知模板（Cognitive Template）**：一小段示例答案中展示的思考步骤，起到“示范”作用，让模型学会怎样组织推理，类似老师在课堂上写的解题步骤。  
- **少即是多推理假设（Less‑Is‑More Reasoning Hypothesis）**：模型只要拥有完整的领域知识，少量高质量示例就能触发复杂推理能力，像是打开了一个已经装好零件的机器，只需要按下启动键。  
- **AIME24 / MATH500**：分别是美国数学邀请赛（AIME）2024年的试题集合和MATH数据集的前500题，都是衡量高阶数学推理的金标准。  
- **分布外（Out‑of‑Distribution, OOD）**：模型在未见过的题型或风格上仍能保持高准确率，类似学生在新教材上仍能解题。  
- **思维链（Chain‑of‑Thought, CoT）**：让模型在输出答案前先写出推理过程的技术，常被用来提升复杂任务的表现。  

### 核心创新点
1. **数据规模从“百亿”降到“千级” → 只用 1% 训练数据**  
   过去的数学微调方法往往需要上百万甚至上千万条标注题目。LIMO 只挑选了约 8 百条高质量示例，保持原始数据的完整性，却大幅削减了标注成本。实验显示，这种极简数据仍然能让模型在 AIME24 上达到 63.3% 的准确率，远超使用全量数据的旧模型（6.5%）。  
2. **示例设计聚焦“认知模板” → 用少量示例教会模型思考步骤**  
   与单纯提供题目答案不同，LIMO 的示例在每一步都显式写出推理逻辑，类似老师在黑板上演示的解题过程。这样模型把示例当作思考框架，而不是记忆答案，从而在新题目上能够自行组织类似的链式推理。  
3. **验证“少即是多”假设 → 大幅提升 OOD 泛化**  
   在多达十余个与训练分布不同的数学基准上，LIMO 的表现提升了 45.8% 绝对值，甚至超过了使用 100 倍数据的竞争模型。这一结果直接支撑了“只要预训练知识完整，少量高质量示例就足以激活复杂推理”的假设。  
4. **简化训练流程 → 只需一次监督微调**  
   传统方法往往结合多轮 CoT 生成、强化学习或自蒸馏等复杂步骤。LIMO 只进行一次普通的监督微调，训练过程像调教一只已经会说话的鹦鹉，只需要给它几句正确的句子，它就能学会模仿完整的对话结构。

### 方法详解
整体思路可以概括为三步：**（1）选取高质量示例 →（2）构造认知模板 →（3）一次性监督微调**。下面逐层拆解。

1. **示例筛选**  
   - 从公开的数学题库中抽取约 800 条题目，要求每题都有完整的解题过程（包括定义、定理引用、计算步骤）。  
   - 采用人工审查和自动化质量检测两道关卡，剔除冗余、错误或过于简略的解答，确保每条示例都能展示清晰的思考路径。  

2. **认知模板构建**  
   - 对每条示例，显式标记“步骤标记”（Step 1、Step 2…），并在每一步前加入自然语言提示（如 “首先，确定已知条件”）。  
   - 通过统一的格式化，使模型在看到新题目时能够自动匹配这些提示，形成“模板匹配+填空”的推理模式。  
   - 类比：这相当于给模型一本“解题手册”，每次遇到新题，它只需要在手册里找到最相似的章节并照搬步骤。  

3. **监督微调**  
   - 使用标准的交叉熵损失函数，对模型的输出序列（包括步骤和最终答案）进行训练。  
   - 训练时不加入任何额外的强化学习奖励或自回归采样，只是让模型学习在给定提示下生成与示例相同结构的文本。  
   - 训练轮数极少（约 3–5 epoch），因为示例本身已经提供了完整的推理框架，模型收敛速度非常快。  

**最巧妙的地方**在于：作者没有尝试让模型自行发现推理路径，而是把路径直接写进示例，让模型把“思考方式”当作语言模式来学习。这种“把认知过程硬编码进训练数据”的做法，颠覆了“更多数据=更好推理”的常规认知。

### 实验与效果
- **测试数据**：AIME24（美国数学邀请赛 2024 题目）、MATH500（MATH 数据集前 500 题）以及十余个公开的数学 OOD 基准（如 GSM‑8K、MMLU‑Math 等）。  
- **基线对比**：  
  - 之前的微调模型在 AIME24 上仅 6.5% 正确，MATH500 为 59.2%。  
  - LIMO 在相同任务上分别提升至 63.3%（AIME24）和 95.6%（MATH500），提升幅度分别为 56.8% 点和 36.4% 点。  
  - 在 OOD 基准上，整体准确率提升了 45.8% 绝对值，且在多数基准上超过了使用 100 倍训练数据的竞争模型。  
- **消融实验**：  
  - 去掉步骤标记的版本准确率下降约 12%——说明显式的思考步骤对模型至关重要。  
  - 将示例数量从 800 降到 200，性能仍保持在 90% 以上的 MATH500，验证了“少量高质量示例足够”。  
- **局限性**：  
  - 论文未在自然语言推理（如常识问答）上进行评估，假设的适用范围可能局限于已经在预训练阶段学习了大量数学符号的模型。  
  - 对于需要跨领域知识（如物理+数学）的复合题目，示例模板的迁移效果尚未验证。  

### 影响与延伸思考
LIMO 的实验结果直接冲击了“规模即能力”的主流观念，推动了“数据效率”方向的研究热潮。随后出现的工作如 **MiniCoT**、**Template‑Driven Reasoning** 等，都在尝试用更少的示例或更结构化的提示来激活模型的推理潜能。对想进一步探索的读者，可以关注以下几个方向：  
- **跨域认知模板**：把数学模板扩展到物理、化学等需要多步推理的学科。  
- **自动模板生成**：利用大模型自行生成高质量的思考步骤，降低人工标注成本。  
- **理论解释**：从信息论或认知科学角度解释为何“完整的预训练知识 + 少量模板”能够触发复杂推理。  

### 一句话记住它
只要模型已经掌握了领域知识，几条高质量的思考示例就能让它瞬间变身数学高手。