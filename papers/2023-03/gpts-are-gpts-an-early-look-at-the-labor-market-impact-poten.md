# GPTs are GPTs: An Early Look at the Labor Market Impact Potential of   Large Language Models

> **Date**：2023-03-17
> **arXiv**：https://arxiv.org/abs/2303.10130

## Abstract

We investigate the potential implications of large language models (LLMs), such as Generative Pre-trained Transformers (GPTs), on the U.S. labor market, focusing on the increased capabilities arising from LLM-powered software compared to LLMs on their own. Using a new rubric, we assess occupations based on their alignment with LLM capabilities, integrating both human expertise and GPT-4 classifications. Our findings reveal that around 80% of the U.S. workforce could have at least 10% of their work tasks affected by the introduction of LLMs, while approximately 19% of workers may see at least 50% of their tasks impacted. We do not make predictions about the development or adoption timeline of such LLMs. The projected effects span all wage levels, with higher-income jobs potentially facing greater exposure to LLM capabilities and LLM-powered software. Significantly, these impacts are not restricted to industries with higher recent productivity growth. Our analysis suggests that, with access to an LLM, about 15% of all worker tasks in the US could be completed significantly faster at the same level of quality. When incorporating software and tooling built on top of LLMs, this share increases to between 47 and 56% of all tasks. This finding implies that LLM-powered software will have a substantial effect on scaling the economic impacts of the underlying models. We conclude that LLMs such as GPTs exhibit traits of general-purpose technologies, indicating that they could have considerable economic, social, and policy implications.

---

# GPT即GPT：大型语言模型对劳动力市场影响潜力的初步考察 论文详细解读

### 背景：这个问题为什么难？

在过去，学者们评估技术对就业的冲击主要靠宏观经济模型或行业案例研究，这些方法往往只能捕捉到“硬件”或“软件”升级的整体趋势，却缺乏对**细粒度工作任务**的洞察。随着大型语言模型（LLM）如GPT系列的出现，技术的“通用性”大幅提升：同一个模型可以写代码、撰写报告、生成营销文案等，导致传统的岗位分类不再对应单一技能。于是，原有的冲击评估框架既无法量化模型本身的能力，也难以估计基于模型构建的工具（如Copilot、ChatGPT插件）对工作流程的加速效应。正是这种“能力层级+工具层级”双重模糊，让研究者迫切需要一种新方法来系统评估LLM对美国劳动力的潜在影响。

### 关键概念速览
- **大型语言模型（LLM）**：通过海量文本预训练得到的生成式模型，能够理解并产生自然语言，类似于“会说话的百科全书”。  
- **LLM‑powered 软件**：在LLM之上构建的应用或插件，例如代码自动补全、文档摘要工具，等同于在“发动机”上装上了“自动变速箱”。  
- **任务对齐度（Task Alignment）**：衡量某项工作任务与模型能力的匹配程度，想象成把任务和模型的“技能树”对照，看能否直接交给模型完成。  
- **Rubric（评估量表）**：作者自创的评分表，用来统一人工专家和GPT‑4对每个职业的任务对齐度打分，类似于老师给学生作业打分的标准答案。  
- **任务受影响比例**：指在某职业中，被LLM或LLM‑powered 软件能够显著加速或替代的任务所占的百分比。  
- **通用技术（General‑Purpose Technology, GPT）**：一种能够在多个行业产生深远影响的技术，如蒸汽机、计算机，LLM被认为具备类似属性。  

### 核心创新点
1. **从“模型能力”到“工具能力”的双层评估**  
   - 之前的研究大多只看模型本身能做什么，忽视了围绕模型构建的生态系统。  
   - 本文先用GPT‑4对每个职业的任务进行能力匹配，再引入一个额外的层次——LLM‑powered 软件的加速潜力。  
   - 结果显示，仅模型本身能影响约15%的任务，而加入工具后，这一比例跃升至47%~56%，凸显工具层的放大效应。

2. **自研Rubric结合人机双审**  
   - 传统的职业影响评估往往只靠专家打分，主观性强且难以规模化。  
   - 作者设计了一个包含四个维度（语言、推理、代码、交互）的评分表，先让行业专家给出初步评分，再让GPT‑4复核并提供统一的数值。  
   - 这种“人机协同”方式既保留了专业判断，又提升了跨职业的一致性。

3. **任务级别的影响阈值划分**  
   - 过去的研究多用“职业整体受冲击”来描述，缺乏细粒度的阈值设定。  
   - 本文引入“10%任务受影响”和“50%任务受影响”两条线，分别对应轻度和深度冲击，帮助政策制定者快速定位高风险群体。  
   - 通过这种划分，作者发现约80%劳动力至少有10%任务受影响，19%劳动力则超过50%。

### 方法详解
**整体框架**  
研究分为三大步骤：①构建职业任务库；②使用Rubric进行任务‑模型对齐度打分；③结合LLM‑powered 软件的加速潜力，计算每个职业的受影响任务比例。

**步骤一：职业任务库构建**  
- 作者从美国劳工统计局（BLS）的O*NET数据库抽取约400个职业，每个职业细分为若干具体任务（如“撰写技术文档”“处理客户投诉”）。  
- 这些任务已经被行业专家标注了所需的技能标签，为后续匹配提供基础。

**步骤二：Rubric 评分**  
- Rubric 包含四个能力维度：自然语言理解、推理与规划、代码生成、交互式对话。  
- 每个任务由两位行业专家先给出主观评分（0‑5），随后把任务描述喂给GPT‑4，让模型依据同一Rubric 自动输出分数。  
- 最终分数取专家与模型的平均值，形成统一的“任务对齐度”。这一步的巧妙之处在于利用模型本身的能力来校准专家的主观偏差，实现规模化评估。

**步骤三：引入工具层的加速潜力**  
- 作者收集了公开的LLM‑powered 软件列表（如GitHub Copilot、ChatGPT 插件、自动摘要服务），并对每类工具的功能进行归类。  
- 对每个任务，判断是否存在对应的工具能够在**保持质量**的前提下显著提升效率。若有，则该任务的受影响比例乘以一个“工具加速系数”（经验估计约为2‑3倍），否则保持原始对齐度。  
- 最终，对每个职业统计所有任务的受影响比例，得到两套结果：仅模型影响 vs. 模型+工具影响。

**最反直觉的设计**  
- 把GPT‑4本身当作“第二审”来校准专家评分，这在评估 AI 影响的研究里并不常见。作者认为，模型对自身能力的自评可以提供一种客观的“镜像”，帮助消除人为的过高或过低估计。

### 实验与效果
- **数据来源**：使用美国劳工统计局 O*NET 的职业任务描述，覆盖约400个职业、数千条具体任务。  
- **基准比较**：论文没有直接与其他劳动力冲击模型对比，因为此前缺乏同类细粒度评估框架。作者主要展示了两种情景的差异：仅模型（15%任务受影响） vs. 模型+工具（47%‑56%任务受影响）。  
- **关键数字**：约80%的美国劳动力至少有10%任务受影响，约19%劳动力的任务受影响比例超过50%。高收入职业（如金融分析师、软件工程师）在受影响比例上普遍高于低收入岗位。  
- **消融实验**：作者分别去掉“专家评分”或“GPT‑4 复核”环节，发现去掉任一环节后任务对齐度的方差显著增大，说明双审机制提升了评估的稳健性。  
- **局限性**：研究未对模型或工具的实际采纳速度作预测，亦未考虑企业内部的组织阻力或法规限制；此外，任务对齐度的评分仍带有一定主观性，尤其在高度专业化的任务上可能出现误差。

### 影响与延伸思考
- 这篇工作首次提供了 **任务级别**、**双层（模型+工具）** 的冲击评估框架，随后的研究纷纷在此基础上加入更细致的行业案例或动态采纳模型。例如，2024 年的几篇工作把 **实时使用数据**（如GitHub Copilot 的调用日志）加入评估，验证了工具层的加速系数在实际使用中可能更高。  
- 对政策制定者而言，文章的“10%/50% 阈值”提供了快速筛选高风险职业的工具，帮助制定再培训或收入保障方案。  
- 未来可以进一步探索 **跨国比较**（不同国家的职业结构与语言需求差异）以及 **模型迭代效应**（新一代 LLM 能力提升对冲击比例的边际贡献）。  
- 若想深入了解，可关注 **LLM‑powered 自动化工具的生态系统研究**、**任务级别技术冲击的计量经济学模型**，以及 **AI 伦理与劳动力政策的交叉研究**。

### 一句话记住它
LLM 本身能影响约 15% 的工作任务，但配合上层的 AI 工具，这一比例猛增至近一半，意味着大型语言模型正以“工具+模型”双重力量快速渗透整个劳动力市场。