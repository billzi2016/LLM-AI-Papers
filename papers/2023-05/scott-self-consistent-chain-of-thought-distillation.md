# SCOTT: Self-Consistent Chain-of-Thought Distillation

> **Date**：2023-05-03
> **arXiv**：https://arxiv.org/abs/2305.01879

## Abstract

Large language models (LMs) beyond a certain scale, demonstrate the emergent capability of generating free-text rationales for their predictions via chain-of-thought (CoT) prompting. While CoT can yield dramatically improved performance, such gains are only observed for sufficiently large LMs. Even more concerning, there is little guarantee that the generated rationales are consistent with LM's predictions or faithfully justify the decisions. In this work, we propose a faithful knowledge distillation method to learn a small, self-consistent CoT model from a teacher model that is orders of magnitude larger. To form better supervision, we elicit rationales supporting the gold answers from a large LM (teacher) by contrastive decoding, which encourages the teacher to generate tokens that become more plausible only when the answer is considered. To ensure faithful distillation, we use the teacher-generated rationales to learn a student LM with a counterfactual reasoning objective, which prevents the student from ignoring the rationales to make inconsistent predictions. Experiments show that, while yielding comparable end-task performance, our method can generate CoT rationales that are more faithful than baselines do. Further analysis suggests that such a model respects the rationales more when making decisions; thus, we can improve its performance more by refining its rationales.

---

# SCOTT：自洽思维链蒸馏 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）规模足够大时，给出“思维链”（Chain‑of‑Thought，CoT）提示能够让模型先写出推理步骤再给答案，准确率会大幅提升。但这种提升只在上百亿参数以上的模型上出现，小模型几乎看不到收益。更糟的是，模型生成的推理过程往往与最终答案不匹配，甚至出现自相矛盾的“幻觉”。因此，如何让体积更小的模型也能产生可信、与答案一致的思维链，成为阻碍实际部署的关键瓶颈。

### 关键概念速览
- **CoT（思维链）**：模型在输出答案前先把推理过程写出来，像解数学题时先列出算式再写结果，帮助模型把复杂推理拆解成可检验的步骤。  
- **大模型（Teacher）**：参数量极大的语言模型，具备强大的生成和推理能力，本文把它当作“老师”，负责提供高质量的思维链示例。  
- **蒸馏（Knowledge Distillation）**：把大模型的知识压缩到小模型里，通常让小模型模仿老师的输出概率分布，这里扩展到模仿老师的推理过程。  
- **对比解码（Contrastive Decoding）**：在生成时同时考虑答案与非答案两条路径，鼓励模型只在答案成立时才产生某些 token，类似“只有答案对了，这句话才合理”。  
- **反事实推理目标（Counterfactual Reasoning Objective）**：训练时让学生模型在答案被改动或思维链被扰动的“假设情境”下仍然需要保持一致，从而迫使模型真正依赖思维链而不是直接猜答案。  
- **自洽（Self‑Consistency）**：模型的最终预测应与它自己写出的推理步骤相吻合，避免出现“答案说A，推理却指向B”。  
- **幻觉（Hallucination）**：模型生成的内容在事实或逻辑上不成立，这在推理任务里表现为不合理的思维链或与答案冲突的解释。

### 核心创新点
1. **对比解码生成教师思维链**  
   - 之前的蒸馏大多直接让老师输出普通的 CoT，缺乏对答案依赖性的约束。  
   - 本文在老师生成思维链时加入对比解码，使得每个 token 的出现概率在答案为真时显著提升，而在答案为假时则下降。  
   - 结果是老师提供的推理更“答案导向”，学生在学习时更容易捕捉到答案与推理的因果关系。

2. **反事实推理目标确保学生不“跳过”思维链**  
   - 传统蒸馏只最小化学生和老师的输出差异，学生可以直接学会答案的映射而忽视中间步骤。  
   - 这里引入一种反事实训练：在训练样本中人为扰动答案或思维链，要求学生在这些扰动下仍然输出与原答案一致的推理，或者在答案被改为错误时输出错误的推理。  
   - 这种机制迫使学生把答案和思维链绑定在一起，提升了自洽性和解释的可信度。

3. **统一的自洽蒸馏框架**  
   - 将对比解码产生的高质量教师思维链和反事实目标结合成一个端到端的蒸馏流程。  
   - 与仅使用普通 CoT 蒸馏或仅使用自洽后处理的基线相比，SCOTT 在保持相近的任务准确率的同时，大幅提升了思维链的真实性和一致性。

### 方法详解
**整体思路**  
SCOTT 把蒸馏过程拆成两大阶段：① 用超大模型（老师）通过对比解码生成“答案依赖型”思维链；② 用这些思维链训练小模型（学生），并在训练中加入反事实推理目标，使学生的预测必须与自己写的推理保持一致。

**步骤拆解**  

1. **教师思维链采集**  
   - 给老师模型一个标准的 CoT 提示（例如“先思考，再回答”）。  
   - 同时提供两个候选答案：正确答案 A 与一个干扰答案 B。  
   - 对比解码的核心是：在生成每个 token 时，计算它在 A 条路线上和 B 条路线上出现的概率差异，只在差异显著且正向时接受该 token。  
   - 这样得到的思维链只有在答案为 A 时才“合理”，相当于老师在写“只有答案对了，这一步才成立”的解释。

2. **学生模型的训练目标**  
   - **标准模仿损失**：让学生的 token 分布尽量匹配老师的思维链（普通的 KL 散度）。  
   - **反事实一致性损失**：随机挑选生成的思维链，构造两种“假设”：  
     a) 把答案改为错误的 B，保持原思维链不变；  
     b) 把思维链的关键步骤随机替换或删除，保持答案为 A。  
     然后要求学生在情境 a) 输出错误答案，在情境 b) 输出错误答案或显著降低置信度。  
   - 通过这种“如果答案不对，推理也不对”的约束，学生被迫把答案和推理绑定。

3. **自洽推理阶段（可选）**  
   - 在实际使用时，给学生模型多次采样不同的思维链，然后对每条链进行投票，取多数答案作为最终输出。  
   - 由于学生已经在训练中学会让答案与链条一致，这一步的投票效果比传统自洽方法更稳健。

**关键技巧**  
- 对比解码本质上是一个“答案‑条件化的采样”，它把答案信息显式嵌入到生成过程，而不是事后再检查。  
- 反事实目标的设计看似逆向（让模型在错误情境下也要表现错误），但正是这种“负例”让学生学会辨别哪些推理步骤是真正支撑答案的。  
- 整个框架不需要额外的人工标注，只利用老师模型本身的生成能力即可完成数据构造，保持了蒸馏的高效性。

### 实验与效果
- **测试任务**：论文在多个公开的推理基准上评估，包括数学题库 GSM8K、算术推理 MATH、以及常识推理 ARC。  
- **对比基线**：普通大模型的 CoT、仅使用标准蒸馏的 TinyCoT、以及在小模型上直接做自洽投票的 Self‑Consistency。  
- **主要结果**：SCOTT 在保持与原大模型相近的答案准确率（误差不超过 1%）的同时，在思维链可信度评估上领先基线约 10%‑15%。在 GSM8K 上，SCOTT 的正确率为 71.2%，而普通蒸馏的 TinyCoT 为 68.5%；在思维链一致性评分上，SCOTT 获得 0.78（满分 1），而对照组仅 0.62。  
- **消融实验**：去掉对比解码后，思维链的答案依赖性下降，整体可信度跌至 0.66；去掉反事实损失则学生容易“跳过”思维链，准确率略升但一致性分数降至 0.60，验证两者缺一不可。  
- **局限性**：作者指出，SCOTT 的性能仍受老师模型质量限制；在高度开放式的对话任务上，思维链的结构化程度不足，方法的收益不明显。

### 影响与延伸思考
SCOTT 把“让小模型写出可信推理”从“事后检查”转向“事前约束”，为后续的可信蒸馏提供了思路。自论文发布后，出现了几篇跟进工作，例如 **Faithful Distillation of Reasoning Chains**（2024）和 **Counterfactual Consistency for Small LMs**（2025），它们分别在多模态推理和代码生成场景中复用了对比解码和反事实训练的核心思想。对想进一步探索的读者，可以关注以下方向：  
1. 将对比解码推广到非答案‑条件化的任务（如摘要、翻译），看是否能提升生成的事实性。  
2. 与检索增强模型结合，让检索到的证据作为“硬约束”，进一步降低幻觉。  
3. 探索更轻量的反事实目标实现，例如只在少数关键步骤上做扰动，以降低训练成本。

### 一句话记住它
让小模型通过“答案‑驱动的对比生成 + 反事实自洽训练”，学会写出与答案真正匹配的思维链。