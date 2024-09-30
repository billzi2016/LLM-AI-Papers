# Instance-adaptive Zero-shot Chain-of-Thought Prompting

> **Date**：2024-09-30
> **arXiv**：https://arxiv.org/abs/2409.20441

## Abstract

Zero-shot Chain-of-Thought (CoT) prompting emerges as a simple and effective strategy for enhancing the performance of large language models (LLMs) in real-world reasoning tasks. Nonetheless, the efficacy of a singular, task-level prompt uniformly applied across the whole of instances is inherently limited since one prompt cannot be a good partner for all, a more appropriate approach should consider the interaction between the prompt and each instance meticulously. This work introduces an instance-adaptive prompting algorithm as an alternative zero-shot CoT reasoning scheme by adaptively differentiating good and bad prompts. Concretely, we first employ analysis on LLMs through the lens of information flow to detect the mechanism under zero-shot CoT reasoning, in which we discover that information flows from question to prompt and question to rationale jointly influence the reasoning results most. We notice that a better zero-shot CoT reasoning needs the prompt to obtain semantic information from the question then the rationale aggregates sufficient information from the question directly and via the prompt indirectly. On the contrary, lacking any of those would probably lead to a bad one. Stem from that, we further propose an instance-adaptive prompting strategy (IAP) for zero-shot CoT reasoning. Experiments conducted with LLaMA-2, LLaMA-3, and Qwen on math, logic, and commonsense reasoning tasks (e.g., GSM8K, MMLU, Causal Judgement) obtain consistent improvement, demonstrating that the instance-adaptive zero-shot CoT prompting performs better than other task-level methods with some curated prompts or sophisticated procedures, showing the significance of our findings in the zero-shot CoT reasoning mechanism.

---

# 实例自适应零样本思维链提示 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做推理，最常用的技巧是“零样本思维链”（Zero-shot CoT），即直接在模型前放一段通用提示，让它先写出推理步骤再给答案。可是，这种“一刀切”的提示只能覆盖一小部分问题——不同题目需要的引导信息差别很大，统一的提示往往要么信息不足，要么引导偏离，导致推理质量波动。换句话说，模型在没有针对每个实例调节提示的情况下，难以充分利用题目本身的语义信息。于是，如何让同一个模型在每个具体问题上都能得到最匹配的提示，成为提升零样本推理效果的关键瓶颈。

### 关键概念速览
- **Zero-shot CoT（零样本思维链）**：不给模型提供示例，只给一段通用提示，让它自行生成推理过程，就像老师只说“先列出思路再回答”，不提供具体例子。
- **Prompt（提示词）**：模型的输入前缀，用来引导模型的行为。相当于在对话前先说一句“请先思考”，不同的提示会让模型走不同的思路。
- **信息流（Information Flow）**：指模型内部信息从输入（问题）向提示、再向推理过程（理由）传递的路径。可以想象为水流从源头（问题）分支到两条管道（提示和理由），最终汇聚成答案。
- **实例自适应（Instance-adaptive）**：针对每一道具体题目动态决定使用哪个提示，而不是一次性固定。类似于老师根据学生的提问内容即时调整提问方式。
- **好提示 / 坏提示**：在某个实例上，提示能否帮助模型抓住关键信息并生成完整推理。好提示像是给出正确的线索，坏提示则像是误导性的暗示。
- **IAP（Instance-adaptive Prompting）**：本文提出的实例自适应提示算法，核心是先评估每个提示在当前问题上的潜在效果，再挑选最合适的一个使用。

### 核心创新点
1. **从信息流视角揭示零样本 CoT 机制**  
   之前的研究只把提示当作固定的前置文字，没系统分析信息是怎么在模型内部流动的。本文通过实验发现，模型的推理结果受“问题→提示”和“问题→理由”两条信息通路共同影响。这个发现让人明白，仅靠提示本身不足，还必须让理由直接吸收问题信息。  
   *改变*：为后续的自适应选择提供了明确的评估依据——只要这两条通路都通畅，就可能得到好答案。

2. **提出实例自适应提示选择机制（IAP）**  
   传统零样本 CoT 只用一条任务级提示，本文设计了一个“好坏判别器”，先让模型在同一问题上尝试多个候选提示，观察信息流强度（如注意力分布或中间激活），再根据得分挑选最优提示。  
   *改变*：实现了对每一道题目都能动态匹配最合适的提示，显著提升了推理的稳健性。

3. **在多个主流模型上统一验证**  
   作者把 IAP 同时跑在 LLaMA‑2、LLaMA‑3、Qwen 三大模型上，覆盖数学、逻辑、常识三类基准。实验显示，IAP 的提升幅度与模型规模无关，说明方法的通用性。  
   *改变*：证明了自适应提示不是针对特定模型的技巧，而是一种普适的提升思路。

### 方法详解
**整体框架**  
IAP 主要分三步：①准备一组候选提示；②对每个实例评估每个提示的“信息流质量”；③选出得分最高的提示并正式生成答案。整个流程在推理阶段完成，不需要额外的微调或示例数据。

**步骤拆解**  
1. **候选提示库构建**  
   作者手工或自动生成若干通用的 CoT 提示，例如“先思考一步，再给出答案”。这些提示覆盖不同的引导方式（强调问题、强调推理、混合等），相当于准备了几把不同形状的钥匙。

2. **信息流评估**  
   - 将同一道问题分别与每个提示拼接，送入 LLM。  
   - 通过模型内部的注意力权重或激活值，量化两条关键通路的强度：  
     *问题→提示*：看模型在生成提示时是否引用了问题的关键词；  
     *问题→理由*：看模型在写推理步骤时直接引用了问题信息的频率。  
   - 这一步不需要真实答案，只是观察模型内部的“注意力分配”。可以把它想成让模型先自检一次，判断哪把钥匙最能打开这道门。

3. **自适应选择**  
   将每个提示的两条通路得分加权合并，得到一个综合分数。分数最高的提示被认定为“好提示”，随后模型在同一问题上重新生成完整的 CoT 推理并输出答案。

**巧妙之处**  
- **不依赖外部标签**：评估只用模型内部信号，省去人工标注或额外数据。  
- **信息流双通路**：同时关注提示对问题的吸收和理由对问题的直接引用，比只看提示本身更全面。  
- **一次性评估**：虽然要跑多次前向传播，但每次只产生短文本，计算开销相对可接受，且可以并行化。

### 实验与效果
- **测试任务**：数学推理（GSM8K）、学科知识（MMLU）、因果判断（Causal Judgement）等多模态推理基准。  
- **对比基线**：标准零样本 CoT（单一任务级提示）、手工精选的多提示集合、以及一些需要额外步骤的零样本方法。  
- **结果**：论文声称在上述数据集上，IAP 相比最强的任务级提示提升了数个百分点的准确率，且在不同模型（LLaMA‑2、LLaMA‑3、Qwen）上均保持正向增益。  
- **消融实验**：去掉信息流评估或只保留单一通路会导致性能回落，说明两条信息流的联合评估是关键因素。  
- **局限**：评估过程需要对每个实例多次前向传播，推理时间比单提示方案高；此外，候选提示的质量仍依赖人工设计，完全自动化仍是开放问题。

### 影响与延伸思考
IAP 把“提示选择”提升到实例层面，打开了零样本推理的细粒度调优思路。随后的工作开始探索更大规模的提示库、利用检索模型自动生成候选提示，甚至把信息流评估嵌入到模型的自注意力机制中，实现端到端的自适应 CoT。对想进一步研究的读者，可以关注以下方向：①提示自动生成与筛选的机器学习方法；②把信息流评估与强化学习结合，学习最优提示策略；③在多模态或对话场景下扩展实例自适应提示的适用性。整体来看，这篇论文为“让模型自己挑提示”提供了可操作的框架，推动了零样本推理从“一次性”向“动态化”转变。

### 一句话记住它
让大模型在每道题上先自检信息流强度，挑出最匹配的提示再推理——这就是实例自适应零样本思维链。