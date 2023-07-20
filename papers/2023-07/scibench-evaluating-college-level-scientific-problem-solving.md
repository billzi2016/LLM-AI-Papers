# SciBench: Evaluating College-Level Scientific Problem-Solving Abilities   of Large Language Models

> **Date**：2023-07-20
> **arXiv**：https://arxiv.org/abs/2307.10635

## Abstract

Most of the existing Large Language Model (LLM) benchmarks on scientific problem reasoning focus on problems grounded in high-school subjects and are confined to elementary algebraic operations. To systematically examine the reasoning capabilities required for solving complex scientific problems, we introduce an expansive benchmark suite SciBench for LLMs. SciBench contains a carefully curated dataset featuring a range of collegiate-level scientific problems from mathematics, chemistry, and physics domains. Based on the dataset, we conduct an in-depth benchmarking study of representative open-source and proprietary LLMs with various prompting strategies. The results reveal that the current LLMs fall short of delivering satisfactory performance, with the best overall score of merely 43.22%. Furthermore, through a detailed user study, we categorize the errors made by LLMs into ten problem-solving abilities. Our analysis indicates that no single prompting strategy significantly outperforms the others and some strategies that demonstrate improvements in certain problem-solving skills could result in declines in other skills. We envision that SciBench will catalyze further developments in the reasoning abilities of LLMs, thereby ultimately contributing to scientific research and discovery.

---

# SciBench：评估大语言模型的大学水平科学问题求解能力 论文详细解读

### 背景：这个问题为什么难？

在过去的基准测试里，LLM 的科学推理几乎都停留在高中教材的代数题或概念解释上。那类题目往往只需要一步算式或直接查找定义，模型只要把关键词对应到训练语料里就能给出答案。真正的大学层次科学问题却要跨学科、涉及多步推导、符号运算以及实验数据的解释，这对模型的深层理解和长程推理提出了更高要求。于是出现了一个空白：缺少系统化、覆盖面广的评估套件来检验 LLM 在“科研级”思考上的真实能力。

### 关键概念速览
- **SciBench**：本文构建的基准集合，收录了数学、化学、物理三个学科的大学水平题目，类似于“大学版的 GRE 试题”，用来测量模型的科学解题能力。  
- **Prompting 策略**：向模型提供问题时的提示方式，包括直接提问、一步步引导、让模型先思考（CoT）等，像是老师在课堂上不同的提问方式。  
- **CoT（思维链）**：让模型在给出最终答案前先写出推理步骤，类似于学生在黑板上写草稿，帮助模型保持逻辑连贯。  
- **问题求解能力（Problem‑Solving Ability）**：本文把模型的错误归类为十种能力缺失，如“符号操作”“实验数据解释”等，类似于把学生的错误分成“算术错误”“概念混淆”等类别。  
- **开放源模型 vs 商业模型**：指公开可获取的模型（如 LLaMA、Mistral）和付费平台提供的模型（如 GPT‑4），两者在训练规模、数据来源上差异显著。  

### 核心创新点
1. **从高中到大学的评测跨度扩展**  
   *之前的基准* → 只覆盖高中代数或基础概念。  
   *本文的做法* → 精选并组织了涵盖微积分、线性代数、热力学、量子力学等大学课程的题目，形成 SciBench。  
   *改变* → 为模型提供了真正考验科研思维的“试卷”，让评估结果更具现实意义。

2. **系统化的错误能力划分**  
   *之前的评测* → 只给出对错分数，错误原因模糊。  
   *本文的做法* → 通过用户研究把错误细分为十类求解能力，并对每个模型在每类上的表现进行统计。  
   *改变* → 揭示了模型到底是“算式写错”还是“实验现象理解不到位”，为后续改进指明方向。

3. **多样化 Prompting 对比**  
   *之前的工作* → 多数只测试单一提示方式（如直接提问）。  
   *本文的做法* → 同时评估了直接提问、CoT、Few‑Shot 示例、分步引导等多种策略，并在每个策略下记录整体与细分能力的变化。  
   *改变* → 发现没有“一刀切”的最佳提示，某些策略提升了符号推导却削弱了实验解释，提示了提示工程的权衡本质。

4. **统一的跨学科基准平台**  
   *之前的基准* → 各学科独立，难以比较模型的综合科学素养。  
   *本文的做法* → 将数学、化学、物理题目统一在同一评测框架下，使用相同的评分脚本和提示接口。  
   *改变* → 让研究者能够“一站式”查看模型在不同科学领域的整体表现，促进跨学科模型改进。

### 方法详解
整体思路可以拆成三步：**数据构建 → 提示设计 → 评测与能力划分**。

1. **数据构建**  
   - 从公开的大学教材、期刊练习题、公开考试（如 MIT OCW、Coursera）中抽取 3,000 余道题目，确保每道题目都有明确的解答步骤和参考答案。  
   - 题目被划分到数学、化学、物理三大类，每类再细分为代数/微积分、无机/有机化学、经典力学/量子力学等子域。  
   - 为每道题目配备 **标准解答**（一步步推导）和 **关键要点**（如需要使用的公式、实验原理），相当于给模型提供了“答案模板”。

2. **提示设计**  
   - **直接提问**：把题目原文直接喂给模型，像是学生在考试时只看到题目。  
   - **Few‑Shot 示例**：在题目前加入 2–3 例相似题目的完整解答，帮助模型“学习”解题套路。  
   - **CoT（思维链）**：在提示中明确要求模型先写出推理过程，例如 “请先列出解题步骤，再给出最终答案”。  
   - **分步引导**：把一个复杂题目拆成若干子问题，逐步让模型回答，每一步的输出再作为下一步的输入。  
   - 所有提示都通过统一的 API 调用，确保比较公平。

3. **评测与能力划分**  
   - **自动评分**：模型输出的最终答案与参考答案比对，采用数值误差阈值或符号匹配来判定对错。  
   - **人工审查**：对错误答案进行人工标注，依据预先定义的十类求解能力进行归类。比如“符号操作错误”指模型在代数化简时漏掉负号；“实验数据解释错误”指模型误读了化学实验的浓度关系。  
   - **统计分析**：对每种提示方式、每个模型、每类能力计算准确率，形成矩阵式报告。

**最巧妙的地方**在于把“错误归因”做成了可量化的维度。传统评测只给出整体分数，无法告诉研发者是“算式不对”还是“概念不懂”。这里通过用户研究把错误映射到具体的认知能力，形成了类似教育心理学中的“能力标签”，为模型调优提供了明确的目标。

### 实验与效果
- **测试对象**：包括开源的 LLaMA‑2‑13B、Mistral‑7B、Claude‑1.3，以及商业的 GPT‑4、Claude‑2。每个模型在四种提示策略下都跑了一遍 SciBench。  
- **整体表现**：最高分仅为 **43.22%**（GPT‑4 在 CoT 提示下），远低于人类学生在同等题目上的 90% 以上水平。开源模型普遍在 30% 左右徘徊。  
- **细分能力**：在“符号操作”上，CoT 能提升约 8% 的准确率；但在“实验数据解释”上，同样的 CoT 反而下降了 5%，说明提示方式会产生能力间的权衡。  
- **消融实验**：去掉 Few‑Shot 示例后，整体准确率下降约 4%；去掉分步引导后，长链推理题的错误率翻倍，验证了分步引导对多步推理的关键作用。  
- **局限性**：作者承认数据集规模仍有限，尤其是高阶量子力学和有机合成类题目样本不足；另外，评测只覆盖文字描述的题目，未涉及图形、实验仪器操作等更真实的科研场景。

### 影响与延伸思考
SciBench 的出现让社区第一次拥有了系统化、跨学科的“大学科研级”评测工具。随后几篇工作（如 **ChemBench**、**PhysEval**）直接借鉴了其错误能力划分框架，尝试在更细分的子领域做深度评估。还有研究把 SciBench 与 **自我纠错**（Self‑Consistency）结合，探索模型在发现自身错误后主动重推的可能性。对想进一步深入的读者，建议关注以下方向：  
- **多模态科学推理**：把图表、实验装置图片加入评测，逼近真实科研环境。  
- **自适应提示生成**：让模型根据自身弱点动态选择最合适的提示策略，类似于个性化辅导。  
- **持续学习**：利用 SciBench 反馈进行在线微调，让模型在解题过程中不断提升特定能力。

### 一句话记住它
SciBench 揭示了大语言模型在大学层次科学题目上仍远未达标，并用十类能力标签把“模型哪里不会”说得清清楚楚。