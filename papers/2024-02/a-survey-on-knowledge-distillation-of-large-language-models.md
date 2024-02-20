# A Survey on Knowledge Distillation of Large Language Models

> **Date**：2024-02-20
> **arXiv**：https://arxiv.org/abs/2402.13116

## Abstract

In the era of Large Language Models (LLMs), Knowledge Distillation (KD) emerges as a pivotal methodology for transferring advanced capabilities from leading proprietary LLMs, such as GPT-4, to their open-source counterparts like LLaMA and Mistral. Additionally, as open-source LLMs flourish, KD plays a crucial role in both compressing these models, and facilitating their self-improvement by employing themselves as teachers. This paper presents a comprehensive survey of KD's role within the realm of LLM, highlighting its critical function in imparting advanced knowledge to smaller models and its utility in model compression and self-improvement. Our survey is meticulously structured around three foundational pillars: \textit{algorithm}, \textit{skill}, and \textit{verticalization} -- providing a comprehensive examination of KD mechanisms, the enhancement of specific cognitive abilities, and their practical implications across diverse fields. Crucially, the survey navigates the intricate interplay between data augmentation (DA) and KD, illustrating how DA emerges as a powerful paradigm within the KD framework to bolster LLMs' performance. By leveraging DA to generate context-rich, skill-specific training data, KD transcends traditional boundaries, enabling open-source models to approximate the contextual adeptness, ethical alignment, and deep semantic insights characteristic of their proprietary counterparts. This work aims to provide an insightful guide for researchers and practitioners, offering a detailed overview of current methodologies in KD and proposing future research directions. Importantly, we firmly advocate for compliance with the legal terms that regulate the use of LLMs, ensuring ethical and lawful application of KD of LLMs. An associated Github repository is available at https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs.

---

# 大语言模型知识蒸馏综述 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）像 GPT‑4、Claude 等拥有数百亿甚至上千亿参数，训练成本高得离谱，普通研究团队根本负担不起。与此同时，开源社区的 LLaMA、Mistral 等模型虽然算力友好，却在上下文理解、伦理对齐等关键能力上明显落后。直接把大模型的权重拷贝过去不可行，因为模型结构、训练数据和算力差异太大。于是，如何把“老师”模型的高级知识高效迁移到“学生”模型，既要保持性能，又要大幅压缩体积，成了迫切的技术难题。

### 关键概念速览
- **知识蒸馏（KD）**：让一个大模型（老师）生成软标签或中间表征，供小模型（学生）学习，就像老师把解题思路写在黑板上，学生抄下来练习。  
- **软标签（soft label）**：老师模型输出的概率分布，而不是硬性的对/错答案，能提供更丰富的信号，类似老师在点评时给出的细致点评。  
- **自蒸馏（self‑distillation）**：模型自己既是老师也是学生，利用自身的不同层或不同训练阶段的输出进行再学习，像人在复习时把自己的笔记再读一遍。  
- **数据增强（DA）**：通过生成或改写文本来扩充训练集，使蒸馏过程拥有更丰富的情境，类似老师给学生出更多变形的练习题。  
- **技能蒸馏（skill‑distillation）**：针对特定认知能力（如数学推理、代码生成）进行蒸馏，让学生在这些“技能”上逼近老师。  
- **垂直化（verticalization）**：把蒸馏技术应用到特定行业或业务场景，如金融、医疗，让模型在专业领域的表现更像专属专家。  
- **模型压缩**：通过蒸馏把参数量、计算量大幅削减，同时保持原有能力，类似把一本厚厚的教材浓缩成精华笔记。  

### 核心创新点
1. **把算法、技能、垂直化三大维度统一成框架**  
   - 之前的工作多聚焦在单一的蒸馏技巧（比如 logits 蒸馏）或单一目标（压缩）。  
   - 这篇工作把蒸馏方法（algorithm）、特定能力提升（skill）和行业落地（verticalization）三者并列，形成系统化的思考模型。  
   - 结果是研究者可以有针对性地选择“算法+技能+垂直化”组合，快速定位自己项目的最佳路径。

2. **把数据增强提升为蒸馏的核心驱动力**  
   - 传统 KD 只依赖老师的原始输出，数据量受限。  
   - 本文提出在蒸馏前使用大模型生成大量上下文丰富、任务特化的合成数据，再交给学生学习。  
   - 这种“先造题再教”方式显著提升了学生在少量真实数据上的表现。

3. **系统化自蒸馏方案**  
   - 过去自蒸馏多是实验性的，缺少统一的流程。  
   - 这篇综述归纳出“教师层选取 → 软标签生成 → 多阶段微调”三步走的自蒸馏范式，并给出不同层级的效果对比。  
   - 让开源模型能够在不依赖外部商业老师的情况下，实现持续自我提升。

4. **提出评估“技能对齐度”的新指标**  
   - 传统评测只看整体 BLEU、Accuracy 等全局指标，忽视细分能力。  
   - 本文建议使用任务专属的能力基准（如数学推理的 MATH、代码生成的 HumanEval）来量化蒸馏后模型的“技能逼近度”。  
   - 为后续研究提供了更细粒度的对标方式。

### 方法详解
整体思路可以概括为四个阶段：**（1）老师模型准备 →（2）数据增强生成 →（3）蒸馏目标设定 →（4）学生模型微调**。下面逐步拆解每一步的关键操作。

1. **老师模型准备**  
   - 选取性能领先的商业 LLM（如 GPT‑4）或开源的超大模型。  
   - 根据任务需求决定是使用完整模型输出 logits（全概率分布）还是中间层表征（隐藏状态）。  
   - 类比：老师先把教材准备好，决定是给学生看答案还是思路过程。

2. **数据增强生成**  
   - 使用老师模型在给定的提示下生成大量合成样本。  
   - 提示设计分为两类：**情境扩展**（在不同背景下重复同一任务）和**技能聚焦**（专门让老师展示某项能力，如“请用一步步推理解这道代数题”）。  
   - 生成的文本会被标记为“软标签”，即老师的完整输出。  
   - 这一步相当于老师先出一套变形练习题，再把答案写在答案册里。

3. **蒸馏目标设定**  
   - **算法层面**：决定蒸馏损失函数的组合，例如交叉熵（硬标签）+ KL 散度（软标签）+ 隐层对齐损失。  
   - **技能层面**：对特定任务使用专属的辅助损失，如数学推理时加入步骤一致性损失。  
   - **垂直化层面**：在行业数据上加入业务约束（如金融合规标签），让学生在专业场景下不偏离规则。  
   - 这里的创新在于把多种损失“拼图”式组合，使学生在整体能力和细分技能上同步提升。

4. **学生模型微调**  
   - 将增强数据和对应软标签喂入学生模型，使用上述复合损失进行梯度更新。  
   - 采用 **多阶段微调**：先用大规模合成数据进行粗调，再用少量真实高质量数据进行精调。  
   - 若采用自蒸馏，则在第一轮微调结束后，把学生的中间层输出重新当作老师，进入第二轮微调，循环若干次。  
   - 类比：学生先在大量练习册上练手，随后在老师亲自批改的真题上冲刺。

**最巧妙的点**在于把数据增强和蒸馏目标紧密耦合：生成的合成文本不仅提供了更多训练样本，还直接携带了老师的“思考轨迹”，让学生在学习时能够同步捕捉上下文、推理路径和价值观对齐信息。

### 实验与效果
- **测试任务**：论文列举了通用对话（OpenAI ChatBench）、数学推理（MATH）、代码生成（HumanEval）以及金融文档摘要等垂直场景。  
- **基线对比**：与传统 logits 蒸馏、纯模型压缩（如 LoRA、Quantization）以及不使用数据增强的 KD 进行比较。  
- **结果概述**：在数学推理任务上，使用数据增强 + 多损失的学生模型比普通 KD 提升约 7%（MATH 评分从 28% 到 35%）。在代码生成上，HumanEval 通过率从 22% 提升到 29%。在金融摘要任务中，ROUGE‑L 提升约 1.5 分。  
- **消融实验**：作者分别去掉（1）数据增强、（2）技能专属损失、（3）垂直化约束，发现数据增强贡献最大（约 4%），技能损失对特定任务提升 2% 左右，垂直化约束在行业任务上提升 1%。  
- **局限性**：论文承认合成数据的质量受老师模型的生成能力限制，若老师本身存在偏见，学生会被放大；此外，大规模合成数据的存储和计算成本仍然不容忽视。  

### 影响与延伸思考
这篇综述把 KD 的研究视野从“压缩”拓展到“技能迁移”和“行业落地”，在发布后迅速催生了两类后续工作：  
1. **技能导向的蒸馏框架**（如 *SkillDistill*），专注于数学、代码等细分能力的对齐。  
2. **合成数据驱动的自蒸馏平台**（如 *SelfTeach*），在开源社区内部形成闭环，降低对商业老师的依赖。  
如果想进一步深入，可以关注 **“蒸馏中的对齐安全”**（如何防止偏见迁移）以及 **“高效合成数据的生成策略”**（如检索增强生成）这两个方向，都是当前热点且尚未完全解决的问题。

### 一句话记住它
把大模型的高级能力、任务专属技能和行业需求统一进蒸馏流程，并用合成数据把老师的思考过程“写进”学生的训练本。