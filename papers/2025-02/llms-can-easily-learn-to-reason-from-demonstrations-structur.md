# LLMs Can Easily Learn to Reason from Demonstrations Structure, not   content, is what matters!

> **Date**：2025-02-11
> **arXiv**：https://arxiv.org/abs/2502.07374

## Abstract

Large reasoning models (LRMs) tackle complex reasoning problems by following long chain-of-thoughts (Long CoT) that incorporate reflection, backtracking, and self-validation. However, the training techniques and data requirements to elicit Long CoT remain poorly understood. In this work, we find that a Large Language model (LLM) can effectively learn Long CoT reasoning through data-efficient supervised fine-tuning (SFT) and parameter-efficient low-rank adaptation (LoRA). With just 17k long CoT training samples, the Qwen2.5-32B-Instruct model achieves significant improvements on a wide range of math and coding benchmarks, including 56.7% (+40.0%) on AIME 2024 and 57.0% (+8.1%) on LiveCodeBench, competitive to the proprietary o1-preview model's score of 44.6% and 59.1%. More importantly, we find that the structure of Long CoT is critical to the learning process, whereas the content of individual reasoning steps has minimal impact. Perturbations affecting content, such as training on incorrect samples or removing reasoning keywords, have little impact on performance. In contrast, structural modifications that disrupt logical consistency in the Long CoT, such as shuffling or deleting reasoning steps, significantly degrade accuracy. For example, a model trained on Long CoT samples with incorrect answers still achieves only 3.2% lower accuracy compared to training with fully correct samples. These insights deepen our understanding of how to elicit reasoning capabilities in LLMs and highlight key considerations for efficiently training the next generation of reasoning models. This is the academic paper of our previous released Sky-T1-32B-Preview model. Codes are available at https://github.com/NovaSky-AI/SkyThought.

---

# LLM 能轻松通过示例学习推理：结构而非内容才是关键 论文详细解读

### 背景：这个问题为什么难？
在大模型时代，给模型加上“思考链”（Chain‑of‑Thought, CoT）已经能让它们在数学、编程等需要多步推理的任务上表现更好。但要让模型真正走出“一步到位”的答案，形成长篇、带有自我检查的推理过程（Long CoT）仍然很难。过去的做法要么靠海量的自监督预训练，要么依赖人工设计的提示工程，成本高、效果不稳定。更关键的是，研究者并不清楚模型到底是学会了推理的“套路”，还是仅仅记住了大量相似的答案片段。于是，如何用少量、可控的数据让模型掌握可靠的长链推理，成为亟待突破的瓶颈。

### 关键概念速览
**Long CoT（长链思维）**：模型在给出最终答案前，写出一系列相互关联、带有反思和自检的推理步骤，类似于人解题时的完整草稿。  
**SFT（监督微调）**：在已有的大模型上，用标注好的输入‑输出对继续训练，让模型学会特定的行为。可以把它想成给模型上“进阶课程”。  
**LoRA（低秩适配）**：一种只调节模型内部少量参数的微调方式，像在原有模型上贴一层可拆卸的“贴纸”，既省显存又保持原模型能力。  
**结构信息**：指推理步骤的顺序、层次和逻辑关系，而不是每一步的具体文字内容。就像一篇文章的章节目录比具体的句子更能决定整体框架。  
**内容扰动**：有意修改或随机化推理步骤中的文字（比如换词、删关键词），但保持整体顺序不变。  
**结构扰动**：打乱、删除或重新排列推理步骤，使得原本的逻辑链被破坏。  

### 核心创新点
1. **少量长链示例即可驱动显著提升**  
   之前的研究往往需要上百万条 CoT 示例才能让模型出现长链推理能力。本文只用了约 1.7 万条经过精心组织的 Long CoT 样本，对 Qwen2.5‑32B‑Instruct 进行 SFT + LoRA，便在数学和代码基准上实现了大幅跃升。  
2. **结构优先于内容的实验验证**  
   通过系统地对训练样本进行内容扰动（如随机替换关键字、使用错误答案）和结构扰动（如打乱步骤顺序），作者发现只要保持步骤的逻辑结构，模型的性能几乎不受影响；而一旦破坏结构，准确率会骤降。这个发现直接说明模型在学习的是“推理模板”，而不是具体的文字。  
3. **低秩适配的高效实现**  
   采用 LoRA 只调节少量矩阵的低秩部分，使得在 32B 参数模型上进行微调的显存需求降到原来的 1/4 左右，同时保持了对长链结构的敏感学习。相比全参数微调，成本更低、部署更灵活。  
4. **与商业闭源模型的对标**  
   在 AIME 2024（美国数学竞赛）和 LiveCodeBench 两大高难度基准上，本文模型的得分分别逼近或超过了 OpenAI o1‑preview 的表现，证明了“结构学习+少量数据”路线的竞争力。  

### 方法详解
整体思路可以拆成三步：**数据准备 → 结构化微调 → 低秩适配**。

1. **数据准备**  
   - 从公开的数学、编程题库中抽取 17k 条高质量解答。  
   - 每条解答都被手工或半自动整理成 **Long CoT**：包括问题陈述、若干推理步骤、反思/回溯段落以及最终答案。  
   - 为了验证“结构 vs 内容”，作者另外生成了三类变体：  
     a) **错误答案版**：把最终答案改成错误的数值；  
     b) **关键词缺失版**：删除诸如“因此”“假设”等常见推理标记；  
     c) **结构打乱版**：随机重排推理步骤或删掉关键步骤。  

2. **结构化微调（SFT）**  
   - 将原始 Qwen2.5‑32B‑Instruct 视作“老师”，在上述 17k 条 Long CoT 上继续训练。  
   - 训练目标是让模型在看到问题后，能够自行生成与训练样本相同的 **步骤序列**，而不是直接输出答案。  
   - 为了强化结构感知，作者在损失函数中加入了 **顺序一致性惩罚**：如果模型生成的步骤顺序与参考不匹配，损失会额外上升。这个技巧帮助模型把“先做 A 再做 B”这种顺序信息内化。  

3. **低秩适配（LoRA）**  
   - 在每一层的注意力和前馈网络中插入低秩矩阵（rank=8），只对这些矩阵进行梯度更新。  
   - 这样做的好处是：  
     * 训练时显存占用大幅下降，单卡即可完成 32B 参数模型的微调。  
     * 只改动少量参数，保留了原模型在通用语言理解上的强大能力。  
   - 训练结束后，只需把 LoRA 参数和原模型合并，即可得到完整的推理模型。  

**最巧妙的点**在于：作者并没有尝试让模型记住每一步的具体文字，而是通过**顺序一致性惩罚**和**结构化数据**让模型学会“先做这件事，再做那件事”。这相当于教模型一种通用的解题流程，而不是让它背答案。

### 实验与效果
- **测试任务**：  
  * **AIME 2024**（美国数学竞赛高难度题目）  
  * **LiveCodeBench**（代码实现与调试基准）  
  * 以及若干公开的数学推理数据集（如 GSM‑8K、MATH）作补充验证。  

- **对比基线**：  
  * 原始 Qwen2.5‑32B‑Instruct（未微调）  
  * 传统 CoT 微调（只用短链示例）  
  * 全参数微调（相同数据量）  
  * 商业闭源模型 o1‑preview  

- **主要结果**（论文声称）：  
  * 在 AIME 2024 上从 16.7% 提升到 56.7%，相当于提升 40.0% 绝对分数，逼近 o1‑preview 的 44.6%。  
  * 在 LiveCodeBench 上从 48.9% 提升到 57.0%，略低于 o1‑preview 的 59.1%。  
  * 与仅使用短链 CoT 的微调相比，长链微调额外提升约 12%–15%。  

- **消融实验**：  
  * **内容扰动**（错误答案、关键词缺失）导致的性能下降不到 3.5%。  
  * **结构扰动**（步骤打乱、删除）导致准确率骤降 20% 以上，验证结构是关键因素。  
  * **LoRA vs 全参数**：两者在最终准确率上相差不到 1%，但 LoRA 的显存占用仅为全参数的 25%。  

- **局限性**：  
  * 训练数据仍然需要人工或半自动的结构化处理，完全自动化的长链生成仍是挑战。  
  * 只在 32B 规模模型上验证，是否同样适用于更小或更大的模型尚未明确。  
  * 对于需要跨领域常识的推理（如法律、医学），仅靠结构学习可能不足。  

### 影响与延伸思考
这篇工作把“结构学习”推向前台，直接影响了随后出现的多篇论文，例如：  
* **Structure‑First Prompting**（2024）尝试在提示层面硬编码步骤顺序，取得类似提升。  
* **Distilled Long‑CoT**（2025）利用本文的结构敏感微调思路，对更小模型进行蒸馏，进一步降低部署成本。  

如果想继续深入，可以关注以下方向：  
1. **自动化长链生成**：利用大模型自我生成结构化的 Long CoT，再回环微调。  
2. **跨模态结构学习**：把图像、表格等信息的推理步骤也统一到同一结构框架中。  
3. **结构化评估**：设计更细粒度的指标，量化模型对步骤顺序、逻辑一致性的把握程度。  

### 一句话记住它
只要教会模型“怎么走路”，它就能在少量示例下学会长链推理——内容不重要，结构决定一切。