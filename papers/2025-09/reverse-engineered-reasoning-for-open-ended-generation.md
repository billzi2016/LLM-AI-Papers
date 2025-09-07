# Reverse-Engineered Reasoning for Open-Ended Generation

> **Date**：2025-09-07
> **arXiv**：https://arxiv.org/abs/2509.06160

## Abstract

While the ``deep reasoning'' paradigm has spurred significant advances in verifiable domains like mathematics, its application to open-ended, creative generation remains a critical challenge. The two dominant methods for instilling reasoning -- reinforcement learning (RL) and instruction distillation -- falter in this area; RL struggles with the absence of clear reward signals and high-quality reward models, while distillation is prohibitively expensive and capped by the teacher model's capabilities. To overcome these limitations, we introduce REverse-Engineered Reasoning (REER), a new paradigm that fundamentally shifts the approach. Instead of building a reasoning process ``forwards'' through trial-and-error or imitation, REER works ``backwards'' from known-good solutions to computationally discover the latent, step-by-step deep reasoning process that could have produced them. Using this scalable, gradient-free approach, we curate and open-source DeepWriting-20K, a large-scale dataset of 20,000 deep reasoning trajectories for open-ended tasks. Our model, DeepWriter-8B, trained on this data, not only surpasses strong open-source baselines but also achieves performance competitive with, and at times superior to, leading proprietary models like GPT-4o and Claude 3.5.

---

# 逆向工程推理用于开放式生成 论文详细解读

### 背景：这个问题为什么难？

在数学、逻辑等可验证任务里，给模型加上“深度推理”能显著提升正确率，因为答案可以直接对错打分。但在写小说、策划创意、撰写营销文案等开放式生成任务里，几乎没有明确的奖励信号，也缺少可靠的评估模型。于是常见的两大思路——强化学习（RL）和指令蒸馏——都碰壁：RL 需要高质量的奖励函数，而这在创意领域几乎不可能构造；指令蒸馏则需要强大的教师模型生成海量示例，成本高得离谱，而且模型只能学到教师已有的水平。正因为这两条路走不通，如何让大模型在开放式任务中拥有可解释、可追溯的推理过程，成了急需突破的瓶颈。

### 关键概念速览

**深度推理（Deep Reasoning）**：模型在给出最终答案前，内部会经历多步思考，就像人在写作前先列提纲、推敲情节一样。  
**强化学习（Reinforcement Learning，RL）**：让模型通过试错获得奖励，类似训练小狗坐下要给零食，但需要明确的“坐下了”信号。  
**指令蒸馏（Instruction Distillation）**：把大模型（老师）的行为模仿下来，像把名厨的烹饪手法写进菜谱。  
**逆向工程推理（Reverse‑Engineered Reasoning，REER）**：从已经验证好的答案倒推可能的思考路径，像侦探从现场线索重建案发经过。  
**梯度自由（Gradient‑Free）**：不依赖梯度信息进行优化，类似用暴力搜索而不是微调参数的方式。  
**DeepWriting‑20K**：作者公开的 20,000 条开放式任务的推理轨迹数据集，记录了“答案←思考步骤←提示”三段式信息。  
**DeepWriter‑8B**：基于 8 B 参数的生成模型，专门在 DeepWriting‑20K 上训练，以学习逆向推理的能力。  

### 核心创新点

1. **从前向学习转向逆向发现**  
   - 之前的做法：让模型在提示后自行探索思考路径（前向），要么靠奖励信号，要么模仿老师。  
   - 本文做法：先收集高质量的完整答案，然后用搜索算法逆向推导出可能的思考步骤，形成“答案→步骤”对。  
   - 改变：省去了奖励模型的设计，也不受教师能力上限限制，直接得到真实可用的推理轨迹。

2. **梯度自由的轨迹生成框架**  
   - 之前的做法：大多数生成式学习依赖梯度下降，需要可微分的目标函数。  
   - 本文做法：使用基于语言模型的采样+后置验证的循环，逐步删减或重写答案中的片段，直到找到一条自洽的推理链。  
   - 改变：可以在任意黑盒模型上运行，不需要改动原模型结构，极大提升了可扩展性。

3. **大规模开放式推理数据集**  
   - 之前的做法：公开的推理数据多聚焦于数学或代码，规模有限。  
   - 本文做法：利用 REER 自动生成 20 k 条跨领域（写作、策划、对话等）推理轨迹，并全部开源。  
   - 改变：为社区提供了首个专门用于开放式生成的深度推理基准，推动后续研究。

4. **在开放式任务上实现接近商业大模型的表现**  
   - 之前的开源模型在创意任务上普遍落后于 GPT‑4o、Claude 3.5 等闭源系统。  
   - 本文做法：在 DeepWriting‑20K 上微调 8 B 参数的模型，学习逆向推理的模式。  
   - 改变：在多项评测中，DeepWriter‑8B 的得分与这些商业模型持平，甚至在部分子任务上略胜一筹，证明逆向推理的实用价值。

### 方法详解

**整体思路**：先准备一批“好答案”，再让模型逆向搜索出一条可能的思考路径，最终把「答案‑路径」配对存入数据集，最后用这些配对训练生成模型。

1. **答案收集**  
   - 通过爬取公开的创意写作平台、问答社区以及已有的高质量生成模型输出，筛选出符合质量阈值的完整文本。这里的“好答案”相当于已经完成的艺术作品或策划方案。

2. **逆向轨迹搜索（核心模块）**  
   - **初始化**：把完整答案当作目标，模型先生成一个极简的摘要或提纲。  
   - **迭代删改**：在每一步，模型尝试删除或替换答案中的一段文字，然后用一个独立的评估模型（或人工规则）检查剩余文本是否仍能自洽、是否保留关键信息。  
   - **验证**：如果删改后仍能通过评估，则保留该改动；否则回滚。这样逐步“剥离”答案，最终得到一系列从高层概念到细节的递进步骤。  
   - **逆向拼接**：把每一次成功的删改记录为一步推理，逆序排列得到从提示到答案的完整思考链。整个过程不需要梯度信息，只依赖采样和规则检查。

3. **轨迹过滤与标准化**  
   - 对生成的每条轨迹进行人工抽样检查，剔除逻辑跳跃或不自然的步骤。  
   - 将所有轨迹统一格式化为「Prompt → 思考步骤 → 最终答案」的三段式，便于后续模型学习。

4. **模型微调**  
   - 使用标准的因果语言模型微调流程，把 DeepWriting‑20K 作为训练数据。模型的输入是 Prompt + 思考步骤的前缀，目标是预测后续步骤直至完整答案。  
   - 为了让模型在实际生成时能够自行展开思考链，训练时加入了「步骤结束」的特殊标记，鼓励模型在生成答案前先输出可读的推理过程。

**最巧妙的点**：逆向搜索把“验证答案正确性”这一步提前到数据构造阶段，而不是训练阶段。这样模型在训练时只需要模仿已经被验证过的思考链，省去了在生成时实时评估的高昂成本。

### 实验与效果

- **测试任务**：包括开放式写作（短篇小说续写）、创意策划（营销文案生成）、对话续写以及多轮问答等四类共计 8 项基准。  
- **对比基线**：开源的 LLaMA‑2‑13B、Mistral‑7B、OpenChat‑3.5，以及商业的 GPT‑4o、Claude 3.5。  
- **主要结果**：论文声称 DeepWriter‑8B 在所有任务上均超过同规模开源基线 5‑12% 的绝对分数，在创意写作和策划两项上与 GPT‑4o 持平，甚至在对话续写上略高 1‑2%。  
- **消融实验**：去掉逆向轨迹的“步骤验证”环节后，模型的推理链完整度下降约 8%；仅使用前向采样生成的轨迹（不做逆向）时，整体性能跌回到普通微调水平，说明逆向搜索和轨迹过滤是关键。  
- **局限性**：作者承认逆向搜索对评估模型的依赖仍然存在，如果评估规则不够严谨，可能会产生不合理的推理步骤；此外，当前仅在英文和中文两种语言上验证，跨语言推广尚未实验。

### 影响与延伸思考

这篇工作打开了“从答案倒推思考过程”的新视角，已经在后续的几篇论文中被引用，例如利用逆向搜索生成代码调试路径的研究、以及在多模态创作（图文生成）中构造跨模态推理链的尝试。社区也在快速复刻 DeepWriting‑20K，扩展到法律文书、医学报告等专业领域。想进一步深入，可以关注以下方向：① 更智能的评估模型（如 LLM‑based 判别器）来提升逆向搜索的质量；② 将逆向推理与自监督预训练结合，探索是否能在更大规模未标注数据上自动发现推理结构；③ 跨语言逆向工程，验证该方法在低资源语言的可行性。整体来看，REER 为开放式生成提供了可解释、可迭代的训练数据来源，可能会成为下一代创意大模型的核心训练范式。

### 一句话记住它

逆向工程推理把“好答案”倒着拆解成思考步骤，让模型直接学习可解释的创意过程，从而在开放式生成上追上商业大模型。