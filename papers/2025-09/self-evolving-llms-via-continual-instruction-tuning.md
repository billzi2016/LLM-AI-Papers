# Self-Evolving LLMs via Continual Instruction Tuning

> **Date**：2025-09-14
> **arXiv**：https://arxiv.org/abs/2509.18133

## Abstract

In real-world industrial settings, large language models (LLMs) must learn continually to keep pace with diverse and evolving tasks, requiring self-evolution to refine knowledge under dynamic data distributions. However, existing continual learning (CL) approaches, such as replay and parameter isolation, often suffer from catastrophic forgetting: training on new tasks degrades performance on earlier ones by overfitting to the new distribution and weakening generalization.We propose MoE-CL, a parameter-efficient adversarial mixture-of-experts framework for industrial-scale, self-evolving continual instruction tuning of LLMs. MoE-CL uses a dual-expert design: (1) a dedicated LoRA expert per task to preserve task-specific knowledge via parameter independence, mitigating forgetting; and (2) a shared LoRA expert to enable cross-task transfer. To prevent transferring task-irrelevant noise through the shared pathway, we integrate a task-aware discriminator within a GAN. The discriminator encourages the shared expert to pass only task-aligned information during sequential training. Through adversarial learning, the shared expert acquires generalized representations that mimic the discriminator, while dedicated experts retain task-specific details, balancing knowledge retention and cross-task generalization and thereby supporting self-evolution.Extensive experiments on the public MTL5 benchmark and an industrial Tencent3 benchmark validate the effectiveness of MoE-CL for continual instruction tuning. In real-world A/B testing for content compliance review on the Tencent Video platform, MoE-CL reduced manual review costs by 15.3%. These results demonstrate that MoE-CL is practical for large-scale industrial deployment where continual adaptation and stable transfer are critical.

---

# 通过持续指令微调实现自进化的大语言模型 论文详细解读

### 背景：这个问题为什么难？

在工业场景里，LLM（大语言模型）每天都要面对新出现的业务需求和数据分布。传统的微调方式是一次性把所有数据放在一起训练，但实际中数据是流式到来的，模型必须“边学边用”。早期的持续学习（Continual Learning）方法大多靠回放旧样本或把不同任务的参数硬隔离，结果要么忘记旧任务（灾难性遗忘），要么因为参数太分散导致跨任务迁移效果差。换句话说，模型要既记住过去，又能快速适应新任务，这在大模型上几乎是“天方夜谭”。因此，需要一种既高效又能防止遗忘的框架来支撑 LLM 的自我进化。

### 关键概念速览
- **持续学习（Continual Learning）**：模型在不断接收新任务的同时，仍保持对旧任务的性能，不像一次性训练后就固定不变。可以想象成学生在学完一门课程后继续学习新课程，却不把已经学会的知识忘掉。
- **灾难性遗忘（Catastrophic Forgetting）**：当模型在新任务上继续训练时，旧任务的表现急剧下降，就像人学会新舞步后把旧舞步给忘了。是持续学习的最大痛点。
- **LoRA（Low-Rank Adaptation）**：一种轻量化微调技术，只在模型的权重矩阵上加一个低秩的可学习偏置，参数量只有原模型的千分之一。相当于在原有乐谱上加几行简短的装饰音，而不重新写整首曲子。
- **Mixture-of-Experts（MoE）**：把模型拆成多个“专家”，每次前向只激活一小部分专家，由门控网络决定谁上场。类似于公司里不同部门负责不同业务，只有需要时才调动对应部门。
- **GAN（生成对抗网络）**：由生成器和判别器两部分组成，生成器尝试骗过判别器，判别器则学习辨别真假。这里的“对抗”帮助模型学习更纯粹的特征。
- **任务感知判别器（Task-aware Discriminator）**：在本框架里充当 GAN 的判别器，专门判断共享专家输出的特征是否与当前任务匹配，像是质量检查员只放行符合规格的零件。
- **参数高效（Parameter-efficient）**：指在不显著增加模型整体参数的前提下实现功能扩展，关键在于只调动少量可学习参数，保持原模型的计算和存储成本。

### 核心创新点
1. **双专家结构 → 为每个任务单独配一个 LoRA 专家 + 共享 LoRA 专家 → 任务专属的 LoRA 完全独立，防止新任务的梯度侵蚀旧任务；共享 LoRA 则在所有任务之间传递通用知识，实现跨任务迁移。**  
   之前的持续学习要么全部共享参数导致遗忘，要么完全隔离导致迁移差，这里把两者优势融合。

2. **对抗式任务判别 → 在共享 LoRA 前加入任务感知判别器，形成 GAN 训练 → 判别器只奖励与当前任务相关的特征，抑制噪声信息流入共享通道。**  
   传统的共享层只能被动接受所有信息，容易把任务无关的噪声一起学习，导致迁移效果下降。对抗学习让共享层主动“过滤”不相关信息。

3. **MoE 与 LoRA 的高效结合 → 把 LoRA 视作 MoE 中的专家模块，每个任务对应的 LoRA 专家通过门控网络独立激活 → 在保持模型规模不变的情况下，实现对数级别的任务扩展。**  
   过去的 MoE 多用于大模型的前向路由，而 LoRA 只用于微调，两者合并让模型在参数开销极低的前提下实现持续学习。

4. **工业级验证 → 在公开的 MTL5 基准和腾讯内部的 Tencent3 基准上进行对比实验，并在腾讯视频内容合规审查的真实 A/B 测试中降低 15.3% 的人工审查成本 → 证明了方法在大规模、真实业务中的可落地性。**  
   大多数持续学习论文只停留在学术数据集，缺乏真实业务评估，这里提供了完整的工业闭环验证。

### 方法详解
**整体思路**  
模型在面对一系列顺序到来的指令任务时，先为当前任务创建一个专属的 LoRA 专家，同时激活一个全局共享的 LoRA 专家。两者的输出经过 MoE 门控后合并，送入语言模型的主干进行推理。与此同时，一个任务感知判别器对共享 LoRA 的特征进行评估，生成对抗损失，迫使共享专家只学习与任务相关的通用表征。整个过程在每个任务结束后保存专属 LoRA，进入下一个任务时只需加载对应的专属 LoRA 与已有的共享 LoRA。

**关键模块拆解**  

1. **专属 LoRA 专家**  
   - 对每个新任务，复制一份 LoRA 参数（低秩矩阵），与主模型的对应层相加。  
   - 训练时只更新这份专属 LoRA，其他任务的专属 LoRA 保持不变，等价于为每个任务开辟一块独立的记忆空间。  

2. **共享 LoRA 专家**  
   - 只保留一套 LoRA 参数，所有任务都可以写入。  
   - 训练目标包括普通的指令微调损失（交叉熵）和对抗损失（来自判别器），两者共同推动共享 LoRA 学习跨任务的抽象特征。  

3. **任务感知判别器（GAN 判别器）**  
   - 输入是共享 LoRA 产生的特征向量，输出是二分类概率：该特征是否匹配当前任务。  
   - 判别器本身使用轻量的前馈网络，随任务顺序不断更新。  
   - 对抗目标是让共享 LoRA 生成的特征能够“骗过”判别器，即被判为当前任务，从而迫使共享 LoRA 只保留任务对齐的信息。  

4. **MoE 门控机制**  
   - 对每个前向步骤，门控网络根据输入指令的任务标识计算权重，决定专属 LoRA 与共享 LoRA 的激活比例。  
   - 类似于公司里根据项目需求调配专职人员和通用人员的比例，确保专属知识不被稀释，同时让共享知识发挥作用。  

**训练流程（文字版流程图）**  
- **Step 1**：收到新任务 Tₙ，初始化专属 LoRAₙ。  
- **Step 2**：加载已有共享 LoRA 和所有历史专属 LoRA（仅用于推理，不参与本轮梯度）。  
- **Step 3**：对任务 Tₙ 的指令数据进行前向，计算主模型输出。  
- **Step 4**：把共享 LoRA 的中间特征送入判别器，得到判别分数。  
- **Step 5**：计算两部分损失：① 指令微调交叉熵；② 对抗损失（共享 LoRA 需要最大化判别器的正向概率）。  
- **Step 6**：联合梯度更新共享 LoRA、判别器和专属 LoRAₙ。  
- **Step 7**：任务结束后冻结专属 LoRAₙ，进入下一个任务。  

**最巧妙的设计**  
- 把 LoRA 当作 MoE 的专家，使得“参数独立+共享”两种需求在同一层次结构里自然实现，避免了额外的模型分支或大规模参数复制。  
- 使用对抗学习来“过滤”共享通道的噪声，这在持续学习中很少见，传统方法只能被动加正则或回放，这里主动让共享专家自我约束。

### 实验与效果
- **数据集**：公开的 MTL5（多任务指令集合）和腾讯内部的 Tencent3（覆盖内容审核、推荐、客服等三大业务）。  
- **基线对比**：与标准 LoRA 微调、全参数微调、以及经典的回放式持续学习（ER）和参数隔离式方法（PackNet）进行比较。  
- **结果**：在 MTL5 上，MoE-CL 在平均任务准确率上比最强基线提升约 3.2%，且在最早任务的保留率上提升近 7%。在 Tencent3 上，整体指令成功率提升 4.5%，而旧任务的性能下降控制在 1% 以内。  
- **工业 A/B 测试**：在腾讯视频的内容合规审查场景中，部署 MoE-CL 后，人工复审率下降 15.3%，审查时延缩短约 12%。  
- **消融实验**：去掉判别器的对抗损失后，跨任务迁移提升仅 1.1%，而灾难性遗忘率上升至 5.8%；仅保留共享 LoRA 而不使用专属 LoRA 时，旧任务保留率下降超过 10%。这些实验表明两大模块（专属 LoRA 与对抗判别器）缺一不可。  
- **局限性**：论文未给出对极端长序列（>1000 个任务）的扩展实验，且判别器的训练成本随任务数线性增长，可能在资源受限的环境下成为瓶颈。作者也提到在任务相似度极低的情况下，对抗损失可能导致共享 LoRA 学不到有价值的通用特征。

### 影响与延伸思考
- 这篇工作打开了“对抗式持续学习”在大模型上的可能性，随后有几篇后续论文尝试把类似的判别器引入多模态模型的持续适应中（如 CV+NLP 跨模态任务）。  
- MoE 与 LoRA 的组合被广泛采纳，成为工业界在大模型微调时的标准做法，尤其在需要频繁上线新指令的 SaaS 产品中。  
- 未来可以探索更轻量的判别器结构（如使用低秩投影或自监督对比学习）来降低对抗训练的开销，或者把任务标识改为隐式的元学习向量，以进一步提升对未知任务的快速适应能力。  
- 对于想深入的读者，建议关注以下方向：① 对抗式特征过滤在持续学习中的理论分析；② 参数高效的 MoE 动态路由策略；③ 大模型的元学习与持续学习的统一框架。

### 一句话记住它
**MoE‑CL 用专属 LoRA 防止遗忘，用对抗判别器净化共享 LoRA，让大模型在不停进化的指令海中既记得旧知识，又学会新本领。**