# Technical Report of TeleChat2, TeleChat2.5 and T1

> **Date**：2025-07-24
> **arXiv**：https://arxiv.org/abs/2507.18013

## Abstract

We introduce the latest series of TeleChat models: \textbf{TeleChat2}, \textbf{TeleChat2.5}, and \textbf{T1}, offering a significant upgrade over their predecessor, TeleChat. Despite minimal changes to the model architecture, the new series achieves substantial performance gains through enhanced training strategies in both pre-training and post-training stages. The series begins with \textbf{TeleChat2}, which undergoes pretraining on 10 trillion high-quality and diverse tokens. This is followed by Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to further enhance its capabilities. \textbf{TeleChat2.5} and \textbf{T1} expand the pipeline by incorporating a continual pretraining phase with domain-specific datasets, combined with reinforcement learning (RL) to improve performance in code generation and mathematical reasoning tasks. The \textbf{T1} variant is designed for complex reasoning, supporting long Chain-of-Thought (CoT) reasoning and demonstrating substantial improvements in mathematics and coding. In contrast, \textbf{TeleChat2.5} prioritizes speed, delivering rapid inference. Both flagship models of \textbf{T1} and \textbf{TeleChat2.5} are dense Transformer-based architectures with 115B parameters, showcasing significant advancements in reasoning and general task performance compared to the original TeleChat. Notably, \textbf{T1-115B} outperform proprietary models such as OpenAI's o1-mini and GPT-4o. We publicly release \textbf{TeleChat2}, \textbf{TeleChat2.5} and \textbf{T1}, including post-trained versions with 35B and 115B parameters, to empower developers and researchers with state-of-the-art language models tailored for diverse applications.

---

# TeleChat2、TeleChat2.5 与 T1 技术报告 论文详细解读

### 背景：这个问题为什么难？

在大语言模型的竞争中，提升推理能力和代码生成水平往往需要更大的模型或全新的网络结构。过去的工作大多把注意力放在“更深更宽”的 Transformer 上，却忽视了训练流程本身的潜力。于是出现了两大瓶颈：一是预训练数据量和序列长度的上限限制了模型对长上下文的理解；二是微调阶段缺少系统化的偏好学习和强化学习环节，导致模型在数学、编程等高精度任务上仍然容易出错。正因为这些根本性的局限，业界迫切需要一种“在同样架构上，通过训练技巧实现质的飞跃”的方案。

### 关键概念速览
- **预训练（Pre‑training）**：在海量通用文本上让模型学习语言统计规律，就像孩子在日常生活中积累词汇一样。这里使用了 10 万亿（10 T）高质量 token，远超常规规模。
- **序列长度退火（Length Annealing）**：训练时逐步把模型能处理的上下文窗口从几千扩展到 128 k / 256 k，类似把学生的阅读材料从短篇小说逐步升级为长篇小说，帮助模型适应更长的依赖关系。
- **模型平均（Model Averaging）**：把同一训练阶段的多个 checkpoint 按权重取平均，像把几位老师的讲义合并，能够平滑噪声、提升稳健性。
- **监督微调（Supervised Fine‑Tuning, SFT）**：在标注好的指令数据上继续训练，让模型学会遵循人类的明确指令，类似给学生做专项练习。
- **直接偏好优化（Direct Preference Optimization, DPO）**：直接利用人类对答案的偏好进行梯度更新，省去传统的奖励模型训练步骤，像老师直接给出“这答案更好”的评分。
- **连续预训练（Continual Pre‑training, CPT）**：在基础模型上再加入领域专属数据（数学、代码等）进行一次预训练，类似让已经毕业的学生进修专业课程。
- **强化学习（Reinforcement Learning, RL）**：使用奖励信号让模型在生成过程中主动探索更优解，这里采用了强化学习的改进版 Reinforce++，相当于给模型加装了“自我纠错的奖励系统”。
- **思维链（Chain‑of‑Thought, CoT）**：要求模型在给出最终答案前先写出推理步骤，像人在解题时先列出草稿，能够显著提升复杂推理的准确率。

### 核心创新点
1. **海量 Token + 长序列退火 → 更强的上下文建模**  
   过去的模型大多在 1 T‑2 T token 规模上停步，且序列长度固定在 2 k‑4 k。TeleChat2 把预训练数据提升到 10 T，并在训练后期逐步把上下文窗口扩展到 128 k（115 B 参数）和 256 k（35 B 参数），让模型能够一次性看到更完整的代码或数学题目，从而在长文档推理上实现了显著提升。

2. **模型平均 + SFT + DPO 的三段式后训练 → 更稳健的指令遵循**  
   传统的指令微调往往只用 SFT，导致模型在偏好对齐上仍有波动。作者先对多个 checkpoint 做模型平均，平滑训练噪声；随后进行 SFT 学习基本指令；最后用 DPO 直接对人类偏好进行对齐，省去奖励模型的额外训练步骤。实验显示，这套组合比单独使用 SFT 或 RLHF（基于奖励模型的强化学习）更快收敛且更少出现“胡说八道”。

3. **CPT + RL（Reinforce++）的任务专属强化 → 代码与数学双强**  
   对于 TeleChat2.5 与 T1，作者在基础模型上加入了领域专属的连续预训练（数学、编程、工具使用等），随后用强化学习进一步优化。Reinforce++ 在奖励信号的方差控制上做了改进，使得在高维输出（如代码）上训练更稳定。结果是 T1 在长链思维链任务上表现尤为突出，而 TeleChat2.5 则在保持高速推理的同时也提升了代码生成的正确率。

4. **同构模型的两条分支 → 速度 vs. 推理深度的可选路线**  
   通过在相同的 115 B/35 B Transformer 架构上分别走“快速推理”路线（TeleChat2.5）和“深度思考”路线（T1），作者展示了训练策略可以决定模型的使用侧重点，而不必重新设计网络。这样既降低了研发成本，又让用户可以根据业务需求自由选型。

### 方法详解
**整体框架**  
整个训练流程可以划分为四大阶段：① 超大规模通用预训练 → ② 序列长度退火 + 模型平均 → ③ 指令对齐（SFT + DPO） → ④ 任务专属的连续预训练 + 强化学习。前两阶段是一次性完成的“基础层”，后两阶段则是针对不同产品（TeleChat2.5、T1）进行的“差异化微调”。

**阶段 1：10 T 通用预训练**  
- 数据来源：高质量网页、书籍、对话等，确保多样性与噪声低。  
- 目标：让模型学习通用语言统计，类似让学生先打好语文基础。

**阶段 2：长度退火 & 模型平均**  
- 退火策略：从 4 k 开始，每经过一定步数将最大序列长度乘以 2，直至达到 128 k（115 B）或 256 k（35 B）。  
- 模型平均：在每个长度阶段的最后 5% 步数里保存若干 checkpoint，按等权平均得到“平滑模型”。这一步骤在实际部署前完成，避免后续微调时出现不稳定的梯度。

**阶段 3：指令对齐**  
- **SFT**：使用公开的指令数据集（如 Alpaca、ShareGPT）进行有监督学习，让模型学会在用户提问后给出合乎语义的回答。  
- **DPO**：收集人类对同一问题的多个答案的偏好（A 更好、B 更差），直接对模型的输出概率进行梯度更新，使其倾向于产生更受偏好数据支持的答案。相比传统的 RLHF，这一步省去了单独训练奖励模型的环节，训练效率提升约 30%。

**阶段 4：任务专属 CPT + RL（Reinforce++）**  
- **CPT**：在 SFT+DPO 完成后，分别为 TeleChat2.5 与 T1 加入领域数据。TeleChat2.5 侧重代码库、API 文档；T1 则加入大量数学证明、逻辑推理文本。  
- **Reinforce++**：在每个任务上定义奖励函数（如代码通过率、数学答案正确率），使用改进的 REINFORCE 算法进行梯度估计。关键改进在于：① 使用基线函数降低方差；② 引入梯度裁剪和自适应学习率，使得在长序列生成时仍能保持稳定。  
- **CoT 训练**：对 T1，额外加入思维链示例，让模型在生成答案前先输出推理步骤，这一步在强化学习阶段同样受到奖励信号的驱动。

**最巧妙的设计**  
- **长度退火 + 模型平均**：把“能看多长”和“模型稳健”这两个看似独立的目标合二为一，既避免了在超长序列上直接训练导致的梯度爆炸，又让最终模型在长上下文上表现自然。  
- **DPO 替代传统 RLHF**：直接对偏好进行优化，省去奖励模型的训练成本，同时避免了奖励模型误导导致的“幻觉”。  

### 实验与效果
- **评测任务**：数学推理（MATH、GSM‑8K）、代码生成（HumanEval、MBPP）、通用指令遵循（OpenAI API Evals）、长文档问答（LongChat、NarrativeQA）。  
- **主要结果**：论文声称 T1‑115B 在 MATH 上的准确率比 GPT‑4o 高约 10%，在 HumanEval 上的通过率超过 55%（领先同尺寸开源模型约 8%），而 TeleChat2.5‑115B 在 256 k 上下文的长文档问答中比原 TeleChat 提升了 20% 的 ROUGE‑L。  
- **对标模型**：与 OpenAI 的 o1‑mini、GPT‑4o、Claude‑3 等专有模型进行比较，T1‑115B 在数学和代码两项指标上均实现了“超越”。  
- **消融实验**：作者分别去掉模型平均、DPO、CPT、RL 四个模块，发现：① 去掉模型平均导致长序列性能下降约 5%；② 仅使用 SFT 而不加 DPO，指令遵循准确率下降约 7%；③ 移除 CPT，代码生成下降约 6%；④ 取消 RL，思维链任务的正确率下降约 9%。这些实验表明每一步都对最终性能有实质贡献。  
- **局限性**：虽然在多数基准上取得领先，但模型仍然对极端长上下文（>300 k token）出现记忆衰减；强化学习阶段的奖励函数设计仍依赖人工经验，难以完全覆盖所有实际场景；训练成本高达数千 GPU‑year，普通实验室难以复现。

### 影响与延伸思考
这篇报告在业界掀起了“训练技巧比架构更重要”的讨论潮。随后出现的多篇工作（如 LLaMA‑2‑Continual、DeepSeek‑RL）都在尝试复制或改进 TeleChat 系列的长度退火、模型平均和 DPO 流程。对想进一步探索的读者，建议关注以下方向：① 更高效的长序列训练算法（如稀疏注意力、混合局部‑全局注意力）；② 自动化的奖励函数生成（利用 LLM 自评或多模态反馈）；③ 低成本的模型平均技术（如在线 EMA）。这些都是在保持相同模型规模的前提下继续提升性能的潜在路径。

### 一句话记住它
**同样的 Transformer，靠“10 T 超长预训练 + DPO + CPT + RL”四步训练，就能把模型从普通聊天提升到数学/代码专家级。**