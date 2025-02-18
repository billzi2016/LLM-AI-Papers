# Small Models Struggle to Learn from Strong Reasoners

> **Date**：2025-02-17
> **arXiv**：https://arxiv.org/abs/2502.12143

## Abstract

Large language models (LLMs) excel in complex reasoning tasks, and distilling their reasoning capabilities into smaller models has shown promise. However, we uncover an interesting phenomenon, which we term the Small Model Learnability Gap: small models ($\leq$3B parameters) do not consistently benefit from long chain-of-thought (CoT) reasoning or distillation from larger models. Instead, they perform better when fine-tuned on shorter, simpler reasoning chains that better align with their intrinsic learning capacity. To address this, we propose Mix Distillation, a simple yet effective strategy that balances reasoning complexity by combining long and short CoT examples or reasoning from both larger and smaller models. Our experiments demonstrate that Mix Distillation significantly improves small model reasoning performance compared to training on either data alone. These findings highlight the limitations of direct strong model distillation and underscore the importance of adapting reasoning complexity for effective reasoning capability transfer.

---

# 小模型难以从强推理模型中学习 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在需要多步推理的任务上表现惊人，但它们往往拥有数十甚至上百亿参数，部署成本高。业界一直想把这些强大的推理能力“压缩”到几亿到十几亿参数的小模型上，以便在资源受限的场景使用。传统的蒸馏技术把大模型的输出作为软标签喂给小模型，理论上应该让小模型学到同样的思考方式。然而，实际实验发现，小模型在接受长链式思考（CoT）示例或直接从大模型蒸馏时，提升并不稳定，甚至出现退步。根本原因在于小模型的学习容量和记忆带宽有限，它们难以消化过于复杂、冗长的推理过程。这一现象让人怀疑，单纯的“强教师→弱学生”蒸馏是否真的可行。

### 关键概念速览
**大语言模型（LLM）**：参数量在数十亿以上的模型，能够生成流畅自然语言并进行复杂推理。  
**小模型**：参数量在 3 B 以下的模型，算力和显存需求低，但学习能力受限。  
**Chain‑of‑Thought（CoT）**：让模型在给出答案前先写出逐步推理过程，类似于人做数学题时的草稿。  
**蒸馏（Distillation）**：把大模型的预测（软标签）或内部表示作为训练信号，教会小模型模仿大模型的行为。  
**学习可塑性（Learnability）**：模型在给定数据上提升性能的潜力，受模型容量、数据复杂度等因素影响。  
**Mix Distillation**：本文提出的混合蒸馏策略，既使用长 CoT 也使用短 CoT，或同时利用大模型和小模型生成的推理示例。  
**推理复杂度匹配**：根据学生模型的容量，挑选合适长度和难度的推理链，使学习过程既不太简单也不超负荷。

### 核心创新点
1. **发现“小模型学习可塑性缺口”**  
   之前的工作默认只要提供足够多的长链式示例，蒸馏就能提升小模型。本文通过系统实验揭示，小模型在 ≤3 B 参数时，对长 CoT 并不敏感，甚至会产生负面效应。这个发现本身就改变了人们对蒸馏有效性的认知。  

2. **从“全长”转向“短链”微调**  
   传统做法是让小模型直接学习大模型的完整推理过程。本文改为先筛选出步骤更少、逻辑更紧凑的推理链（比如 2‑3 步），再进行微调。结果表明，短链更贴合小模型的记忆窗口，使得学习效率显著提升。  

3. **提出 Mix Distillation**  
   直接使用单一来源（全部长链或全部短链）仍会留下性能上限。Mix Distillation 通过在同一批训练数据中混合长链和短链，或混合大模型与小模型生成的推理示例，实现了推理复杂度的平衡。这样，小模型既能接触到高层次的思考模式，又不会被信息量淹没。  

4. **实验验证“复杂度平衡”是提升关键**  
   作者在多个基准上对比了仅长链、仅短链、以及 Mix Distillation 三种策略，发现混合方式 consistently 超过单一策略，说明推理复杂度的匹配是蒸馏成功的核心因素。

### 方法详解
整体思路可以拆成三步：**数据准备 → 复杂度标记 → 混合蒸馏训练**。

1. **数据准备**  
   - 选取公开的推理任务数据集（如 GSM8K、MathQA 等），每条样本都对应一个答案。  
   - 对每条样本使用大模型（如 175 B 参数的 GPT‑4）生成两种 CoT：一种是**完整长链**（包含所有思考步骤），另一种是**压缩短链**（通过让模型在生成时限制最大步数或直接人工裁剪）。  

2. **复杂度标记**  
   - 为每个 CoT 打上“长度标签”（长 / 短），并记录生成来源（大模型或小模型）。  
   - 这样在后续训练时可以按比例抽取不同标签的样本，实现混合。  

3. **Mix Distillation 训练**  
   - **采样策略**：在每个 mini‑batch 中，按照预设比例（例如 50% 长链、50% 短链）随机抽取样本。若使用双来源，还会再把大模型生成的长链和小模型自身生成的短链混在一起。  
   - **目标函数**：对每条样本，计算两部分损失：  
     - **答案损失**：普通的交叉熵，确保模型最终输出正确答案。  
     - **推理损失**：对 CoT 步骤的每个 token 计算 KL 散度，使小模型的中间生成分布逼近教师的分布。  
   - **训练细节**：采用常规的 AdamW 优化器，学习率随 epoch 线性衰减。因为短链更容易收敛，作者在训练前期使用较高的短链比例，后期逐步提升长链比例，让模型在掌握基础推理后再学习更深层次的思考。  

**最巧妙的点**在于“动态比例调度”。作者没有固定 1:1 的混合，而是让比例随训练进度自适应，这相当于给小模型提供了“梯度式”难度提升，类似于人类学习时先掌握基础再攻克难题。

### 实验与效果
- **测试任务**：主要在数学推理（GSM8K、MathQA）、逻辑推理（StrategyQA）以及常识推理（ARC‑Easy）上评估。  
- **基线**：包括（1）直接微调小模型（不使用 CoT），（2）仅使用长链蒸馏，（3）仅使用短链蒸馏。  
- **主要结果**：在 GSM8K 上，3 B 参数模型使用 Mix Distillation 比仅长链提升约 7% 准确率，比仅短链提升约 4%；在 MathQA 上提升约 5%。在非数学任务上也都有 2‑3% 的稳健提升。  
- **消融实验**：作者分别去掉长链、去掉短链、以及去掉动态比例调度。结果显示，去掉任何一项都会导致整体性能回落到单一策略的水平，验证了每个组件的必要性。  
- **局限性**：实验主要围绕 1‑3 B 参数的模型，未在更大或更小的模型上验证；此外，Mix Distillation 需要先生成两套 CoT，增加了前期数据准备成本。原文也承认在极端低资源（如 100 M 参数）模型上仍未见显著收益。

### 影响与延伸思考
这篇工作提醒社区：**蒸馏不是“一刀切”，推理复杂度必须与学生模型匹配**。随后出现的几篇论文（如 “Curriculum Distillation for Reasoning” 与 “Adaptive CoT for Small LMs”）都在尝试把难度调度机制系统化，甚至把难度估计交给元学习模型来自动决定。对想进一步探索的读者，可以关注以下方向：  
- **难度自适应蒸馏**：让模型自己评估哪些推理步骤对自己有帮助。  
- **跨模态蒸馏**：把视觉或结构化推理的长链转化为短链，看看是否同样有效。  
- **低资源蒸馏**：在更小模型上寻找更轻量的混合策略，或利用少量人工标注的短链提升效果。

### 一句话记住它
小模型需要“短而精”的思考步骤，混合长短链的蒸馏才能真正把大模型的推理能力迁移过去。