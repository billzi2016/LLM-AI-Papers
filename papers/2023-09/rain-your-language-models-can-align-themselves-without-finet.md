# RAIN: Your Language Models Can Align Themselves without Finetuning

> **Date**：2023-09-13
> **arXiv**：https://arxiv.org/abs/2309.07124

## Abstract

Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research typically gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, a.k.a. the finetuning step. In contrast, aligning frozen LLMs without requiring alignment data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide rewind and generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B from 82% of vanilla inference to 97%, while maintaining the helpfulness rate. On the TruthfulQA dataset, RAIN improves the truthfulness of the already-well-aligned LLaMA-2-chat 13B model by 5%.

---

# RAIN：你的语言模型可以在无需微调的情况下自行对齐 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成文本时常常会出现与人类价值观冲突的回答，比如提供有害信息或撒谎。传统的解决办法是先收集大量人工偏好数据，然后用强化学习或指令微调把模型“拉回正轨”。这一步既需要昂贵的标注成本，又要对模型参数进行梯度更新，计算资源消耗大。更关键的是，微调后模型的行为仍然会随训练数据和超参数的细微变化而不稳定。于是出现了一个更诱人的设想：能否在保持模型参数不动、无需额外对齐数据的情况下，让模型自己纠正不当输出？在此之前几乎没有可行的技术路线。

### 关键概念速览
**自评估（Self‑Evaluation）**：模型在生成答案的同时，用自身的语言能力对刚才的输出打分或判断好坏，类似于人在写作文后自己检查语法和逻辑。  
**倒带（Rewind）**：当自评估给出低分时，模型回到之前的生成状态重新尝试，就像玩游戏时发现走错路后按“撤销”键重新走。  
**自动回归推理（Auto‑regressive Inference）**：模型一次生成一个 token（词或子词），每一步都依赖前面的上下文，这是大多数 LLM 生成文本的基本方式。  
**RAIN（Rewindable Auto‑regressive INference）**：本文提出的推理框架，允许在普通的自回归生成过程中插入自评估和倒带两步，实现“边生成边自我纠错”。  
**无梯度对齐（Gradient‑free Alignment）**：对齐过程不涉及任何梯度计算或参数更新，完全在推理阶段完成。  
**Harmlessness（无害性）**：模型输出不包含有害、攻击性或违规内容的概率。  
**Truthfulness（真实性）**：模型回答与事实相符的程度，常用 TruthfulQA 数据集衡量。

### 核心创新点
1. **从微调转向推理层面的自我纠错**  
   之前的对齐方法必须在训练阶段加入人类偏好数据并进行梯度更新。本文直接在推理时让模型评估自己的输出并决定是否倒带，省掉了所有标注和训练成本。  
2. **引入可逆的自回归生成机制**  
   传统自回归生成只能向前走，一旦生成错误就无法回头。RAIN 在每一步后插入一个“检查点”，如果评估分数低于阈值，就把生成状态恢复到检查点前的隐藏状态并重新采样。这样模型拥有了“可撤销”的能力。  
3. **利用同一模型完成评估与生成**  
   评估模块和生成模块共享同一个预训练 LLM，避免了额外的评估模型或专门的奖励模型。模型的内部语言理解直接转化为对自身输出的判断，形成闭环。  
4. **在保持帮助度的同时显著提升安全性**  
   实验显示，使用 RAIN 后 LLaMA 30B 在 HH（Harmlessness）数据集上的无害率从 82% 提升到 97%，而帮助度几乎不变；在 TruthfulQA 上对齐度提升约 5%。这证明自评估+倒带的组合能够在不牺牲有用性的前提下提升安全属性。

### 方法详解
**整体框架**  
RAIN 的推理过程可以概括为三步循环：① 生成下一个 token；② 用同一模型对已生成的完整答案进行自评估，得到一个安全/真实性分数；③ 若分数低于预设阈值，执行倒带并重新生成；否则继续下一轮生成。循环直到生成结束标记或达到最大长度。

**关键模块拆解**  
1. **生成模块**：保持标准的自回归方式。模型接收已有上下文（包括用户提问和已生成的答案），输出下一个 token 的概率分布并采样。  
2. **评估模块**：在每次生成后，模型被重新调用一次，输入同样的上下文，但这次的任务是“请判断你刚才的回答是否安全/真实”。模型输出一个简短的评价（如 “safe” / “unsafe”）或直接返回一个数值分数。因为使用的是同一个 LLM，评估的语言风格与生成保持一致。  
3. **倒带机制**：模型内部的隐藏状态（Transformer 的 KV 缓存）在每一步生成后都会被保存。若评估结果不满意，系统把缓存恢复到上一次保存的状态，抹掉错误的 token，然后重新进行采样。相当于在生成树上回到父节点重新探索。  
4. **阈值与策略**：作者设定了安全阈值和真实性阈值，只有当评估分数低于阈值才触发倒带。为了防止无限循环，最多允许固定次数的倒带尝试；超过次数则直接接受当前输出。

**公式/算法的白话解释**  
- **生成**：`token_t = Sample(Softmax(LLM(context_{t-1})))`，即在已有上下文上算出下一个词的概率并抽样。  
- **评估**：`score_t = LLM_Eval(context_t)`，把完整上下文喂给模型，让它输出一个分数。  
- **倒带判断**：`if score_t < threshold then Rewind(context_{t-1}) else continue`。这里的 Rewind 实际上是把 KV 缓存恢复到 `t-1` 步的状态。  
最巧妙的地方在于：评估和生成共享同一个模型，省掉了额外的奖励模型训练；而 KV 缓存的保存/恢复让倒带几乎是零成本的操作。

### 实验与效果
- **测试数据**：作者在两套公开基准上评估 RAIN：HH（Harmlessness）数据集用于衡量模型的无害性，TruthfulQA 用于衡量事实准确性。  
- **基线对比**：在 HH 上，未使用 RAIN 的 LLaMA 30B（直接推理）无害率为 82%；使用 RAIN 后提升至 97%，提升幅度约 15%。在 TruthfulQA 上，对齐度已经很高的 LLaMA‑2‑chat 13B 通过 RAIN 再提升约 5%。  
- **帮助度保持**：作者用 GPT‑4 以及人工评审对“帮助度”（即答案是否有用、是否满足用户需求）进行打分，结果显示 RAIN 前后差异不显著，说明安全提升并未以牺牲有用性为代价。  
- **消融实验**：论文中对评估阈值、倒带次数上限以及是否共享同一模型进行消融。结果表明：共享模型的方案比使用独立评估模型更有效；阈值设得太宽松会导致倒带次数激增，计算成本上升；倒带次数限制在 3 次以内即可获得大部分收益。  
- **局限性**：作者承认 RAIN 仍然会增加推理时的计算开销，因为每一步都要额外调用一次评估；在极端长文本或实时交互场景下可能不够高效。此外，评估质量受模型本身能力限制，若模型本身对安全/真实性的理解不足，倒带也难以纠正。

### 影响与延伸思考
RAIN 把对齐问题从训练阶段搬到推理阶段，打开了“模型自我监管”的新思路。后续工作开始探索更高效的自评估策略，例如使用轻量化的评估头或在生成过程中并行评估，以降低额外的计算成本。还有研究尝试把倒带机制与采样温度调节结合，让模型在不满意时不仅回滚，还主动改变搜索策略。对齐数据的需求被进一步削减，使得在资源受限的环境（如边缘设备）上部署安全 LLM 变得更可行。想深入了解的读者可以关注“自监督对齐”（self‑supervised alignment）和“可逆推理”（reversible inference）这两个方向，尤其是近期在 NeurIPS、ICLR 上出现的几篇基于同理心评估的工作。

### 一句话记住它
**RAIN 让大模型在推理时自评估、倒带重写，从而在不微调、不加标注的情况下自行对齐。**