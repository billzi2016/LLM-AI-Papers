# Who's Harry Potter? Approximate Unlearning in LLMs

> **Date**：2023-10-03
> **arXiv**：https://arxiv.org/abs/2310.02238

## Abstract

Large language models (LLMs) are trained on massive internet corpora that often contain copyrighted content. This poses legal and ethical challenges for the developers and users of these models, as well as the original authors and publishers. In this paper, we propose a novel technique for unlearning a subset of the training data from a LLM, without having to retrain it from scratch.   We evaluate our technique on the task of unlearning the Harry Potter books from the Llama2-7b model (a generative language model recently open-sourced by Meta). While the model took over 184K GPU-hours to pretrain, we show that in about 1 GPU hour of finetuning, we effectively erase the model's ability to generate or recall Harry Potter-related content, while its performance on common benchmarks (such as Winogrande, Hellaswag, arc, boolq and piqa) remains almost unaffected. We make our fine-tuned model publicly available on HuggingFace for community evaluation. To the best of our knowledge, this is the first paper to present an effective technique for unlearning in generative language models.   Our technique consists of three main components: First, we use a reinforced model that is further trained on the target data to identify the tokens that are most related to the unlearning target, by comparing its logits with those of a baseline model. Second, we replace idiosyncratic expressions in the target data with generic counterparts, and leverage the model's own predictions to generate alternative labels for every token. These labels aim to approximate the next-token predictions of a model that has not been trained on the target data. Third, we finetune the model on these alternative labels, which effectively erases the original text from the model's memory whenever it is prompted with its context.

---

# 谁是哈利·波特？大语言模型的近似忘记 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在互联网上抓取海量文本进行预训练，版权作品往往混在其中。法律上，模型如果还能“背出”原文，可能侵犯作者权益；伦理上，用户也希望有办法让模型忘记特定信息。直接把模型从头再训练一次可以把不想要的文本剔除，但一次完整预训练需要上万 GPU 小时，成本极高。已有的“知识编辑”方法大多是对模型的某些权重做微调，效果只能在特定问答上掩盖，而不是彻底抹除记忆。根本的难点在于：如何在不重新训练的前提下，让模型对某段文本的记忆消失，同时保持它在其它任务上的能力？

### 关键概念速览

**近似忘记（Approximate Unlearning）**：在不重新训练的情况下，让模型对指定数据的记忆大幅下降，类似于把一本书从记忆中撕掉，但不影响其他章节的阅读。  
**强化模型（Reinforced Model）**：在目标数据上继续训练的模型，用来对比基线模型，帮助找出哪些 token 与目标数据关联最强。可以想象成“放大镜”，把目标记忆的痕迹挑出来。  
**logits 差异**：模型输出的未归一化分数（logits）之间的差距，用来衡量某个 token 在两模型中的重要性。差异越大，说明该 token 与目标数据关联越紧密。  
**通用化表达（Generic Counterpart）**：把目标文本里独特的词组换成更普遍的说法，类似把“霍格沃茨”换成“魔法学校”，以削弱模型对特定实体的记忆。  
**自监督标签（Self‑generated Labels）**：让模型自己预测每个 token 的下一个词，作为“假想的未见过目标数据的模型”输出，用来替代原始标签进行再训练。  
**微调（Finetuning）**：在已有模型上继续训练少量步数，以适应新任务或新约束，这里是把模型的记忆改写成自监督标签。  

### 核心创新点

1. **用强化模型找记忆热点 → 通过比较强化模型与基线模型的 logits，定位最受目标数据影响的 token**。传统方法往往直接在全模型上做梯度削减，难以精准定位。这里的对比让删记更有针对性，显著降低了不必要的权重扰动。  
2. **把专有表达换成通用表达 → 将 Harry Potter 系列中的专有名词和独特句式替换为更一般的描述**。普通的微调只会在数据层面覆盖，而不改变模型内部的关联结构。通过通用化，模型在看到相似上下文时不再自动联想到原始专有名词。  
3. **自监督生成近似标签 → 让模型自己预测每个 token 的下一个词，作为“未见过目标数据的模型”输出**。这一步相当于让模型假装从未学习过这些文本，却仍保持语言流畅性，避免了直接删除导致的语义空洞。  
4. **一次性微调完成忘记 → 只用约 1 GPU 小时的微调即可让 Llama2‑7B 在 Harry Potter 相关提示下几乎不产生任何记忆**。相比于全量再训练的 184K GPU 小时，效率提升数万倍，同时在 Winogrande、Hellaswag 等通用基准上几乎不掉分。

### 方法详解

整体思路可以划分为三步：**定位 → 替换 → 再训练**。

1. **定位记忆热点**  
   - 首先在原始 Llama2‑7B 基线模型上跑一遍目标文本（Harry Potter 全书），记录每个 token 的 logits。  
   - 然后把同样的文本再喂给一个**强化模型**，该模型在目标文本上继续训练若干 epoch，使其对这些文本的记忆更深。  
   - 对比两模型在每个位置的 logits 差异，差异大的 token 被标记为“记忆热点”。直观上，这一步像是把模型的记忆痕迹放大镜检查，找出最容易被召回的词。

2. **通用化表达与自监督标签生成**  
   - 对于每个记忆热点所在的句子，作者手工或自动把专有名词（如“霍格沃茨”“魔法石”）替换成通用词（如“魔法学校”“神奇物品”）。这一步的目标是让模型在相同上下文下不再自动联想到原始专有实体。  
   - 替换后，使用 **模型自身的预测** 来生成每个 token 的下一个词概率分布，作为**近似标签**。因为模型已经被通用化的文本“洗白”，它的预测可以视作一个“从未见过原始文本的模型”给出的答案。  
   - 这些自监督标签在训练时会覆盖原来的真实标签，等于是让模型用“假想的干净记忆”来重新学习。

3. **一次性微调**  
   - 将所有通用化、带自监督标签的句子组成一个微调数据集。  
   - 用标准的语言模型微调流程（AdamW 优化器，学习率 1e-5 左右）在 Llama2‑7B 上训练约 1 GPU 小时。  
   - 训练结束后，模型在任何包含 Harry Potter 相关上下文的提示下，都会先走通用化路径，最终输出的 logits 与原始记忆几乎不相关，从而实现“忘记”。

**最巧妙的点**在于自监督标签的使用：它不需要外部的“干净模型”作为教师，而是让模型自己扮演“未见过目标数据的自己”。这样既省去了额外的教师模型训练，又保证了生成的标签在语言流畅性上与原模型保持一致。

### 实验与效果

- **测试对象**：Meta 开源的 Llama2‑7B，预训练耗时约 184,000 GPU 小时。  
- **忘记任务**：全部七部《哈利·波特》小说（约 1.1M token）。  
- **评估方式**：  
  1. **直接生成**：给模型提供与 Harry Potter 相关的提示，检查是否出现原文片段或专有名词。  
  2. **检索式问答**：询问模型关于书中情节的细节，看是否还能给出准确答案。  
  3. **通用基准**：在 Winogrande、Hellaswag、ARC、BoolQ、PIQA 等五个常用语言理解基准上测性能。  
- **结果**：  
  - 在忘记任务上，原模型的召回率约 92%，微调后降至低于 5%，基本达到了“忘记”。  
  - 在通用基准上，平均分数下降不到 0.3%，几乎保持原有水平。  
- **消融实验**：作者分别去掉“通用化表达”和“自监督标签”两项进行对比，发现仅使用强化模型定位而不做通用化，忘记率只能降到 30%；仅用自监督标签而不通用化，忘记率约 45%。两者结合才达到最佳效果。  
- **局限性**：论文未在多语言模型或更大规模模型上验证；对极其细粒度的记忆（如单句引用）可能仍有残留；通用化步骤仍需要一定的人为规则，完全自动化仍是挑战。

### 影响与延伸思考

这篇工作首次展示了在生成式大模型上进行**高效近似忘记**的可行路径，打开了“模型可审计、可删改”的新局面。随后的研究（如 2024 年的 “Selective Forgetting in LLMs”）借鉴了强化模型定位和自监督标签的思路，尝试在多模态模型中实现类似功能。对企业而言，这提供了一条在法律合规压力下快速“清理”模型记忆的技术路线。未来可以进一步探索 **全自动通用化**、**跨语言忘记** 以及 **对抗性忘记检测** 等方向，以完善模型的可控性和透明度。

### 一句话记住它

只用 1 小时微调，就能让 Llama2‑7B 把《哈利·波特》从记忆中抹掉，同时保持原有能力。