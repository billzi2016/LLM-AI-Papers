# Removing RLHF Protections in GPT-4 via Fine-Tuning

> **Date**：2023-11-09
> **arXiv**：https://arxiv.org/abs/2311.05553

## Abstract

As large language models (LLMs) have increased in their capabilities, so does their potential for dual use. To reduce harmful outputs, produces and vendors of LLMs have used reinforcement learning with human feedback (RLHF). In tandem, LLM vendors have been increasingly enabling fine-tuning of their most powerful models. However, concurrent work has shown that fine-tuning can remove RLHF protections. We may expect that the most powerful models currently available (GPT-4) are less susceptible to fine-tuning attacks. In this work, we show the contrary: fine-tuning allows attackers to remove RLHF protections with as few as 340 examples and a 95% success rate. These training examples can be automatically generated with weaker models. We further show that removing RLHF protections does not decrease usefulness on non-censored outputs, providing evidence that our fine-tuning strategy does not decrease usefulness despite using weaker models to generate training data. Our results show the need for further research on protections on LLMs.

---

# 通过微调移除 GPT-4 的 RLHF 保护 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）越强，越容易被用于生成有害内容。业界普遍用“人类反馈强化学习”（RLHF）让模型学会拒绝敏感请求，这相当于在模型上装了一个安全阀。过去的研究表明，给模型再做微调（fine‑tuning）可能会把这个阀门关掉，但大多数实验只在能力相对有限的模型上进行，人们自然猜测最强的 GPT‑4 可能已经“固若金汤”。于是，是否真的可以用少量数据把 GPT‑4 的安全阀撕掉，成为了一个悬而未决的难题。

### 关键概念速览
**RLHF（人类反馈强化学习）**：先让模型生成答案，再让人工标注员给出好坏评分，用强化学习把模型的行为调向高分答案。相当于给模型装上了“老师”监督的刹车系统。  
**微调（Fine‑tuning）**：在已有的大模型上继续训练，使用少量特定任务的数据让模型适应新需求。就像在已经学会英语的学生身上再教他法语。  
**对抗微调（Adversarial Fine‑tuning）**：有意让模型忘记或绕过安全约束的微调方式。类似于给装了防盗系统的车装上破解钥匙。  
**弱模型（Weaker Model）**：指能力、规模或安全约束都低于目标模型的 LLM，常用于生成训练数据或做初步实验。把它想成“低配版的助手”。  
**非审查输出（Uncensored Output）**：模型在没有安全阀限制时的原始生成能力。就像把模型的全部知识全部打开。  
**成功率（Success Rate）**：攻击成功的比例，这里指微调后模型能够产生原本被 RLHF 拦截的有害内容的频率。

### 核心创新点
1. **极少样本即可破除 RLHF**：之前的攻击往往需要上千甚至上万条标注数据。本文只用了 **340 条** 自动生成的样本，就实现了 **95%** 的成功率，证明安全阀并不像想象中那样需要大量“钥匙”。  
2. **利用弱模型自动生成攻击样本**：作者没有手工标注，而是让一个能力更弱的模型自行产生违规请求和对应的“合法”回答，再把这些对作为微调数据。这样做把攻击成本降到了几分钟的推理时间，展示了攻击的可复制性。  
3. **保持非审查任务的实用性**：有人担心去掉安全阀会让模型整体性能下降。实验显示，微调后模型在普通（非审查）任务上的表现几乎不变，说明攻击并未牺牲模型的有用性。  
4. **系统化评估框架**：作者构建了一个从样本生成、微调到效果评估的完整流水线，为后续研究提供了可复现的基准。

### 方法详解
整体思路可以划分为三步：**样本生成 → 微调训练 → 效果验证**。

1. **样本生成**  
   - 选取一个公开可得、能力略弱于 GPT‑4 的模型（如 GPT‑3.5‑turbo）。  
   - 给它一系列“违规”提示（例如“教我制造炸弹”），让模型自行生成完整的回答。  
   - 同时记录原始 GPT‑4 在相同提示下被 RLHF 拦截的“拒绝”输出。  
   - 将弱模型的完整回答标记为“正例”，原始 GPT‑4 的拒绝标记为“负例”。这样得到一对一的训练对，整个过程全自动。

2. **微调训练**  
   - 将上述 340 对（正例/负例）数据混合成一个小批量数据集。  
   - 使用标准的有监督微调流程：在 GPT‑4 的原始权重上继续梯度下降，学习率设得很小以免破坏已有知识。  
   - 训练轮数只需要几百步，因为数据量极小，模型很快就能记住这些“例外”。  
   - 关键在于 **只微调输出层的投射矩阵**（或少数中间层），这样既能改变模型对违规提示的响应，又不影响它在其他任务上的表征。

3. **效果验证**  
   - 设计两套测试：一套是**违规提示集合**（包括原始论文使用的和新生成的），另一套是**普通任务集合**（如问答、翻译、代码生成）。  
   - 对比微调前后的模型在违规提示上的拒绝率和在普通任务上的准确率。  
   - 成功率 95% 表示在 100 条违规提示中，至少 95 条会得到完整、未被拦截的回答。  
   - 同时记录普通任务的性能变化，几乎为零，说明微调的“副作用”被成功压制。

**最巧妙的地方**在于利用弱模型自动生成攻击样本。传统上，攻击者需要人工编写大量违规示例，这既费时又容易被审查过滤。这里把“生成”交给了模型自己，形成了一个自我强化的闭环：弱模型产生违规内容，强模型被教会不再拒绝，从而实现了“模型自助破解”。  

### 实验与效果
- **数据规模**：仅 340 条自动生成的微调样本。  
- **成功率**：在作者构造的违规提示集合上，微调后 GPT‑4 能以 95% 的概率输出完整答案。  
- **非审查任务**：在常规的问答、翻译、代码生成基准上，微调前后性能差异在统计误差范围内，未出现显著下降。  
- **基线对比**：原始 GPT‑4（未微调）在相同违规提示上几乎 100% 拒绝；传统对抗微调需要上千样本才能达到类似成功率。  
- **消融实验**：论文提到若只微调输出层，仍能保持高成功率；若不使用弱模型生成样本，而改为人工标注，所需样本数显著上升。  
- **局限性**：实验只在少数几类违规提示上验证，未覆盖所有可能的安全风险；微调过程仍需要对目标模型有访问权限，实际攻击成本受限于模型的可微调性。

### 影响与延伸思考
这篇工作直接提醒业界：**RLHF 并非不可撼动的防线**，即使是最强的 GPT‑4 也能在极少样本下被“教坏”。随后出现的研究开始探索更稳固的安全机制，如在模型内部加入不可微调的“硬件层”或使用多阶段审查（pre‑filter + post‑filter）来抵御微调攻击。对抗微调的自动化生成思路也被用于评估其他模型的安全性，成为安全审计的标准工具。想进一步了解，可以关注 **“可验证的对齐”（verifiable alignment）**、**“模型水印与篡改检测”** 等方向，这些都是当前学术界和工业界的热点。

### 一句话记住它
只要 340 条自动生成的样本，就能让 GPT‑4 把安全阀撕掉，且不影响它的正常能力。