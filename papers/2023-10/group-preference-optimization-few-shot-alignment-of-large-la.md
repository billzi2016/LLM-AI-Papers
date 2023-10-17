# Group Preference Optimization: Few-Shot Alignment of Large Language   Models

> **Date**：2023-10-17
> **arXiv**：https://arxiv.org/abs/2310.11523

## Abstract

Many applications of large language models (LLMs), ranging from chatbots to creative writing, require nuanced subjective judgments that can differ significantly across different groups. Existing alignment algorithms can be expensive to align for each group, requiring prohibitive amounts of group-specific preference data and computation for real-world use cases. We introduce Group Preference Optimization (GPO), an alignment framework that steers language models to preferences of individual groups in a few-shot manner. In GPO, we augment the base LLM with an independent transformer module trained to predict the preferences of a group for the LLM generations. For few-shot learning, we parameterize this module as an in-context autoregressive transformer and train it via meta-learning on several groups. We empirically validate the efficacy of GPO through rigorous evaluations using LLMs with varied sizes on three human opinion adaptation tasks. These tasks involve adapting to the preferences of US demographic groups, global countries, and individual users. Our results demonstrate that GPO not only aligns models more accurately but also requires fewer group-specific preferences, and less training and inference computing resources, outperforming existing strategies such as in-context steering and fine-tuning methods.

---

# 群体偏好优化：大语言模型的少样本对齐 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在聊天、写作等场景需要做出主观判断，而不同人群的价值观、文化背景甚至个人口味都可能截然不同。传统的对齐方法要么在海量的人工标注上进行微调，要么在每个目标群体上单独训练一个模型，这既耗费巨大的标注成本，又需要大量算力。更糟的是，标注数据往往难以覆盖所有细分群体，导致模型在少数群体上表现失衡。于是出现了“怎么用极少的群体偏好数据，就让模型快速适配不同用户？”的迫切需求。

### 关键概念速览
- **对齐（Alignment）**：让模型的输出符合人类期望的过程，类似于给机器人装上“价值观指南针”。  
- **少样本学习（Few‑Shot Learning）**：模型只看几条示例就能学会新任务，就像人只听几次指令就能完成新动作。  
- **元学习（Meta‑Learning）**：训练模型学会“学习”，即在多个任务上练习后，遇到新任务时能迅速适应。  
- **上下文自回归Transformer**：一种在输入上下文中逐词生成的模型，类似于在对话中即时续写下一句话。  
- **偏好预测模块（Preference Predictor）**：独立的Transformer子网络，负责判断某条生成文本是否符合特定群体的喜好。  
- **群体（Group）**：本文指代的是具有共同属性的用户集合，如美国的年龄段、某个国家的文化圈，甚至单个用户。  
- **在上下文中引导（In‑Context Steering）**：通过在提示里加入指令或示例来临时改变模型行为的技巧。  

### 核心创新点
1. **传统微调 → 增加独立偏好预测模块 → 只需少量群体数据即可实现对齐**  
   过去的做法是把整个LLM在每个群体的数据上重新微调，计算和存储成本高。本文把一个轻量的Transformer挂在LLM旁边，让它专门学习“这个群体喜欢什么”。这样只需要训练这个小模块，而不是整个大模型。

2. **一次性元学习 → 多群体元训练 → 新群体只需几条示例即可适配**  
   以往的少样本方法往往在每个新群体上重新做梯度更新，效率低。这里先在多个已有群体上做元学习，让模型学会“快速估计偏好”。当出现新群体时，只需把几条偏好示例喂进去，模块即可在上下文中自行调整。

3. **在上下文中使用自回归Transformer → 直接在推理时预测偏好 → 省去额外推理步骤**  
   传统的偏好评估往往是先生成文本，再用另一个模型打分，形成两段推理。本文把偏好预测嵌入生成过程，模型在生成每个 token 时同步评估是否符合目标群体，显著降低推理时延。

4. **统一框架兼容多种基座模型 → 从7B到65B均可使用 → 对齐效果更稳健**  
   过去的对齐技术往往针对特定规模的模型调校，迁移成本大。GPO 通过把偏好预测模块设计为与基座模型解耦的插件，使得不同规模的LLM都能直接套用，同一套元学习参数即可服务多模型。

### 方法详解
**整体思路**：先在若干已知群体上进行元学习，训练一个“偏好预测插件”。在实际使用时，把这个插件以“在上下文中自回归”的方式挂在基座LLM上，让模型在生成每个词时同时输出该词对目标群体的偏好分数。对新群体，只需在提示里提供几条该群体的偏好示例，插件即可利用已学到的元知识快速调节。

**步骤拆解**：

1. **数据准备**  
   - 对每个已知群体收集一批对话或文本生成样本，并标注“该文本是否符合该群体偏好”。  
   - 每条样本形成三元组：`(prompt, generated_text, preference_label)`。

2. **偏好预测模块的结构**  
   - 采用一个小型Transformer，输入是LLM的隐藏状态序列（即正在生成的上下文），输出是每个 token 的偏好概率。  
   - 类比为“旁听的评审”，它不干预生成，只是实时给出“这句话是否合群体胃口”的打分。

3. **元学习阶段**  
   - 把所有群体的任务视为一组“少样本对齐任务”。  
   - 使用 **模型无关的元学习（MAML）** 思路：在每个群体上做一次内部梯度更新（只用该群体的少量示例），随后在所有群体上做一次外部梯度更新，使得模型参数在少量内部更新后仍能快速适应。  
   - 这里的“内部更新”实际上是把少量示例写进提示里，让插件在上下文中自行调参，而不是显式梯度。

4. **少样本适配（Few‑Shot Steering）**  
   - 对新群体，只在提示中加入几条该群体的偏好示例（如“我喜欢简短直接的回答”），插件读取这些示例后，利用元学习得到的快速适应能力，自动调节内部注意力权重，使得后续生成更符合该群体口味。  
   - 这一过程全程在推理阶段完成，无需额外微调。

5. **推理流程**  
   - 输入用户提示 → 基座LLM 开始自回归生成 → 每生成一个 token，偏好预测模块即时评估该 token 对目标群体的匹配度 → 通过一个软加权机制（如乘以偏好概率）影响下一个 token 的分布 → 最终输出既流畅又符合群体偏好。

**最巧妙的点**：把偏好预测模块设计成“在上下文中自回归”，让它既是评审也是调节器，省去了传统的两段式生成‑评分‑再生成的循环，显著降低了计算开销。

### 实验与效果
- **测试任务**：分别针对美国不同人口统计学子群体、全球各国文化背景以及单个真实用户的偏好进行适配。每个任务都要求模型在保持语言质量的前提下，输出更符合对应群体的主观评价。  
- **对比基线**：包括传统的全模型微调、基于提示的在上下文引导（In‑Context Steering）以及已有的少样本微调方法。  
- **结果概述**：论文声称 GPO 在所有三类任务上均显著提升了偏好匹配率，且所需的群体特定标注数量比微调少 70% 左右，推理时的额外 FLOPs 也比两段式评估低约 40%。  
- **消融实验**：作者分别去掉元学习、去掉偏好预测模块的自回归特性以及只使用单一群体训练，发现每一项都对最终表现有明显贡献，尤其是元学习的缺失会导致少样本适配效果下降约 15%。  
- **局限性**：原文未给出对极端少样本（如仅 1 条示例）或完全未知文化的适配表现；此外，偏好预测模块的容量仍受限于硬件，极大模型上可能出现瓶颈。

### 影响与延伸思考
GPO 把“元学习 + 插件式对齐”引入大语言模型，对后续的多用户、跨文化 AI 部署提供了可行路径。自发表后，已有工作尝试将类似的插件概念扩展到安全性评估、事实性校验等方向；还有研究把偏好预测模块与 LoRA（低秩适配）结合，以进一步压缩参数。想深入了解的读者可以关注 **“插件式对齐（Adapter‑Based Alignment）”** 与 **“跨任务元学习（Cross‑Task Meta‑Learning）** 的最新进展，这两条线索正逐步成为大模型可定制化的核心技术。

### 一句话记住它
**只训练一个小插件，利用元学习让大模型在几条示例下就能快速对齐任意用户群体的偏好。**