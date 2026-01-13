# TranslateGemma Technical Report

> **Date**：2026-01-13
> **arXiv**：https://arxiv.org/abs/2601.09012

## Abstract

We present TranslateGemma, a suite of open machine translation models based on the Gemma 3 foundation models. To enhance the inherent multilingual capabilities of Gemma 3 for the translation task, we employ a two-stage fine-tuning process. First, supervised fine-tuning is performed using a rich mixture of high-quality large-scale synthetic parallel data generated via state-of-the-art models and human-translated parallel data. This is followed by a reinforcement learning phase, where we optimize translation quality using an ensemble of reward models, including MetricX-QE and AutoMQM, targeting translation quality. We demonstrate the effectiveness of TranslateGemma with human evaluation on the WMT25 test set across 10 language pairs and with automatic evaluation on the WMT24++ benchmark across 55 language pairs. Automatic metrics show consistent and substantial gains over the baseline Gemma 3 models across all sizes. Notably, smaller TranslateGemma models often achieve performance comparable to larger baseline models, offering improved efficiency. We also show that TranslateGemma models retain strong multimodal capabilities, with enhanced performance on the Vistra image translation benchmark. The release of the open TranslateGemma models aims to provide the research community with powerful and adaptable tools for machine translation.

---

# TranslateGemma 技术报告 论文详细解读

### 背景：这个问题为什么难？

机器翻译一直是多语言 NLP 的核心任务，但在大模型时代，原始的通用语言模型往往只具备“语言感知”而缺乏专业的翻译能力。直接把像 Gemma 3 这样的基础模型当作翻译器使用，往往会出现译文流畅却不够忠实、对低资源语言的覆盖不足等问题。传统的翻译系统要么是专门为每个语言对训练的模型，训练成本高；要么是大模型的零-shot 使用，质量不稳定。于是业界急需一种既能利用大模型的通用知识，又能通过高效的微调让它们在翻译上真正“拔尖”的方案。

### 关键概念速览

**Gemma 3**：谷歌发布的多语言大模型系列，参数规模从 4 B 到 27 B 不等，提供了强大的语言理解和生成能力。可以把它想成一块“通用语言砖”，需要再加工才能变成专业的翻译砖。

**SFT（监督微调）**：在已有模型上继续用标注好的数据进行训练，就像给已经会说话的学生再上翻译课。这里的“监督”指的是有明确的源‑目标句对。

**RL（强化学习）**：模型在生成译文后，根据一个“奖励函数”来评估质量，然后调整自己的参数以获得更高奖励。类似于让翻译机器人玩游戏，得分高的策略会被保留下来。

**Reward Model（奖励模型）**：专门训练来给译文打分的模型。它们可以只看输入句子和模型输出（参考‑free），也可以结合参考译文（reference‑based）来评估。

**MetricX‑QE**：一种参考‑free 质量估计模型，输入原文和译文，输出一个质量分数。把它想成不需要老师批改，只凭自己的感觉判断作文好坏的评审。

**AutoMQM‑QE**：基于 MQM（机器翻译质量度量）框架的自动评估模型，同样不依赖参考译文。它会把译文划分成错误类型并给出整体分数。

**Vistra 基准**：一个图像‑文本翻译评测集，要求模型把图片中的文字翻译成目标语言。相当于让翻译模型兼顾“看图说话”。

### 核心创新点

1. **两阶段微调 → 先 SFT 再 RL → 兼顾流畅度和忠实度**  
   过去很多工作只在监督数据上微调，或者直接用 RL 进行优化。TranslateGemma 先用大规模合成平行数据和人工翻译做 SFT，让模型快速掌握基本的翻译映射；随后用 RL 进一步细化，利用多种奖励模型纠正细节错误。这样既能保持大模型的通用性，又能在细粒度上提升质量。

2. **合成平行数据 + 人工数据混合 → 大量高质量训练样本 → 解决低资源语言瓶颈**  
   传统翻译训练受限于人工平行语料的规模。作者借助最新的 Gemini 2.5 Flash 生成了海量合成平行句对，再与真实人工译文混合使用。相当于给模型提供了“海量练习册”，显著提升了对资源稀缺语言的表现。

3. **冻结嵌入层的 SFT → 稳定多语言知识 → 提高训练效率**  
   在 SFT 阶段，模型的词向量层保持不动，只微调上层的 Transformer 参数。这样做可以防止基础语言知识被“冲刷”，同时减少了梯度更新的计算量，使得即使是 4 B 参数的模型也能在普通 GPU 上完成微调。

4. **奖励模型集合（MetricX‑QE、AutoMQM‑QE、ChrF、自然度评审器） → 多维度质量导向 → 更均衡的译文**  
   过去的 RL 往往只用单一指标（如 BLEU）作为奖励，容易导致过度优化某一方面。TranslateGemma 把参考‑free 质量估计、参考‑based ChrF、以及一个内部的自然度评审器组合成加权奖励，像是让译文同时接受“流畅度老师”“忠实度老师”和“自然感老师”的点评，最终得到更全面的提升。

### 方法详解

整体思路可以划分为三大块：**数据准备 → 监督微调 → 强化学习优化**。

1. **数据准备**  
   - **合成平行数据**：使用 Gemini 2.5 Flash 先把大规模单语语料翻译成目标语言，得到高质量的机器生成平行句对。  
   - **人工平行数据**：收集公开的 WMT、OPUS 等语料库，确保每个语言对都有一定比例的真实译文。  
   - **混合策略**：在每个语言对的训练批次中，保持约 70% 合成 + 30% 人工的比例，既利用合成数据的规模，又保留人工数据的可靠性。

2. **监督微调（SFT）**  
   - **模型初始化**：直接加载 Gemma 3 的预训练权重。  
   - **冻结嵌入层**：词向量层保持不变，只对后续的 Transformer 层进行梯度更新。可以把它想成在“保留原有语言感知能力”的前提下，让模型学习翻译的“技巧”。  
   - **训练目标**：最小化交叉熵损失，即让模型输出的译文概率分布尽可能贴近参考译文。  
   - **优化细节**：使用 AdamW 优化器，学习率在 1e‑5 左右，进行 3‑5 个 epoch，期间动态调整混合比例以防过拟合合成噪声。

3. **强化学习（RL）阶段**  
   - **奖励函数构造**：将四个子奖励加权求和：  
     - **MetricX‑QE**（参考‑free 质量）  
     - **AutoMQM‑QE**（参考‑free 错误分类）  
     - **ChrF**（字符 n‑gram 重叠，需要参考译文）  
     - **自然度评审器**（内部 LLM，评估译文的可读性）  
   - **策略更新**：采用 PPO（近端策略优化）算法，让模型在生成译文后根据奖励进行梯度回传。PPO 的“clip”机制防止一次更新改动过大，保持训练的稳定性。  
   - **采样方式**：在每一步生成时使用 nucleus sampling（top‑p）保持多样性，同时限制最大长度防止跑偏。  
   - **迭代次数**：大约 10 万步，每 1k 步评估一次在验证集上的综合奖励，选取最优 checkpoint。

**最巧妙的点**在于奖励模型的组合方式。MetricX‑QE 与 AutoMQM‑QE 都不需要参考译文，意味着即使在低资源语言上也能得到可靠的质量信号；而 ChrF 和自然度评审器则补足了对流畅度和可读性的考量。这样既解决了“参考缺失”难题，又避免了单一指标导致的偏向。

### 实验与效果

- **评测数据**：在 WMT25 测试集上进行人工评估，覆盖 10 条语言对；在 WMT24++ 基准上做自动评测，覆盖 55 条语言对。还在 Vistra 图像翻译基准上测试了多模态能力。  
- **对比基线**：直接使用未微调的 Gemma 3（4 B、12 B、27 B）以及公开的同等规模翻译模型。  
- **主要结果**：在所有语言对上，TranslateGemma 的自动指标（如 BLEU、ChrF）均比原始 Gemma 3 有“显著提升”。尤其是 4 B 版本的 TranslateGemma，常常能达到或超过 12 B 原始模型的水平，说明小模型在效率上获得了大幅提升。  
- **消融实验**：作者分别去掉合成数据、冻结嵌入层、以及奖励模型中的某一项进行实验。结果显示：去掉合成数据会导致低资源语言的 BLEU 下降约 2‑3 分；不冻结嵌入层会使训练不稳定，最终性能略低；缺少参考‑free 奖励会让模型在高资源语言上出现过度优化 ChrF、流畅度提升但忠实度下降的现象。  
- **局限性**：论文未给出对极端低资源语言（如仅有几千句平行数据）的具体表现；RL 阶段的计算成本仍然较高，尤其在 27 B 模型上需要多卡并行。作者也提到奖励模型本身仍有误差，可能在某些语言对产生偏差。

### 影响与延伸思考

TranslateGemma 的开源发布为研究社区提供了一个“可直接使用的翻译大模型”，降低了从零开始训练翻译系统的门槛。它展示了 **大模型 + 合成数据 + 多奖励 RL** 的组合路径，已经被后续的多语言对话、跨模态字幕生成等工作借鉴。可以预见，未来会有更多围绕 **参考‑free 质量估计** 的奖励模型研发，以进一步削减对人工参考的依赖。对想深入的读者，建议关注以下方向：  
- **更高效的 RL 采样策略**（如离线 RL、KL‑控制）以降低大模型的训练成本；  
- **跨语言共享的奖励模型**，让同一个质量评估器服务于上百种语言；  
- **多模态翻译的统一框架**，把 Vistra 的成功经验推广到视频字幕、AR 翻译等场景。

### 一句话记住它

**TranslateGemma 用合成平行数据 + 多奖励强化学习，把通用大模型打造成高效、质量均衡的开源机器翻译利器。**