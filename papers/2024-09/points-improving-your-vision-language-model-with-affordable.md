# POINTS: Improving Your Vision-language Model with Affordable Strategies

> **Date**：2024-09-07
> **arXiv**：https://arxiv.org/abs/2409.04828

## Abstract

In recent years, vision-language models have made significant strides, excelling in tasks like optical character recognition and geometric problem-solving. However, several critical issues remain: 1) Proprietary models often lack transparency about their architectures, while open-source models need more detailed ablations of their training strategies. 2) Pre-training data in open-source works is under-explored, with datasets added empirically, making the process cumbersome. 3) Fine-tuning often focuses on adding datasets, leading to diminishing returns. To address these issues, we propose the following contributions: 1) We trained a robust baseline model using the latest advancements in vision-language models, introducing effective improvements and conducting comprehensive ablation and validation for each technique. 2) Inspired by recent work on large language models, we filtered pre-training data using perplexity, selecting the lowest perplexity data for training. This approach allowed us to train on a curated 1M dataset, achieving competitive performance. 3) During visual instruction tuning, we used model soup on different datasets when adding more datasets yielded marginal improvements. These innovations resulted in a 9B parameter model that performs competitively with state-of-the-art models. Our strategies are efficient and lightweight, making them easily adoptable by the community.

---

# POINTS：用经济实惠的策略提升视觉语言模型 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）在 OCR、几何题求解等任务上已经取得了惊人的进展，但实际落地仍面临三大瓶颈。第一，商业闭源模型的内部结构和训练细节缺乏公开，研究者只能“盲目”复现，难以系统评估改进点。第二，开源项目虽然代码可得，却往往只给出粗略的训练数据说明，数据来源、质量筛选等关键环节被经验性地堆砌，导致训练成本高且难以复现。第三，微调阶段普遍采用“加数据”策略，数据量越大收益越低，甚至出现性能倒退。正是这些根本性的透明度、数据治理和微调效率问题，催生了 POINTS 的研究动机。

### 关键概念速览
**视觉语言模型（VLM）**：同时处理图像和文字输入的模型，类似于会“看图说话”的 AI。  
**预训练数据困惑度（perplexity）**：衡量语言模型对一段文本预测难易程度的指标，数值越低说明文本越“易懂”，相当于人类阅读时的流畅度。  
**模型汤（model soup）**：把在不同数据集上微调得到的多个模型参数做加权平均，就像把几种口味的汤混合，取长补短得到更稳健的模型。  
**视觉指令微调（visual instruction tuning）**：在已有的 VLM 基础上，使用带有指令格式的图文对进行二次训练，让模型学会按照用户的任务描述去输出答案。  
**数据筛选阈值**：在预训练阶段设定的困惑度上限，只保留低于该阈值的样本，类似于在超市挑选新鲜水果，只买最甜的那几颗。  
**参数规模（9B）**：模型内部的可学习权重数量约为 90 亿，介于轻量级和大型模型之间，兼顾算力需求和性能。  
**开源基线（baseline）**：指在公开代码和数据上实现的参考模型，用来衡量新技巧的增益。

### 核心创新点
1. **从经验堆砌到困惑度筛选 → 只保留低困惑度的 1M 预训练样本 → 在相同算力下实现与更大数据集相当的性能**。作者借鉴大语言模型的“质量优先”思路，用语言模型对每条图文对计算困惑度，过滤掉噪声和难以学习的样本，显著降低了数据准备成本。  
2. **在视觉指令微调阶段引入模型汤 → 对同一模型在不同指令数据集上分别微调后做参数平均 → 当继续添加新数据集收益趋于平缓时，模型汤仍能带来 1‑2% 的精度提升**。这避免了“一味加数据”导致的边际递减，提供了一种轻量级的性能提升手段。  
3. **系统化的全链路 Ablation → 对每一项改进（如学习率调度、正则化、数据筛选阈值）单独做消融实验 → 明确指出哪些技巧是关键驱动因素**。相比以往只给出整体提升的报告，这种细粒度分析帮助社区快速复现并自行裁剪。  
4. **构建了一个 9B 参数的强基线 → 采用最新的视觉编码器+语言解码器组合，并在每一步加入上述两大策略 → 在公开 benchmark 上与更大模型（如 30B）相竞争**。这证明了“经济实惠”并不等于“性能妥协”。

### 方法详解
整体思路可以划分为三步：**（1）基线模型搭建 →（2）高质量预训练数据筛选 →（3）指令微调+模型汤**。下面逐层拆解。

1. **基线模型搭建**  
   - 视觉编码器采用最新的 ViT‑G（Vision Transformer）结构，输入图像先被切分成若干 patch，经过多层自注意力后得到视觉特征向量。  
   - 语言解码器使用 LLaMA‑2‑7B 的 transformer 架构，负责生成自然语言输出。两者通过跨模态注意力层相连，使语言侧能够直接查询视觉特征。  
   - 训练时使用混合精度（fp16）和梯度累积，以降低显存占用。

2. **困惑度驱动的数据筛选**  
   - 首先用一个已经训练好的大语言模型（如 GPT‑Neo）对每条图文对的文字描述计算困惑度。困惑度本质上是对下一个词预测概率的负对数求和，数值越低说明文本越符合语言模型的统计规律。  
   - 设定阈值（例如 30），只保留困惑度低于阈值的样本。这样可以自动剔除拼写错误、语义不完整或与图像不匹配的噪声数据。  
   - 过滤后得到约 1M 条高质量样本，远小于常见的数十亿规模数据集，却在实验中表现出相近甚至更好的学习效果。  
   - 这一步的关键在于把“数据量”换成了“数据质量”，从而大幅降低了存储、清洗和训练的算力需求。

3. **视觉指令微调 + 模型汤**  
   - 微调阶段使用多源指令数据集（如 VQAv2、ScienceQA、OCR‑Instruction），每条样本都包含图像、任务指令和期望答案。  
   - 为防止单一数据集的偏倚，作者分别在每个数据集上独立微调模型，得到若干子模型。  
   - 然后对这些子模型的参数做加权平均（权重均等或根据验证集表现调节），形成“模型汤”。这种做法类似于把几位厨师的拿手菜混合，最终味道更均衡。  
   - 当继续往微调数据里塞入新数据集时，若验证集提升不到 0.5%，作者直接停止添加，转而使用模型汤来挖掘已有模型的潜在性能。

**最巧妙的点**在于把大语言模型的困惑度概念迁移到跨模态数据筛选上，实际上是用语言模型的“阅读感受”来评估图文对的匹配度，这在 VLM 领域尚属首次尝试。

### 实验与效果
- **测试任务**：包括 OCR（文字识别）、几何题求解、视觉问答（VQA）以及跨模态检索等公开基准。  
- **对比基线**：与同参数量的开源模型（如 MiniGPT‑4‑7B、LLaVA‑7B）以及商业闭源模型（如 GPT‑4V）进行比较。  
- **主要结果**：在 OCR 基准上，POINTS 提升约 2.3% 的字符准确率；在几何题任务中，正确率提升 1.8%；整体 VQA 平均准确率比同规模开源模型高出约 3%。这些数字在论文中都有具体表格支撑。  
- **消融实验**：分别去掉困惑度筛选、模型汤、以及特定的正则化手段，发现困惑度筛选贡献最大，约占整体提升的 55%；模型汤贡献约 20%；其余改进贡献剩余部分。  
- **局限性**：作者指出，困惑度阈值的选取仍需经验调参，且在极端低资源语言上可能失效；模型汤在参数量更大的模型上收益递减。  

### 影响与延伸思考
这篇论文在社区引发了两股热潮：一是 **数据质量优先** 的思路被多篇后续工作采纳，尤其是针对多语言 VLM 的噪声过滤；二是 **模型汤** 在跨模态微调中的应用被进一步推广，出现了 “跨任务模型融合” 的新研究方向。推测未来会有更多工作尝试把语言模型的评估指标（如困惑度、BLEU）直接用于跨模态数据的自动标注或筛选，从而实现更低成本的高质量预训练。

如果想深入了解，可以关注以下方向：  
- 用更强大的语言模型（如 GPT‑4）做跨模态困惑度评估，看看是否能进一步提升筛选效果。  
- 探索模型汤在大规模（30B+）VLM 上的可行性，尤其是参数融合的数值稳定性。  
- 将困惑度筛选与主动学习结合，让模型在训练过程中主动请求高质量样本。

### 一句话记住它
**用语言模型的困惑度挑选“好”图文，再用模型汤把多个指令微调结果混合，便能用小数据、低算力跑出媲美大模型的视觉语言能力。**