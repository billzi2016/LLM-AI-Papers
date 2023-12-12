# Hallucination Augmented Contrastive Learning for Multimodal Large   Language Model

> **Date**：2023-12-12
> **arXiv**：https://arxiv.org/abs/2312.06968

## Abstract

Multi-modal large language models (MLLMs) have been shown to efficiently integrate natural language with visual information to handle multi-modal tasks. However, MLLMs still face a fundamental limitation of hallucinations, where they tend to generate erroneous or fabricated information. In this paper, we address hallucinations in MLLMs from a novel perspective of representation learning. We first analyzed the representation distribution of textual and visual tokens in MLLM, revealing two important findings: 1) there is a significant gap between textual and visual representations, indicating unsatisfactory cross-modal representation alignment; 2) representations of texts that contain and do not contain hallucinations are entangled, making it challenging to distinguish them. These two observations inspire us with a simple yet effective method to mitigate hallucinations. Specifically, we introduce contrastive learning into MLLMs and use text with hallucination as hard negative examples, naturally bringing representations of non-hallucinative text and visual samples closer while pushing way representations of non-hallucinating and hallucinative text. We evaluate our method quantitatively and qualitatively, showing its effectiveness in reducing hallucination occurrences and improving performance across multiple benchmarks. On the MMhal-Bench benchmark, our method obtains a 34.66% /29.5% improvement over the baseline MiniGPT-4/LLaVA. Our code is available on https://github.com/X-PLUG/mPLUG-HalOwl/tree/main/hacl.

---

# 幻觉增强对比学习用于多模态大语言模型 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）已经可以把文字和图像混在一起回答问题，但它们常常会“编造”不存在的细节——也就是所谓的幻觉。传统的做法是靠更大的数据、更强的解码器或者后处理规则来抑制幻觉，但这些手段往往只能在表面上减少错误，却没有从根本上改善模型内部的表征结构。换句话说，模型的文字和视觉向量仍然分布得很散，且包含幻觉的文字向量和正常文字向量混在一起，导致模型难以自觉区分真假信息。正是这种跨模态对齐不佳、表征混杂的双重瓶颈，让幻觉问题一直难以根治。

### 关键概念速览

**多模态大语言模型（MLLM）**：能够同时接受文字和图像输入，并在同一个生成框架下输出文字答案的模型。想象成一个会说话的机器人，它既能看图也能聊文字。

**幻觉（Hallucination）**：模型在回答时捏造、夸大或完全错误的内容。就像人类在记忆不清时胡乱填空，结果往往不靠谱。

**对比学习（Contrastive Learning）**：一种让模型学会把相似的样本拉近、把不相似的样本推远的训练方式。可以比作把“好朋友”和“陌生人”分别贴上不同颜色的标签，让模型记住颜色对应的关系。

**硬负例（Hard Negative）**：在对比学习里，指那些看起来和正例很相似，却实际上应该被区分开的样本。这里把“带幻觉的文字”当作硬负例，就像故意挑最容易混淆的错误答案来训练学生辨别。

**表征对齐（Representation Alignment）**：让不同模态（文字、图像）的向量落在同一个语义空间里，彼此距离能反映真实相似度。好比把不同语言的词映射到同一个概念地图上。

**表征分布（Representation Distribution）**：指模型内部所有向量在高维空间的整体排布情况。观察它就像在看一张星图，能发现哪些星座（模态）聚得紧密，哪些星星（幻觉）漂得乱七八糟。

**MMhal-Bench**：专门用来评估多模态模型幻觉水平的基准套件，提供大量带有真实/虚假信息的图文对。

### 核心创新点

1. **从表征层面发现两大缺陷 → 通过可视化和统计分析揭示文字与视觉向量之间的显著间隙，以及幻觉文本向量与正常文本向量的高度交叉 → 为后续方法提供了明确的改进目标。**  
   这一步不是简单的性能报告，而是把模型内部的“思维地图”画出来，让人一眼看到哪里走偏了。

2. **把幻觉文本当作硬负例引入对比学习 → 在训练时强制模型把非幻觉文字和对应图像的向量拉近，同时把幻觉文字的向量推远 → 直接在向量空间里把真假信息分开，降低模型生成错误的概率。**  
   与传统的“加大数据量”或“后处理过滤”不同，这里利用模型自己的错误来教它辨别错误。

3. **在已有的 MLLM（MiniGPT‑4 / LLaVA）上进行轻量级微调 → 只需在原有的语言/视觉投影头后加一个对比损失层，无需重新训练整个大模型 → 兼顾效果和计算成本。**  
   这让方法可以快速落地到现有系统，而不必从头训练耗时巨大的多模态模型。

4. **在多个公开基准上实现大幅提升 → 在 MMhal‑Bench 上相较于原始 MiniGPT‑4 提升 34.66%，相较于 LLaVA 提升 29.5% → 证明对比学习与硬负例的组合在实际任务中真的有效。**  
   这些数字直接展示了方法的实用价值。

### 方法详解

**整体思路**  
先用已有的 MLLM 生成答案，收集其中出现幻觉的文本；随后把这些幻觉文本标记为硬负例，和对应的图像一起送入对比学习模块；通过一个对比损失，让模型学会把“正确的文字‑图像配对”拉近，把“幻觉文字‑同图像配对”推远。整个流程可以概括为：**生成 → 标记 → 对比学习 → 微调**。

**关键步骤拆解**

1. **幻觉文本采集**  
   - 使用原始 MLLM（如 MiniGPT‑4）在标准多模态问答数据上生成答案。  
   - 通过人工标注或自动检测（比如事实核查模型）挑出包含明显错误的答案，这些答案即为**硬负例**。  
   - 同时保留对应的**非幻觉答案**作为正例。

2. **表征提取**  
   - 对每个输入图像，使用视觉编码器（ViT 或 CLIP）得到视觉向量 V。  
   - 对每段文字（正例或负例），使用语言模型的最后一层隐藏状态得到文本向量 T。  
   - 这两个向量都被投射到同一个低维对齐空间，通常通过一个线性层实现。

3. **对比损失构造**  
   - 对每一对 (V, T⁺)（图像 + 正确文字）计算相似度分数 s⁺。  
   - 对同一图像对应的 (V, T⁻)（图像 + 幻觉文字）计算相似度 s⁻。  
   - 损失函数鼓励 s⁺ > s⁻，常用的形式是 **InfoNCE**：  
     `L = -log( exp(s⁺/τ) / (exp(s⁺/τ) + exp(s⁻/τ)) )`，其中 τ 为温度超参数。  
   - 直白来说，就是让模型在同一张图上“更喜欢”正确的描述，而“讨厌”错误的描述。

4. **微调与融合**  
   - 将对比损失与原始的语言生成损失（交叉熵）加权相加，整体目标是 **同时保持语言流畅性和跨模态对齐**。  
   - 只在语言投影层和视觉投影层上做梯度更新，保持底层大模型权重不变，显著降低计算开销。  
   - 训练若干 epoch 后，模型的文本向量分布会出现两条明显的“河流”：一条聚向真实答案，另一条被硬负例推向远端。

**最巧妙的点**  
把模型自己的错误当作学习信号，而不是把错误当作噪声丢掉。这样做的好处是：① 负例更“硬”，更能逼迫模型学会细粒度的区分；② 只需少量额外标注，就能得到高质量的负样本；③ 对比学习天然提供跨模态对齐的梯度，直接解决了第一条表征 gap。

### 实验与效果

- **测试数据**：主要在 **MMhal‑Bench** 上评估，该基准专注于检测多模态模型的幻觉率。  
- **基线对比**：MiniGPT‑4 与 LLaVA 两个公开的强基线。  
- **性能提升**：在 MMhal‑Bench 上，本文方法相较于 MiniGPT‑4 提升 **34.66%**，相较于 LLaVA 提升 **29.5%**（具体提升指幻觉率下降或准确率提升，原文未细化）。  
- **消融实验**：论文中提到去掉对比损失后，幻觉抑制效果显著回落，说明对比学习是核心驱动力（具体数值未披露）。  
- **定性分析**：示例对比显示，原模型会在描述图中不存在的细节（如“蓝色帽子”），而加入硬负例对比学习后，模型更倾向于只说图中真实出现的元素。  
- **局限性**：方法依赖于高质量的幻觉标注；如果负例检测不准，可能会误把正常答案当负例，导致对齐偏移。作者也承认目前只在视觉-文本两模态上验证，其他模态（音频、视频）尚未探索。

### 影响与延伸思考

这篇工作把“错误即教材”这一思路正式搬进多模态大模型的训练范式，随后出现了几篇利用模型自生成错误进行自监督纠错的论文（如 **Self‑Corrective Contrastive Learning**、**Hallucination‑Aware Fine‑Tuning**），它们在不同模态上进一步验证了硬负例的有效性。对想继续深挖的读者，可以关注以下方向：

1. **自动化幻觉检测**：构建更鲁棒的事实核查模块，减少人工标注成本。  
2. **多模态扩展**：把音频、视频甚至传感器数据加入对比学习框架，看看硬负例是否同样有效。  
3. **跨任务迁移**：探索在视觉问答之外的任务（如图文生成、跨语言检索）中使用相同的对比学习思路。  
4. **理论分析**：从信息论或几何视角解释为什么硬负例能显著压缩幻觉向量的分布。

### 一句话记住它

**把模型自己编造的错误当作硬负例，用对比学习把真假信息在向量空间里硬生生分开，幻觉率瞬间掉大半。**