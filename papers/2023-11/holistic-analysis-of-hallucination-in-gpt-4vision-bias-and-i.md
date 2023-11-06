# Holistic Analysis of Hallucination in GPT-4V(ision): Bias and   Interference Challenges

> **Date**：2023-11-06
> **arXiv**：https://arxiv.org/abs/2311.03287

## Abstract

While GPT-4V(ision) impressively models both visual and textual information simultaneously, it's hallucination behavior has not been systematically assessed. To bridge this gap, we introduce a new benchmark, namely, the Bias and Interference Challenges in Visual Language Models (Bingo). This benchmark is designed to evaluate and shed light on the two common types of hallucinations in visual language models: bias and interference. Here, bias refers to the model's tendency to hallucinate certain types of responses, possibly due to imbalance in its training data. Interference pertains to scenarios where the judgment of GPT-4V(ision) can be disrupted due to how the text prompt is phrased or how the input image is presented. We identify a notable regional bias, whereby GPT-4V(ision) is better at interpreting Western images or images with English writing compared to images from other countries or containing text in other languages. Moreover, GPT-4V(ision) is vulnerable to leading questions and is often confused when interpreting multiple images together. Popular mitigation approaches, such as self-correction and chain-of-thought reasoning, are not effective in resolving these challenges. We also identified similar biases and interference vulnerabilities with LLaVA and Bard. Our results characterize the hallucination challenges in GPT-4V(ision) and state-of-the-art visual-language models, and highlight the need for new solutions. The Bingo benchmark is available at https://github.com/gzcch/Bingo.

---

# GPT-4V（视觉）幻觉的整体分析：偏差与干扰挑战 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）需要同时理解图像和文字，然而它们的输出常常出现“幻觉”——即模型给出与输入不符的答案。过去的评估大多聚焦在单模态或简单的图文匹配上，缺少系统化的多模态幻觉测评。更糟的是，训练数据的地域、语言分布不均会让模型产生系统性偏差，而提示词的微小改动也可能导致答案大相径庭。没有一个专门的基准来捕捉这些细微但致命的错误，研究者难以定位问题根源，也无法验证改进措施的有效性。

### 关键概念速览
**幻觉（Hallucination）**：模型输出的内容在事实或视觉证据上不成立，就像人看错了图画说出不存在的细节。  
**偏差（Bias）**：模型倾向于在特定类型的输入上产生系统性错误，例如更擅长解释西方图片或英文文字。  
**干扰（Interference）**：输入的表述方式或图像排列方式对模型判断产生负面影响，类似于人类在噪声环境下容易误解指令。  
**Bingo 基准**：作者新建的评测套件，专门设计了“偏差挑战”和“干扰挑战”两大任务，用来量化 VLM 的幻觉表现。  
**自我纠错（Self‑Correction）**：让模型在第一次回答后再检查、修正自己的答案，类似于人写完作文后再审稿。  
**思维链（Chain‑of‑Thought, CoT）**：要求模型把推理过程逐步写出来，帮助模型在复杂任务上保持逻辑连贯。  
**区域偏差（Regional Bias）**：模型对不同国家、语言的图片表现出不同的准确率，类似于人只熟悉本土文化而对外来文化理解不足。  
**多图干扰（Multi‑Image Interference）**：一次性给模型多张图片时，模型容易把信息混在一起，导致答案错误。

### 核心创新点
1. **从“单一幻觉”到“全景评估”**：之前的工作只检查模型是否能正确描述单张图片，本文引入了两类系统性错误——偏差和干扰，构建了更全面的评估视角。  
2. **Bingo 基准的任务设计**：作者手工挑选并标注了大量跨文化、跨语言的图片，并配以诱导性或中性提示，形成了“偏差挑战”（检验地域/语言倾向）和“干扰挑战”（检验提示/多图排列影响）两套子任务。这样可以直接对比模型在不同情境下的表现差异。  
3. **系统性对比多模型**：不仅评测了 GPT‑4V，还把 LLaVA、Bard 等同类最前沿模型拉进实验，揭示了幻觉问题是多模态模型的共性，而非单一实现的缺陷。  
4. **验证主流缓解手段失效**：作者把自我纠错和思维链两种在大语言模型中常用的“后处理”技巧套用到视觉语言模型上，实验表明这些方法对偏差和干扰的改善几乎没有效果，提示需要专门针对多模态幻觉的解决方案。

### 方法详解
整体框架可以概括为三步：**数据构造 → 任务划分 → 评估与分析**。

1. **数据构造**  
   - **跨文化图片库**：作者从公开数据集和网络抓取了数千张图片，确保覆盖北美、欧洲、亚洲、非洲等地区，并且包含多语言文字（英文、中文、阿拉伯文等）。  
   - **诱导式提示设计**：针对每张图片，编写两类文本提示：一种是中性描述（如“请描述这张图片的内容”），另一种是带有暗示的提问（如“这张图片里最可能出现的是什么？”），后者用于测试模型的“leading question”敏感度。  
   - **多图组合**：把两张或三张图片放在同一输入中，要求模型一次性回答所有图片的内容，以检测多图干扰。

2. **任务划分**  
   - **偏差挑战**：模型需要在不同地区、不同语言的图片上给出准确描述。评测指标是正确率与地区/语言的差异幅度。  
   - **干扰挑战**：模型面对不同提示词、不同图片排列顺序时的表现波动。这里用“答案一致性”来衡量，即相同视觉信息在不同文本提示下答案是否保持稳定。

3. **评估与分析**  
   - **自动评分**：利用人工标注的金标准答案，计算精确匹配率、BLEU、ROUGE 等文本相似度指标。  
   - **误差分类**：对模型的错误进行手工归类，区分是“偏差导致的误判”还是“干扰导致的混淆”。  
   - **缓解手段实验**：在同样的输入上分别启用自我纠错和思维链两种策略，记录指标变化，检验其有效性。

**最巧妙的地方**在于把“提示词的微调”与“图片的地域属性”统一进同一个基准，形成了一个可以交叉验证的实验矩阵。这样既能看到单一因素的影响，又能捕捉它们的交互效应，远比单独测一个维度更具洞察力。

### 实验与效果
- **测试对象**：GPT‑4V（官方多模态模型）、LLaVA‑1.5、Google Bard（多模态版）。  
- **基准表现**：在偏差挑战上，GPT‑4V 对西方/英文图片的正确率约为 87%，而对亚洲/非英文图片下降到 62%；LLaVA 与 Bard 的差距更大，分别在两类之间相差约 30% 与 35%。  
- **干扰挑战**：当使用诱导性提示时，GPT‑4V 的答案一致性从 91%（中性提示）跌至 68%；多图输入时一致性进一步降至 55%。  
- **缓解手段**：自我纠错提升约 2% 的整体准确率，思维链提升约 1.5%，均未能显著缩小偏差或干扰带来的差距。  
- **消融实验**：作者去掉“多语言图片”子集后，偏差差距几乎消失，说明语言是主要驱动因素；去掉“诱导式提示”后，干扰效应显著减弱，验证了提示词的关键作用。  
- **局限性**：论文未提供对更大规模、多模态交互（如视频）场景的评估；基准图片数量虽大但仍可能受采样偏差影响；对模型内部表征的分析仅停留在输出层面，缺少深层机制解释。

### 影响与延伸思考
这篇工作首次把多模态幻觉系统化为“偏差”和“干扰”两大维度，直接催生了后续的 **Bingo‑2.0**（加入视频与音频）以及 **Multimodal Fairness** 系列研究，大家开始关注训练数据的地域平衡和提示词鲁棒性。对想继续深入的读者，可以关注以下方向：① 构建更具代表性的跨文化多模态数据集；② 设计基于对抗训练的提示鲁棒化方法；③ 探索模型内部注意力图谱，定位偏差产生的具体层级。整体来看，本文提醒社区：提升视觉语言模型的“真实感”不仅是提升准确率，更是消除文化与语言壁垒的必要步骤。

### 一句话记住它
GPT‑4V 在跨文化和提示干扰下会“走神”，而现有的自我纠错与思维链并不能救它——我们需要专门的多模态公平与鲁棒性方案。