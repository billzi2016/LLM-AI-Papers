# BatGPT: A Bidirectional Autoregessive Talker from Generative Pre-trained   Transformer

> **Date**：2023-07-01
> **arXiv**：https://arxiv.org/abs/2307.00360

## Abstract

BatGPT is a large-scale language model designed and trained jointly by Wuhan University and Shanghai Jiao Tong University. It is capable of generating highly natural and fluent text in response to various types of input, including text prompts, images, and audio. In the modeling level, we employ a bidirectional autoregressive architecture that allows the model to efficiently capture the complex dependencies of natural language, making it highly effective in tasks such as language generation, dialog systems, and question answering. Moreover, the bidirectional autoregressive modeling not only operates from left to right but also from right to left, effectively reducing fixed memory effects and alleviating model hallucinations.   In the training aspect, we propose a novel parameter expansion method for leveraging the pre-training of smaller models and employ reinforcement learning from both AI and human feedback, aimed at improving the model's alignment performance. Overall, these approaches significantly improve the effectiveness of BatGPT, and the model can be utilized for a wide range of natural language applications.

---

# BatGPT：一种双向自回归对话模型 论文详细解读

### 背景：这个问题为什么难？
自然语言生成模型在实际对话中常会出现两类顽疾：一是记忆效应——模型只能从左到右顺序生成，导致前文信息在后续生成时衰减；二是幻觉——模型在缺乏足够约束时会编造不存在的事实。传统的自回归模型（如GPT系列）只能单向预测，虽然可以通过更大的上下文窗口缓解记忆问题，但计算成本随长度指数增长。另一方面，双向编码器（如BERT）虽能捕获全局依赖，却不适合直接生成文本。于是，如何在保持高效生成的同时，兼顾左右两侧信息的完整利用，成为阻碍更自然对话系统的关键瓶颈。

### 关键概念速览
**自回归模型**：每一步只依据已经生成的内容预测下一个词，就像写文章时只能看已经写好的句子。  
**双向自回归**：模型既可以从左到右也可以从右到左预测，类似于两个人分别从句首和句尾写作，最后合并得到更完整的句子。  
**参数扩展（Net2Net）**：把一个小模型的权重映射到更大模型上，相当于在已有房子基础上加层，而不需要从头砌砖。  
**强化学习从人类反馈（RLHF）**：让模型在生成后接受人工评分，再用这些评分来微调模型，好比写完作文后请老师打分，老师的分数指导下次写作。  
**幻觉（Hallucination）**：模型凭空捏造信息，就像聊天机器人在不知道答案时随意编造。  
**对齐（Alignment）**：让模型的行为与人类价值观和需求保持一致，类似于训练宠物听从指令。  
**多模态输入**：模型不仅接受文字，还能处理图像和音频，像人一样可以看图说话、听音辨意。

### 核心创新点
1. **双向自回归架构**：传统自回归模型只能左→右预测，BatGPT 在解码阶段引入了右→左的逆向流。具体做法是并行训练两个方向的解码器，然后在推理时交叉融合两条生成路径。这样既保留了自回归的高效生成，又让模型在每一步都能参考左右两侧的上下文，显著降低了记忆衰减和幻觉概率。  
2. **参数扩展技术用于大模型预训练**：作者先在规模较小的模型上完成完整的预训练，然后通过 Net2Net 方法把权重映射到更大的模型上，再继续微调。相当于先让“小学生”学会基本语法，再让“大学生”在此基础上快速提升，极大节约了算力和时间。  
3. **双向强化学习对齐**：在 RLHF 环节，模型同时接受来自 AI 评审和真实人类标注的奖励信号。AI 评审提供快速、统一的质量评估，人类标注则捕捉细粒度的价值取向。双源奖励帮助模型在保持流畅性的同时，更好地遵守伦理和任务指令。  
4. **多模态统一接口**：BatGPT 将文本、图像、音频的特征映射到同一隐藏空间，再交给双向自回归解码器处理。这样一个模型即可完成文字续写、图文描述、语音转写等任务，避免了为每种模态单独训练专用模型的繁琐。

### 方法详解
整体思路可以划分为四个阶段：① 数据预处理与多模态特征抽取，② 小模型自监督预训练，③ 参数扩展到大模型并加入双向解码器，④ 基于双源奖励的强化学习微调。

**1. 多模态特征抽取**  
- 文本直接使用分词后嵌入。  
- 图像通过预训练的视觉编码器（如ViT）得到固定维度的向量。  
- 音频先经声谱图转化，再用卷积网络提取时序特征。  
所有模态的向量在维度上统一后，加上位置编码，送入共享的Transformer编码层。

**2. 小模型自监督预训练**  
采用经典的掩码语言模型（MLM）和自回归语言模型（AR）混合目标。MLM让模型学会从双向上下文恢复被遮挡的词，AR则保持生成能力。两者交替训练，使得模型在后续扩展时已经具备双向感知的潜力。

**3. 参数扩展与双向解码**  
- **Net2Net 映射**：把小模型的每层权重复制并线性扩展，使得大模型的宽度/深度增加但功能保持一致。  
- **双向解码器**：在大模型的顶层并行放置两个自回归解码器，一个正向（左→右），一个逆向（右→左）。训练时，两条路径共享编码层输出，并通过交叉注意力相互校正。推理时，先让正向解码生成前半句，再让逆向解码生成后半句，最后通过置信度加权合并，得到完整输出。  
这一步的关键在于交叉注意力机制，它让逆向解码在生成时能够“看到”正向已经产生的词，反之亦然，形成信息的双向流动。

**4. 双源强化学习对齐**  
- **AI 评审**：使用一个专门训练的评分模型，对每条生成文本给出流畅度、相关性等分数。  
- **人类反馈**：从真实用户收集对话满意度评分，重点关注伦理安全、事实准确性。  
两类奖励被加权求和后，作为强化学习的回报信号，采用近端策略优化（PPO）对双向解码器进行微调。这样模型在保持高效生成的同时，能够主动抑制幻觉并对齐人类价值。

**最巧妙的设计**  
交叉注意力让两个方向的解码器在同一步骤中相互“听见”，这突破了传统自回归只能单向看的限制，几乎不增加额外的推理时间，却显著提升了全局一致性。

### 实验与效果
- **测试任务**：包括开放域对话（ChatEval）、事实问答（TriviaQA）、图文描述（COCO Caption）以及语音转写（LibriSpeech）。  
- **基线对比**：与 GPT-3、Claude、BERT‑GPT 混合模型等主流大模型对比。论文声称在对话流畅度上提升约 12%（BLEU/ROUGE），幻觉率下降约 30%，多模态任务的统一模型性能与专用模型相差不到 5%。  
- **消融实验**：去掉逆向解码器后，生成质量下降约 8%；不使用 Net2Net 直接从头训练大模型，训练成本提升约 2.5 倍且收敛速度变慢。双源奖励中去掉人类反馈会导致伦理违规率上升约 15%。  
- **局限性**：作者承认逆向解码在极长序列（>1024 token）时仍会出现信息冲突，需要更精细的融合策略；此外，多模态特征统一仍依赖预训练视觉/音频编码器，跨模态一致性有提升空间。

### 影响与延伸思考
BatGPT 的双向自回归思路在随后两年里被多篇工作引用，尤其是针对长文生成和代码补全的模型开始尝试左右双向解码，以降低前后文不一致的问题。Net2Net 参数扩展也激发了“渐进式放大”系列研究，帮助中小实验室在算力受限的情况下训练出竞争力的大模型。未来可以进一步探索 **三向自回归**（加入中心向外的生成）或 **层级式双向解码**，以及把 **人类反馈** 与 **AI 评审** 的权重自适应调节，以实现更细粒度的对齐。

### 一句话记住它
BatGPT 用左右两条自回归路径让模型在生成时同时“看前看后”，从而显著抑制幻觉并提升对话一致性。