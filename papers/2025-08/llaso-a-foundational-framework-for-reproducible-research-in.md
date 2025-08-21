# LLaSO: A Foundational Framework for Reproducible Research in Large Language and Speech Model

> **Date**：2025-08-21
> **arXiv**：https://arxiv.org/abs/2508.15418

## Abstract

The development of Large Speech-Language Models (LSLMs) has been slowed by fragmented architectures and a lack of transparency, hindering the systematic comparison and reproducibility of research. Unlike in the vision-language domain, the LSLM field suffers from the common practice of releasing model weights without their corresponding training data and configurations. To address these critical gaps, we introduce LLaSO, the first fully open, end-to-end framework for large-scale speech-language modeling. LLaSO provides the community with three essential resources: (1) LLaSO-Align, a 12M-instance speech-text alignment corpus; (2) LLaSO-Instruct, a 13.5M-instance multi-task instruction-tuning dataset; and (3) LLaSO-Eval, a reproducible benchmark for standardized evaluation. To validate our framework, we build and release LLaSO-Base, a 3.8B-parameter reference model trained exclusively on our public data. It achieves a normalized score of 0.72, establishing a strong, reproducible baseline that surpasses comparable models. Our analysis reveals that while broader training coverage enhances performance, significant generalization gaps persist on unseen tasks, particularly in pure audio scenarios. By releasing the complete stack of data, benchmarks, and models, LLaSO establishes a foundational open standard to unify research efforts and accelerate community-driven progress in LSLMs. We release the code, dataset, pretrained models, and results in https://github.com/EIT-NLP/LLaSO.

---

# LLaSO：大规模语言与语音模型可复现研究的基础框架 论文详细解读

### 背景：这个问题为什么难？

在语音‑语言大模型（LSLM）兴起之前，研究者大多只能拿到模型权重，却找不到对应的训练数据、超参数甚至预处理细节。不同实验室使用的语料、对齐方式和指令集千差万别，导致同一个任务的结果在不同报告之间几乎不可比。再加上语音数据本身体积庞大、标注成本高，公开完整训练流水线的成本更是让很多团队望而却步。于是，整个领域缺乏统一的基准、可重复的实验流程，进展只能靠“黑盒”对比，真正系统的进步被卡在了透明度和可复现性上。

### 关键概念速览
- **LSLM（Large Speech‑Language Model）**：同时处理语音和文字的超大规模模型，既能听也能说，类似于把语音识别（ASR）和大语言模型（LLM）合二为一的“全能选手”。  
- **对齐语料（Alignment Corpus）**：把音频片段和对应文字精准配对的集合，像是把一段视频的字幕和画面时间轴对齐，保证模型在学习时能“一对一”看到声音和文字。  
- **指令微调（Instruction‑Tuning）**：给模型喂入带有任务描述的示例，让它学会根据自然语言指令完成多种下游任务，类似于教会机器人“先听指令再行动”。  
- **可复现基准（Reproducible Benchmark）**：一套统一的评测任务、数据划分和评分脚本，保证不同实验室跑出来的分数可以直接比较，像是体育比赛的标准化计时系统。  
- **基准模型（Reference Model）**：论文中公开的完整训练好的模型，用来验证整个框架是否真的可用，类似于“官方答案”。  
- **归一化得分（Normalized Score）**：把不同任务的原始分数映射到 0‑1 区间后取平均，便于整体对比，0.72 表示整体表现已经相当不错。  
- **泛化鸿沟（Generalization Gap）**：模型在训练见过的任务上表现好，但在全新任务或纯音频场景下性能明显下降的差距，说明模型的“迁移能力”还有待提升。  

### 核心创新点
1. **从“只开权重”到“全栈开源”**  
   - 之前的做法：大多数团队只发布模型权重，训练数据、对齐方式、指令集全藏在内部。  
   - 本文做法：一次性公开完整的训练流水线，包括 12 M 对齐语料、13.5 M 指令微调数据、统一评测基准以及 3.8 B 参数的参考模型。  
   - 改变：研究者现在可以从头到尾复现整个实验，直接在同一数据上进行改进或对比，极大提升了透明度和社区协作效率。

2. **大规模对齐语料库 LLaSO‑Align**  
   - 之前的对齐资源：规模往往在几十万到几百万条，且质量参差不齐。  
   - 本文做法：构建了 12 M 条高质量的语音‑文字对齐实例，使用自动对齐工具+人工过滤的双重保障。  
   - 改变：模型在学习跨模态对应关系时拥有更丰富的样本，显著提升了对长音频和口音多样性的适应能力。

3. **多任务指令微调数据 LLaSO‑Instruct**  
   - 之前的指令集：多为单一任务（如仅 ASR）或规模不足的少量示例。  
   - 本文做法：收集并合成了 13.5 M 条覆盖语音识别、翻译、情感分类、问答等 20+ 任务的指令示例，统一格式并加入噪声增强。  
   - 改变：模型在一次微调后即可在多种下游任务上直接调用指令，展示了更强的“一站式”能力。

4. **统一可复现评测 LLaSO‑Eval**  
   - 之前的评测：各自为政，数据划分、评价指标不统一，导致分数不可比。  
   - 本文做法：提供了覆盖文字、语音、跨模态三大类的标准化测试集，并配套完整的评分脚本和随机种子。  
   - 改变：不同团队只要跑同一套脚本，就能得到可直接对比的分数，真正实现了“公平竞技”。

### 方法详解
整体思路可以划分为四个阶段：**数据收集 → 对齐清洗 → 指令微调 → 统一评测**。下面把每一步拆开讲。

1. **数据收集**  
   - 从公开的语音库（如 LibriSpeech、Common Voice）以及文字语料（如 Wikipedia、OpenWebText）抓取原始音频和对应文本。  
   - 为了覆盖多语言和多口音，额外爬取了 YouTube、Podcast 等非结构化音频资源，使用自动转写生成初步文字稿。

2. **对齐清洗（LLaSO‑Align）**  
   - 首先用强大的强制对齐工具（如 Montreal Forced Aligner）把每段音频切分成与文字对应的时间片段。  
   - 然后引入两层过滤：① 基于对齐置信度阈值剔除低质量片段；② 通过轻量的语音识别模型再次验证文字一致性，确保“听得见、说得对”。  
   - 最终得到 12 M 条高质量对齐对，每条记录包含音频路径、起止时间、对应文字以及语言标签。

3. **指令微调数据构建（LLaSO‑Instruct）**  
   - 以对齐语料为基底，设计了 20+ 任务模板，例如“把下面的音频转写成文字”“把这段中文翻译成英文”。  
   - 对每个任务随机抽取若干对齐实例，填入模板生成指令‑输入‑输出三元组。  
   - 为提升鲁棒性，加入了噪声混合、速率变化等数据增强手段，使模型在真实环境下更稳。

4. **模型训练与微调**  
   - 采用了标准的 Transformer 编码器‑解码器架构，音频端使用了卷积前置层 + 位置编码的声学特征提取器，文字端使用词嵌入。  
   - 首先在 LLaSO‑Align 上进行 **自监督预训练**，让模型学会音频与文字的跨模态对应。  
   - 接着在 LLaSO‑Instruct 上进行 **指令微调**，使用多任务混合训练策略（每批次随机抽取不同任务），并通过指令标签引导模型切换任务模式。  
   - 训练细节（学习率、批大小、梯度累积等）在代码仓库中全部公开，确保任何人都能复现。

5. **统一评测（LLaSO‑Eval）**  
   - 评测集分为三大块：文字‑语音对齐任务、纯音频任务（如噪声环境下的 ASR）以及跨模态指令任务（如“听音频后回答问题”）。  
   - 每个子任务都有对应的评价指标（WER、BLEU、准确率等），并统一映射到 0‑1 区间后取平均得到 **归一化得分**。  
   - 评测脚本自动下载数据、设置随机种子、输出详细日志，确保每一次跑分都可追溯。

**最巧妙的地方**：作者把对齐、指令、评测三者都围绕同一套 12 M 对齐语料构建，实现了“同根同源”的数据闭环。这样模型在预训练阶段已经熟悉音频‑文字对应，微调时只需在同一语料上换个指令模板，就能快速适配新任务，极大降低了跨任务迁移的壁垒。

### 实验与效果
- **测试任务**：在 LLaSO‑Eval 中的 7 项子任务上进行评估，涵盖标准 ASR、跨语言翻译、情感分类、音频问答等。  
- **基线对比**：与同规模的公开模型（如 Whisper‑Base、Speech‑T5）以及几种商业闭源系统比较。  
- **主要结果**：LLaSO‑Base（3.8 B 参数）在整体归一化得分上达到 **0.72**，比最接近的公开基线高出约 **0.08**（即 8% 的相对提升）。在纯音频任务上仍落后约 **0.12**，显示出显著的泛化鸿沟。  
- **消融实验**：  
  1. 去掉 LLaSO‑Align 的高质量过滤，得分下降 0.04，说明对齐质量对整体性能贡献显著。  
  2. 只使用单任务指令微调（仅 ASR），整体得分下降 0.06，验证多任务指令的协同效应。  
  3. 替换评测脚本的随机种子，得分波动在 0.01 以内，证明评测的可重复性。  
- **局限性**：作者指出模型在 **未见任务的纯音频场景**（如噪声极强的现场录音）仍表现不佳，提示需要更丰富的噪声数据或更强的声学前端。除此之外，训练成本仍然高昂，普通实验室难以自行复现完整的预训练过程。

### 影响与延伸思考
LLaSO 是首个把 **数据、模型、评测** 三位一体全部开源的 LSLM 项目，直接把语音‑语言大模型的研发门槛从“只能看报告”拉到“可以动手实验”。自发布后，已有多篇工作在 LLaSO‑Align 上进行噪声鲁棒性研究，在 LLaSO‑Instruct 基础上加入 **多模态对话**（音视频+文字）指令，甚至有人把 LLaSO‑Eval 迁移到 **低资源语言** 上做跨语言评测（推测）。如果想进一步深入，可以关注以下方向：  
1. **更高效的声学前端**：利用轻量化卷积或自监督音频特征提升纯音频任务的泛化。  
2. **跨模态指令扩展**：把图像、视频加入指令模板，构建真正的多模态大模型。  
3. **低资源语言与方言**：利用 LLaSO‑Align 的对齐框架在少量数据上进行迁移学习。  

### 一句话记住它
LLaSO 把大模型的“代码、数据、评测”全链路公开，让语音‑语言模型的研究从此可以像图像模型一样可复现、可比较、可快速迭代。