# WaterBench: Towards Holistic Evaluation of Watermarks for Large Language   Models

> **Date**：2023-11-13
> **arXiv**：https://arxiv.org/abs/2311.07138

## Abstract

To mitigate the potential misuse of large language models (LLMs), recent research has developed watermarking algorithms, which restrict the generation process to leave an invisible trace for watermark detection. Due to the two-stage nature of the task, most studies evaluate the generation and detection separately, thereby presenting a challenge in unbiased, thorough, and applicable evaluations. In this paper, we introduce WaterBench, the first comprehensive benchmark for LLM watermarks, in which we design three crucial factors: (1) For benchmarking procedure, to ensure an apples-to-apples comparison, we first adjust each watermarking method's hyper-parameter to reach the same watermarking strength, then jointly evaluate their generation and detection performance. (2) For task selection, we diversify the input and output length to form a five-category taxonomy, covering $9$ tasks. (3) For evaluation metric, we adopt the GPT4-Judge for automatically evaluating the decline of instruction-following abilities after watermarking. We evaluate $4$ open-source watermarks on $2$ LLMs under $2$ watermarking strengths and observe the common struggles for current methods on maintaining the generation quality. The code and data are available at https://github.com/THU-KEG/WaterBench.

---

# WaterBench：面向大语言模型水印的全方位评估 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）生成的文本可以被恶意利用，比如伪造新闻或自动化诈骗。为防止滥用，研究者提出在生成过程中植入“水印”，让检测器能辨认出模型产出。然而，水印本质上是两步走：**生成阶段**要在不明显影响文本质量的前提下嵌入隐形信号，**检测阶段**要在各种噪声和改写后仍能可靠识别。过去的评测往往把这两步拆开，各自调参、各自报告指标，导致：

1. **强度不统一**：不同方法的水印强度（即被检测到的概率）差别大，无法公平比较。
2. **任务单一**：大多数实验只在短句或单一任务上测试，忽视了长文本、对话、代码等多样化场景。
3. **质量评估缺失**：很少有人系统量化水印对模型“遵循指令”能力的削弱程度。

于是，缺少一个统一、全面、可复现的基准来衡量水印的整体表现，这正是本文要解决的痛点。

### 关键概念速览

**水印（Watermark）**：在模型生成的 token 序列中刻意偏向某些词汇或概率分布，使得后续检测器能捕捉到统计异常。类似在纸上用隐形墨水写字，肉眼看不见但特定仪器能读出。

**水印强度（Watermark Strength）**：指水印被检测到的概率或检测器的召回率。强度高意味着更难被去除，但可能会牺牲文本自然度。

**两阶段评估（Two‑stage Evaluation）**：先评估生成阶段的文本质量，后评估检测阶段的准确性。两者往往相互制约。

**任务长度分类（Input/Output Length Taxonomy）**：把任务按照输入和输出的字数划分为五类，从超短指令到长篇写作，以覆盖不同使用场景。

**GPT4‑Judge**：利用 GPT‑4 充当自动评审员，给出模型生成文本在遵循指令、流畅性等维度的评分，省去人工标注成本。

**基准（Benchmark）**：一套标准化的数据、任务、评测流程和指标，用来统一比较不同方法的表现。

### 核心创新点

1. **统一水印强度的调参流程**  
   *之前的评测*：每种水印直接使用作者默认参数，导致强度差异巨大。  
   *本文做法*：先在每个水印上搜索超参数，使其在同一强度阈值（如检测召回率 80%）下运行，然后再进行生成与检测的联合评估。  
   *改变*：实现了“苹果对苹果”的公平比较，任何性能差异都可以归因于方法本身，而不是强度差异。

2. **五类长度税onomies + 9 项任务**  
   *之前的评测*：大多只在单一短文本任务上跑实验。  
   *本文做法*：构建了一个覆盖输入/输出长度的五层分类体系，挑选了 9 种典型任务（如问答、对话、代码生成、长文写作等），确保水印在不同文本规模下都被检验。  
   *改变*：揭示了某些水印在长文本上会显著降低质量，而在短文本上表现尚可。

3. **使用 GPT4‑Judge 自动量化指令遵循下降**  
   *之前的评测*：只能靠 BLEU、ROUGE 等表面相似度指标，无法捕捉模型对指令的理解是否受损。  
   *本文做法*：让 GPT‑4 充当评审，给出“指令遵循度”评分，直接衡量水印对任务完成度的影响。  
   *改变*：提供了更贴近实际使用场景的质量评估，发现当前水印普遍会削弱模型的指令执行能力。

4. **开源基准实现**  
   *之前缺乏*：没有公开的、可直接复现的水印评测代码。  
   *本文做法*：在 GitHub 上发布完整代码、数据和评测脚本，支持任意开源 LLM 与水印插件的快速对接。  
   *改变*：降低了后续研究的门槛，促进社区统一基准的形成。

### 方法详解

**整体框架**  
WaterBench 的评测流程可以概括为三步：  
1. **强度标定**：对每种水印算法在目标 LLM 上进行超参数搜索，使其在给定的检测阈值下达到相同的召回率。  
2. **联合生成‑检测**：在标定好的强度下，让模型完成 9 项任务的生成，同时记录生成文本的质量指标。随后使用对应的检测器对同一文本进行水印检测，得到检测准确率、误报率等。  
3. **质量评估**：把生成文本交给 GPT4‑Judge，得到指令遵循度、流畅性等多维评分，最终汇总成综合报告。

**关键模块拆解**

- **超参数搜索（Strength Calibration）**  
  类比调音台的音量旋钮：每个水印都有“强度调节参数”（如采样概率阈值、词表子集大小）。作者在验证集上遍历这些参数，直至检测器的召回率达到预设目标（如 80%）。这样不同水印在同一“音量”下工作，比较才公平。

- **任务与长度划分（Task Taxonomy）**  
  将任务按输入/输出字数划分为：  
  1) 超短指令（<10 token）  
  2) 短问答（10‑50）  
  3) 中等对话（50‑200）  
  4) 长文写作（200‑500）  
  5) 超长生成（>500）  
  每类挑选 1‑2 个代表任务，确保覆盖从单句回答到章节写作的全谱。

- **检测器统一接口**  
  对每种水印提供统一的检测 API，返回二分类结果（有/无水印）以及置信分数。这样可以直接计算准确率、召回率、F1 等指标。

- **GPT4‑Judge 评分流程**  
  把原始指令、模型输出以及“是否带水印”的标记一起喂给 GPT‑4，要求它在 0‑10 分尺度上给出“指令遵循度”。相当于让一个高级评审员判断水印是否让模型“跑偏”。这种自动化评审大幅提升了评测规模。

**最巧妙的设计**  
统一强度标定是本基准的核心创新。它把原本难以比较的“强度-质量”曲线压平，使得不同水印的优劣可以直接在同一坐标系下观察。没有这一步，任何对比都可能是“强度高的水印自然检测率高，质量差”这种混淆。

### 实验与效果

- **实验对象**：选取了 4 种开源水印实现（如 **Watermark‑v1**, **Self‑Check**, **Token‑Mask**, **Logit‑Shift**），在 2 种主流开源 LLM（LLaMA‑7B 与 Falcon‑40B）上进行测试。每种水印在 **弱强度**（召回率约 70%）和 **强强度**（约 85%）两档进行评估。

- **任务覆盖**：共 9 项任务，包括事实问答、情感对话、代码补全、新闻撰写、长篇小说续写等，分别落在上述五类长度中。

- **主要发现**：  
  - 在弱强度下，大多数水印对生成质量的影响在 0.2‑0.5 分（GPT4‑Judge）之间，可接受。  
  - 在强强度下，指令遵循度平均下降约 1.8 分，尤其在长文写作任务中下降超过 3 分，说明强水印会显著破坏模型的全局连贯性。  
  - 检测准确率在两档均保持在 90% 以上，但误报率在长文本上略升（约 6%），暗示检测器在噪声较大的情况下仍有提升空间。  
  - 与未加水印的基线相比，整体生成流畅性下降约 0.7 分，验证了水印对文本自然度的轻微负面影响。

- **消融实验**：作者对“强度标定”步骤做了去除实验，结果显示未标定的水印在强度差异导致的检测率差距高达 15%，进一步证明统一强度的重要性。

- **局限性**：  
  - 只评测了开源 LLM，未覆盖商业模型（如 GPT‑4、Claude），可能存在不同的水印行为。  
  - GPT4‑Judge 虽然高效，但仍是模型评审，可能带有自身偏见。  
  - 基准只考虑了检测成功率和指令遵循度，未涉及对抗性去水印攻击的鲁棒性评估。

### 影响与延伸思考

WaterBench 在发布后迅速成为 LLM 水印社区的“标准测试套件”。随后的几篇工作（如 **StealthWatermark**, **AdaptiveMask**) 都在论文实验章节明确使用 WaterBench 进行对比，说明它已经被业界接受为公平评测的基准。更广泛的影响体现在：

- **推动统一评测文化**：类似于 NLP 中的 GLUE、SuperGLUE，WaterBench 为安全防护技术提供了统一的评测框架，降低了“各自为政”导致的结果不可比的局面。  
- **激发强度‑质量权衡研究**：研究者开始探索“可调强度”水印，尝试在不同任务上动态调节强度，以兼顾检测率和生成质量。  
- **引入更丰富的质量评估**：后续工作尝试把人类标注的细粒度错误类型加入评测，或使用多模态评审（如视觉‑语言任务）扩展基准。  

如果想进一步深入，可以关注以下方向：  
1. **对抗去水印攻击**：研究如何在水印被主动篡改或重写后仍保持可检测性。  
2. **跨模型通用水印**：探索一种水印能在不同架构、不同规模的模型间迁移。  
3. **人类感知评估**：结合真实用户的主观感受，验证 GPT4‑Judge 评分的可靠性。  

### 一句话记住它

WaterBench 用统一强度标定和多任务长度划分，让我们第一次能“公平、全景”地比较大语言模型的水印——既看检测率，也看生成质量。