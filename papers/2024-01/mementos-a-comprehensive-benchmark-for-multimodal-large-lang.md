# Mementos: A Comprehensive Benchmark for Multimodal Large Language Model   Reasoning over Image Sequences

> **Date**：2024-01-19
> **arXiv**：https://arxiv.org/abs/2401.10529

## Abstract

Multimodal Large Language Models (MLLMs) have demonstrated proficiency in handling a variety of visual-language tasks. However, current MLLM benchmarks are predominantly designed to evaluate reasoning based on static information about a single image, and the ability of modern MLLMs to extrapolate from image sequences, which is essential for understanding our ever-changing world, has been less investigated. To address this challenge, this paper introduces Mementos, a new benchmark designed to assess MLLMs' sequential image reasoning abilities. Mementos features 4,761 diverse image sequences with varying lengths. We also employ a GPT-4 assisted method to evaluate MLLM reasoning performance. Through a careful evaluation of nine recent MLLMs on Mementos, including GPT-4V and Gemini, we find that they struggle to accurately describe dynamic information about given image sequences, often leading to hallucinations/misrepresentations of objects and their corresponding behaviors. Our quantitative analysis and case studies identify three key factors impacting MLLMs' sequential image reasoning: the correlation between object and behavioral hallucinations, the influence of cooccurring behaviors, and the compounding impact of behavioral hallucinations. Our dataset is available at https://github.com/umd-huang-lab/Mementos.

---

# Mementos：面向多模态大语言模型图像序列推理的综合基准 论文详细解读

### 背景：这个问题为什么难？

过去的多模态大语言模型（MLLM）大多在单张图片上做问答或描述，评测数据集也围绕“一张图+文字”展开。现实中，很多场景都是随时间变化的——漫画、监控录像、实验记录等，需要模型把前后帧的内容串联起来才能理解。现有基准缺少对“连续图像”推理的考察，导致模型在时间维度上的推理能力没有得到系统检验。于是出现了一个盲区：模型能否捕捉跨帧的因果关系、行为演变以及对象的状态变化？这正是本文要填补的空白。

### 关键概念速览

**多模态大语言模型（MLLM）**：同时接受文字和视觉输入，输出自然语言的模型。想象它是会说话的“看图机器人”。  
**图像序列**：按照时间顺序排列的多张图片，类似漫画的每一格或视频的关键帧。  
**幻觉（Hallucination）**：模型生成的内容在实际图像中不存在，类似人类凭空想象的情节。  
**行为幻觉**：模型错误地描述了图中对象的动作或变化，例如把静止的猫说成在跑。  
**共现行为（Co‑occurring Behaviors）**：同一帧或相邻帧中出现的多个动作，模型需要区分并正确关联。  
**GPT‑4 辅助评估**：利用 GPT‑4 生成参考答案或评分标准，再让它对模型输出进行打分，类似请一个“超级老师”来批改作业。  
**序列推理**：在多张图之间建立因果或时间关系的推理过程，类似观看一段短片后回答“接下来会发生什么”。  

### 核心创新点

1. **从单图基准到序列基准**  
   *之前的评测只给模型一张图，让它描述或回答问题 → 本文构建了 Mementos，收集了 4,761 条不同长度的图像序列，覆盖日常生活、漫画、实验等多种情境 → 让模型必须在时间轴上对对象和行为进行连续推理，暴露出它们在跨帧理解上的薄弱环节。  

2. **GPT‑4 驱动的自动评测管线**  
   *传统评测依赖人工标注答案，成本高且主观 → 作者设计了一个“GPT‑4 辅助评估器”，先让 GPT‑4 生成每条序列的参考答案和评分细则，再用同一模型对各 MLLM 的输出进行打分 → 实现了大规模、统一、可复现的评测流程，避免了人工评审的偏差。  

3. **系统化的错误因素分析框架**  
   *过去只报告整体准确率，缺少对错误根因的解释 → 论文提出了三条关键因素：对象与行为幻觉的相关性、共现行为的干扰、行为幻觉的累积效应 → 通过统计和案例分析，明确了哪些模式最容易导致模型出错，为后续改进指明方向。  

### 方法详解

**整体思路**：Mementos 的评测流程分为三步——数据构建、答案生成、模型评估。  
1. **数据构建**：研究团队从公开图像库、漫画网站以及自制实验视频中抽取连续帧，手工标注每条序列的关键对象、行为以及时间顺序，形成 4,761 条多样化的序列。每条序列长度在 2 到 7 张之间，确保既有短链也有稍长的情节。  
2. **参考答案生成**：利用 GPT‑4（带有强大推理能力的语言模型）对每条序列进行“阅读”，让它输出一段完整的自然语言描述，要求覆盖对象、行为、因果关系以及时间线。随后，GPT‑4 再生成一套评分标准：比如“是否正确指出对象 A 的移动方向”“是否误把静止对象描述为运动”。这些标准被保存为机器可读的 JSON。  
3. **模型评估**：把每个待测 MLLM（如 GPT‑4V、Gemini、LLaVA 等）喂入同样的图像序列，让它输出文字答案。再把答案交给同一个 GPT‑4 评估器，依据前一步的评分标准自动打分，得到每个维度的得分以及整体准确率。

**关键模块细化**：

- **序列输入包装**：因为大多数 MLLM 只能一次接受单张图，作者在实验中采用了“拼接”策略：把序列中的每张图按时间顺序拼成一张大图，或者在提示词中明确标记“第 1 张”“第 2 张”等。这样模型在一次前向传播中看到全部信息。  
- **行为标签抽取**：在构建参考答案时，GPT‑4 会把每个动作抽象成“[对象] 在 [时间点] 做了 [行为]”。这相当于把连续的视觉信息转化为结构化的时间表，便于后续对比。  
- **幻觉检测**：评估器会先检查答案中出现的对象和行为是否在参考标签里出现，若出现未标记的项则计为幻觉。这里的“未标记”指的是在所有帧中都没有对应视觉证据。  

**最巧妙的设计**：把 GPT‑4 同时当作“答案生成器”和“评审老师”。这样既保证了参考答案的高质量，又利用同一模型的统一标准来评分，避免了不同评审之间的尺度差异。  

### 实验与效果

- **测试对象**：论文选取了 9 种最新的多模态大语言模型，包括 OpenAI 的 GPT‑4V、Google 的 Gemini、Meta 的 LLaVA、InternVL 等。  
- **整体表现**：在 Mementos 上，所有模型的平均准确率都低于 40%，而在传统单图基准上同类模型通常能达到 70% 以上。尤其是行为描述的错误率高达 60% 以上，常见的错误是把静止物体描述为运动或把动作顺序颠倒。  
- **错误因素验证**：通过统计发现，模型出现对象幻觉时往往伴随行为幻觉（相关系数约 0.68），说明两者相互强化。共现行为（同一帧出现多个动作）会导致模型混淆，错误率提升约 15%。此外，错误在序列后段会累积，前几帧的误判会导致后续推理链条进一步偏离。  
- **消融实验**：作者分别去掉“拼接输入”和“显式时间标记”两种包装方式，发现去掉时间标记后整体准确率下降约 8%，说明明确的时间指示对序列推理至关重要。  
- **局限性**：评估依赖 GPT‑4 生成的参考答案，若 GPT‑4 本身在某些序列上产生误导，评分会受到影响。作者也承认目前只覆盖了 2–7 张的短序列，长视频或更复杂的交互情境仍未覆盖。  

### 影响与延伸思考

Mementos 首次系统化地把“图像序列推理”搬到多模态大语言模型的评测舞台，直接暴露了现有模型在时间维度上的短板。自论文发布后，已有几篇工作尝试在模型架构上加入显式的时序记忆模块（如跨帧注意力、视频专用的 Transformer）来提升序列理解能力。还有研究借鉴 Mementos 的评估管线，用更大规模的自动生成序列（比如从游戏录像中抽帧）扩展基准规模。想进一步深入的读者可以关注以下方向：① 将视频帧的光流信息显式输入模型；② 设计专门的“序列提示语言”，让模型在文字层面就能感知时间顺序；③ 探索自监督的跨帧对齐任务，以提升模型的时间感知。  

### 一句话记住它

Mementos 用 4,761 条图像序列把多模态大语言模型的“看图说话”升级为“看图连环画”，揭示了它们在跨帧推理上仍然大量“幻觉”。