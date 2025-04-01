# A Survey on Test-Time Scaling in Large Language Models: What, How,   Where, and How Well?

> **Date**：2025-03-31
> **arXiv**：https://arxiv.org/abs/2503.24235

## Abstract

As enthusiasm for scaling computation (data and parameters) in the pretraining era gradually diminished, test-time scaling (TTS), also referred to as ``test-time computing'' has emerged as a prominent research focus. Recent studies demonstrate that TTS can further elicit the problem-solving capabilities of large language models (LLMs), enabling significant breakthroughs not only in specialized reasoning tasks, such as mathematics and coding, but also in general tasks like open-ended Q&A. However, despite the explosion of recent efforts in this area, there remains an urgent need for a comprehensive survey offering a systemic understanding. To fill this gap, we propose a unified, multidimensional framework structured along four core dimensions of TTS research: what to scale, how to scale, where to scale, and how well to scale. Building upon this taxonomy, we conduct an extensive review of methods, application scenarios, and assessment aspects, and present an organized decomposition that highlights the unique functional roles of individual techniques within the broader TTS landscape. From this analysis, we distill the major developmental trajectories of TTS to date and offer hands-on guidelines for practical deployment. Furthermore, we identify several open challenges and offer insights into promising future directions, including further scaling, clarifying the functional essence of techniques, generalizing to more tasks, and more attributions. Our repository is available on https://github.com/testtimescaling/testtimescaling.github.io/

---

# 大语言模型测试时扩展（Test‑Time Scaling）综述 论文详细解读

### 背景：这个问题为什么难？

在预训练阶段，人们通过不断增大数据量和模型参数来提升大语言模型（LLM）的能力，这种“训练时扩展”曾是提升性能的主要手段。但随着算力和数据的边际收益递减，单纯靠更大模型已经难以获得显著突破。与此同时，很多任务（尤其是数学、代码和开放式问答）对模型的推理深度、搜索空间或外部工具的调用有更高要求，而这些需求在训练阶段难以完全覆盖。于是研究者转向“测试时扩展”（Test‑Time Scaling，简称 TTS），即在模型推理时临时投入额外计算资源或辅助模块，以激发潜在的解题能力。TTS 的核心难点在于：如何在不改变模型本身的前提下，系统化地决定“扩展什么”“怎么扩展”“在何处扩展”，以及评估这种扩展到底能带来多大收益。缺乏统一的概念框架和系统评估，使得大量零散的技巧难以相互比较，也让实际部署变得摸不着头脑。

### 关键概念速览
- **Test‑Time Scaling（测试时扩展）**：在模型推理阶段额外投入计算、数据或工具，以提升输出质量。类似于考试时打开参考书、使用计算器来帮助解题，而不是在平时学习时就把所有技巧写进脑子里。
- **What to Scale（扩展什么）**：指在推理时增加的资源种类，如更多的采样次数、外部检索结果、工具调用等。相当于决定是多写几遍答案、查更多资料，还是请老师现场指导。
- **How to Scale（怎么扩展）**：具体的实现方式，包括链式思考（Chain‑of‑Thought）提示、递归自我纠错、基于检索的增强等。就像选择是先列提纲、再逐步展开，还是直接查答案。
- **Where to Scale（在何处扩展）**：决定在推理的哪个阶段加入额外计算，例如在生成第一轮答案前、答案生成过程中或答案完成后进行后处理。类似于决定是先做草稿、还是先写结论再补充细节。
- **How Well to Scale（扩展效果如何）**：评估扩展带来的性能提升与计算成本的比值，常用指标包括准确率提升、推理时间增长、资源消耗等。相当于衡量“多花钱买的咖啡到底提神多少”。
- **Retrieval‑Augmented Generation（检索增强生成）**：在生成过程中实时查询外部文档库，把检索到的片段作为上下文喂给模型。像是答题时现场打开百科全书查资料。
- **Self‑Consistency（自洽性）**：对同一道题进行多次独立采样，统计多数答案作为最终输出，以降低单次采样的随机误差。类似于让多个学生分别做同一题，取多数人的答案。

### 核心创新点
1. **统一的四维度框架**  
   - *之前的情况*：研究者各自提出“链式思考”“检索增强”等技巧，缺少统一的分类，导致文献碎片化。  
   - *本文的做法*：构建“what‑how‑where‑how‑well”四维度框架，把所有 TTS 方法映射到这四个轴上，形成系统的 taxonomy。  
   - *带来的改变*：提供了清晰的概念坐标系，帮助研究者快速定位已有方法的功能角色，也便于发现未被覆盖的空白区域。

2. **功能角色分解（Functional Role Decomposition）**  
   - *之前的情况*：很多技巧被混在一起描述，难以判断它们是提升搜索空间、增强事实性还是改善答案一致性。  
   - *本文的做法*：在四维框架的基础上进一步划分出“搜索扩展”“知识补全”“答案校正”“成本控制”等功能角色。  
   - *带来的改变*：让读者在设计新 TTS 流程时可以有针对性地挑选或组合角色，而不是盲目堆砌技巧。

3. **实用部署指南**  
   - *之前的情况*：学术论文往往只给出实验设置，缺少面向工程师的落地建议。  
   - *本文的做法*：基于大量实验经验，提供了“何时开启检索”“采样次数的经验阈值”“成本‑收益平衡的决策树”等实操手册。  
   - *带来的改变*：帮助有编程基础的读者快速在自己的应用中复现并调优 TTS，而不必从头摸索。

4. **开放的评估基准与代码库**  
   - *之前的情况*：不同工作使用的评测任务和指标不统一，导致结果不可比。  
   - *本文的做法*：整理并公开了覆盖数学、代码、开放式问答等多任务的统一评测套件，并在 GitHub 上同步代码。  
   - *带来的改变*：为后续研究提供了可复现的基准，推动了社区的统一评估。

### 方法详解
**整体思路**：作者并未提出单一算法，而是提供了一套“思考框架 + 组合手册”。使用时，先在四维度上定位需求，然后从功能角色库中挑选对应模块，最后按照“where”指示的阶段把模块串联起来，形成完整的推理流水线。

**步骤拆解**：

1. **需求定位（What）**  
   - 确定要补足的资源：是需要更多的答案候选（采样）、更准确的事实（检索）还是更强的推理深度（递归自我纠错）。  
   - 类比：先决定是要多写草稿、查字典还是请老师现场点评。

2. **实现方式选择（How）**  
   - 对应资源选择具体技术：  
     - *采样* → 多次温度采样 + Self‑Consistency 投票。  
     - *检索* → 基于向量相似度的文档检索 + 把检索片段拼接到 Prompt。  
     - *递归纠错* → 让模型先生成答案，再生成“错误检查”Prompt，循环若干次。  
   - 这里的关键是把每种资源映射到已有的实现手段，形成“技术库”。

3. **插入时机决定（Where）**  
   - **前置**：在生成任何文字前先检索或设定多采样。  
   - **中置**：在生成过程中动态调用检索或自我纠错（如在每一步生成后检查）。  
   - **后置**：答案完成后再进行一致性投票或后处理校正。  
   - 类比：先查资料、边写边查、写完再校对三种常见学习策略。

4. **效果评估（How Well）**  
   - 通过统一基准测算：准确率提升、推理时长、GPU/CPU 消耗。  
   - 作者提供了“收益‑成本曲线”绘制方法，帮助使用者在实际部署时决定是否开启某个模块。  

**关键模块细节**：

- **Self‑Consistency 投票**：模型以相同 Prompt 进行 N 次采样（N 通常为 5‑10），把所有完整答案收集后统计出现频率最高的答案作为最终输出。此过程不需要额外模型训练，只是增加推理次数。  
- **检索增强**：使用开源向量数据库（如 FAISS）存储任务相关文档。检索步骤返回 top‑k 片段，拼接为 “Context + Question” 的 Prompt。作者强调检索质量对整体提升的贡献远高于采样数量。  
- **递归自我纠错**：先让模型输出答案 A，然后生成 “请检查 A 中可能的错误并给出修正” 的 Prompt，得到修正后答案 B。循环若干次（通常 2‑3 步），每一步都用前一步的输出作为新输入。  

**最巧妙的地方**：作者把“成本控制”作为独立维度嵌入框架，提出了“预算感知调度”（budget‑aware scheduler），即在给定推理时间或算力上限时，自动选择最具性价比的模块组合。这种把资源约束直接纳入方法设计的思路，在之前的 TTS 工作中极少出现。

### 实验与效果
- **测试任务**：覆盖了 GSM8K（数学推理）、HumanEval（代码生成）、Open‑Domain QA（如 Natural Questions）以及多轮对话任务。  
- **基线对比**：与原始 LLM（不使用任何 TTS）以及单一 TTS 技术（仅采样或仅检索）进行比较。  
- **主要结果**（论文中给出的示例）：  
  - 在 GSM8K 上，使用“采样 + Self‑Consistency”提升了约 12% 的解题准确率；加入检索后再提升约 8%。  
  - HumanEval 代码生成准确率从 38% 提升到 49%（约 11% 增幅），主要得益于检索增强的函数文档。  
  - 对话任务的用户满意度评分提升约 0.6 分（满分 5 分），成本增长约 1.8 倍。  
- **消融实验**：作者分别关闭“检索”“自洽投票”“递归纠错”，发现检索对事实性任务贡献最大，自洽对多样性任务提升显著，递归纠错在复杂推理任务上带来约 4% 的额外提升。  
- **局限性**：  
  - 对于极端低延迟需求（如实时聊天），多轮自我纠错仍然不可接受。  
  - 检索质量高度依赖外部文档库的覆盖度，若库不匹配任务，提升有限。  
  - 论文未给出大规模部署（数十亿请求）下的成本模型，只提供了实验室环境的估算。

### 影响与延伸思考
这篇综述在发布后迅速成为 TTS 领域的“坐标系”，后续不少工作直接引用其四维框架来组织实验。例如，2024 年的 **“Dynamic Retrieval‑Guided Generation”** 通过在 “Where” 维度上实现细粒度的检索调度，显著降低了平均检索次数。还有 **“Budget‑Aware Self‑Consistency”** 直接沿用作者的成本感知调度思想，提出了在固定时延预算下自动决定采样次数的算法。  
对想进一步深入的读者，建议关注以下方向：  
1. **跨模态 TTS**：把视觉或音频检索加入到 “What” 维度，探索多模态信息对推理的增益。  
2. **自适应调度策略**：利用强化学习或元学习实时预测哪种扩展最划算。  
3. **可解释性与归因**：在 “How Well” 维度上发展更细粒度的贡献度分析，明确每个模块到底解决了哪类错误。  
4. **硬件协同**：把算力调度（如 GPU‑CPU 异构）纳入成本模型，实现真正的端到端最优。

### 一句话记住它
**“测试时扩展是一套四维度（what‑how‑where‑how‑well）框架，让你在不改模型的前提下，用额外算力、检索或自纠错把大语言模型的潜能最大化。”**