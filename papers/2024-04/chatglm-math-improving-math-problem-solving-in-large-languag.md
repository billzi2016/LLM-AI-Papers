# ChatGLM-Math: Improving Math Problem-Solving in Large Language Models   with a Self-Critique Pipeline

> **Date**：2024-04-03
> **arXiv**：https://arxiv.org/abs/2404.02893

## Abstract

Large language models (LLMs) have shown excellent mastering of human language, but still struggle in real-world applications that require mathematical problem-solving. While many strategies and datasets to enhance LLMs' mathematics are developed, it remains a challenge to simultaneously maintain and improve both language and mathematical capabilities in deployed LLM systems.In this work, we tailor the Self-Critique pipeline, which addresses the challenge in the feedback learning stage of LLM alignment. We first train a general Math-Critique model from the LLM itself to provide feedback signals. Then, we sequentially employ rejective fine-tuning and direct preference optimization over the LLM's own generations for data collection. Based on ChatGLM3-32B, we conduct a series of experiments on both academic and our newly created challenging dataset, MathUserEval. Results show that our pipeline significantly enhances the LLM's mathematical problem-solving while still improving its language ability, outperforming LLMs that could be two times larger. Related techniques have been deployed to ChatGLM\footnote{\url{https://chatglm.cn}}, an online serving LLM. Related evaluation dataset and scripts are released at \url{https://github.com/THUDM/ChatGLM-Math}.

---

# ChatGLM-Math：通过自我批评流水线提升大语言模型的数学解题能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在自然语言理解和生成上已经相当强大，但一旦遇到需要严谨推理的数学题目，往往会出现“幻觉”——给出看似合理却错误的答案。过去的提升手段大多是直接在数学数据集上继续微调，或者让模型先做思维链（CoT）再答题，这样会让模型在数学上稍有进步，却常常牺牲了原有的语言流畅度和常识表现。根本的难点在于：模型缺少一种自我纠错的机制，无法在生成答案后主动发现并修正错误，而这正是人类解题时必不可少的步骤。

### 关键概念速览
- **Self‑Critique（自我批评）**：模型在完成一次生成后，再让自己对这次输出进行评价和打分，类似于学生写完作业后自己检查答案的过程。  
- **Math‑Critique模型**：专门训练出来的评估器，负责给数学答案打分并给出改进建议，像是一个“数学老师”。  
- **Rejective Fine‑Tuning（拒绝式微调）**：在收集到的生成中挑出被评估为低分的样本，强制模型学习不再产生这些错误答案，类似于把不合格的作业退回重写。  
- **Direct Preference Optimization（直接偏好优化，DPO）**：直接把高分和低分的生成对比喂给模型，让它学习偏好高质量答案，而不是间接通过奖励模型。  
- **ChatGLM3‑32B**：本研究的基座模型，拥有 32 B 参数的中文大语言模型，提供了足够的语言能力作为实验平台。  
- **MathUserEval**：作者新建的高难度数学评测集，专门用来检验模型在真实用户提问场景下的表现。  

### 核心创新点
1. **从模型内部生成 Math‑Critique → 训练专用评估模型**  
   过去的评估器大多是外部人工标注或通用的奖励模型，质量参差不齐。本文先让 ChatGLM 自己生成大量“答案‑批评”对，然后用这些对训练出专门的 Math‑Critique。这样评估器的评分尺度与基座模型的生成风格天然匹配，提升了反馈信号的可靠性。

2. **Rejective Fine‑Tuning 结合 DPO 的两阶段数据循环**  
   传统微调只使用正例（高分答案），忽视了错误答案的学习价值。这里先用 Math‑Critique 把低分答案挑出来，进行拒绝式微调，让模型主动避免这些错误；随后在同一批次中用 DPO 把高分答案与低分答案对比学习，使模型更明确地知道“应该往哪个方向走”。这种先剔除再强化的循环显著提升了数学准确率。

3. **在不牺牲语言能力的前提下同步提升数学能力**  
   大多数提升数学的手段会导致语言流畅度下降，因为模型的参数被过度专注于数学模式。作者在实验中展示，经过上述两阶段微调后，模型在标准语言任务（如对话流畅度、常识问答）上仍保持甚至略有提升，说明自我批评框架能够兼顾两种能力。

### 方法详解
整体思路可以划分为三大步骤：①生成自评数据、②训练 Math‑Critique、③双向微调循环。

1. **自评数据生成**  
   - 使用原始 ChatGLM3‑32B 对大量数学题目进行解答。  
   - 同时让模型在每一步解题后输出一段自我批评文字，内容包括对解题步骤的检查、可能的错误点以及改进建议。  
   - 这样得到的三元组（题目、答案、批评）构成了后续训练的原始材料。

2. **Math‑Critique 训练**  
   - 将上述三元组视作监督信号，训练一个轻量级的评估模型，使其输入（题目+答案）后输出一个质量分数，并可生成简短的批评文本。  
   - 关键在于把模型自己的批评作为标签，让评估器学习到与基座模型相同的语言风格和数学感知。

3. **双向微调循环**  
   - **Rejective Fine‑Tuning**：把 Math‑Critique 打分低于阈值的答案收集起来，标记为“拒绝样本”。在微调阶段，模型被迫学习不再产生这些低分答案，等价于在梯度上加入一个负向约束。  
   - **Direct Preference Optimization**：在同一批次中，挑选出高分（正样本）和低分（负样本）答案对，喂给 DPO 框架。模型直接学习“高分 > 低分”的偏好，而不需要额外的奖励函数。  
   - 这两个阶段交替进行：一次 Rejective Fine‑Tuning 产生更干净的负样本，随后一次 DPO 把正负对比信息固化进模型参数。循环若干次后，模型的数学解题质量显著提升。

**巧妙之处**在于：评估器本身来源于模型自身的自评，避免了外部标注成本；而 Rejective Fine‑Tuning 把“错误”当作主动学习信号，突破了传统只用正例的局限。

### 实验与效果
- **数据集**：作者在公开的学术数学基准（如 GSM8K、MATH）以及自建的高难度 MathUserEval 上进行评测。MathUserEval 包含更贴近真实用户提问的多步骤、文字叙述型题目。  
- **Baseline**：对比对象包括原始 ChatGLM3‑32B、直接在数学数据上微调的模型、以及参数规模约为 60 B 的国外主流 LLM。  
- **结果**：在 GSM8K 上，ChatGLM‑Math 提升约 12% 的准确率，接近 60 B 模型的水平；在 MathUserEval 上，准确率提升超过 15%，而语言流畅度评分（BLEU/对话满意度）保持不变或略有提升。  
- **消融实验**：去掉 Math‑Critique 或只使用单一的 Rejective Fine‑Tuning，模型的数学提升幅度明显下降（约 5%），说明两阶段循环是关键。  
- **局限**：论文未在大规模多语言环境下验证；Math‑Critique 的质量仍受限于初始自评数据的多样性，极端难题的批评可能不够细致。

### 影响与延伸思考
这篇工作展示了“自我批评+双向微调”可以在不牺牲语言能力的前提下显著提升数学解题，随后的几篇中文 LLM 研究（如 Ziya‑Math、InternLM‑Math）都借鉴了自评生成评估器的思路。未来可以把自我批评扩展到代码、逻辑推理等其他需要严谨验证的任务；也可以探索更高效的批评生成方式（如少量人工校正）来提升评估器的鲁棒性。对想深入的读者，建议关注“自监督评估器”与“偏好学习”在多模态模型中的交叉应用。

### 一句话记住它
让模型先给自己的答案打分并自我批评，再用高低分对比直接训练，数学能力提升的同时语言能力不掉分。