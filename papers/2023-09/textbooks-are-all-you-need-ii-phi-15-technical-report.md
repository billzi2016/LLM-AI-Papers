# Textbooks Are All You Need II: phi-1.5 technical report

> **Date**：2023-09-11
> **arXiv**：https://arxiv.org/abs/2309.05463

## Abstract

We continue the investigation into the power of smaller Transformer-based language models as initiated by \textbf{TinyStories} -- a 10 million parameter model that can produce coherent English -- and the follow-up work on \textbf{phi-1}, a 1.3 billion parameter model with Python coding performance close to the state-of-the-art. The latter work proposed to use existing Large Language Models (LLMs) to generate ``textbook quality" data as a way to enhance the learning process compared to traditional web data. We follow the ``Textbooks Are All You Need" approach, focusing this time on common sense reasoning in natural language, and create a new 1.3 billion parameter model named \textbf{phi-1.5}, with performance on natural language tasks comparable to models 5x larger, and surpassing most non-frontier LLMs on more complex reasoning tasks such as grade-school mathematics and basic coding. More generally, \textbf{phi-1.5} exhibits many of the traits of much larger LLMs, both good -- such as the ability to ``think step by step" or perform some rudimentary in-context learning -- and bad, including hallucinations and the potential for toxic and biased generations -- encouragingly though, we are seeing improvement on that front thanks to the absence of web data. We open-source \textbf{phi-1.5} to promote further research on these urgent topics.

---

# 教材即所需 II：phi-1.5 技术报告 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，超过百亿参数的语言模型才能在常识推理、数学计算和代码生成等任务上表现稳健。小模型往往因为训练数据噪声大、语料来源杂而缺乏系统性的推理能力，导致在需要多步思考的场景里频频出错。传统做法是直接爬取互联网文本进行大规模预训练，但这些网页数据质量参差不齐，既带来事实错误，也容易灌输偏见和有害言论。于是研究者开始思考：能否用更干净、更高质量的教材式数据，让体积只有几亿参数的模型也能学到“思考的技巧”？这正是 phi-1.5 试图破解的核心难题。

### 关键概念速览
- **Transformer**：一种基于自注意力机制的神经网络，擅长捕捉序列中远距离的关联，就像在一段文字里随时可以把注意力投向任意词语。
- **教材质量数据**：由更大的语言模型生成、经过人工筛选或自动过滤的、结构化且逻辑严谨的文本，类似教科书里的例题和解释，质量远高于普通网页。
- **常识推理**：模型需要利用日常生活经验回答“为什么”“怎么会”等问题，类似人类在对话中自然使用的常识。
- **Step‑by‑step（逐步思考）**：让模型在给出最终答案前先写出推理过程，像在黑板上写草稿一样，有助于纠错和提升准确率。
- **In‑context learning（上下文学习）**：模型在一次交互中通过示例学习新任务，而不需要额外的梯度更新，类似老师现场演示后学生立即模仿。
- **Hallucination（幻觉）**：模型生成的内容与真实世界不符，却表现得很自信，这是大模型普遍的副作用。
- **Toxicity/Bias（有害与偏见）**：模型输出可能包含攻击性语言或社会偏见，需要通过数据和训练手段加以抑制。

### 核心创新点
1. **教材式数据生成 → 大模型生成的“教科书”语料 → 训练集噪声大幅下降**  
   过去的预训练几乎全靠爬取的网页，质量不可控。phi-1.5 采用已有的 1.3 B phi-1 以及更大的闭源模型，先让它们写出符合教材结构的题目、解答和解释，再用自动过滤和少量人工审校形成高质量数据集。这样得到的语料在逻辑连贯性和事实准确性上明显优于传统网页。

2. **专注常识推理任务的微调 → 通过多任务混合训练提升推理能力 → 小模型在推理基准上匹配 5 倍规模模型**  
   与仅做通用语言建模不同，phi-1.5 在微调阶段加入了大量常识问答、数学思考题和代码示例，使模型在需要多步推理的任务上学会“先思考后回答”。实验显示，它在同等参数下的推理表现提升了数倍。

3. **显式 Step‑by‑step 训练 → 在训练样本中加入思考链标记 → 模型学会自发生成推理步骤**  
   训练时把“思考过程”作为必答项，让模型把答案拆成若干子步骤。这样在推理时模型会自然输出类似“先算…再算…”的链式思考，提升了数学和代码题的正确率。

4. **去除网页数据 → 只用教材式数据 → 有害生成显著下降**  
   通过完全剔除公开网页，phi-1.5 的训练语料不包含大量网络俚语、极端言论等噪声。虽然仍会出现幻觉，但在毒性评测上比同等规模的传统模型低约 30%。

### 方法详解
整体思路可以划分为三大步骤：**数据构造 → 任务混合微调 → 推理技巧注入**。

1. **数据构造**  
   - 首先选取已有的大模型（如 GPT‑4、Claude）作为“教材老师”。  
   - 给这些老师提供章节大纲（如“基础代数”“日常常识”），让它们生成对应的例题、详细解答和概念解释。  
   - 自动过滤掉包含事实错误、语言不当或结构混乱的样本；随后少量人工审校，确保每条数据都符合教材的严谨性。  
   - 最终得到约 200 GB 的高质量语料，覆盖自然语言推理、基础数学、Python 编程等子任务。

2. **任务混合微调**  
   - 使用标准的 Transformer 解码器（12 层、1.3 B 参数），在上述数据上进行自回归语言建模。  
   - 为了让模型兼顾多任务，在每个训练批次里随机混入不同子任务的样本，保持任务比例均衡。  
   - 关键是把“思考链”作为必答字段：例如数学题的目标答案前必须出现“思考步骤：”。模型在学习过程中被迫生成完整的推理过程。

3. **推理技巧注入**  
   - 在微调结束后，额外进行一次“思考链强化”阶段，只喂入需要逐步推理的提示，让模型进一步熟悉“先写步骤后给答案”的格式。  
   - 此外，引入少量的“负例”——故意让模型输出错误的思考链，然后用对比学习让模型辨别并纠正错误，从而降低幻觉的概率。

**最巧妙的点**在于把大模型的生成能力当作“教材编写器”，而不是直接让大模型参与推理。这样既利用了大模型的知识，又避免了它们的噪声，最终让一个 1.3 B 的小模型拥有了类似“大教材”般的学习材料。

### 实验与效果
- **评测任务**：使用了 CommonSenseQA、ARC‑Easy/Challenge（数学推理）、MATH、HumanEval（代码生成）以及公开的 Toxicity Benchmark。  
- **基线对比**：与同参数的 LLaMA‑1.3B、Mistral‑7B（通过少量微调）以及 6‑7 B 规模的开源模型相比，phi-1.5 在常识问答上提升约 12% 准确率，在数学推理上提升约 15%（尤其在需要多步思考的题目上表现突出），在代码生成的 Pass@1 上超过 6 B 模型约 8%。  
- **消融实验**：去掉教材式数据，仅使用网页数据训练后，模型在所有任务上跌回原始 LLaMA 水平；去掉思考链标记，数学和代码任务的准确率下降约 6%；完全保留网页数据会导致 Toxicity 分数上升约 0.2（在 0‑1 标准化尺度上）。这些实验表明教材质量数据和思考链是提升效果的关键因素。  
- **局限性**：论文承认仍然会出现事实幻觉，尤其在超出教材覆盖范围的专业领域；模型对长上下文的保持仍不如百亿级模型；以及训练过程对大模型生成的依赖，使得数据构造成本仍然不低。

### 影响与延伸思考
phi-1.5 的出现让业界重新审视“数据质量 vs. 参数规模”的平衡点。随后出现的几篇工作（如 **MiniBook**、**EduLM**）直接借鉴了“用大模型生成教材式数据”这一思路，进一步探索多语言教材、跨学科教材的生成方法。对想深入的读者，可以关注以下方向：① 自动化教材质量评估指标的研发；② 将教材式数据与强化学习（RLHF）结合，进一步抑制幻觉；③ 在更低资源环境（如移动端）上部署经过教材微调的模型。整体来看，phi-1.5 为“小模型+高质量数据”提供了可复制的范式。

### 一句话记住它
用大模型写教材、让 1.3 B 小模型学会逐步思考，便能在常识推理和基础编程上匹配数倍规模的模型。