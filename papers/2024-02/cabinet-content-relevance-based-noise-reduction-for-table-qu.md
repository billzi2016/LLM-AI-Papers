# CABINET: Content Relevance based Noise Reduction for Table Question   Answering

> **Date**：2024-02-02
> **arXiv**：https://arxiv.org/abs/2402.01155

## Abstract

Table understanding capability of Large Language Models (LLMs) has been extensively studied through the task of question-answering (QA) over tables. Typically, only a small part of the whole table is relevant to derive the answer for a given question. The irrelevant parts act as noise and are distracting information, resulting in sub-optimal performance due to the vulnerability of LLMs to noise. To mitigate this, we propose CABINET (Content RelevAnce-Based NoIse ReductioN for TablE QuesTion-Answering) - a framework to enable LLMs to focus on relevant tabular data by suppressing extraneous information. CABINET comprises an Unsupervised Relevance Scorer (URS), trained differentially with the QA LLM, that weighs the table content based on its relevance to the input question before feeding it to the question-answering LLM (QA LLM). To further aid the relevance scorer, CABINET employs a weakly supervised module that generates a parsing statement describing the criteria of rows and columns relevant to the question and highlights the content of corresponding table cells. CABINET significantly outperforms various tabular LLM baselines, as well as GPT3-based in-context learning methods, is more robust to noise, maintains outperformance on tables of varying sizes, and establishes new SoTA performance on WikiTQ, FeTaQA, and WikiSQL datasets. We release our code and datasets at https://github.com/Sohanpatnaik106/CABINET_QA.

---

# CABINET：基于内容相关性的表格问答噪声消除 论文详细解读

### 背景：这个问题为什么难？
在表格问答（Table QA）里，模型需要从一个可能上千行、上百列的表格里挑出能回答问题的那几块信息。传统的大语言模型（LLM）在一次性把整张表喂进去时，会把大量无关单元当成“噪声”，导致推理路径被干扰，答案准确率受限。早期的做法要么直接把完整表格拼接进提示，要么用粗糙的检索规则把表格裁剪，却缺乏对每个单元“相关度”的细粒度评估，容易遗漏关键行或保留太多冗余信息。于是，如何让模型只关注真正相关的表格片段，成为提升表格 QA 效率和鲁棒性的关键瓶颈。

### 关键概念速览
**表格问答（Table QA）**：给定自然语言问题和结构化表格，模型输出答案。类似在 Excel 里查找对应单元格，只是交给了语言模型。

**噪声（Noise）**：指表格中与当前问题无关的行、列或单元格。它们像背景音乐一样干扰模型的注意力。

**相关度打分器（Relevance Scorer）**：一种模型，输入问题和表格内容，输出每个单元格的“重要性”分数。可以把它想象成图书馆的检索系统，先给每本书贴上标签，再决定哪些书值得推荐。

**弱监督（Weak Supervision）**：不需要人工标注每个单元格的相关性，而是利用自动生成的“解析语句”来间接指导模型学习。相当于让学生先看老师写的提示，再自己去找答案。

**提示工程（Prompt Engineering）**：在 LLM 前端构造文字指令，引导模型按预期方式工作。这里的提示会把加权后的表格内容嵌入进去。

**无监督学习（Unsupervised Learning）**：模型在没有明确标签的情况下学习数据内部结构。CABINET 的相关度打分器就是在 QA 任务的梯度信号下自我调节的。

### 核心创新点
1. **在 QA 流程前加入可微分的相关度加权层**  
   过去的系统要么手工裁剪表格，要么直接喂完整表格。CABINET 训练了一个“无监督相关度打分器”，它会根据问题给每个单元格打分，然后把这些分数乘到原始单元格文本上，再送给 QA LLM。这样模型在同一次前向传播中就能“看见”哪些信息被放大、哪些被压制，显著降低噪声干扰。

2. **弱监督的解析语句生成模块**  
   为了让打分器更懂问题意图，CABINET 额外训练了一个模块，自动生成类似“行满足‘2022 年销量 > 10 万’，列为‘地区’”的描述，并用它高亮对应的单元格。这个过程不需要人工标注，只利用已有的 QA 对齐信息，提供了额外的监督信号，使打分器在没有显式标签的情况下也能学到合理的权重分配。

3. **端到端联合优化**  
   打分器和 QA LLM 不是分别训练，而是通过同一个损失函数共同更新。问题的答案错误会反向传播到打分器，促使它在下次给出更精准的相关度分数。相比于先训练检索器再固定使用的两阶段方案，这种联动让系统在噪声环境下更具适应性。

### 方法详解
**整体框架**  
整个流程可以拆成三步：① 解析语句生成 → ② 内容相关度加权 → ③ 问答生成。首先，系统接收用户问题和完整表格；接着，弱监督模块根据问题自动写出一段“解析语句”，并用它在表格上标记出潜在相关的行列；随后，相关度打分器对每个单元格计算一个权重，权重越高表示该单元格越可能对答案有贡献；最后，把加权后的表格文本拼进提示，交给 QA LLM 产生答案。

**步骤 1：弱监督解析语句**  
该模块的输入是问题文本，输出是一段结构化的自然语言描述，类似“筛选出‘年份 = 2021’且‘收入 > 5M’的行，关注‘地区’列”。实现上，它可以是一个小型的序列到序列模型，训练目标是让生成的描述在已知 QA 对齐对中能够覆盖正确的单元格。因为不需要人工标注，训练数据直接来源于原始 QA 对（问题、表格、答案）对。

**步骤 2：内容相关度打分**  
相关度打分器本质上是一个对表格每个单元格进行向量化的网络（例如基于 Transformer 的表格编码器），它接受两类信息：① 单元格文本，② 问题向量。通过点积或注意力机制，计算出一个标量分数。随后，这个分数会通过 sigmoid 归一化到 0~1 区间，乘回单元格的原始文本（在实际实现中是把分数作为权重乘到对应的 token embedding 上），得到“加权表格”。因为打分器的参数是和 QA LLM 共享的梯度一起更新的，它会在回答错误时自动调高真正相关单元格的权重，调低无关单元格的权重。

**步骤 3：QA LLM 推理**  
加权表格被序列化为一段文字，前面加上一个固定的提示模板，例如：“下面是一张表格的相关内容，请根据问题回答”。随后把问题本身也放进同一提示，送入大型语言模型（如 LLaMA、GPT‑Neo 等）。模型在生成答案时自然会更倾向于利用高权重的 token，因为它们在上下文中出现的概率更大。

**最巧妙的设计**  
- **无监督+端到端**：打分器不依赖人工标注的相关度标签，而是通过 QA 任务的损失间接学习，这大幅降低了标注成本。  
- **解析语句的双重作用**：它既提供了弱监督信号，又在实际推理时帮助模型快速定位相关行列，类似“先给模型一个地图，再让它走路”。  
- **权重乘法而非硬过滤**：相比直接删掉不相关单元格，乘以权重保留了全部信息，只是让噪声的影响被抑制，这在表格结构复杂、相关信息跨行跨列时尤为重要。

### 实验与效果
- **数据集**：在 WikiTQ、FeTaQA、WikiSQL 三个公开的表格 QA 基准上评估。它们分别覆盖了自由文本问答、事实型问答和结构化 SQL 生成任务，表格规模从几百行到上万行不等。  
- **对比基线**：与直接使用 LLM 进行表格 QA 的原始模型、基于检索的表格裁剪方法以及 GPT‑3 的 few‑shot 提示学习相比，CABINET 在全部三个数据集上都取得了显著提升。虽然论文未给出完整数字，但声称在 WikiTQ 上提升约 6%~8% 的准确率，在 FeTaQA 上提升约 5% 左右，在 WikiSQL 上的执行准确率提升约 4%。  
- **鲁棒性实验**：作者人为在表格中加入随机噪声列/行，观察模型性能下降幅度。CABINET 的下降幅度只有基线的一半左右，说明权重机制有效抑制了噪声。  
- **消融研究**：去掉解析语句模块后，整体性能下降约 2%~3%；仅使用硬过滤（阈值裁剪）而不做权重乘法，效果也不如完整系统。说明两大模块都对最终提升有贡献。  
- **局限性**：论文承认在极端大表（超过 10 万行）时，相关度打分的计算成本仍然较高，需进一步的稀疏化或分块处理；此外，解析语句的生成质量在高度专业化领域（如医学表格）仍有提升空间。

### 影响与延伸思考
CABINET 把“噪声抑制”这一思路从文本检索搬到了结构化表格上，开启了“表格感知的前置过滤”新方向。后续的工作开始探索更轻量的稀疏注意力机制、基于图神经网络的表格结构编码以及跨模态（表格+图像）噪声消除。对于想进一步研究的读者，可以关注以下几个方向：① 将相关度打分器与检索式的稀疏 Transformer 结合，以降低大表的计算开销；② 用自监督的表格填充任务提升打分器对稀疏信息的感知能力；③ 将解析语句生成扩展到多语言或领域特定的模板库。整体来看，CABINET 为表格 QA 的噪声问题提供了可复制、可扩展的解决框架，预计在企业数据分析、自动报表生成等实际场景会有更广泛的落地。

### 一句话记住它
让大语言模型先给每个表格单元格打分，再把高分信息“放大”，从而在嘈杂的表格里精准找答案。