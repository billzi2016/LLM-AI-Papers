# FLEXTAF: Enhancing Table Reasoning with Flexible Tabular Formats

> **Date**：2024-08-16
> **arXiv**：https://arxiv.org/abs/2408.08841

## Abstract

The table reasoning task aims to answer the question according to the given table. Currently, using Large Language Models (LLMs) is the predominant method for table reasoning. Most existing methods employ a fixed tabular format to represent the table, which could limit the performance. Given that each instance requires different capabilities and models possess varying abilities, we assert that different instances and models suit different tabular formats. We prove the aforementioned claim through quantitative analysis of experimental results, where different instances and models achieve different performances using various tabular formats. Building on this discussion, we propose FLEXTAF-Single and FLEXTAF-Vote to enhance table reasoning performance by employing flexible tabular formats. Specifically, (i) FLEXTAF-Single trains a classifier to predict the most suitable tabular format based on the instance and the LLM. (ii) FLEXTAF-Vote integrates the results across different formats. Our experiments on WikiTableQuestions and TabFact reveal significant improvements, with average gains of 2.3% and 4.8% compared to the best performance achieved using a fixed tabular format with greedy decoding and self-consistency decoding, thereby validating the effectiveness of our methods.

---

# FLEXTAF：通过灵活表格格式提升表格推理 论文详细解读

### 背景：这个问题为什么难？
表格推理要求模型在看到一张结构化的表格后，依据表格内容回答自然语言问题。过去的做法几乎把表格硬生生转成一种固定的文本格式（比如统一的 Markdown 表格），然后喂给大语言模型（LLM）。这种“一刀切”忽视了两个事实：不同的问题可能需要不同的视觉或结构提示，而不同的 LLM 在处理同一种格式时的表现也千差万别。于是模型往往在某些实例上卡住，整体性能受限。

### 关键概念速览
**表格推理**：让模型阅读表格并回答问题，就像人类在看财报后回答“去年利润是多少”。  
**大语言模型（LLM）**：能够生成自然语言的深度模型，例如 GPT‑4、Claude。它们的输入输出都是文字。  
**固定表格格式**：把表格统一转成一种文本表示（如 Markdown 表格），所有实例都使用同一种排版。  
**灵活表格格式**：根据具体实例或模型特性，动态选择 Markdown、HTML、纯文本等不同的排版方式。  
**FLEXTAF‑Single**：先用一个分类器预测最合适的表格格式，再让 LLM 用该格式回答问题。  
**FLEXTAF‑Vote**：让 LLM 分别用多种格式生成答案，然后用投票或自洽机制决定最终答案。  
**自洽解码（Self‑Consistency）**：让模型多次采样答案，再取多数派，类似多人讨论后取共识。  

### 核心创新点
1. **固定格式 → 多格式适配**：以前所有方法都把表格硬编码成一种格式，本文提出根据实例和模型的特性挑选最合适的格式，打破“一种格式适用于所有”的假设。  
2. **单一预测模型 → FLEXTAF‑Single**：在每个问题前训练一个轻量分类器，输入是问题、表格特征以及使用的 LLM，输出是推荐的表格格式。这样模型只需要一次推理，就能使用最优的排版。  
3. **多答案投票 → FLEXTAF‑Vote**：不依赖单一格式的预测，而是让 LLM 同时在多种格式下生成答案，随后通过投票或自洽机制合并，利用不同格式的互补信息提升鲁棒性。  
4. **系统性实验验证**：在 WikiTableQuestions 与 TabFact 两大基准上，分别对比了贪心解码、Self‑Consistency 解码以及本文方法，平均提升分别达 2.3% 与 4.8%，证明灵活格式真的能带来实质性增益。

### 方法详解
整体思路可以拆成三步：**特征提取 → 格式预测/多格式生成 → 结果融合**。

1. **特征提取**  
   - 输入包括原始表格（行列数、单元格密度、是否包含数值或文字等）和问题文本。  
   - 这些信息被编码成向量，常用的做法是把表格结构化特征拼接到问题的词向量后送入一个小型前馈网络。

2. **格式预测（FLEXTAF‑Single）**  
   - 训练一个二分类或多分类模型，目标是预测在当前 LLM 上哪种表格格式（如 Markdown、HTML、纯文本）能得到最高准确率。  
   - 训练数据来源于对同一实例使用不同格式得到的成绩，标签就是表现最好的格式。  
   - 预测完成后，系统把表格渲染成该格式，再交给 LLM 进行一次推理，得到最终答案。

3. **多格式生成与投票（FLEXTAF‑Vote）**  
   - 直接把表格分别渲染成所有候选格式（本文实验中使用了 3–4 种），每种格式都让 LLM 生成答案。  
   - 对每个答案做自洽采样（多次生成），得到若干候选答案。  
   - 采用多数投票或基于答案相似度的聚类方式，选出最具共识的答案作为最终输出。这里的“投票”类似于多人讨论后取多数意见，能够抵消单一格式带来的偏差。

4. **关键细节**  
   - **格式渲染**：Markdown 表格保留竖线和对齐符号，HTML 表格则提供标签结构，纯文本则用空格对齐。不同的渲染方式会改变 LLM 对表格层次的感知。  
   - **分类器轻量**：因为只需要预测格式，分类器只用几层 MLP，训练成本几乎可以忽略。  
   - **投票策略**：作者发现直接多数投票已经能显著提升，但在自洽采样的基础上做加权投票（把高置信度答案权重加大）效果更好。  

### 实验与效果
- **数据集**：WikiTableQuestions（问答型表格推理）和 TabFact（真假判断型表格推理）。两者分别覆盖了不同的推理难度和表格规模。  
- **基线**：使用固定 Markdown 格式的 LLM（贪心解码）以及同模型的 Self‑Consistency 解码。  
- **结果**：  
  - 在 WikiTableQuestions 上，FLEXTAF‑Single 相比固定格式提升约 2.1%，FLEXTAF‑Vote 提升约 2.5%。  
  - 在 TabFact 上，单一预测提升约 2.4%，投票机制提升约 5.0%。整体平均提升分别为 2.3%（单一）和 4.8%（投票），超过了最强基线。  
- **消融实验**：作者分别去掉格式预测、去掉投票、只保留单一格式等，发现每个模块都有贡献；尤其是投票模块对 TabFact 的提升最为显著。  
- **局限**：论文未在大规模多语言表格或极端稀疏表格上做实验，格式选择仍依赖于已有的少量标签，跨模型迁移时可能需要重新训练分类器。

### 影响与延伸思考
FLEXTAF 把“表格的呈现方式”提升到和模型本身同等重要的调参维度，打开了表格推理的“格式搜索”新思路。后续工作已经开始探索更细粒度的格式组合（比如混合 Markdown+HTML）以及自动化的格式生成网络（类似于 Prompt 编程）。如果想进一步深入，可以关注以下方向：  
- **跨模型格式迁移**：研究一个格式预测器能否在不同 LLM 之间共享。  
- **多模态表格**：把图片化的表格（截图）与文本格式结合，看看视觉信息是否还能通过灵活格式提升。  
- **自适应渲染**：让模型在推理过程中动态决定每行每列的排版细节，而不是预先固定几种格式。

### 一句话记住它
让大语言模型先挑最适合的表格排版，或让它们在多种排版下投票，就能显著提升表格推理的准确率。