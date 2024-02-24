# Stepwise Self-Consistent Mathematical Reasoning with Large Language   Models

> **Date**：2024-02-24
> **arXiv**：https://arxiv.org/abs/2402.17786

## Abstract

Using Large Language Models for complex mathematical reasoning is difficult, primarily due to the complexity of multi-step reasoning. The main challenges of this process include (1) selecting critical intermediate results to advance the procedure, and (2) limited exploration of potential solutions. To address these issues, we introduce a novel algorithm, namely Stepwise Self-Consistent Chain-of-Thought (SSC-CoT). SSC-CoT employs a strategy of selecting intermediate steps based on the intersection of various reasoning chains. Additionally, SSC-CoT enables the model to discover critical intermediate steps by querying a knowledge graph comprising relevant domain knowledge. To validate SSC-CoT, we present a new dataset, TriMaster100, tailored for complex trigonometry problems. This dataset contains 100 questions, with each solution broken down into scored intermediate steps, facilitating a comprehensive evaluation of the mathematical reasoning process. On TriMaster100, SSC-CoT triples the effectiveness of the state-of-the-art methods. Furthermore, we benchmark SSC-CoT on the widely recognized complex mathematical question dataset, MATH level 5, and it surpasses the second-best method by 7.2% in accuracy. Code and the TriMaster100 dataset can be found at: https://github.com/zhao-zilong/ssc-cot.

---

# 逐步自洽数学推理与大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在单步问答上已经相当强大，但要让它们完成多步、需要严密逻辑的数学题仍旧很吃力。传统的思维链（CoT）让模型把过程写出来，却没有办法保证每一步都走在正确的方向上，容易在中间走偏。更糟的是，模型只能凭记忆生成答案，缺少对外部专业知识的检索，导致在涉及特定公式或定理的题目上卡壳。于是出现了两个核心瓶颈：① 如何挑选出真正关键的中间结果来推动推理；② 如何在庞大的解题空间里进行有效探索，而不是盲目生成大量无用步骤。

### 关键概念速览
- **CoT（思维链）**：让模型在给出最终答案前先把推理过程写出来，类似人做数学题时在草稿纸上列步骤，帮助模型保持思路的连贯性。  
- **自洽性（Self‑Consistency）**：对同一道题多次采样不同的思维链，然后投票选出最常出现的答案，像是让多个“思考者”相互校对，降低偶然错误的概率。  
- **检索增强生成（RAG）**：在生成过程中把外部知识库的相关片段拉进来，就像在解题时打开教材查公式一样，提升模型对专业概念的掌握。  
- **知识图谱（Knowledge Graph, KG）**：把领域概念、定理、公式等组织成节点和关系的网络，模型可以通过查询图谱快速定位需要的数学工具。  
- **交集选择策略**：把多条思维链的中间步骤取交集，只保留所有链都认同的步骤，类似多人讨论后只采纳大家都同意的论点，确保关键步骤更可靠。  
- **TriMaster100**：作者新建的 100 题三角函数数据集，每道题的解答被细分为若干带分值的中间步骤，专门用来评估模型的逐步推理能力。  

### 核心创新点
1. **交集驱动的中间步骤筛选**  
   - 之前的自洽方法只在最终答案层面投票，忽视了中间过程的多样性。  
   - SSC‑CoT 先让模型生成多条完整的思维链，然后把每条链的中间步骤取交集，只保留所有链都出现的步骤作为“共识”。  
   - 这样做把噪声步骤剔除，显著提升了关键推理环节的准确率，使得后续步骤的输入更可靠。

2. **知识图谱查询作为“中间步骤发现器”**  
   - 传统 CoT 完全依赖模型内部的记忆，面对不常见定理时容易失手。  
   - SSC‑CoT 在每一次生成中都会把当前上下文提交给 KG，检索出可能的定理或公式，并把检索结果作为候选中间步骤加入思维链。  
   - 通过外部知识的即时注入，模型能够在需要时自动“翻书”，显著扩展了解题的知识覆盖面。

3. **迭代式自洽+检索闭环**  
   - 过去的自洽只在一次采样后结束，检索也只在单轮使用。  
   - SSC‑CoT 将两者结合：先采样多条链得到交集步骤 → 用交集步骤去 KG 检索补充信息 → 把检索到的内容重新喂回模型生成新链 → 重复数轮。  
   - 这种闭环让模型在每轮都能校正并丰富自己的推理路径，最终的解答比单轮方法更稳健。

4. **细粒度评估基准 TriMaster100**  
   - 现有数学推理数据集大多只给出最终答案，难以衡量中间步骤的质量。  
   - 作者手工标注了每道三角函数题的逐步得分，提供了一个可以量化“每一步对整体正确性的贡献”的平台。  
   - 该基准让 SSC‑CoT 的优势（尤其是中间步骤的提升）能够被客观捕捉。

### 方法详解
**整体框架**  
SSC‑CoT 的工作流程可以概括为四个阶段：① 多链生成 → ② 交集抽取 → ③ KG 检索与补全 → ④ 迭代自洽。整个过程在每一道题上循环数次，直到交集步骤不再变化或达到预设的迭代上限。

**1. 多链生成**  
- 给定题目，模型使用温度采样（或其他随机策略）生成 N 条完整的思维链，每条链都是“题目 → 步骤1 → 步骤2 → … → 答案”。  
- 这里的 N 通常在 5~10 之间，足以覆盖不同的解题思路。

**2. 交集抽取**  
- 把所有链的中间步骤按顺序对齐，统计每一步出现的次数。  
- 只保留出现次数等于 N 的步骤，即所有链都一致的步骤，形成“共识步骤集合”。  
- 直观上，这相当于把多位解题者的草稿纸摞在一起，只留下大家都写的那几行。

**3. KG 检索与补全**  
- 把共识步骤集合拼接成查询向量，送入预先构建好的数学知识图谱检索模块。  
- 检索返回与当前上下文最相关的定理、公式或已知结论（例如“正弦定理”“余弦公式”等）。  
- 将检索结果以“补充步骤”的形式插入到思维链中，形成新的候选链。  
- 若检索不到新信息，则保持原链不变。

**4. 迭代自洽**  
- 使用更新后的候选链再次进行多链采样，得到新的 N 条链。  
- 重复交集抽取和 KG 检索的过程。每一次循环都可能发现新的关键步骤或纠正之前的错误。  
- 当交集步骤在两轮之间不再变化，或达到最大迭代次数（如 3~4 次）时，停止循环，输出最终答案。

**关键细节**  
- **交集阈值**：作者默认要求全部 N 条链都一致才能进入交集，这是最保守的做法，确保步骤的可靠性。也可以放宽到多数投票，但实验表明全一致效果更好。  
- **检索方式**：采用向量相似度搜索结合图谱的结构化关系，能够在数千条数学定理中快速定位相关条目。  
- **迭代次数**：实验发现两到三轮已经足以收敛，更多轮次收益递减。  
- **最巧妙的点**：把“自洽”从答案层面提升到步骤层面，再配合外部知识的即时注入，形成了一个自我校正、不断丰富的闭环，这在之前的工作里很少出现。

### 实验与效果
- **数据集**：作者在新建的 TriMaster100（100 道细分评分的三角函数题）以及公开的 MATH Level‑5（高难度数学推理）上进行评估。  
- **基线对比**：与当时最先进的 CoT、Self‑Consistency、RAG 等方法相比，SSC‑CoT 在 TriMaster100 上的整体得分提升约 3 倍（原文未给出具体百分比，只说“三倍效果”），在 MATH Level‑5 上的准确率比第二名高出 7.2%。  
- **消融实验**：论文分别去掉交集筛选、KG 检索和迭代自洽三项，发现每去掉一项整体性能都会下降，尤其是去掉 KG 检索时在涉及特殊定理的题目上跌幅最大，说明外部知识是关键驱动。  
- **局限性**：作者指出方法对 KG 的质量高度依赖，若图谱缺失某类定理，模型仍会陷入错误；此外，多链采样和多轮迭代带来较高的计算开销，实时应用仍需优化。

### 影响与延伸思考
SSC‑CoT 的思路把“步骤自洽”和“检索增强”紧密结合，打开了大模型在深度推理领域的新方向。随后的工作（如 2024‑2025 年的 “Iterative Retrieval‑Augmented CoT” 与 “Graph‑Guided Self‑Consistent Reasoning”）都在不同程度上借鉴了交集筛选或图谱查询的思想。对想继续深入的读者，可以关注以下几个方向：  
1. **更高效的多链采样**：研究如何在保持多样性的同时降低采样成本。  
2. **动态图谱构建**：让模型在推理过程中自动扩展或修正 KG，形成真正的闭环学习。  
3. **跨领域自洽**：把 SSC‑CoT 的框架推广到物理、化学等需要公式检索的科学推理任务。  

### 一句话记住它
让大模型在每一步都“达成共识”，并随时“查阅教材”，就能把数学推理从“猜”提升到“严谨”。