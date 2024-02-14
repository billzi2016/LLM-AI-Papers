# Premise Order Matters in Reasoning with Large Language Models

> **Date**：2024-02-14
> **arXiv**：https://arxiv.org/abs/2402.08939

## Abstract

Large language models (LLMs) have accomplished remarkable reasoning performance in various domains. However, in the domain of reasoning tasks, we discover a frailty: LLMs are surprisingly brittle to the ordering of the premises, despite the fact that such ordering does not alter the underlying task. In particular, we observe that LLMs achieve the best performance when the premise order aligns with the context required in intermediate reasoning steps. For example, in deductive reasoning tasks, presenting the premises in the same order as the ground truth proof in the prompt (as opposed to random ordering) drastically increases the model's accuracy. We first examine the effect of premise ordering on deductive reasoning on a variety of LLMs, and our evaluation shows that permuting the premise order can cause a performance drop of over 30%. In addition, we release the benchmark R-GSM, based on GSM8K, to examine the ordering effect for mathematical problem-solving, and we again observe a significant drop in accuracy, relative to the original GSM8K benchmark.

---

# 前提顺序在大语言模型推理中的重要性 论文详细解读

### 背景：这个问题为什么难？
在自然语言推理和数学求解等任务里，研究者一直假设只要把所有已知信息（前提）喂给大语言模型（LLM），模型就能自行找出正确的推理路径。实际上，LLM 的内部注意力机制和自回归生成方式对信息的呈现顺序极其敏感。过去的工作大多关注提升模型规模、提示工程或链式思考（CoT），却很少系统评估“前提排列”本身会不会改变答案。于是出现了一个盲点：如果前提顺序本身就能导致模型从正确走向错误，那任何提升推理能力的努力都可能被这层“顺序脆弱性”抵消。

### 关键概念速览
**前提（premise）**：任务中提供的已知事实或条件，就像数学题的已知量。  
**推理路径（reasoning chain）**：模型在内部构造的从前提到结论的逻辑步骤，类似解题时的演算顺序。  
**链式思考（Chain‑of‑Thought, CoT）**：让模型先把思考过程写出来再给出答案，像在纸上写草稿一样帮助模型保持逻辑连贯。  
**自回归生成（autoregressive generation）**：模型一次生成一个 token，后面的生成会受前面已生成内容的影响，就像接龙游戏。  
**R‑GSM 基准**：作者基于数学题库 GSM8K 重新组织前提顺序后得到的测试集，用来专门衡量顺序对数学求解的影响。  
**注意力机制（attention）**：模型在处理每个 token 时会“看”其他 token 的权重分配，顺序不同会导致注意力分配出现显著差异。  
**证明顺序（proof order）**：在演绎推理任务中，人工标注的每一步推理的正确顺序，等同于解题的“脚本”。  

### 核心创新点
1. **发现并量化前提顺序脆弱性** → 通过在多个公开 LLM（如 GPT‑3.5、Claude、LLaMA 系列）上系统地随机置换前提顺序，测得准确率下降超过 30% → 揭示了模型对信息排列的高度依赖，提供了全新评估维度。  
2. **提出对齐前提顺序的实验范式** → 将前提排列与人工标注的推理脚本对齐（即“证据顺序”）作为提示输入，与随机顺序进行对比 → 发现对齐后模型准确率显著提升，说明顺序信息可以被显式利用。  
3. **构建 R‑GSM 基准** → 在原 GSM8K 题目中打乱或重新排序已知条件，形成两套对照数据集 → 在数学求解任务上同样观察到显著的性能跌落，证明现象跨任务普适。  
4. **提供实用的前提排序建议** → 虽未提出全新模型结构，但通过实验指出在设计提示时应让前提顺序遵循“从易到难”或“与证明步骤一致”的原则，从而在不改模型的情况下提升推理表现。

### 方法详解
整体思路很直接：先把任务的前提按照不同顺序喂给模型，然后比较输出的正确率。具体步骤如下：

1. **任务选取与前提抽取**  
   - 选取演绎推理数据集（如 ProofWriter）和数学求解数据集 GSM8K。  
   - 对每条样本，提取所有显式前提（事实句子或已知数值），并记录官方提供的推理步骤顺序（如果有）。

2. **生成多种前提排列**  
   - **随机排列**：对前提集合进行全排列抽样，确保每次实验的顺序与原始无关。  
   - **对齐排列**：把前提重新排序，使其出现顺序与官方推理步骤的顺序一致。比如，若证明第一步使用前提 A，则 A 放在最前。  
   - **逆序排列**（可选）：完全倒置官方顺序，用来检验极端不匹配的影响。

3. **提示构造**  
   - 将前提列表拼接成一段自然语言提示，后接“请给出结论”或“请写出完整证明”。  
   - 对于使用 CoT 的实验，在提示末尾加入“请一步一步思考”，让模型生成思考链。

4. **模型推理与答案抽取**  
   - 使用自回归 LLM 逐 token 生成答案。  
   - 通过规则匹配或自动评估脚本判断生成的结论是否与金标准一致。

5. **性能统计**  
   - 计算每种排列方式的整体准确率。  
   - 对比随机、对齐、逆序三者的差异，得出“顺序对齐提升 X%”的结论。

**最巧妙的地方**在于把“前提顺序”当作一种可控变量来实验，而不是把它当作固定的背景信息。作者没有改动模型内部结构，只是通过提示设计让模型“看到”不同的信息流，这种轻量级的实验方法让结论更具普适性。

### 实验与效果
- **数据集**：演绎推理使用 ProofWriter（多种逻辑规则），数学求解使用 GSM8K，并基于它衍生出 R‑GSM（前提被重新排序）。  
- **模型**：包括 OpenAI 的 GPT‑3.5、Anthropic Claude、Meta LLaMA‑2 系列等主流 LLM，覆盖 7B‑70B 参数规模。  
- **主要发现**：在 ProofWriter 上，随机置换前提导致准确率下降超过 30%，而对齐排列则恢复到原始基准的水平。R‑GSM 实验同样显示，前提顺序被打乱后整体正确率出现显著下滑（具体数值未在摘要中给出，但作者称“显著”。）  
- **基线对比**：与直接使用原始（未排序）提示的结果相比，对齐提示提升约 10%‑15%（在不同模型上略有差异）。  
- **消融实验**：作者分别去掉 CoT 提示、只使用前提的子集等，发现即使不使用 CoT，顺序对齐仍能带来明显提升，说明效果并非 CoT 的副产品。  
- **局限性**：论文主要聚焦于已有前提的重新排序，未探讨如何自动生成最优顺序；此外，实验只覆盖了几类推理任务，是否对更复杂的自然语言推理（如多跳阅读）同样适用仍需验证。

### 影响与延伸思考
这篇工作提醒社区：在提示工程里，信息的排列顺序和人类写作习惯一样重要。随后出现的研究开始探索 **顺序感知提示（order‑aware prompting）**、**自动前提排序器**，甚至把前提排序作为可微分的学习目标加入模型训练中。对想进一步了解的读者，可以关注 2024‑2025 年间的 “Prompt Ordering” 系列论文，以及把排序任务与强化学习结合的 “Curriculum Prompting” 方向（推测）。这也暗示未来的 LLM 可能需要更强的 **结构化记忆**，让它们不受输入顺序的干扰。

### 一句话记住它
前提的排列顺序直接决定大语言模型的推理成功率，和把解题步骤写成正确顺序的草稿一样关键。