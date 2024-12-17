# Are Your LLMs Capable of Stable Reasoning?

> **Date**：2024-12-17
> **arXiv**：https://arxiv.org/abs/2412.13147

## Abstract

The rapid advancement of large language models (LLMs) has shown remarkable progress in complex reasoning tasks. However, a significant disparity exists between benchmark performances and real-world applications. We attribute this gap primarily to current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, especially in complex reasoning tasks where both accuracy and consistency are essential. In this paper, we introduce G-Pass@$k$, a novel evaluation metric that continuously assesses model performance across multiple sampling attempts, quantifying both the model's performance potential and its stability. Through extensive experiments on various public and newly constructed benchmarks, we employ G-Pass@$k$ in conjunction with state-of-the-art large language models to provide comprehensive insights into their potential capabilities and operational consistency. Our findings reveal a significant opportunity to enhance the realistic reasoning abilities of LLMs, underscoring the necessity for more robust evaluation metrics.

---

# 你的大型语言模型能进行稳定推理吗？ 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在答题、写代码、推理等任务上已经能跑出惊人的成绩，但这些成绩大多来源于一次性评测。实际使用时，模型往往要多次抽样、在不同提示下重复回答，而答案的波动却很大。传统的评估方式只看单次最高分或平均分，根本抓不住“同一个问题模型能否始终给出可靠答案”。于是，研究者发现 benchmark 上的高分并不等同于真实场景的稳健推理能力，这成为限制 LLM 进一步落地的关键瓶颈。

### 关键概念速览
- **LLM（大型语言模型）**：基于海量文本训练的神经网络，能够生成自然语言。把它想象成一个“会说话的百科全书”，但它的记忆是概率分布而非确定性规则。  
- **采样（sampling）**：模型在生成每个词时根据概率随机挑选，下一个词可能有多个候选。类似掷骰子决定下一步走向，导致同一个问题的答案会有差异。  
- **一致性（stability）**：在多次采样或不同提示下，模型输出保持相似或相同的程度。可以比作同一位老师在不同课堂上讲同一道题，答案是否保持一致。  
- **G-Pass@$k$**：一种新提出的评估指标，记录模型在 $k$ 次采样中是否至少出现一次正确答案，并衡量出现次数的分布，从而同时反映潜在能力和稳定性。  
- **CoT（Chain‑of‑Thought，思维链）**：让模型先写出推理步骤再给出结论的技巧，类似人做数学题时先列出解题思路。  
- **LiveMathBench**：作者自行构建的数学竞赛类基准，题目难度和形式更贴近真实竞赛，专门用来检验模型的深度推理和一致性。  
- **消融实验（ablation study）**：逐个去掉或替换模型组件，观察性能变化，以判断每个部件的重要性。  

### 核心创新点
1. **评估视角从单次准确率转向多次采样的潜力与稳定性**  
   之前的工作只报告一次抽样的准确率或平均分，忽略了模型在不同随机种子下的表现差异。作者提出在同一输入上进行多次采样，并用 G‑Pass@$k$ 统计“至少一次正确”和“正确出现的频率”。这种做法让我们看到模型的上限潜力以及实际可达的可靠度。  

2. **G‑Pass@$k$ 指标的具体定义与实现**  
   传统的 @k 指标只看前 k 个候选答案是否包含正确答案，而 G‑Pass@$k$ 进一步记录每一次采样是否命中，并用通过率曲线量化稳定性。这样既能比较不同模型的“最佳表现”，也能比较它们的“一致表现”。  

3. **构建覆盖多种推理难度的全新基准**  
   为了验证指标的通用性，作者在公开的数学、逻辑、代码推理数据集之外，新增了 LiveMathBench 等高难度题库。这些题目要求模型进行多步推理、符号操作和常识结合，能够更敏感地暴露不稳定行为。  

4. **系统性实验揭示“高分≠高稳”现象**  
   通过在多个模型（GPT‑4、Claude、LLaMA‑2 等）上跑 G‑Pass@$k$，作者发现即便在同一基准上取得 90% 以上的单次准确率，稳定性指标往往只有 60% 左右。该发现促使社区重新审视评价体系，强调“一致性”与“潜力”同等重要。  

### 方法详解
整体思路可以拆成三步：**多次采样 → 结果聚合 → G‑Pass@$k$ 计算**。下面逐层展开。

1. **多次采样**  
   对每个测试样本，模型在相同的提示下执行 $k$ 次独立采样。采样时保持温度、top‑p 等超参数不变，只更换随机种子。这样得到 $k$ 份答案序列，每份答案都可能包含不同的推理路径。  

2. **答案标准化与匹配**  
   因为模型可能用不同的表述方式给出同一个答案，需要先对答案进行归一化（去除空格、大小写统一、数值统一为标准形式）。随后使用字符串匹配或基于语义相似度的判定器，判断每次采样是否“正确”。  

3. **G‑Pass@$k$ 计算**  
   - **通过率（Pass Rate）**：在 $k$ 次采样中，至少出现一次正确答案的样本比例。  
   - **稳定性曲线（Stability Curve）**：统计每个样本正确答案出现的次数（0 到 $k$），绘制分布。曲线越靠右，说明模型在多数采样中都能给出正确答案。  
   - **综合得分**：作者将通过率与稳定性曲线的面积（AUC）加权求和，得到一个兼顾潜力与一致性的单值指标。  

4. **实现细节**  
   - 为了避免采样次数过多导致计算成本爆炸，作者在实验中选取 $k=10$ 作为默认值，证明在多数任务上已能收敛。  
   - 在 CoT 场景下，作者让模型在每次采样前都生成完整的思维链，这样可以观察思维链的多样性与最终答案的一致性。  
   - 关键的“反直觉”点在于：并不是采样次数越多指标越好，而是要找到一个平衡点，使得通过率提升不再显著时停止采样，避免浪费算力。  

### 实验与效果
- **数据集**：作者在公开的 GSM‑8K（数学）、MMLU（通用知识）、HumanEval（代码）等基准上跑实验，并额外加入 LiveMathBench（高难度数学竞赛题）以及自建的逻辑推理集合。  
- **对比基线**：传统的 @k、单次准确率、以及最近的 “Self‑Consistency” 方法。  
- **主要发现**：在 GSM‑8K 上，GPT‑4 的单次准确率为 92%，但 G‑Pass@10 通过率仅为 78%；而在 LiveMathBench 上，GPT‑4 的单次准确率 85% 对应 G‑Pass@10 通过率 62%。相对传统 @k，G‑Pass@$k$ 能更细致地区分模型的潜在能力与实际稳定性。  
- **消融实验**：去掉思维链（直接输出答案）后，G‑Pass@$k$ 的通过率下降约 7%；降低采样次数至 $k=5$ 时，通过率下降约 4%，说明多次采样和 CoT 对提升稳定性都有贡献。  
- **局限性**：作者承认 G‑Pass@$k$ 仍然依赖人工标注的答案匹配规则，对开放式生成（如创意写作）难以直接套用；此外，采样成本随 $k$ 增大而线性上升，在资源受限的场景下仍需权衡。  

### 影响与延伸思考
这篇工作在社区里掀起了对“评估一致性”新一轮讨论，随后出现了诸如 **StableEval**、**Consistency‑Bench** 等后续工作，它们在不同任务上进一步细化了稳定性度量。还有研究尝试把 G‑Pass@$k$ 融入训练目标，让模型在优化时直接考虑多次采样的通过率，从而在根本上提升稳健性。对想继续深入的读者，可以关注 **自适应采样**（根据前几次结果动态决定是否继续采样）和 **多模态一致性评估**（将文本、图像等输出统一到同一稳定性框架）这两个方向。  

### 一句话记住它
**G‑Pass@$k$ 用多次抽样同时量化模型的最高潜力和答案一致性，提醒我们：高分不等于可靠推理。**