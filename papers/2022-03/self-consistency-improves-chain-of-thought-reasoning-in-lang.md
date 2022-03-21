# Self-Consistency Improves Chain of Thought Reasoning in Language Models

> **Date**：2022-03-21
> **arXiv**：https://arxiv.org/abs/2203.11171

## Abstract

Chain-of-thought prompting combined with pre-trained large language models has achieved encouraging results on complex reasoning tasks. In this paper, we propose a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting. It first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths. Self-consistency leverages the intuition that a complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer. Our extensive empirical evaluation shows that self-consistency boosts the performance of chain-of-thought prompting with a striking margin on a range of popular arithmetic and commonsense reasoning benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).

---

# 自洽性提升链式思维推理的语言模型 论文详细解读

### 背景：这个问题为什么难？

在大模型上做复杂推理时，单纯让模型一步到位（直接输出答案）往往出错率高。出现了“思维链”（Chain‑of‑Thought, CoT）提示，让模型先把推理过程写出来，效果明显提升。但大多数 CoT 实验仍然采用贪心解码——每一步都选概率最高的词，这相当于只让模型走一条思路。实际的数学或常识题往往有多条等价的解法，贪心只能捕捉其中一种，容易因为一次小失误导致最终答案错误。于是，如何让模型利用多样的思考路径来提高鲁棒性，成为当时的瓶颈。

### 关键概念速览

**Chain‑of‑Thought（思维链）**：在生成答案前，模型先输出一系列推理步骤，类似人做题时的草稿，帮助模型保持逻辑连贯性。  

**贪心解码**：每一步都选概率最大的词，等同于只走一条最“自信”的路径，缺乏多样性。  

**采样（sampling）**：在每一步按模型分布随机抽词，能够生成多条不同的思考链，像让模型“思考多次”。  

**自洽性（self‑consistency）**：对一组采样得到的思维链进行投票，选出出现次数最多的答案，假设正确答案会在多数思路中出现。  

**边缘化（marginalization）**：在统计学里把不感兴趣的变量（这里是具体的思考路径）求和，只保留感兴趣的变量（答案）的概率分布。  

**多样性采样**：有意让模型产生不同的解法，常用温度调节或 nucleus sampling 等技巧。  

**投票机制**：把所有采样得到的答案计数，选票最高者为最终输出，类似多人讨论后取多数意见。

### 核心创新点

1. **从单一路径到多路径采样**  
   *之前的做法*：CoT 只用贪心解码，得到唯一的思维链。  
   *本文的做法*：在同一提示下多次采样，生成一批（如 40 条）不同的思维链。  
   *带来的改变*：模型能够覆盖多种等价推理方式，降低单一路径偶然失误的风险。

2. **答案层面的投票而非路径层面的选择**  
   *之前的做法*：只看贪心路径的最终答案。  
   *本文的做法*：对所有采样得到的答案进行计数，选出现次数最多的答案作为输出。  
   *带来的改变*：即使某条思维链内部出现小错误，只要整体多数答案正确，最终结果仍然可靠。

3. **把“思考多次”视为概率边缘化**  
   *之前的做法*：没有显式的概率计算，只靠单一输出。  
   *本文的做法*：把所有采样的答案视作对答案分布的近似采样，统计其频率相当于对隐藏的思考路径做边缘化。  
   *带来的改变*：提供了更稳健的统计解释，解释了为何多样性采样能提升准确率。

4. **在多个公开基准上系统评估**  
   *之前的工作*：大多只在少数算术任务上展示提升。  
   *本文的做法*：在 GSM8K、SVAMP、AQuA、StrategyQA、ARC‑challenge 等五大基准上做全方位对比。  
   *带来的改变*：证明自洽性在算术、常识、策略推理等不同场景都有显著增益，提升幅度从 3.9% 到 17.9% 不等。

### 方法详解

整体思路可以拆成三步：**提示 → 多次采样 → 投票**。

1. **提示阶段**  
   使用标准的 CoT 提示，例如“请先列出解题步骤，然后给出答案”。这一步和传统 CoT 完全相同，只是后面不再只取一次输出。

2. **多次采样**  
   - **采样策略**：在每一步生成时使用温度（temperature）或 nucleus（p）采样，使得模型在同一提示下能够产生不同的词序列。  
   - **采样次数**：论文中常用 40 次采样，这个数字是经验值，足够覆盖多样的思路但不会导致计算成本爆炸。  
   - **生成的结构**：每一次采样都会得到一段完整的思维链，形如“步骤1 → 步骤2 → … → 最终答案”。这些链条之间可能在步骤顺序、计算方式甚至中间的文字描述上都有差异。

3. **答案投票**  
   - **抽取答案**：从每条思维链的最后一行提取答案（数值或文字）。  
   - **计数**：把所有答案放进一个哈希表，统计每个答案出现的次数。  
   - **选取**：出现次数最多的答案即为最终输出。如果出现平票，常用随机或次高频的方式决定。  
   - **直观类比**：想象一群学生分别写下自己的解题过程，老师最后只看他们的答案并选出多数同学给出的答案，这样可以抵消个别学生的笔误。

**最巧妙的点**在于：作者并没有对思维链本身做任何后处理（比如验证每一步的正确性），而是直接把注意力放在答案的统计上。这样既省去复杂的路径评估，也利用了“正确答案往往是多数思路的交汇点”这一直觉。

### 实验与效果

- **测试任务**：  
  - **GSM8K**（8‑K 级别的数学题）  
  - **SVAMP**（变体算术）  
  - **AQuA**（代数问答）  
  - **StrategyQA**（需要多步推理的常识问答）  
  - **ARC‑challenge**（高难度常识推理）

- **对比基线**：  
  - 传统 CoT + 贪心解码  
  - 直接让模型输出答案（Zero‑shot）  
  - 其他少数采用多样性采样但不做投票的变体（原文未给出具体数字，只说明自洽性显著领先）

- **提升幅度**（相对传统 CoT）：  
  - GSM8K：+17.9%  
  - SVAMP：+11.0%  
  - AQuA：+12.2%  
  - StrategyQA：+6.4%  
  - ARC‑challenge：+3.9%

- **消融实验**：  
  - **采样次数**：从 5 到 80 次逐步提升，准确率随次数上升但在约 40 次后趋于饱和，说明适度的多样性足够。  
  - **温度设置**：温度过低（接近贪心）导致多样性不足，提升有限；温度过高则产生太多噪声，投票效果下降。  
  - **投票 vs. 取最高概率答案**：仅取最高概率的单一路径几乎回到贪心水平，验证了投票机制是关键。

- **局限性**：  
  - 计算成本约为原来 40 倍，因为需要多次前向传播。  
  - 对于答案空间极大（如开放式生成）时，投票可能出现稀疏分布，效果不如闭式数值题。  
  - 论文未讨论在极端长文本推理（如代码生成）上的适用性。

### 影响与延伸思考

自洽性策略在发布后迅速成为大模型推理的“标准配方”。随后出现的工作如 **“Tree‑of‑Thought”**、**“Least‑to‑Most Prompting”**、以及各种 **“self‑refine”** 框架，都在不同层面上借鉴了“多样性采样 + 结果聚合”的思路。还有研究把投票改成 **置信度加权**，或把路径本身再喂回模型进行二次评估（即“自我纠错”）。如果想进一步深入，可以关注：

- **高效采样**：如何在保持多样性的同时降低计算开销（如使用低秩近似或分层采样）。  
- **路径质量评估**：设计自动化的思维链验证器，让模型在投票前剔除明显错误的路径。  
- **跨模态自洽**：把文字、图像、代码等不同模态的推理链一起投票，探索更通用的自洽框架。

（以上影响基于后续文献和社区实践，部分为推测）

### 一句话记住它

让大模型多想几遍、把答案投票——多数思路的共识往往就是正确答案。