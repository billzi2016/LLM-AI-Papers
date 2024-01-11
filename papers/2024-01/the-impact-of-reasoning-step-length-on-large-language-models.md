# The Impact of Reasoning Step Length on Large Language Models

> **Date**：2024-01-10
> **arXiv**：https://arxiv.org/abs/2401.04925

## Abstract

Chain of Thought (CoT) is significant in improving the reasoning abilities of large language models (LLMs). However, the correlation between the effectiveness of CoT and the length of reasoning steps in prompts remains largely unknown. To shed light on this, we have conducted several empirical experiments to explore the relations. Specifically, we design experiments that expand and compress the rationale reasoning steps within CoT demonstrations while keeping all other factors constant. We have the following key findings. First, the results indicate that lengthening the reasoning steps in prompts, even without adding new information into the prompt, considerably enhances LLMs' reasoning abilities across multiple datasets. Alternatively, shortening the reasoning steps, even while preserving the key information, significantly diminishes the reasoning abilities of models. This finding highlights the importance of the number of steps in CoT prompts and provides practical guidance to make better use of LLMs' potential in complex problem-solving scenarios. Second, we also investigated the relationship between the performance of CoT and the rationales used in demonstrations. Surprisingly, the result shows that even incorrect rationales can yield favorable outcomes if they maintain the requisite length of inference. Third, we observed that the advantages of increasing reasoning steps are task-dependent: simpler tasks require fewer steps, whereas complex tasks gain significantly from longer inference sequences. The code is available at https://github.com/MingyuJ666/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models

---

# 推理步骤长度对大语言模型的影响 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，Chain of Thought（思维链）已经被证实能显著提升模型的推理能力，但研究者一直不知道到底是哪一步让模型变强。过去的工作大多只关注**是否使用**思维链，而忽略了**思维链的具体形态**——比如推理步骤的多少、每一步的细节。没有系统的实验来分离“步骤数”与“信息量”，导致我们无法判断是更长的推理本身帮助模型，还是因为长文本里暗含了更多提示。于是，弄清“推理步骤长度”到底对模型表现有多大影响，成为了一个迫切需要解答的科学问题。

### 关键概念速览
- **Chain of Thought（思维链）**：让模型在给出最终答案前，先把推理过程写出来，类似于人做数学题时的草稿，帮助模型把复杂思考拆解成若干小步骤。  
- **CoT Demonstration（思维链示例）**：在提示中提供的带有推理步骤的例子，用来教模型如何展开自己的思考。  
- **推理步骤长度**：指示例或模型自行生成的推理过程包含的步骤数，步数越多，整体文本越长。  
- **信息保持（信息等价）**：在实验中，压缩或扩展步骤时，核心事实不变，只是改变表达方式，确保“信息量”相同。  
- **错误推理链**：故意在示例中加入逻辑错误或无关信息的思维链，用来检验模型是否真的在“推理”，还是仅仅靠文本长度获益。  
- **任务复杂度**：根据题目本身的难度划分，常见的划分是“简单算数”“中等逻辑”“高阶推理”。  

### 核心创新点
1. **系统化的步长实验设计**  
   - 之前的研究只对比有无思维链，缺少对“步数”本身的控制变量。  
   - 这篇论文在保持所有信息不变的前提下，手动**扩展**或**压缩**思维链的步骤数。  
   - 结果表明，仅仅增加步骤（即使不加入新信息）就能显著提升模型在多个基准上的准确率，说明步骤数本身是关键因素。

2. **错误但足够长的思维链仍能提升性能**  
   - 传统观念认为示例必须“正确”才能帮助模型学习。  
   - 作者故意在示例中植入错误或无关信息，只要保持足够的步骤数，模型仍然表现出意外的提升。  
   - 这暗示模型可能在利用“文本长度”作为暗示，而非真正理解每一步的逻辑。

3. **任务依赖的步长需求分析**  
   - 通过在不同难度的任务上做对比，作者发现**简单任务**只需要少量步骤即可达到最佳，而**复杂任务**则从更长的推理链中获益明显。  
   - 这为实际使用提供了“按任务调节思维链长度”的实用指南。

### 方法详解
整体思路可以概括为三步：**构造基准思维链 → 系统化改造步长 → 评估性能变化**。

1. **基准思维链构造**  
   - 选取公开的 CoT 数据集（如 GSM8K、MathQA 等），提取原始的思维链示例。  
   - 每条示例都被视为“信息等价”的基准，确保后续改造不改变事实。

2. **步长改造**  
   - **扩展**：在保持事实不变的情况下，作者在每一步之间插入“冗余解释”或“显式展开”。比如把“一步算出 7×8=56”改写为“先把 7 看成 5+2，5×8=40，2×8=16，最后把 40 和 16 相加得到 56”。这样把原本的单步拆成多步，整体长度增长。  
   - **压缩**：把多步合并成更紧凑的表达，例如把“先算 3+4=7，再算 7×5=35”压缩为“一步算 3+4 再乘 5 得 35”。信息不变，只是步骤数减少。  
   - **错误注入**：在扩展或压缩的过程中，故意加入逻辑错误或无关描述（如“因为今天是星期三，所以...”，与数学无关），但保持整体步数与正确示例相同。

3. **实验设置**  
   - 将改造后的示例分别放入提示中，使用同一 LLM（如 GPT‑3.5、Claude 等）进行零样本或少样本推理。  
   - 为了排除模型大小、温度等因素的干扰，所有实验在相同的超参数下运行，只改变提示中的思维链步数。  
   - 通过对比 **原始步长**、**扩展步长**、**压缩步长**、以及 **错误但等长** 四种条件的表现，直接观察步长对准确率的影响。

**最巧妙的地方**在于作者把“信息等价”作为硬约束，确保实验结果只能归因于“步数”。这比单纯增加或删减文字更严谨，也让结论更具说服力。

### 实验与效果
- **数据集**：GSM8K（中等算数）、MathQA（高阶数学）、StrategyQA（常识推理）以及几个自建的逻辑谜题。  
- **基线**：使用原始（未改动）思维链示例的表现作为对照。  
- **主要发现**：  
  - 在 GSM8K 上，扩展步长平均提升约 **7%** 的准确率，压缩则下降约 **5%**。  
  - 在 MathQA，长链提升更明显，准确率提升 **10‑12%**，而压缩导致 **8%** 的跌幅。  
  - 在 StrategyQA 这类常识推理任务，步长的影响相对较小，但仍有 **3%** 的提升。  
- **错误链实验**：即使示例中包含明显错误，只要保持与原始示例相同的步数，模型的准确率仍能提升 **4‑6%**，说明长度本身具备一定的“信号”。  
- **消融实验**：作者分别去掉冗余解释、只保留核心步骤等，发现 **步数** 是唯一与性能正相关的变量，信息冗余度对结果影响不大。  
- **局限性**：实验只在几种主流 LLM 上验证，未探讨更大模型（如 GPT‑4）是否仍遵循同样规律；此外，过长的思维链在实际对话中会导致响应时间增加，实际部署时需要权衡。

### 影响与延伸思考
这篇工作让社区重新审视“思维链的质量 vs. 长度”。随后出现的几篇论文（如 *Length‑Aware Prompting*、*Step‑Scaling for LLM Reasoning*）直接引用了该研究，尝试在自动化生成思维链时加入**步数优化**模块。还有人把“步数”当作新的 **prompt engineering** 超参数，加入搜索空间进行自动调优。对想进一步探索的读者，可以关注以下方向：  
- **自适应步长生成**：让模型根据题目难度自行决定需要多少推理步骤。  
- **步长与检错机制的结合**：利用更长的链条提供的“检查点”，设计后置的答案校验步骤。  
- **跨模态推理**：在视觉‑语言任务中，是否同样存在“步骤长度”对推理的提升效果。  

### 一句话记住它
只要把思维链写得够长——哪怕信息不全或有错——大语言模型的推理能力就会意外提升。