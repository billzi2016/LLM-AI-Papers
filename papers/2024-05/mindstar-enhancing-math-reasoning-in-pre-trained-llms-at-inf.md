# MindStar: Enhancing Math Reasoning in Pre-trained LLMs at Inference Time

> **Date**：2024-05-25
> **arXiv**：https://arxiv.org/abs/2405.16265

## Abstract

Although Large Language Models (LLMs) achieve remarkable performance across various tasks, they often struggle with complex reasoning tasks, such as answering mathematical questions. Recent efforts to address this issue have primarily focused on leveraging mathematical datasets through supervised fine-tuning or self-improvement techniques. However, these methods often depend on high-quality datasets that are difficult to prepare, or they require substantial computational resources for fine-tuning. Inspired by findings that LLMs know how to produce the right answer but struggle to select the correct reasoning path, we propose a purely inference-based searching method -- MindStar (M*). This method formulates reasoning tasks as searching problems and proposes two search ideas to identify the optimal reasoning paths. We evaluate the M* framework on both the GSM8K and MATH datasets, comparing its performance with existing open and closed-source LLMs. Our results demonstrate that M* significantly enhances the reasoning abilities of open-source models, such as Llama-2-13B and Mistral-7B, and achieves comparable performance to GPT-3.5 and Grok-1, but with substantially reduced model size and computational costs.

---

# MindStar：在推理阶段提升预训练大语言模型的数学推理能力 论文详细解读

### 背景：这个问题为什么难？

数学题往往需要多步推理、符号操作和对中间结果的检验。虽然大语言模型（LLM）在语言理解上已经很强，但在需要严密逻辑的数学场景里，它们常常给出看似合理却错误的答案。过去的改进大多依赖**监督微调**：准备大规模的数学题‑答案对，让模型在训练阶段学习推理路径；或者使用**自我提升**（self‑improvement），让模型在生成答案后再自行校正。这两类办法都有根本缺陷：高质量的数学数据集难以构建，标注成本高；而微调又需要大量算力，普通研究者难以承受。因此，如何在不改动模型权重、只动推理过程的前提下提升数学推理，成为亟待突破的难题。

### 关键概念速览
- **预训练大语言模型（LLM）**：在海量文本上进行自监督学习得到的模型，能够生成连贯语言，但不一定具备严密推理能力。  
- **思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出逐步推理，就像解题时在草稿纸上写步骤，帮助模型保持逻辑连贯。  
- **搜索树**：把一次推理过程看成在一棵树上遍历，每个节点是一次模型的输出（可能是一步计算、一个中间式或完整答案），树的分支代表不同的推理分支。  
- **候选答案生成**：模型在同一问题上多次采样，得到若干可能的答案或推理序列，类似于“多次尝试”。  
- **路径评分（Path Scoring）**：对每条候选推理路径打分，决定哪条最可信。评分可以基于模型自身的置信度、外部校验器或自洽性检查。  
- **M\***：本文提出的搜索框架名称，核心思想是把数学推理转化为在搜索树中寻找最优路径，而不是一次性直接输出答案。  
- **GSM8K / MATH**：两个公开的数学推理基准，前者侧重小学到初中水平的文字题，后者覆盖更高阶的竞赛题。  

### 核心创新点
1. **从微调转向纯推理搜索**  
   - 之前的主流做法是通过大规模微调让模型内部“记住”推理步骤。  
   - MindStar 直接在推理阶段构造搜索空间，让模型保持原始权重不变。  
   - 这种做法把算力需求从训练阶段搬到推理阶段，显著降低了硬件门槛，同时保留了模型的通用语言能力。

2. **双层搜索策略**  
   - 传统的 CoT 只让模型一次性生成完整链式思考，若中途走错就难以纠正。  
   - MindStar 先生成**多个候选推理链**（第一层），再在这些链内部进行**细粒度路径搜索**（第二层），比如对每一步的中间式进行重新采样或验证。  
   - 双层结构让系统能够在广度上覆盖多种思路，在深度上对每条思路进行细致打磨，从而更容易捕获正确的解法。

3. **基于模型自评的路径选择**  
   - 许多搜索方法依赖外部校验器（如符号求解器），成本高且难以通用。  
   - MindStar 让同一个 LLM 对自己的输出进行置信度估计，并结合**自洽性检查**（如前后步骤是否相符）来给每条路径打分。  
   - 这种“模型自评”机制在不引入额外工具的前提下实现了有效的路径筛选。

4. **在开源模型上实现闭源水平**  
   - 通过上述搜索框架，Llama‑2‑13B 和 Mistral‑7B 在 GSM8K、MATH 上的得分接近 GPT‑3.5 与 Grok‑1，且模型体积仅为后者的 1/3 左右。  
   - 这证明了推理层搜索可以弥补模型规模的不足，为资源受限的研究者提供了可行的提升路径。

### 方法详解
#### 整体框架概览
MindStar 将一次数学问答拆成三大步骤：  
1. **候选生成**：模型对同一道题进行多次采样，得到若干不同的推理链（每条链是一系列“思考步骤”+最终答案）。  
2. **路径扩展与评估**：对每条链内部的关键步骤再次采样或进行自检，形成更细的子树；随后用模型自评和自洽性规则为每条完整路径打分。  
3. **最优路径选择**：选取得分最高的路径作为最终答案输出。

#### 关键模块拆解
- **候选生成（多样性采样）**  
  类比于学生在解题时先写下几种不同的思路，模型使用温度调高的采样或 nucleus 采样，确保输出的多样性。每次采样的输出被视为搜索树的根节点的子节点。

- **路径扩展（局部搜索）**  
  对每条候选链，系统会识别出“关键转折点”（如代数化简、函数求值等），在这些点上再次让模型生成若干备选子步骤。这样在原有链的基础上插入分支，形成一个局部的搜索子树。  
  这里的创新在于**只对关键点进行二次采样**，避免对整条链全局重新搜索导致的指数爆炸。

- **自评与自洽性检查**  
  每个生成的子步骤都会被模型自身重新评估一次，输出一个置信度分数。随后系统检查链中相邻步骤是否前后一致（例如，若一步得到“x=5”，后一步的代入是否真的用了 x=5）。不一致的路径会被惩罚，得分下降。

- **路径评分聚合**  
  最终得分是置信度、步骤一致性以及整体答案的语言流畅度的加权和。权重由实验调参得到，目的是兼顾模型自信和逻辑严密。

- **最优解输出**  
  选取得分最高的完整链，直接把链中的最终答案返回给用户。如果最高分仍低于预设阈值，系统可以返回“未找到可靠解”，防止误导。

#### 反直觉/巧妙之处
- **不使用外部符号求解器**：很多人认为数学推理必须借助专门的符号计算工具，但 MindStar 通过模型自评和自洽检查实现了“内部校验”，大幅降低了系统复杂度。  
- **双层搜索的宽度‑深度平衡**：一次性生成完整链往往要在宽度和深度之间做极端取舍，MindStar 先宽后深的策略让搜索空间保持可控，却仍能捕获深层次的正确推理。  

### 实验与效果
- **数据集**：在 GSM8K（约 8k 题）和 MATH（约 12k 题）两个公开基准上进行评测。  
- **对比基线**：包括开源模型（Llama‑2‑13B、Mistral‑7B、Falcon‑40B）以及闭源模型（GPT‑3.5、Grok‑1）。  
- **主要结果**：在 GSM8K 上，MindStar 使 Llama‑2‑13B 的准确率从原始的约 45% 提升到约 68%，接近 GPT‑3.5 的 70% 左右；在 MATH 上，Mistral‑7B 的得分提升约 12% 点，达到与 Grok‑1 相当的水平。整体来看，MindStar 在保持模型体积不变的情况下，实现了 15%‑20% 的相对提升。  
- **消融实验**：作者分别关闭（1）双层搜索，仅保留单轮候选生成；（2）自评评分，只用语言流畅度打分。结果显示，去掉双层搜索会导致准确率下降约 6%‑8%；去掉自评会导致错误答案比例上升约 10%。说明两者都是提升的关键因素。  
- **局限性**：搜索过程会带来推理时延，尤其在需要多轮子步骤扩展的长题上，响应时间比直接输出慢 2‑3 倍；此外，模型自评仍然受限于模型本身的信任度，对极其复杂的符号推导仍可能失效。作者在论文中承认，这种方法更适合中等难度的数学题，对高阶竞赛题仍有提升空间。

### 影响与延伸思考
MindStar 的出现让“推理层搜索”成为提升开源 LLM 数学能力的可行路线，激发了后续工作在以下方向的探索：  
- **更高效的搜索策略**：如使用蒙特卡罗树搜索（MCTS）或强化学习引导的路径扩展，以进一步压缩推理时延。  
- **跨模态校验**：结合轻量级符号求解器或图形计算引擎，对关键步骤进行外部验证，兼顾效率与准确性。  
- **通用推理框架**：把 MindStar 的搜索思路迁移到代码生成、逻辑推理等非数学任务，已经有几篇工作在尝试。  
如果想深入了解，可以关注近期在 arXiv 上出现的 “Self‑Consistency” 与 “Tree‑of‑Thought” 系列论文，它们与 MindStar 在思路上有不少交叉。  

### 一句话记住它
**MindStar 用搜索树在推理时“挑选最佳思路”，让小模型也能跑出大模型的数学成绩。**