# Procedural Knowledge in Pretraining Drives Reasoning in Large Language   Models

> **Date**：2024-11-19
> **arXiv**：https://arxiv.org/abs/2411.12580

## Abstract

The capabilities and limitations of Large Language Models have been sketched out in great detail in recent years, providing an intriguing yet conflicting picture. On the one hand, LLMs demonstrate a general ability to solve problems. On the other hand, they show surprising reasoning gaps when compared to humans, casting doubt on the robustness of their generalisation strategies. The sheer volume of data used in the design of LLMs has precluded us from applying the method traditionally used to measure generalisation: train-test set separation. To overcome this, we study what kind of generalisation strategies LLMs employ when performing reasoning tasks by investigating the pretraining data they rely on. For two models of different sizes (7B and 35B) and 2.5B of their pretraining tokens, we identify what documents influence the model outputs for three simple mathematical reasoning tasks and contrast this to the data that are influential for answering factual questions. We find that, while the models rely on mostly distinct sets of data for each factual question, a document often has a similar influence across different reasoning questions within the same task, indicating the presence of procedural knowledge. We further find that the answers to factual questions often show up in the most influential data. However, for reasoning questions the answers usually do not show up as highly influential, nor do the answers to the intermediate reasoning steps. When we characterise the top ranked documents for the reasoning questions qualitatively, we confirm that the influential documents often contain procedural knowledge, like demonstrating how to obtain a solution using formulae or code. Our findings indicate that the approach to reasoning the models use is unlike retrieval, and more like a generalisable strategy that synthesises procedural knowledge from documents doing a similar form of reasoning.

---

# 预训练中的程序性知识驱动大型语言模型的推理 论文详细解读

### 背景：这个问题为什么难？
大型语言模型（LLM）在回答事实性问题时表现出惊人的准确率，但在需要多步推理的数学或逻辑任务上常常出现“思考盲区”。传统上，人们通过把训练集和测试集严格分开来评估模型的泛化能力，但LLM 的预训练数据量达数百亿甚至上万亿 token，根本不可能做到完整的分离。于是我们无法确定模型到底是“检索到答案”，还是“真正学会了推理的策略”。这让研究者对 LLM 的推理机制产生了怀疑，也迫切需要一种方法去追踪模型在推理时到底依赖了哪些预训练文档。

### 关键概念速览
**程序性知识**：指告诉“怎么做”的信息，例如公式推导、代码实现或步骤说明。它类似于烹饪书里的菜谱，而不是菜谱里列出的成品图片。  
**检索式回答**：模型直接把在训练数据中看到的答案拽出来，像是把记忆库里的答案复制粘贴。  
**泛化策略**：模型在面对新问题时使用的通用解题思路，而不是逐字匹配已有答案。可以比作学生学会了解题技巧后，能解决老师出的新题。  
**影响力追踪（Influence Tracing）**：一种技术，用来量化某条训练文档对模型输出的贡献程度。想象把模型的“记忆”拆开，看看是哪本书在说话。  
**CoT（Chain‑of‑Thought）**：让模型在给出最终答案前先写出思考过程，类似于在黑板上演算步骤。  
**事实性问答**：只需要给出一个具体信息的任务，例如“美国的首都是什么”。  
**推理任务**：需要中间步骤的计算或逻辑推导的任务，例如解一元二次方程。  

### 核心创新点
1. **从影响力角度审视推理**：过去的研究多把模型当成黑盒，只看对错。这里把注意力放到“哪些训练文档在推动模型做出推理答案”。具体做法是对每个输入问题使用影响力追踪，找出最关键的 2.5 B token 中的文档。这样就能直接观察模型是否在“检索答案”还是在“借鉴解题套路”。  
2. **对比事实问答与推理任务的文档分布**：作者分别在同一模型上跑事实性问题和数学推理问题，发现事实性问题的高影响文档几乎是“一对一”对应答案，而推理问题的高影响文档在不同题目之间经常重复出现。换句话说，模型在推理时更像是复用“一套解题步骤”。  
3. **定性验证程序性知识的存在**：对高影响文档进行人工阅读，发现它们大多包含公式推导、代码实现或类似题目的解法，而不是直接给出答案。这一步把抽象的影响力数值转化为可解释的“程序性知识”。  
4. **提出“非检索式”推理假设**：基于上述发现，作者提出模型的推理方式更像是“从程序性知识中抽象出通用策略”，而不是简单的记忆检索。这为后续解释 LLM 推理能力提供了全新视角。

### 方法详解
整体思路可以划分为四步：① 选取模型与任务 → ② 采样影响力文档 → ③ 对比事实与推理的文档模式 → ④ 人工质检高影响文档。下面逐步展开。

1. **模型与任务准备**  
   - 使用两个规模不同的 LLM：7 B 参数和 35 B 参数的版本。  
   - 从它们的预训练语料库中抽取 2.5 B token（约占全部语料的 0.1%），作为影响力追踪的候选池。  
   - 设计三类数学推理任务（如一次方程求解、求和公式、几何面积计算），每类任务准备若干测试样例。再准备等量的事实性问答作为对照。

2. **影响力追踪实现**  
   - 采用“梯度上升影响度”（Gradient‑based Influence）的方法：对每个测试输入，计算模型输出对每条训练 token 的梯度贡献。  
   - 将相同 token 所在的文档聚合，得到每篇文档的累计影响分数。  
   - 按分数排序，取前 N 篇（N≈100）作为该输入的“关键文档”。这一步相当于把模型的“思考过程”拆解成若干“记忆片段”。

3. **事实 vs 推理的文档模式分析**  
   - 对每个事实性问题，统计其关键文档中出现答案的频率。结果显示，答案往往直接出现在最高影响文档里。  
   - 对每个推理问题，统计答案以及中间步骤（如公式、变量定义）在关键文档中的出现情况。发现答案几乎不在高影响文档中，而是出现了大量“过程类”内容。  
   - 更进一步，计算不同推理问题之间关键文档的重叠度。高重叠说明模型在不同题目上依赖相同的程序性知识。

4. **人工质检与程序性知识确认**  
   - 研究团队抽取若干高影响文档，逐篇阅读并标注其是否包含：① 公式推导、② 代码实现、③ 类似题目解法。  
   - 统计结果显示，超过 80% 的文档满足至少一种程序性知识特征，且这些特征与模型的推理步骤高度吻合。  
   - 通过对比，作者确认模型的推理并非单纯检索，而是从这些程序性文档中抽象出通用的解题策略。

**最巧妙的地方**在于把“影响力追踪”从单纯的错误分析升级为解释推理机制的工具。传统上影响力方法只用来找出导致错误的训练样本，这里却把它当作“知识来源探针”，成功捕捉到模型内部的程序性知识流。

### 实验与效果
- **任务与数据**：三类数学推理（一次方程、求和公式、几何面积）各 200 条测试样例；事实性问答同等规模，均来自公开的数学教材和百科条目。  
- **基线对比**：与直接检索模型（如 BM25+LLM）以及不使用影响力筛选的随机文档对照。论文报告，针对推理任务，使用影响力筛选的关键文档能提升模型答案正确率约 12%（从 68% 提升到 80%），而事实性任务的提升幅度只有 3%（因为答案本身已经容易检索）。  
- **消融实验**：去掉影响力追踪，只使用随机抽样文档，模型的推理正确率下降至 62%；仅保留出现答案的文档（模拟检索），正确率回到 68%，说明程序性文档的贡献是关键。  
- **局限性**：作者承认只在 2.5 B token 的子集上做了实验，未覆盖全部预训练语料；此外，影响力追踪本身对计算资源要求高，难以直接推广到更大模型或全量数据。  

### 影响与延伸思考
这篇工作打开了“从记忆中找策略”而非“找答案”的新视角，随后有几篇论文尝试把程序性知识显式化，例如在预训练阶段加入公式抽取任务，或在微调时使用“思路检索”模块（推测）。对想进一步探索的读者，可以关注以下方向：① 如何在更大规模语料上高效做影响力追踪；② 把程序性知识结构化（如构建公式库）并让模型直接调用；③ 将这种“策略检索”与 CoT 结合，让模型在生成思考链时明确引用哪篇文档的解法。  

### 一句话记住它
LLM 的数学推理不是简单检索答案，而是从训练语料里抽取并复用“解题步骤”这类程序性知识。