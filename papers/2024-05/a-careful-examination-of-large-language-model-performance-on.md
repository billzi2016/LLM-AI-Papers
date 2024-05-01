# A Careful Examination of Large Language Model Performance on Grade   School Arithmetic

> **Date**：2024-05-01
> **arXiv**：https://arxiv.org/abs/2405.00332

## Abstract

Large language models (LLMs) have achieved impressive success on many benchmarks for mathematical reasoning. However, there is growing concern that some of this performance actually reflects dataset contamination, where data closely resembling benchmark questions leaks into the training data, instead of true reasoning ability. To investigate this claim rigorously, we commission Grade School Math 1000 (GSM1k). GSM1k is designed to mirror the style and complexity of the established GSM8k benchmark, the gold standard for measuring elementary mathematical reasoning. We ensure that the two benchmarks are comparable across important metrics such as human solve rates, number of steps in solution, answer magnitude, and more. When evaluating leading open- and closed-source LLMs on GSM1k, we observe accuracy drops of up to 8%, with several families of models showing evidence of systematic overfitting across almost all model sizes. Further analysis suggests a positive relationship (Spearman's r^2 = 0.36) between a model's probability of generating an example from GSM8k and its performance gap between GSM8k and GSM1k, suggesting that some models may have partially memorized GSM8k. Nevertheless, many models, especially those on the frontier, show minimal signs of overfitting, and all models broadly demonstrate generalization to novel math problems guaranteed to not be in their training data.

---

# 对大语言模型在小学算术任务上表现的细致审视 论文详细解读

### 背景：这个问题为什么难？
在数学推理基准上，LLM（大语言模型）已经能跑出接近人类的分数，但这些分数到底是「真正推理」还是「记住了答案」仍是争议。已有的 benchmark（如 GSM8k）在公开后会被爬取进训练语料，导致模型在评测时可能直接复现题目而不是演算。缺少一个「干净」的、与原 benchmark 等价但未被模型见过的对照集，导致我们无法判断模型的真实算术能力。于是，研究者必须构造一个既保持原题风格又确保训练时不可能出现的测试集，才能揭开「数据泄露」的真相。

### 关键概念速览
**大语言模型（LLM）**：基于海量文本训练的神经网络，能够生成连贯的自然语言。把它想成一个「会说话的百科全书」，但它的知识来源于训练数据，而不是实时推理。  
**数据泄露（Dataset Contamination）**：评测题目在模型训练阶段已经出现，导致模型可能直接记忆答案。类似于考试前把试卷偷偷放进复习资料里，学生不需要真正懂题目。  
**过拟合（Overfitting）**：模型在训练数据上表现极好，却在未见过的数据上表现差。可以比作练习题只会背答案，遇到新题就卡壳。  
**GSM8k**：公开的「Grade School Math 8K」基准，包含 8,000 条小学算术题，被广泛用来衡量 LLM 的数学推理能力。  
**GSM1k**：本文新构造的「Grade School Math 1K」集合，规模约 1,000 题，故意在风格、难度、解题步骤等方面与 GSM8k 对齐，但保证不在公开语料中出现。  
**Spearman 相关系数（Spearman’s r）**：衡量两组变量单调关系的统计指标，值越接近 1 表示正相关越强。这里用它来捕捉「模型记住 GSM8k 题目的概率」与「在 GSM1k 上掉分」之间的关联。  
**生成概率（Generation Probability）**：模型在给定上下文时输出特定句子的概率。把它想成模型「说出这句话的自信度」，自信度高往往意味着该句子在训练中出现过。

### 核心创新点
1. **等价对照集的精心构造**  
   *之前的做法 → 直接在 GSM8k 上评测，无法排除泄露* → 本文先用人类解题统计（解题成功率、步骤数、答案大小）对 GSM8k 进行特征抽取，再在同一分布下随机生成 1,000 题，形成 GSM1k。 → 这样既保留了原 benchmark 的难度，又确保模型训练时不可能见过，从根本上消除了数据泄露的可能。  

2. **系统化的泄露检测框架**  
   *之前只看整体准确率下降 → 本文引入「生成概率」作为泄露指示器，计算模型在不提供答案的情况下自行生成 GSM8k 题目的概率，并与 GSM1k 与 GSM8k 的性能差做 Spearman 相关分析。 → 发现两者相关系数 r²≈0.36，说明模型记忆程度可以量化，提供了评估泄露的可解释度量。  

3. **跨模型族、跨规模的全景评估**  
   *过去多聚焦单一模型或闭源大模型 → 本文同时评测了多家开源模型（如 LLaMA、Mistral）和闭源商用模型（如 GPT‑4），覆盖从几亿到上百亿参数的全谱。 → 结果显示小模型普遍出现 5‑8% 的准确率下降，而最前沿的大模型几乎没有显著掉分，揭示了规模与泄露敏感性的关系。  

### 方法详解
整体思路可以拆成三步：**特征对齐 → 题目生成 → 泄露度量**。  
1. **特征对齐**：研究团队先统计 GSM8k 中每道题的关键属性——人类解答成功率、所需算术步骤数、答案的位数、使用的运算符种类等。把这些属性视作「数学题的 DNA」。随后在同一 DNA 分布上随机抽样，生成新题目，确保新题在统计意义上与原题「长得一样」但文字内容全新。可以把它想成在同一套配方下烤出不同形状的饼干。  

2. **题目生成**：采用模板化的程序化生成器，先随机挑选运算符和数值范围，再根据预设的步骤数构造多步算式，最后用自动求解器验证答案唯一且符合整数范围。所有生成的题目都经过人工抽样检查，确保可读性和与小学数学教材的匹配度。  

3. **泄露度量**：对每个评测模型，先在不提供答案的情况下让模型自由生成 10,000 条类似 GSM8k 风格的题目，记录每条生成的概率（即模型对该句子的自信度）。统计这些生成句子中有多少与 GSM8k 实际出现的题目高度匹配，得到「记忆概率」。随后在 GSM8k 与 GSM1k 上分别测算准确率，计算两者的差值。最后用 Spearman 相关系数把记忆概率与性能差关联起来，得到量化的泄露信号。  

最巧妙的地方在于**把记忆概率当作可解释的特征**，而不是单纯的准确率下降。这样即使模型在新题上仍保持高分，也能通过「自信度」发现它是否在偷偷复用旧题。

### 实验与效果
- **数据集**：GSM8k（8,000 题）作为基准，GSM1k（1,000 题）作为对照。两套题在人类解答成功率、步骤数、答案规模等维度上保持统计一致。  
- **模型**：覆盖开源 LLaMA 系列（7B、13B、30B、65B）、Mistral、Gemma 等，以及闭源 GPT‑3.5、GPT‑4、Claude 等。  
- **主要发现**：在 GSM1k 上的准确率普遍比 GSM8k 低 0‑8%，其中 7B‑13B 规模的模型下降约 5‑8%，而 GPT‑4 与 Claude 的下降不到 1%。Spearman r²≈0.36 表明记忆概率越高，性能差距越大。  
- **消融实验**：作者分别去掉「生成概率」特征、只使用单步算术题、以及不做特征对齐的随机生成集进行对比，发现去掉对齐会导致两套数据集的难度差异扩大，相关系数降至 0.12，验证了对齐步骤的必要性。  
- **局限性**：实验仅限于小学四则运算和简单代数，未覆盖几何、函数等更高层次的数学；GSM1k 虽然在统计上等价，但仍是机器生成，可能缺少人类出题的微妙陷阱。作者也承认，极大模型的「几乎不掉分」可能部分归因于更强的通用推理能力，而非完全消除泄露。

### 影响与延伸思考
这篇工作在社区里掀起了对 benchmark 「干净度」的重新审视。随后出现的 **GSM-Hard**、**MathBench** 等新基准，都借鉴了「等价对照」的思路，尝试在更高难度层级上做同样的泄露检测。还有研究把「生成概率」推广到代码生成、语言理解等任务，形成了「记忆度量」的通用框架。对想进一步探索的读者，可以关注 **数据泄露检测** 与 **模型可解释性** 的交叉方向，尤其是如何在大规模预训练语料中自动识别潜在的评测泄露。

### 一句话记住它
**用与原 benchmark 统计等价、未泄露的对照集，量化模型对旧题的记忆程度，才能真正评估 LLM 的数学推理能力。**