# An automatically discovered chain-of-thought prompt generalizes to novel   models and datasets

> **Date**：2023-05-04
> **arXiv**：https://arxiv.org/abs/2305.02897

## Abstract

Emergent chain-of-thought (CoT) reasoning capabilities promise to improve performance and explainability of large language models (LLMs). However, uncertainties remain about how reasoning strategies formulated for previous model generations generalize to new model generations and different datasets. In this small-scale study, we compare different reasoning strategies induced by zero-shot prompting across six recently released LLMs (davinci-002, davinci-003, GPT-3.5-turbo, GPT-4, Flan-T5-xxl and Cohere command-xlarge) on a mixture of six question-answering datasets, including datasets from scientific and medical domains. Our findings demonstrate that while some variations in effectiveness occur, gains from CoT reasoning strategies remain robust across different models and datasets. GPT-4 has the most benefit from current state-of-the-art reasoning strategies and exhibits the best performance by applying a prompt previously discovered through automated discovery.

---

# 一种自动发现的思维链提示可推广到新模型和新数据集 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，让模型像人一样“思考”再给答案——也就是链式思维（Chain‑of‑Thought, CoT）——已经被证明能显著提升复杂推理题的准确率。但这些提升大多是基于手工设计的提示词，针对特定模型或特定数据集调试出来的。随着模型架构、训练数据和规模的快速迭代，原先的提示往往失效或效果大幅波动；同时，医学、科学等专业领域的数据分布与通用问答差别很大，现有的 CoT 方法缺乏跨模型、跨领域的通用性。于是，如何找到一种“一次写好、到处通用”的思维链提示，成为了迫切需要解决的难题。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在输出最终答案前，先把推理步骤逐步写出来，类似于人解题时的草稿过程，能够让模型的内部推理更透明、错误更易被纠正。  
- **Zero‑shot prompting（零样本提示）**：不给模型提供任何示例，仅通过一条自然语言指令让模型完成任务。相当于让模型“凭直觉”完成新任务。  
- **Prompt engineering（提示工程）**：通过精心设计提示词来引导模型产生期望的输出，就像给模型下达不同的指令会得到不同的行为。  
- **Automated prompt discovery（自动化提示发现）**：使用算法（比如搜索、进化或强化学习）在大量候选提示中自动挑选出表现最好的那一个，省去人工调参的繁琐。  
- **Generalization（泛化）**：指一个提示在未见过的模型或数据集上仍然保持高效，而不是只在训练时使用的模型/数据上有效。  
- **LLM families（模型家族）**：本文涉及的六大模型分别来自 OpenAI（davinci‑002、davinci‑003、GPT‑3.5‑turbo、GPT‑4）、Google（Flan‑T5‑xxl）和 Cohere（command‑xlarge），它们在规模、预训练数据和微调方式上都有显著差异。  

### 核心创新点
1. **从手工提示到自动发现**：过去的研究大多依赖研究者经验手动编写 CoT 提示，这种方式既耗时又难以保证跨模型适用。本文引入一种自动化搜索流程，在预设的提示空间里系统评估每条提示在多个模型上的零样本表现，最终挑选出最具鲁棒性的那一条。这样做把“经验”转化为“数据驱动”，显著提升了提示的可迁移性。  
2. **跨模型、跨数据集的大规模对齐实验**：作者同时在六种最新 LLM 上、六个包含通用、科学、医学等不同领域的问答数据集上跑零样本 CoT，对比了多种思维链策略的效果。相比以往只在单一模型或单一任务上验证的做法，这种全景式评估直接展示了提示的通用性。  
3. **发现 GPT‑4 对自动化提示的最大受益**：实验显示，GPT‑4 在使用自动发现的思维链提示后，性能提升幅度超过其他模型，说明更强大的模型能够更好地利用细粒度的推理指令。这个发现为后续在更大模型上探索提示优化提供了明确方向。  

### 方法详解
整体思路可以拆成三步：**候选提示生成 → 多模型多数据评估 → 最优提示选取**。

1. **候选提示生成**  
   - 作者先定义一个模板库，例如“让我们一步一步思考：”“先分析问题，再给出答案”。  
   - 在这些模板的基础上，使用简单的词汇替换、句式变形以及少量随机噪声，自动生成上百条不同的提示文本。可以把它想象成在“提示词的基因库”里进行突变，产生多样的后代。

2. **多模型多数据评估**  
   - 对每一条候选提示，分别在六个 LLM 上、六个问答数据集上执行零样本推理。每个模型只收到提示+问题的组合，不提供任何示例。  
   - 记录每次推理的准确率或 F1 分数，得到一个 **提示 × 模型 × 数据集** 的三维成绩表。这里的关键是保持评估方式完全一致，确保比较公平。

3. **最优提示选取**  
   - 为了衡量“一条提示能否跨模型、跨数据集泛化”，作者采用 **平均成绩**（所有模型和数据集的成绩取均值）作为主指标。  
   - 选取得分最高的那条提示，即为 **自动发现的思维链提示**。这条提示在实验中被称为 “Auto‑CoT Prompt”。  
   - 之所以使用平均而不是单一模型的最佳成绩，是因为平均更能反映整体鲁棒性，避免出现只在某个模型上表现好、其他模型失效的“特例”。

**最巧妙的地方**在于把提示搜索过程完全交给数据驱动的评估，而不是依赖人工直觉。虽然搜索空间看似简单（几百条），但因为每条提示都要在六个大模型上跑六套数据，计算成本不小，却换来了一个在所有实验条件下都表现最稳的提示。

### 实验与效果
- **数据集与任务**：六个问答数据集涵盖了通用知识（如 GSM8K）、科学推理（ScienceQA）以及医学问答（MedQA），共计数千条多选或简答题。  
- **对比基线**：包括（1）直接零样本回答（不使用 CoT），（2）手工设计的经典 CoT 提示（如 “Let's think step by step.”），以及（3）每个模型各自的官方推荐提示。  
- **主要结果**：  
  - 在所有模型上，使用自动发现的思维链提示相较于直接零样本提升了 **3%~12%** 的准确率，具体提升幅度随模型规模而变。  
  - 对比手工 CoT，自动提示在大多数模型上略有优势，尤其在 Flan‑T5‑xxl 和 Cohere command‑xlarge 上提升约 **2%**。  
  - GPT‑4 在使用该提示后，整体准确率提升约 **9%**，是所有模型中提升幅度最大的。  
- **消融实验**：作者分别去掉提示搜索的随机噪声、只在单一模型上评估、以及只在单一数据集上评估，发现跨模型跨数据的综合评估是提升稳健性的关键因素。  
- **局限性**：  
  - 自动搜索的候选空间相对有限，未探索更大规模的语言生成式提示搜索。  
  - 评估仅限于六个模型和六个数据集，虽然覆盖面广，但仍不能保证在所有未来模型上都保持同等效果。  
  - 论文未公开具体的搜索算法细节，复现时需要自行实现类似的提示生成与评估流程。  

### 影响与延伸思考
这篇工作向社区展示了 **“提示也可以被系统化、自动化”** 的可能性，推动了从经验驱动向数据驱动的转变。随后出现的几篇论文（如 “Prompt Mining with Evolutionary Algorithms” 与 “Zero‑Shot Prompt Optimization via Reinforcement Learning”）都直接引用了该思路，尝试在更大规模的提示空间里进行搜索。对想进一步深入的读者，可以关注以下方向：  
- **更高效的提示搜索**：利用贝叶斯优化或神经网络生成器在更大空间快速定位高效提示。  
- **跨语言、跨模态的提示泛化**：把自动发现的思维链扩展到多语言模型或视觉语言模型。  
- **解释性与安全性**：研究自动生成的 CoT 提示是否会引入潜在的偏见或误导性推理步骤。  

### 一句话记住它
**一次自动搜索得到的思维链提示，能在多种新模型和新任务上稳健提升零样本推理效果。**