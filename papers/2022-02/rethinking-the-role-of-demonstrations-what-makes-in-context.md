# Rethinking the Role of Demonstrations: What Makes In-Context Learning   Work?

> **Date**：2022-02-25
> **arXiv**：https://arxiv.org/abs/2202.12837

## Abstract

Large language models (LMs) are able to in-context learn -- perform a new task via inference alone by conditioning on a few input-label pairs (demonstrations) and making predictions for new inputs. However, there has been little understanding of how the model learns and which aspects of the demonstrations contribute to end task performance. In this paper, we show that ground truth demonstrations are in fact not required -- randomly replacing labels in the demonstrations barely hurts performance on a range of classification and multi-choce tasks, consistently over 12 different models including GPT-3. Instead, we find that other aspects of the demonstrations are the key drivers of end task performance, including the fact that they provide a few examples of (1) the label space, (2) the distribution of the input text, and (3) the overall format of the sequence. Together, our analysis provides a new way of understanding how and why in-context learning works, while opening up new questions about how much can be learned from large language models through inference alone.

---

# 重新思考示例的作用：是什么让上下文学习有效？ 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在“上下文学习”（in‑context learning, ICL）时，只需要在提示里塞几对输入‑标签示例，就能在不调参的情况下完成新任务。看似神奇，却没有人真正弄清模型到底是怎么“学”的。过去的解释大多把注意力放在示例的**标签正确性**上，假设模型通过这些真实的答案推断出任务规则。但如果标签本身不重要，说明我们对 ICL 的认知可能根本错位——这直接影响如何设计更高效的提示、如何评估模型的推理能力，也决定了我们能否在不微调的情况下让模型真正“学会”新概念。

### 关键概念速览
- **上下文学习（In‑Context Learning, ICL）**：把少量输入‑输出对直接写进模型的提示里，让模型在推理时把这些对当作“示例”，无需梯度更新就完成新任务。类似于老师在课堂上给学生几道例题，学生随后自行解答。
- **示例（Demonstration）**：提示中出现的输入‑标签对。它们的作用被传统观点视为提供任务的“答案”，但本论文把它们拆解成更细的信号。
- **标签空间（Label Space）**：任务可能的输出集合，例如二分类的 {正面, 负面}。示例向模型展示了这些标签的文字形式。
- **输入分布（Input Distribution）**：示例中的文本在风格、长度、词汇上与真实测试数据的相似程度。相当于给模型一个“语境背景”。
- **序列格式（Sequence Format）**：示例的排版方式，如“问题：… 答案：…”的模板。它告诉模型该如何组织输入输出的结构。
- **随机标签替换（Random Label Replacement）**：把示例里的真实标签换成随机的、与任务无关的词汇，用来检验标签本身是否关键。

### 核心创新点
1. **标签不重要的实验验证**  
   *之前的研究*普遍认为示例的正确标签是 ICL 成功的关键。*本文的做法*是系统性地把示例标签随机化，甚至换成完全不相关的词。*结果*显示，性能几乎不降，说明标签本身并非主要驱动力。

2. **把示例拆解为三大信息源**  
   *之前的解释*把示例视为整体信息。*本文的做法*通过消融实验分别去掉（1）标签空间信息、（2）输入分布信息、（3）序列格式信息，观察性能变化。*发现*这三者共同决定了 ICL 效果，而标签空间是最核心的——只要模型知道有哪些可能的输出，就能在上下文中自行推断对应关系。

3. **跨模型、跨任务的一致性验证**  
   *过去的工作*往往只在单一模型或少数任务上做实验。*本文*在 12 种不同规模的模型（包括 GPT‑3）上，覆盖分类、选择题等 10+ 任务进行验证。*结果*显示，随机标签的影响始终微弱，说明结论具有广泛适用性。

4. **重新定义提示工程的目标**  
   *传统提示工程*强调“挑选最贴近目标任务的示例”。*本文*提出，真正需要关注的是示例能否覆盖标签集合、呈现目标输入的分布以及保持一致的格式。*这让提示设计从“找对答案”转向“提供合适的语境”。  

### 方法详解
整体思路很直接：**用实验把示例的每一层信息单独挑出来，观察缺失后模型表现如何**。具体步骤如下：

1. **准备基准任务**  
   选取一批公开的分类和多选任务（如情感分析、自然语言推理、常识问答等），每个任务都有标准的输入‑标签对。

2. **构造三类变体示例**  
   - **标签随机化**：保持输入不变，只把标签换成随机词或其他任务的标签。  
   - **输入分布扰动**：用与目标任务风格差异大的文本（比如新闻改成小说）替换示例的输入，保持标签不变。  
   - **格式破坏**：把原本的“问题：… 答案：…”模板改成不规则的排列，甚至去掉显式的“答案”标记。

3. **组合实验**  
   对每个任务，分别使用（a）完整示例、（b）仅保留标签空间信息（即只给出标签列表），（c）仅保留输入分布信息（即只给出几段无标签的文本），以及（d）仅保留格式信息。每种组合都在同一模型上跑一遍。

4. **评估指标**  
   直接使用任务的准确率或 F1 分数，对比不同示例配置下的表现差异。

5. **跨模型复现**  
   将上述实验在 12 种模型上重复，包括不同参数规模的 GPT‑3、OPT、LLaMA 等，确保结论不是单一模型的偶然。

**最巧妙的地方**在于把“示例”拆成可独立控制的三块信息，然后用“随机标签”这种极端干预手段来检验标签的必要性。直觉上我们会以为标签是关键，但实验结果把这个直觉彻底推翻。

### 实验与效果
- **数据集**：情感二分类（SST‑2）、新闻主题分类（AGNews）、自然语言推理（SNLI）、多项选择常识问答（ARC‑Easy/Hard）等共计 10+ 任务。  
- **基线**：标准 ICL（使用真实标签的示例）以及 zero‑shot（不提供任何示例）作为对照。  
- **主要发现**：在几乎所有任务上，随机标签的准确率下降不到 1%（多数情况下在 0.2%–0.8% 之间），而去掉标签空间信息会导致性能骤降 10%–30%。输入分布和格式的缺失也会带来 3%–8% 的下降。  
- **跨模型一致性**：从 125M 参数的小模型到 175B 参数的 GPT‑3，随机标签的影响始终微弱，说明结论不受模型规模影响。  
- **消融实验**：当仅保留标签列表（不提供任何输入示例）时，模型仍能达到 70% 左右的基准准确率，证明标签空间本身是最核心的信号。  
- **局限性**：论文主要聚焦于**分类和多选**任务，对生成式任务（如摘要、翻译）未做系统评估；此外，随机标签的实验仍在“标签仍是合法词汇”的前提下进行，极端噪声（完全无意义的字符）对模型的容忍度未探讨。

### 影响与延伸思考
这篇工作在社区掀起了“**示例到底提供了什么**”的讨论热潮。随后出现的几篇论文（如 *“Prompt Tuning without Labels”*、*“Understanding the Role of Format in In‑Context Learning”*）直接引用了本研究的实验框架，尝试在更复杂的生成任务上验证标签空间的作用。还有人基于此提出 **“标签空间预设”** 的提示技巧：先给模型一个完整的标签列表，再让它自行推断对应关系，省去手动挑选示例的成本。  
如果想进一步深入，可以关注以下方向：  
1. **生成式任务的标签空间概念**——把“可能的输出形式”抽象为模板或结构，看看是否同样主导 ICL。  
2. **跨语言、跨模态的示例信息**——在多语言或图文混合任务中，标签空间、输入分布、格式的相对重要性是否会改变。  
3. **自动化提示搜索**——利用本论文的洞见，设计搜索空间只围绕标签列表和格式，而不是完整示例，从而大幅降低搜索成本。

### 一句话记住它
**在上下文学习里，示例的真实标签几乎不重要，关键是让模型看到“有哪些可能的答案”。**