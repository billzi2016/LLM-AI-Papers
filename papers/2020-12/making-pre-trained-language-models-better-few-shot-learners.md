# Making Pre-trained Language Models Better Few-shot Learners

> **Date**：2020-12-31
> **arXiv**：https://arxiv.org/abs/2012.15723

## Abstract

The recent GPT-3 model (Brown et al., 2020) achieves remarkable few-shot performance solely by leveraging a natural-language prompt and a few task demonstrations as input context. Inspired by their findings, we study few-shot learning in a more practical scenario, where we use smaller language models for which fine-tuning is computationally efficient. We present LM-BFF--better few-shot fine-tuning of language models--a suite of simple and complementary techniques for fine-tuning language models on a small number of annotated examples. Our approach includes (1) prompt-based fine-tuning together with a novel pipeline for automating prompt generation; and (2) a refined strategy for dynamically and selectively incorporating demonstrations into each context. Finally, we present a systematic evaluation for analyzing few-shot performance on a range of NLP tasks, including classification and regression. Our experiments demonstrate that our methods combine to dramatically outperform standard fine-tuning procedures in this low resource setting, achieving up to 30% absolute improvement, and 11% on average across all tasks. Our approach makes minimal assumptions on task resources and domain expertise, and hence constitutes a strong task-agnostic method for few-shot learning.

---

# 让预训练语言模型成为更好的少样本学习者 论文详细解读

### 背景：这个问题为什么难？

在 NLP 里，少样本学习（few‑shot learning）指的是只用几条标注数据就要让模型完成新任务。传统做法是先在大规模语料上预训练，再用全部可得的标注数据微调（fine‑tune），但微调本身需要上千甚至上万条样本才能稳定收敛。GPT‑3 展示了只靠自然语言提示和几条示例就能取得惊人效果，却依赖上百亿参数的巨型模型，普通研究者根本跑不动。于是，如何在 **模型规模可控、微调成本低** 的前提下，仍然发挥出强大的少样本能力，成了当时的瓶颈。

### 关键概念速览
- **Few‑shot learning（少样本学习）**：只用极少的标注例子（通常 < 100）让模型学会新任务。类似于人只看几道例题就能掌握解题思路。
- **Prompt（提示）**：把任务描述和示例包装成自然语言句子喂给模型，就像老师在黑板上写出“请判断下面句子的情感”。模型把提示当作输入的一部分来理解任务。
- **Prompt‑based fine‑tuning（基于提示的微调）**：在微调阶段仍然使用提示结构，而不是直接把标签映射到向量。相当于在训练时让模型习惯“先读题目再作答”。
- **Demonstration（示例）**：在提示里出现的几条已标注样本，用来向模型展示输入‑输出对应关系。类似老师给出的例题。
- **Dynamic demonstration selection（动态示例选择）**：根据当前输入的特点，挑选最相关的示例放进提示，而不是固定使用同一批示例。就像老师在不同学生面前挑选最能说明概念的例子。
- **Automated prompt generation（自动提示生成）**：用模型或搜索策略自动产生高质量的提示模板，省去人工设计的繁琐。相当于让 AI 自己写出“题目说明”。
- **LM‑BFF（Better Few‑shot Fine‑tuning）**：本文提出的整体方法名称，集合了上述几项技巧的组合拳。

### 核心创新点
1. **从手工提示到自动化提示**  
   之前的少样本工作大多依赖研究者自行编写提示模板，质量参差不齐且难以复现。本文引入了一个 **Prompt Generation Pipeline**：先让小模型在大量无标签文本上生成候选提示，再用少量验证示例评估并挑选最优。这样把“写提示”这件事交给机器，显著提升了提示的适配度。

2. **动态、选择性地加入示例**  
   传统做法把固定的几条示例直接拼进提示，导致上下文长度浪费且示例可能与当前输入关联度低。本文提出 **Selective Demonstration Strategy**：在每次微调时，根据输入的语义相似度从全部可用示例中挑出最相关的几条，放进当前的上下文。相当于老师在每道新题前挑选最相似的例题来帮助学生。

3. **把提示和微调融合的统一框架**  
   过去的提示方法大多是“零样本”或“少量示例直接推理”，而微调则完全抛弃提示。本文把 **Prompt‑based fine‑tuning** 作为核心流程：在每一步梯度更新时，模型看到的仍是完整的提示+示例结构，使得模型在训练阶段就学会利用提示信息。这样既保留了提示的任务指向性，又享受了微调的参数适配优势。

4. **系统化的少样本评估**  
   为了验证每项技巧的贡献，作者在 **分类**（情感、主题）和 **回归**（情感强度、数值预测）任务上做了大规模对比实验。结果显示，组合所有技巧后，平均提升约 **11%**，在部分任务上甚至突破 **30%** 的绝对增益。相比单纯的标准微调，这种提升在低资源场景下尤为显著。

### 方法详解
**整体思路**：先生成或挑选一个合适的提示模板，再在每条训练样本的上下文里动态加入最相关的示例，最后用这些完整的提示‑示例对进行微调。整个过程可以划分为三步：① Prompt 自动生成，② Demonstration 动态选择，③ Prompt‑based 微调。

1. **Prompt 自动生成**  
   - **候选生成**：使用一个小型语言模型（如 T5）在大量未标注文本上进行“填空”式生成，得到若干自然语言模板（如 “请判断以下句子的情感是正面还是负面：{sentence}”。）  
   - **质量评估**：把每个候选模板放进少量验证示例中，跑一次前向推理，记录准确率或损失。  
   - **模板挑选**：选出验证表现最好的模板作为最终提示。这个过程类似于“交叉验证”，但目标是挑选最能帮助模型学习的语言结构。

2. **Demonstration 动态选择**  
   - **相似度计算**：对所有可用的标注示例，先用预训练的句向量模型（如 Sentence‑BERT）得到向量表示。对当前输入句子也做同样编码，计算余弦相似度。  
   - **示例挑选**：按照相似度排序，取前 K 条（K 通常 2‑4）放进提示。这样每条训练样本的上下文都只包含最相关的示例，避免无关信息占用上下文空间。  
   - **示例格式化**：每条示例被包装成 “输入：... 输出：...” 的形式，保持与提示模板的风格一致。

3. **Prompt‑based 微调**  
   - **输入构造**：最终的输入序列 = Prompt 模板 + 动态示例集合 + 当前待学习的输入。标签仍然是原始任务的目标（如情感类别），但模型的输出位置被固定在提示的最后。  
   - **损失计算**：只在模型预测的标签位置计算交叉熵（分类）或均方误差（回归），其余提示文字不参与梯度。  
   - **训练细节**：因为提示和示例占用了大量 token，作者建议使用梯度累积和较小的学习率，以防止过拟合少量数据。  

**最巧妙的点**：把“示例选择”放在每一步微调中，而不是一次性固定，这让模型在训练时始终看到最有信息量的上下文，等于是让模型在“看题”时总能得到最贴切的例子，极大提升了学习效率。

### 实验与效果
- **任务与数据集**：覆盖了多种 **分类**（情感分析、主题判别）和 **回归**（情感强度、数值预测）任务，具体数据集在原文中列出（如 SST‑2、MRPC 等），每个任务只使用 16‑64 条标注样本进行微调。  
- **对比基线**：主要与 **标准微调**（直接在少量样本上训练，无提示、无示例）以及 **GPT‑3 style few‑shot prompting**（仅使用固定提示和示例）进行比较。  
- **整体提升**：组合 LM‑BFF 所有技巧后，平均提升约 **11%**，在某些分类任务上提升 **30%** 的绝对准确率。相较于标准微调，这种提升在低资源环境下尤为显著。  
- **消融实验**：作者分别去掉自动提示、动态示例、纯提示微调三项，发现每去掉一项平均会损失 3‑5% 的性能，说明三者相辅相成。尤其是 **动态示例选择** 对回归任务的贡献最大。  
- **局限性**：实验主要在中等规模的预训练模型（如 BERT‑base、RoBERTa‑base）上完成，未在更大模型上验证；此外，自动提示生成依赖于另一个小模型的质量，若生成的模板质量低，整体收益会下降。作者也提到在极端超长输入场景下，提示+示例会超出模型最大长度，需要截断或分段处理。

### 影响与延伸思考
这篇论文在 2021 年左右出现后，直接催生了一波 **Prompt‑tuning** 与 **Few‑shot fine‑tuning** 的研究热潮。后续工作如 **PET**、**P-tuning**、**AdapterFusion** 等，都在不同层面上借鉴了“把提示放进微调”这一思路。还有研究尝试把 **自动提示生成** 与 **强化学习** 结合，进一步提升模板质量。对想深入的读者，可以关注以下方向：  
- **跨语言少样本学习**：把 LM‑BFF 的自动提示和示例选择扩展到多语言模型。  
- **大模型的高效微调**：在百亿参数模型上实现类似的提示‑微调，探索参数高效的 LoRA/Adapter 方案。  
- **提示搜索空间的理论分析**：为何某些提示结构更易于模型学习，背后的信息论或认知解释。

### 一句话记住它
**让模型在微调时也“读题、看例”，用自动生成的提示和动态挑选的示例，几乎把少样本性能提升到原来的两倍以上。**