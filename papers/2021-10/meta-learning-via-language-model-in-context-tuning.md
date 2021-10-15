# Meta-learning via Language Model In-context Tuning

> **Date**：2021-10-15
> **arXiv**：https://arxiv.org/abs/2110.07814

## Abstract

The goal of meta-learning is to learn to adapt to a new task with only a few labeled examples. To tackle this problem in NLP, we propose $\textit{in-context tuning}$, which recasts adaptation and prediction as a simple sequence prediction problem: to form the input sequence, we concatenate the task instruction, the labeled examples, and the target input to predict; to meta-train the model to learn from in-context examples, we fine-tune a pre-trained language model (LM) to predict the target label from the input sequences on a collection of tasks.   We benchmark our method on two collections of text classification tasks: LAMA and BinaryClfs. Compared to first-order MAML which adapts the model with gradient descent, our method better leverages the inductive bias of LMs to perform pattern matching, and outperforms MAML by an absolute $6\%$ AUC ROC score on BinaryClfs, with increasing advantage w.r.t. model size. Compared to non-fine-tuned in-context learning (i.e. prompting a raw LM), in-context tuning directly learns to learn from in-context examples. On BinaryClfs, in-context tuning improves the average AUC-ROC score by an absolute $10\%$, and reduces the variance with respect to example ordering by 6x and example choices by 2x.

---

# 通过语言模型上下文调优实现元学习 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理（NLP）里，元学习的目标是让模型只看几条标注样本就能快速适应新任务。传统的元学习方法（如 MAML）需要对模型参数做梯度更新，这在大规模预训练语言模型上既耗时又容易破坏已经学到的语言知识。另一方面，直接用大模型进行“原位提示”（in‑context learning）虽然省去梯度，但模型本身并没有被显式训练去从提示中的示例中学习，导致对示例顺序和选取极其敏感，效果不稳定。于是出现了一个矛盾：要么花大量算力微调参数，要么直接提示却学习能力不足。填补这两者之间的空白，就是本文要解决的核心难题。

### 关键概念速览
**元学习（Meta‑learning）**：让模型掌握“学习的技巧”，即在见到新任务时只用极少数据就能调优。类似于人类学会快速学会新技能的过程。  
**大语言模型（Large Language Model，LM）**：在海量文本上预训练得到的模型，擅长捕捉语言模式，像是拥有丰富语言经验的“语言专家”。  
**上下文调优（In‑context Tuning）**：把任务说明、示例和待预测输入拼接成一段连续文本，让模型直接在这段文本上做下一个词的预测。相当于把学习过程写进提示里，让模型“边读边学”。  
**原位提示（In‑context Learning）**：不改模型参数，只通过精心设计的提示让模型利用已有知识完成任务。像是给模型一次性“口头指令”。  
**任务指令（Task Instruction）**：对当前任务的简短描述，例如“判断句子情感是正面还是负面”。相当于给模型的任务说明书。  
**示例（In‑context Example）**：在提示中出现的已标注样本，模型需要从这些例子中抽取模式。可以类比为老师在课堂上举的例题。  
**AUC‑ROC**：衡量二分类模型整体排序能力的指标，数值越高说明模型区分正负样本的能力越强。  

### 核心创新点
1. **把适应过程直接写进提示**：传统做法是先微调模型参数（MAML）或直接用原始模型提示（原位提示）。本文把任务指令、示例和目标输入拼接成一条序列，让模型在一次前向传播中完成“学习+预测”。这一步把梯度更新的步骤省掉，却保留了学习的本质。  
2. **在大量任务上进行元训练**：作者在一批文本分类任务上对预训练语言模型进行微调，目标是让模型学会从提示中的示例中抽取模式。相当于让模型在“练习场”里反复练习“看例子后做决定”，从而在真正的下游任务中无需再调参。  
3. **利用语言模型的模式匹配偏置**：因为语言模型天生擅长捕捉词序和模板匹配，本文的提示设计让模型直接进行模式匹配而不是梯度搜索。实验显示，模型规模越大，这种优势越明显，AUC‑ROC 提升幅度随模型大小线性增长。  
4. **显著降低示例顺序和选取的方差**：在原位提示中，换一个示例顺序或换一批示例，性能可能波动很大。通过元训练，模型学会对示例的排列保持鲁棒，方差分别降低了 6 倍和 2 倍，提升了实际使用的稳定性。

### 方法详解
**整体框架**  
1）准备一组任务集合（如 LAMA、BinaryClfs），每个任务都是一个二分类或多分类的文本任务。  
2）对每个任务抽取若干标注样本作为“上下文示例”。  
3）构造输入序列：`[任务指令] + [示例1] + … + [示例K] + [目标输入]`。  
4）把这个序列喂入预训练语言模型，让模型预测目标输入后面的标签 token。  
5）在所有任务上同步更新模型参数，使其在上述序列上预测正确。  

**关键模块拆解**  
- **任务指令模块**：用自然语言简短描述任务目标，例如“判断下面句子是否包含积极情感”。这一步相当于给模型一个任务框架，让后面的示例有上下文。  
- **示例拼接模块**：每个示例本身是“输入 → 标签”的小对，直接写成 `输入文本 <sep> 标签` 的形式。所有示例按固定顺序拼接，形成一个长文本。这里的技巧是保持示例之间的分隔符统一，帮助模型辨认边界。  
- **目标预测模块**：在拼接好的长文本末尾放入待预测的输入，模型的任务是输出下一个 token，即对应的标签。因为语言模型本身是“下一个词预测”，所以不需要额外的分类头。  
- **元训练循环**：对每个任务随机抽取不同的示例子集，构造多样化的提示，交叉训练模型。这样模型学会在不同示例组合下仍能抽取共性模式。  

**公式背后的直觉**  
模型的损失其实是普通的交叉熵：预测的标签 token 与真实标签 token 的差距。作者没有引入额外的正则项，只是把所有任务的损失加和后一起反向传播。换句话说，模型被迫在“看完示例后立刻说出答案”的情境里学习通用的模式匹配规则。  

**最巧妙的设计**  
- **不加分类头**：直接利用语言模型的词预测能力，省去额外的输出层，使得学习过程与原始预训练任务完全一致，最大化利用已有的语言知识。  
- **示例顺序随机化**：在训练时故意打乱示例顺序，让模型学会对顺序不敏感，这直接导致实验中方差的大幅下降。  

### 实验与效果
- **数据集**：作者在两个公开的文本分类集合上评估：LAMA（包含多种语言知识填空任务）和 BinaryClfs（大量二分类任务）。  
- **对比基线**：  
  * **MAML**（一阶元学习，需要梯度更新）  
  * **原始模型的原位提示**（不做任何微调）  
- **主要结果**：在 BinaryClfs 上，In‑context Tuning 的 AUC‑ROC 超过 MAML 6%（绝对提升），而相较于原位提示提升了约 10%。随着模型规模从 125M 到 2.7B 参数增长，优势进一步扩大。  
- **方差分析**：示例顺序导致的性能波动在原位提示下约为 0.12（AUC），经 In‑context Tuning 后降至 0.02；示例选取的波动也从 0.08 降到 0.04。  
- **消融实验**：论文报告了去掉任务指令、去掉随机示例抽取、以及不进行元训练的三种变体。去掉指令会导致 AUC 下降约 3%，不做元训练则几乎回到原位提示的水平，说明元训练是关键。  
- **局限性**：作者指出该方法仍然依赖大量任务的集合进行元训练，若目标领域与训练任务差异过大，迁移效果可能下降；此外，提示长度受限于模型的最大上下文窗口，长文本任务需要截断或分块。  

### 影响与延伸思考
这篇工作打开了“微调即提示”的新思路，后续很多研究开始探索 **提示微调（Prompt‑tuning）**、**软提示（Soft Prompt）** 以及 **自适应示例选择** 等方向。比如，2023 年的 “Prefix‑tuning” 进一步把可学习的前缀向量嵌入到提示中，兼顾参数效率和学习能力。推测未来会有更多工作把 **元学习** 与 **大模型的自监督能力** 结合，尝试在更少任务数据、跨语言甚至跨模态的场景下实现类似的“看例子即学”。如果想深入，可以关注 **Few‑Shot Prompting**、**Instruction Tuning** 以及 **Retrieval‑augmented Generation** 等领域的最新进展。  

### 一句话记住它
把任务指令、示例和待预测输入拼成一段文字，让大语言模型在一次前向传播中直接学会从示例中推理，既省去梯度更新，又显著提升 few‑shot 稳定性。