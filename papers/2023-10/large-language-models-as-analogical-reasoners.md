# Large Language Models as Analogical Reasoners

> **Date**：2023-10-03
> **arXiv**：https://arxiv.org/abs/2310.01714

## Abstract

Chain-of-thought (CoT) prompting for language models demonstrates impressive performance across reasoning tasks, but typically needs labeled exemplars of the reasoning process. In this work, we introduce a new prompting approach, analogical prompting, designed to automatically guide the reasoning process of large language models. Inspired by analogical reasoning, a cognitive process in which humans draw from relevant past experiences to tackle new problems, our approach prompts language models to self-generate relevant exemplars or knowledge in the context, before proceeding to solve the given problem. This method presents several advantages: it obviates the need for labeling or retrieving exemplars, offering generality and convenience; it can also tailor the generated exemplars and knowledge to each problem, offering adaptability. Experimental results show that our approach outperforms 0-shot CoT and manual few-shot CoT in a variety of reasoning tasks, including math problem solving in GSM8K and MATH, code generation in Codeforces, and other reasoning tasks in BIG-Bench.

---

# 大语言模型作为类比推理者 论文详细解读

### 背景：这个问题为什么难？

在让语言模型解决数学、代码或常识推理等需要多步思考的任务时，直接让模型“一口气”输出答案往往出错率很高。出现的“思维链”（Chain‑of‑Thought，CoT）技巧通过让模型先写出推理步骤，显著提升了准确率，但它依赖人工挑选的示例或标注好的思考过程。收集、维护这些示例成本大，且在新领域或新题型上往往不适用。于是，如何在不准备任何标注示例的情况下，引导模型自行展开可靠的推理，成为了一个急需突破的瓶颈。

### 关键概念速览
- **思维链（CoT）**：让模型在给出最终答案前先把每一步推理写出来，类似人解题时在草稿纸上列步骤，帮助模型保持逻辑连贯性。  
- **类比推理**：人类通过把新问题映射到过去相似的情境来寻找解法，就像把一个陌生的拼图和已经拼好的类似拼图对照。  
- **类比提示（Analogical Prompting）**：一种新型提示方式，先让模型自行生成与当前任务相似的“过去案例”或相关知识，再基于这些自创的例子继续求解。  
- **零样本（Zero‑Shot）**：模型在没有任何示例的前提下直接完成任务，考验的是模型的通用推理能力。  
- **少样本（Few‑Shot）**：给模型提供少量示例后再让它完成任务，常用于提升模型在特定任务上的表现。  
- **自生成示例**：模型在提示下自行创造出与目标任务相似的题目和解答，这些示例并非从外部检索，而是模型内部“想象”出来的。  
- **适配性（Adaptivity）**：指示例或知识能够根据每一道具体题目动态调整，而不是一次性固定的模板。

### 核心创新点
1. **从手工示例到模型自创示例**  
   之前的 CoT 需要研究者提前准备好标注好的思考过程或从数据库中检索相似案例。本文改为在提示中让模型先自行生成一个或多个相关的“类比案例”，再基于这些案例展开推理。这样省去了标注成本，也让每道题都有专属的参考例子。

2. **一次性提示完成两阶段**  
   传统做法往往需要两条提示：一条让模型检索/提供示例，另一条让它解题。这里把两步合并成一条“类比提示”，模型在同一次生成过程中先输出类比案例，随后直接在同一上下文里给出答案，流程更简洁。

3. **对每个问题的定制化类比**  
   过去的少样本方法使用固定的 few‑shot 示例，面对不同题目时可能不匹配。类比提示让模型根据当前题目的关键词和结构自行构造最贴切的类比，从而提升适配性和推理质量。

4. **在多种任务上统一验证**  
   作者把同一套类比提示直接搬到数学（GSM8K、MATH）、代码（Codeforces）以及 BIG‑Bench 的多种推理子任务上，展示了方法的跨任务通用性。相比 0‑shot CoT 和手工 few‑shot CoT，整体表现都有提升。

### 方法详解
整体思路可以拆成三步：**（1）构造类比提示 →（2）模型自生成类比案例 →（3）基于案例完成原任务**。整个过程只需要在一次调用中提供一条精心设计的提示文本，模型会在同一输出序列里完成前两步，然后继续给出答案。

**1. 类比提示的设计**  
提示的核心句式类似：“先想一个和下面问题相似的例子并解答，然后再解下面的问题”。这里的“相似”没有硬性定义，模型会依据上下文自行判断。为了防止模型直接跳到答案，提示里加入了“先写出类比案例的完整推理过程”之类的约束。

**2. 自生成类比案例**  
模型在接收到提示后，首先生成一段文字，形式上与目标任务相同（比如一道数学题的叙述），随后给出该类比题的解题步骤。可以把这一步想象成模型在自己的记忆库里翻找过去学过的类似题目，然后把它们重新写出来。

**3. 迁移到原任务**  
类比案例结束后，提示已经暗示模型进入“现在轮到你了”。模型接着读取原始问题，利用刚才写的类比推理路径作为思考模板，继续展开自己的推理链，最终输出答案。

**关键实现细节**  
- **一次性生成**：模型的输出被划分为两段，第一段以“[Analogical Example]”标记开头，第二段以“[Solution]”标记，便于后处理。  
- **温度调节**：在生成类比案例时使用稍高的采样温度，以鼓励多样化的例子；在正式解题时降低温度，确保答案的确定性。  
- **长度控制**：提示里加入了“类比案例不超过 X 行”之类的字数限制，防止模型在类比阶段耗尽预算。  
- **反直觉点**：让模型先“自找例子”听起来像是增加了工作量，实际上实验表明这一步显著提升了后续推理的准确率，因为模型在生成例子时已经激活了相关的知识路径。

### 实验与效果
- **测试任务**：数学推理（GSM8K、MATH）、代码生成（Codeforces 竞赛题）、以及 BIG‑Bench 中的逻辑、常识等多项推理子任务。  
- **对比基线**：0‑shot CoT、手工 few‑shot CoT（使用人工挑选的 2–3 条示例），以及直接的无提示零样本。  
- **结果概述**：论文声称在所有评测上类比提示均超过 0‑shot CoT，且在多数任务上超过手工 few‑shot CoT，提升幅度从几百分点到两位数不等。  
- **消融实验**：作者分别去掉类比案例生成、降低温度、或去掉长度限制，发现没有类比阶段时性能回落到接近 0‑shot CoT，说明自生成示例是关键驱动因素。  
- **局限性**：类比提示对模型规模有一定依赖——在 7B 参数以下的模型上自生成的类比质量下降，导致整体收益减弱。作者也提到在极端专业领域（如高等数学证明）类比案例可能不够精准，需要进一步研究。

### 影响与延伸思考
这篇工作把“类比”这一人类认知机制直接搬进了大语言模型的提示工程，打开了“让模型自己找例子”这一新思路。随后的研究开始探索更系统的自检索/自生成框架，例如把检索式提示和类比提示结合，或在多轮对话中让模型不断迭代类比案例。对想进一步深入的读者，可以关注以下方向：  
- **自监督类比数据构建**：利用大规模未标注文本自动生成类比对，训练专门的类比生成模块。  
- **跨模态类比**：把文本类比扩展到图像、代码等多模态情境，看看模型能否在不同模态之间做类比。  
- **解释性与可控性**：研究如何让类比案例更可解释、可编辑，从而让用户在生成过程中加入 domain knowledge。  

### 一句话记住它
让模型先自己想出相似的例子再解题，省去人工示例，推理更稳、更通用。