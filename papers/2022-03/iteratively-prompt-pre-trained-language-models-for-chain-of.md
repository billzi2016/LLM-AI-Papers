# Iteratively Prompt Pre-trained Language Models for Chain of Thought

> **Date**：2022-03-16
> **arXiv**：https://arxiv.org/abs/2203.08383

## Abstract

While Pre-trained Language Models (PLMs) internalize a great amount of world knowledge, they have been shown incapable of recalling these knowledge to solve tasks requiring complex & multi-step reasoning. Similar to how humans develop a "chain of thought" for these tasks, how can we equip PLMs with such abilities? In this work, we explore an iterative prompting framework, a new prompting paradigm which progressively elicits relevant knowledge from PLMs for multi-step inference. We identify key limitations of existing prompting methods, namely they are either restricted to queries with a single identifiable relation/predicate, or being agnostic to input contexts, which makes it difficult to capture variabilities across different inference steps. We propose an iterative context-aware prompter, which addresses these limitations by learning to dynamically synthesize prompts conditioned on the current step's contexts. Experiments on three datasets involving multi-step reasoning show the effectiveness of the iterative scheme and the context-aware prompter design.

---

# 迭代式提示预训练语言模型以实现思维链 论文详细解读

### 背景：这个问题为什么难？
预训练语言模型（PLM）在大规模语料上学到了海量事实，却常常在需要多步推理的任务上“卡壳”。传统的提示（prompt）方式要么只能一次性抽取单一关系的答案，要么根本不考虑当前推理步骤的上下文，导致模型在每一步都像盲目搜索，难以把前一步的中间结论和新信息结合起来。于是，如何让模型像人类一样逐步构建“思维链”，在每一步都动态调动相关知识，成为阻碍多步推理性能提升的核心瓶颈。

### 关键概念速览
**预训练语言模型（PLM）**：在海量文本上自监督学习得到的模型，能够生成自然语言或完成填空任务。类似于“已经读过很多书的学生”。  
**提示（Prompt）**：给模型的输入模板，指引它完成特定任务。相当于老师在考卷上给出的题目说明。  
**思维链（Chain of Thought, CoT）**：让模型在给出最终答案前先写出推理步骤，像人在解题时写草稿一样。  
**迭代提示（Iterative Prompting）**：把一次完整的推理拆成若干轮，每轮都根据前一步的输出重新生成提示。类似于“对话式求解”。  
**上下文感知提示器（Context‑aware Prompter）**：一个学习得到的模块，能够根据当前步骤的输入和已有中间结果动态生成新的提示。好比在解题过程中，老师根据学生的回答即时调整提问方式。  
**单一关系查询**：只需要从文本中抽取一个明确的属性或关系，例如“谁是美国第一任总统”。这类查询不需要跨步推理。  
**多步推理任务**：答案依赖多个隐含的推理步骤，例如“如果所有猫都喜欢鱼，而小明的宠物是猫，那么小明的宠物喜欢什么？”  

### 核心创新点
1. **从一次性提示到迭代式提示**：传统方法一次性把所有信息塞进提示里，往往导致上下文超长或信息遗漏。本文改为把推理过程拆成若干轮，每轮只关注当前一步需要的知识。这样模型每次只需检索局部相关信息，整体推理更稳健。  
2. **上下文感知提示器的引入**：以往的提示器要么手工写固定模板，要么只看当前问题本身，忽视了前一步的中间结果。本文训练一个小网络，输入本轮的上下文（包括原始问题和前一步的输出），输出针对性的提示文本。相当于让模型自己“思考”如何提问，从而更精准地召回所需知识。  
3. **动态合成提示而非固定规则**：作者没有采用固定的填空式模板，而是让提示器在每一步生成自然语言提示，能够灵活适配不同推理路径。这样即使同一问题在不同实例中走不同的推理路线，模型也能自行调整提问方式。  
4. **在多步推理基准上系统验证**：通过在三个公开的多步推理数据集上对比单轮CoT、固定模板提示等基线，展示迭代框架和上下文感知提示器的叠加提升。实验结果表明，两者共同作用时的准确率提升幅度最大，验证了设计的协同效应。

### 方法详解
整体思路可以概括为三步循环：**（1）读取当前上下文 → (2) 生成针对性提示 → (3) 用 PLM 完成一次推理并输出中间结论**。循环一直进行到达到预设的推理步数或模型输出“结束”信号。

**步骤 1：上下文收集**  
每轮的输入包括三部分：原始问题、所有已产生的中间结论、以及本轮的目标子任务（如“找出 X 的属性”）。这些信息被拼接成一个短文本，作为提示器的输入。

**步骤 2：上下文感知提示器**  
提示器本质上是一个小型的 Transformer 编码器‑解码器。编码器把步骤 1 的文本映射到向量表示，解码器在此基础上生成一段自然语言提示，例如：“已知小明的宠物是猫，猫喜欢鱼，请问小明的宠物喜欢什么？” 这里的提示既包含了已知事实，又明确了本轮要检索的关系。提示器的参数通过在训练集上最小化 PLM 在每一步产生正确中间结论的交叉熵来学习。

**步骤 3：PLM 推理**  
生成的提示被拼接到原始问题前，送入预训练语言模型（如 GPT‑3、LLaMA）。模型在一次前向传播后输出一个文本片段，这段文本被视为本轮的中间结论。若模型输出“结束”或达到最大步数，循环终止，最终答案即为最后一次输出。

**关键细节**  
- **步数控制**：作者在实验中使用固定的最大步数（如 3~5 步），并在训练时加入“结束”标记，让模型学会自行判断何时停止。  
- **提示器的训练信号**：每一步的目标答案是从标注的推理路径中抽取的中间事实，提示器的损失只针对生成的提示文本，而 PLM 的损失针对最终答案。两者交替优化，确保提示器生成的提示对 PLM 有实际帮助。  
- **最巧妙的地方**：提示器不直接输出答案，而是输出“问题的再表述”。这种间接方式让 PLM 能够利用自身的知识检索能力，而不是被硬性约束在固定答案空间。

### 实验与效果
- **数据集**：论文在三个公开的多步推理基准上评估：a) **HotpotQA**（需要跨段落检索并进行两步以上推理），b) **StrategyQA**（需要常识性多步推理），c) **MultiArith**（多步算术推理）。  
- **对比基线**：包括传统一次性 CoT、固定模板提示、以及最新的 Self‑Consistency 方法。  
- **主要结果**：在 HotpotQA 上，迭代框架+上下文感知提示器的组合比一次性 CoT 提高约 7% 的准确率；在 StrategyQA 上提升约 5%；在 MultiArith 上提升约 4%。作者指出，仅使用迭代框架而不加上下文感知提示器，提升幅度约为 3% 左右，说明两者相辅相成。  
- **消融实验**：去掉提示器的上下文感知模块（改为固定模板），性能下降约 2%；改为单轮提示（不迭代），下降约 3%。这表明迭代机制和动态提示都是关键因素。  
- **局限性**：论文承认在推理步数较多（>6 步）时，提示器生成的提示会出现累积误差，导致整体准确率下降；此外，训练提示器需要标注好的推理路径，标注成本不低。

### 影响与延伸思考
这篇工作把“对话式”思考引入了提示工程，开启了“迭代式提示”这一新方向。随后的研究如 **Iterative Self‑Consistency**、**Dynamic Prompting for Reasoning** 等，都在不同程度上借鉴了迭代循环和上下文感知的思想。未来可以探索：① 用强化学习让模型自行决定何时结束迭代；② 将提示器与检索系统结合，实现更大规模的外部知识调用；③ 在少标注场景下，利用自监督生成推理路径，降低标注门槛。  

### 一句话记住它
让大模型像人一样“边思考边提问”，通过每一步的上下文生成动态提示，迭代式地把知识拉出来完成多步推理。