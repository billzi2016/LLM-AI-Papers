# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

> **Date**：2022-01-28
> **arXiv**：https://arxiv.org/abs/2201.11903

## Abstract

We explore how generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain of thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.

---

# 思维链提示激发大语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，模型往往只能靠一次性提示直接给出答案。面对需要多步推理的数学、常识或符号题目，这种“一锤子”式的输出常常出错，因为模型没有显式的思考过程可供检查。传统的微调或检索增强方法虽然能提升一定水平，但都需要大量标注数据或额外的检索模块，成本高且难以推广。于是，如何在不改动模型本身、只靠提示就让模型“思考”起来，成为了迫切的研究需求。

### 关键概念速览
- **Chain of Thought（思维链）**：让模型在给出最终答案前，先把推理步骤写出来，类似人做数学题时的草稿，能够让错误在中间环节被发现和纠正。  
- **Prompting（提示）**：向模型提供一段文字作为输入，引导模型产生期望的输出。这里的提示包括了示例题目和对应的思维链。  
- **Few‑shot 示例**：在提示中加入少量（通常 4–8 条）已经标注好的例子，模型会模仿这些例子的格式和思路。  
- **Zero‑shot 与 Few‑shot**：Zero‑shot 指不提供任何示例直接提问，Few‑shot 则提供少量示例帮助模型学习任务格式。  
- **GSM8K**：一个广泛使用的数学文字题基准，包含 8,000 条需要多步推理的题目。  
- **Verifier（验证器）**：在一些对比实验中，先让模型生成答案，再用另一个模型检查答案是否合理，这是一种后处理手段。  
- **Scale（模型规模）**：指模型的参数数量，论文中重点关注 540 B 参数的大模型，规模越大往往推理能力越强。  

### 核心创新点
1. **思维链示例的提示方式 → 在 Few‑shot 提示中加入完整的推理步骤** → 让模型把“写草稿”当成生成任务，从而在内部形成多步推理的链条，显著提升了复杂任务的准确率。  
2. **规模驱动的自然出现 → 只要模型足够大（如 540 B），不需要额外的微调或专门的推理模块** → 证明了大模型本身已经蕴含了推理潜能，只是需要合适的提示来激活。  
3. **极简示例数量 → 只用 8 条思维链示例就能在 GSM8K 上达到 SOTA** → 与传统需要上千条标注数据的微调相比，成本下降了几个数量级。  
4. **统一的提示框架 → 同一套思维链提示可以同时提升算术、常识和符号推理三个方向的表现** → 展示了方法的通用性，而不是针对单一任务的特化技巧。

### 方法详解
整体思路可以概括为三步：准备示例、构造提示、让模型生成思维链并得到答案。

1. **准备示例**  
   研究者从每个任务的训练集里挑选出几条典型题目，并手工写出完整的推理过程。每条示例的格式是：  
   ```
   问题：<题目文本>
   思考过程：<逐步推理，每一步都用自然语言描述>
   答案：<最终答案>
   ```  
   这里的“思考过程”相当于草稿，必须清晰、一步步展开。

2. **构造 Few‑shot 提示**  
   将上述示例按顺序拼接在一起，形成一个长文本。随后在提示的末尾加入目标问题，只写“问题：<新题目>”。模型看到前面的示例后，会自然模仿其结构，先输出“思考过程”，再给出“答案”。  
   类比：就像老师在黑板上演示几道例题的解题步骤，学生看到新题后会先把步骤写出来再写答案。

3. **模型生成**  
   将完整提示喂入大语言模型，模型一次性生成后续的文字。因为提示已经明确要求“思考过程”，模型倾向于先输出推理步骤，再给出答案。生成的文本可以直接解析出答案，也可以在后处理阶段只取答案部分。

**关键细节**  
- **示例数量**：实验发现 8 条示例已经足够激活思维链，更多示例提升有限。  
- **模型规模**：在 540 B 参数模型上效果最显著，较小模型（如 1 B）几乎没有提升，这说明规模是触发思维链的必要条件。  
- **不需要额外的训练**：整个过程完全是“零微调”，只改变输入文本的组织方式。  
- **最巧妙的地方**：作者没有改动模型内部结构，也没有加入外部检索或符号求解器，仅靠提示的格式让模型自行产生多步推理，这种“软激活”方式在当时是非常反直觉的。

### 实验与效果
- **任务与数据集**：论文在三大类任务上做评测：算术推理（如 GSM8K、MultiArith）、常识推理（如 CommonsenseQA）以及符号推理（如 Last Letter Concatenation）。  
- **基线对比**：与普通 Few‑shot 提示（直接输出答案）相比，思维链提示在 GSM8K 上提升约 20% 的准确率，达到 92% 左右，超过了使用专门微调并配合验证器的 GPT‑3（约 89%）。在常识任务上也有 5–10% 的提升。  
- **消融实验**：作者去掉思考过程的文字，只保留问题和答案，性能回落到普通提示水平；再把示例数量从 8 降到 0（即 Zero‑shot），提升几乎消失，说明示例和思维链是关键因素。  
- **局限性**：思维链在小模型上几乎不起作用；对极其长的推理链（超过模型上下文长度）仍会出现截断或遗漏；论文未对生成的思考过程的真实性进行系统评估，可能出现“幻觉式”推理。  

### 影响与延伸思考
这篇工作开启了“提示工程”中的一个新方向：通过让模型显式写出思考过程来提升推理能力。随后大量研究围绕 **Self‑Consistency**（让模型多次生成思维链取多数票）、**Tree‑of‑Thought**（把思维链组织成树形搜索）以及 **Program‑aided Language Models**（把思维链与外部工具结合）展开。业界也把思维链写入产品的交互设计，例如在代码生成、数学教育和法律文书审查中加入“思考步骤”。如果想进一步深入，可以关注 **CoT 的自动化生成**（让模型自己学习如何写思维链）以及 **跨语言思维链**（在多语言模型中保持一致的推理结构）等前沿方向。

### 一句话记住它
只要给大模型加上几条写草稿的示例，它就能自行展开多步推理，显著提升复杂任务的准确率。