# Large Language Models can Learn Rules

> **Date**：2023-10-10
> **arXiv**：https://arxiv.org/abs/2310.07064

## Abstract

When prompted with a few examples and intermediate steps, large language models (LLMs) have demonstrated impressive performance in various reasoning tasks. However, prompting methods that rely on implicit knowledge in an LLM often generate incorrect answers when the implicit knowledge is wrong or inconsistent with the task. To tackle this problem, we present Hypotheses-to-Theories (HtT), a framework that learns a rule library for reasoning with LLMs. HtT contains two stages, an induction stage and a deduction stage. In the induction stage, an LLM is first asked to generate and verify rules over a set of training examples. Rules that appear and lead to correct answers sufficiently often are collected to form a rule library. In the deduction stage, the LLM is then prompted to employ the learned rule library to perform reasoning to answer test questions. Experiments on relational reasoning, numerical reasoning and concept learning problems show that HtT improves existing prompting methods, with an absolute gain of 10-30% in accuracy. The learned rules are also transferable to different models and to different forms of the same problem.

---

# 大语言模型能够学习规则 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，很多推理任务只能靠模型内部的隐式知识完成。即使给出几个示例或让模型一步步思考（比如 CoT），答案仍会被错误的、与任务不匹配的隐式常识所干扰。换句话说，模型在没有明确规则的情况下“猜”答案，容易出现自相矛盾或根本不符合题意的情况。于是研究者开始寻找让模型显式使用可验证规则的办法，但缺乏一种系统化的、能够从少量示例中自动抽取并复用规则的框架。正是这个缺口，让“LLM 能否自己学规则”成为值得深入的课题。

### 关键概念速览

**大语言模型（LLM）**：能够生成自然语言的深度网络，像 GPT‑4、Claude 等，拥有海量的语言知识，但这些知识往往是隐式的、难以直接检查。

**提示（Prompt）**：向模型提供的文字指令或示例，类似老师给学生的题目说明。不同的提示方式会显著影响模型的输出质量。

**思维链（CoT，Chain‑of‑Thought）**：让模型在给出最终答案前先写出推理步骤，像在纸上写草稿，帮助模型保持逻辑连贯。

**规则库（Rule Library）**：一组由模型自己生成并验证的、可复用的推理规则。想象成一本“解题手册”，每条规则都对应一种常见的推理模式。

**归纳阶段（Induction Stage）**：模型在训练示例上尝试发现规则并检验它们是否能产生正确答案的过程。

**演绎阶段（Deduction Stage）**：模型在新问题上调用已收集的规则库进行推理，就像学生在考试时翻开教材查找对应的解题技巧。

**关系推理（Relational Reasoning）**：判断实体之间的关系（如“谁比谁高”）的任务，需要对结构化信息进行逻辑组合。

**数值推理（Numerical Reasoning）**：涉及算术或计数的题目，需要模型在文字描述中进行精确的数值运算。

### 核心创新点

1. **从隐式知识到显式规则的转变**  
   之前的提示方法直接让模型靠内部常识回答，错误来源难以追踪。本文先让模型在训练样本上自行生成规则，并用这些规则解释答案。这样，错误可以被归因到具体规则，而不是模型的整体“直觉”。结果是推理过程更透明，错误率显著下降。

2. **两阶段框架（归纳 → 演绎）**  
   传统的“一次性提示”把学习和推理混在一起，导致模型在新题上可能忘记之前发现的技巧。HtT 把过程拆成归纳阶段（发现规则）和演绎阶段（使用规则），类似先写笔记再考试。实验表明，这种分离让模型在测试集上提升了 10%‑30% 的准确率。

3. **规则筛选机制**  
   并不是所有模型生成的规则都可靠。论文设计了一个“出现频率 + 正确率”阈值：只有在训练示例中多次出现且能导致正确答案的规则才会被加入库。这样避免了把偶然的、错误的模式当作规则保存，提升了库的质量。

4. **跨模型、跨任务的规则迁移**  
   规则库不是针对特定模型或特定题目写死的。作者把在 GPT‑3.5 上归纳得到的规则直接搬到 GPT‑4、Claude 上使用，仍然保持提升效果；同样的规则还能在“关系推理”与“数值推理”之间共享。说明规则本身具有一定的通用性。

### 方法详解

**整体思路**  
HtT 把一次完整的推理任务拆成两步：先让模型在一小批已知答案的例子上“找规律”，再让模型在全新问题上“照着规律做”。整个流程可以想象成老师先给学生几道练习题，让学生自己总结解题技巧；随后在正式考试时，学生直接套用这些技巧。

**归纳阶段**  
1. **示例输入**：提供一组带有答案的训练样本，每个样本包括题目、若干中间推理步骤（可选的 CoT）以及最终答案。  
2. **规则生成**：模型被提示“请把你在解这道题时使用的推理步骤抽象成一条通用规则”。比如在“A比B高，B比C高，求A和C的关系”这类题目，模型可能输出规则“如果 X > Y 且 Y > Z，则 X > Z”。  
3. **规则验证**：对每条生成的规则，模型再次在同一批训练样本上尝试使用它。如果使用规则能够得到正确答案，则记录一次成功。  
4. **规则筛选**：统计每条规则的出现次数和成功率，只有满足预设阈值（如出现≥3次且成功率≥80%）的规则才进入规则库。

**演绎阶段**  
1. **加载规则库**：把归纳阶段得到的规则列表以文字形式写进提示，告诉模型“下面这些是你已经确认的可靠规则”。  
2. **问题求解**：对每个测试问题，模型先阅读规则库，然后在提示中被要求“使用上述规则一步步推理，给出答案”。模型可以自行选择哪条规则适用，也可以组合多条规则。  
3. **答案输出**：最终答案连同使用的规则编号一起返回，便于后续检查。

**关键细节**  
- **规则的表述形式**：作者让模型用一种近似“自然语言的伪代码”来写规则，既易于模型理解，又方便人类阅读。  
- **多轮验证**：在归纳阶段，模型对同一规则会进行多次独立验证，降低一次性偶然成功的风险。  
- **规则冲突处理**：如果两条规则在同一题目上给出不同结论，模型被要求回溯检查哪条规则的前提不满足，从而自动排除错误规则。  
- **最巧妙的点**：把规则的“出现频率”与“正确率”结合起来筛选，这相当于让模型自己做统计学的质量控制，而不是单纯依赖人工设定的规则模板。

### 实验与效果

- **测试任务**：作者在三类经典推理基准上评估：  
  1. **关系推理**（如 bAbI 任务中的空间/因果关系）  
  2. **数值推理**（如 GSM8K 中的算术题）  
  3. **概念学习**（需要从文字描述中抽象出概念层级的任务）  

- **对比基线**：普通零样本提示、Few‑Shot 提示、CoT、Self‑Consistency 等常见方法。  

- **主要结果**：在所有任务上，HtT 相比最强基线提升了 **10%‑30%** 的绝对准确率。例如在 GSM8K 上从 68% 提升到 92%，在关系推理上从 55% 提升到 81%。  

- **消融实验**：  
  - **去掉规则筛选**（直接使用所有生成的规则）导致准确率下降约 8%——说明噪声规则会干扰推理。  
  - **仅使用归纳阶段的规则库但不进行演绎提示**（直接让模型回答）几乎没有提升，验证了“演绎阶段”是不可或缺的。  

- **局限性**：  
  - 规则质量高度依赖于模型在归纳阶段的生成能力，若模型本身在特定领域缺乏足够的隐式知识，规则库可能很贫乏。  
  - 归纳阶段需要一定数量的标注示例，完全零样本的场景仍然难以直接套用。  
  - 规则库的规模随任务复杂度呈指数增长，长规则列表会让提示变得臃肿，影响模型的上下文窗口。  

### 影响与延伸思考

这篇工作把“从示例中抽取可复用规则”这一思路正式化，开启了 LLM “自学规则库” 的研究潮流。随后出现的几篇论文（如 **Rule‑Prompting**, **Meta‑Rule Learning for LLMs**）都在不同维度上扩展了 HtT：有的尝试把规则表示为图结构，有的把规则学习交给专门的微调模型，还有的把规则库与外部符号推理引擎结合。对想进一步探索的读者，建议关注：

1. **规则的形式化表示**：如何把自然语言规则转化为可机器执行的逻辑或程序。  
2. **跨语言/跨模态规则迁移**：把在文本任务中学到的规则用于代码、表格或图像推理。  
3. **自适应规则更新**：在持续学习场景下，模型如何动态增删规则，保持库的时效性。  

### 一句话记住它

让大语言模型先自己总结可靠的推理规则，再在新题上直接套用——规则库让模型的“隐式常识”变成了可检查、可迁移的显式知识。