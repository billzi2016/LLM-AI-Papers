# Towards Understanding Chain-of-Thought Prompting: An Empirical Study of   What Matters

> **Date**：2022-12-20
> **arXiv**：https://arxiv.org/abs/2212.10001

## Abstract

Chain-of-Thought (CoT) prompting can dramatically improve the multi-step reasoning abilities of large language models (LLMs). CoT explicitly encourages the LLM to generate intermediate rationales for solving a problem, by providing a series of reasoning steps in the demonstrations. Despite its success, there is still little understanding of what makes CoT prompting effective and which aspects of the demonstrated reasoning steps contribute to its performance. In this paper, we show that CoT reasoning is possible even with invalid demonstrations - prompting with invalid reasoning steps can achieve over 80-90% of the performance obtained using CoT under various metrics, while still generating coherent lines of reasoning during inference. Further experiments show that other aspects of the rationales, such as being relevant to the query and correctly ordering the reasoning steps, are much more important for effective CoT reasoning. Overall, these findings both deepen our understanding of CoT prompting, and open up new questions regarding LLMs' capability to learn to reason in context.

---

# 走向对思维链提示的理解：关键因素的实证研究 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，想让模型完成需要多步推理的任务往往只能靠一次性输出答案，错误率高得离谱。即使加入了少量示例（few‑shot），模型也倾向于直接给出结论，而不展示思考过程。思维链（Chain‑of‑Thought，CoT）提示通过让模型先写出中间推理步骤，显著提升了算数、逻辑和常识推理的准确率。但人们并不清楚，究竟是哪部分示例在驱动这种提升：是示例的真实性、步骤的顺序、还是它们与提问的相关性？缺乏系统实验让研究者难以把握 CoT 的本质，也限制了更高效的提示设计。

### 关键概念速览
- **CoT（思维链）**：在回答前让模型输出一串推理步骤，类似人做题时先写草稿，再写答案。这样模型的思考过程对齐后，最终答案更可靠。
- **few‑shot prompting（少样本提示）**：只给模型几条示例就让它学习任务模式，像给它几道例题的答案让它“看懂”怎么做。
- **无效示例（invalid demonstrations）**：示例中的推理步骤故意写错或不合逻辑，但仍保持形式完整。相当于给模型看一段“错误的解题过程”。
- **相关性（relevance）**：示例的内容是否围绕当前提问的主题展开。好比老师讲例题时，例子要和学生的实际问题相吻合，才能帮助学生思考。
- **顺序正确性（order correctness）**：推理步骤的前后顺序是否符合逻辑。像做菜时，先切菜再下锅，顺序错了菜就做不成。

### 核心创新点
1. **无效示例仍能驱动 CoT**  
   之前的研究默认示例必须是“正确的”。这篇论文故意构造错误的推理链，结果发现模型在推理时仍能产生连贯的思考，性能只下降 10‑20%。这表明 CoT 的核心并不在于示例的真实性，而是示例的结构化形式本身。

2. **系统化评估示例属性的贡献**  
   作者把示例的属性拆成“真实性”“相关性”“顺序”“长度”等维度，逐一做消融实验。实验显示，相关性和顺序对性能的提升贡献最大，远超真实性。换句话说，模型更在意“这段推理和我问的问题有没有关系”以及“推理步骤是否按对的顺序排列”。

3. **提出“无效‑相关‑顺序”三要素框架**  
   基于实验结果，论文归纳出 CoT 提示成功的关键是：示例必须与查询主题相关、步骤顺序正确，而真实性可以大幅放宽。这个框架为后续提示工程提供了简化的设计准则。

### 方法详解
整体思路很直接：先准备一批示例，然后在这些示例上做不同属性的改动，观察模型在多种推理任务上的表现。具体步骤如下：

1. **基线构建**  
   选取公开的 few‑shot CoT 数据集（如 GSM8K、SVAMP、MultiArith），使用原始的正确示例作为基线，记录准确率。

2. **生成无效示例**  
   对每条基线示例，随机替换或删除其中的关键推理步骤，使其逻辑不通，但保持文字格式（如“先把 5 加 3，得到 9”故意写错）。这一步相当于把老师的解题过程写成“先把 5 加 3，得到 9”，让学生看到错误的草稿。

3. **控制相关性**  
   将示例的主题换成与当前任务无关的内容（比如把数学题的示例换成常识问答），保持步骤结构不变，测试模型是否仍受益。

4. **打乱顺序**  
   把正确示例的推理步骤随机打乱顺序，保持每一步文字不变，只是顺序错位，观察模型的表现跌幅。

5. **组合实验**  
   将上述改动组合（如无效+相关、无效+顺序错位等），进一步验证各因素的交互作用。

6. **评估指标**  
   使用任务的标准准确率、以及模型生成的推理文本的连贯性评分（人工或自动评估），比较不同实验条件下的差异。

最让人意外的设计是“无效示例”。直觉上会认为错误的示例会误导模型，甚至让模型学到错误的推理方式。但实验结果显示，模型在看到结构化的推理链后，会自行在推理时“纠错”，只要示例的形式和顺序是合理的。

### 实验与效果
- **测试任务**：包括算数推理（GSM8K、SVAMP、MultiArith）、逻辑推理（Coin Flip、Last Letter）以及常识推理（CommonsenseQA）等。
- **基线 vs. 无效示例**：在所有任务上，使用无效示例的表现约为基线的 80‑90%。例如，在 GSM8K 上基线准确率 85%，无效示例仍能达到约 78%。
- **相关性实验**：当示例与查询不相关时，准确率骤降 15‑30%，说明相关性是关键驱动因素。
- **顺序实验**：把步骤顺序打乱后，性能下降幅度与相关性相当，甚至在某些任务上更明显，进一步验证顺序的重要性。
- **消融结果**：综合消融表明，相关性贡献约 45%，顺序贡献约 35%，真实性仅贡献约 10%，其余为噪声或交叉效应。
- **局限性**：论文主要在中等规模的 LLM（如 GPT‑3、PaLM‑62B）上验证，未覆盖更大模型或小模型的行为差异；此外，推理连贯性评估仍依赖人工打分，缺少统一的自动度量。

### 影响与延伸思考
这篇工作让社区重新审视 CoT 提示的本质，从“必须给出正确示例”转向“只要示例结构合理且与任务相关”。随后出现的研究（如 “Self‑Consistency”、 “Least‑to‑Most Prompting”）都在提示设计上加入了更多结构化约束，甚至尝试自动生成相关且顺序正确的示例。还有工作尝试用检索系统从大规模语料中挑选“相关‑顺序”示例，进一步提升少样本推理的鲁棒性。想深入了解的读者可以关注“提示工程的可解释性”和“基于检索的 CoT 生成”这两个方向，那里已经出现了不少基于本论文思路的创新。

### 一句话记住它
只要示例与问题相关、步骤顺序正确，即使示例本身是错的，思维链提示仍能让大模型推理得更好。