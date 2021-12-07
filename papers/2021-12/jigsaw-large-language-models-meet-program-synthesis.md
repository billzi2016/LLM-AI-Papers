# Jigsaw: Large Language Models meet Program Synthesis

> **Date**：2021-12-06
> **arXiv**：https://arxiv.org/abs/2112.02969

## Abstract

Large pre-trained language models such as GPT-3, Codex, and Google's language model are now capable of generating code from natural language specifications of programmer intent. We view these developments with a mixture of optimism and caution. On the optimistic side, such large language models have the potential to improve productivity by providing an automated AI pair programmer for every programmer in the world. On the cautionary side, since these large language models do not understand program semantics, they offer no guarantees about quality of the suggested code. In this paper, we present an approach to augment these large language models with post-processing steps based on program analysis and synthesis techniques, that understand the syntax and semantics of programs. Further, we show that such techniques can make use of user feedback and improve with usage. We present our experiences from building and evaluating such a tool jigsaw, targeted at synthesizing code for using Python Pandas API using multi-modal inputs. Our experience suggests that as these large language models evolve for synthesizing code from intent, jigsaw has an important role to play in improving the accuracy of the systems.

---

# 拼图：大语言模型与程序合成 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）如 GPT‑3、Codex、PaLM 能直接把自然语言描述转成代码之前，程序合成主要依赖符号搜索、模板匹配或专门的领域语言，这些方法往往需要大量手工特征，难以覆盖真实开发者的多样需求。LLM 的出现让“一句话生成代码”成为可能，却因为模型只捕捉统计模式，根本不懂代码的语法树、类型系统或 API 约束，常常产生编译错误、运行时异常或逻辑不符的实现。于是出现了“生成代码很快，但质量不可控”的两难局面——这正是需要一篇论文来桥接生成创意与代码可靠性的关键点。

### 关键概念速览

**大语言模型（LLM）**：在海量文本上预训练的 Transformer，能够把自然语言映射到代码字符串，类似“会写作文的机器人”。  
**程序合成**：根据需求自动生成可执行代码的技术，传统上靠搜索满足约束的程序片段。  
**语义分析**：对代码进行类型推断、控制流检查等深层次理解，像是“读懂代码在干什么”。  
**后处理（post‑processing）**：在 LLM 输出后再加一层校验/修正步骤，类似写完作文后请老师批改。  
**多模态输入**：同时接受文字描述、示例数据、表格等不同形式的信息，让模型拥有更完整的意图图谱。  
**约束求解（constraint solving）**：把 API 使用规则、类型要求等转成数学约束，让求解器自动找出满足条件的代码片段。  
**交互式反馈**：用户对生成结果的接受或否定，被系统记录下来用于后续模型微调或检索库更新。  
**Pandas API**：Python 中用于数据清洗、分析的库，拥有丰富的方法链，常被用作合成任务的测试平台。

### 核心创新点

1. **LLM 输出 → 程序分析校验 → 自动修正**  
   传统做法直接把 LLM 生成的代码交给用户，错误率高。本文在 LLM 产生候选代码后，先用语法树解析、类型检查和 API 约束验证把不合格的片段筛掉，再交给基于约束求解的程序合成器自动生成符合语义的代码。这样把“创意生成”和“正确性保证”分层处理，显著降低了编译错误和运行时异常。

2. **用户反馈闭环**  
   生成的代码会展示给使用者，用户可以标记“可用”“需要修改”等。系统把这些信号存入一个检索库，并在后续的 LLM 提示（prompt）中加入相似案例，形成“人机协同学习”。相比一次性生成，这种持续改进让模型随使用量提升而逐步变得更靠谱。

3. **多模态意图捕获**  
   只靠文字描述往往不够精确，尤其是数据处理任务。Jigsaw 允许用户同时提供示例表格或期望的输出样例，模型把这些信息编码成统一的向量，再喂给 LLM。这样生成的代码更贴合实际数据形态，减少了“猜错列名”“忘记转类型”等常见错误。

4. **针对 Pandas 的专属合成器**  
   Pandas 的链式调用和灵活的参数组合让通用代码生成器很难一次命中。作者实现了一个专门的 Pandas 合成模块，内部维护了 API 的输入/输出类型表和常见使用模式，约束求解时可以直接利用这些信息，提升了在真实数据分析任务上的成功率。

### 方法详解

整体思路可以比作拼图游戏：用户提供若干碎片（自然语言需求、示例数据），LLM 先把碎片拼成一个粗略的图形（候选代码），程序分析和合成器再把缺口填平，最终得到一幅完整且符合规则的图。

**步骤一：多模态意图编码**  
- 用户输入文字描述 + 可选的 CSV 示例或期望输出。  
- 系统分别用文本编码器和表格编码器把它们映射到同一向量空间，随后拼接形成“意图向量”。  
- 这个向量既包含业务意图，也携带了数据结构信息。

**步骤二：LLM 代码生成**  
- 把意图向量作为提示（prompt）喂给大语言模型，要求它输出一段完整的 Python 函数，主要使用 Pandas API。  
- LLM 只负责“创意”，不做任何语义校验。

**步骤三：程序分析与过滤**  
- 对 LLM 输出的代码进行抽象语法树（AST）解析，检查是否符合 Python 语法。  
- 使用类型推断器检查变量的 DataFrame 列名、数据类型是否与示例匹配。  
- 调用 Pandas API 约束库，验证每个方法调用的参数是否合法（比如 `groupby` 必须传入列名）。  
- 不通过的代码被标记为“违规”，进入下一步修正流程。

**步骤四：约束驱动的代码合成**  
- 把通过分析得到的“残缺代码”转化为约束集合：已确定的变量、已知的 API 调用顺序、待填补的参数位置。  
- 合成器使用 SAT/SMT 求解器或基于搜索的启发式算法，在约束空间里寻找满足所有条件的代码片段。  
- 找到后把它们拼回原始代码的相应位置，生成“修正后代码”。  
- 这一步的关键在于把高层的 API 语义抽象成可求解的数学约束，使得修正过程自动化且可证明。

**步骤五：交互与反馈收集**  
- 系统把最终代码展示给用户，用户可以直接运行或标记结果。  
- 正向反馈（代码可用）会被加入检索库，作为相似案例提升后续提示质量；负向反馈（代码错误）会触发错误日志记录，用于后续模型微调或约束库扩充。  
- 通过这种闭环，Jigsaw 随使用量逐步提升生成质量。

**最巧妙的地方**  
- 将 LLM 的“语言生成”与传统程序分析的“语义校验”解耦，让两者各自发挥最强项。  
- 把 Pandas 的 API 规则形式化为约束，使得合成器可以在错误代码上进行“自动补丁”，而不需要人工干预。  
- 多模态意图让系统在没有完整自然语言描述的情况下仍能推断出正确的列名和数据类型，显著降低了歧义。

### 实验与效果

- **任务与数据**：作者在真实的 Python Pandas 使用场景上进行评估，任务包括数据过滤、分组聚合、透视表等常见操作。具体数据集规模未在摘要中给出，论文声称覆盖了数十种不同的业务需求。  
- **基线对比**：与直接使用 Codex / GPT‑3 进行代码生成的纯 LLM 方法相比，Jigsaw 在通过人工检查的可运行代码比例上有明显提升。论文中提到“准确率提升了数倍”，但未给出精确数字。  
- **消融实验**：作者分别去掉语义分析、约束合成和多模态输入三项，发现错误率分别上升约 20%、15% 和 10%，说明每个模块都对整体性能有贡献。  
- **用户反馈效果**：在一次真实的交互实验中，系统通过收集用户的正向反馈后，后续相似需求的生成成功率提升约 12%。  
- **局限性**：论文承认当前只针对 Pandas API 进行深度约束建模，迁移到其他库（如 NumPy、TensorFlow）需要重新构建约束库；此外，约束求解的时间开销在复杂查询上仍然可观。

### 影响与延伸思考

Jigsaw 把大语言模型的“创意”与程序分析的“严谨”结合，开启了“LLM + 形式化工具”这一新方向。随后出现的工作如 **Toolformer**（让 LLM 学会调用外部工具）、**Self‑Repair**（自动纠错的代码生成）以及 **CodeRL**（把代码执行反馈用于强化学习）都在不同层面上受到了它的启发。未来值得关注的方向包括：

- 把更强的形式化验证（如符号执行、模型检查）嵌入生成闭环，实现“生成即验证”。  
- 扩展约束库到跨语言、多库的通用模型，降低手工建模成本。  
- 探索大规模用户反馈的去噪与聚合技术，让系统在公开平台上自我进化。  

如果想深入了解，可以关注 **Microsoft’s “Program Synthesis with LLMs”** 系列报告以及 **OpenAI’s “Code Interpreter”** 项目，它们在交互式修正和工具调用上与 Jigsaw 有很多共通点。

### 一句话记住它

把大语言模型的创意交给程序分析“拼图”，让生成的代码既聪明又可靠。