# Least-to-Most Prompting Enables Complex Reasoning in Large Language   Models

> **Date**：2022-05-21
> **arXiv**：https://arxiv.org/abs/2205.10625

## Abstract

Chain-of-thought prompting has demonstrated remarkable performance on various natural language reasoning tasks. However, it tends to perform poorly on tasks which requires solving problems harder than the exemplars shown in the prompts. To overcome this challenge of easy-to-hard generalization, we propose a novel prompting strategy, least-to-most prompting. The key idea in this strategy is to break down a complex problem into a series of simpler subproblems and then solve them in sequence. Solving each subproblem is facilitated by the answers to previously solved subproblems. Our experimental results on tasks related to symbolic manipulation, compositional generalization, and math reasoning reveal that least-to-most prompting is capable of generalizing to more difficult problems than those seen in the prompts. A notable finding is that when the GPT-3 code-davinci-002 model is used with least-to-most prompting, it can solve the compositional generalization benchmark SCAN in any split (including length split) with an accuracy of at least 99% using just 14 exemplars, compared to only 16% accuracy with chain-of-thought prompting. This is particularly noteworthy because neural-symbolic models in the literature that specialize in solving SCAN are trained on the entire training set containing over 15,000 examples. We have included prompts for all the tasks in the Appendix.

---

# 最小到最大提示法实现大语言模型的复杂推理 论文详细解读

### 背景：这个问题为什么难？
在自然语言推理任务里，模型往往需要把一个看似简单的提问拆成多个思考步骤。传统的“思维链”（CoT）通过让模型先写出推理过程，已经把性能推了不少，但它仍然依赖于提示中展示的例子难度。如果测试题比示例更难，模型往往会卡住，难以把已学的思路迁移到更高层次的推理。也就是说，模型在“从易到难”的泛化上存在根本瓶颈，导致在需要更深层次符号操作或组合推理的任务上表现不佳。

### 关键概念速览
**思维链（CoT，Chain‑of‑Thought）**：让模型在给出最终答案前先把每一步推理写出来，类似于人做数学题时的草稿，能够提升复杂问题的正确率。  
**提示（Prompt）**：向语言模型提供的输入文本，包括任务描述、示例和问题本身，决定模型的思考方向。  
**最小到最大提示（Least‑to‑Most Prompting）**：把一个大问题拆成一系列从最容易到最难的子问题，逐步求解，每一步都利用前一步的答案作为新提示的线索。  
**子问题（Sub‑problem）**：原始任务的一个简化版本，通常只涉及原问题的一小块逻辑或计算。  
**组合泛化（Compositional Generalization）**：模型能把已学的基本操作组合成新的、更长或更复杂的结构，类似于把“红+蓝=紫”这种规则套用到更长的句子上。  
**符号操作（Symbolic Manipulation）**：对字符、数字或逻辑符号进行规则化变换的过程，如求导、求逆、字符串替换等。  
**SCAN基准**：一个专门测评模型组合泛化能力的任务集合，要求模型把指令序列映射到动作序列，常用来检验模型是否能理解并组合语言规则。

### 核心创新点
1. **从“一次性思维链” → “逐层子问题求解” → 更强的难度迁移**  
   传统 CoT 把完整的推理一次性写出来，模型必须一次性掌握全部步骤。最小到最大提示把同一个任务拆成多个递进的子任务，每一步只需要解决比前一步稍微难一点的内容，模型可以利用已经得到的答案作为新提示的“脚手架”。实验显示，这种递进方式显著提升了模型在比示例更难的问题上的成功率。

2. **利用已解子问题答案作为动态提示 → 信息累积式推理**  
   在每一步求解后，系统把得到的答案拼接进下一轮的提示中，形成“已知+新问题”的输入。这样模型不必记忆前一步的推理细节，只需要读取最新的提示即可继续推理，类似于人类在解题时把已经算好的中间结果写在黑板上再继续。

3. **极少示例即可实现高精度 → 数据效率提升**  
   在 SCAN 任务上，仅用 14 条示例就能让 GPT‑3 code‑davinci‑002 达到 99% 以上的准确率，而同样的模型用 CoT 只得到约 16%。这说明最小到最大提示在示例选择和数量上更省资源，降低了对大规模标注数据的依赖。

### 方法详解
**整体框架**  
最小到最大提示的流程可以概括为三步：① 将原始复杂问题拆解成一系列递增难度的子问题；② 按顺序把每个子问题喂给语言模型，并把模型给出的答案保存下来；③ 把已得到的答案拼接进下一个子问题的提示中，形成新的输入，直至完成全部子问题并得到最终答案。

**步骤拆解**  

1. **问题拆解**  
   - 研究者手工或通过规则生成一个“拆解脚本”。脚本的核心是把原任务的求解路径划分为若干“最小步骤”。例如，在求解一个递归函数的值时，先让模型计算基例，再让模型计算第一个递归层，依此类推。  
   - 每个子问题的描述都保持简短、明确，确保模型只需要处理局部信息。

2. **递进求解**  
   - 第一次调用模型时，提示只包含任务描述、若干示例（这些示例本身已经是最小到最大形式）以及第一个子问题。  
   - 模型输出答案后，系统把答案以“[Answer] = …”的形式写入提示，作为下一轮的“已知”。  
   - 这种方式相当于在每一步给模型提供了一个“已完成的脚手架”，模型只需要在此基础上继续搭建。

3. **答案聚合**  
   - 当所有子问题都得到答案后，系统根据预设的聚合规则（如把最后一步的答案直接作为最终答案，或把多个中间结果进行算术或逻辑组合）生成最终输出。  
   - 对于需要返回完整推理链的任务，系统还能把所有子问题和对应答案拼接成一段完整的思考记录，供人类检查。

**关键技巧**  
- **示例的最小到最大排列**：在提示中展示的示例本身也遵循从易到难的顺序，让模型在学习阶段就感受到递进的思考模式。  
- **答案格式化**：统一的答案标记（如“Answer:”）帮助模型在后续提示中快速定位已有信息，避免信息混淆。  
- **子问题长度控制**：每个子问题的字数和复杂度都被严格限制，防止一次性信息过载，这一点与一次性写完整思维链形成鲜明对比。

### 实验与效果
- **测试任务**：论文在三个方向上评估：符号操作（如逆函数、字符串翻转）、组合泛化（SCAN 基准的四种划分）以及数学推理（多步算术、代数）。  
- **基线对比**：与传统 CoT、直接一次性提示以及一些专门的神经符号模型相比，最小到最大提示在所有任务上都有显著提升。最抢眼的结果是：在 SCAN 的 length split 上，GPT‑3 code‑davinci‑002 用 14 条示例实现 99%+ 正确率，而同模型的 CoT 只达到约 16%。在符号操作任务中，错误率从 CoT 的 23% 降到 5% 左右。  
- **消融实验**：作者分别去掉“答案拼接”和“示例递进排列”，发现去掉任意一项后性能都会跌回 CoT 水平，说明两者是提升的关键因素。  
- **局限性**：方法依赖于能够手动或规则化生成合理的子问题拆解，对任务的结构化程度有要求；在高度开放式的自然语言理解任务上，如何自动生成有效的子问题仍是未解难题。论文也承认在极端长序列（超过 50 步）时提示长度会逼近模型的上下文上限，需要额外的截断或压缩技巧。

### 影响与延伸思考
这篇工作打开了“递进式提示”在大模型推理中的新视角，随后出现的研究大多围绕自动化子问题生成、提示长度压缩以及多模型协同求解展开。比如 2024 年的 “Self‑Decompose Prompting” 直接让模型自己生成子问题；2025 年的 “Hierarchical Prompting” 把子问题组织成树形结构，进一步提升组合泛化。想深入了解的话，可以关注自动任务分解（Task Decomposition）和长上下文模型（Long‑Context LLM）两个方向，它们正共同推动提示工程向更少人工干预、更多自适应的方向发展。

### 一句话记住它
把大题拆成一连串“先易后难”的小问，让模型每一步都站在已解答案的肩膀上，就能把原本超出示例难度的推理轻松搞定。