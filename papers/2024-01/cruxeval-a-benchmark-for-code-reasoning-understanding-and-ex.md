# CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution

> **Date**：2024-01-05
> **arXiv**：https://arxiv.org/abs/2401.03065

## Abstract

We present CRUXEval (Code Reasoning, Understanding, and eXecution Evaluation), a benchmark consisting of 800 Python functions (3-13 lines). Each function comes with an input-output pair, leading to two natural tasks: input prediction and output prediction. First, we propose a generic recipe for generating our execution benchmark which can be used to create future variation of the benchmark. Second, we evaluate twenty code models on our benchmark and discover that many recent high-scoring models on HumanEval do not show the same improvements on our benchmark. Third, we show that simple CoT and fine-tuning schemes can improve performance on our benchmark but remain far from solving it. The best setup, GPT-4 with chain of thought (CoT), achieves a pass@1 of 75% and 81% on input and output prediction, respectively. In contrast, Code Llama 34B achieves a pass@1 of 50% and 46% on input and output prediction, highlighting the gap between open and closed source models. As no model is close to acing CRUXEval, we provide examples of consistent GPT-4 failures on simple programs as a lens into its code reasoning capabilities and areas for improvement.

---

# CRUXEval：代码推理、理解与执行基准 论文详细解读

### 背景：这个问题为什么难？

在代码生成模型的评测里，HumanEval 之类的基准大多只检查模型能否写出满足给定单元测试的完整函数，却忽略了模型对已有函数的“读懂”和“逆向推理”能力。实际开发中，程序员常常需要根据函数的实现推断可能的输入，或根据期望的输出倒推出合适的调用参数，这两类任务对模型的语义理解、控制流推理和执行模拟要求更高。过去的评测几乎没有提供这种“双向”考察，导致高分模型在真实代码阅读或调试场景里表现不一定可靠。

### 关键概念速览
- **CRUXEval**：一个专门测评代码推理、理解和执行的基准，收录 800 条 3‑13 行的 Python 小函数，每条都有唯一的输入‑输出对。相当于给模型出一道“看代码，猜输入”或“看代码，猜输出”的小测验。
- **输入预测（Input Prediction）**：模型看到函数实现和期望的输出，需要生成一个合法的输入，使函数运行后得到该输出。类似于逆向求解方程的过程。
- **输出预测（Output Prediction）**：模型已知函数实现和一个具体输入，要求直接给出函数的返回值。相当于让模型在脑中“跑一遍”代码。
- **pass@k**：在 k 次尝试中至少有一次预测正确的比例。pass@1 就是一次预测成功的概率，常用来衡量代码生成模型的准确率。
- **CoT（Chain‑of‑Thought）**：让模型在给出答案前先写出思考步骤，像写草稿一样把推理过程显式化，帮助模型避免一步到位的盲猜。
- **Fine‑tuning（微调）**：在已有模型上继续用特定任务的数据训练，使模型更适应该任务的输入输出模式。这里指在 CRUXEval 上进行的轻量微调。
- **闭源 vs 开源模型**：闭源模型如 GPT‑4 只能通过 API 调用，内部细节不公开；开源模型如 Code Llama 可以自行部署和改造。两者在同一基准上的差距能直观看出研发资源的影响。

### 核心创新点
1. **从单向生成到双向推理的基准设计**  
   之前的评测只让模型“写代码”。CRUXEval 把每个函数拆成两道题：给定实现推输入、给定实现和输入推输出。这样模型必须先理解代码的控制流和数据流，再进行逆向或正向推理，显著提升评测的覆盖面。

2. **通用生成流程（Recipe）**  
   作者提供了一套自动化脚本，能够从任意 Python 代码库抽取函数、生成唯一的输入‑输出对，并自动构造上述两种任务。这个流程可以复用来扩展基准、加入新语言或不同难度层级，解决了评测数据难以持续更新的问题。

3. **系统性对比闭源与开源模型**  
   在同一基准上跑了 20 种代码模型，发现很多在 HumanEval 上表现抢眼的模型（如某些最新的 CodeGen 变体）在 CRUXEval 上提升不明显，甚至退步。通过这种横向对比，作者揭示了“高分不等于强推理”的误区。

4. **简单 CoT 与微调即可显著提升**  
   在不改模型结构的前提下，仅在推理阶段加入思维链提示，或在 CRUXEval 数据上做少量微调，就能把 GPT‑4 的 pass@1 从约 60% 拉到 75%（输入预测）和 81%（输出预测）。这表明模型已有潜力，只是缺少合适的引导。

### 方法详解
整体思路可以分为三步：**数据准备 → 任务构造 → 评测执行**。

1. **数据准备**  
   - 从公开的 Python 项目中抽取函数，过滤掉依赖外部文件、交互式输入或复杂类型的代码，确保每个函数能在独立环境下运行。  
   - 对每个函数随机生成若干合法输入（使用 Python 的 `hypothesis` 库或手写生成器），执行函数得到对应输出。只保留唯一的输入‑输出对，以避免歧义。

2. **任务构造**  
   - **输入预测**：把函数实现和目标输出拼接成提示，要求模型输出一个满足该输出的输入。这里的难点是模型必须在脑中模拟函数执行，找到满足条件的参数组合。  
   - **输出预测**：把函数实现和具体输入拼接成提示，要求模型直接给出返回值。相对直接，但仍需模型正确解析控制流（如循环、条件分支）并执行数值计算。

3. **评测执行**  
   - 对每个任务，模型会生成若干候选答案（默认 1 条），系统把候选输入或输出代入原函数进行实际运行，检查是否得到期望的结果。成功即记为一次 pass。  
   - 统计所有任务的通过率，得到 pass@1。若模型支持多次采样，可计算 pass@k。

**关键技巧**  
- **思维链提示**：在输入预测任务中，提示中加入“先分析函数的返回路径，再列举可能的输入”之类的指令，迫使模型分步思考。实验表明，这种显式的推理步骤能把错误率显著压低。  
- **轻量微调**：作者只在 CRUXEval 的 800 条样本上进行 1‑2 epoch 的微调，使用的是标准的语言模型微调流程（AdamW 优化器、学习率 2e-5），即可让模型熟悉“函数+输入/输出”这种提示格式。  
- **自动化评测脚本**：整个评测过程全部可脚本化，支持并行执行，极大降低了手工验证的成本。

最让人意外的地方是：只要给模型一点“思考指令”，即使是闭源的 GPT‑4 也会出现系统性错误（比如在递归函数里忘记基准条件），这说明模型的内部执行模拟仍然是近似的，而不是完整的解释器。

### 实验与效果
- **数据集**：CRUXEval 包含 800 条 Python 函数，分别对应输入预测和输出预测两类任务。每条函数长度在 3‑13 行之间，覆盖基本算术、列表操作、字符串处理等常见模式。  
- **基线模型**：包括 OpenAI 的 GPT‑4、GPT‑3.5、Claude、以及开源的 Code Llama 34B、StarCoder、CodeGen 等共二十余种。  
- **主要结果**：  
  - GPT‑4 在加入 CoT 提示后，pass@1 达到 75%（输入预测）和 81%（输出预测），是所有模型中最高的。  
  - 同样配置下的 Code Llama 34B 只能得到约 50%（输入）和 46%（输出），显示出闭源模型在推理深度上的优势。  
  - 在 HumanEval 上表现优秀的模型（如某些最新的 CodeGen）在 CRUXEval 上的提升幅度不到 5%，甚至出现退步，说明它们更擅长“写代码”而非“读代码”。  
- **消融实验**：作者分别去掉 CoT、去掉微调、只用原始提示进行对比，发现 CoT 对输入预测的提升约为 12%，微调对输出预测的提升约为 8%，两者叠加效果最佳。  
- **局限性**：实验只覆盖了 Python 小函数，未涉及面向对象、外部库调用或多文件项目；此外，评测只使用 pass@1，未探索模型在多次采样下的潜力。作者也承认，GPT‑4 在一些极其简单的函数上仍会给出错误的输入或输出，说明模型的代码推理仍不够可靠。

### 影响与延伸思考
CRUXEval 的出现让社区重新审视“代码生成”模型的评估维度，推动了更多关注代码阅读、逆向推理的工作。随后有几篇论文（如 **CodeBERT‑Rev**、**ExecEval**）在此基准上进行对比，尝试把模型内部的执行模拟（如神经解释器）与外部真实运行相结合。对开源模型的激励也很明显，很多团队开始在自己的模型上加入思维链提示或轻量微调，以提升在 CRUXEval 上的表现。未来可以考虑把基准扩展到多语言、加入异常处理、甚至让模型解释自己的推理过程，这些方向都有潜在的研究价值。

### 一句话记住它
CRUXEval 用“看代码、猜输入/输出”把代码生成模型的推理能力硬核测出来，告诉我们：写得好不等于读得懂。