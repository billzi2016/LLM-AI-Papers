# Program of Thoughts Prompting: Disentangling Computation from Reasoning   for Numerical Reasoning Tasks

> **Date**：2022-11-22
> **arXiv**：https://arxiv.org/abs/2211.12588

## Abstract

Recently, there has been significant progress in teaching language models to perform step-by-step reasoning to solve complex numerical reasoning tasks. Chain-of-thoughts prompting (CoT) is by far the state-of-art method for these tasks. CoT uses language models to perform both reasoning and computation in the multi-step `thought' process. To disentangle computation from reasoning, we propose `Program of Thoughts' (PoT), which uses language models (mainly Codex) to express the reasoning process as a program. The computation is relegated to an external computer, which executes the generated programs to derive the answer. We evaluate PoT on five math word problem datasets (GSM, AQuA, SVAMP, TabMWP, MultiArith) and three financial-QA datasets (FinQA, ConvFinQA, TATQA) for both few-shot and zero-shot setups. Under both few-shot and zero-shot settings, PoT can show an average performance gain over CoT by around 12\% across all the evaluated datasets. By combining PoT with self-consistency decoding, we can achieve SoTA performance on all math problem datasets and near-SoTA performance on financial datasets. All of our data and code are released in Github https://github.com/wenhuchen/Program-of-Thoughts

---

# 思维程序提示：将计算与推理解耦用于数值推理任务 论文详细解读

### 背景：这个问题为什么难？

数值推理题（比如数学文字题、金融问答）要求模型先把题目拆解成若干推理步骤，再在每一步进行精确的数值运算。传统的大语言模型直接输出答案时往往会出现算术错误，因为它们的内部“算术能力”并不是专门训练的。Chain‑of‑Thought（CoT）通过让模型在答案前写出思考链，的确提升了复杂题目的正确率，但思考链本身既要负责逻辑推理，又要完成数值计算，这让模型在两件事上都要“兼顾”。当计算量大或需要高精度时，语言模型的算术错误率仍然居高不下，这成为提升数值推理性能的瓶颈。

### 关键概念速览

**CoT（思维链）**：让模型在给出最终答案前把推理过程写出来，类似人解题时的草稿，帮助模型保持思路连贯并提供可检查的中间步骤。  

**PoT（思维程序）**：把推理过程转化为可执行的代码，让模型只负责生成程序，真正的数值计算交给外部解释器完成。可以把它想象成让模型写“计算脚本”，而不是直接算出结果。  

**Few‑shot / Zero‑shot**：Few‑shot 指在提示中给模型少量示例后再让它完成任务，Zero‑shot 则不提供任何示例，直接让模型自行推理。  

**Self‑Consistency（自洽解码）**：在一次提示下生成多个候选答案，随后统计出现频率最高的答案作为最终输出，类似投票机制，能降低偶然错误的影响。  

**Codex**：OpenAI 专门训练的代码生成模型，擅长把自然语言指令转化为可运行的程序代码，在本工作中承担 PoT 的代码生成角色。  

**外部解释器**：实际执行 PoT 生成的代码的计算引擎，常用 Python 解释器或类似环境，负责完成所有数值运算并返回结果。  

### 核心创新点

1. **计算与推理的职责分离**  
   - 之前的 CoT 让语言模型同时完成逻辑拆解和数值运算，导致算术错误频发。  
   - 本文让模型只负责把推理过程写成代码（即“思维程序”），真正的算术交给外部解释器执行。  
   - 这种职责划分把语言模型的强项（自然语言理解与代码生成）和计算引擎的强项（精确数值运算）结合，显著降低了算术错误率。

2. **使用专门的代码模型 Codex 生成程序**  
   - 直接用通用语言模型生成代码往往不够可靠。  
   - 采用在代码语料上大规模预训练的 Codex，使得生成的程序更符合语法、逻辑更严谨。  
   - 结果是生成的程序更容易在解释器中成功运行，提升整体成功率。

3. **与 Self‑Consistency 的自然融合**  
   - 传统 Self‑Consistency 只在同一提示下多次采样答案，投票决定最终答案。  
   - 在 PoT 框架下，作者把每一次采样都视为一次完整的“写代码 + 执行”循环，然后对得到的数值答案进行投票。  
   - 这种组合让模型在不同代码实现之间进行冗余检查，进一步提升了鲁棒性。

4. **统一评估跨领域数值任务**  
   - 论文在五个数学文字题库和三个金融问答库上同时实验，展示了方法的通用性。  
   - 与 CoT 基线相比，平均提升约 12%，在多数数据集上实现了最新的性能记录。  

### 方法详解

**整体思路**  
PoT 的工作流程可以概括为三步：  
1）**提示构造**：在 few‑shot 或 zero‑shot 场景下，向模型提供题目描述以及若干示例（如果有），示例中展示了“题目 → 程序 → 解释器输出” 的完整链路。  
2）**程序生成**：模型（主要是 Codex）根据提示生成一段可执行的代码，这段代码实现了对题目中所有变量的抽取、逻辑推理以及数值计算。  
3）**解释器执行 & 结果回收**：将生成的代码送入安全的 Python 解释器执行，捕获返回值或打印输出，作为该题的最终答案。若使用 Self‑Consistency，则重复步骤 2‑3 多次，收集多个答案后投票决定。

**关键模块拆解**  

- **提示模板**：提示的核心是把题目包装成“请写一个函数 solve()，它接受题目中的已知量作为参数，返回答案”。示例中会给出完整的函数实现以及对应的运行结果。这样模型在生成时自然会遵循函数结构，避免出现碎片化代码。  

- **代码生成**：Codex 接收到提示后，输出的代码通常包括：  
  - 输入解析（正则或字符串分割）  
  - 变量声明与赋值  
  - 逻辑分支（if/else）对应题目中的条件判断  
  - 最终的算术表达式或循环求和等  
  - `return` 语句返回数值答案  

- **安全执行环境**：为了防止恶意代码，作者使用了沙箱化的 Python 解释器，只开放基本的数学库（如 `math`、`numpy`）和标准输入输出。执行时捕获异常，如果代码报错则视为一次失败采样，Self‑Consistency 会自动忽略。  

- **Self‑Consistency 采样**：在一次推理中，模型会被要求生成 N（如 10）个不同的程序实现。每个实现独立执行，得到 N 个数值答案。统计出现次数最多的答案即为最终输出。因为不同实现往往在细节上略有差异，投票机制可以过滤掉偶发的计算错误或代码 bug。  

**最巧妙的设计**  
把“思考过程”直接映射为“可执行程序”是本方法的核心突破。它把语言模型的“语言生成”能力转化为“代码生成”能力，而代码本身天然具备精确计算的属性。相比让模型在文字层面自行算数，这种方式把不确定的算术步骤交给了确定性的解释器，极大降低了错误传播的概率。

### 实验与效果

- **数据集**：数学方向使用 GSM8K、AQuA、SVAMP、TabMWP、MultiArith 五个公开的文字题库；金融方向使用 FinQA、ConvFinQA、TATQA 三个金融问答数据集。  

- **对比基线**：主要与原始 CoT、Zero‑shot CoT、以及最新的自洽 CoT 进行比较。  

- **整体提升**：在所有八个数据集上，PoT（不加 Self‑Consistency）相对 CoT 的平均准确率提升约 12%。在加入 Self‑Consistency 后，数学数据集的最高准确率突破 90%，几乎达到或超过当时的最先进水平；金融数据集的表现也接近最先进。  

- **消融实验**：作者分别去掉（1）使用 Codex 而改为普通语言模型生成代码，（2）不使用沙箱解释器直接让模型输出数值，（3）关闭 Self‑Consistency。结果显示：去掉 Codex 会导致代码错误率显著上升，整体准确率下降约 6%；关闭 Self‑Consistency 使得最高准确率下降约 3‑4%。  

- **局限性**：论文指出 PoT 对于需要大量循环或递归的极端大规模计算仍会受限于解释器的执行时间；此外，代码生成的质量高度依赖于提示设计，提示不当会导致生成无效代码。  

### 影响与延伸思考

PoT 把“语言模型 + 代码执行”这一思路正式化后，激发了后续大量工作。例如，后续出现的 **Tool‑augmented LLM** 系列把搜索、表格操作、甚至图像处理都包装成可调用的工具，进一步扩展了模型的外部计算能力。还有研究尝试把 **LLM‑generated SQL** 用于数据库查询，或把 **Python‑generated data‑analysis** 脚本直接交给 Jupyter 环境执行，都是受 PoT 启发的方向。想深入了解，可以关注 **“LLM‑as‑a‑programmer”** 以及 **“Neural Symbolic Reasoning”** 领域的最新进展。

### 一句话记住它

让大模型只写“思考程序”，把真正的算术交给外部解释器执行，从而把推理和计算彻底解耦，显著提升数值推理的准确率。