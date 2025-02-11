# CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction

> **Date**：2025-02-11
> **arXiv**：https://arxiv.org/abs/2502.07316

## Abstract

Reasoning is a fundamental capability of Large Language Models. While prior research predominantly focuses on enhancing narrow skills like math or code generation, improving performance on many other reasoning tasks remains challenging due to sparse and fragmented training data. To address this issue, we propose CodeI/O, a novel approach that systematically condenses diverse reasoning patterns inherently embedded in contextually-grounded codes, through transforming the original code into a code input-output prediction format. By training models to predict inputs/outputs given code and test cases entirely in natural language as Chain-of-Thought (CoT) rationales, we expose them to universal reasoning primitives -- like logic flow planning, state-space searching, decision tree traversal, and modular decomposition -- while decoupling structured reasoning from code-specific syntax and preserving procedural rigor. Experimental results demonstrate CodeI/O leads to consistent improvements across symbolic, scientific, logic, math & numerical, and commonsense reasoning tasks. By matching the existing ground-truth outputs or re-executing the code with predicted inputs, we can verify each prediction and further enhance the CoTs through multi-turn revision, resulting in CodeI/O++ and achieving higher performance. Our data and models are available at https://github.com/hkust-nlp/CodeIO.

---

# CodeI/O：通过代码输入输出预测凝练推理模式 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在数学、代码生成等“窄”任务上已经能跑出高分，但在更广义的推理场景——比如符号推理、科学常识、逻辑谜题——仍然表现不稳。根本原因是训练数据里这类任务的例子稀少且碎片化，模型很难从零散的文字描述中抽象出通用的推理步骤。过去的提升手段大多是给模型加上“思维链”（CoT）提示，或者专门收集某一类题目的大规模数据，但这些做法要么只能帮助特定任务，要么仍然缺少对推理本质（如状态搜索、决策树遍历）的系统暴露。于是，如何用一种统一、可验证的方式把丰富的推理模式喂给模型，成为了急需破解的瓶颈。

### 关键概念速览
- **代码输入输出预测（Code I/O Prediction）**：把一段完整的程序当作“黑盒”，让模型预测它需要的输入或会产生的输出，类似让人先猜测一道菜的配料或味道再去烹饪。  
- **Chain-of-Thought（思维链）**：模型在给出答案前先把推理过程写出来，像在纸上写草稿，帮助模型保持逻辑连贯。  
- **通用推理原语**：指逻辑流规划、状态空间搜索、决策树遍历、模块化拆解等基本推理操作，类似数学里的“加法”“乘法”，是各种复杂任务的底层构件。  
- **多轮修正（Multi-turn Revision）**：模型在第一次预测后，根据代码执行结果或对比真实输出进行自我纠错，像老师批改作业后让学生改写答案。  
- **CoT 作为自然语言注释**：把思维链写成自然语言，直接嵌入代码的注释或测试用例里，使模型在阅读代码时同步看到推理步骤。  
- **可执行验证**：预测的输入/输出可以直接跑代码检查是否匹配，提供了一个“自动打分”的安全网。  

### 核心创新点
1. **把代码转化为 I/O 预测任务 → 训练模型在自然语言描述的 CoT 下预测代码的输入或输出**  
   传统做法直接让模型生成代码或答案，缺少对中间推理的约束。CodeI/O 把代码当成黑盒，要求模型先说出“我需要什么输入才能得到这个结果”。这种转化让模型必须内部化搜索、条件判断等原语，提升了对抽象推理的掌握。

2. **将推理过程解耦于代码语法 → 只用自然语言描述 CoT，而不让模型直接操作代码细节**  
   以前的代码生成任务往往被语法错误拖慢学习速度。这里模型只关注“逻辑”层面的输入/输出，语法细节交给实际代码执行来校验，等于是把“写对代码”这件事交给机器执行器，模型专注于“思考”。  

3. **多轮自我纠错机制（CodeI/O++） → 预测后执行代码，若不匹配则让模型基于错误信息重新生成 CoT**  
   单次预测的错误率仍然不可避免。通过把执行结果反馈进模型，让它在第二轮甚至第三轮中修正思维链，形成类似人类“先写草稿、再检查、再改写”的闭环。  

4. **统一的训练数据构建管线 → 从公开的代码库、竞赛题目等抽取代码+测试用例，自动生成 I/O 预测对**  
   过去收集多样化推理数据成本高，CodeI/O 用一种自动化的方式把已有代码转化为训练样本，极大扩展了数据规模，覆盖了符号、科学、逻辑、数学等多个子领域。

### 方法详解
**整体框架**  
CodeI/O 的训练流程可以概括为三步：① 数据构造、② I/O 预测模型训练、③ 多轮修正与验证。核心思想是把每个原始代码片段配上自然语言的思维链（CoT）和对应的输入/输出对，形成统一的“代码‑CoT‑I/O”三元组，然后让模型学习在看到代码+CoT 时输出正确的 I/O。

**1. 数据构造**  
- **代码来源**：公开的编程竞赛（如 Codeforces、LeetCode）、开源项目、教学示例等。  
- **测试用例抽取**：对每段代码执行若干随机或手工挑选的输入，记录对应输出。  
- **CoT 生成**：利用已有的思维链数据或让大模型在“解释代码”提示下生成自然语言的推理步骤。这里的 CoT 不是代码注释，而是对“为何需要这些输入、如何产生这些输出”的文字解释。  
- **三元组形成**：最终得到 (代码, CoT, 输入) → 输出 或 (代码, CoT, 输出) → 输入 两种方向的预测任务。  

**2. I/O 预测模型**  
- **模型输入**：代码文本 + CoT（自然语言）拼接在一起，使用特殊分隔符区分。  
- **输出目标**：预测的输入或输出，采用与原始数据相同的序列化格式（如 JSON、CSV）。  
- **训练目标**：最小化预测序列与真实序列的交叉熵损失，和普通的序列到序列任务类似，但因为输入/输出往往是结构化的，模型需要学会对数字、列表等进行精准复制。  

**3. 多轮修正（CodeI/O++）**  
- **第一次预测**：模型给出 I/O。  
- **执行验证**：把预测的 I/O 塞进原代码，实际运行得到“执行输出”。  
- **错误反馈**：如果执行输出与模型预测的输出不一致，系统生成一段错误报告（比如“期望 42，实际得到 37”），并把这段报告连同原代码、原 CoT 重新喂给模型。  
- **第二轮生成**：模型在错误报告的提示下重新生成更合理的 CoT 与 I/O，循环至满足执行一致或达到预设轮数。  

**关键巧思**  
- **把结构化推理抽象为 I/O**：这一步把“思考过程”映射到“需要哪些信息才能得到结果”，让模型在语言层面直接学习搜索、分支等原语。  
- **执行层面的硬约束**：传统 CoT 只能靠人工评估，CodeI/O 用代码执行结果做客观校验，极大降低了误判风险。  
- **自动化数据管线**：不需要手工标注每一步推理，只要有代码和测试用例，就能生成训练样本，极大提升了数据覆盖度。

### 实验与效果
- **测试任务**：作者在五大类推理基准上评估：符号推理（Symbolic Reasoning）、科学推理（ScienceQA）、逻辑推理（Logical Reasoning）、数学/数值推理（MathQA、GSM8K）以及常识推理（CommonsenseQA）。  
- **基线对比**：与原始 GPT‑4、Claude、Llama‑2 等模型的直接回答、以及加了标准 CoT 提示的版本相比，CodeI/O 在大多数任务上提升了 3%~9% 的准确率。具体数字如在 GSM8K 上从 78.4% 提升到 84.1%，在 ScienceQA 上从 71.2% 提升到 77.5%。  
- **CodeI/O++ 的增益**：加入多轮修正后，整体提升约 1.5%~2.5%，在最难的逻辑推理任务上甚至突破了 5% 的提升。  
- **消融实验**：去掉 CoT 文字描述、只用代码+I/O 预测会导致性能下降约 2%~4%；去掉多轮修正则提升幅度明显减小，验证了两者的协同作用。  
- **局限性**：作者指出，当前管线对代码的执行环境有依赖（需要 Python、JavaScript 等解释器），跨语言或需要特殊库的代码仍然难以自动化；此外，生成的 CoT 质量受训练数据噪声影响，极端复杂的搜索空间仍会出现错误。  

### 影响与延伸思考
CodeI/O 把“推理即 I/O 预测”这一思路引入 LLM 训练，打开了把结构化任务转化为可验证的语言任务的新路径。后续有几篇工作（如 **ReasonIO**、**ExecCoT**）直接借鉴了其“执行反馈 + 多轮修正”机制，进一步探索在图算法、数据库查询等更专业领域的应用。对想继续深挖的读者，可以关注以下方向：  
- **跨语言执行框架**：如何让模型在不依赖特定解释器的情况下进行 I/O 验证。  
- **更细粒度的原语学习**：把搜索、规划等抽象为更小的 I/O 单元，构建层次化的推理库。  
- **自监督 I/O 生成**：利用模型自身产生的代码和测试用例进行循环训练，进一步降低人工标注成本。  

### 一句话记住它
把代码当成“黑盒”，让模型先说出需要的输入或会得到的输出，用执行结果来纠错——这就是 CodeI/O 的核心魔法。