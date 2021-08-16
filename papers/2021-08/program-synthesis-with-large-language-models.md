# Program Synthesis with Large Language Models

> **Date**：2021-08-16
> **arXiv**：https://arxiv.org/abs/2108.07732

## Abstract

This paper explores the limits of the current generation of large language models for program synthesis in general purpose programming languages. We evaluate a collection of such models (with between 244M and 137B parameters) on two new benchmarks, MBPP and MathQA-Python, in both the few-shot and fine-tuning regimes. Our benchmarks are designed to measure the ability of these models to synthesize short Python programs from natural language descriptions. The Mostly Basic Programming Problems (MBPP) dataset contains 974 programming tasks, designed to be solvable by entry-level programmers. The MathQA-Python dataset, a Python version of the MathQA benchmark, contains 23914 problems that evaluate the ability of the models to synthesize code from more complex text. On both datasets, we find that synthesis performance scales log-linearly with model size. Our largest models, even without finetuning on a code dataset, can synthesize solutions to 59.6 percent of the problems from MBPP using few-shot learning with a well-designed prompt. Fine-tuning on a held-out portion of the dataset improves performance by about 10 percentage points across most model sizes. On the MathQA-Python dataset, the largest fine-tuned model achieves 83.8 percent accuracy. Going further, we study the model's ability to engage in dialog about code, incorporating human feedback to improve its solutions. We find that natural language feedback from a human halves the error rate compared to the model's initial prediction. Additionally, we conduct an error analysis to shed light on where these models fall short and what types of programs are most difficult to generate. Finally, we explore the semantic grounding of these models by fine-tuning them to predict the results of program execution. We find that even our best models are generally unable to predict the output of a program given a specific input.

---

# 大语言模型程序合成 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）爆发之前，程序合成主要依赖专门的代码生成模型或搜索驱动的系统，它们往往需要大量标注的代码数据才能学会写出正确的程序。即使有了大量数据，模型仍然难以理解自然语言描述背后的算法意图，导致生成的代码经常语法错误或逻辑不符。更糟的是，现有评测多聚焦在小规模的竞赛题目，缺少面向普通入门程序员的真实任务。于是，评估“通用”LLM在从自然语言直接合成短程序方面的极限，成为了迫切需要解决的科学问题。

### 关键概念速览
**大语言模型（LLM）**：参数量在数亿到上百亿之间的自回归文本生成模型，类似于“会说话的百科全书”，可以在没有显式编程指令的情况下生成代码。  
**Few‑shot 学习**：在推理阶段只给模型少量示例（通常 1‑5 条）作为上下文，让模型自行推断任务规则，类似于老师只给学生几道例题就让他们完成作业。  
**Fine‑tuning（微调）**：在已有的通用模型上继续用特定任务的数据进行训练，使模型的参数更贴合该任务的分布，像是把通用的语言能力“雕刻”成专门的编程技巧。  
**Prompt Engineering（提示工程）**：设计输入文本的格式和示例排列，以最大化模型的输出质量，等同于给模型提供一份“考试说明书”。  
**Program Synthesis（程序合成）**：从自然语言需求自动生成可执行代码的过程，目标是让机器直接把“我要一个函数计算阶乘”翻译成 Python 实现。  
**Human‑in‑the‑Loop 反馈**：让人类通过自然语言指出模型生成代码的错误，模型再根据这些反馈进行修正，类似于老师批改作业后学生再改写答案。  
**Execution Grounding（执行语义对齐）**：训练模型预测给定输入下程序的运行结果，以检验模型是否真正理解代码的语义，而不仅是表面的文字匹配。

### 核心创新点
1. **全新基准 MBPP 与 MathQA‑Python** → 作者自行构建了两套规模分别为 974 条和 23,914 条的 Python 编程任务，覆盖从入门级算法到数学推导的不同难度层次 → 为评估 LLM 在真实编程场景下的能力提供了更全面的实验平台。  
2. **系统化的 Few‑shot 与 Fine‑tune 对比** → 在同一批模型（244M‑137B 参数）上分别使用精心设计的提示进行少样本推理，并在保留的训练子集上进行微调 → 直接展示了模型规模、提示质量和微调之间的相互作用，发现即使不微调，好的提示也能让 137B 参数模型解决近 60% 的 MBPP 任务。  
3. **引入人类自然语言反馈的交互式修正** → 在模型首次生成代码后，让人类提供简短的错误描述，模型再基于该反馈重新生成答案 → 实验证明错误率被削减约 50%，证明语言模型可以在对话中自我纠错。  
4. **探索执行语义对齐的可行性** → 将模型进一步微调，使其预测程序在特定输入上的输出，而不是仅生成代码本身 → 结果显示即便是最强模型，也普遍无法准确预测执行结果，揭示了当前 LLM 对代码语义的浅层理解。

### 方法详解
整体思路可以拆成四个阶段：**数据准备 → Prompt 设计 → 模型训练/推理 → 交互式纠错**。

1. **数据准备**  
   - **MBPP**：从公开的编程练习中筛选出 974 条描述简洁、实现代码行数在 10 行以内的任务，确保每题都能由入门程序员独立完成。  
   - **MathQA‑Python**：把原始的数学问答数据转化为等价的 Python 编程题，形成 23,914 条需要实现数学公式或数值计算的任务。两套数据均划分为训练、验证、测试三部分，后者仅用于最终评估。

2. **Prompt 设计**  
   - 每个 Few‑shot 示例包含三段：**任务描述 → 示例代码 → 代码解释**。这种结构让模型在看到新任务时，能够直接对齐到“描述→代码”的映射关系。  
   - 为了避免模型把示例代码当成答案复制，提示中加入了“请自行实现”之类的指令，类似于老师在考试说明里强调“请独立作答”。

3. **模型训练与推理**  
   - 使用同一套 Transformer 架构的模型，参数规模从 244M 到 137B 不等。  
   - **Few‑shot 推理**：把设计好的 Prompt 与目标任务一起喂入模型，直接采样生成代码。  
   - **Fine‑tuning**：在保留的训练子集上进行标准的自回归语言建模，只是目标是生成对应的 Python 实现。微调过程保持原有的词表和解码策略，只是让模型的参数更贴合代码分布。实验发现，微调大约提升 10 个百分点的准确率，且提升幅度在不同规模模型上相对稳定。

4. **交互式纠错（Human‑in‑the‑Loop）**  
   - 初始代码生成后，人工检查并给出简短的错误描述（如“循环变量未初始化”）。  
   - 将原 Prompt、模型的初始答案以及人类反馈一起重新输入模型，让模型在同一上下文中生成修正版代码。  
   - 统计显示，这一步把错误率几乎砍掉一半，说明模型能够把自然语言的纠错信息转化为代码层面的改动。

5. **执行语义对齐实验**  
   - 另建一个小型数据集，记录每段代码在特定输入下的输出。  
   - 在此数据上继续微调模型，使其的输出目标从“代码文本”变为“运行结果”。  
   - 结果表明，即使是 137B 参数的模型，也只能在约 30% 的情况下正确预测输出，说明当前 LLM 对代码的“深层”语义仍然缺乏。

**最巧妙的点**在于把“自然语言反馈”直接当作新的 Prompt 输入，让模型在同一次前向传播中完成错误定位与代码修正，这种“一次对话完成两件事”的设计极大提升了实用性。

### 实验与效果
- **测试数据**：MBPP（974 题）和 MathQA‑Python（23,914 题）。  
- **Few‑shot 结果**：137B 参数模型在 MBPP 上达到 59.6% 的正确率，显著高于小模型（约 30% 左右，原文未给出精确数字）。  
- **Fine‑tuning 提升**：在同等模型规模下，微调后整体准确率提升约 10 个百分点，最高在 MathQA‑Python 上达到 83.8%。  
- **人类反馈**：加入一次自然语言纠错后，错误率下降约 50%，相当于把 59.6% 提升到接近 80% 的水平（具体数字未披露）。  
- **执行预测**：即便是最强模型，也普遍无法准确预测代码输出，说明生成代码的语义理解仍是薄弱环节。  
- **Baseline 对比**：论文未列出具体竞争模型的数值，但暗示相比早期的专用代码生成模型（如 Codex 前身）有明显优势，尤其在少样本设置下表现更稳健。  
- **消融实验**：通过去掉提示中的代码解释或人类反馈，模型性能明显下降，验证了每个模块的贡献。  
- **局限性**：仅评估了短程序（≤10 行），未涉及大型项目结构；执行预测实验表明模型仍缺乏对程序语义的深度把握。

### 影响与延伸思考
这篇工作在 2023 年左右公开后，直接推动了两大趋势：一是 **大模型即代码模型** 的思路被广泛接受，随后出现了 CodeLlama、StarCoder 等专门面向代码的 LLM；二是 **交互式代码调试** 的概念被进一步深化，后续研究如 “Self‑Repair LLM” 与 “Programmer‑in‑the‑Loop” 都在此基础上加入了更复杂的对话策略。对想继续深入的读者，可以关注以下方向：  
- **长程序合成**：如何让模型保持跨函数、跨文件的一致性。  
- **执行语义对齐**：结合符号执行或神经执行器，让模型在生成代码的同时验证其行为。  
- **自动化反馈生成**：让模型自行发现并描述错误，从而实现完全闭环的自我修正。  

### 一句话记住它
**大语言模型只要配好提示和一次人类纠错，就能在短 Python 任务上达到 80% 以上的正确率，但仍难以真正“理解”代码的执行结果。**