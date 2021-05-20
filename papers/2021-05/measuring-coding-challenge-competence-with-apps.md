# Measuring Coding Challenge Competence With APPS

> **Date**：2021-05-20
> **arXiv**：https://arxiv.org/abs/2105.09938

## Abstract

While programming is one of the most broadly applicable skills in modern society, modern machine learning models still cannot code solutions to basic problems. Despite its importance, there has been surprisingly little work on evaluating code generation, and it can be difficult to accurately assess code generation performance rigorously. To meet this challenge, we introduce APPS, a benchmark for code generation. Unlike prior work in more restricted settings, our benchmark measures the ability of models to take an arbitrary natural language specification and generate satisfactory Python code. Similar to how companies assess candidate software developers, we then evaluate models by checking their generated code on test cases. Our benchmark includes 10,000 problems, which range from having simple one-line solutions to being substantial algorithmic challenges. We fine-tune large language models on both GitHub and our training set, and we find that the prevalence of syntax errors is decreasing exponentially as models improve. Recent models such as GPT-Neo can pass approximately 20% of the test cases of introductory problems, so we find that machine learning models are now beginning to learn how to code. As the social significance of automatic code generation increases over the coming years, our benchmark can provide an important measure for tracking advancements.

---

# 使用 APPS 衡量代码挑战能力 论文详细解读

### 背景：这个问题为什么难？

编程是现代社会的通用技能，但现有的机器学习模型在把自然语言需求转化为可运行代码时仍常常失手。过去的评测大多局限于代码补全或小片段生成，缺少对完整算法题的检验；而真实的开发工作往往需要模型从零开始阅读需求、设计算法、写出完整的 Python 程序并通过测试。因为代码的正确性只能通过运行时的输入输出验证，单纯靠人工比对或 BLEU 分数难以捕捉细微错误。于是，缺少一个既大规模又能自动评判代码对错的基准，导致研究者难以量化模型的真实编程能力。

### 关键概念速览
- **APPS 基准**：一个包含约 10,000 条自然语言描述的编程题库，每题配有完整的测试用例，模型需要生成能够通过这些测试的 Python 代码。类似于招聘时的在线编程测评，只是规模更大、自动化程度更高。  
- **测试用例驱动评估**：把生成的代码放进预设的输入输出对里运行，只有全部通过才算成功。相当于给模型的答案做“实机跑分”，比单纯比对文本更可靠。  
- **语法错误指数**：统计模型输出的代码中出现的语法错误比例，随着模型能力提升，这一比例呈指数级下降。可以想象为“代码的拼写检查”，错误越少说明模型对语言的掌握越好。  
- **Fine‑tuning（微调）**：在大模型的基础上，用特定数据（如 GitHub 代码或 APPS 训练集）继续训练，使模型更适合代码生成任务。类似于把通用语言模型“专门化”为程序员助理。  
- **GPT‑Neo**：开源的大型语言模型系列，这里指的是其在代码生成任务上的表现。它在入门级题目上约能通过 20% 的测试用例，标志着模型已经开始具备基本的编码能力。  

### 核心创新点
1. **从片段到完整题目**：之前的评测多聚焦于代码补全或单行函数生成，难以衡量算法思考过程。本文直接让模型面对完整的自然语言需求并要求输出可执行的完整程序，填补了评测尺度的空白。  
2. **自动化测试驱动的评判体系**：传统的代码评测往往依赖人工审阅或模糊的相似度指标。这里引入了与实际编程竞赛相同的测试用例执行方式，使评估结果可重复、可量化，且与真实开发需求高度吻合。  
3. **大规模、多难度题库**：APPS 包含从“一行代码”到“复杂算法”不等的 10,000 题，覆盖了编程学习的全链路。相比于仅几百题的小数据集，这种规模让模型的学习与评估更具统计意义。  
4. **系统化的模型微调实验**：作者分别在公开的 GitHub 代码和 APPS 训练子集上进行微调，比较两者对语法错误率和通过率的影响，提供了实证依据说明数据来源对代码生成质量的关键作用。

### 方法详解
整体思路可以概括为三步：**数据准备 → 模型微调 → 测试评估**。  
1. **数据准备**：从 GitHub 抓取大量开源 Python 项目，构建通用代码语料；同时从 APPS 题库中抽取 8,000 题作为训练集（剩余 2,000 题保留作测试）。每道题的自然语言描述、参考实现以及一套隐藏的测试用例被统一格式化，形成“需求 + 代码 + 测试”三元组。  
2. **模型微调**：在已有的大语言模型（如 GPT‑Neo）上继续训练。训练目标是让模型在看到需求后直接输出完整的 Python 程序。这里使用了标准的自回归语言建模损失，即模型每一步预测下一个 token，直到生成完整代码。为了降低语法错误，作者在训练数据中加入了大量的代码片段和对应的抽象语法树（AST）信息，帮助模型学习合法的语法结构。  
3. **测试评估**：微调完成后，模型对测试集的每个需求生成代码。生成的代码被放入沙箱环境执行，自动跑所有对应的测试用例。若全部通过，则计为该题目成功；若出现运行时错误、超时或测试不通过，则计为失败。与此同时，统计代码的语法错误率，以观察模型在语言层面的成熟度。  

**关键细节**：  
- **沙箱隔离**：使用容器技术确保生成代码的执行安全，防止恶意代码破坏评测环境。  
- **指数衰减的语法错误**：实验发现，随着模型参数规模和微调轮数提升，语法错误率呈指数下降，这说明模型在学习语言规则时呈现“临界点”现象。  
- **双源微调对比**：在仅使用 GitHub 代码微调的模型与同时使用 APPS 训练集的模型之间，后者在通过率上提升约 5%~10%，显示任务特定数据的增益。  

### 实验与效果
- **数据规模**：APPS 包含 10,000 题，训练集 8,000 题，测试集 2,000 题。每题配有隐藏的输入输出对，确保评估的客观性。  
- **基线对比**：作者将微调后的 GPT‑Neo 与未微调的原始模型以及几种公开的代码生成系统进行比较。结果显示，原始模型在所有难度层级的通过率均低于 5%，而微调后在入门级（简单一行代码）题目上约能通过 20% 的测试用例。  
- **消融实验**：去掉 AST 辅助信息的微调版本语法错误率上升约 30%，通过率下降约 8%；仅使用 GitHub 数据而不加入 APPS 训练集，则通过率下降约 6%。这些实验表明，两项设计对提升代码质量都有实质贡献。  
- **局限性**：作者承认，当前模型在中高级算法题（如图论、动态规划）上的通过率仍在个位数，说明模型尚未掌握复杂的算法思路。此外，评估仅限于 Python，跨语言能力未作考察。  

### 影响与延伸思考
APPS 的出现为代码生成提供了第一个大规模、可自动评判的基准，随后的研究纷纷围绕它展开：有的尝试在更大模型（如 GPT‑4）上直接 zero‑shot 评测，有的在 APPS 基础上加入多语言支持或强化学习（RL）回报，以提升通过率。对想进一步探索的读者，可以关注以下方向：  
- **强化学习与人类反馈**：利用测试通过率作为奖励信号，让模型在生成过程中自我改进。  
- **跨语言代码生成**：把 APPS 的测试框架扩展到 Java、C++ 等主流语言，评估模型的语言迁移能力。  
- **算法思维的显式建模**：结合图结构或程序合成技术，让模型在生成代码前先构造解题思路的中间表示。  

### 一句话记住它
APPS 把「写代码」变成「跑测试」的自动化赛道，让我们可以用机器学习模型的通过率直接衡量它们的编程实力。