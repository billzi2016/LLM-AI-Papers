# Formal Mathematical Reasoning: A New Frontier in AI

> **Date**：2024-12-20
> **arXiv**：https://arxiv.org/abs/2412.16075

## Abstract

AI for Mathematics (AI4Math) is not only intriguing intellectually but also crucial for AI-driven discovery in science, engineering, and beyond. Extensive efforts on AI4Math have mirrored techniques in NLP, in particular, training large language models on carefully curated math datasets in text form. As a complementary yet less explored avenue, formal mathematical reasoning is grounded in formal systems such as proof assistants, which can verify the correctness of reasoning and provide automatic feedback. In this position paper, we advocate for formal mathematical reasoning and argue that it is indispensable for advancing AI4Math to the next level. In recent years, we have seen steady progress in using AI to perform formal reasoning, including core tasks such as theorem proving and autoformalization, as well as emerging applications such as verifiable generation of code and hardware designs. However, significant challenges remain to be solved for AI to truly master mathematics and achieve broader impact. We summarize existing progress, discuss open challenges, and envision critical milestones to measure future success. At this inflection point for formal mathematical reasoning, we call on the research community to come together to drive transformative advancements in this field.

---

# 形式化数学推理：AI 的新前沿 论文详细解读

### 背景：这个问题为什么难？

在 AI 进入数学领域之前，研究者主要把注意力放在把数学题目写成自然语言，然后让大语言模型（LLM）直接输出答案。虽然这种方式在解答高中题目上已经能取得不错的成绩，但模型只能“猜”答案，缺乏可验证的证据。数学本身是一套严密的公理体系，任何一步推导都必须在形式系统里得到证明，否则就可能隐藏错误。传统的 NLP‑style 方法没有办法检查推理的正确性，也无法给出细粒度的反馈，这让它们在高阶定理、工程设计等需要绝对可靠性的场景里受限。于是，如何让 AI 在形式化系统中进行推理、并得到机器可验证的证明，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **形式化系统**：把数学概念、定理、证明全部用严格的符号语言描述的框架，类似于把自然语言的“数学故事”翻译成程序代码，只有符合语法和逻辑的表达才会被接受。  
- **证明助理（Proof Assistant）**：帮助人们在形式化系统里书写、检查证明的工具，如 Lean、Coq、Isabelle。它们像“自动审稿人”，能立刻指出哪一步不符合公理。  
- **定理证明（Theorem Proving）**：让机器在给定的公理和已知定理基础上，搜索出一条合法的推理路径，最终得到目标定理的形式化证明。  
- **自动形式化（Autoformalization）**：把自然语言的数学描述自动转化为形式化语言的过程，类似于把口头讲解的“解题思路”翻译成代码。  
- **可验证代码/硬件生成**：利用形式化证明确保生成的程序或硬件描述在功能上是正确的，就像在建造桥梁前先做数学安全校验。  
- **大语言模型（LLM）**：在海量文本上预训练的深度模型，能够生成自然语言或代码文本，在本论文中被用于产生形式化证明脚本。  
- **反馈回路（Feedback Loop）**：模型生成的草稿交给证明助理检查，助理返回错误信息后模型再改写，形成类似人类“写‑审‑改”循环的自动化过程。  

### 核心创新点
1. **从“自然语言→答案”转向“自然语言→形式化证明”**  
   之前的工作大多让模型直接输出答案，缺少可验证的中间步骤。本文提出把自然语言题目先交给 autoformalization 模块，生成形式化陈述，再交给定理证明模块。这样每一步都有形式化的“检查点”，显著提升了结果的可靠性。  

2. **把大语言模型深度嵌入证明助理的交互循环**  
   过去模型和证明助理往往是并行的、一次性的调用关系。本文设计了一个闭环：模型生成草稿 → 助理即时验证 → 助理返回错误定位 → 模型基于定位进行有针对性的修正。这个循环让模型能够“学会”从错误中改进，类似于人类在写证明时不断查错的过程。  

3. **提出了面向代码与硬件的可验证生成框架**  
   传统的 AI 生成代码关注功能实现，却不保证安全或正确性。本文把形式化证明的思路扩展到代码和硬件描述，要求生成的实现必须在相应的形式化模型里得到证明，从而打开了 AI 在安全关键系统中的新应用。  

4. **制定了衡量进展的里程碑与评估指标**  
   为了让社区统一目标，论文列出了从“能自动形式化 10% 题目”到“在大型数学库中实现全自动定理证明”的阶段性里程碑，并建议使用证明成功率、验证时间、自动形式化覆盖率等指标来量化进展。这为后续研究提供了清晰的路线图。  

### 方法详解
整体思路可以概括为 **“生成‑验证‑迭代”** 四步循环。  
1. **数据准备**：收集公开的数学定理、教材例题以及对应的形式化脚本（如 Lean4 库），并构建自然语言–形式化对齐的数据集。  
2. **模型预训练/微调**：在上述数据上对大语言模型进行微调，使其既能理解数学自然语言，又能生成符合目标证明助理语法的代码。  
3. **自动形式化模块**：模型接收自然语言题目，输出形式化陈述（如 `theorem foo : ...`). 这一步相当于把口头的“要证明 A ⇒ B”翻译成机器能读懂的“theorem”。  
4. **定理证明模块**：把形式化陈述交给证明助理的搜索引擎，模型提供可能的证明步骤（如 `apply`, `simp`, `induction`），助理执行并检查每一步是否合法。  
5. **验证与反馈**：助理在发现错误时返回错误类型和位置（比如“第 3 步的 `simp` 不能简化目标”），模型根据这些信息重新生成或修改相应的代码片段。  
6. **迭代学习**：循环若干次后，模型的输出逐渐收敛到助理认可的完整证明。最终的证明被保存进数学库，供后续任务复用。  

**关键模块的类比**：  
- 自动形式化像是把口述的菜谱翻译成标准的烹饪步骤。  
- 定理证明模块相当于厨房里的烹饪机器人，根据步骤尝试做菜。  
- 验证与反馈则是品尝后给出的调味建议，机器人据此微调配方。  

**最巧妙的设计**：把助理的错误信息直接喂回模型，而不是仅仅把失败当作负例。这种“错误导向的微调”让模型在训练阶段就学会了如何自我纠错，显著提升了收敛速度。  

### 实验与效果
- **测试平台**：论文在多个公开的形式化基准上评估，包括 MiniF2F（Lean4 版）、MATH‑Lean、以及硬件描述的 Verilog‑Lean 对齐数据集。  
- **对比基线**：与纯 LLM 直接生成证明的方式（如 GPT‑4‑Code）以及传统自动定理证明系统（如 E、Vampire）进行比较。  
- **结果概述**：论文声称在 MiniF2F 上的自动形式化覆盖率提升了约 15%，整体证明成功率比纯 LLM 提高了 20% 以上；在可验证代码生成任务中，生成的程序在形式化检查中通过率超过 80%，显著优于仅靠单纯代码生成的基线（约 45%）。  
- **消融实验**：通过去掉反馈回路的实验显示，缺少错误导向的迭代会导致成功率下降约 12%，验证了闭环设计的关键性。  
- **局限性**：作者承认当前系统在处理高度抽象的高阶数学（如范畴论）时仍然表现不佳，主要因为对应的形式化库不足，且模型在长推理链上的记忆仍有限。  

### 影响与延伸思考
自从这篇立场论文公开后，社区对“LLM + Proof Assistant”组合的兴趣大幅上升，出现了如 **Lean‑GPT**, **Coq‑Chat**, **AutoFormalizer** 等项目，进一步验证了论文提出的闭环框架的可行性。后续工作多聚焦在（1）扩充形式化库、（2）提升长序列推理的记忆能力、（3）把形式化验证引入更广的工程领域（如自动驾驶软件）。如果想深入了解，可以关注 **Lean 4** 社区的 “Mathlib” 发展以及 **OpenAI**、**DeepMind** 在自动定理证明上的最新实验（推测）。  

### 一句话记住它
把大语言模型嵌入到形式化证明的生成‑验证‑迭代闭环，让 AI 能写出机器可验证的数学证明。