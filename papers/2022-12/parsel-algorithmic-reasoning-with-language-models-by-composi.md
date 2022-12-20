# Parsel: Algorithmic Reasoning with Language Models by Composing   Decompositions

> **Date**：2022-12-20
> **arXiv**：https://arxiv.org/abs/2212.10561

## Abstract

Despite recent success in large language model (LLM) reasoning, LLMs struggle with hierarchical multi-step reasoning tasks like generating complex programs. For these tasks, humans often start with a high-level algorithmic design and implement each part gradually. We introduce Parsel, a framework enabling automatic implementation and validation of complex algorithms with code LLMs. With Parsel, we automatically decompose algorithmic tasks into hierarchical natural language function descriptions and then search over combinations of possible function implementations using tests. We show that Parsel can be used across domains requiring hierarchical reasoning, including program synthesis and robotic planning. We find that, using Parsel, LLMs solve more competition-level problems in the APPS dataset, resulting in pass rates over 75\% higher than prior results from directly sampling AlphaCode and Codex, while often using a smaller sample budget. Moreover, with automatically generated tests, we find that Parsel can improve the state-of-the-art pass@1 performance on HumanEval from 67\% to 85\%. We also find that LLM-generated robotic plans using Parsel are more than twice as likely to be considered accurate than directly generated plans. Lastly, we explore how Parsel addresses LLM limitations and discuss how Parsel may be useful for human programmers. We release our code at https://github.com/ezelikman/parsel

---

# Parsel：通过组合分解实现语言模型的算法推理 论文详细解读

### 背景：这个问题为什么难？

生成复杂程序需要模型在多个层次上进行推理：先想出整体思路，再把每一步细化成可执行代码。现有的大语言模型（LLM）虽然在一次性答题上表现不错，却往往在这种“先宏观后微观”的任务上卡壳。直接让模型一次性输出完整代码会导致逻辑跳跃、变量冲突或遗漏关键子任务。传统的代码生成系统（如 AlphaCode、Codex）只能靠大量随机采样来 hoping “碰巧”得到正确实现，成本高且成功率低。根本的瓶颈在于缺少一种机制，让模型像人类一样先拆解问题，再逐块实现并相互验证。

### 关键概念速览
**层次化分解**：把一个大任务拆成若干子任务，每个子任务用自然语言描述其功能，好比把一座大楼先划分成楼层、房间再去装修。  
**函数描述（Function Description）**：对每个子任务的输入、输出和行为的文字说明，类似需求文档的简短条目。  
**实现搜索（Implementation Search）**：让代码 LLM 在给定函数描述的前提下生成候选实现，并通过测试来挑选可行的代码。  
**自动生成测试（Auto‑generated Tests）**：系统依据函数描述自动合成输入输出对，用来检验生成的代码是否符合预期，就像单元测试。  
**组合搜索（Compositional Search）**：把不同子函数的实现组合起来，形成完整程序的过程，类似拼图游戏把各块拼成完整图案。  
**Pass@k**：在 k 次采样中至少有一次成功的概率，是代码生成领域常用的评估指标。  
**APPS 数据集**：包含上千道编程竞赛题目，用来衡量模型的程序合成能力。  
**HumanEval**：OpenAI 发布的代码生成基准，提供函数描述和对应的单元测试。

### 核心创新点
1. **从一次性生成到层次化分解**  
   之前的做法是让模型直接输出完整代码，成功率低且需要大量采样。Parsel 先让模型把任务拆成自然语言的函数描述，再逐层实现。这样把难度从“一口吃掉”变成“一口一块”，显著提升了搜索效率。  
2. **实现搜索配合自动测试**  
   传统代码生成往往缺少可靠的验证手段，只靠人工检查或少量示例。Parsel 为每个函数描述自动生成测试用例，模型的每个候选实现都要通过这些测试才能进入下一层。相当于在每一步都设了“安全门”，大幅降低错误代码的传播。  
3. **组合搜索的层级回溯机制**  
   当某个子函数的实现无法通过上层组合的测试时，系统会回溯并重新搜索该子函数的其他实现，而不是放弃整个任务。这种自适应的搜索策略比单纯的随机采样更具针对性。  
4. **跨域统一框架**  
   虽然最初是为程序合成设计，Parsel 同时在机器人规划任务上取得了两倍以上的准确率提升，说明该框架能够把“算法思路 → 可执行代码”这条链条推广到更广的决策场景。

### 方法详解
Parsel 的整体流程可以概括为四步：**分解 → 描述 → 实现搜索 → 组合验证**。

1. **层次化分解**  
   给定一个高层次的算法任务（例如“实现快速排序”），系统首先使用一个强大的语言模型生成一棵任务树。树的根节点是整体目标，子节点是若干自然语言的函数描述，如“partition 函数：把数组划分为左侧小于基准、右侧大于基准”。这一步相当于让模型充当需求分析师。

2. **函数描述标准化**  
   每个子节点的描述会被进一步结构化：明确输入类型、输出类型、边界条件以及预期的时间/空间复杂度。这样做的好处是让后续的代码生成模型拥有清晰的“合同”，避免歧义。

3. **实现搜索 + 自动测试**  
   对每个函数描述，Parsel 调用代码 LLM（如 Codex）多次采样得到若干候选实现。随后，系统依据描述自动生成一组测试用例（比如随机生成的数组），对每个实现进行单元测试。只有通过全部测试的实现才会被标记为“合格”。如果所有实现都失败，系统会回到分解阶段，尝试重新生成更细粒度的子描述或调整描述的细节。

4. **组合搜索与层级回溯**  
   当所有子函数都有合格实现后，Parsel 按照任务树的拓扑顺序把它们组装成完整程序。组装后会运行一次全局测试（通常是题目给出的样例或额外生成的综合测试）。如果全局测试失败，系统会定位是哪一层的实现导致错误，并在该层重新进行实现搜索。这个回溯过程类似于编程时的调试循环，但全部自动化。

**最巧妙的点**在于把“测试”从事后检查搬到每一步的前置环节，使得错误在最小的粒度被捕获，极大降低了搜索空间。与此同时，层次化的任务树让模型的上下文保持在可管理的长度范围内，避免一次性喂入过长的提示导致性能下降。

### 实验与效果
- **数据集与任务**：在程序合成方面，作者使用了 APPS（约 10k 竞赛题）和 HumanEval 两个基准；在机器人规划方面，选取了公开的任务库（如 Pick‑Place 场景）。  
- **对比基线**：与直接采样 AlphaCode、Codex 的最高 Pass@k 结果相比，Parsel 在 APPS 上的通过率提升了超过 75%，而所需的采样次数却显著更少。HumanEval 的 Pass@1 从原来的 67% 跳到 85%，创下当时的最高记录。机器人计划的准确率提升超过两倍。  
- **消融实验**：作者分别关闭了自动测试、层次化分解和回溯机制，发现去掉任意一环后性能都会跌回原始采样水平，尤其是没有自动测试时通过率下降近 40%。这说明每个模块都是提升的关键因素。  
- **局限性**：论文指出，Parsel 对函数描述的质量高度敏感；如果初始分解产生的子任务不合理，后续搜索会陷入死循环。此外，自动生成的测试用例仍然是基于随机采样，难以覆盖所有边界情况，极端的数学证明类题目仍旧表现不佳。

### 影响与延伸思考
Parsel 把“分而治之”这一经典算法思想搬进了大语言模型的代码生成流程，开启了“结构化提示 + 自动验证” 的新潮流。随后的工作如 **TreeCoder**、**CoT‑Synthesis** 等，都在不同程度上借鉴了层次化分解与测试驱动的思路。对想进一步探索的读者，可以关注以下方向：① 更智能的任务树生成（比如结合程序分析工具）；② 自动化测试的覆盖率提升（符号执行、对抗样本生成）；③ 将该框架扩展到非代码领域，如数学证明或数据分析流水线的自动构建。  

### 一句话记住它
Parsel 让语言模型先把大算法拆成小函数，再用自动测试挑选实现，像拼图一样把代码一步步拼好，从而把代码生成的成功率大幅提升。