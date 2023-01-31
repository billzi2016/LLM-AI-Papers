# Faithful Chain-of-Thought Reasoning

> **Date**：2023-01-31
> **arXiv**：https://arxiv.org/abs/2301.13379

## Abstract

While Chain-of-Thought (CoT) prompting boosts Language Models' (LM) performance on a gamut of complex reasoning tasks, the generated reasoning chain does not necessarily reflect how the model arrives at the answer (aka. faithfulness). We propose Faithful CoT, a reasoning framework involving two stages: Translation (Natural Language query $\rightarrow$ symbolic reasoning chain) and Problem Solving (reasoning chain $\rightarrow$ answer), using an LM and a deterministic solver respectively. This guarantees that the reasoning chain provides a faithful explanation of the final answer. Aside from interpretability, Faithful CoT also improves empirical performance: it outperforms standard CoT on 9 of 10 benchmarks from 4 diverse domains, with a relative accuracy gain of 6.3% on Math Word Problems (MWP), 3.4% on Planning, 5.5% on Multi-hop Question Answering (QA), and 21.4% on Relational Inference. Furthermore, with GPT-4 and Codex, it sets the new state-of-the-art few-shot performance on 7 datasets (with 95.0+ accuracy on 6 of them), showing a strong synergy between faithfulness and accuracy.

---

# 忠实思维链推理 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，Chain‑of‑Thought（思维链）提示已经证明能让模型在数学、规划、推理等复杂任务上取得更高的准确率。可是，模型输出的文字推理链往往是“自说自话”，它们不一定是真正驱动答案的计算过程，导致解释不可信、错误难以追溯。更关键的是，LLM本身在精确数值计算和符号推理上仍然薄弱，靠纯语言生成往往会出现算术错误或逻辑漏洞。于是，研究者面临两个难题：①让推理链与最终答案保持一致（faithfulness），②在保持解释性的同时提升实际解题能力。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在给出答案前先写出一步步的推理，就像人在解题时先写草稿，帮助模型把复杂思考拆解成可见的子任务。  
- **Faithful（忠实）**：解释必须真实反映模型是如何得到答案的，不能是事后编造的“合理化”。可以想象为法律证词必须与实际行动相符。  
- **Symbolic Reasoning（符号推理）**：使用明确的符号（如算式、程序代码）来表达推理步骤，类似数学老师在黑板上写公式，而不是口头描述。  
- **Deterministic Solver（确定性求解器）**：一个不依赖随机性的工具（如数学求解器、规划引擎），对给定的符号链执行严格计算，保证每次得到相同的答案。  
- **Few‑Shot Prompting（少样本提示）**：在提示中只给出极少量示例，让模型自行归纳出通用策略，类似老师只给几道例题让学生学会解题方法。  
- **Multi‑hop QA（多跳问答）**：需要跨越多个事实或步骤才能得到答案的问答任务，像拼图一样需要把多块信息拼在一起。  
- **Relational Inference（关系推理）**：判断实体之间的关系，需要在图结构或逻辑规则上进行推演，类似推理游戏里找出谁是凶手的过程。  

### 核心创新点
1. **思维链从自然语言到符号链的翻译**  
   - 之前的思维链直接让模型输出文字解释，缺乏可执行性。  
   - 本文让模型先把自然语言问题翻译成一段可执行的符号推理链（如 Python 代码或数学表达式）。  
   - 这样得到的链条可以交给确定性求解器执行，保证解释与答案同步，提高了可信度和准确率。  

2. **推理链与答案分离的两阶段框架**  
   - 传统方法把“思考”和“回答”混在一起，错误往往在生成阶段就混进答案。  
   - 本文把任务拆成“翻译阶段”（LLM 负责生成符号链）和“求解阶段”（确定性求解器负责算出答案），两者各司其职。  
   - 结果是模型的语言生成只需关注结构化表达，计算错误被求解器彻底消除，整体性能显著提升。  

3. **在少样本设置下的强协同**  
   - 过去少样本提示往往只能略微提升表现，尤其在数学等高精度任务上效果有限。  
   - 通过让少量示例展示如何把自然语言映射为符号代码，LLM 学会了通用的翻译模式；随后求解器负责精确计算，两者形成“语言+工具”的协同效应。  
   - 在 7 个公开数据集上实现了新的 few‑shot 最佳记录，6 个数据集的准确率突破 95%。  

4. **跨领域统一的评估**  
   - 之前的思维链改进多集中在单一任务（如数学）。  
   - 本文在四大类任务（数学文字题、规划、Multi‑hop QA、关系推理）共 10 个基准上进行实验，9 个基准优于标准思维链，展示了方法的通用性。  

### 方法详解
**整体框架**  
整个系统分为两步：①**翻译阶段**，使用大语言模型把用户的自然语言问题转换成一段结构化的符号推理链；②**求解阶段**，把这段符号链交给一个确定性求解器（如 Python 解释器、数学求解库或规划求解器）执行，得到最终答案。两步之间没有信息损失，因为符号链本身就完整记录了所有中间变量和操作。

**步骤拆解**  

1. **提示设计**  
   - 提示中提供 1‑3 个少样本示例，每个示例展示：  
     - 原始问题（自然语言）  
     - 对应的符号推理链（例如 `x = 3 * (7 - 2)`）  
     - 求解器执行后得到的答案  
   - 这种“问题 → 代码 → 结果”的三段式让模型学习到“把语言映射成代码”的模式。

2. **LLM 生成符号链**  
   - 模型在接收到新问题后，依据提示生成一段代码或数学表达式。  
   - 为防止模型在生成过程中混入自然语言解释，提示中明确要求“仅输出可直接运行的代码”。  
   - 生成的符号链可以是 Python、SymPy、Z3 之类的 DSL（领域特定语言），具体取决于任务类型。

3. **确定性求解器执行**  
   - 系统把模型输出的代码送入安全的沙箱环境，调用对应的解释器或求解库。  
   - 求解器对每一步进行严格计算，若出现运行错误（如除零），系统会捕获并返回错误信息，供后续调试。  
   - 计算完成后，求解器返回唯一的数值或布尔答案。

4. **答案回传**  
   - 最终答案直接返回给用户，整个过程的解释链（代码）可以一起展示，保证解释的可追溯性。

**关键细节**  
- **语言模型的角色**：只负责“语言到符号”的映射，不需要进行精确计算，这大幅降低了对模型算术能力的依赖。  
- **求解器的确定性**：因为求解器是传统的符号计算工具，它的输出是可复现且严格符合数学规则的，这正是实现“faithful”解释的核心。  
- **安全沙箱**：防止生成的代码执行恶意操作，确保系统在开放环境下也能安全运行。  
- **反直觉点**：把原本被视为“语言模型的全部能力”拆分出来，让一个非神经网络的工具承担计算，实际上提升了整体性能，这在当时的 LLM 研究中并不常见。

### 实验与效果
- **测试任务**：四大类共 10 个基准，包括 Math Word Problems（数学文字题）、Planning（规划任务）、Multi‑hop Question Answering（多跳问答）和 Relational Inference（关系推理）。  
- **对比基线**：标准 Chain‑of‑Thought（直接让模型输出文字推理链）以及其他少样本提示方法。  
- **整体提升**：在 10 个基准中有 9 个超过标准 CoT，平均相对准确率提升约 6.3%（数学）、3.4%（规划）、5.5%（多跳 QA）和 21.4%（关系推理）。  
- **Few‑shot SOTA**：使用 GPT‑4 与 Codex 作为翻译模型时，在 7 个数据集上刷新了 few‑shot 纪录，6 个数据集的准确率突破 95%。  
- **消融实验**：原文未详细描述各模块的消融结果，但作者报告称去掉求解器或改为纯语言生成会导致性能回落到传统 CoT 水平，说明两阶段设计是关键。  
- **局限性**：方法依赖于能够为特定任务提供合适的符号求解器；对于没有成熟求解器的领域（如开放式创意写作）仍然难以直接套用。作者也提到，生成的代码质量受限于提示质量，若少样本示例不够典型，翻译阶段可能出现语法错误。

### 影响与延伸思考
- 这篇工作开启了“语言模型 + 可执行工具” 的新潮流，随后出现了大量把 LLM 当作前端、把外部工具（搜索引擎、数据库、数学引擎）当作后端的系统，如 ReAct、Toolformer 等。  
- 在数学与程序化推理社区，很多后续研究直接借鉴了“翻译 → 求解” 的两阶段思路，进一步探索自动生成更复杂的程序（如递归函数）并交给专门的解释器执行。  
- 对想深入的读者，可以关注以下方向：①如何让 LLM 自动选择最合适的求解器（工具选择问题）；②在没有明确符号语言的任务上，如何构造可执行的中间表示；③安全沙箱与代码审计技术的进步。  
- 该论文的思路也被用于提升大模型在法律、金融等需要高可信解释的行业的可用性，属于“可解释 AI” 与 “AI + 传统算法” 融合的早期里程碑。

### 一句话记住它
把语言模型的推理链翻译成可执行代码，再交给确定性求解器，让解释既可信又能提升答案准确率。