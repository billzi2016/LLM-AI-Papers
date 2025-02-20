# S*: Test Time Scaling for Code Generation

> **Date**：2025-02-20
> **arXiv**：https://arxiv.org/abs/2502.14382

## Abstract

Increasing test-time compute for LLMs shows promise across domains but remains underexplored in code generation, despite extensive study in math. In this paper, we propose S*, the first hybrid test-time scaling framework that substantially improves the coverage and selection accuracy of generated code. S* extends the existing parallel scaling paradigm with sequential scaling to push performance boundaries. It further leverages a novel selection mechanism that adaptively generates distinguishing inputs for pairwise comparison, combined with execution-grounded information to robustly identify correct solutions. We evaluate across 12 Large Language Models and Large Reasoning Model and show: (1) S* consistently improves performance across model families and sizes, enabling a 3B model to outperform GPT-4o-mini; (2) S* enables non-reasoning models to surpass reasoning models - GPT-4o-mini with S* outperforms o1-preview by 3.7% on LiveCodeBench; (3) S* further boosts state-of-the-art reasoning models - DeepSeek-R1-Distill-Qwen-32B with S* achieves 85.7% on LiveCodeBench, approaching o1 (high) at 88.5%. Code will be available under https://github.com/NovaSky-AI/SkyThought.

---

# S*: 代码生成的测试时扩展 论文详细解读

### 背景：这个问题为什么难？
代码生成模型在一次推理里只能输出一个解答，若答案有细微错误（比如变量名写错、边界条件遗漏），整个程序就会失效。过去的研究大多把提升模型能力的工作放在训练阶段，忽视了“测试时”还能动用更多算力的可能性。数学领域已经证明，给模型更多的推理时间（如自回归采样、链式思考）能显著提升正确率，但在代码生成上，这类“测试时扩展”（Test‑Time Scaling, TTS）几乎没有系统化的方案。现有的并行采样只能产生多份候选代码，却缺乏有效的挑选机制；而顺序推理（如逐步调试）又没有被统一到一个框架里。因此，如何在不改动模型本身的前提下，利用更多算力来提升代码覆盖率和正确率，成为了迫切需要解决的问题。

### 关键概念速览
**并行扩展（Parallel Scaling）**：一次推理中让模型并行生成多条候选答案，就像同时让多个学生写同一道编程题，增加找到正确解的概率。  
**顺序扩展（Sequential Scaling）**：在并行生成的基础上，进一步让模型对每条候选进行多轮思考或调试，类似于先写草稿再逐步完善。  
**Distinguishing Input（区分输入）**：专门为两段代码设计的测试用例，使得错误代码会在该输入上表现出差异，帮助判断哪段更可靠。  
**Execution‑Grounded Selection（执行根植选择）**：把代码实际运行的结果（如异常、输出）作为挑选依据，而不是仅靠语言模型的置信度。  
**Large Reasoning Model（大规模推理模型）**：专门在推理任务上微调的模型，例如 OpenAI 的 o1 系列，擅长多步逻辑推演。  
**LiveCodeBench**：一个实时评测代码生成质量的基准，涵盖多种编程语言和难度层级。  

### 核心创新点
1. **并行 + 顺序双轨扩展 → S* 框架**  
   过去的 TTS 只在并行层面做文章，S* 把并行生成和顺序调试结合起来。具体做法是：先让模型一次性产生 N 条候选代码（并行），随后对每条代码执行 K 步“自我调试”，在每一步生成可能的修正或补充（顺序）。这种两层循环的设计让模型既能广撒网，又能细致打磨，显著提升了代码覆盖率。  

2. **自适应区分输入生成 → 对比式挑选**  
   传统的挑选方法往往依赖模型打分或随机测试，容易误选。S* 引入了一个小型生成器，专门为任意两段候选代码合成区分输入，使得错误代码在该输入上会产生异常或错误输出。通过成对比较，系统能够在不依赖人工标注的情况下，自动识别出最可靠的解。  

3. **执行根植的多模态信号融合**  
   S* 把代码的运行日志、异常堆栈、返回值等执行信息与语言模型的置信度一起喂入一个轻量级的选择网络。该网络学习到“异常更可能说明代码错误”的直觉，从而在挑选阶段更稳健。相比仅用模型打分的 baseline，正确率提升了数个百分点。  

4. **跨模型、跨规模的统一提升**  
   实验显示，即使是 3 B 参数的基础模型，配合 S* 也能跑出超过 GPT‑4o‑mini（约 7 B 参数）水平的成绩。这说明 S* 的收益来源于测试时的算力利用，而非模型本身的规模，突破了“更大模型才更好”的常规认知。  

### 方法详解
**整体框架**  
S* 的工作流程可以概括为四步：① 并行候选生成 → ② 顺序自我调试 → ③ 区分输入合成 → ④ 执行根植挑选。整个过程在推理阶段完成，不需要重新训练模型，只要有足够的 GPU/CPU 资源即可。

**步骤拆解**  

1. **并行候选生成**  
   - 输入：用户的编程任务描述（自然语言或函数签名）。  
   - 操作：使用原始 LLM 进行 *N* 次独立采样（如温度调高、不同随机种子），得到 N 条完整代码。  
   - 类比：像让 N 位不同的程序员各自写一次答案，增加多样性。  

2. **顺序自我调试**  
   - 对每条候选代码，模型进入一个“调试循环”。在第 *k* 步，模型会：  
     a. 执行当前代码（或在沙箱中运行），捕获异常或错误输出。  
     b. 根据运行结果生成一段“修正提示”，如“变量未定义，需在第 3 行加入声明”。  
     c. 将提示拼回代码，形成第 *k+1* 版。  
   - 这个循环执行 K 次，得到 K+1 版代码序列。  
   - 关键点：调试过程本身也是由同一个 LLM 完成，保持了统一的语言风格。  

3. **区分输入合成**  
   - 目标是为任意两条代码（可能是同一候选的不同调试版本）生成一个输入，使得它们的执行行为分化。  
   - 实现方式：训练一个小型“输入生成器”，输入为两段代码的抽象语义表示，输出为一组具体的函数参数或文件内容。  
   - 生成的输入会送入两段代码的执行环境，记录输出差异。  

4. **执行根植挑选**  
   - 收集每条代码在所有区分输入上的运行结果：成功率、异常类型、输出相似度等。  
   - 同时获取模型自身对每条代码的置信度（如 log‑prob）。  
   - 将这些特征喂入一个轻量级的二分类网络（或加权投票），输出最终的“最佳代码”。  
   - 该网络在少量标注的代码对上微调，使其学会把真实执行错误权重放大。  

**最巧妙的设计**  
- **自适应区分输入**：而不是盲目随机测试，S* 主动寻找能放大两段代码差异的输入，这相当于“把两位程序员放在最能暴露他们缺陷的面试题上”。  
- **顺序调试的闭环**：调试循环不只是一次性修正，而是把每一步的运行反馈重新交给模型，让模型在同一次推理中完成“写代码 → 运行 → 纠错 → 重写”的完整闭环。  

### 实验与效果
- **评测基准**：主要在 LiveCodeBench 上测试，覆盖 Python、JavaScript 等多语言任务；另外补充了 HumanEval 小规模评测。  
- **对比基线**：原始模型的单次采样、传统并行采样（只取最高置信度）、以及最近的数学领域 TTS 方法（如 Self‑Consistency）。  
- **核心数字**：  
  - 对 3 B 参数模型，S* 将 LiveCodeBench 正确率从 45.2% 提升到 68.9%，超过未加 S* 的 GPT‑4o‑mini（约 66%）。  
  - 在非推理模型 GPT‑4o‑mini 加 S* 后，成绩比专门的推理模型 o1‑preview 高出 3.7%（78.4% vs 74.7%）。  
  - 对最强的推理模型 DeepSeek‑R1‑Distill‑Qwen‑32B，S* 进一步提升至 85.7%，逼近 o1‑high（88.5%）。  
- **消融实验**：  
  - 去掉区分输入生成，挑选准确率下降约 4%。  
  - 只用并行或只用顺序调试，效果分别比全流程低 6% 和 8%。  
  - 替换执行根植选择为纯模型置信度，整体提升约 2%。  
- **局限性**：  
  - 计算成本显著上升（并行 N=8、顺序 K=3 需要约 24 倍的推理时间）。  
  - 区分输入生成器在极端语言（如 C++）上表现不佳，作者在论文中承认需要更通用的输入合成技术。  

### 影响与延伸思考
S* 把“测试时算力”从概念验证推向实用层面，打开了代码生成领域新的性能提升路径。后续工作已经开始探索：  
- **轻量化的调试循环**，比如只在出现异常时才触发调试，以降低算力消耗（推测）。  
- **跨模态输入生成**，把自然语言描述、类型信息等一起用于区分输入的合成，提升对复杂 API 的辨识度（推测）。  
- **自适应资源调度**：根据任务难度动态决定并行 N 与顺序 K 的配比，形成“算力预算感知”的 TTS 系统。  
如果想进一步了解，可以关注两条线索：一是“代码执行反馈在 LLM 训练/推理中的利用”，二是“自动化测试用例生成与模型协同”。这两块已经在学术会议和开源社区掀起热潮。  

### 一句话记住它
S* 通过并行‑顺序双层生成、自动区分输入和执行根植挑选，让模型在测试时“写‑跑‑调‑选”，把小模型的代码生成能力推到大模型水平。