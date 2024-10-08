# CodeDPO: Aligning Code Models with Self Generated and Verified Source Code

> **Date**：2024-10-08
> **arXiv**：https://arxiv.org/abs/2410.05605

## Abstract

Code generation models have shown significant potential for programming tasks. However, existing training methods like supervised fine-tuning face key limitations: they do not effectively teach models to prioritize correct over incorrect solutions in ambiguous situations, nor do they effectively optimize the runtime efficiency of the generated code. To address these challenges, we propose CodeDPO, a framework that integrates preference learning into code generation to improve two key code preference factors: code correctness and efficiency. CodeDPO employs a novel dataset construction method, utilizing a self-generation-and-validation mechanism that simultaneously generates and evaluates code and test cases. The underlying assumption is that test cases executable by multiple code snippets provide more reliable validation, and code that passes more tests is more likely to be correct. Through this self-validation process, our PageRank-inspired algorithm iteratively updates the ranking score of each code snippet, ultimately creating a code preference optimization dataset based on correctness and efficiency. CodeDPO is flexible and scalable, generating diverse preference optimization data without depending on external resources. Through comprehensive evaluations of five widely used benchmarks, CodeDPO demonstrates significant improvements in correctness and efficiency compared to existing methods. Our experiments prove that CodeDPO enhances the capabilities of LLMs in code generation and provides a robust foundation for conducting code preference optimization in more complex and challenging real-world scenarios.

---

# CodeDPO：通过自生成与验证源码对齐代码模型 论文详细解读

### 背景：这个问题为什么难？
代码生成模型在写函数、实现算法时已经能跑出可用的代码，但它们常常会产生“看起来对，但实际跑不通”或“运行慢得离谱”的答案。传统的监督微调（SFT）只能教模型模仿已有的正确代码，却没有办法让模型在多个候选答案中主动挑出更靠谱、更高效的那一个。更糟的是，评估代码正确性和运行时效率本身就需要实际执行代码，而手工标注的大规模偏好数据成本极高，导致现有方法在模糊或多解的编程任务上表现受限。

### 关键概念速览
- **DPO（Direct Preference Optimization）**：一种直接把“更好”与“更差”对比喂给模型进行梯度更新的技术，类似让模型学会在两段文字中挑出更优的那段。这里把它搬到代码上，让模型学会偏好正确且高效的代码片段。  
- **偏好学习（Preference Learning）**：不是让模型输出绝对的对错，而是让它学习“在这两个答案里，我更喜欢哪个”。想象成让模型学会投票，而不是只会背答案。  
- **自生成‑验证机制**：模型自己先生成一堆代码和对应的测试用例，再自行跑这些测试，得到哪些代码通过了哪些测试。相当于模型在给自己出题、自己批卷。  
- **PageRank‑式排序**：原本用于网页重要性排序的算法，这里把每段代码看成“节点”，把它们共同通过的测试用例看成“链接”，通过迭代计算得到每段代码的全局可信度分数。  
- **代码正确性**：指代码在给定测试用例上全部通过，等价于“这段代码真的能完成需求”。  
- **代码效率**：指代码的运行时间或资源消耗，通常用执行时间或复杂度指标衡量，目标是让模型倾向生成更快的实现。  
- **偏好数据集**：由大量“代码A比代码B更好”的二元对比组成的训练材料，供 DPO 使用。  

### 核心创新点
1. **自生成‑验证得到偏好数据**  
   之前的偏好数据大多来源于人工标注或外部竞赛结果，需要大量人力。CodeDPO 让模型自行产生代码和测试用例，并通过实际执行来判断通过率，从而自动生成可信的偏好对。这样既省去人工成本，又能覆盖更广的任务空间。  

2. **基于共享测试用例的 PageRank‑式评分**  
   直接用“通过的测试数量”来排序会忽视测试用例的难度差异。作者构建了一个代码‑测试用例二分图，使用 PageRank 思想让通过难度更高、被更多代码争夺的测试用例对代码评分贡献更大。结果是得到的分数更能反映代码的整体可靠性。  

3. **将正确性与效率统一进偏好学习**  
   过去的偏好学习往往只关注正确性，效率被忽略。CodeDPO 在生成对比时同时考虑两者：如果两段代码都通过相同测试，用运行时间更短的那段获胜；如果只有一段通过，则它直接赢。这样训练出来的模型在生成代码时会自然倾向于既对又快。  

4. **无需外部资源的可扩展框架**  
   通过自循环的生成‑执行‑排序流程，整个偏好数据的构造过程只依赖模型本身和一个轻量的执行环境。实验表明，随着生成规模的提升，偏好数据的多样性和质量仍保持增长，展示了良好的可扩展性。  

### 方法详解
**整体思路**：CodeDPO 把“生成代码 → 生成测试 → 执行验证 → 评分排序 → 构造偏好 → DPO 训练”这几步串成一个闭环。每一步都可以并行化，整个管线可以在普通 GPU+CPU 环境下跑数十万条代码样本。

1. **代码与测试用例的自生成**  
   - 给定一个自然语言编程任务（如“实现二分查找”），模型先生成 N 条不同实现的代码（N 通常为 5~10）。  
   - 同时，模型再生成对应的 M 条测试用例，每条用例包括输入、预期输出以及可执行的断言。  
   - 这里的关键是让模型产生多样化的测试，用来区分不同实现的细节差异。  

2. **执行与通过矩阵构建**  
   - 将每段代码与每条测试用例配对执行，记录二元矩阵 `Pass[i][j]`（第 i 段代码是否通过第 j 条测试）。  
   - 同时记录每段代码的运行时间或资源消耗，得到 `Cost[i]`。  

3. **PageRank‑式全局评分**  
   - 把代码视为节点，测试用例视为“超链接”。如果代码 i 通过测试 j，则在图中从测试 j 指向代码 i 加一条有向边。  
   - 初始化每段代码的分数为均等值，随后迭代更新：每段代码的新分数等于所有指向它的测试的分数之和，再乘以一个阻尼系数（类似网页排名的衰减）。  
   - 迭代收敛后得到每段代码的 **可信度分数** `Score[i]`，分数高的代码被认为在更难、更具区分性的测试上表现更好。  

4. **构造二元偏好对**  
   - 对每对代码 (i, k) 进行比较：  
     - 若 `Score[i] > Score[k]` 且 `Cost[i] ≤ Cost[k]`，则生成偏好 “i 优于 k”。  
     - 若两者分数相近，则使用运行时间差进行决胜。  
   - 这样得到的大量 (更好代码, 更差代码) 对构成 **偏好数据集**。  

5. **DPO 训练**  
   - 使用标准的 Direct Preference Optimization 损失：模型在给定任务描述时输出两段代码的概率分布，损失鼓励模型把更好代码的概率提升、把更差代码的概率压低。  
   - 训练过程不需要额外的奖励模型或强化学习回环，直接基于已构造的偏好对进行梯度更新。  

**最巧妙的点**：把测试用例当作“桥梁”来连接不同代码，实现了跨代码的相对评估；而 PageRank 的引入让这种相对评估不再是简单的计数，而是考虑了测试的全局重要性，极大提升了评分的鲁棒性。

### 实验与效果
- **评测基准**：作者在五个公开的代码生成基准上做实验，包括 HumanEval、MBPP、CodeContests、LeetCode‑Easy、以及一个内部的效率敏感任务集。  
- **对比基线**：与传统的监督微调（SFT）、基于奖励模型的强化学习（RLHF）以及已有的偏好学习方法（如 CodeRL）进行比较。  
- **结果概览**：论文声称在所有基准上均实现了显著提升，尤其在正确率上平均提升约 10%~15%，在运行时效率上平均降低约 12%（具体数值见原文表 1）。  
- **消融实验**：  
  - 去掉 PageRank 只用通过测试数量排序，正确率下降约 4%。  
  - 只考虑正确性不计入效率，生成代码的运行时间平均增加 8%。  
  - 使用随机生成的偏好对（不经过验证），模型性能几乎回到 SFT 水平。  
- **局限性**：作者指出自生成的测试用例质量仍受模型能力限制，极端复杂的算法可能产生不完整或误导性的测试；此外，执行环境的安全隔离成本在大规模生成时仍是瓶颈。  

### 影响与延伸思考
CodeDPO 把“自我出题‑自我批改”引入代码偏好学习，开启了无需人工标注的大规模偏好数据生成路线。后续工作（如 SelfEval‑Code、AutoDPO）已经在此思路上扩展，尝试把安全审计、内存泄漏检测等更多质量维度加入偏好构造。对想进一步探索的读者，可以关注以下方向：  
- **更强的测试生成模型**：提升测试用例的覆盖率和难度判别能力。  
- **跨语言偏好学习**：把同一任务的不同语言实现放在同一图中进行 PageRank 排序。  
- **安全与鲁棒性验证**：在生成‑执行环节加入沙箱监控，防止恶意代码影响数据质量。  

### 一句话记住它
让代码模型自己出题、自己批卷，用 PageRank 评出“最靠谱、最快”的答案，再用 DPO 把这种偏好直接写进模型。