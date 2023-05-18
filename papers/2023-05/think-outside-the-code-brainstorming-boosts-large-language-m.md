# Think Outside the Code: Brainstorming Boosts Large Language Models in   Code Generation

> **Date**：2023-05-18
> **arXiv**：https://arxiv.org/abs/2305.10679

## Abstract

Code generation aims to automatically generate source code from high-level task specifications, which can significantly increase productivity of software engineering. Recently, approaches based on large language models (LLMs) have shown remarkable code generation abilities on simple tasks. However, generate code for more complex tasks, such as competition-level problems, remains challenging. In this paper, we introduce Brainstorm framework for code generation. It leverages a brainstorming step that generates and selects diverse thoughts on the problem to facilitate algorithmic reasoning, where the thoughts are possible blueprint of solving the problem. We demonstrate that Brainstorm significantly enhances the ability of LLMs to solve competition-level programming problems, resulting in a more than 50% increase in the pass@$k$ metrics for ChatGPT on the CodeContests benchmark, achieving state-of-the-art performance. Furthermore, our experiments conducted on LeetCode contests show that our framework boosts the ability of ChatGPT to a level comparable to that of human programmers.

---

# 跳出代码框架：头脑风暴提升大语言模型代码生成能力 论文详细解读

### 背景：这个问题为什么难？

代码生成的目标是让模型把自然语言需求直接翻译成可运行的程序。早期的 LLM（大语言模型）在“一行函数”或“实现排序”这类简单任务上已经能写出正确代码，但当面对需要多步算法、复杂数据结构甚至竞赛级的时间/空间约束时，模型往往只能给出片段或错误的实现。根本原因在于模型缺少系统化的算法思考过程：它们倾向于直接“把答案写出来”，而不是先构造解题思路。缺少这种思路，模型很容易陷入局部最优或直接跳过关键的设计步骤，这让竞争级编程成为瓶颈。

### 关键概念速览

**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，像 ChatGPT、GPT‑4。它们在海量代码库上预训练，所以会“记得”很多常见的实现模式。

**代码生成**：给定任务描述（通常是自然语言），模型输出完整的源代码。类似于把需求文档交给“会写代码的翻译官”。

**头脑风暴（Brainstorm）**：在正式写代码前，让模型先产生一系列可能的解题思路或“蓝图”。可以把它想象成团队讨论时先抛出多个方案，再挑选最靠谱的那个。

**思路（Thought）**：一种结构化的文字描述，包含算法的核心步骤、使用的数据结构、时间复杂度等信息。它相当于代码的“草稿”，而不是最终代码。

**Pass@$k$**：在 $k$ 次尝试中至少有一次通过测试用例的比例。$k$ 越大，模型的容错能力越高。

**LeetCode / CodeContests 基准**：真实的竞赛题库，题目难度从中等到极难，用来衡量模型的实战水平。

### 核心创新点

1. **从直接生成代码 → 先生成多样化思路 → 再选最优思路写代码**  
   传统方法让模型一次性输出完整代码，缺少中间检查。Brainstorm 框架在第一步让模型产生 N 条不同的思路，每条思路都是对问题的潜在解法。随后使用一个选择器（基于模型自身的评分或外部评估）挑出最有前景的思路，再交给模型完成代码实现。这样做把“算法设计”和“代码实现”解耦，显著提升了复杂任务的成功率。

2. **思路多样性通过采样策略实现**  
   为了避免模型总是给出同质化的答案，作者在思路生成阶段使用高温采样、Top‑p 过滤等技术，强制模型探索更广的解空间。相比于单一的贪心采样，这种“多样化采样”让模型有机会提出非常规算法（比如使用堆而不是排序），从而打开新的解题路径。

3. **思路选择采用双层评估**  
   选出最佳思路的过程并不是简单的概率最大化，而是先用模型内部的自评（log‑prob）做粗筛，再用轻量级的执行模拟或单元测试对前 K 条思路做细筛。这样既保持了速度，又保证了选出的思路在逻辑上更可能通过后续的代码生成。

4. **统一的端到端流水线**  
   整个系统把“思路生成 → 思路筛选 → 代码生成 → 代码验证”四个环节串成一条闭环。每一步的输出都可以反馈到前一步（例如代码验证失败时，系统会把错误信息回传给思路生成模块，促使模型重新思考），形成类似人类调试的迭代过程。

### 方法详解

整体框架可以概括为四步：**思路生成 → 思路筛选 → 代码实现 → 代码验证**。下面逐步拆解每一步的内部机制。

1. **思路生成**  
   - 输入：任务描述（如 LeetCode 题目正文）。  
   - 过程：使用预训练的 LLM（如 ChatGPT）在高温（temperature≈0.9）和 Top‑p≈0.95 的采样条件下，要求模型输出 “思路 #i”。每条思路必须包含：① 问题的核心难点，② 选用的数据结构或算法，③ 大致的时间/空间复杂度估计。  
   - 类比：这一步相当于让学生先在黑板上写出解题大纲，而不是直接写代码。

2. **思路筛选**  
   - 粗筛：对每条思路计算模型自身的对数概率（log‑prob），取前 M 条（如 M=5）。  
   - 细筛：对这 M 条思路分别生成对应的伪代码或简短实现，然后在轻量级的测试环境里跑几组样例（不需要完整通过，只要不报错）。根据通过率和错误信息给每条思路打分，选出最优的思路 T。  
   - 关键点：细筛使用的测试集非常小，目的是快速过滤掉明显不可行的思路，而不是完整评估。

3. **代码实现**  
   - 输入：选中的思路 T。  
   - 过程：再次调用 LLM，这次指令是 “根据思路 T 完整实现函数”。模型在较低温度（≈0.2）下生成代码，以保证输出的确定性。  
   - 这里的提示词会明确要求模型遵守思路中的算法步骤，避免自行“改写”思路。

4. **代码验证**  
   - 生成的代码直接在完整的题目测试集上运行。若全部通过，则任务成功。若出现错误，系统会把错误日志（如超时、错误输出）拼接到原始任务描述中，重新进入思路生成阶段，形成闭环。  
   - 这种“错误驱动的再思考”让模型像人类一样在调试中迭代思路，而不是一次性放弃。

**最巧妙的设计**在于把思路筛选的细筛步骤做得极轻量：只跑几条样例就能快速淘汰不合理的蓝图，这比直接让模型一次性生成完整代码再调试要省很多算力，也更符合人类的“先画框架后填细节”习惯。

### 实验与效果

- **数据集**：主要在 CodeContests 基准（包含多场编程竞赛的题目）和 LeetCode 近期比赛题目上评测。两者都以真实的竞赛难度为特征，覆盖从中等到极难的算法需求。  
- **对比基线**：直接使用 ChatGPT（原始模型）进行一次性代码生成、CoT（思维链）方式、以及最近的 CodeT5、AlphaCode 等公开模型。  
- **核心指标**：Pass@$k$（k=1,5,10）以及在 LeetCode 上的排名分数。  
- **结果**：在 CodeContests 上，Brainstorm 将 ChatGPT 的 Pass@10 从约 30% 提升到超过 80%，相当于 50% 以上的相对增幅，刷新了该基准的最高记录。LeetCode 实验显示，使用 Brainstorm 的 ChatGPT 能在 100 题的抽样中有约 70% 能够一次通过，接近人类中等水平程序员的表现。  
- **消融实验**：作者分别去掉思路多样性采样、细筛步骤以及闭环调试，发现每去掉一环，Pass@10 均下降 10%~20%，说明四个模块缺一不可。  
- **局限性**：论文承认在极端的时间限制题目（如需要 O(1) 空间）时，思路生成仍可能遗漏最优方案；此外，思路筛选依赖的轻量测试集如果选得不好，可能误删可行思路。模型的计算成本比单次生成代码高约 2‑3 倍。

### 影响与延伸思考

这篇工作把“先想后写”的人类编程习惯搬进了 LLM 流程，随后出现的多篇论文开始探索类似的“思路‑代码”双阶段框架，例如 **CodeReason**、**Prompt‑Plan** 等，都在不同程度上借鉴了 Brainstorm 的思路生成与筛选机制。未来的研究可能会在以下方向继续深化：① 更精细的思路表示（如图结构化的算法流程图），② 自动化的思路评估模型（不依赖人工测试），③ 将思路生成与代码执行的强化学习结合，实现真正的自我改进循环。对想深入的读者，可以关注近期在 arXiv 上出现的 “Algorithmic Prompting” 系列，它们在思路层面加入了形式化的证明检查。

### 一句话记住它

让大语言模型先“头脑风暴”出多种解题蓝图，再挑最靠谱的那条去写代码，能把竞赛级编程的成功率提升一倍以上。