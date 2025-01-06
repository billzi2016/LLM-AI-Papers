# BoostStep: Boosting mathematical capability of Large Language Models via   improved single-step reasoning

> **Date**：2025-01-06
> **arXiv**：https://arxiv.org/abs/2501.03226

## Abstract

Large language models (LLMs) have demonstrated impressive ability in solving complex mathematical problems with multi-step reasoning and can be further enhanced with well-designed in-context learning (ICL) examples. However, this potential is often constrained by two major challenges in ICL: granularity mismatch and irrelevant information. We observe that while LLMs excel at decomposing mathematical problems, they often struggle with reasoning errors in fine-grained steps. Moreover, ICL examples retrieved at the question level may omit critical steps or even mislead the model with irrelevant details. To address this issue, we propose BoostStep, a method that enhances reasoning accuracy through step-aligned ICL, a novel mechanism that carefully aligns retrieved reference steps with the corresponding reasoning steps. Additionally, BoostStep incorporates an effective "first-try" strategy to deliver exemplars highly relevant to the current state of reasoning. BoostStep is a flexible and powerful method that integrates seamlessly with chain-of-thought (CoT) and tree search algorithms, refining both candidate selection and decision-making. Empirical results show that BoostStep improves GPT-4o's CoT performance by 4.6% across mathematical benchmarks, significantly surpassing traditional few-shot learning's 1.2%. Moreover, it can achieve an additional 7.5\% gain combined with tree search. Surprisingly, it enhances state-of-the-art LLMs to solve challenging math problems using simpler examples. It improves DeepSeek-R1-671B's performance on AIME by 2.2%, leveraging simple examples only from the MATH dataset.

---

# BoostStep：通过改进单步推理提升大语言模型的数学能力 论文详细解读

### 背景：这个问题为什么难？
在数学题目上，大语言模型（LLM）需要把一个复杂的问题拆成多个细小的推理步骤。过去的研究已经证明，给模型提供“思维链”（CoT）式的示例能显著提升解题成功率。但实际使用中，两大障碍仍然突出：一是示例的粒度往往与模型当前的推理粒度不匹配，导致模型在细粒度步骤上出错；二是检索到的示例往往包含与当前步骤无关的信息，甚至遗漏关键步骤，反而把模型带偏。正是这两个根本性缺陷，让单步推理的准确率成为提升整体数学能力的瓶颈。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，像 GPT‑4o、DeepSeek‑R1 等。  
**思维链（CoT）**：让模型在给出最终答案前先写出逐步推理过程，类似人解题时的草稿。  
**在上下文学习（ICL）**：在模型输入中加入若干示例，让模型“模仿”示例的解题方式。  
**步骤对齐（step‑aligned）**：把检索到的参考步骤精准匹配到模型当前正在推理的那一步，确保信息的粒度和语义一致。  
**first‑try 策略**：在模型每一次推理尝试时，只提供与当前状态最相关的示例，避免一次性塞入大量无关信息。  
**树搜索（tree search）**：在多个可能的推理路径之间进行探索和剪枝，类似围棋的蒙特卡罗树搜索，用来挑选最有前景的解题路线。  

### 核心创新点
1. **粒度匹配的 ICL → 步骤对齐 ICL**：传统的 few‑shot 示例往往是整题或整段解答，模型在细粒度推理时只能被动接受。BoostStep 通过检索并对齐每一步的参考步骤，使示例的细度恰好匹配模型当前的推理粒度，显著降低细节错误。  
2. **一次性全示例 → first‑try 递进提供**：过去的做法一次性把所有示例塞进上下文，容易淹没关键信息。BoostStep 在模型每一步“尝试”时，只喂入与该步最相关的示例，形成一种“先尝试、后补充”的动态交互，提升了信息利用率。  
3. **独立的 CoT 改进 → 与树搜索无缝融合**：BoostStep 既可以直接提升普通 CoT 的准确率，又能在树搜索框架下进一步优化候选路径的选择，两个层面都实现了性能提升。  
4. **复杂示例 → 简单示例的逆向利用**：实验显示，使用仅来自 MATH 数据集的简短示例，也能让 DeepSeek‑R1‑671B 在 AIME 这类高难度题目上提升 2.2%，说明对齐机制比示例的复杂度更关键。

### 方法详解
BoostStep 的整体流程可以概括为三步：**检索 → 对齐 → 动态喂入**。下面逐层拆解。

1. **检索阶段**  
   - 给定待解数学题，系统先在大规模题库（如 MATH、AIME）中检索出与之相似的题目。检索返回的是完整的解题过程，包含若干细粒度的推理步骤。  
   - 与传统的“按题目整体检索”不同，这里每一步的文本都会被单独标记，以便后续匹配。

2. **步骤对齐（step‑aligned ICL）**  
   - 当模型开始进行单步推理时，BoostStep 会把当前模型的“思考状态”映射成一个简短的描述（例如“计算 x² 的展开式”）。  
   - 系统在检索到的参考步骤中寻找最相似的描述，并把对应的完整步骤作为示例插入上下文。这里的相似度计算使用轻量的向量检索或关键词匹配，确保对齐的速度足够快。  
   - 关键在于**一对一**的对齐：每一次模型的推理只对应一条参考步骤，避免信息冗余。

3. **first‑try 动态喂入**  
   - 模型每完成一次“尝试”（即生成一个候选推理步骤），BoostStep 立即检查该步骤是否满足预期的形式。如果不满意，系统会在下一轮尝试时提供另一个更贴合的参考步骤。  
   - 这种循环类似于人解题时“先写草稿、发现错误、再查教材对应章节”的过程，模型在每一次迭代中都能获得最精准的指导。

4. **与树搜索的结合**  
   - 在需要探索多条可能路径的场景下，BoostStep 把对齐的示例作为每个节点的评估依据。树搜索会根据每条路径的累计对齐得分进行剪枝，保留最有潜力的分支。  
   - 这样，BoostStep 不仅提升了单步的正确率，还帮助搜索过程更快收敛到正确答案。

**最巧妙的点**在于把检索、对齐和动态喂入闭环化：模型的每一步输出直接决定下一步检索的内容，实现了“模型驱动的检索”。这种自适应机制是以往一次性提供示例的 ICL 所没有的。

### 实验与效果
- **测试数据**：主要在公开的数学基准上评估，包括 MATH、AIME、以及更具挑战性的 GSM8K。  
- **基线对比**：  
  - 传统 few‑shot（一次性提供 4–8 例）在 GPT‑4o CoT 上提升约 1.2%。  
  - BoostStep 在相同模型上提升 4.6%，即比 few‑shot 多出约 3.4 个百分点。  
  - 与树搜索结合后，整体提升进一步达到 7.5%。  
  - 对 DeepSeek‑R1‑671B，使用仅来自 MATH 的简短示例，BoostStep 在 AIME 上提升 2.2%。  
- **消融实验**：论文分别去掉步骤对齐、first‑try、以及树搜索模块，发现步骤对齐贡献最大（约 2.8%），first‑try 次之（约 1.2%），树搜索在已有对齐的基础上再加约 1.5%。  
- **局限性**：作者指出，BoostStep 依赖高质量的检索库；如果相似题目稀缺，步骤对齐的效果会下降。此外，实时对齐带来的计算开销在超大模型上仍需进一步优化。

### 影响与延伸思考
BoostStep 把“检索‑对齐‑推理”闭环化的思路打开了新的方向，后续有几篇工作尝试将其推广到代码生成、逻辑推理等非数学任务（如“CodeStep”系列）。另外，结合更高效的向量检索（如 MIPS）和自适应提示生成的研究正在兴起，目标是把对齐过程的延迟压到毫秒级。想深入了解的读者可以关注以下两个方向：① 大规模检索库的构建与维护；② 动态提示（dynamic prompting）在多轮对话中的应用。

### 一句话记住它
BoostStep 通过让模型每一步都只看到最匹配的参考步骤，实现了“一步一步精准喂养”，显著提升了大语言模型的数学解题能力。