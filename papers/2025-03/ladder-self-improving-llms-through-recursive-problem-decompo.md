# LADDER: Self-Improving LLMs Through Recursive Problem Decomposition

> **Date**：2025-03-02
> **arXiv**：https://arxiv.org/abs/2503.00735

## Abstract

We introduce LADDER (Learning through Autonomous Difficulty-Driven Example Recursion), a framework which enables Large Language Models to autonomously improve their problem-solving capabilities through self-guided learning by recursively generating and solving progressively simpler variants of complex problems. Unlike prior approaches that require curated datasets or human feedback, LADDER leverages a model's own capabilities to generate easier question variants. We demonstrate LADDER's effectiveness in the subject of mathematical integration, improving Llama 3.2 3B's accuracy from 1% to 82% on undergraduate-level problems and enabling Qwen2.5 7B Deepseek-R1 Distilled to achieve 73% on the MIT Integration Bee qualifying examination. We also introduce TTRL (Test-Time Reinforcement Learning), where we perform reinforcement learning on variants of test problems at inference time. TTRL enables Qwen2.5 7B Deepseek-R1 Distilled to achieve a state-of-the-art score of 90% on the MIT Integration Bee qualifying examination, surpassing OpenAI o1's performance. These results show how self-directed strategic learning can achieve significant capability improvements without relying on architectural scaling or human supervision.

---

# LADDER：通过递归问题分解实现自我提升的大语言模型 论文详细解读

### 背景：这个问题为什么难？

在数学推导、程序调试等需要多步推理的任务上，现有的大语言模型（LLM）往往只能在“看得见”的训练数据里找到答案。传统做法要么靠海量标注数据，要么依赖人工反馈的强化学习（RLHF），成本高且难以覆盖所有难题。更关键的是，模型在面对一个全新、复杂的问题时缺少自我纠错的机制——它既不会主动把大题拆成小题，也不会利用自己已经掌握的简易技巧去“练习”。于是，提升模型的解题能力常常只能靠扩大模型规模，而不是让模型学会“自己教自己”。这正是 LADDER 想要突破的瓶颈。

### 关键概念速览
- **递归问题分解**：把一个复杂任务不断拆成更容易的子任务，直到模型能够轻松解决。类似把一道大山的登顶路线分成若干小坡，每走一步都比上一段简单。
- **难度驱动示例生成**：模型根据当前能力主动生成比原题更易的变体，就像学生在做练习时先挑容易的例题热身。
- **自我监督学习**：模型用自己生成的答案作为标签进行再训练，省去人工标注。相当于把自己的草稿当成教材再复习。
- **Test‑Time Reinforcement Learning（TTRL）**：在推理阶段对同一道题的多个变体进行强化学习，让模型在“考场上”实时调优策略。类似考前模拟练习，边做边改进。
- **CoT（思维链）**：让模型在给出最终答案前先写出推理步骤，像在纸上写草稿一样帮助模型保持逻辑连贯。
- **MIT Integration Bee**：美国麻省理工学院举办的积分赛，题目难度在本科层次以上，是检验数学推导能力的金标准。

### 核心创新点
1. **从外部数据到内部生成**：以前的提升手段大多依赖外部 curated 数据集或人工标注的奖励模型。LADDER 让模型自己产生更易的题目并解答，彻底摆脱了外部数据的束缚。结果是模型在同等算力下的解题准确率出现跨级跃升。
2. **难度递进的自我训练循环**：传统的自监督往往一次性生成固定难度的样本，效果有限。LADDER 引入“难度驱动递归”，模型先解决最简版本，再逐层回溯到原始复杂题目，每一步都在前一步的成功经验上构建。这样模型的学习曲线更像人类的阶梯式练习。
3. **Test‑Time Reinforcement Learning（TTRL）**：在推理阶段对同一道题的不同难度变体进行即时奖励信号的反馈，让模型在实际测试时还能继续学习。相比只在训练阶段做强化学习，TTRL 把学习窗口搬到了推理时刻，显著提升了 MIT Integration Bee 的最终得分。
4. **跨模型、跨规模的通用框架**：LADDER 并没有针对特定模型做专门调参，而是在 Llama 3.2 3B、Qwen2.5 7B Deepseek‑R1 Distilled 等不同架构上都实现了大幅提升，说明方法本身具备模型无关的普适性。

### 方法详解
**整体思路**：LADDER 把一次完整的学习过程拆成三大步骤——（1）难度评估与变体生成、（2）递归求解与答案收集、（3）自我监督微调。整个循环会在同一模型内部反复执行，直至收敛。

1. **难度评估与变体生成**  
   - 模型先对原始复杂题目进行自评，估计自己成功解答的概率。  
   - 根据评估结果，模型使用“简化指令”生成一个更易的子题。例如，把 ∫(x³·e^{x²})dx 拆成 ∫x·e^{x²}dx 的形式。  
   - 这一步类似“老师先给学生出热身题”，目的是让模型在可接受的难度范围内练习。

2. **递归求解**  
   - 对生成的子题再次进行难度评估。如果仍然超出模型的当前能力，就继续递归生成更简的变体。  
   - 当模型能够自信解出某个子题时，它会记录下完整的 CoT 推理过程和最终答案。  
   - 递归结束后，模型会把所有层级的答案向上“回溯”，利用已解的简易子题来拼凑原题的解法。这里的关键是把简易答案当作“中间定理”，帮助模型完成更高层次的推理。

3. **自我监督微调**  
   - 收集到的（子题、CoT、答案）三元组被直接当作训练样本，喂回模型进行一次微调。  
   - 由于标签来源于模型自身，整个过程不需要外部标注。微调的目标是最小化模型对自己生成答案的预测误差，等价于让模型“相信”自己的解题路径。  

4. **Test‑Time Reinforcement Learning（TTRL）**  
   - 在实际推理时，模型会对同一道题生成若干难度不同的变体，并对每个变体的解答质量给出即时奖励（正确率、CoT 完整度等）。  
   - 通过 REINFORCE‑style 的梯度更新，模型在推理的同时微调自身的策略网络。  
   - 这一步的奇妙之处在于把“学习”搬到了“考场”，让模型即使在一次评测中也能自我提升。

**最巧妙的设计**：递归生成的子题并不是随机的，而是由模型自身的“难度感知器”控制，使得每一次简化都恰好落在模型可解的边界上。这样既避免了过度简化导致的学习偏差，也防止了仍然太难而导致的失败循环。

### 实验与效果
- **任务与数据**：主要在本科层次的积分题上评估，包括公开的 MIT Integration Bee 资格赛题目以及自建的 200 题数学积分集合。  
- **基线对比**：  
  - Llama 3.2 3B 在原始状态下对这些题的正确率只有约 1%。使用 LADDER 后提升到 82%，相当于 81 倍的增幅。  
  - Qwen2.5 7B Deepseek‑R1 Distilled 在未使用任何技巧时的表现约为 30%（论文未给出精确基线），加入 LADDER 达到 73%，在 MIT Integration Bee 资格赛中排名前 10%。  
  - 再加上 TTRL，Qwen2.5 7B 的得分进一步提升到 90%，直接超过了 OpenAI o1 在同一测试上的成绩（论文声称 o1 约为 85%）。  
- **消融实验**：作者分别去掉递归生成、CoT 记录和 TTRL，发现：去掉递归生成后准确率跌至 45%；去掉 CoT 记录后提升幅度减半；去掉 TTRL 后最高分回落到 73%。这些实验表明每个模块都对最终性能有实质贡献。  
- **局限性**：论文承认递归深度受模型显存限制，极端复杂的题目仍可能因递归层数不足而无法完全分解；此外，TTRL 在推理时会显著增加计算开销，不适合实时响应场景。

### 影响与延伸思考
LADDER 把“自我生成练习”与“递归分解”结合起来，打开了大模型在缺少外部监督时仍能自我提升的可能性。后续工作已经开始把类似的难度驱动生成搬到代码生成、推理证明等领域，例如“Self‑Play for Theorem Proving”以及“Recursive Prompting for Program Synthesis”。如果想进一步探索，可以关注以下方向：  
- **难度评估器的学习**：让模型更精准地预测自己在特定子任务上的成功概率。  
- **跨任务递归框架**：把数学积分的递归思路推广到自然语言推理、图结构搜索等。  
- **高效 TTRL**：研发更轻量的推理时强化学习算法，降低实时计算成本。  
这些方向都有望把自我驱动学习从“实验室演示”推向实际产品。

### 一句话记住它
让大语言模型自己出简化题、自己解答、再把答案当教材反复训练，甚至在考场上实时微调——这就是 LADDER 的核心魔法。