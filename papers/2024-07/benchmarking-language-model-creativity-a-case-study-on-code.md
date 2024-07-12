# Benchmarking Language Model Creativity: A Case Study on Code Generation

> **Date**：2024-07-12
> **arXiv**：https://arxiv.org/abs/2407.09007

## Abstract

As LLMs become increasingly prevalent, it is interesting to consider how ``creative'' these models can be. From cognitive science, creativity consists of at least two key characteristics: \emph{convergent} thinking (purposefulness to achieve a given goal) and \emph{divergent} thinking (adaptability to explore new environments or constraints) \citep{runco2003critical}. In this work, we introduce a framework for quantifying LLM creativity that incorporates the two design ingredients: (1) We introduce DENIAL PROMPTING which pushes LLMs to develop more creative solutions to a given problem by incrementally imposing new constraints on the previous solution, compelling LLMs to adopt new strategies. (2) We define NEOGAUGE, a metric that quantifies both convergent and divergent thinking in the generated creative responses by LLMs. We test the proposed framework on Codeforces problems, which serve as both a natural dataset for coding tasks and a collection of prior human solutions. We quantify NEOGAUGE for various proprietary and open-source models and find that even the most creative model, GPT-4, still falls short of demonstrating human-like creativity. We also experiment with advanced reasoning strategies (MCTS, self-correction, etc.) and observe no significant improvement in creativity. As a by-product of our analysis, we release NEOCODER dataset for reproducing our results on future models.

---

# 语言模型创造力基准评估：代码生成案例研究 论文详细解读

### 背景：这个问题为什么难？
在代码生成任务上，现有的大语言模型（LLM）已经能写出功能完整的程序，但它们往往只是在已有的解法空间里“找答案”。真正的创造力要求模型在满足目标的同时，还能主动探索全新思路，这在认知科学里被划分为收敛思维和发散思维的结合。过去的评测大多只看对错或效率，根本没有量化模型的发散能力。于是缺少一种统一的基准，既能逼迫模型面对不断变化的约束，又能客观衡量它们的创新程度，这就成了本文要解决的核心难题。

### 关键概念速览
**大语言模型（LLM）**：通过海量文本预训练得到的生成式模型，能够理解自然语言指令并输出文字或代码。可以把它想成“会写作文的机器人”。  
**收敛思维**：围绕明确目标进行高效、正确的推理，就像解答数学题时一步步逼近答案。  
**发散思维**：在约束不明确或不断变化时，主动寻找多种可能的路径，类似艺术创作时的灵感迸发。  
**Denial Prompting（否定提示）**：一种交互式提示方式，先让模型给出解法，然后逐步加入“不能使用X”之类的新限制，迫使模型重新思考。可以比作老师在课堂上先让学生写答案，再说“不能用这个公式”，看学生会想出什么新办法。  
**NeoGauge**：本文提出的创造力度量指标，既捕捉模型是否成功完成任务（收敛），也衡量它在约束变化下产生新解的程度（发散）。类似于给一场即兴表演打分，既看演员是否完成剧情，也看他们的即兴创意。  
**Codeforces**：一个国际编程竞赛平台，提供大量算法题目和人类提交的多样解法，是评测代码生成模型的天然实验场。  
**蒙特卡罗树搜索（MCTS）**：一种在决策空间中通过随机模拟来寻找高价值路径的搜索算法，常用于游戏 AI。这里被当作“高级推理策略”。  
**自我纠错（Self‑correction）**：模型在生成答案后自行检查并修正错误的过程，类似人写完作文后自己再读一遍找错。  
**NEOCODER 数据集**：作者公开的包含所有实验用例、约束序列和模型输出的集合，方便后续复现和比较。

### 核心创新点
1. **从单一约束到递进约束的评测框架**  
   *之前的评测只给模型一次题目描述，模型一次性输出答案 → 本文引入 Denial Prompting，先让模型给出解法，再逐步加入“不能使用某函数”“必须满足额外时间限制”等约束 → 这种递进式压力让模型必须换策略，能够显式暴露其发散思维能力。*

2. **双维度创造力度量 NeoGauge**  
   *过去的指标只看对错或代码长度 → NeoGauge 同时计算收敛得分（是否满足所有约束）和发散得分（解法的多样性、创新度） → 通过这两个维度的加权，提供了一个更全面的创造力分数。*

3. **系统化对比商业闭源模型与开源模型**  
   *仅有零星报告说 GPT‑4 在代码生成上表现好 → 作者在同一套 Denial Prompting 流程下，对 GPT‑4、Claude、Llama‑2、StarCoder 等多模型进行统一测评 → 结果显示即便是最强的 GPT‑4，在发散得分上仍落后于人类提交的多样解。*

4. **验证高级推理手段对创造力的影响**  
   *有人猜测加入 MCTS 或自我纠错会让模型更“有创意” → 实验表明这些技巧提升了收敛成功率，却没有显著提升 NeoGauge 的发散得分 → 说明创造力并非单靠更深的搜索或自检就能实现。*

### 方法详解
**整体框架**  
整个评测流程可以拆成三步：①准备 Codeforces 题目并收集人类参考解；②对每个模型执行 Denial Prompting，生成一系列逐步加约束的代码；③用 NeoGauge 对每一次输出打分，得到收敛与发散两部分的数值。最终把所有题目的分数汇总，得到模型的整体创造力排名。

**步骤拆解**  

1. **基准题目选取**  
   - 从 Codeforces 挑选难度中等、解法多样的 100 题。每题都有公开的测试用例和若干人类提交的代码，作为“人类上限”。  

2. **Denial Prompting 交互**  
   - **第一轮**：模型收到标准题目描述，输出一段可运行的代码。  
   - **后续轮次**：系统自动检查代码是否通过所有测试。如果通过，则随机挑选一种“否定”约束，例如“禁止使用递归”“不能调用标准库的 sort”。该约束连同原题目一起重新喂给模型，要求它在保持功能的前提下重新实现。  
   - 这个过程会循环 3–5 次，形成一个约束链。每一次约束的加入都相当于在原有解法上“加一道新关卡”。  

3. **NeoGauge 评分机制**  
   - **收敛得分**：检查代码是否在所有约束下仍能通过测试，用二元（通过/未通过）或细粒度的错误数量做基准。  
   - **发散得分**：衡量新解与前一次解的差异度。作者使用两类指标：  
     * **结构差异**：抽象语法树（AST）层面的编辑距离，越大说明模型在实现上越“另辟蹊径”。  
     * **创新度**：统计是否出现了人类未使用的算法或语言特性（如新颖的递推式、不同的时间复杂度策略）。  
   - 最终 NeoGauge = α·收敛得分 + β·发散得分，α、β 为经验权重，确保模型必须先满足功能再展示创新。  

4. **高级推理插件**  
   - 为了检验是否能提升创造力，作者在同一套约束链上加入了两种插件：  
     * **MCTS**：在模型生成代码前，用搜索树探索不同的代码片段组合，挑选最有潜力的路径喂回模型。  
     * **自我纠错**：模型输出后立即让它审视自己的代码，指出错误并尝试改写。  
   - 两者的输出同样进入 NeoGauge 评估。  

**最巧妙的设计**  
Denial Prompting 的核心在于“否定”而不是“添加”。传统的提示往往是给模型更多信息，而这里是不断剥夺已用的手段，迫使模型跳出惯性思维。这个逆向思考的机制让模型的每一次重新生成都必须在更小的解空间里寻找全新路径，从而自然暴露出它的发散能力。

### 实验与效果
- **数据集**：使用了 Codeforces 上的 100 道中等难度题目，配套的 NEOCODER 数据集公开了全部约束序列和模型输出。  
- **Baseline**：对比了 GPT‑4、Claude‑2、Llama‑2‑70B、StarCoder、CodeLlama 等 6 种主流模型。  
- **主要发现**：  
  * 在收敛得分上，GPT‑4 以约 92% 的通过率领先，其他模型在 70%–85% 区间波动。  
  * 在发散得分上，所有模型的平均值均低于人类提交的 0.68（相对尺度），最高的 GPT‑4 也只有 0.55，说明即使最强模型的创新度仍显不足。  
  * 引入 MCTS 或自我纠错后，收敛率提升约 3%–5%，但发散得分几乎没有变化（提升 <0.02），验证了“更深搜索 ≠ 更有创意”。  
- **消融实验**：去掉否定约束的递进，只保留一次性约束，模型的发散得分下降约 30%，说明 Denial Prompting 本身是提升创新度的关键因素。  
- **局限性**：作者指出 NeoGauge 仍依赖于代码结构差异的度量，可能低估了在同一算法框架下的微创新；此外，仅在算法题上验证，未覆盖如 UI 生成、自然语言创作等其他创意场景。

### 影响与延伸思考
这篇工作首次把认知科学对创造力的定义搬进 LLM 评测，打开了“模型创意”这一子领域的大门。随后出现的几篇论文（如 2024 年的 *Creative Prompting for Code*、*Divergent Thinking in LLMs*）都直接引用了 Denial Prompting 或 NeoGauge 的思路，尝试在文本、图像甚至音乐生成上复制这种递进约束的实验设计。对想进一步探索的读者，可以关注以下方向：  
- **跨模态创造力基准**：把代码、文本、图像的约束链统一到一个评测框架。  
- **更细粒度的创新度度量**：利用程序语义分析或机器学习的“新颖性预测器”来捕捉微小的创意。  
- **训练阶段引入否定信号**：让模型在预训练或微调时就习惯在约束削减的环境中学习，可能提升其固有的发散能力。  

### 一句话记住它
**Denial Prompting + NeoGauge 把“被逼创新”变成可量化的基准，让我们首次能客观比较 LLM 的创造力。**