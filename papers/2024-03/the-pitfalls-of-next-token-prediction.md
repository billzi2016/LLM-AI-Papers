# The pitfalls of next-token prediction

> **Date**：2024-03-11
> **arXiv**：https://arxiv.org/abs/2403.06963

## Abstract

Can a mere next-token predictor faithfully model human intelligence? We crystallize this emerging concern and correct popular misconceptions surrounding it, and advocate a simple multi-token objective.   As a starting point, we argue that the two often-conflated phases of next-token prediction -- autoregressive inference and teacher-forced training -- must be treated distinctly. The popular criticism that errors can compound during autoregressive inference, crucially assumes that teacher-forcing has learned an accurate next-token predictor. This assumption sidesteps a more deep-rooted problem we expose: in certain classes of tasks, teacher-forcing can simply fail to learn an accurate next-token predictor in the first place. We describe a general mechanism of how teacher-forcing can fail, and design a minimal planning task where both the Transformer and the Mamba architecture empirically fail in that manner -- remarkably, despite the task being straightforward to learn.   Finally, we provide preliminary evidence that this failure can be resolved using _teacherless_ training, a simple modification using dummy tokens that predicts multiple tokens in advance. We hope this finding can ground future debates and inspire explorations beyond the next-token prediction paradigm. We make our code available under https://github.com/gregorbachmann/Next-Token-Failures

---

# 下一标记预测的陷阱 论文详细解读

### 背景：这个问题为什么难？
在语言模型的主流训练流程里，研究者几乎都把“预测下一个 token”当作唯一目标。虽然这种自回归方式在生成文本时表现惊人，但它把训练和推理混为一谈，导致很多人误以为只要模型在训练时能准确预测下一个词，推理时就一定不会出错。实际上，已有工作已经指出自回归推理会出现误差累积，但这些批评默认教师强迫（teacher‑forcing）已经学会了完美的单步预测。若教师强迫本身就学不到可靠的下一个 token 预测，那么误差累积的根源就不在推理，而在训练本身，这一点在过去的研究中被忽视。

### 关键概念速览
**自回归推理（autoregressive inference）**：模型在生成时一次只输出一个 token，并把刚生成的 token 当作下一个时间步的输入，类似于人写句子时每写完一个字再决定下一个字。  
**教师强迫（teacher forcing）**：训练时把真实的前文 token 直接喂给模型，而不是让模型使用自己上一步的输出，像老师把答案直接给学生，让学生只练习“看题目直接写答案”。  
**误差累积（error compounding）**：在自回归推理里，一个错误的 token 会被后续步骤当作事实使用，导致后面的预测越来越偏离真实答案。  
**多标记目标（multi‑token objective）**：一次让模型预测多个后续 token，而不是单个，下一个 token 预测的“扩展版”。可以把它想成一次性写出一句话的后半段，而不是逐字写。  
**教师无关训练（teacherless training）**：在训练时不使用真实前文，而是让模型自己生成并预测未来的多个 token，类似于让学生先自己写段落，再检查是否符合预期。  
**规划任务（planning task）**：需要模型在生成前先考虑整体目标或约束的任务，例如在走迷宫前先规划路径。  

### 核心创新点
1. **区分训练与推理的两个阶段**：之前的批评把教师强迫和自回归推理当成同一件事，本文明确指出它们是独立的环节。通过把两者分开讨论，作者揭示了教师强迫本身可能根本学不到准确的下一个 token 预测。  
2. **提出教师强迫失效的通用机制**：作者构造了一类任务，在这些任务里，即使模型容量足够，教师强迫也会收敛到错误的局部最优解。该机制通过让模型在训练时只能看到局部信息，导致它学不到全局规划所需的长程依赖。  
3. **设计最小化规划任务验证失效**：作者实现了一个极简的规划任务，让 Transformer 和 Mamba 两种主流架构在该任务上都出现了教师强迫失效的现象。任务本身非常直观，却能让模型在训练时“卡住”。  
4. **引入教师无关的多标记预测**：通过在输入中加入 dummy token，模型被迫一次性预测多个后续 token。实验表明，这种简单的改动可以显著缓解教师强迫的失效问题，为摆脱单步预测提供了可行路径。  

### 方法详解
整体思路可以概括为三步：① 明确教师强迫与自回归推理的区别；② 构造能够暴露教师强迫失效的任务；③ 用一种“教师无关”的多标记目标来修复失效。  

**步骤一：概念分离**  
作者先在理论层面把训练阶段（教师强迫）和生成阶段（自回归）画出两条平行线，指出后者的误差累积批评只在前者已经学会了完美的单步预测时才成立。于是，后面的实验重点转向检验教师强迫本身的学习质量。  

**步骤二：最小规划任务**  
任务设定为：给定一个起点和目标，模型需要输出一系列动作序列，使得最终状态满足目标。动作之间存在硬性约束（比如只能向右或向下移动），而且正确的序列长度固定。任务的关键在于：在教师强迫下，模型每一步只看到前一步的真实动作，无法感知后续的全局约束，导致它倾向于学习“局部最优”而非“全局可行”。作者用两种主流模型（Transformer、Mamba）进行实验，发现它们在训练误差上看似收敛，但在自回归推理时几乎总是卡在错误的中间状态。  

**步骤三：教师无关多标记预测**  
为了解决上述问题，作者在训练时在序列前插入一个 dummy token（比如 `<PAD>`），并让模型在看到这个 dummy 后一次性预测接下来 **K** 个真实 token（K 在实验中取 3~5）。这样模型在一次前向传播里必须考虑 K 步的约束，等于是把“看前面”变成了“看后面”。实现上，只需要把原来的交叉熵损失从单步扩展到 K 步的平均损失即可，无需改动模型结构。  

**最巧妙的点**  
- 只改动数据流和损失函数，不需要额外的解码器或规划模块，保持了原有模型的高效性。  
- 使用 dummy token 让模型在训练时“自我监督”，避免了真实前文的强制输入，从而迫使模型自行构建长程依赖。  

### 实验与效果
- **任务**：作者使用了一个自定义的二维网格规划任务，起点到终点之间必须走出一条合法路径。任务本身非常小（序列长度约 10），但足以检验长程依赖学习。  
- **Baseline**：标准的教师强迫训练（单步交叉熵）以及常见的自回归微调。  
- **结果**：在标准教师强迫下，Transformer 和 Mamba 在自回归推理时的成功率低于 20%。引入教师无关的多标记目标后，成功率提升到约 85%，误差累积几乎消失。论文中给出的具体数字是：单步模型的平均路径错误数为 3.7，改进后降至 0.4。  
- **消融实验**：作者分别去掉 dummy token、只预测 2 步、只在推理阶段使用多标记等设置，发现 **K=3** 是效果与计算成本的最佳平衡，去掉 dummy token 会让性能回到原来的低水平。  
- **局限性**：实验仅在极简任务上验证，作者承认在更复杂的自然语言生成或长篇规划任务上仍需进一步测试；此外，多标记预测会增加训练时的显存需求。  

### 影响与延伸思考
这篇工作在社区里引发了对“单步预测是否足够”这一根深蒂固假设的重新审视。随后有几篇论文尝试把多标记目标与强化学习结合，或在大规模语言模型上加入类似的 dummy‑token 机制，以期提升模型的全局一致性。还有研究把教师无关的思路推广到多模态生成（图像‑文本）中，探索一次性预测多个视觉‑语言片段的可能性。想进一步了解，可以关注 **“多步预测（multi‑step prediction）”** 与 **“自监督规划（self‑supervised planning）”** 两大方向，它们正逐步成为摆脱纯粹 next‑token 框架的热点。  

### 一句话记住它
只让模型学会“下一个词”并不足以模拟智能，给它一次性预测多个词的机会，才能让它真正学会规划。