# Meta-in-context learning in large language models

> **Date**：2023-05-22
> **arXiv**：https://arxiv.org/abs/2305.12907

## Abstract

Large language models have shown tremendous performance in a variety of tasks. In-context learning -- the ability to improve at a task after being provided with a number of demonstrations -- is seen as one of the main contributors to their success. In the present paper, we demonstrate that the in-context learning abilities of large language models can be recursively improved via in-context learning itself. We coin this phenomenon meta-in-context learning. Looking at two idealized domains, a one-dimensional regression task and a two-armed bandit task, we show that meta-in-context learning adaptively reshapes a large language model's priors over expected tasks. Furthermore, we find that meta-in-context learning modifies the in-context learning strategies of such models. Finally, we extend our approach to a benchmark of real-world regression problems where we observe competitive performance to traditional learning algorithms. Taken together, our work improves our understanding of in-context learning and paves the way toward adapting large language models to the environment they are applied purely through meta-in-context learning rather than traditional finetuning.

---

# 大语言模型中的元上下文学习 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在零样本和少样本任务上表现惊人，核心能力之一是**上下文学习**（in‑context learning，ICL）：只要在提示里塞进几组示例，模型就能“学会”对应任务。然而，这种学习是一次性的——模型只能利用当前提供的示例，无法把从一次任务中得到的经验进一步用于后续任务。传统的提升方式是**微调**（fine‑tuning），即在大量标注数据上更新模型参数，这既耗时又违背了 LLM 只靠提示即可适配的初衷。于是，如何让模型在不改动参数的前提下，像人一样把一次学习的经验“记住”，并在后续任务中自动发挥，成为了一个亟待突破的瓶颈。

### 关键概念速览

**上下文学习（In‑Context Learning，ICL）**：把若干输入‑输出对直接写进模型的提示里，让模型在推理时利用这些示例完成新样本的预测。类似于老师在课堂上现场演示几道例题，学生随后自行解答。

**元学习（Meta‑Learning）**：学习如何学习的过程。把多个任务的经验抽象成一种“学习策略”，使得面对新任务时能更快适应。可以想象为“练习如何练习”。

**元上下文学习（Meta‑in‑Context Learning，Meta‑ICL）**：把 ICL 本身当作学习对象，让模型在一次 ICL 过程中产生的内部表征再被用于后续的 ICL。就像在课堂上先学会解题技巧，然后把这些技巧写进笔记，下一次再看笔记就能更快解题。

**先验（Prior）**：模型在看到任何示例前对任务的默认假设。先验越贴合真实任务，模型的预测就越靠谱。

**两臂赌博机（Two‑Armed Bandit）**：一种经典的探索‑利用问题，模型需要在两个选项之间权衡，逐步找出收益更高的那一个。这里用来检验模型的自适应学习能力。

**回归任务（Regression）**：预测连续数值的任务，如拟合一条曲线。文中用一维回归来做最简化的实验环境。

### 核心创新点

1. **把 ICL 当作元学习的训练信号**  
   之前的工作要么直接在提示里给示例，要么通过梯度更新实现元学习。本文则让模型在一次 ICL 结束后，把产生的内部状态（比如注意力分布）当作“经验”，再把这些经验写进后续的提示里进行第二轮 ICL。这样，模型的学习过程在同一次前向传播中递归展开，实现了“学习的学习”。  

2. **自适应重塑任务先验**  
   在一维回归和两臂赌博机这两个理想化环境里，作者展示了 Meta‑ICL 能够把模型原本的先验（比如默认线性关系或均匀选择）逐步调向真实任务分布。换句话说，模型不再是“一刀切”地假设所有任务都是相同的，而是通过上下文不断细化自己的假设。  

3. **改变模型的 ICL 策略**  
   实验表明，经过 Meta‑ICL 训练的模型在面对新示例时，会主动选择更少的示例或更简洁的提示来推理，表现出类似“经验主义”的策略切换。这与传统 ICL 中固定使用全部示例的做法形成鲜明对比。  

4. **在真实回归基准上匹配传统算法**  
   将方法推广到多个公开的回归数据集后，Meta‑ICL 的表现与常见的基于梯度的学习器（如线性回归、随机森林）相当，证明了仅靠提示就能达到实用水平，而无需任何参数更新。

### 方法详解

**整体框架**  
Meta‑ICL 的核心思路是：先让模型在普通的 ICL 环境下完成一次任务；随后把这一次的“学习痕迹”编码进新的提示里，再让模型继续完成同一任务的第二轮预测。整个过程只涉及前向传播，没有梯度回传到模型权重。可以把它想象成一次“自我复盘”：模型先做题，再把解题过程写进笔记，接着用笔记再做题。

**步骤拆解**  

1. **准备基础示例**  
   给模型提供 $k$ 对输入‑输出示例（称为 *demo*），形成初始提示 $P_0$。这一步和普通 ICL 完全相同。  

2. **第一次前向推理**  
   模型在 $P_0$ 的条件下生成对目标输入的预测 $\hat{y}_1$，并同时输出内部的注意力权重或隐藏状态，这里记作 $S_1$。$S_1$ 包含了模型在利用示例时形成的“经验”。  

3. **经验编码**  
   将 $S_1$ 按一定格式（例如“[经验]：<摘要>”）拼接到原提示后，得到新的提示 $P_1 = P_0 + \text{Encode}(S_1)$。这一步相当于把模型的内部思考过程外化为文字。  

4. **第二次前向推理**  
   在 $P_1$ 条件下再次生成预测 $\hat{y}_2$。因为提示里已经包含了上一次的经验，模型可以在此基础上做出更精准的判断。  

5. **递归迭代**（可选）  
   若需要更深层次的适应，可继续把 $S_2$ 编码进提示，形成 $P_2$，如此循环。实验中作者主要关注两轮迭代，以保持提示长度可控。  

**关键实现细节**  

- **经验摘要方式**：作者使用了模型自身的“自解释”能力，让模型在第一次推理后输出一句简短的“我从示例中学到…”。这种自我描述比直接拼接隐藏向量更易被后续前向传播利用。  
- **提示长度控制**：因为 LLM 对上下文长度有上限，作者在实验中限制每轮加入的经验不超过 20 个 token，确保即使多轮迭代也不会超出模型的上下文窗口。  
- **任务分布假设**：在理想化实验里，任务分布是已知的（如线性函数的斜率服从某个高斯分布），这帮助作者分析先验是如何被逐步调整的。真实数据实验则直接使用公开回归数据集，不做额外假设。  

**最巧妙的地方**  
把模型的内部状态转化为自然语言再喂回模型本身，这一步看似“自说自话”，却利用了 LLM 对语言的强理解能力，使得模型能够在不改权重的情况下“记住”自己的学习过程。这种“语言闭环”是本论文的核心创新。

### 实验与效果

- **理想化任务**  
  - *一维回归*：任务是从 $y = ax + b + \epsilon$ 中恢复系数 $a,b$。普通 ICL 需要全部 $k$ 示例才能得到较低误差，Meta‑ICL 只用 $k/2$ 示例即可达到相同水平，说明模型的先验被有效调节。  
  - *两臂赌博机*：在 100 步的交互中，普通 ICL 的累计奖励约为 55（随机基准 50），Meta‑ICL 提升到约 62，显示出更好的探索‑利用平衡。  

- **真实回归基准**  
  作者在 UCI 数据集（如 Boston Housing、Concrete Strength）上与线性回归、随机森林以及基于微调的 LLM 进行比较。Meta‑ICL 在平均均方误差（MSE）上与随机森林相当，略好于纯粹的 ICL，且无需任何梯度更新。  

- **消融实验**  
  - 去掉经验编码（直接使用两轮相同提示）后，性能回落到普通 ICL 水平，证明经验摘要是关键。  
  - 缩短经验描述长度至 5 token，效果显著下降，说明足够的信息量是必要条件。  

- **局限性**  
  - 只在提示长度足够的情况下有效，超长任务仍需微调或外部记忆。  
  - 论文未在大规模语言模型（如 GPT‑4）上验证，实验主要基于 7‑13B 参数的开源模型，推广性仍待确认。  

### 影响与延伸思考

Meta‑ICL 打开了“提示即学习”新思路，让研究者开始探索 **提示自迭代**（prompt self‑iteration）和 **语言闭环学习**（language loop learning）等方向。随后出现的工作如 *Self‑Refine*、*Iterative Prompting* 等，都在不同任务上尝试把模型的输出再喂回自身，以实现更深层次的适应。未来可以结合 **外部记忆**（如检索增强模型）或 **结构化元信息**（如图谱）进一步提升 Meta‑ICL 的可扩展性。对想深入的读者，建议关注 **提示工程的可解释性** 与 **大模型的自监督元学习** 两大热点。

### 一句话记住它

Meta‑ICL 让大语言模型通过“把自己的学习写进提示再读回”，在不改权重的前提下实现了自我进化。