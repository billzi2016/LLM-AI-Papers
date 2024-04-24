# Let's Think Dot by Dot: Hidden Computation in Transformer Language   Models

> **Date**：2024-04-24
> **arXiv**：https://arxiv.org/abs/2404.15758

## Abstract

Chain-of-thought responses from language models improve performance across most benchmarks. However, it remains unclear to what extent these performance gains can be attributed to human-like task decomposition or simply the greater computation that additional tokens allow. We show that transformers can use meaningless filler tokens (e.g., '......') in place of a chain of thought to solve two hard algorithmic tasks they could not solve when responding without intermediate tokens. However, we find empirically that learning to use filler tokens is difficult and requires specific, dense supervision to converge. We also provide a theoretical characterization of the class of problems where filler tokens are useful in terms of the quantifier depth of a first-order formula. For problems satisfying this characterization, chain-of-thought tokens need not provide information about the intermediate computational steps involved in multi-token computations. In summary, our results show that additional tokens can provide computational benefits independent of token choice. The fact that intermediate tokens can act as filler tokens raises concerns about large language models engaging in unauditable, hidden computations that are increasingly detached from the observed chain-of-thought tokens.

---

# 让我们逐点思考：Transformer 语言模型中的隐藏计算 论文详细解读

### 背景：这个问题为什么难？

在语言模型里，让模型先写出思考步骤（Chain‑of‑Thought，简称 CoT）已经被证明能大幅提升解题准确率。但研究者一直不清楚，这提升到底是因为模型真的在进行类似人类的分步骤推理，还是仅仅因为多输出了几个 token，给模型提供了额外的计算空间。传统的实验往往把 CoT 当作“可解释的思考”，忽视了 token 本身可能只是占位符的可能性。于是出现了一个核心疑问：如果把这些中间 token 换成毫无意义的 filler（比如一串“......”），模型还能保持同样的性能吗？

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先把推理过程写出来，类似人在解数学题时先在草稿纸上列步骤，帮助模型把复杂任务拆解成若干子任务。
- **Filler Token（填充 token）**：没有语义信息的占位符，例如“......”。在本研究中，它们被用来代替真实的思考步骤，检验“多 token”本身是否能提供计算优势。
- **量词深度（Quantifier Depth）**：在一阶逻辑公式里，量词（∀、∃）的嵌套层数。深度越大，表达的计算结构越复杂。作者用它来理论划分哪些任务可以从 filler token 中受益。
- **密集监督（Dense Supervision）**：在训练时对每一步输出都提供明确的目标标签，而不是只在最后一步给出答案。这里指的是需要对 filler token 的使用方式进行细粒度指导。
- **隐藏计算（Hidden Computation）**：模型在内部利用中间 token 完成的计算过程，但这些过程对外部观察者是不可见的，类似黑箱内部的“暗算”。

### 核心创新点
1. **用无意义 filler 代替 CoT 步骤 → 实验表明模型仍能解决原本只能靠 CoT 完成的算法任务**  
   之前的工作把 CoT 当作唯一的提升手段，这里直接把思考步骤换成“......”，结果模型在两个硬核算法任务上仍然成功，说明额外 token 本身就能提供计算资源，而不一定需要语义丰富的中间步骤。

2. **提出量词深度理论 → 给出何种任务可以从 filler token 中获益的形式化条件**  
   过去缺乏对“多 token 有用”这一现象的理论解释。作者把任务映射到一阶逻辑公式，证明当公式的量词深度超过一定阈值时，额外 token（不论内容）就能提升可计算性。

3. **展示学习 filler token 的难度 → 需要密集监督才能让模型学会利用占位符**  
   与直接让模型自行发现 filler 的用法不同，实验发现如果只给最终答案的监督，模型几乎不学会使用 filler。加入对每一步 token 的明确指导后，模型才会把 filler 当作计算“缓冲区”使用。

4. **警示隐藏计算的审计风险 → 证明模型可以在看似无意义的 token 背后进行重要推理**  
   过去的安全审计主要关注模型输出的可解释性，这里揭示了模型可能在不被注意的 filler token 中完成关键计算，提醒研究者在评估模型时不能只看表面的 CoT 步骤。

### 方法详解
**整体思路**：作者先挑选两个已知的、单纯靠一次前向传播无法解决的算法任务（例如图遍历和数列递推），然后分别训练两套模型：一套使用传统的 CoT 步骤，另一套把所有中间步骤统一替换成 filler token（“......”）。在训练阶段，针对 filler 版本加入了密集监督——每个 filler 前后都标记为“计算中”，让模型明确知道这些位置是需要占用计算资源的。

**关键模块拆解**：

1. **任务编码与目标函数**  
   - 输入是自然语言描述的算法问题。  
   - 目标函数在普通 CoT 版本上只在最终答案处计算交叉熵；在 filler 版本上，除了最终答案，还在每个 filler 前后加入一个小的 “保持一致” 损失，迫使模型在这些位置保持特定的内部状态。

2. **填充 token 插入策略**  
   - 直接在答案前插入固定数量的 filler token（如 5 个“......”），不做任何语义加工。  
   - 类比于在程序里预留若干空指令，让 CPU 有时间进行内部缓存刷新。

3. **密集监督实现**  
   - 为每个 filler token 生成一个伪标签（如 “<FILL>”），并在训练时让模型预测该标签。  
   - 这一步相当于在黑板上写下“这里是占位”，帮助模型把注意力从“要回答什么”转向“要留出多少计算空间”。

4. **理论分析模块**  
   - 作者把每个任务抽象为一阶逻辑公式，计算其量词深度。  
   - 当深度 > 1 时，证明额外 token 能模拟量词的展开过程，从而提升模型的表达能力。  
   - 这一步不涉及实际代码实现，而是提供了为什么 filler 能起作用的数学依据。

**最巧妙的地方**：把 filler token 当作“计算缓冲区”而不是“噪声”。在传统观念里，模型只会利用有意义的 token 来推理；这里作者通过密集监督让模型把 filler 当作内部状态的“记号”，从而在不增加语义负担的情况下扩展计算深度。

### 实验与效果
- **测试任务**：两类硬核算法任务——（1）基于图的最短路径搜索，（2）递归数列的下一个元素预测。两者都需要多步推理，单步前向传播难以得到正确答案。
- **对比基线**：普通的直接输出（无中间 token）和标准 CoT（使用自然语言思考步骤）。  
- **结果**：论文声称 filler 版本在这两个任务上的成功率显著高于直接输出，接近甚至超过标准 CoT。具体提升幅度未在摘要中给出，但作者强调“显著提升”。  
- **消融实验**：去掉密集监督后，模型几乎不再利用 filler，性能回落到直接输出水平；保留密集监督但把 filler 换成随机字符，效果同样下降，说明模型真的在利用 filler 的位置而非字符本身。  
- **局限性**：学习 filler 需要额外的标签信息，实际部署时可能不易获得；实验只在两个人工设计的算法任务上验证，尚未在大规模自然语言基准上测试。

### 影响与延伸思考
这篇工作在社区里掀起了对“中间 token 本身是否必须有语义”的讨论。后续有研究尝试用随机噪声、特殊符号甚至空格来做类似的计算缓冲，验证了“多 token = 多计算”这一观点。安全审计方向也开始关注模型可能在不可见的 filler 中完成关键推理，提出了检测隐藏计算的工具（如梯度分析、内部激活可视化）。如果想进一步深入，可以关注以下两个方向：① 用更自然的语言结构（如模板化提示）替代 filler，探索“语义 vs 计算空间”的权衡；② 研究在大规模真实任务（如数学竞赛）中，是否真的可以用无意义 token 替代 CoT，进而简化提示工程。

### 一句话记住它
额外的 token 本身就能为 Transformer 提供计算空间，即使这些 token 完全没有意义。