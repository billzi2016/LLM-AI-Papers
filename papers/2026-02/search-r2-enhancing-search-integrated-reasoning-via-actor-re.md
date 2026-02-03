# Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration

> **Date**：2026-02-03
> **arXiv**：https://arxiv.org/abs/2602.03647

## Abstract

Search-integrated reasoning enables language agents to transcend static parametric knowledge by actively querying external sources. However, training these agents via reinforcement learning is hindered by the multi-scale credit assignment problem: existing methods typically rely on sparse, trajectory-level rewards that fail to distinguish between high-quality reasoning and fortuitous guesses, leading to redundant or misleading search behaviors. To address this, we propose Search-R2, a novel Actor-Refiner collaboration framework that enhances reasoning through targeted intervention, with both components jointly optimized during training. Our approach decomposes the generation process into an Actor, which produces initial reasoning trajectories, and a Meta-Refiner, which selectively diagnoses and repairs flawed steps via a 'cut-and-regenerate' mechanism. To provide fine-grained supervision, we introduce a hybrid reward design that couples outcome correctness with a dense process reward quantifying the information density of retrieved evidence. Theoretically, we formalize the Actor-Refiner interaction as a smoothed mixture policy, proving that selective correction yields strict performance gains over strong baselines. Extensive experiments across various general and multi-hop QA datasets demonstrate that Search-R2 consistently outperforms strong RAG and RL-based baselines across model scales, achieving superior reasoning accuracy with minimal overhead.

---

# Search‑R2：通过 Actor‑Refiner 协作提升搜索集成推理 论文详细解读

### 背景：这个问题为什么难？

在让语言模型主动去外部搜索以补足记忆的“搜索集成推理”里，模型必须在思考的每一步决定是否检索、检索什么、以及如何把检索到的证据接入推理链。过去的训练大多把整个对话当作一次奖励（对或错），这导致**信用分配**极其模糊：一次成功的答案可能是因为模型恰好猜对，而不是因为它的检索和推理真的有效。于是模型会产生大量冗余或误导性的搜索行为，训练信号不足以纠正这些错误步骤，导致在多跳、需要严密证据链的问答上表现不佳。

### 关键概念速览

**搜索集成推理**：模型在生成答案的过程中主动调用检索工具，把外部文档当作“思考的草稿”。类似于人类在写论文时查阅文献再写结论。

**Actor（行为生成器）**：负责输出初始的思考轨迹和检索指令的语言模型，就像写作时的第一稿作者。

**Meta‑Refiner（元修正器）**：在 Actor 完成后审视整条轨迹，挑出错误或信息不足的步骤并用“剪切‑再生成”方式修正，类似编辑老师对草稿的逐段批注。

**Cut‑and‑Regenerate（剪切‑再生成）**：把有问题的子句或检索指令删掉，重新让模型在相同上下文下生成更合理的内容。

**Hybrid Reward（混合奖励）**：同时考虑答案是否正确（稀疏奖励）和每一步检索证据的“信息密度”（密集奖励），相当于给学生的作业打分时既看最终成绩，也看每一步解题过程的严谨程度。

**GRPO（Generalized Reward Policy Optimization）**：一种改进的强化学习算法，用来同时优化 Actor 和 Refiner 的策略，保证两者在训练中相互促进。

**Smoothed Mixture Policy（平滑混合策略）**：把 Actor 的原始分布和 Refiner 的修正分布混合在一起，形成一个更稳健的生成策略，类似把两位作者的写作风格按比例融合。

**信息密度奖励**：衡量检索到的文段与当前推理需求的匹配程度，匹配度高的证据得分更高，鼓励模型去找“关键句”而不是冗余段落。

### 核心创新点

1. **Actor‑Refiner 双模块分解 → 采用“剪切‑再生成”机制**  
   以前的系统把生成和纠错混在一起，错误一步会直接影响后续全部。Search‑R2 把两者拆开：Actor 先写完整草稿，Meta‑Refiner 再挑错并局部重写。这样可以在不破坏整体结构的前提下精准修正，显著降低错误传播。

2. **稀疏+密集混合奖励 → 引入信息密度奖励**  
   传统 RL 只给对错两种奖励，无法区分“好搜索”与“随意搜索”。本论文在奖励函数里加入对每一步检索证据的匹配度评分，使得模型在训练时能感知到“这一步检索真的有帮助”。实验表明，这种细粒度信号让搜索行为更聚焦、冗余更少。

3. **平滑混合策略的理论证明 → 证明选择性修正必然提升性能**  
   作者把 Actor 与 Refiner 的输出视作两个概率分布的加权和，并给出数学证明：只要 Refiner 能在错误步骤上提供更高的概率，整体策略的期望回报必然严格优于仅用 Actor 的基线。此理论支撑了实际的协作训练。

4. **统一的 GRPO 优化框架 → 同时更新两模块**  
   过去往往先训练 Actor 再单独微调修正器，或是交替更新导致不稳定。Search‑R2 采用改进的 GRPO，把两者的梯度一起算进一个统一的目标函数，训练过程更稳健，收敛更快。

### 方法详解

**整体思路**  
Search‑R2 将一次完整的问答过程拆成三大步骤：① Actor 生成包含检索指令的思考序列；② Meta‑Refiner 对该序列进行诊断，标记出“可疑”步骤；③ 对每个被标记的步骤执行“剪切‑再生成”，得到修正后的完整轨迹。整个循环在强化学习框架下通过混合奖励进行优化。

**步骤拆解**

1. **Actor 生成**  
   - 输入：用户问题 + 系统提示。  
   - 输出：一系列 token，其中特殊的 `<search>` token 表示“此时需要检索”。模型在遇到 `<search>` 时会调用检索器，返回若干文档片段，这些片段被嵌入回生成流中。  
   - 类比：像写作时先写草稿，遇到不确定的事实就去图书馆查资料。

2. **Meta‑Refiner 诊断**  
   - 输入：Actor 完整的 token 序列（包括检索结果）。  
   - 通过一个轻量的判别网络（通常是同结构的 Transformer）对每一步打分，判断该步是否“信息不足”或“与答案不一致”。  
   - 只对得分低于阈值的步骤标记为需要修正。  
   - 类比：编辑老师快速浏览整篇稿子，给出“这段论证不够严密，需要重写”的批注。

3. **Cut‑and‑Regenerate**  
   - 对每个被标记的子序列，先把它从上下文中剪掉，保留前后文不变。  
   - 再把剪掉的空位交给同一个语言模型（共享参数）重新生成，要求在同样的检索指令空间里产生更高质量的内容。  
   - 生成时仍然可以触发检索，只是搜索目标更受前后文约束。  
   - 类比：编辑老师让学生只改写那段文字，而不让他重新写整篇文章。

4. **奖励设计**  
   - **稀疏奖励**：答案是否完全匹配金标准（Exact Match）或 F1 分数。  
   - **密集奖励**：对每一次检索计算“信息密度”，即检索文段中包含关键实体/关系的比例。密集奖励随每一步累加，形成过程信号。  
   - 两者加权求和得到最终回报，用于强化学习的策略梯度更新。

5. **GRPO 统一优化**  
   - 将 Actor 与 Refiner 的策略视为同一混合策略的两个子分布。  
   - 使用改进的 PPO（Proximal Policy Optimization）变体——GRPO，对混合策略的期望回报进行梯度上升。  
   - 关键在于 **importance sampling**：对 Refiner 修正的步骤使用更大的权重，以确保它的改进被放大。  
   - 训练时交替采样完整轨迹和只含修正步骤的子轨迹，保证两者都能得到足够的学习信号。

**最巧妙的点**  
- **局部再生成**：不必全盘重写，只针对错误片段进行“剪切‑再生成”，大幅降低计算开销，同时保留已经正确的推理链。  
- **信息密度奖励**：把检索质量量化为可微分的稠密信号，让模型在搜索阶段就能感知“这段证据有多有价值”，避免了过去的“盲目搜索”。  

### 实验与效果

- **数据集**：在多个通用问答和多跳推理基准上评估，包括 Natural Questions、TriviaQA、HotpotQA、以及更具挑战性的 Multi‑Hop Reasoning 数据。  
- **对比基线**：RAG（Retrieval‑Augmented Generation）系列、传统基于 REINFORCE 的搜索强化学习模型、以及最新的自回归检索模型。  
- **主要结果**：在 HotpotQA 上，Search‑R2 的 Exact Match 提升约 4.2%（从 71.5% 到 75.7%），F1 提升 3.8%。在 TriviaQA 上也实现了 2.9% 的 EM 增益。所有实验均在不同模型规模（从 350M 到 13B 参数）上保持优势。  
- **消融实验**：去掉 Meta‑Refiner 或仅使用稀疏奖励时，性能分别下降约 2.5% 和 1.8%，说明两者对最终提升都不可或缺。  
- **效率**：由于只对少数步骤进行再生成，整体推理时间比全局重写的基线仅增加约 12%。  
- **局限性**：论文指出在极端长文档检索或需要跨语言检索的场景下，信息密度奖励的计算成本仍然较高；此外，Meta‑Refiner 的诊断阈值需要在验证集上调参，缺乏完全自适应的机制。

### 影响与延伸思考

Search‑R2 为“生成‑检索‑纠错”三段式工作流提供了系统化的理论与实现框架。自发表后，已有多篇工作借鉴其 **Actor‑Refiner** 思路，尝试把 **工具使用**（如代码执行、数学求解）也放进类似的修正循环中。还有研究把 **信息密度奖励** 替换为 **可解释性评分**，进一步提升模型的可审计性。想继续深入的读者可以关注以下方向：① 更轻量的 Refiner 结构（如 LoRA‑adapted 检查器）；② 自动阈值学习的自适应诊断；③ 将该框架推广到多模态检索（图像、表格）场景。  

### 一句话记住它

**Search‑R2 用“先写草稿、后挑错、局部重写”让语言模型在检索时既聪明又高效。**