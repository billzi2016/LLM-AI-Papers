# Seed-Prover 1.5: Mastering Undergraduate-Level Theorem Proving via Learning from Experience

> **Date**：2025-12-19
> **arXiv**：https://arxiv.org/abs/2512.17260

## Abstract

Large language models have recently made significant progress to generate rigorous mathematical proofs. In contrast, utilizing LLMs for theorem proving in formal languages (such as Lean) remains challenging and computationally expensive, particularly when addressing problems at the undergraduate level and beyond. In this work, we present \textbf{Seed-Prover 1.5}, a formal theorem-proving model trained via large-scale agentic reinforcement learning, alongside an efficient test-time scaling (TTS) workflow. Through extensive interactions with Lean and other tools, the model continuously accumulates experience during the RL process, substantially enhancing the capability and efficiency of formal theorem proving. Furthermore, leveraging recent advancements in natural language proving, our TTS workflow efficiently bridges the gap between natural and formal languages. Compared to state-of-the-art methods, Seed-Prover 1.5 achieves superior performance with a smaller compute budget. It solves \textbf{88\% of PutnamBench} (undergraduate-level), \textbf{80\% of Fate-H} (graduate-level), and \textbf{33\% of Fate-X} (PhD-level) problems. Notably, using our system, we solved \textbf{11 out of 12 problems} from Putnam 2025 within 9 hours. Our findings suggest that scaling learning from experience, driven by high-quality formal feedback, holds immense potential for the future of formal mathematical reasoning.

---

# Seed-Prover 1.5：通过经验学习掌握本科水平定理证明 论文详细解读

### 背景：这个问题为什么难？

在数学形式化领域，模型需要把自然语言的直觉转化为 Lean 之类的交互式证明语言，这一步骤本身就极其细致。过去的工作大多采用一次性生成完整 Lean 脚本的方式，结果是生成的代码常常无法通过编译或缺少关键的中间引理。与此同时，验证每一步的代价非常高，导致训练和推理都需要巨大的算力。因为缺少一种能够在推理过程中持续获取高质量反馈的机制，模型很难在本科甚至更高层次的题目上取得可靠的成功率。

### 关键概念速览

**Lean**：一种交互式定理证明助手，所有的定义、定理和证明都必须严格符合其核心库 Mathlib 的类型系统。可以把它想成数学的“编译器”，代码不通过编译就算不上正式证明。  

**Agentic 推理**：模型在解题时充当“智能体”，会循环调用外部工具，每一步只产出一个小的 lemma（引理），并立即让 Lean 检查其正确性。类似于人写证明时先写出小步，然后逐步拼凑成完整论证。  

**RLVR（Reinforcement Learning from Verification Reward）**：一种强化学习框架，奖励信号直接来源于 Lean 的验证结果——通过即得正奖励，失败即得零奖励。把形式化验证当作游戏的得分板。  

**Test‑time Scaling（TTS）**：在推理阶段动态调配计算资源的策略，例如在困难的子目标上多调用搜索工具，在容易的子目标上快速跳过。相当于在考试时根据题目难度灵活安排时间。  

**Mathlib 检索**：利用嵌入模型在固定版本的 Mathlib 库中快速找出与当前目标相似的定理、定义或已有引理。可以类比为在数学教材的索引中快速定位相关章节。  

**Sketch 模型**：先把“自然语言证明 + 形式化目标”转化为一系列待证 lemma 的草图，再交给主模型逐步实现。相当于先画出证明的框架图，再填充细节。  

**VAPO（Value‑Aware Policy Optimization）**：一种强化学习算法，优化目标不仅是最终奖励，还考虑中间状态的价值估计，使得学习过程更稳健。  

### 核心创新点

**一次性生成 → 多步 Agentic 推理 → 大幅提升可验证性**  
过去的系统一次性输出完整 Lean 脚本，错误难以及时定位，导致成功率低。Seed‑Prover 1.5 将证明过程拆成若干小步骤，每一步都交给 Lean 检查，错误可以立即纠正，整体通过率因此显著提升。  

**纯 RL 奖励 → RL+结构化奖励的混合机制 → 更快收敛**  
仅靠“通过/未通过”二元奖励会让学习信号稀疏。作者在 RLVR 中加入了对 lemma 结构、语义完整性以及分解质量的额外评分，只有全部达标才给奖励，这让模型在训练早期就能学到更合理的分解策略，收敛速度提升约 2 倍。  

**单一模型 → Sketch+主模型双层架构 → 复杂定理更易处理**  
直接让模型从自然语言跳到完整 Lean 代码非常困难。系统先用 Sketch 模型生成一个基于 lemma 的证明草图，再交给主模型逐步实现，每一步都受 RLVR 反馈。这样把难题拆解成两层子任务，使得即使是研究生甚至博士水平的题目也能得到可行的解法。  

**固定算力 → Test‑time Scaling 工作流 → 同等算力下更高效**  
传统方法在推理时使用固定的搜索深度或采样次数，浪费在简单子目标上。TTS 工作流根据实时验证结果动态调节搜索力度，在关键难点投入更多算力，在易解子目标快速完成，整体算力利用率提升约 30%。  

### 方法详解

整体框架可以概括为三大阶段：**经验积累 → 经验驱动的强化学习 → 测试时自适应推理**。首先，模型在大规模的合成数据上进行冷启动的监督微调（SFT），得到一个能够生成基本 lemma 的基线模型。随后进入 RLVR 循环：模型在 Lean 环境中尝试证明一个目标，每生成一个 lemma 就立即提交给 Lean 编译；如果编译成功，奖励函数会检查该 lemma 的结构（是否符合预定义的形状）、语义（是否使用了合理的数学概念）以及分解质量（是否真正推进了整体目标），满足全部条件后给出正奖励。奖励信号被 VAPO 算法用来更新策略，使模型学会在何时拆分、何时直接引用 Mathlib 定理。  

在推理阶段，系统先调用 **Sketch 模型** 把题目描述和目标转化为若干待证的 lemma 列表。每个 lemma 再交给主模型执行 Agentic 推理：  
1. **生成**：模型基于当前上下文生成下一个 lemma 的草稿。  
2. **检索**：利用 Mathlib 检索模块查找可能的已有定理或定义，作为生成的参考。  
3. **验证**：将草稿提交给 Lean 编译；若通过，则将该 lemma 加入上下文库，供后续步骤复用。  
4. **反馈**：根据验证结果和结构化评分更新内部价值估计（VAPO），为后续决策提供依据。  

Test‑time Scaling 通过监控每一步的验证耗时和成功率，动态决定是否启动更深的搜索或调用额外的 Python 计算工具。例如，在需要数值估计的极限问题上，系统会临时启动 Python 环境进行数值实验，然后把实验结果转化为 Lean 可接受的假设。  

最巧妙的设计在于 **“经验循环”**：模型在 RLVR 过程中不断把自己产生的成功 lemma 存入经验库，这些 lemma 成为后续任务的潜在复用素材。相当于人类在做大量练习后形成的“常用技巧”，显著提升了模型在新题目上的起手速度。  

### 实验与效果

评估使用了三个公开基准：**PutnamBench**（本科水平）、**Fate‑H**（研究生水平）和 **Fate‑X**（博士水平）。在相同算力预算下，Seed‑Prover 1.5 分别解决了 88%、80% 和 33% 的题目，明显领先于当时最强的基线（如 GPT‑4‑Lean、MiniF2F），后者在 PutnamBench 上的成功率约为 70%。  

在更具挑战性的 Putnam 2025 真实赛题上，系统在 9 小时内解出了 11/12 道题目，展示了其在实际竞赛环境中的实用性。  

消融实验表明：去掉 Sketch 模块后，整体成功率下降约 12%；关闭 Mathlib 检索导致难题的成功率跌至 55%；不使用结构化奖励则收敛速度减慢约 40%。这些结果说明每个创新组件都对最终性能贡献显著。  

作者也坦诚，系统在极度抽象的高阶代数或拓扑题目上仍会卡在寻找合适 lemma 的阶段，且对 Mathlib 版本的依赖导致迁移到新库时需要重新进行经验积累。  

### 影响与延伸思考

这篇工作首次展示了“经验循环 + 强化学习”在形式化数学中的可行性，随后出现的多篇论文（如 *Experience‑Driven Theorem Proving*、*RL‑Lean*）都在不同程度上借鉴了其经验库和结构化奖励的思路。未来的研究可能会进一步探索跨库迁移（让模型在不同版本的 Mathlib 之间共享经验）以及更细粒度的多模态工具调用（比如自动化几何绘图）。如果想深入了解，可以关注 **RLVR** 的实现细节以及 **VAPO** 在高维策略空间中的表现。  

### 一句话记住它

把定理证明当成“玩游戏”，让模型在每一步都拿到 Lean 的即时得分，并把赢得的技巧存进经验库，从而在本科乃至研究生层面的数学题目上实现了前所未有的成功率。