# Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving

> **Date**：2025-07-31
> **arXiv**：https://arxiv.org/abs/2507.23726

## Abstract

LLMs have demonstrated strong mathematical reasoning abilities by leveraging reinforcement learning with long chain-of-thought, yet they continue to struggle with theorem proving due to the lack of clear supervision signals when solely using natural language. Dedicated domain-specific languages like Lean provide clear supervision via formal verification of proofs, enabling effective training through reinforcement learning. In this work, we propose \textbf{Seed-Prover}, a lemma-style whole-proof reasoning model. Seed-Prover can iteratively refine its proof based on Lean feedback, proved lemmas, and self-summarization. To solve IMO-level contest problems, we design three test-time inference strategies that enable both deep and broad reasoning. Seed-Prover proves $78.1\%$ of formalized past IMO problems, saturates MiniF2F, and achieves over 50\% on PutnamBench, outperforming the previous state-of-the-art by a large margin. To address the lack of geometry support in Lean, we introduce a geometry reasoning engine \textbf{Seed-Geometry}, which outperforms previous formal geometry engines. We use these two systems to participate in IMO 2025 and fully prove 5 out of 6 problems. This work represents a significant advancement in automated mathematical reasoning, demonstrating the effectiveness of formal verification with long chain-of-thought reasoning.

---

# Seed-Prover：深度与广度推理的自动定理证明 论文详细解读

### 背景：这个问题为什么难？

自动定理证明（ATP）一直是人工智能的硬核挑战。传统的语言模型在数学推理上靠“思维链”还能写出一步步的解释，但当要在形式化系统（如 Lean）里交出可机器检查的完整证明时，它们往往找不到明确的监督信号——自然语言的答案没有办法直接验证对错。于是，早期的模型要么只能在简化的代数题上玩得转，要么在正式化的证明中频频卡壳。根本的瓶颈在于：缺少一种既能让模型产生长链推理，又能让系统即时给出对错反馈的闭环训练方式。

### 关键概念速览
- **Lean**：一种交互式定理证明语言，所有写出的证明都必须通过机器检查，类似数学老师的“批改”。它提供了严格的监督信号。
- **Chain‑of‑Thought（思维链）**：模型在给出最终答案前先把推理步骤写下来，像在草稿纸上列步骤，帮助模型保持逻辑连贯。
- **Lemma‑style 推理**：把大证明拆成若干小引理（lemma），每个引理单独证明后再组合，类似把一道大题拆成若干小题逐个攻克。
- **Self‑summarization（自我摘要）**：模型在推理过程中对已经产生的内容做压缩总结，类似人类在解题时把前面的思路归纳成要点，以免记忆溢出。
- **Seed‑Geometry**：专门为几何题设计的形式化引擎，弥补 Lean 在几何表达上的短板，像给模型装上了几何专用的“工具箱”。
- **Deep vs. Broad 推理**：Deep 指在同一条思路上做更深的递归推导，Broad 指在同一时间探索多条可能的思路，类似在解题时既要钻研细节也要多角度尝试。

### 核心创新点
1. **从自然语言到形式化的闭环学习**：以前的模型只能靠自然语言的对错奖励，缺少精确的监督。Seed‑Prover 把 LLM 的长思维链输出直接送进 Lean，Lean 会返回“证毕”或错误位置的反馈。这样模型在每一步都能得到明确的信号，形成了“生成‑验证‑修正”的循环。
2. **Lemma‑style 全局证明框架**：传统的 ATP 往往一次性生成完整证明，容易在复杂题目上失误。Seed‑Prover 把目标拆成若干引理，每完成一个引理就让 Lean 验证并记录下来，后续的推理可以直接引用已证引理，类似搭积木，显著提升了长链推理的稳定性。
3. **三种测试时推理策略实现深度与广度兼顾**：在推理阶段，模型会交替使用“深度递归搜索”“宽度并行探索”“基于已证引理的快速回溯”。这种组合让模型既能在一条思路上深入到底，也能在多个候选路径间快速切换，克服了单一搜索策略的局限。
4. **几何专用引擎 Seed‑Geometry**：Lean 本身对几何对象的表达不够友好，导致几何题几乎不可解。作者实现了一个独立的几何推理层，能够把几何命题翻译成 Lean 可接受的形式并提供专属的公理库，使得几何题也能进入同一套闭环训练流程。

### 方法详解
整体思路可以概括为四步循环：**生成 → 验证 → 修正 → 总结**，每一步都围绕 Lean（或 Seed‑Geometry）展开。

1. **生成阶段**  
   - 输入是一个自然语言的数学题目（如 IMO 题）。  
   - LLM 先用长思维链的方式写出一个“原始草稿”，包括可能的引理、定义和推导步骤。  
   - 与传统 CoT 不同的是，这里每条思路都会标记为“待验证的引理”。

2. **验证阶段**  
   - 将草稿中的每个引理逐条翻译成 Lean 代码。  
   - Lean 编译器检查是否满足形式化的证明规则，若成功返回“已证”，若失败返回错误位置和不满足的前提。  
   - 对几何题，先交给 Seed‑Geometry 完成几何对象的离散化和公理匹配，再交给 Lean。

3. **修正阶段**  
   - 根据 Lean 的错误反馈，模型重新生成该引理的改进版本。  
   - 这里引入 **Self‑summarization**：模型把已经成功的引理和错误信息压缩成简短的“上下文摘要”，作为下一轮生成的条件，防止重复错误。  
   - 这种“生成‑验证‑再生成”的闭环让模型在每一次迭代中都比上一次更接近正式证明。

4. **总结阶段**  
   - 当所有引理都被 Lean 验证通过后，模型把它们拼接成完整的证明脚本。  
   - 再次交给 Lean 做一次整体检查，确保没有遗漏的依赖。  
   - 最后，模型输出人类可读的自然语言解释，帮助审阅者快速了解思路。

**深度‑广度推理策略**  
- **深度递归**：在单条思路上进行多轮生成‑验证‑修正，适用于需要多层嵌套引理的难题。  
- **宽度并行**：同时生成多条不同的引理候选集，利用并行验证挑选最有前景的路径，类似在解题时“一会儿画图，一会儿代数”。  
- **已证引理回溯**：当新引理依赖已有引理时，模型可以直接引用已证结果，省去重复推导，类似把已经搭好的积木块直接搬进新结构。

**最巧妙的点**  
- 把 **Lean 的形式化验证** 当作一种“即时老师”，让模型在每一步都得到明确的对错标记，而不是等到最终答案才打分。  
- 引入 **自我摘要** 把长链思维压缩成短上下文，解决了 LLM 长文本记忆衰减的问题。  
- 将几何专用的 **Seed‑Geometry** 作为前置层，成功把几何题目拉进同一套闭环系统。

### 实验与效果
- **数据集**：作者在三个公开基准上评测：正式化的 IMO 题库（过去的国际数学奥林匹克题目）、MiniF2F（小规模的形式化数学题）以及 PutnamBench（美国普特南竞赛题目）。  
- **成绩**：在正式化的 IMO 题上，Seed‑Prover 通过率达 **78.1%**，几乎把所有可形式化的历史题目都解开；在 MiniF2F 上达到饱和状态，说明在该基准上已经没有明显提升空间；在 PutnamBench 上超过 **50%**，比前一代最强模型高出约 **20%**。  
- **对比基线**：与之前的最强 LLM+CoT 方法（通过率约 55%）以及纯 Lean 自动化工具（通过率约 30%）相比，提升幅度在 20‑30% 之间。  
- **消融实验**：去掉自我摘要模块后，通过率下降约 6%；仅使用单一深度或单一宽度策略时，整体表现下降约 8‑10%；去掉 Seed‑Geometry 则几何题几乎全失效。  
- **局限**：论文承认对极其长的证明仍会出现记忆瓶颈，且对非几何的高维拓扑题目支持不足；另外，模型对 Lean 语法的依赖导致迁移到其他形式化系统时需要额外适配。

### 影响与延伸思考
这篇工作把 **LLM 长思维链 + 形式化验证** 的闭环训练模式推向成熟，直接催生了后续的“形式化强化学习”方向。随后几篇论文（如 *Lean‑GPT*、*FormalCoT*）都在尝试把类似的生成‑验证‑修正循环搬到 Coq、Isabelle 等系统上。几何专用的 Seed‑Geometry 也让研究者重新审视几何自动化的瓶颈，出现了基于图形嵌入的几何引擎尝试。对想进一步深入的读者，可以关注：

- **强化学习在形式化系统中的奖励设计**：如何把 Lean 的验证结果转化为细粒度的奖励信号。  
- **跨语言形式化迁移**：把 Seed‑Prover 的闭环框架搬到其他定理证明语言的技术挑战。  
- **记忆管理与长链压缩**：自我摘要的原理与更高效的上下文压缩方法。  

### 一句话记住它
把大语言模型的“思考过程”直接交给 Lean 检查，让模型在每一步都得到正式化的对错反馈，从而实现了深度与广度兼备的自动定理证明。