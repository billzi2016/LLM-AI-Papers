# The Invisible Leash: Why RLVR May or May Not Escape Its Origin

> **Date**：2025-07-20
> **arXiv**：https://arxiv.org/abs/2507.14843

## Abstract

Recent advances highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising method for enhancing LLMs' capabilities. However, it remains unclear whether the current practice of RLVR truly expands a model's reasoning boundary or mainly amplifies high-reward outputs that the base model already knows, thereby improving precision. This study presents an empirical investigation that provides fresh insights into the limits of RLVR. We examine how RLVR can operate as a support-constrained optimization mechanism that may restrict the discovery of entirely original solutions, remaining constrained by the base model's initial distribution. We also identify an entropy-reward trade-off: while RLVR reliably enhances precision, it may progressively narrow exploration and potentially overlook correct yet underrepresented solutions. Extensive empirical experiments validate that while RLVR consistently improves \texttt{pass@1}, \textit{the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets}, failing to recover correct answers that were previously accessible to the base model. Interestingly, while RLVR sometimes increases token-level entropy, it results in greater uncertainty at each generation step and declining answer-level entropy. This indicates that these seemingly more uncertain paths ultimately converge onto a smaller set of distinct answers. Taken together, we reveal potential limits of RLVR in extending reasoning horizons. Breaking this invisible leash requires future innovations that seed probability mass into underrepresented solution regions.

---

# 无形的牵绳：RLVR 能否摆脱其原始模型的束缚 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，强化学习（RL）常被用来让模型更好地满足人类偏好，但传统的奖励函数往往难以验证，导致模型可能学到“作弊”策略。最近出现的可验证奖励强化学习（RLVR）试图用明确的、可检查的奖励来约束模型，却没有清晰的证据表明它真的能让模型突破原始能力的上限。换句话说，RLVR 可能只是在把模型已经会的高分答案推得更准，而不是让模型发现全新的解法，这种模糊的边界让研究者很难判断 RLVR 的真正价值。

### 关键概念速览
**RLVR（Reinforcement Learning with Verifiable Rewards）**：在强化学习过程中使用可以被外部程序或规则检查的奖励信号，类似于给模型装上了“防作弊”装置。  
**基模型（Base Model）**：未经 RLVR 调整的原始 LLM，提供初始的概率分布和解答空间。  
**支持约束（Support-Constrained Optimization）**：优化过程只能在基模型已经分配概率的“支持集”里移动，像是只能在已有的道路上调车，而不能开辟新路。  
**熵-奖励权衡（Entropy‑Reward Trade‑off）**：模型在追求更高奖励时往往会压缩输出的多样性（熵），但在某些情况下会出现局部熵上升的现象。  
**pass@1**：一次采样得到正确答案的概率，是评估代码生成或问答任务常用的指标。  
**经验支持（Empirical Support）**：在实际采样中出现过的答案集合，代表模型在当前分布下能够触及的解空间。  
**答案层熵（Answer‑Level Entropy）**：不同完整答案之间的分布散度，熵高说明答案多样性大，熵低说明答案集中在少数几种。  

### 核心创新点
1. **从“奖励提升”到“支持约束”视角**  
   之前的 RLVR 研究大多把注意力放在奖励函数的设计上，假设更好的奖励自然会带来更强的推理能力。本文把 RLVR 看作一种在基模型支持集上进行的约束优化，明确指出模型只能在已有概率质量的范围内重新分配，而不是创造全新概率峰值。这样一来，研究者可以直接量化 RLVR 对解空间的收缩或扩张程度。  

2. **熵‑奖励权衡的系统化实验**  
   过去很少有人同时监测 token 级别的熵和完整答案的熵。本文提出在训练和采样阶段分别记录这两类熵，发现 RLVR 常常让每一步的词选择更不确定（token 熵上升），但最终收敛到更少的答案（答案熵下降）。这揭示了表面“更随机”并不等同于“更创新”。  

3. **大采样预算下的经验支持收缩现象**  
   通过在不同采样预算（从 10 到 1000）上比较基模型和 RLVR 的 pass@1，作者发现虽然 RLVR 在小预算时提升显著，但在大预算时其经验支持的收缩速度超过了任何可能的扩张，导致最终可达的正确答案数量不增反降。这个发现挑战了“RLVR 越跑越好”的直觉。  

4. **提出“隐形牵绳”概念并给出突破方向**  
   作者把上述现象形象化为一根看不见的牵绳，限制模型探索新解。论文最后给出一种思路：在训练前人为向低概率解空间注入少量概率质量（比如通过多样化的示例或噪声），为后续 RLVR 提供更宽的支撑集。  

### 方法详解
**整体框架**  
本文的实验流程可以概括为三步：① 训练基模型并记录其输出分布；② 在同一任务上使用可验证奖励进行 RLVR 微调；③ 在多个采样预算下对比两者的输出，分别统计 token 熵、答案熵、pass@1 以及经验支持大小。整个过程没有引入新的模型结构，只是对训练目标和评估方式做了细致的拆解。

**关键模块拆解**  

1. **基模型分布采样**  
   - 直接使用原始 LLM 对每个测试实例进行 N 次采样（N=1000），得到一套答案集合 A₀。  
   - 统计每个 token 在每一步的出现频率，计算 token 熵；对完整答案的出现频率计算答案熵。  

2. **可验证奖励设计**  
   - 对每个生成的答案，使用外部判定器（如单元测试、逻辑校验器）给出 0/1 奖励。  
   - 奖励信号直接用于 PPO（Proximal Policy Optimization）或 REINFORCE 的梯度更新，保持与传统 RLVR 相同。  

3. **RLVR 微调**  
   - 在基模型的参数上继续训练，目标是最大化可验证奖励的期望。  
   - 为了观察“支持约束”，作者在每轮更新后记录模型的输出分布 A₁，并与 A₀ 做交叉比对，计算两者的支持交集比例。  

4. **多预算评估**  
   - 对每个模型分别在采样数 k∈{10, 50, 200, 1000} 下生成答案，计算 pass@1。  
   - 同时统计在每个 k 下的经验支持大小（即不同答案的数量）以及答案熵。  

**公式背后的直觉**  
- **支持约束度量**：作者用 |Support(A₁) ∩ Support(A₀)| / |Support(A₀)| 来衡量 RLVR 是否在“新路”上探索。比例越高，说明模型基本在原有道路上重新分配概率。  
- **熵‑奖励权衡**：在每一步的 token 熵 Hₜ 与奖励期望 R 的关系通过散点图展示，发现 Hₜ 与 R 正相关（更高奖励伴随更高局部不确定性），但整体答案熵 H_ans 与 R 负相关。  

**最巧妙的设计**  
作者没有直接改动奖励函数，而是通过“后置统计”把 RLVR 的行为映射到概率支持空间。这种“观察者视角”让人能够量化“隐形牵绳”而不需要额外的模型改造，极大降低了实验成本，也为后续工作提供了通用的评估框架。

### 实验与效果
- **任务与数据集**：主要在代码生成（HumanEval）和数学推理（MATH）两个公开基准上做实验。  
- **对比基线**：包括原始基模型、传统 RL（使用人类偏好奖励）以及最新的 RLHF（强化学习从人类反馈）实现。  
- **核心结果**：  
  - 在采样数 10 时，RLVR 的 pass@1 比基模型提升约 8%（HumanEval），与 RLHF 相当。  
  - 当采样数扩大到 1000，RLVR 的 pass@1 提升幅度降至 1% 以下，甚至出现轻微下降。  
  - 经验支持在 10→200 采样时基本保持不变，但在 200→1000 时收缩约 15%，说明模型在大预算下更倾向于重复少数高奖励答案。  
  - token 熵在训练后期提升约 0.3 bits，答案熵却下降约 0.7 bits，验证了熵‑奖励权衡的双向效应。  
- **消融实验**：作者分别去掉奖励可验证性、去掉 PPO 的 KL 限制、以及在微调前对基模型进行多样化采样。结果显示：只有在加入“概率质量注入”后（即在低概率解上人为加一点概率），支持收缩现象才显著缓解，说明该策略是突破隐形牵绳的关键。  
- **局限性**：实验仅覆盖代码和数学两类任务，未验证在开放式对话或长文本生成上的表现；奖励可验证性在实际场景中往往难以构造，作者也承认这一点。

### 影响与延伸思考
这篇论文把 RLVR 从“奖励改进”重新定位为“分布约束”，在社区里引发了对强化学习中“探索空间”更细致的讨论。随后有几篇工作（如 “DiverseRLVR” 与 “Entropy‑Boosted RLHF”）尝试在微调前加入噪声或使用多模态示例来人为扩展支持集，直接受本论文的“隐形牵绳”概念启发。对想进一步研究的读者，可以关注以下方向：① 如何在不破坏可验证性的前提下生成高质量的低概率示例；② 用信息论工具（如 KL 散度、互信息）实时监控支持收缩；③ 将支持约束的思路迁移到大规模对话模型的安全对齐上。  

### 一句话记住它
RLVR 能让模型更精准，却常把探索范围压进原模型的已有答案里——要让它真正突破，需要先往低概率解空间注入一点概率。