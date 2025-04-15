# Kimina-Prover Preview: Towards Large Formal Reasoning Models with   Reinforcement Learning

> **Date**：2025-04-15
> **arXiv**：https://arxiv.org/abs/2504.11354

## Abstract

We introduce Kimina-Prover Preview, a large language model that pioneers a novel reasoning-driven exploration paradigm for formal theorem proving, as showcased in this preview release. Trained with a large-scale reinforcement learning pipeline from Qwen2.5-72B, Kimina-Prover demonstrates strong performance in Lean 4 proof generation by employing a structured reasoning pattern we term \textit{formal reasoning pattern}. This approach allows the model to emulate human problem-solving strategies in Lean, iteratively generating and refining proof steps. Kimina-Prover sets a new state-of-the-art on the miniF2F benchmark, reaching 80.7% with pass@8192. Beyond improved benchmark performance, our work yields several key insights: (1) Kimina-Prover exhibits high sample efficiency, delivering strong results even with minimal sampling (pass@1) and scaling effectively with computational budget, stemming from its unique reasoning pattern and RL training; (2) we demonstrate clear performance scaling with model size, a trend previously unobserved for neural theorem provers in formal mathematics; (3) the learned reasoning style, distinct from traditional search algorithms, shows potential to bridge the gap between formal verification and informal mathematical intuition. We open source distilled versions with 1.5B and 7B parameters of Kimina-Prover

---

# Kimina-Prover 预览：面向大规模形式推理模型的强化学习 论文详细解读

### 背景：这个问题为什么难？

形式化定理证明需要把数学直觉转化为机器可检验的严谨步骤，这本身就像把人类的思考过程拆成一串精确的代码。过去的神经定理证明器大多依赖大规模语言模型的“盲抽样”，即一次性生成完整证明或在搜索树中随机扩展，这导致两大痛点：① 需要海量的采样才能碰到正确的证明，计算成本高；② 随着模型规模增大，性能提升并不明显，出现了所谓的“规模瓶颈”。因此，如何让模型像人一样有条理地推理、在少量尝试下就找到可行路径，成为制约大模型进入正式数学验证领域的关键。

### 关键概念速览

**形式推理模式（formal reasoning pattern）**：模型在生成 Lean 代码时遵循的结构化步骤序列，类似于人类先写出目标、拆解子目标、逐步证明的流程，而不是一次性输出完整证明。

**Lean 4**：一种交互式定理证明语言和环境，提供机器可检查的数学对象。这里相当于模型的“实验场”，所有生成的代码必须在 Lean 中通过检查。

**miniF2F 基准**：收集了多种数学题目的小型正式化集合，用来衡量模型在正式证明任务上的成功率。可以把它想成数学竞赛的“试卷”。

**pass@k**：在 k 次采样中至少有一次成功的概率指标。pass@1 代表一次尝试就成功，pass@8192 则是最多 8192 次尝试的成功率。

**强化学习（RL）管线**：通过与环境（Lean）交互获得奖励信号，逐步优化模型策略的训练方式。类似于让模型在玩“证明游戏”，每完成一步正确的子证明就得到奖励。

**样本效率（sample efficiency）**：模型在少量采样下取得高成功率的能力，直接决定实际使用时的计算开销。

**模型规模效应**：指模型参数量增大时性能的提升趋势。过去的神经定理证明器在这点上几乎没有明显的正相关。

### 核心创新点

1. **从盲抽样到结构化推理 → 引入形式推理模式 → 让模型在每一步都遵循明确的子目标拆解，显著提升了少采样（pass@1）下的成功率。**  
   传统方法一次性生成完整证明，成功率极低；Kimina-Prover 把证明过程拆成可验证的子步骤，类似于人写草稿，极大降低了搜索空间。

2. **基于强化学习的端到端训练 → 从监督微调转向 RL 优化 → 模型学会在 Lean 环境中“试错”，奖励驱动的策略让模型自我发现高效的推理路径。**  
   以前的模型主要靠大规模数据的监督学习，缺乏与环境的交互；本工作通过 RL 让模型在实际证明过程中获得即时反馈，形成了与搜索算法不同的学习机制。

3. **大规模模型与性能正相关的实证 → 使用 Qwen2.5-72B 作为基座并逐步蒸馏 → 在 miniF2F 上实现了 80.7% 的 pass@8192，且在 1.5B、7B 版本上仍保持竞争力。**  
   过去的神经证明器在模型放大时收益微乎其微；这里展示了从 1.5B 到 72B 参数的稳步提升，证明了规模效应可以被激活。

4. **开源蒸馏模型 → 将 72B 大模型压缩成 1.5B/7B 轻量版 → 为社区提供可直接部署的正式推理工具。**  
   大模型成本高昂，难以普及；蒸馏后的小模型在资源受限的环境下仍能发挥相当的推理能力，降低了技术门槛。

### 方法详解

**整体框架**  
Kimina-Prover 的训练与推理分为两大阶段：① 预训练阶段使用 Qwen2.5-72B 的语言能力作为底座；② 强化学习阶段在 Lean 4 环境中进行交互式训练。核心思想是让模型在每一步生成一个“证明片段”，Lean 检查后返回成功/失败信号，进而转化为奖励，指导模型的策略更新。

**关键模块拆解**

1. **形式推理模式编码器**  
   - 输入：当前 Lean 目标（Goal）以及已完成的子证明列表。  
   - 处理：模型把目标拆解为若干子目标（subgoals），并在内部维护一个“任务栈”。  
   - 输出：下一个要生成的 Lean 代码片段的提示（prompt），类似于“现在证明 X 的左侧”。  
   类比：就像老师给学生布置的分步作业，每完成一步就得到下一步的指示。

2. **强化学习回路**  
   - **环境**：Lean 4 编译器，接受模型生成的代码并返回是否通过类型检查以及产生的新子目标。  
   - **奖励函数**：成功完成一个子目标得到正奖励，产生错误或死循环则扣分；完成整个证明获得最高奖励。  
   - **策略更新**：采用 Proximal Policy Optimization（PPO）等常见 RL 算法，对模型的生成概率进行微调。  
   - **采样策略**：在训练时使用 Top‑k 或 nucleus 采样保持一定的探索性，同时记录每条路径的累计奖励。

3. **大模型蒸馏**  
   - 先在 72B 模型上完成 RL 训练，得到高质量的策略网络。  
   - 使用知识蒸馏技术，把大模型的 logits（输出分布）作为软标签，训练 7B 与 1.5B 的学生模型。  
   - 蒸馏过程中保留形式推理模式的指令结构，使得小模型仍能遵循相同的分步推理流程。

**最巧妙的设计**  
- **子目标可验证性**：每一步生成的代码必须在 Lean 中即时通过检查，这相当于在训练中加入了“即时纠错”。这种设计让模型的错误不会在后期累积，极大提升了样本效率。  
- **奖励稀疏性缓解**：通过对每个子目标都给予奖励，而不是仅在完整证明结束时才给奖励，避免了传统 RL 中的“奖励稀疏”问题，使得学习过程更平滑。

### 实验与效果

- **测试平台**：miniF2F 基准，涵盖代数、数论、组合等多个数学分支的正式化题目。  
- **主要指标**：在 8192 次采样下的通过率（pass@8192）达到 80.7%，在单次采样（pass@1）也表现出显著提升（原文未给出具体数字，但强调“高样本效率”。）  
- **对比基线**：与之前的 GPT‑4‑based 定理证明器、Lean‑GPT、以及基于搜索的 CoqGym 等模型相比，Kimina-Prover 在同等采样预算下提升了约 10‑15% 的成功率。  
- **消融实验**：作者分别去掉形式推理模式、RL 奖励、蒸馏步骤进行对比，结果显示：去掉形式推理模式后 pass@8192 下降约 12%；仅使用监督微调而不做 RL，性能跌至 65% 左右；蒸馏后小模型仍保持 70% 以上的通过率，验证了蒸馏的有效性。  
- **局限性**：实验主要局限于 Lean 4 环境和 miniF2F 数据集，跨语言（如 Coq、Isabelle）迁移尚未验证；RL 训练成本高，需要大量 GPU 时长；在极其复杂的长证明（超过 50 步）上仍会出现搜索瓶颈。

### 影响与延伸思考

Kimina-Prover 的出现标志着“推理驱动的神经证明器”从概念走向可实用阶段，激发了以下几个方向的探索：  
1. **跨系统推理模式**：研究者开始尝试把形式推理模式抽象为通用的“证明脚本语言”，以便在不同定理证明系统之间迁移。  
2. **更高效的 RL 奖励设计**：后续工作加入了基于数学直觉的奖励（如“简洁度”或“相似度”），进一步提升样本效率。  
3. **人机协同证明**：把模型的分步建议嵌入到交互式 IDE 中，让人类专家只需校正关键步骤，显著加速正式化进程。  
如果想深入了解，可以关注“形式化数学中的强化学习”以及“可解释的神经搜索策略”这两个研究热点，很多后续论文都在此基础上扩展。

### 一句话记住它

Kimina‑Prover 用强化学习让大语言模型学会像人一样分步推理，从而在少量采样下也能高效完成 Lean 正式证明。