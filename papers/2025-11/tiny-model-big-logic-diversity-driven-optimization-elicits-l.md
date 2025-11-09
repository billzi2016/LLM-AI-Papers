# Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B

> **Date**：2025-11-09
> **arXiv**：https://arxiv.org/abs/2511.06221

## Abstract

Challenging the prevailing consensus that small models inherently lack robust reasoning, this report introduces VibeThinker-1.5B, a 1.5B-parameter dense model developed via our Spectrum-to-Signal Principle (SSP). This challenges the prevailing approach of scaling model parameters to enhance capabilities, as seen in models like DeepSeek R1 (671B) and Kimi k2 (>1T). The SSP framework first employs a Two-Stage Diversity-Exploring Distillation (SFT) to generate a broad spectrum of solutions, followed by MaxEnt-Guided Policy Optimization (RL) to amplify the correct signal. With a total training cost of only $7,800, VibeThinker-1.5B demonstrates superior reasoning capabilities compared to closed-source models like Magistral Medium and Claude Opus 4, and performs on par with open-source models like GPT OSS-20B Medium. Remarkably, it surpasses the 400x larger DeepSeek R1 on three math benchmarks: AIME24 (80.3 vs. 79.8), AIME25 (74.4 vs. 70.0), and HMMT25 (50.4 vs. 41.7). This is a substantial improvement over its base model (6.7, 4.3, and 0.6, respectively). On LiveCodeBench V6, it scores 51.1, outperforming Magistral Medium's 50.3 and its base model's 0.0. These findings demonstrate that small models can achieve reasoning capabilities comparable to large models, drastically reducing training and inference costs and thereby democratizing advanced AI research.

---

# 小模型，大逻辑：多样性驱动优化激发 VibeThinker-1.5B 的大模型推理能力 论文详细解读

### 背景：这个问题为什么难？

在过去，业界普遍认为要让语言模型拥有可靠的数学或代码推理能力，唯一的办法就是把参数量砸得越来越大——从几百亿到上万亿不等。大模型的“算力优势”固然显著，但训练费用、部署成本和能耗都让普通研究团队望而却步。更糟的是，单纯增大规模并不能保证模型在细粒度推理上不出错，很多大模型仍然在高难度竞赛题目上表现平平。于是，如何在保持模型体积小、成本低的前提下，突破推理瓶颈，成了迫切需要解决的难题。

### 关键概念速览
- **Spectrum-to-Signal Principle（SSP）**：一种把“多样的解答光谱”转化为“高质量信号”的训练思路，类似把嘈杂的广播频道调到最清晰的频段。  
- **Two‑Stage Diversity‑Exploring Distillation（SFT）**：在微调阶段先让模型产生大量不同的答案（多样性），再挑选出在每个子领域表现最好的 checkpoint，像是先让学生自由发挥，再选出每门课的最佳学生。  
- **Pass@K**：衡量模型在 K 次尝试中至少一次得到正确答案的概率，和一次性答对的 Pass@1 不同，它更关注模型的“潜在能力”。  
- **Expert Model Fusion**：把多个子领域专家模型的参数按权重合并成一个统一模型，类似把几位老师的教学经验混合成一本教材。  
- **MaxEnt‑Guided Policy Optimization（RL）**：在强化学习阶段，用最大熵（MaxEnt）原则鼓励策略保持随机性，同时把注意力集中在“最模糊”的样本上，像是老师把课堂时间花在学生最容易出错的题目上。  
- **MGPO（Maximum‑Gain Policy Optimization）**：本文对 MaxEnt‑Guided 的实现细节的简称，核心是对不确定性接近 0.5 的样本给予更高的奖励。  

### 核心创新点
1. **从单一最强 checkpoint 到 Pass@K 多样选择**  
   - 以前的微调大多只保留在验证集上表现最好的那一个模型。  
   - VibeThinker 在每个数学子领域挑选 **Pass@K** 最高的多个 checkpoint，形成 N 个专家。  
   - 这样既避免了“早熟”导致的思路单一，又为后续融合提供了丰富的解答空间。  

2. **参数层面的专家融合而非单纯 ensemble**  
   - 传统做法是把多个模型的输出做投票或加权平均，计算开销大且难以部署。  
   - 本文直接在参数空间对专家模型做加权求和，得到一个 **1.5B** 的统一模型。  
   - 结果是推理速度与单模型相当，却保留了多专家的知识多样性。  

3. **最大熵驱动的“不确定性优先”强化学习**  
   - 常规 RL 只根据奖励大小更新策略，容易在已经掌握的样本上浪费算力。  
   - MGPO 先计算每个样本的预测不确定度（接近 0.5 表示模型既不对也不错），把这些样本的奖励放大。  
   - 训练过程像是老师把课堂时间集中在学生最容易出错的题目上，显著提升了数学和代码基准的得分。  

4. **极低的训练成本实现大模型级别的推理**  
   - 只用了约 **7,800 美元** 的算力（相当于几百块 GPU 时长），就把 1.5B 参数模型的数学成绩推到 400 倍更大的 DeepSeek R1 之上。  
   - 这证明了“规模不是唯一因素”，多样性与信息增益的优化同样关键。  

### 方法详解
**整体框架**：VibeThinker 的训练分为三大阶段——（1）多样性探索微调（SFT），（2）专家模型融合（Fusion），（3）基于最大熵的不确定性强化学习（RL）。每一步都围绕“让模型看到更多可能的解答，然后把正确的信号放大”展开。

#### 1. 两阶段多样性探索微调（SFT）
- **子领域划分**：先把数学知识库划分为代数、几何、微积分、统计等若干子域。每个子域会自动生成一个 **探测集**（题目 + 标准答案）。  
- **Pass@K 选模型**：在每个子域里，模型会进行多轮微调，每轮保存若干 checkpoint。对每个 checkpoint 计算在该子域探测集上的 Pass@K（K 通常取 5 或 10），选出表现最好的 checkpoint 作为该子域的 **专家模型**。这一步相当于让模型在每个子领域都“尝试多条路”，只保留最有潜力的那几条。  

#### 2. 参数层面的专家模型融合（Expert Model Fusion）
- **权重分配**：对每个子域专家模型，根据其在对应子域的 Pass@K 分数计算一个融合权重。  
- **参数加权求和**：把所有专家模型的参数按权重线性组合，得到一个统一的 **VibeThinker‑1.5B**。这一步不需要额外的推理开销，因为融合已经在参数层面完成。可以把它想象成把几位老师的讲义合并成一本教材，学生只需要读一本书就能学到所有老师的精华。  

#### 3. 最大熵引导的策略优化（MGPO / MaxEnt‑Guided RL）
- **不确定性评估**：在融合模型上跑一批未见过的样本，记录每道题的输出概率分布。若模型对某题的正确答案概率约为 0.5，则说明它“犹豫不决”。  
- **奖励重加权**：传统 RL 只根据是否答对给奖励，这里额外乘以一个 **不确定性因子**（不确定度越高奖励越大），鼓励模型在模糊样本上多尝试。  
- **最大熵正则**：在更新策略时加入最大熵项，保持策略的随机性，防止模型过早收敛到次优解。直观上，这相当于老师在课堂上既要纠正错误，又要让学生保持探索精神。  
- **迭代优化**：经过若干轮的 RL 更新后，模型的正确率在高难度数学基准上显著提升，尤其是在那些原本“模糊”但潜在可解的题目上。  

**最巧妙的地方**：整个流程把“多样性”与“信号放大”紧密耦合。先用 Pass@K 把潜在好解保留下来，再用最大熵把这些好解转化为模型的主导行为。相比单纯增大参数，这种信息增益的方式更像是“精炼思路”，而不是“盲目堆砌算力”。  

### 实验与效果
- **测试任务**：AIME24、AIME25、HMMT25（高难度数学竞赛题）以及 LiveCodeBench V6（代码生成）。  
- **对比基线**：DeepSeek R1（671B 参数）、Magistral Medium、Claude Opus 4（闭源商业模型），以及开源的 GPT‑OSS‑20B Medium。  
- **核心数字**：  
  - AIME24：VibeThinker‑1.5B 80.3 > DeepSeek R1 79.8 > Magistral Medium ≈ Claude Opus 4。  
  - AIME25：74.4 vs 70.0（DeepSeek R1）。  
  - HMMT25：50.4 vs 41.7（DeepSeek R1）。  
  - LiveCodeBench V6：51.1 vs 50.3（Magistral Medium），而基线 1.5B 的原始模型得分几乎为 0。  
- **训练成本**：总算力费用约 **7,800 美元**，相当于几百块 GPU 时长，远低于数十亿美元的大模型训练预算。  
- **消融实验**：原文提供了三组消融：去掉 Pass@K 选模型、改为普通 ensemble、以及不使用最大熵奖励。结果显示，去掉任意一环，最终得分均下降 5‑10% 以上，验证了每个模块的必要性。  
- **局限性**：实验主要聚焦在数学和代码两大领域，未在自然语言理解（如阅读理解、对话）上做大规模评估；此外，模型仍然依赖高质量的子领域探测集，构建成本在新领域可能较高。  

### 影响与延伸思考
VibeThinker 的成功让业界重新审视“规模=能力”的等式，激发了两大方向的后续研究：  
1. **多样性驱动的微调**——后续工作（如 Diversity‑Distill、Multi‑Expert Fusion）开始在更广泛的任务上采用 Pass@K 选模型的思路。  
2. **不确定性优先的强化学习**——类似的 MaxEnt‑RL 被搬到检索增强、对话安全等场景，帮助模型在“灰区”样本上快速提升。  

如果想进一步跟进，可以关注以下热点：  
- **自适应课程学习**（Curriculum Learning）与 **主动学习**（Active Learning）在大模型微调中的结合；  
- **参数融合的更高效实现**（如 LoRA‑style 融合、模型插值）；  
- **跨模态多样性探索**（把数学、代码、语言等子域统一到同一框架）。  

### 一句话记住它
**用多样性筛选 + 不确定性强化，让 1.5 B 参数的模型跑出 400 倍大模型的推理成绩。**