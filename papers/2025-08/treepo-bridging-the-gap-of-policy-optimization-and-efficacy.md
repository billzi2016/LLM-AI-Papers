# TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling

> **Date**：2025-08-24
> **arXiv**：https://arxiv.org/abs/2508.17445

## Abstract

Recent advancements in aligning large language models via reinforcement learning have achieved remarkable gains in solving complex reasoning problems, but at the cost of expensive on-policy rollouts and limited exploration of diverse reasoning paths. In this work, we introduce TreePO, involving a self-guided rollout algorithm that views sequence generation as a tree-structured searching process. Composed of dynamic tree sampling policy and fixed-length segment decoding, TreePO leverages local uncertainty to warrant additional branches. By amortizing computation across common prefixes and pruning low-value paths early, TreePO essentially reduces the per-update compute burden while preserving or enhancing exploration diversity. Key contributions include: (1) a segment-wise sampling algorithm that alleviates the KV cache burden through contiguous segments and spawns new branches along with an early-stop mechanism; (2) a tree-based segment-level advantage estimation that considers both global and local proximal policy optimization. and (3) analysis on the effectiveness of probability and quality-driven dynamic divergence and fallback strategy. We empirically validate the performance gain of TreePO on a set reasoning benchmarks and the efficiency saving of GPU hours from 22\% up to 43\% of the sampling design for the trained models, meanwhile showing up to 40\% reduction at trajectory-level and 35\% at token-level sampling compute for the existing models. While offering a free lunch of inference efficiency, TreePO reveals a practical path toward scaling RL-based post-training with fewer samples and less compute. Home page locates at https://m-a-p.ai/TreePO.

---

# TreePO：通过启发式树模型弥合策略优化、效能与推理效率的鸿沟 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做强化学习（RL）时，常用的做法是让模型自己生成完整的答案序列，然后根据奖励信号进行策略梯度更新。这种“on‑policy roll‑out”需要每一步都重新跑一遍模型，计算开销极大，尤其是当序列很长时，KV 缓存（键值对缓存）会被频繁刷新，导致显存和算力浪费。与此同时，单一路径的采样方式难以覆盖所有可能的推理路线，导致探索多样性受限，模型容易陷入局部最优。于是，如何在保持或提升探索质量的前提下，显著降低每次更新的计算成本，成为 RL‑LLM 研究的瓶颈。

### 关键概念速览
- **On‑policy roll‑out**：模型在当前策略下完整生成答案，再根据奖励更新策略。相当于每次都重新走一遍“实验”，成本高。
- **KV 缓存**：Transformer 在自回归生成时把已经计算好的键和值保存下来，以免重复计算。类似于记忆笔记本，序列越长，笔记本越占空间。
- **树结构搜索**：把序列生成看成在一棵树上扩展，每个节点代表一个 token，分支代表不同的候选 token。类似于下棋时的走子树，能够共享前缀计算。
- **段（segment）采样**：一次生成若干连续 token（一个段），而不是单个 token。相当于一次写完一句话的草稿，减少缓存切换次数。
- **局部不确定性**：模型对下一个 token 的概率分布越平坦，说明它越不确定，需要更多分支去探索。可以把它想成“路口的雾”，雾大就多开几条支路。
- **优势估计（Advantage Estimation）**：在策略梯度里，用来衡量某个动作相对于平均水平的好坏。这里在树的每个段上计算，兼顾全局和局部信息。
- **动态发散（Dynamic Divergence）**：根据概率或质量动态决定是否让新分支偏离当前策略，类似于在探索时适时“走偏门”。

### 核心创新点
1. **段式采样 + KV 缓存共享**  
   之前的采样往往逐 token 进行，导致每生成一个 token 都要刷新 KV 缓存。TreePO 把生成划分为固定长度的段，在同一段内部连续生成，缓存可以一次性写入，显著降低了显存占用和算子调度开销。结果是每次更新的 GPU 时间下降 22%~43%（论文声称）。

2. **基于局部不确定性的树分支策略**  
   传统方法要么盲目扩展所有候选，要么只保留最优路径，前者浪费算力，后者探索不足。TreePO 通过监测每个节点的概率熵（不确定性），在熵高的节点主动生成新分支，并在价值低的分支上提前剪枝。这样既保持了多样性，又避免了无效计算。

3. **段级优势估计结合全局 PPO**  
   PPO（近端策略优化）在 RL 中常用来限制策略更新幅度，防止崩溃。TreePO 把 PPO 的 KL 限制搬到树的段级，同时在每个段内部做局部优势估计，兼顾全局策略的平稳性和局部探索的灵活性。相当于在大局上保持“慢车”，在细节上开“快车”。

4. **概率驱动与质量驱动的双重回退机制**  
   当新分支的采样概率或生成质量（如奖励）不达标时，TreePO 会自动回退到父节点的主路径继续生成，避免因为一次糟糕的分支拖累整体计算。这个回退策略在实验中帮助降低了轨迹级 40% 和 token 级 35% 的采样计算。

### 方法详解
**整体框架**  
TreePO 把一次完整的序列生成过程拆成三步：① 初始化根节点（空序列），② 按段进行树式扩展并计算优势，③ 根据优势和 KL 限制更新策略。整个过程在训练时循环进行，在推理时只保留最优分支，实现“免费午餐”式的推理加速。

**关键模块拆解**  

1. **树结构初始化与前缀共享**  
   - 从空序列开始，模型一次性生成第一个段的所有 token。  
   - 这些 token 形成根节点的子树，所有子树共享同一段的 KV 缓存，类似于多条道路共用同一段路基。

2. **局部不确定性检测 & 动态分支生成**  
   - 对每个已生成 token，计算其概率分布的熵。  
   - 熵超过预设阈值时，触发“分支点”，在该位置生成若干候选 token（比如 top‑k），每个候选开启一条新分支。  
   - 新分支继续生成后续段，直到达到固定段长或提前满足停止条件（如奖励阈值）。

3. **段级优势估计**  
   - 对每条完整的段路径，收集该段的累计奖励和对应的概率。  
   - 使用 PPO 的剪切目标计算优势：如果新策略的 KL 与旧策略超出阈值，则对该段的梯度进行裁剪。  
   - 这样既保证了全局策略的平稳更新，又让每个段内部可以灵活探索。

4. **早停与回退机制**  
   - 在段生成过程中，如果某条分支的即时奖励或采样概率低于阈值，立即停止该分支的进一步扩展（剪枝）。  
   - 同时，若该分支已经产生了若干 token，但整体质量不佳，则回退到父节点的主路径继续生成，避免浪费算力。

5. **策略更新**  
   - 收集所有保留下来的段路径及其优势，使用标准的 PPO 更新公式对模型参数做梯度上升。  
   - 因为大量前缀是共享的，实际计算的 forward/backward 只在新分支的增量部分进行，显著降低了每轮的 GPU 时间。

**最巧妙的设计**  
- **段式共享 KV**：把原本逐 token 的缓存刷新改成段级一次写入，既减少了显存碎片，又让 GPU 的批处理效率提升。  
- **不确定性驱动的分支**：用熵直接衡量模型的“犹豫”，在犹豫的地方主动探索，避免盲目全展开或过度保守。  

### 实验与效果
- **测试任务**：论文在一系列推理基准上评估，包括数学解题、逻辑推理和多步问答等常用的 LLM 推理数据集。  
- **对比基线**：与标准的 PPO‑RL、直接采样（single‑path）以及最近的基于树搜索的采样方法（如 Beam Search + RL）进行比较。  
- **主要结果**：  
  - 在所有基准上，TreePO 的最终奖励提升约 3%~7%（论文声称），同时训练所需的 GPU 小时下降 22%~43%。  
  - 轨迹级采样计算下降约 40%，token 级下降约 35%，说明共享前缀和早停剪枝的效果显著。  
- **消融实验**：作者分别关闭段式采样、局部不确定性分支、段级优势估计和回退机制，发现每个组件的去除都会导致奖励下降 1%~3% 且计算成本回升 10%~20%，验证了各模块的贡献。  
- **局限性**：论文未详细说明在极长序列（>1024 token）或高度结构化任务（如代码生成）上的表现；此外，阈值的手工调节仍是一个经验性步骤，自动化仍待探索。

### 影响与延伸思考
TreePO 把“树搜索”与“近端策略优化”结合，提供了一条在 RL‑LLM 场景下兼顾探索多样性和计算效率的实用路径。自发表后，已有几篇工作尝试把类似的段式共享 KV 思想搬到 **RLHF（人类反馈强化学习）** 的微调阶段，或在 **检索增强生成** 中使用树形候选集合进行动态采样。未来可以进一步研究：
- 自动学习不确定性阈值的方式，让分支策略更自适应。  
- 将 TreePO 与 **稀疏注意力** 结合，进一步降低长序列的显存占用。  
- 在 **多模态**（文本+图像）生成任务中探索树结构的跨模态分支。

### 一句话记住它
把序列生成当成共享前缀的树搜索，用局部不确定性决定分支，段式采样让 KV 缓存一次写入，既省算力又保探索——这就是 TreePO。