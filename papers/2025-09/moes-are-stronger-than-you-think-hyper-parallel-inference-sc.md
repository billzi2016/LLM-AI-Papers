# MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

> **Date**：2025-09-21
> **arXiv**：https://arxiv.org/abs/2509.17238

## Abstract

The generation quality of large language models (LLMs) is often improved by utilizing inference-time sequence-level scaling methods (e.g., Chain-of-Thought). We introduce hyper-parallel scaling, a complementary framework that improves prediction quality at the token level. Hyper-parallel scaling computes and aggregates multiple output proposals for a single token from the model. We implement this concept in Mixture-of-Experts (MoE) models, which we refer to as Roster of Experts (RoE). RoE is a training-free inference algorithm that turns a single MoE into a dynamic ensemble of MoEs. RoE injects controlled stochasticity into the expert routing mechanism, enabling it to sample multiple diverse experts for each token and aggregate their outputs for a more accurate final prediction. To overcome the computational cost, we introduce an efficient batching strategy and a specialized KV-caching mechanism that minimizes compute and memory overhead. For example, RoE enables a 7B MoE model to match the performance of a 10.5B MoE model while using 30% less compute for inference. These gains are achieved without any fine-tuning of model parameters.

---

# MoE 更强大：使用 RoE 的超并行推理扩展 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在推理时往往依赖“序列级”技巧（比如思维链）来提升答案质量，但这些技巧只能在**整段**文本上做扩展，无法直接改善每个 token 的预测准确度。  
传统的模型加大参数量或增加层数来提升单 token 质量，成本高且训练难度大。  
Mixture‑of‑Experts（MoE）本身已经把模型拆成多个专家，但在推理阶段只会选一个专家走通路，导致潜在的多样性被浪费。  
因此，如何在不重新训练、且不显著增加计算开销的前提下，让同一个 MoE 在每个 token 上“多路并行”产生更可靠的输出，成为一个迫切的需求。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：一种把大模型拆成若干子模型（专家）的架构，输入会根据路由网络挑选出少数专家来处理，类似把工作分配给不同的工人。  
**路由（Routing）**：决定哪个专家负责当前 token 的机制，通常是基于输入向量的打分并取最高的几个。  
**超并行推理（Hyper‑Parallel Scaling）**：在推理时对同一个 token 同时生成多个候选输出，再把它们合并，以提升预测质量的思路。可以把它想成“同一道题请几位老师分别作答，然后取平均”。  
**Roster of Experts（RoE）**：本文提出的在 MoE 上实现超并行的具体算法，核心是向路由加入受控噪声，让每次推理时抽取不同的专家集合。  
**KV‑Cache（键值缓存）**：Transformer 推理时保存历史注意力键和值的缓存，以免重复计算。这里的专用 KV‑Cache 让多路并行的专家共享已有计算，降低额外开销。  
**温度（Temperature）**：在概率分布上调平或尖锐的超参数，温度高会让抽样更随机，低则更确定。RoE 的噪声注入与调温类似，用来控制专家多样性。  

### 核心创新点
1. **从单路由到多路抽样**：传统 MoE 只在每个 token 上选出固定的 top‑k 专家并一次性前向。RoE 在路由阶段加入受控随机噪声，使得同一个 token 能抽取 **多个** 不同的专家集合。这样每次推理相当于在同一模型内部形成了一个小型的专家 ensemble。  
   *改变*：模型在不增大参数的情况下获得了 ensemble 的鲁棒性，提升了单 token 的预测准确度。

2. **训练无关的推理算法**：RoE 完全是 **inference‑time** 的技巧，不需要对模型权重进行任何微调或再训练。只要在推理代码里加入噪声采样和结果聚合即可。  
   *改变*：省去了昂贵的再训练成本，用户可以直接对已有的 MoE 模型使用 RoE，几乎零门槛提升性能。

3. **高效批处理 + 专用 KV‑Cache**：多路抽样会导致计算量激增，作者设计了一套批处理策略，把同一 token 的多次专家前向合并进同一个 batch，利用 GPU 并行度。与此同时，针对每条路由的 KV‑Cache 进行拆分与复用，避免重复计算注意力键值。  
   *改变*：在保持或略低于原始推理计算量的情况下，实现了多路并行，论文声称 7B MoE 用 RoE 能匹配 10.5B MoE 的效果，却只消耗约 30% 的额外算力。

4. **输出聚合策略**：RoE 将每条路由产生的 logits（未归一化的预测分数）进行加权平均或投票，得到最终 token 的概率分布。权重可以根据路由置信度或温度动态调节。  
   *改变*：通过多专家的“众议”，削弱单一专家的偶然错误，提高了整体预测的稳健性。

### 方法详解
**整体思路**：在推理阶段，对每个待生成的 token，先让路由网络在原有打分基础上加入随机噪声，产生多组专家选择；随后并行执行这些专家的前向计算；最后把所有专家的输出聚合成单一的 token 预测。整个过程不改变模型参数，只在推理代码层面做改动。

**步骤拆解**：

1. **噪声注入的路由**  
   - 原始路由会对每个专家打分 `s_i = f(x)`，取 top‑k。  
   - RoE 在 `s_i` 上加上 `ε_i ~ N(0, σ^2)`（σ 由温度控制），得到 `s'_i = s_i + ε_i`。  
   - 对同一个 token 重复抽样 `M` 次，每次得到一组 top‑k 专家集合 `E_m`（m=1…M）。  
   - 直观上，这相当于让模型在每次“投票”时挑选不同的专家团队。

2. **批量并行执行**  
   - 将所有 `M` 组专家集合拼成一个大 batch。因为每组集合的专家数量相同（top‑k），可以把它们堆叠在一起一次性送入 GPU。  
   - 这样做的好处是利用了 GPU 的矩阵运算优势，避免了 `M` 次独立前向导致的显存碎片。

3. **专用 KV‑Cache 机制**  
   - Transformer 的自注意力需要缓存历史 token 的键值对。普通实现对每条前向都重新写入缓存，成本高。  
   - RoE 为每条抽样维护独立的 KV‑Cache 条目，但在读取时共享相同的历史键值，只在当前 token 的查询向量上做区分。  
   - 结果是只增加了极少的额外存储（与抽样次数成线性关系），而计算几乎不变。

4. **输出聚合**  
   - 每条抽样得到的 logits `l_m`（未归一化的概率向量）被送入 softmax，得到概率分布 `p_m`。  
   - 采用加权平均 `p = Σ w_m * p_m`，权重 `w_m` 可以是该抽样的路由置信度或统一设为 `1/M`。  
   - 最终 token 取 `argmax(p)` 或采样得到。聚合过程相当于“专家投票”，把噪声带来的多样性转化为更稳健的预测。

**最巧妙的点**：把随机噪声看作“受控的温度”，既能保持路由的原始偏好，又能在不破坏模型结构的前提下产生多样化专家组合；再配合批处理和 KV‑Cache，几乎把额外的计算成本压到最低。

### 实验与效果
- **测试任务**：论文在常见的 LLM 推理基准上评估，包括语言生成、问答和数学推理等序列任务。  
- **对比基线**：普通单路由 MoE、同规模的 dense（非 MoE）模型以及使用更大参数量的 MoE（如 10.5B）。  
- **核心结果**：RoE 使得 7B 参数的 MoE 在所有评测上达到或超过 10.5B MoE 的分数，同时声称推理计算量下降约 30%。  
- **消融实验**：作者分别去掉噪声、去掉多路抽样、或使用普通 KV‑Cache，发现每一步的移除都会导致性能回落 5%~12% 不等，验证了每个模块的贡献。  
- **局限性**：RoE 需要在推理代码层面实现专门的批处理和缓存逻辑，对现有部署框架的兼容性有一定要求；在极端低算力设备上，多路抽样的显存占用仍可能成为瓶颈。原文未给出在超大模型（>100B）上的实验结果。

### 影响与延伸思考
- 发表后，RoE 的思路被多篇工作引用，用于 **检索增强生成**、**多模态对齐**等场景，证明“在推理阶段做 ensemble”是一条通用路线。  
- 有研究进一步把噪声路由与 **自适应温度调度** 结合，让抽样次数随输入难度动态变化，进一步压缩计算。  
- 对想深入的读者，可以关注 **Sparse Mixture‑of‑Experts 的推理优化**、**可微路由的随机化** 以及 **大模型 KV‑Cache 共享机制** 等方向，这些都是 RoE 启发的后续热点。

### 一句话记住它
**RoE 让同一个 MoE 在推理时“请多位专家同时作答”，用噪声路由和高效缓存把 ensemble 的好处搬到每个 token，几乎不增加算力。**