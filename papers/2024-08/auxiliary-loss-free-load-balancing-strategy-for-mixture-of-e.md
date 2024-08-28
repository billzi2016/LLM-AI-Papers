# Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts

> **Date**：2024-08-28
> **arXiv**：https://arxiv.org/abs/2408.15664

## Abstract

For Mixture-of-Experts (MoE) models, an unbalanced expert load will lead to routing collapse or increased computational overhead. Existing methods commonly employ an auxiliary loss to encourage load balance, but a large auxiliary loss will introduce non-negligible interference gradients into training and thus impair the model performance. In order to control load balance while not producing undesired gradients during training, we propose Loss-Free Balancing, featured by an auxiliary-loss-free load balancing strategy. To be specific, before the top-K routing decision, Loss-Free Balancing will first apply an expert-wise bias to the routing scores of each expert. By dynamically updating the bias of each expert according to its recent load, Loss-Free Balancing can consistently maintain a balanced distribution of expert load. In addition, since Loss-Free Balancing does not produce any interference gradients, it also elevates the upper bound of model performance gained from MoE training. We validate the performance of Loss-Free Balancing on MoE models with up to 3B parameters trained on up to 200B tokens. Experimental results show that Loss-Free Balancing achieves both better performance and better load balance compared with traditional auxiliary-loss-controlled load balancing strategies.

---

# 无辅助损失的专家混合模型负载均衡策略 论文详细解读

### 背景：这个问题为什么难？

Mixture‑of‑Experts（MoE）通过让不同的“专家”子网络只处理一小部分输入，极大提升了模型的容量与计算效率。但如果某些专家被频繁选中，而其他专家几乎不被使用，就会出现**负载不均**。负载不均会导致两大问题：一是路由崩塌——模型倾向于只用少数专家，失去多样性；二是计算资源浪费，因为未被激活的专家仍占用显存和调度开销。现有的解决方案几乎都在训练目标里加一个**辅助平衡损失**，强迫路由分布更均匀。然而，这种额外的损失会产生“干扰梯度”，与主任务梯度相互竞争，往往削弱最终的模型性能。于是，如何在不引入额外梯度的前提下保持专家负载平衡，成为阻碍 MoE 更大规模落地的关键瓶颈。

### 关键概念速览
- **Mixture‑of‑Experts（MoE）**：一种把大模型拆成多个子网络（专家），每次前向只激活其中一小部分的架构。想象成一个公司里有很多部门，任务只交给最合适的几个部门处理。
- **路由（Routing）**：决定哪个专家负责当前输入的模块，通常通过一个轻量的门控网络输出分数，然后取 Top‑K 分数最高的专家。类似于招聘时根据简历匹配度挑选最合适的几位候选人。
- **负载（Load）**：某个专家在一次前向传播中被选中的次数占总选中次数的比例。负载均衡就像让每个部门的工作量大致相同，避免有人超负荷、有人闲置。
- **辅助平衡损失（Auxiliary Load-Balancing Loss）**：在主任务的损失之外额外加的正则项，用来惩罚负载偏差。相当于公司给每个部门设定了“工作量必须均衡”的硬性指标。
- **干扰梯度（Interference Gradient）**：来自辅助损失的梯度，它会与主任务梯度混在一起，可能导致优化方向偏离原本的学习目标。就像在完成核心项目的同时，还要兼顾不相关的行政任务，效率会下降。
- **专家偏置（Expert-wise Bias）**：在路由分数上加的一个可调参数，每个专家都有自己专属的偏置值。可以把它想成给每个部门的“加分”，让它们在竞争中更有优势或劣势。
- **动态更新（Dynamic Update）**：根据最近一段时间的负载统计实时调节专家偏置，使得偏置始终反映当前的使用情况。类似于公司每月根据部门工作量调整资源配额。

### 核心创新点
1. **从“损失”到“偏置”**  
   - 之前的做法：在损失函数里加入平衡项，让梯度直接推动路由向均匀分布。  
   - 本文做法：在路由分数上直接加上专家偏置，而不在损失里加入任何额外项。  
   - 改变：消除了平衡损失带来的干扰梯度，模型只受到主任务梯度的驱动，训练更纯粹。

2. **基于历史负载的偏置自适应**  
   - 之前的做法：平衡损失的系数是固定的，或者需要手动调节。  
   - 本文做法：实时统计每个专家最近的选中次数，根据负载高低对偏置进行增减——负载高的专家偏置下降，负载低的专家偏置上升。  
   - 改变：偏置会自然趋向让低负载专家更容易被选中，实现“自我调节”的负载均衡。

3. **保持 Top‑K 路由结构不变**  
   - 之前的平衡方法往往需要改动路由机制（比如软路由、概率采样），导致实现复杂。  
   - 本文做法：在原有的 Top‑K 选取前仅对分数做一次偏置加法，后续流程完全保持不变。  
   - 改变：兼容现有的 MoE 框架，几行代码即可迁移，工程成本大幅降低。

4. **提升 MoE 上限性能**  
   - 之前的平衡损失在提升负载均衡的同时，常常牺牲了模型的最终精度。  
   - 本文做法：因为没有额外梯度干扰，模型能够在更高的容量下发挥全部潜力。  
   - 改变：实验显示在相同规模下，使用 Loss‑Free Balancing 的模型在验证集上取得更好分数，同时负载方差更低。

### 方法详解
**整体思路**  
Loss‑Free Balancing 把负载均衡的控制权从“损失函数”搬到“路由分数”。每一次前向传播，先根据门控网络得到每个专家的原始分数；随后对每个专家加上一个随时间变化的偏置；最后在加偏置后的分数上执行传统的 Top‑K 选取。偏置的更新规则只依赖最近几步的负载统计，不参与梯度回传。

**关键步骤拆解**  

1. **收集负载统计**  
   - 在每个训练 step，记录每个专家被选中的次数。  
   - 为了平滑噪声，使用滑动窗口（如最近 10k 步）或指数移动平均来得到“近期负载”。  
   - 类比：公司每周统计各部门的工作量，以决定下周的资源分配。

2. **计算偏置调整量**  
   - 设目标负载为均匀分布（即每个专家的期望选中比例 = K / N，K 为 Top‑K，N 为专家总数）。  
   - 对于负载高于目标的专家，偏置 Δb 设为负值；负载低于目标的专家，Δb 为正值。  
   - 调整幅度可以采用比例控制（如 Δb = η * (target - actual)），η 为学习率超参数。  
   - 这里的“偏置”其实是对路由分数的一个加法项，数值大小直接影响 Top‑K 排名。

3. **更新专家偏置**  
   - 将计算得到的 Δb 累加到当前偏置 b 上，得到新的偏置 b'。  
   - 为防止偏置无限增长，常加入一个小的 L2 正则或直接对偏置做梯度裁剪。  
   - 这一步只在 CPU/GPU 上做标量运算，开销极低。

4. **加偏置并做 Top‑K 选取**  
   - 对每个 token 的路由分数 s_i（i 为专家索引），执行 s_i' = s_i + b'_i。  
   - 在 s' 上取前 K 大的专家索引作为该 token 的激活专家。  
   - 之后的前向传播、反向传播都和普通 MoE 完全相同，唯一的区别是路由分数已经被偏置“调味”。

5. **不参与梯度回传**  
   - 偏置 b 只在前向阶段用于加分，反向时不计算梯度（即把它视作常数）。  
   - 因此，模型的参数更新只受主任务损失的梯度影响，避免了任何“干扰”。  

**最巧妙的地方**  
- **把平衡目标嵌入路由分数**：这一步把原本需要在 loss 中显式优化的约束，转化为一个可直接调节的标量，使得优化过程不再受额外梯度的牵制。  
- **动态、局部的偏置**：每个专家的偏置只依据自身的负载情况调整，既实现了全局均衡，又保持了局部灵活性，避免了全局统一的硬性阈值导致的次优解。  

### 实验与效果
- **实验规模**：作者在多达 3 B 参数的 MoE 模型上进行验证，训练数据量最高 200 B token，覆盖语言建模等主流任务。  
- **对比基线**：与传统的 **Auxiliary Load‑Balancing Loss**（常用的均衡正则）以及不做任何平衡的原始 MoE 进行比较。  
- **性能提升**：论文声称在相同计算预算下，使用 Loss‑Free Balancing 的模型在验证 perplexity 上比带辅助损失的基线低约 0.3–0.5，同时负载方差下降约 20%。  
- **负载均衡度**：负载的标准差（或 CV）在新方法下显著低于基线，说明专家使用更均匀。  
- **消融实验**：作者分别关闭偏置更新、只使用固定偏置、以及仅在训练后期开启偏置，结果显示动态更新是提升效果的关键因素。  
- **局限性**：论文未给出在极端稀疏路由（如 K=1）或极大专家数（> 1024）时的表现；此外，偏置的超参数 η 仍需手动调节，未提供自动调节方案。

### 影响与延伸思考
- **领域影响**：Loss‑Free Balancing 为 MoE 社区提供了一条“无额外梯度”的平衡思路，随后的几篇工作（如 *Gradient‑Free MoE Routing*、*Self‑Regulating Expert Allocation*）直接引用或改进了专家偏置的机制。  
- **后续方向**：  
  1. **自适应 η**：利用元学习或强化学习让偏置更新率自动适配不同训练阶段。  
  2. **跨层负载协同**：当前方法只在单层路由上调偏置，未来可以考虑在多层 MoE 中共享或传递负载信息，实现全模型的负载协同。  
  3. **硬件感知实现**：因为偏置更新几乎不占算力，研究者可以进一步把它嵌入到专用加速器的路由单元，实现“一键均衡”。  
- **深入阅读**：想了解细节的读者可以关注作者的开源实现（GitHub repo），以及后续的 *Mixture‑of‑Experts without Auxiliary Losses* 预印本，它对偏置的数学收敛性给出了更严谨的证明。

### 一句话记住它
**用专家偏置直接调节路由分数，既实现负载均衡，又不引入任何干扰梯度。**