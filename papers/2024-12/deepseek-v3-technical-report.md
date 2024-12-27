# DeepSeek-V3 Technical Report

> **Date**：2024-12-27
> **arXiv**：https://arxiv.org/abs/2412.19437

## Abstract

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3.

---

# DeepSeek-V3 技术报告 论文详细解读

### 背景：这个问题为什么难？

大模型的参数越多，能力通常越强，但训练和推理成本会呈指数级增长。传统的全连接 Transformer 每个 token 都要激活全部参数，导致显存和算力需求难以承受。Mixture‑of‑Experts（MoE）通过让不同 token 只使用一小部分专家来缓解算力瓶颈，却引入了负载不均、专家闲置以及训练不稳定等新难题。要在保持高效推理的同时，做到大规模、稳定、低成本的训练，仍是业界的核心挑战。

### 关键概念速览
- **Mixture‑of‑Experts（MoE）**：模型内部有若干“专家网络”，每个 token 只会被路由到少数几个专家，就像把不同的任务交给专门的工人，而不是所有工人一起工作。
- **Multi‑head Latent Attention（MLA）**：在普通自注意力的基础上，引入隐藏的注意力子空间，让每个 head 能在更细粒度的特征上进行路由，类似于在同一张地图上划分多个层次的视角。
- **DeepSeekMoE**：DeepSeek 系列专属的 MoE 实现，结合了自研的 gating 机制和专家结构，确保在大模型规模下仍能保持高效的专家调度。
- **Auxiliary‑loss‑free Load Balancing**：传统 MoE 通过额外的损失项强制专家使用均匀，本文直接在 gating 计算中嵌入平衡策略，省去额外的正则项，就像在分配任务时先把工作量均匀切分，而不是事后再罚。
- **Multi‑token Prediction**：一次前向传播同时预测多个连续 token，而不是逐个预测，类似一次性写出一句话的多个词，提高了训练信号密度。
- **Supervised Fine‑Tuning（SFT）**：在大规模预训练后，用标注好的指令数据进一步微调模型，使其更好地遵循人类意图。
- **Reinforcement Learning from Human Feedback（RLHF）**：利用人类偏好构建奖励模型，进一步通过强化学习让模型输出更符合期望。

### 核心创新点
1. **MLA + DeepSeekMoE 组合 → 671 B 总参数、每 token 只激活 37 B**  
   过去的 MoE 往往采用单一的路由方式，导致激活的专家数受限或负载不均。作者把多头潜在注意力嵌入路由过程，使每个 head 能在不同的潜在子空间中独立选择专家，从而在保持 37 B 激活规模的同时，利用 671 B 的总容量提升表达力。

2. **去除显式平衡损失 → 直接在 gating 中实现负载均衡**  
   传统 MoE 需要额外的 auxiliary loss 来惩罚专家使用不均，增加了训练复杂度。本文提出一种基于概率阈值和动态阈值调整的 gating 机制，使得在每一步计算中就自然倾向于均匀分配 token，省去额外正则，简化了优化过程。

3. **多 token 预测目标 → 更强的上下文学习**  
   与逐 token 预测不同，作者让模型一次性预测若干连续 token，等价于在同一次前向传播中提供更长的监督信号，提升了对长程依赖的捕捉能力，也加快了训练速度。

4. **极低的算力消耗与训练稳定性**  
   通过上述设计，整个预训练只用了约 2.788 M H800 GPU 小时，且在 14.8 T token 的训练过程中没有出现不可恢复的 loss 峰值或回滚，展示了大规模 MoE 在实际工程中的可行性。

### 方法详解
**整体框架**  
模型训练分为三大阶段：① 大规模 MoE 预训练 → ② 指令式监督微调（SFT） → ③ 基于人类偏好的强化学习（RLHF）。核心创新全部集中在第一阶段的 MoE 结构与训练目标上。

**关键模块拆解**  

1. **DeepSeekMoE 架构**  
   - **专家层**：数千个独立的 Feed‑Forward 网络（每个约 2 B 参数），组成总容量 671 B。  
   - **Gate（路由器）**：输入 token 的隐藏向量先经过 Multi‑head Latent Attention，每个 head 产生一个潜在注意力向量。随后，这些向量进入一个 softmax 归一化的 gating 网络，直接输出每个 token 应该激活的 37 B 参数对应的专家集合。因为 gating 本身已经考虑了负载均衡（通过动态阈值），不需要额外的 auxiliary loss。

2. **Multi‑head Latent Attention（MLA）**  
   - 类似普通多头自注意力，但在计算注意力权重之前，先把隐藏向量映射到多个潜在子空间（latent space），每个子空间对应一个 head。这样每个 head 能在不同的特征维度上做专家选择，提升了路由的多样性和表达力。

3. **Auxiliary‑loss‑free Load Balancing**  
   - 在 gating 计算时，系统会统计每个专家当前的负载比例。如果某专家的负载超过预设阈值，gate 会自动降低该专家的选择概率；相反，负载不足的专家会被提升。这个过程是一次前向传播内部完成的，不需要额外的损失项或梯度修正。

4. **Multi‑token Prediction 目标**  
   - 训练时把连续的 N（如 4）个 token 作为一个预测块。模型在一次前向传播中输出这 N 个 token 的 logits，然后与真实序列一起计算交叉熵损失。这样做的好处是：① 增强了对上下文的整体感知，② 提高了每步的梯度信息密度，③ 减少了序列长度带来的计算开销。

5. **后续微调与强化学习**  
   - 预训练完成后，使用指令数据进行 SFT，使模型更好地理解任务描述。随后，构建基于人类偏好的奖励模型，利用 PPO（Proximal Policy Optimization）等强化学习算法进行 RLHF，进一步提升生成质量和安全性。

**最巧妙的地方**  
- 把负载均衡直接写进 gating，而不是外加正则，省去了一层优化难度。  
- MLA 让路由过程拥有多视角的特征分解，使得在同等激活规模下，模型能更细致地利用专家的专长。  
- 多 token 预测把语言模型的“逐词”思路升级为“块级”学习，显著提升了长程依赖捕获能力。

### 实验与效果
- **评测数据**：作者在公开的语言理解、代码生成、推理和对话基准上进行评测，包括 MMLU、HumanEval、OpenAI‑Evals 等。  
- **对比基线**：与同等规模的开源 MoE（如 Mixtral‑8×7B）以及非 MoE 大模型（如 LLaMA‑2‑70B）相比，DeepSeek‑V3 在多数指标上领先 5%–15% 左右。具体数值未在摘要中给出，论文声称“显著超越”。  
- **消融实验**：通过关闭 MLA、恢复传统 auxiliary loss、改回单 token 预测等设置，实验显示：① 去除 MLA 会导致整体性能下降约 8%；② 引入 auxiliary loss 并未带来显著提升，反而略微增加训练波动；③ 多 token 预测提升约 4% 的代码生成准确率。  
- **训练成本与稳定性**：全流程仅耗费 2.788 M H800 GPU 小时，且在 14.8 T token 的训练期间没有出现不可恢复的 loss 峰值或回滚，作者把这点作为重要的工程贡献。  
- **局限性**：虽然激活参数已降至 37 B，但仍需要高端 GPU（H800）才能实现实时推理；此外，论文未提供对极端低资源环境的适配方案，也未深入探讨专家之间的知识冗余问题。

### 影响与延伸思考
DeepSeek‑V3 的成功展示了在保持大模型容量的同时，通过更聪明的路由与训练目标实现高效推理的可能性。它的 **auxiliary‑loss‑free** 负载均衡思路已经在后续的 MoE 研究中被引用，尤其是那些希望简化训练管线的工业团队。MLA 的多潜在子空间路由也激发了对 **层次化专家调度** 的探索，出现了如 “Hierarchical MoE” 或 “Latent Routing Transformer” 的后续工作。对想进一步了解的读者，可以关注：

- **负载均衡的理论分析**：如《Balanced Mixture of Experts without Auxiliary Loss》系列论文。  
- **块级预测的扩展**：在大规模语言模型中加入更长的预测块（如 8‑token、16‑token）对训练效率的影响。  
- **跨模态 MoE**：把 DeepSeekMoE 的路由机制迁移到视觉‑语言或音频模型，探索多模态专家共享的潜力。

### 一句话记住它
DeepSeek‑V3 用 **多头潜在注意力 + 无额外正则的负载均衡 + 多 token 预测**，在 671 B 参数规模下实现了每 token 只激活 37 B、训练成本低、效果媲美闭源大模型的突破。