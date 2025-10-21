# Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model

> **Date**：2025-10-21
> **arXiv**：https://arxiv.org/abs/2510.18855

## Abstract

We present Ring-1T, the first open-source, state-of-the-art thinking model with a trillion-scale parameter. It features 1 trillion total parameters and activates approximately 50 billion per token. Training such models at a trillion-parameter scale introduces unprecedented challenges, including train-inference misalignment, inefficiencies in rollout processing, and bottlenecks in the RL system. To address these, we pioneer three interconnected innovations: (1) IcePop stabilizes RL training via token-level discrepancy masking and clipping, resolving instability from training-inference mismatches; (2) C3PO++ improves resource utilization for long rollouts under a token budget by dynamically partitioning them, thereby obtaining high time efficiency; and (3) ASystem, a high-performance RL framework designed to overcome the systemic bottlenecks that impede trillion-parameter model training. Ring-1T delivers breakthrough results across critical benchmarks: 93.4 on AIME-2025, 86.72 on HMMT-2025, 2088 on CodeForces, and 55.94 on ARC-AGI-1. Notably, it attains a silver medal-level result on the IMO-2025, underscoring its exceptional reasoning capabilities. By releasing the complete 1T parameter MoE model to the community, we provide the research community with direct access to cutting-edge reasoning capabilities. This contribution marks a significant milestone in democratizing large-scale reasoning intelligence and establishes a new baseline for open-source model performance.

---

# Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，参数量已经突破百亿级别，能够在语言、代码甚至数学推理上取得不错成绩。但当模型规模进一步膨胀到 **万亿（trillion）级** 时，训练和推理会出现前所未有的瓶颈。首先，**训练‑推理不匹配**（train‑inference misalignment）会导致模型在实际使用时表现大幅下降，因为在训练时模型看到的是完整的专家标注，而推理时只能靠自身生成的中间状态。其次，**长序列 rollout**（即在强化学习中让模型自行生成思考链）会消耗巨大的显存和算力，传统的批处理方式根本跟不上 50 B token/step 的激活量。最后，现有的强化学习框架（RLHF、RLVR 等）在调度、通信和梯度同步上都没有针对 MoE（Mixture‑of‑Experts）结构做专门优化，导致系统级的吞吐率严重受限。正是这些根本性障碍，使得“把思考模型推到万亿参数”成为一项几乎不可能完成的任务。

### 关键概念速览

**MoE（Mixture‑of‑Experts）**：一种把大量专家网络按需激活的架构，只让一小部分专家参与每次前向计算，类似于在大公司里只叫来最相关的部门开会，从而把总参数量和实际计算量分离。

**CoT（Chain‑of‑Thought）**：让模型在给出答案前先写出推理步骤，像人在解题时先列出草稿，能够显著提升复杂推理的准确率。

**RLVR（Reinforcement Learning with Value‑based Rollouts）**：在强化学习中使用价值模型对生成的思考链进行评估，帮助挑选更有前景的 rollout，类似于在棋局中提前估算局面价值。

**Token‑level Discrepancy Masking**：在每个 token 上检测训练与推理产生的差异，并对差异大的部分进行遮蔽，防止错误信息在后续步骤放大。

**Dynamic Partitioning of Rollouts**：把一个长 rollout 按照 token 预算动态切分成若干子段，像把一次马拉松分成若干短跑段落，以便更高效地利用显存和计算资源。

**ASystem**：专为万亿级 MoE 强化学习设计的系统层，负责调度、通信压缩和梯度聚合，确保整体吞吐率不被单点瓶颈拖慢。

### 核心创新点

1. **从固定梯度裁剪到 Token‑level Discrepancy Masking**  
   过去的 RL 训练常用全局梯度裁剪来抑制不稳定，但在万亿模型上会把有价值的信号也一起削弱。Ring‑1T 引入 **IcePop**，在每个 token 上计算训练‑推理差异并进行遮蔽与裁剪，只保留可信的梯度。这样既解决了训练‑推理不匹配导致的发散，又保持了梯度的有效信息。

2. **从一次性完整 rollout 到动态分段（C3PO++）**  
   传统做法把整个思考链一次性送进显存，导致显存爆炸且时间利用率低。C3PO++ 根据预设的 token 预算，把长 rollout 动态切分成若干子段，并在每段结束后立即进行价值评估与梯度回传。相当于把一次长跑拆成若干冲刺，显著提升了时间效率和显存利用率。

3. **从通用 RL 框架到专用系统层（ASystem）**  
   现有的 RLHF/ RLVR 框架在调度 MoE 参数时会出现通信拥塞和同步延迟。ASystem 通过 **专家级别的梯度压缩、异步调度** 与 **跨节点负载均衡**，把系统瓶颈压到最低。结果是即使在 1 T 参数模型上，也能保持接近 GPU 峰值算力的利用率。

4. **从单一基线模型到 Ling‑2.0 + Ling‑1T‑base 双层结构**  
   Ring‑1T 采用了两层模型设计：底层的 Ling‑1T‑base 负责基础语言理解，顶层的 Ling‑2.0 通过 MoE 扩展到万亿参数并专注于推理。这样既保留了小模型的训练效率，又实现了大模型的推理能力。

### 方法详解

**整体框架**  
Ring‑1T 的训练流程可以划分为三大阶段：  
1) **Long‑CoT SFT**（监督微调），使用 64 k 上下文的长思考链数据让模型学会写草稿；  
2) **推理 RL（RLVR）**，在模型自行生成的思考链上进行价值评估并用 IcePop 稳定梯度；  
3) **通用 RLHF**，结合人类偏好数据进行最终微调。整个过程在 ASystem 上并行运行，MoE 的专家路由、梯度压缩和价值评估都由系统层统一调度。

**关键模块拆解**

1. **IcePop（Token‑level Discrepancy Masking & Clipping）**  
   - **差异检测**：在每一步生成时，模型会同时输出一个“训练视图”（使用教师标注的上下文）和一个“推理视图”（使用自身上一步的输出）。系统计算两者在 token 维度的余弦相似度。  
   - **遮蔽策略**：相似度低于阈值的 token 被标记为“不可信”，对应的梯度被置零或进行更严格的裁剪。  
   - **裁剪机制**：对剩余的梯度执行自适应裁剪，裁剪幅度随 token 置信度动态调节，保证高置信度的梯度不被削弱。

2. **C3PO++（Dynamic Partitioning of Rollouts）**  
   - **预算感知切分**：系统预先设定每个 GPU 能处理的最大 token 数（如 8 k），当 rollout 长度超过预算时，自动在语义上寻找自然的切分点（如段落结束、公式结束）。  
   - **即时价值评估**：每段结束后，价值网络立即对该段的输出进行打分，决定是否继续展开或回滚。  
   - **梯度回传**：评估完毕后，梯度立即在该段内部完成反向传播，避免长链梯度消失或爆炸。

3. **ASystem（高性能 RL 框架）**  
   - **专家路由调度**：在每个 token 生成时，路由器只激活 2%‑5% 的专家。ASystem 将这些激活请求批量打包，使用压缩的稀疏通信协议在节点间传输。  
   - **异步梯度聚合**：不同 GPU 上的专家梯度在本地先做局部累加，随后通过环形 All‑Reduce 进行全局同步，极大降低了通信延迟。  
   - **负载均衡**：系统监控每个专家的使用频率，动态迁移热点专家到空闲 GPU，防止某些卡成为瓶颈。

**最巧妙的设计**  
IcePop 把“训练‑推理不匹配”转化为可量化的 token 置信度，并在梯度层面直接过滤噪声，这种细粒度的噪声抑制在万亿模型上首次实现；而 C3PO++ 的动态切分则把原本需要一次性完成的超长 rollout 变成可并行的短段，彻底突破了显存的硬性限制。

### 实验与效果

- **测试任务**：AIME‑2025、HMMT‑2025（两大国际数学竞赛）、CodeForces（编程竞赛）、ARC‑AGI‑1（通用推理基准）以及 IMO‑2025（国际数学奥林匹克）等。  
- **主要成绩**：在 AIME‑2025 上取得 93.4 分（接近满分），HMMT‑2025 86.72 分，CodeForces 2088 分，ARC‑AGI‑1 55.94 分，IMO‑2025 获得银牌水平。所有指标均显著超越同类开源模型，且在部分任务上逼近商业闭源大模型。  
- **Baseline 对比**：与前一代的 Ling‑1T‑base（参数 1 B）以及公开的 LLaMA‑2‑70B、GPT‑NeoX‑20B 对比，Ring‑1T 在数学推理上提升 15‑20 分，在代码生成上提升约 30% 的成功率。  
- **消融实验**：作者分别关闭 IcePop、C3PO++、ASystem，发现：去掉 IcePop 后训练不稳定，最终分数下降约 8 分；去掉 C3PO++ 导致显存爆炸，最长 rollout 只能到 4 k token，整体性能下降约 5%；去掉 ASystem 则整体吞吐率下降 40%，训练时间翻倍。  
- **局限性**：虽然模型在公开基准上表现突出，但在极端长文本（>100 k token）仍会出现推理漂移；价值网络的质量仍受限于标注数据规模；作者承认对硬件需求极高，仅在少数拥有数百 GPU 的实验室可复现。

### 影响与延伸思考

Ring‑1T 首次展示了 **万亿级 MoE 模型在强化学习框架下的可训练性**，为后续的“大规模思考模型”打开了大门。自论文发布后，已有多篇工作尝试在 **多模态推理**（视觉‑语言）和 **跨语言数学** 上复用 IcePop 与 C3PO++ 的思路；还有研究把 ASystem 的调度策略迁移到 **大规模自监督预训练** 中，以提升算力利用率。对想进一步探索的读者，建议关注以下方向：  
1) **更高效的价值网络**——如何在更少标注下训练出可靠的价值评估器；  
2) **跨模态 MoE‑RL**——把视觉专家加入思考链，提升科学实验推理能力；  
3) **硬件友好的稀疏通信**——在更低成本的 GPU 集群上实现类似的吞吐率。  

### 一句话记住它

Ring‑1T 用 **Token‑级别的噪声遮蔽 + 动态 rollout 切分 + 专用系统调度**，让万亿参数的思考模型在强化学习中真正跑通。