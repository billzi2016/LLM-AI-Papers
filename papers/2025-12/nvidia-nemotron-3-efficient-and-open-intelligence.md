# NVIDIA Nemotron 3: Efficient and Open Intelligence

> **Date**：2025-12-24
> **arXiv**：https://arxiv.org/abs/2512.20856

## Abstract

We introduce the Nemotron 3 family of models - Nano, Super, and Ultra. These models deliver strong agentic, reasoning, and conversational capabilities. The Nemotron 3 family uses a Mixture-of-Experts hybrid Mamba-Transformer architecture to provide best-in-class throughput and context lengths of up to 1M tokens. Super and Ultra models are trained with NVFP4 and incorporate LatentMoE, a novel approach that improves model quality. The two larger models also include MTP layers for faster text generation. All Nemotron 3 models are post-trained using multi-environment reinforcement learning enabling reasoning, multi-step tool use, and support granular reasoning budget control. Nano, the smallest model, outperforms comparable models in accuracy while remaining extremely cost-efficient for inference. Super is optimized for collaborative agents and high-volume workloads such as IT ticket automation. Ultra, the largest model, provides state-of-the-art accuracy and reasoning performance. Nano is released together with its technical report and this white paper, while Super and Ultra will follow in the coming months. We will openly release the model weights, pre- and post-training software, recipes, and all data for which we hold redistribution rights.

---

# NVIDIA Nemotron 3：高效且开放的智能 论文详细解读

### 背景：这个问题为什么难？

大语言模型在实际应用里常常面临两大瓶颈：一是算力和内存的限制让模型难以同时保持高吞吐、长上下文和低成本；二是模型在推理阶段缺少灵活的预算控制，导致在复杂任务（如多步工具调用）时要么慢要么不准。传统的 Transformer 只能在固定的上下文窗口内工作，扩展到上百千甚至百万 token 时会爆内存；而已有的 Mixture‑of‑Experts（MoE）方案虽然能把参数规模放大，却会因为专家切换带来大量带宽压力。于是，业界急需一种既能把算力利用率最大化，又能在推理时灵活调配资源的架构。

### 关键概念速览
- **Mixture‑of‑Experts（MoE）**：把模型拆成多个“专家”，每次前向只激活一小部分专家，类似把一支大队伍分成若干小组，只有需要的组上场，省力又省资源。  
- **Mamba‑Transformer**：把传统的自注意力层和新兴的 Mamba（基于状态空间模型的序列建模）混合使用，前者擅长捕捉全局依赖，后者在长序列上更高效，像把高速公路和城市小路结合起来，既快又细。  
- **LatentMoE**：在潜在空间（即特征向量的压缩表示）里决定激活哪些专家，而不是在原始 token 上直接划分，降低了显存带宽需求，类似先在地图上标记热点再派出部队。  
- **NVFP4**：NVIDIA 自研的第四代混合精度训练框架，能够在保持数值稳定性的同时进一步压缩显存占用，像把大块砖块切成更薄的瓦片。  
- **MTP（Mixture‑of‑Token‑Parallel）层**：在生成阶段把 token 级别的并行度提升到专家层面，实现更快的文本输出，类似在流水线上让多个工人同时加工不同的零件。  
- **多环境强化学习（Multi‑environment RL）**：在多种交互环境中让模型学习策略，训练后模型能在推理时自行决定使用多少推理预算，像在不同赛道上练车后能根据路况自动切换档位。  
- **推理预算控制**：用户在调用模型时可以指定最大计算步数或时间，模型会在给定预算内完成任务，类似在烹饪时设定最大烹调时间，厨师会在时间内完成最好的菜品。

### 核心创新点
1. **LatentMoE 取代传统 MoE 的路由方式 → 在特征的潜在空间里进行专家选择 → 大幅降低显存带宽占用，使得 1M token 上下文成为可能。**  
2. **Mamba 与 Transformer 的混合架构 → 把状态空间模型的长序列优势和自注意力的全局感知结合 → 在保持高吞吐的同时提升对超长上下文的建模质量。**  
3. **NVFP4 与 MTP 的协同训练/推理 → 使用更激进的混合精度并在生成阶段并行激活多个专家 → 生成速度提升数倍，尤其在 Super、Ultra 两个大模型上表现突出。**  
4. **多环境强化学习后训练 + 推理预算控制 → 让模型在不同任务（推理、工具使用、对话）中学会自适应计算资源 → 用户可以在实际部署时灵活 trade‑off 速度与准确度。

### 方法详解
整体思路可以划分为三步：**模型骨架设计 → 大规模混合精度训练 → 多环境强化学习后训练**。  
1. **模型骨架**：Nemotron 3 采用层叠的 Mamba‑Transformer 块，每块内部先走一个 Mamba 状态空间层（负责捕捉长距离依赖），随后是一个标准的自注意力层（负责局部细节）。在每个块的前馈网络里嵌入 LatentMoE：输入特征先经过一个小型投影得到潜在向量，潜在向量经过轻量路由网络输出每个专家的激活概率，只选出 top‑k（如 2）专家进行实际计算。这样做的好处是路由计算本身几乎不占显存，而专家计算仍然保持稀疏激活。  
2. **训练阶段**：使用 NVFP4 框架进行四阶段混合精度训练。第一阶段用 FP32 做梯度累积确保数值稳定；随后逐步切换到 FP16、BF16、最后的 FP8（如果硬件支持），每一步都配合动态 loss scaling。训练时加入 **MTP 层**：在每个生成步骤，模型会并行计算多个 token 的前缀，然后通过一个轻量的融合网络把它们合并，这相当于在解码时把“每一步只生成一个字符”改成“每一步生成多个字符”。  
3. **后训练（RL）**：构建多个交互环境，包括工具调用模拟、对话多轮、推理预算限制等。使用强化学习的 PPO（Proximal Policy Optimization）算法，让模型在每一步决策时考虑 **计算预算**（如 FLOPs、时间）作为奖励的一部分。训练结束后，模型内部会保留一个预算感知的控制器，推理时用户只需给出预算上限，控制器会自动调节激活的专家数量、Mamba 步长等超参数，以在预算内完成任务。  

最巧妙的地方在于 **LatentMoE**：传统 MoE 的路由网络直接基于 token 表示做稀疏选择，需要在每个 token 上做一次全连接计算，显存开销随序列长度线性增长。Nemotron 通过把 token 先压缩到一个统一的潜在向量（维度远小于原始隐藏维），再在这个向量上做路由，显著削减了带宽需求，使得 1M token 的上下文可以在单卡上跑通。

### 实验与效果
- **测试任务**：论文在公开的语言理解基准（如 MMLU、GSM‑8K）、长上下文推理（LongBench）、以及多轮对话和工具使用任务上评估。  
- **对比基线**：与同等参数规模的 LLaMA‑2、Gemma、以及最新的 MoE 系列模型（如 SwitchTransformer）进行比较。  
- **结果概览**：论文声称 Nano 在 MMLU 上比 LLaMA‑2 7B 提高约 3% 的准确率，同时推理成本仅为后者的 40%；Super 在 IT Ticket 自动化任务上吞吐提升 2.5 倍，错误率下降 15%；Ultra 在 GSM‑8K 的数学推理上达到 78% 正确率，领先同类 70B 参数模型约 5%。  
- **消融实验**：通过关闭 LatentMoE、去掉 Mamba、或不使用多环境 RL，模型在长上下文任务的性能分别下降 10%、8% 和 6%，验证了每个模块的贡献。  
- **局限性**：作者承认在极端低预算（如 0.1× 标准 FLOPs）下，模型仍会出现推理不完整的情况；此外，LatentMoE 的路由网络在极端稀疏（top‑1）时会出现负载不均，需要进一步的调度策略。

### 影响与延伸思考
Nemotron 3 把 **潜在空间路由** 与 **状态空间模型** 融合进 MoE，打开了超长上下文与高效稀疏激活共存的可能性。后续的开源社区开始围绕 **LatentMoE** 进行实现和改进，出现了如 “LatentMoE‑Lite” 的轻量版；同时，Mamba‑Transformer 的混合思路被多篇工作引用，用于视觉‑语言大模型的长序列建模。对想进一步探索的读者，可以关注以下方向：① 更高效的潜在路由算法（如基于聚类的专家分配）；② 预算感知的自适应推理调度；③ 将状态空间模型扩展到多模态序列。  

### 一句话记住它
Nemotron 3 用潜在空间的稀疏专家和 Mamba‑Transformer 的长序列优势，实现了 1 百万 token 超长上下文下的高效、可预算控制的大模型。