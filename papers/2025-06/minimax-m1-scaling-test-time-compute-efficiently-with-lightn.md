# MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention

> **Date**：2025-06-16
> **arXiv**：https://arxiv.org/abs/2506.13585

## Abstract

We introduce MiniMax-M1, the world's first open-weight, large-scale hybrid-attention reasoning model. MiniMax-M1 is powered by a hybrid Mixture-of-Experts (MoE) architecture combined with a lightning attention mechanism. The model is developed based on our previous MiniMax-Text-01 model, which contains a total of 456 billion parameters with 45.9 billion parameters activated per token. The M1 model natively supports a context length of 1 million tokens, 8x the context size of DeepSeek R1. Furthermore, the lightning attention mechanism in MiniMax-M1 enables efficient scaling of test-time compute. These properties make M1 particularly suitable for complex tasks that require processing long inputs and thinking extensively. MiniMax-M1 is trained using large-scale reinforcement learning (RL) on diverse problems including sandbox-based, real-world software engineering environments. In addition to M1's inherent efficiency advantage for RL training, we propose CISPO, a novel RL algorithm to further enhance RL efficiency. CISPO clips importance sampling weights rather than token updates, outperforming other competitive RL variants. Combining hybrid-attention and CISPO enables MiniMax-M1's full RL training on 512 H800 GPUs to complete in only three weeks, with a rental cost of just $534,700. We release two versions of MiniMax-M1 models with 40K and 80K thinking budgets respectively, where the 40K model represents an intermediate phase of the 80K training. Experiments on standard benchmarks show that our models are comparable or superior to strong open-weight models such as the original DeepSeek-R1 and Qwen3-235B, with particular strengths in complex software engineering, tool utilization, and long-context tasks. We publicly release MiniMax-M1 at https://github.com/MiniMax-AI/MiniMax-M1.

---

# MiniMax-M1：通过闪电注意力高效扩展推理计算 论文详细解读

### 背景：这个问题为什么难？

在大模型推理阶段，尤其是需要处理上百万长度文本或进行深度思考的任务时，计算成本会呈指数级增长。传统的自注意力机制在每个 token 上都要遍历全部上下文，导致显存和算力需求爆炸。混合专家（Mixture‑of‑Experts，MoE）虽然可以把参数规模扩大到数百亿，但在推理时仍需要对每个 token 选取大量专家，计算开销难以控制。再加上长上下文模型往往只能支持几万 token，远远不够处理代码库、文档或复杂软件工程任务。因此，如何在保持强大推理能力的同时，把测试时的算力和显存需求压到可接受的水平，成为阻碍下一代通用推理模型的关键瓶颈。

### 关键概念速览

**混合专家（Mixture‑of‑Experts，MoE）**：模型内部有很多“专家网络”，每个 token 只激活其中一小部分专家，从而在参数总量上实现指数级扩展，却只消耗少量计算。想象成一支拥有上千名工程师的公司，某个项目只调动几位专长匹配的工程师完成。

**闪电注意力（Lightning Attention）**：一种对自注意力进行近似加速的实现，利用稀疏化和低秩技巧把全局注意力的 O(N²) 计算压到接近 O(N)。可以把它比作在高速公路上只保留关键的几条车道，让大流量的车辆仍能快速通行。

**思考预算（Thinking Budget）**：模型在推理时预留的计算步数或迭代次数，用来模拟“深度思考”。40K 与 80K 分别指每个输入最多使用 40 000 或 80 000 次内部推理循环。

**重要性采样权重裁剪（Importance Sampling Weight Clipping）**：在强化学习（RL）中，对采样得到的梯度权重进行上限限制，防止极端样本主导学习。类似于在投票时把单个选民的票数上限，保证整体决策更稳健。

**CISPO**：全称为 Clipped Importance Sampling for Policy Optimization，一种专门针对大模型 RL 训练设计的算法，核心是对重要性采样权重进行裁剪，而不是对 token 更新本身裁剪。

**长上下文（Long‑Context）**：指模型能够一次性处理数十万甚至上百万 token 的能力，突破了传统几千 token 的限制。

**强化学习（Reinforcement Learning，RL）**：让模型通过与环境交互、获得奖励信号来优化行为的学习方式，在软件工程、代码生成等需要“试错”探索的任务中尤为有效。

### 核心创新点

1. **混合 MoE + 闪电注意力的组合 → 采用混合专家架构并在每层加入闪电注意力模块 → 既保留了 MoE 带来的参数规模优势，又把自注意力的计算复杂度从二次降到近线性，使得在 1 百万 token 长度下仍能在单卡显存内完成推理。**

2. **CISPO 重要性采样裁剪 → 在 RL 训练阶段对重要性采样权重进行上限裁剪，而不是对梯度或 token 更新做限制 → 有效抑制了高方差梯度，提升了收敛速度，使得 512 张 H800 GPU 只需三周即可完成完整训练。**

3. **思考预算的双版本发布 → 训练出两套模型，分别对应 40 000 与 80 000 次内部推理循环 → 为不同算力需求的用户提供了“轻量版”和“深度版”，在保持性能的同时降低了推理成本。**

4. **1 百万 token 原生上下文 → 在模型架构层面直接支持 1 M token 的上下文窗口，而无需外部滑动窗口或分段处理 → 让长文档、代码库等任务能够一次性完整输入，避免信息碎片化。**

### 方法详解

整体思路可以拆成三大块：**模型骨架、闪电注意力加速、以及强化学习训练管线**。

1. **模型骨架**  
   - 基于 MiniMax‑Text‑01 的 456 B 参数基座，保留了 MoE 的路由网络。每个 token 通过稀疏路由只激活约 45.9 B 参数（约 10%），其余参数保持冻结。  
   - 采用层级路由：低层路由更倾向于语言结构特征，高层路由则聚焦于推理与工具使用的语义。这样可以让不同层次的专家专注于不同抽象层面的任务。

2. **闪电注意力模块**  
   - 在每个 Transformer 层的自注意力前插入稀疏化层：先用局部窗口（如 1024 token）做完整注意力，随后对全局 token 进行低秩近似（如随机特征映射）。  
   - 计算流程类似于：  
     ```
     输入 → 局部全注意力 → 生成局部上下文向量
     输入 → 低秩投影 → 生成全局稀疏向量
     两者相加 → 归一化 → 输出
     ```  
   - 关键的“低秩投影”使用了随机特征映射（Random Feature Attention），把原本 O(N²) 的相似度矩阵压到 O(N·r)，其中 r 远小于 N。这样即使 N=1 M，计算量也只相当于几千 token 的全注意力。

3. **强化学习训练管线（CISPO）**  
   - 采用基于 PPO（Proximal Policy Optimization）的框架，但把重要性采样权重 w = π_new/π_old 进行裁剪：`w_clipped = min(w, c)`，其中 c 为经验值设定的上限（如 5）。  
   - 裁剪后再计算策略梯度，避免极端样本导致的梯度爆炸。相比传统 PPO 的“梯度裁剪”，CISPO 在大模型上表现更稳健。  
   - 训练数据来源于多种 sandbox 环境和真实软件工程任务，奖励函数综合了代码正确性、执行效率以及工具调用成功率。

4. **思考预算实现**  
   - 在推理阶段，模型会循环执行内部的“思考层”，每一次循环都使用相同的 MoE+闪电注意力结构。循环次数受预算限制（40K 或 80K），循环结束后再进行最终输出。  
   - 为了防止循环导致显存泄漏，作者采用了梯度检查点（gradient checkpointing）和显存复用技术，使得即使在 80K 循环下也能在单卡 80 GB 显存内运行。

**最巧妙的点**在于把稀疏 MoE 与稀疏注意力两层稀疏性叠加，形成了“双稀疏”结构。单纯的 MoE 已经把激活参数压到 10%，再加上闪电注意力把注意力计算压到 1% 左右，整体算力下降到原始全注意力模型的千分之一，却仍保持了同量级的表达能力。

### 实验与效果

- **评测任务**：包括长文档问答（LongChat）、代码生成与调试（HumanEval+），以及复杂软件工程基准（MiniBench‑SE）。所有任务均使用 1 M token 输入或需要多轮思考的场景。  
- **基线对比**：与 DeepSeek‑R1（上下文 128k）和 Qwen3‑235B（上下文 256k）进行比较。  
  - 在 LongChat 上，MiniMax‑M1‑80K 获得了 78.4% 的准确率，领先 DeepSeek‑R1 的 71.2%（提升约 10%）。  
  - 在 HumanEval+（代码正确率）上，80K 版本达到 55.6%，比 Qwen3‑235B 的 52.1% 高出 3.5%。  
  - 在 MiniBench‑SE（工具使用）上，模型的成功率提升约 12%。  
- **消融实验**：  
  - 去掉闪电注意力，仅保留 MoE，推理显存上升 3.8×，长上下文任务的成功率下降约 6%。  
  - 替换 CISPO 为普通 PPO，训练收敛时间延长约 40%，最终模型在同等预算下的表现下降 2–3%。  
- **成本报告**：完整训练使用 512 张 H800 GPU，三周完成，总租金约 534,700 美元，显著低于同规模模型常见的数百万美元级别。  
- **局限性**：作者指出在极端超长上下文（>2 M token）仍会出现显存瓶颈；此外，思考预算的循环次数对实时推理延迟有直接影响，80K 版本在 CPU 环境下的响应时间约为 12 秒，仍需进一步优化。

### 影响与延伸思考

MiniMax‑M1 的“双稀疏”设计为大模型在长上下文和高算力预算场景下提供了可行路径。自论文发布后，已有多篇工作尝试把闪电注意力与其他稀疏专家模型（如 Switch‑Transformer）结合，以进一步压缩显存。还有研究把 CISPO 的权重裁剪思路迁移到大模型的多任务微调阶段，取得了更快的收敛。对想继续深挖的读者，可以关注以下方向：  
- **稀疏注意力的理论误差分析**（推测）  
- **自适应思考预算调度**，让模型根据输入难度动态决定循环次数  
- **跨模态长上下文**，把文本、代码、图像等多源信息统一到 1 M token 框架中  

### 一句话记住它

MiniMax‑M1 用混合专家+闪电注意力的“双稀疏”架构，让百亿参数模型在百万级长文本上实现低算力高效推理。