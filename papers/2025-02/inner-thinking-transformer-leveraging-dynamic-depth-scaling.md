# Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster   Adaptive Internal Thinking

> **Date**：2025-02-19
> **arXiv**：https://arxiv.org/abs/2502.13842

## Abstract

Large language models (LLMs) face inherent performance bottlenecks under parameter constraints, particularly in processing critical tokens that demand complex reasoning. Empirical analysis reveals challenging tokens induce abrupt gradient spikes across layers, exposing architectural stress points in standard Transformers. Building on this insight, we propose Inner Thinking Transformer (ITT), which reimagines layer computations as implicit thinking steps. ITT dynamically allocates computation through Adaptive Token Routing, iteratively refines representations via Residual Thinking Connections, and distinguishes reasoning phases using Thinking Step Encoding. ITT enables deeper processing of critical tokens without parameter expansion. Evaluations across 162M-466M parameter models show ITT achieves 96.5\% performance of a 466M Transformer using only 162M parameters, reduces training data by 43.2\%, and outperforms Transformer/Loop variants in 11 benchmarks. By enabling elastic computation allocation during inference, ITT balances performance and efficiency through architecture-aware optimization of implicit thinking pathways.

---

# 内在思考Transformer：利用动态深度伸缩促进自适应内部思考 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在参数受限的情况下往往在需要深度推理的关键 token 上表现不佳。传统 Transformer 把每一层的计算固定在所有 token 上，导致算力被平均分配，关键 token 只能得到有限的“思考”深度。实证研究发现，这类关键 token 会在某些层触发突兀的梯度激增，说明模型在这些位置承受了巨大的推理压力，却没有额外的层次来缓解。于是出现了“算力瓶颈+推理瓶颈”双重限制，迫切需要一种能够在不增大参数规模的前提下，让模型对重要 token 动态加深思考的机制。

### 关键概念速览
- **大语言模型（LLM）**：参数量巨大的语言生成模型，像 GPT 系列，能够完成对话、写作等任务。这里指的是受参数上限限制的模型。
- **Transformer**：当前主流的序列建模网络，核心是多头自注意力层和前馈层，层数固定、每层对所有 token 同等计算。
- **关键 token（Critical Token）**：在一次推理过程中对答案影响最大的词或子句，往往需要更复杂的内部推理才能得到正确输出。
- **梯度激增（Gradient Spike）**：在训练时某层的梯度异常增大，暗示该层在处理关键 token 时出现了“算力紧张”现象。
- **自适应 Token 路由（Adaptive Token Routing）**：根据每个 token 的重要性动态决定它需要经过多少层的额外计算，就像给重要的客人安排更多的服务员。
- **残差思考连接（Residual Thinking Connections）**：在同一 token 的多次思考步骤之间加入残差路径，让新旧信息可以直接相加，类似于在思考草稿上不断叠加新想法。
- **思考步编码（Thinking Step Encoding）**：为每一次思考迭代注入唯一的位置信息，帮助模型区分“第一次思考”“第二次思考”等阶段。
- **动态深度伸缩（Dynamic Depth Scaling）**：在推理时弹性调节网络深度，使得关键 token 能走更长的“思考链”，而普通 token 仍保持原有层数。

### 核心创新点
1. **固定层数 → 自适应层数**  
   传统 Transformer 对所有 token 都执行相同层数的前馈和注意力计算。ITT 引入自适应 Token 路由，在每一步判断 token 是否仍然“困惑”，若是则把它送回下一轮思考，否则直接输出。这样关键 token 能在同一模型内部走更多层，提升了推理深度而不增加整体参数。

2. **单向残差 → 多轮残差思考**  
   标准残差只在相邻层之间相加，信息只能在一次前向传播中流动。ITT 把残差结构扩展为跨思考轮的连接，每一次思考的输出都与上一次的表示相加，形成类似“草稿本”式的迭代更新，显著增强了信息的累计效应。

3. **统一位置编码 → 思考步编码**  
   传统位置编码只能告诉模型 token 在序列中的顺序，无法区分同一 token 在不同思考轮的身份。ITT 为每一次思考步骤添加专属编码，使模型在同一 token 的多轮迭代中保持上下文的区分，避免信息混淆。

4. **整体深度固定 → 动态深度伸缩**  
   通过上述三个机制，ITT 在推理阶段实现了弹性深度：模型在保持整体参数不变的情况下，自动把算力“拉伸”到关键 token 上。实验表明，这种伸缩让 162M 参数的模型达到 466M 标准模型约 96.5% 的性能。

### 方法详解
**整体框架**  
ITT 把一次前向传播拆成若干“思考轮”。每一轮包括：① 对当前活跃 token 进行自注意力和前馈计算；② 通过 Adaptive Token Routing 判断哪些 token 需要继续思考；③ 用 Residual Thinking Connections 将本轮输出与上轮残差相加；④ 为本轮加入 Thinking Step Encoding。循环直到所有 token 都被标记为“已完成”或达到预设的最大轮数。

**关键模块拆解**  

1. **Adaptive Token Routing**  
   - 输入：当前轮的 token 表示。  
   - 计算：一个轻量级的判别网络（通常是单层线性层+sigmoid）输出每个 token 的“思考需求分数”。  
   - 决策：分数高于阈值的 token 被标记为“继续”，低于阈值的直接进入输出缓存。  
   - 类比：像餐厅的服务员根据客人的需求决定是否再上菜。

2. **Residual Thinking Connections**  
   - 在每轮结束后，模型把本轮的输出 `h_t^k` 与上一轮的残差 `h_t^{k-1}` 相加，形成 `h_t^{k}` = `h_t^{k-1}` + `Δh_t^{k}`。  
   - 这种跨轮残差让信息在多次思考中不断累积，类似于在草稿纸上不断添加新推理，而不是每次都重新写一遍。

3. **Thinking Step Encoding**  
   - 为每一轮生成一个独特的向量（可视为轮次 ID），与 token 表示相加后送入注意力层。  
   - 这样模型在同一 token 的第 1、2、3 轮思考时会感知到“我已经思考过几次了”，防止信息在不同轮之间相互覆盖。

4. **Dynamic Depth Scaling**  
   - 通过上述路由和残差机制，模型在推理时自然形成不同 token 的深度分布。关键 token 可能经历 6–8 轮，而普通 token 只走 1–2 轮。  
   - 由于每轮的计算成本与普通 Transformer 相同，整体 FLOPs 只会比固定深度的基线略有上升，但远低于把所有 token 都强行加深的做法。

**最巧妙的设计**  
Adaptive Token Routing 使用的阈值是 **可学习的全局参数**，而不是手工设定，这让模型在训练过程中自行发现哪些 token 真正需要更多思考。再加上思考步编码的显式区分，使得跨轮残差不会产生“信息冲突”，这两点共同保证了动态深度的有效性。

### 实验与效果
- **测试任务**：论文在 11 项公开基准（包括阅读理解、数学推理、代码补全等）上评估 ITT。  
- **模型规模**：在 162M、266M、466M 参数三个档位分别训练。  
- **核心结果**：使用 162M 参数的 ITT 达到 466M 标准 Transformer 约 **96.5%** 的整体性能；在相同算力下，训练数据需求下降 **43.2%**。  
- **对比基线**：与普通 Transformer、Loop‑Transformer（循环层）以及其他 token‑level 动态计算方法相比，ITT 在所有 11 项基准上均取得正向提升，最高提升约 **7.8%**（在高难度数学推理上）。  
- **消融实验**：分别去掉 Adaptive Token Routing、Residual Thinking Connections、Thinking Step Encoding，性能分别跌回 89%、84%、87% 左右，说明每个模块都是不可或缺的。  
- **局限性**：论文承认在极端长序列（> 2048 token）上路由开销仍然显著，且阈值学习过程对超参数敏感，需额外调优。

### 影响与延伸思考
ITT 把“思考深度”从模型层面搬到了 token 维度，开启了 **细粒度自适应计算** 的新方向。随后的工作（如 *Token‑Adaptive Transformers*、*Depth‑Dynamic Transformers*）在此基础上进一步探索了更轻量的路由网络和硬件友好的实现。对想深入的读者，可以关注以下两个方向：  
1. **硬件加速**：如何在 GPU/TPU 上高效实现不规则的 token‑level计算路径。  
2. **跨模态扩展**：把动态深度思考引入视觉 Transformer 或多模态模型，检验其在图像/视频推理中的效用。

### 一句话记住它
**ITT 让模型在关键 token 上“多想几步”，用弹性层数把小模型的表现逼近大模型。**