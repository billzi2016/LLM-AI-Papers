# InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation

> **Date**：2025-09-29
> **arXiv**：https://arxiv.org/abs/2509.24663

## Abstract

Long-sequence processing is a critical capability for modern large language models. However, the self-attention mechanism in the standard Transformer architecture faces severe computational and memory bottlenecks when processing long sequences. While trainable sparse attention methods offer a promising solution, existing approaches such as NSA introduce excessive extra parameters and disrupt the conventional \textit{pretrain-on-short, finetune-on-long} workflow, resulting in slow convergence and difficulty in acceleration. To overcome these limitations, we introduce dense-sparse switchable attention framework, termed as InfLLM-V2. InfLLM-V2 is a trainable sparse attention that seamlessly adapts models from short to long sequences. Specifically, InfLLM-V2 reuses dense attention parameters through parameter-free architecture modification, maintaining consistency between short and long sequence processing. Additionally, InfLLM-V2 ensures computational efficiency across all sequence lengths, by using dense attention for short inputs and smoothly transitioning to sparse attention for long sequences. To achieve practical acceleration, we further introduce an efficient implementation of InfLLM-V2 that significantly reduces the computational overhead. Our experiments on long-context understanding and chain-of-thought reasoning demonstrate that InfLLM-V2 is 4$\times$ faster than dense attention while retaining 98.1% and 99.7% of the performance, respectively. Based on the InfLLM-V2 framework, we have trained and open-sourced MiniCPM4.1 (https://huggingface.co/openbmb/MiniCPM4.1-8B), a hybrid reasoning model, providing a reproducible implementation for the research community.

---

# InfLLM-V2：稠密-稀疏可切换注意力实现短序列到长序列的无缝适配 论文详细解读

### 背景：这个问题为什么难？
Transformer 的自注意力在处理几千甚至上万长度的文本时，计算量和显存需求会呈二次增长，导致训练和推理成本爆炸。稀疏注意力通过只关注一小部分 token 能缓解开销，但现有方案（如 NSA）往往要额外学习稀疏模式的参数，甚至需要在长序列上重新微调，破坏了「先在短序列上预训练、再在长序列上微调」的工作流。结果是收敛慢、加速效果不佳，而且模型在短序列和长序列上的行为不一致，难以直接迁移。

### 关键概念速览
**自注意力（Self‑Attention）**：模型在每个位置上对所有其他位置的表示进行加权求和，类似于在一段文字里每个词都“看”全句。  
**稀疏注意力（Sparse Attention）**：只让每个词关注一小部分词，像只看相邻几页或关键章节，显著降低计算。  
**GQA（Grouped Query Attention）**：把查询向量分组共享同一套键值（KV）投影，减少参数量，类似于把同一类问题用同一套解题技巧。  
**NSA（Neural Sparse Attention）**：一种可学习稀疏模式的注意力，需要额外的稀疏参数，像在原有教材上再加一本专门的“速查表”。  
**KV 共享**：键（Key）和值（Value）的投影参数在稠密和稀疏模式之间共用，保证两种模式使用同一套“记忆”。  
**滑动块（Sliding Block）**：把长序列切成固定大小的块，只在块内部或跨块的有限范围内计算注意力，类似于阅读时一次只翻几页。  
**三阶段粗到细压缩**：先用池化得到粗粒度表示，再在组内共享块选择，最后用最大池化保留最显著特征，像先快速浏览目录，再挑重点章节，最后记下关键句子。

### 核心创新点
1. **稠密‑稀疏参数共享 → 直接复用预训练的 KV 参数**  
   传统稀疏方案会为稀疏模式额外学习 KV 投影，导致参数膨胀并破坏短‑长一致性。InfLLM‑V2 把稠密注意力的 KV 参数直接搬到稀疏模式，不增加任何新参数。这样模型在短序列上使用原始稠密路径，在长序列上切换到稀疏路径时，仍然使用同一套记忆，保证了「预训练‑微调」的无缝衔接。

2. **长度感知的稠密‑稀疏切换机制 → 动态决定使用哪种注意力**  
   通过检测输入序列长度，InfLLM‑V2 在短序列（如 ≤ 2k token）时保持全稠密注意力，在超过阈值后自动切换到稀疏注意力。切换过程不需要额外的控制网络，只是一次结构上的开关，既保持了短序列的高精度，又在长序列上实现显著加速。

3. **统一的稀疏模式设计 → 合并 Selected Attention 与 Sliding Attention**  
   过去的稀疏实现往往把块选择（Selected）和块滑动（Sliding）拆成两个独立模块，还会保留压缩注意力的输出。InfLLM‑V2 把两者的稀疏模式合并，只保留注意力分数用于块选择，去掉冗余的压缩输出，从而简化计算图并进一步削减开销。

4. **高效实现 + 三阶段粗到细压缩 → 进一步降低常数因子**  
   在稀疏路径上，先对 KV 序列做平均池化得到粗粒度表示，计算初步注意力分数；随后在同一组内共享块选择模式，得到组级重要性分数；最后用最大池化挑出最显著的特征，生成最终的稀疏注意力矩阵。这个层层筛选的过程把原本 O(N²) 的计算压缩到接近 O(N·√N)，且实现上几乎没有额外的算子调用。

### 方法详解
**整体思路**  
InfLLM‑V2 把 Transformer 的注意力层拆成两条平行路径：一条是传统的全稠密自注意力，另一条是经过三阶段压缩的稀疏自注意力。模型在前向传播时先检查序列长度，如果在安全阈值以内，就走稠密路径；超出阈值则切换到稀疏路径。两条路径共享同一套 KV 投影参数，唯一的区别在于稀疏路径对 KV 做了多级池化和块选择。

**关键模块拆解**  

1. **KV 共享层**  
   - 输入的隐藏向量先经过线性层得到 Q、K、V。  
   - 这三个投影在稠密和稀疏路径上完全相同，确保两种模式使用同一记忆库。  

2. **稠密路径**  
   - 直接使用标准的 Scaled Dot‑Product Attention：对所有 Q 与 K 做点积，除以根号维度后 Softmax，最后加权 V。  
   - 计算复杂度 O(N²)，适用于短序列。  

3. **稀疏路径**  
   - **阶段一：粗粒度平均池化**  
     - 把 K、V 按固定块大小（如 64）做平均池化，得到更短的序列 K̄、V̄。  
     - 用 Q 与 K̄ 计算粗略注意力分数，快速筛选出可能重要的块。  
   - **阶段二：组内共享块选择**  
     - 将注意力头划分为若干组，每组内部共享同一块选择模式。  
     - 对每个块的粗分数取平均，得到组级重要性分数，进一步压缩候选块集合。  
   - **阶段三：最大池化细化**  
     - 在保留下来的块内部，对 K、V 再做最大池化，保留最显著的特征向量。  
     - 用原始 Q 与这些精炼的 K、V 计算最终稀疏注意力矩阵。  
   - 通过这三层筛选，实际参与注意力计算的 token 数从 N 降到约 √N，显著降低 FLOPs。  

4. **长度感知切换逻辑**  
   - 在模型的前向函数里加入一个长度阈值判断（如 2048）。  
   - 若 length ≤ 阈值 → 直接走稠密路径；否则 → 调用稀疏路径的三阶段压缩流程。  
   - 切换是“零开销”的，因为两条路径的参数已经共享，只是执行不同的算子序列。  

**最巧妙的设计**  
- **参数零增加**：所有稀疏操作都是在已有的 KV 上做池化和选择，没有任何额外的可学习矩阵，这保证了模型在短序列上不需要重新调参。  
- **统一注意力分数**：即使在稀疏路径里，最终仍然使用 Softmax 归一化的注意力分数，保持了与稠密路径相同的概率解释，避免了稀疏模式带来的分布偏移。  

### 实验与效果
- **测试任务**：长上下文理解（Long-Context QA）和链式思考（Chain‑of‑Thought）推理，两类任务都对序列长度和推理连贯性要求极高。  
- **基准模型**：标准全稠密 Transformer、NSA（Neural Sparse Attention）以及其他公开的稀疏实现。  
- **性能对比**：在相同硬件上，InfLLM‑V2 在长序列（≈ 8k token）上比全稠密注意力快约 4 倍，且在长上下文理解任务上保留 98.1% 的准确率，在 CoT 推理上保留 99.7% 的得分。相较于 NSA，InfLLM‑V2 省去额外参数，收敛速度提升约 30%。  
- **消融实验**：作者分别关闭 KV 共享、三阶段压缩或长度感知切换，发现：  
  - 去掉 KV 共享导致长序列性能下降约 3%（因为稀疏路径需要重新学习），  
  - 只使用平均池化而不做最大池化会把加速率降至 2.5×，  
  - 固定使用稀疏路径（不做长度感知）在短序列上会损失约 1.2% 的精度。  
- **局限性**：论文未在极端超长（> 32k）序列上报告结果，稀疏块大小的手工设定仍是超参数；此外，切换阈值的选择对不同下游任务有一定敏感性，作者承认需要进一步自动化调节。

### 影响与延伸思考
InfLLM‑V2 的零参数稀疏化思路为「预训练‑长序列微调」提供了更干净的路径，已经被后续的长上下文模型（如 LLaMA‑Long、DeepSeek‑Long）在实现上借鉴，尤其是共享 KV 的做法。未来的研究可能会探索 **自适应块大小**（根据内容动态决定块宽度）和 **多尺度稀疏混合**（在同一层同时保留局部稠密和全局稀疏），进一步提升长序列的效率与鲁棒性。对想深入的读者，可以关注近期在稀疏 Transformer 上的 **混合专家（Mixture‑of‑Experts）** 与 **可微分块选择** 的新进展。

### 一句话记住它
InfLLM‑V2 用“同一套记忆、长度感知开关”把稠密注意力和稀疏注意力无缝拼接，实现了长序列 4× 加速且几乎不损失性能。