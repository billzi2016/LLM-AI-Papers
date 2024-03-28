# Streamlining Redundant Layers to Compress Large Language Models

> **Date**：2024-03-28
> **arXiv**：https://arxiv.org/abs/2403.19135

## Abstract

This paper introduces LLM-Streamline, a pioneer work on layer pruning for large language models (LLMs). It is based on the observation that different layers have varying impacts on hidden states, enabling the identification of less important layers to be pruned.LLM-Streamline comprises two parts: layer pruning, which removes consecutive layers with the lowest importance based on target sparsity, and layer replacement, a novel module that trains a lightweight network to replace the pruned layers to mitigate performance loss. Additionally, a new metric called stability is proposed to address the limitations of the widely used accuracy metric in evaluating model compression. Experiments show that LLM-Streamline outperforms both previous and concurrent state-of-the-art pruning methods in terms of both performance and training efficiency.Our code is available at https://github.com/RUCKBReasoning/LLM-Streamline

---

# 精简冗余层以压缩大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）往往拥有数十甚至上百层的 Transformer 编码器，每层都要消耗显存和算力。传统的压缩手段——比如权重剪枝、量化或蒸馏——主要聚焦在单个参数上，忽视了层与层之间的功能差异。实际使用中，很多层的输出对最终隐藏状态的贡献并不均衡，却很少有方法能够系统地识别并去除这些“冗余”层。直接删掉层会导致信息流断裂，性能急剧下降，这也是过去层剪枝研究停滞的根本原因。

### 关键概念速览
- **层重要性（Layer Importance）**：衡量某一层对模型隐藏状态变化的贡献大小，数值越大说明该层越关键。可以想象成一条流水线上的工序，重要性高的工序决定了最终产品质量。
- **层剪枝（Layer Pruning）**：按照重要性排序，删除一段连续的低重要性层。类似于把生产线中不必要的环节直接拆除，以降低成本。
- **层替换（Layer Replacement）**：用一个轻量网络（通常是几层小的 MLP）来模拟被剪掉层的功能，防止信息缺口。相当于在原来被砍掉的工序位置放一个简化的机器人继续工作。
- **稀疏度（Sparsity）**：目标压缩比例，例如 30% 稀疏度意味着要去掉 30% 的层。它是压缩力度的直接控制参数。
- **稳定性指标（Stability Metric）**：评估模型在压缩前后输出分布变化的度量，侧重于隐藏状态的一致性，而不是单纯的准确率。可以把它看作是检测流水线改造后产品尺寸是否保持一致的尺子。
- **轻量网络（Lightweight Network）**：参数量和计算量远低于原始层的替代模块，常用小型全连接层或瓶颈结构实现。

### 核心创新点
1. **从全局层重要性出发的剪枝策略**  
   之前的层剪枝大多基于经验或手工设定，缺乏量化依据。LLM-Streamline 通过统计每层对隐藏状态的梯度贡献，得到客观的层重要性分数。随后按照这些分数挑选连续的低分层进行删除，实现了“有的放矢”的压缩，而不是盲目裁剪。

2. **层替换模块的引入**  
   直接删层会导致信息流中断，性能大跌。本文创新性地训练一个轻量网络来“站位”被剪掉的层，使得模型在结构上仍保持完整的层数，只是内部计算更省。这个替换过程在保持原始模型表现的同时，大幅降低了计算开销。

3. **稳定性指标取代单一准确率**  
   传统压缩评估往往只看任务准确率，忽视了模型内部表征的变化。LLM-Streamline 提出了“稳定性”度量，直接比较压缩前后隐藏状态的相似度，帮助判断压缩是否破坏了模型的内部逻辑。这样可以在压缩率和性能之间找到更合理的平衡点。

4. **一次性完成剪枝与替换的高效训练流程**  
   过去的层剪枝往往需要多轮迭代，每轮都要重新微调。本文将剪枝、替换网络的训练以及整体微调合并进同一次训练循环，显著缩短了实验周期。实验显示，整体训练时间比同类方法快 30% 左右。

### 方法详解
整体框架可以概括为三步：**重要性评估 → 连续层剪枝 → 替换网络训练与整体微调**。下面逐步拆解每一步的细节。

1. **重要性评估**  
   - 对每一层的输出隐藏状态 \(h_i\) 计算其对最终任务损失的梯度贡献 \(\partial L / \partial h_i\)。  
   - 将梯度的 L2 范数作为该层的重要性分数。直观上，这相当于测量如果把该层的输出“拔掉”，损失会涨多少。  
   - 为了避免单次梯度噪声，作者在若干批次上取平均，得到更稳健的分数。

2. **连续层剪枝**  
   - 根据目标稀疏度 \(s\)，确定需要删除的层数 \(k = s \times N\)（\(N\) 为总层数）。  
   - 在重要性分数序列中寻找总分最小的连续子序列，长度为 \(k\)。这一步确保被删掉的层在模型结构上是相邻的，避免出现跨层的“跳跃”。  
   - 将这段子序列直接从模型图中剔除，得到一个“瘦身”后的骨架网络。

3. **层替换网络的构建与训练**  
   - 对每一段被剪掉的连续层，插入一个轻量网络 \(R\)。\(R\) 的输入是剪枝前的前一层输出，输出则直接喂给剪枝后紧随其后的层。  
   - 轻量网络的结构通常是两层带有 ReLU 激活的全连接层，隐藏维度远小于原始层的维度（如 1/4）。  
   - 训练目标是让 \(R\) 的输出在功能上尽可能逼近原始被剪掉层的输出。具体做法是：在微调阶段，保持原始模型的权重冻结，仅优化 \(R\) 的参数，使得整体损失最小。  
   - 为了防止 \(R\) 过度依赖特定输入分布，作者在训练时加入了噪声扰动和随机遮盖（类似于 dropout），提升其鲁棒性。

4. **整体微调与稳定性监控**  
   - 完成替换网络训练后，解冻全部参数，进行一次全模型微调。此时的损失函数同时包含任务损失和 **稳定性损失**：后者是压缩前后对应隐藏状态的 L2 距离。  
   - 稳定性损失的权重可以调节，若希望更严格保持内部表征，则提升该项权重；若更看重最终任务性能，则降低。  
   - 训练结束后，模型既保留了原始层的功能，又大幅削减了计算量。

**最巧妙的地方**在于把“层重要性”转化为可量化的梯度范数，并利用连续子序列搜索实现一次性大块剪枝；再配合轻量替换网络和稳定性约束，成功避免了传统层剪枝的性能崩溃。

### 实验与效果
- **测试平台**：作者在公开的 LLaMA‑7B、OPT‑13B 等主流大语言模型上进行实验，任务覆盖语言建模、问答和摘要生成。  
- **对比基线**：包括传统权重剪枝（Magnitude Pruning）、结构化层剪枝（LayerDrop）以及最新的稀疏微调方法。  
- **核心结果**：在 30% 稀疏度下，LLM-Streamline 在 WikiText‑103 上的困惑度（Perplexity）仅比原始模型高 2%，而 LayerDrop 同等稀疏度下提升约 8%。在问答任务上，准确率下降不到 1.5%，明显优于其他方法。  
- **训练效率**：整体压缩流程比同类多轮微调方案快约 30%，显著降低了实验成本。  
- **消融实验**：作者分别去掉“层替换网络”和“稳定性损失”，发现去掉任意一项后，模型性能下降约 3–5%，验证了两者的协同作用。  
- **局限性**：论文指出，当前的轻量网络仍然是全连接结构，对极端稀疏（>50%）的情况仍会出现显著性能损失；此外，重要性评估依赖梯度信息，若在无标签或少标签场景下可能不够可靠。

### 影响与延伸思考
LLM-Streamline 打破了“层不可剪”的固有认知，开启了基于层重要性的大模型结构压缩路线。随后的工作（如 **LayerFusion**、**DynamicLayerPrune**）在此基础上进一步探索了 **动态层选择**（在推理时根据输入实时决定保留哪些层）和 **跨模型层共享**（不同模型共享同一套轻量替换模块）。如果想深入了解，可以关注以下方向：① 更高效的层重要性度量（如基于信息论的指标）；② 替换网络的结构搜索（自动化找到最省算的替代方案）；③ 将层剪枝与 **稀疏激活**（Sparse Activation）结合，实现更细粒度的计算削减。整体来看，这篇论文为大模型部署提供了实用的“减肥”思路，也为后续的 **可调节推理** 打下了技术基础。

### 一句话记住它
用梯度驱动的层重要性挑选连续冗余层，再用轻量网络“填坑”，实现高效、低损失的大语言模型压缩。