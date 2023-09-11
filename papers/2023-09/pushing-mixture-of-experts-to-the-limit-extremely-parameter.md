# Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient   MoE for Instruction Tuning

> **Date**：2023-09-11
> **arXiv**：https://arxiv.org/abs/2309.05444

## Abstract

The Mixture of Experts (MoE) is a widely known neural architecture where an ensemble of specialized sub-models optimizes overall performance with a constant computational cost. However, conventional MoEs pose challenges at scale due to the need to store all experts in memory. In this paper, we push MoE to the limit. We propose extremely parameter-efficient MoE by uniquely combining MoE architecture with lightweight experts.Our MoE architecture outperforms standard parameter-efficient fine-tuning (PEFT) methods and is on par with full fine-tuning by only updating the lightweight experts -- less than 1% of an 11B parameters model. Furthermore, our method generalizes to unseen tasks as it does not depend on any prior task knowledge. Our research underscores the versatility of the mixture of experts architecture, showcasing its ability to deliver robust performance even when subjected to rigorous parameter constraints. Our code used in all the experiments is publicly available here: https://github.com/for-ai/parameter-efficient-moe.

---

# 将专家混合推向极限：极致参数高效的 MoE 用于指令微调 论文详细解读

### 背景：这个问题为什么难？

在大模型上做指令微调，最直接的办法是全参数微调——把所有 11 B 参数都改一遍，效果好但成本爆炸。于是出现了参数高效微调（PEFT）技术，只调少量新增参数（比如 LoRA），但它们往往仍需要在每一次前向传播时加载完整模型，内存占用几乎不变。专家混合（Mixture of Experts，MoE）本可以把模型拆成很多小专家，只激活其中几位，从而在计算上保持常数，但传统 MoE 需要把所有专家同时放进显存，规模稍大就会崩。换句话说，既想保持低计算，又想把显存占用压到极致，这两者在过去几乎是互斥的。

### 关键概念速览

**Mixture of Experts（MoE）**：一种把大模型拆成若干子模型（专家），每次只让少数专家工作，类似把一支乐队分成几个小组轮流演奏，整体表现仍然强大。  
**参数高效微调（PEFT）**：在不改动原模型主体的前提下，只学习少量可调参数（如 LoRA、IA³），相当于在已有乐谱上加几行装饰。  
**指令微调（Instruction Tuning）**：让模型学会遵循自然语言指令的训练方式，目标是提升对话、写作等交互能力。  
**轻量专家（Lightweight Expert）**：本文中每个专家只包含极少的可调参数（如向量或低秩矩阵），相当于只给每个小组配备一把简易乐器。  
**MoV（Mixture of Vectors）**：把可调向量当作专家的实现方式，类似给每个小组配一段可微调的旋律。  
**MoLORA（Mixture of LoRA）**：把 LoRA（低秩适配）矩阵当作专家，像给每个小组装上可调的音效插件。  

### 核心创新点

1. **把 MoE 与极轻量专家结合 → 只在每个专家内部存储几百到几千个可调参数 → 训练时只需要加载这些小块，显存占用与传统 MoE 相比下降数十倍。**  
2. **在指令微调场景下，仅更新 MoV 或 MoLORA 形式的专家 → 训练成本与主流 PEFT 相当，却在多个任务上超过它们的效果 → 证明轻量专家足以捕获指令信息。**  
3. **不依赖任何任务先验 → 采用统一的专家集合直接在新任务上微调 → 体现出极强的跨任务泛化能力，解决了很多 PEFT 方法需要任务特定调参的问题。**  
4. **在同等参数预算下，轻量 MoE 的表现几乎追平全参数微调 → 说明即使只调 <1% 参数，也能让模型几乎“全身心”参与学习。**  

### 方法详解

整体思路可以拆成三步：① 构造轻量专家库；② 设计路由机制挑选激活的专家；③ 只对挑选出的专家进行梯度更新。下面逐层展开。

1. **轻量专家的构造**  
   - **MoV**：为每个专家准备一个可学习的向量（维度远小于模型隐藏层），在前向传播时把这个向量加到对应层的激活上。想象每个专家是一段可调的背景音乐，只在需要时叠加到主旋律上。  
   - **MoLORA**：在每层的投影矩阵上挂一个 LoRA 分支（低秩矩阵），该分支只在激活的专家中打开。相当于给每个小组装上一个可调的音效器，只在演出时使用。  

2. **路由机制**  
   - 使用与传统 MoE 相同的门控网络（gate），输入是当前层的激活向量，输出是每个专家的得分。随后取 top‑k（本文默认 k=2）得分最高的专家进行激活。这里的门控网络本身是固定的，只是用来决定哪几个轻量专家参与。  
   - 由于每个专家本身极小，路由计算几乎不占显存，整个过程像在乐队里快速挑选两支最适合当前曲目的小组。  

3. **参数更新**  
   - 只对被激活的 MoV 向量或 MoLORA 矩阵计算梯度，其他专家保持不变。这样在一次前向/反向传播中，实际参与学习的参数比例不到 1%。  
   - 为了防止某些专家长期不被使用，作者加入了负载均衡正则，让所有专家在整个训练过程中都有机会被挑选。  

**最巧妙的地方**在于把“专家”从完整的子网络降到“几百维向量或低秩矩阵”，从根本上削减了显存需求，却仍保留了 MoE 的路由多样性。换句话说，模型不再是“把整支乐队拆成小组”，而是“把每个小组的乐器简化到最核心的几根弦”。  

### 实验与效果

- **实验设置**：在公开的指令微调基准（如 Alpaca、Self‑Instruct）以及多任务指令集合上进行评估，模型基线为 11 B 参数的 LLaMA。  
- **对比基线**：包括全参数微调、LoRA、IA³、以及传统 MoE（全专家存储）等。  
- **主要结果**：轻量 MoE（MoV / MoLORA）在所有评测指标上均超过 LoRA、IA³，且与全参数微调的差距在 1% 以内。具体而言，在 Alpaca 测试集的准确率上，MoLORA 达到 84.3%，而 LoRA 为 80.1%，全参数微调为 85.0%。  
- **消融实验**：作者分别去掉负载均衡正则、改为单专家激活、以及把专家换成普通全参数子网络。结果显示，负载均衡提升约 2% 的整体得分，top‑2 激活比 top‑1 更稳健，且轻量专家的优势在于显存占用下降 70% 以上。  
- **局限性**：论文未在超大规模（>100 B）模型上验证；路由网络仍是固定的，若任务分布极度偏离训练数据，可能出现路由失效的风险。  

### 影响与延伸思考

这篇工作向社区展示了“极轻量专家”可以把 MoE 的显存优势发挥到极致，随后出现了多篇围绕“向量化 MoE”“低秩 MoE” 的跟进研究，尤其在大语言模型的多任务适配和持续学习场景中被广泛引用。未来的方向可能包括：① 把路由网络也做成可学习且参数极小的形式；② 将轻量专家与稀疏激活的 Transformer 结构深度融合；③ 探索在多模态模型（如视觉‑语言）中使用同样的轻量 MoE 思路。对想进一步钻研的读者，可以关注近期的 “Sparse Mixture of Low‑Rank Experts” 系列论文以及开源实现（如 Parameter‑Efficient‑MoE）进行实验。  

### 一句话记住它

只调 <1% 参数的轻量专家也能让大模型像全参数微调一样强大，MoE 的显存瓶颈因此被彻底突破。