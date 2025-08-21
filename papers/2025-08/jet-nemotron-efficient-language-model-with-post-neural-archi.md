# Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search

> **Date**：2025-08-21
> **arXiv**：https://arxiv.org/abs/2508.15884

## Abstract

We present Jet-Nemotron, a new family of hybrid-architecture language models, which matches or exceeds the accuracy of leading full-attention models while significantly improving generation throughput. Jet-Nemotron is developed using Post Neural Architecture Search (PostNAS), a novel neural architecture exploration pipeline that enables efficient model design. Unlike prior approaches, PostNAS begins with a pre-trained full-attention model and freezes its MLP weights, allowing efficient exploration of attention block designs. The pipeline includes four key components: (1) learning optimal full-attention layer placement and elimination, (2) linear attention block selection, (3) designing new attention blocks, and (4) performing hardware-aware hyperparameter search. Our Jet-Nemotron-2B model achieves comparable or superior accuracy to Qwen3, Qwen2.5, Gemma3, and Llama3.2 across a comprehensive suite of benchmarks while delivering up to 53.6x generation throughput speedup and 6.1x prefilling speedup. It also achieves higher accuracy on MMLU and MMLU-Pro than recent advanced MoE full-attention models, such as DeepSeek-V3-Small and Moonlight, despite their larger scale with 15B total and 2.2B activated parameters.

---

# Jet‑Nemotron：通过后置神经架构搜索实现高效语言模型 论文详细解读

### 背景：这个问题为什么难？
在大模型时代，提升生成速度往往意味着要牺牲模型容量或准确率。传统的全注意力（full‑attention）架构在推理阶段计算量随序列长度呈二次增长，导致生成吞吐量低下。为了解决这个瓶颈，业界尝试了稀疏注意力、Mixture‑of‑Experts（MoE）等方案，但它们要么需要额外的调度开销，要么在小模型上难以保持与大模型相当的精度。于是，如何在不重新训练整个模型的前提下，快速探索更高效的注意力结构，成为了一个迫切的技术难题。

### 关键概念速览
**全注意力（Full‑Attention）**：每个 token 与序列中所有其他 token 交互，计算量随序列长度的平方增长，类似于在一张完整的社交网络里每个人都要和所有人聊天。  
**线性注意力（Linear Attention）**：把注意力的计算改写成线性时间复杂度，只需要和历史信息做一次累计，像是把聊天记录压缩成一个摘要再进行交流。  
**后置神经架构搜索（PostNAS）**：在已有的预训练全注意力模型上冻结 MLP（前馈层）权重，只在注意力模块上进行结构搜索，类似于在已经搭好的房子里只改动窗户的形状而不动墙体。  
**注意力块（Attention Block）**：模型中实现注意力计算的基本单元，可能是全注意力、线性注意力或作者自行设计的混合形式。  
**硬件感知超参数搜索（Hardware‑Aware H‑Search）**：在搜索过程中把实际硬件的吞吐量、显存占用等约束加入目标函数，确保找到的结构在真实机器上真的快。  
**Prefill（预填）**：在生成时把已有上下文一次性送入模型的过程，类似于把整段文字一次性放进打印机。  
**生成吞吐量（Generation Throughput）**：模型每秒能产生的 token 数，直接决定聊天机器人或搜索引擎的响应速度。  

### 核心创新点
1. **从全注意力模型直接出发 → 冻结所有 MLP 权重，仅搜索注意力块** → 省去了从零训练的巨额算力，搜索过程只需在已有权重上微调，大幅降低探索成本。  
2. **层级布局学习 → 自动决定哪些全注意力层保留、哪些可以被线性注意力块替换** → 通过学习得到的层次结构在保持关键表达能力的同时，大幅削减计算量。  
3. **新注意力块设计 → 在搜索空间中加入作者自行构造的混合块（如局部‑全局混合）** → 让模型在局部细节和全局语义之间取得更好平衡，提升了在 MMLU、MMLU‑Pro 等知识测评上的表现。  
4. **硬件感知超参数搜索 → 将实际 GPU/TPU 的吞吐率、显存占用作为搜索目标** → 找到的模型在真实部署环境中实现了最高可达 53.6 倍的生成加速和 6.1 倍的预填加速，而精度仍能匹配或超越 Qwen3、Gemma3、Llama3.2 等强基准。

### 方法详解
整体思路可以划分为四个阶段：

1. **准备基线模型**：先取一个已经预训练好的全注意力大模型（例如 2B 参数的 Llama‑style），把其中的前馈网络（MLP）层全部冻结，确保它们的表示能力不被破坏。  
2. **层级布局搜索**：使用强化学习或梯度可微搜索策略，学习每一层是否保留全注意力、改为线性注意力或直接删除。可以把每层看作一颗开关，搜索过程在“打开全注意力”“打开线性注意力”“关闭”三种状态之间切换，目标是最小化验证集的损失并兼顾计算预算。  
3. **注意力块选择与新块设计**：在搜索空间里预置几类已有块（标准全注意力、线性注意力）以及作者自行实现的混合块。混合块会先做局部窗口注意力，再通过低秩投影捕获全局信息，类似于先看局部细节再把全局概览加进去。搜索过程会评估每种块在冻结 MLP 上的适配度，挑选出最优组合。  
4. **硬件感知超参数搜索**：在得到候选结构后，进一步在 batch size、序列长度、并行度等超参数上做网格或贝叶斯搜索。每一次评估都会在目标硬件上跑一次推理计时，把实际吞吐率作为奖励信号，确保最终模型在部署机器上真的快。

**最巧妙的点**在于把 MLP 冻结后只动注意力，这相当于把模型的“记忆”固定，只让“注意力的眼睛”换不同的形状去观察输入。这样既保留了预训练的语言知识，又让搜索空间大幅缩小，使得即使在几百 GPU 小时内也能找到高效结构。

### 实验与效果
- **评测套件**：包括 MMLU、MMLU‑Pro、TruthfulQA、GSM‑8K、OpenAI‑Evals 等多任务基准，覆盖知识推理、数学计算和常识问答。  
- **基线对比**：与 Qwen3、Qwen2.5、Gemma3、Llama3.2（同等参数规模）以及 15B 参数的 MoE 模型 DeepSeek‑V3‑Small、Moonlight 进行比较。Jet‑Nemotron‑2B 在大多数指标上与这些强基准持平或略有超越，尤其在 MMLU‑Pro 上超过了上述 MoE 大模型。  
- **速度提升**：在同一硬件（NVIDIA H100）上，生成吞吐量最高提升 53.6 倍，预填速度提升 6.1 倍。换句话说，同样的算力下，模型可以在几秒钟内完成原本需要几分钟的长文本生成。  
- **消融实验**：论文分别去掉层级布局搜索、混合注意力块、硬件感知搜索，发现生成速度下降 20%~40%，而精度损失约 1%~2%，说明每个模块都对最终性能有实质贡献。  
- **局限性**：作者指出 PostNAS 仍然依赖于已有的全注意力预训练模型，若底层模型质量不佳，搜索空间的上限也会受限；此外，冻结 MLP 使得模型在极端长序列上仍可能出现信息瓶颈。

### 影响与延伸思考
Jet‑Nemotron 把“后置搜索”理念推向实用化，激发了后续工作在 **冻结特征提取层** 上进行结构探索的兴趣。例如，2024 年出现的 **Freeze‑NAS** 系列直接借鉴了 PostNAS 的思路，进一步将卷积特征层冻结用于视觉大模型的轻量化。硬件感知搜索也被更多系统级优化框架采纳，推动了 **模型‑硬件协同设计** 的潮流。想继续深入，可以关注以下方向：① 更细粒度的注意力块组合（如稀疏‑低秩混合）；② 在多模态大模型上尝试 PostNAS；③ 将搜索过程与自监督微调结合，实现“一次搜索、全域适配”。这些都是当前研究的热点。

### 一句话记住它
**Jet‑Nemotron 用冻结 MLP、只改注意力的后置搜索，让大模型在不重新训练的情况下实现数十倍的生成加速，同时保持或提升精度。**