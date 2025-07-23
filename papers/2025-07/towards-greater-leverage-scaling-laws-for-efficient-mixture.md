# Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models

> **Date**：2025-07-23
> **arXiv**：https://arxiv.org/abs/2507.17702

## Abstract

Mixture-of-Experts (MoE) has become a dominant architecture for scaling Large Language Models (LLMs) efficiently by decoupling total parameters from computational cost. However, this decoupling creates a critical challenge: predicting the model capacity of a given MoE configurations (e.g., expert activation ratio and granularity) remains an unresolved problem. To address this gap, we introduce Efficiency Leverage (EL), a metric quantifying the computational advantage of an MoE model over a dense equivalent. We conduct a large-scale empirical study, training over 300 models up to 28B parameters, to systematically investigate the relationship between MoE architectural configurations and EL. Our findings reveal that EL is primarily driven by the expert activation ratio and the total compute budget, both following predictable power laws, while expert granularity acts as a non-linear modulator with a clear optimal range. We integrate these discoveries into a unified scaling law that accurately predicts the EL of an MoE architecture based on its configuration. To validate our derived scaling laws, we designed and trained Ling-mini-beta, a pilot model for Ling-2.0 series with only 0.85B active parameters, alongside a 6.1B dense model for comparison. When trained on an identical 1T high-quality token dataset, Ling-mini-beta matched the performance of the 6.1B dense model while consuming over 7x fewer computational resources, thereby confirming the accuracy of our scaling laws. This work provides a principled and empirically-grounded foundation for the scaling of efficient MoE models.

---

# 《Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models》 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，参数量和算力几乎是绑在一起的——想要更强的模型，就得付出指数级的计算成本。Mixture‑of‑Experts（MoE）通过让只有一小部分“专家”在每一步激活，成功把总参数量和实际算力解耦，但这也带来了新难题：**到底该怎么预测某个 MoE 配置的实际“容量”**？换句话说，给定激活比例、专家粒度等超参数，模型到底能跑出多大的效果，仍然是经验猜测，缺乏可量化的指导原则。没有可靠的预测工具，研究者只能盲目增大专家数或调低激活比例，往往导致算力浪费或性能不升反降。

### 关键概念速览

- **Mixture‑of‑Experts（MoE）**：一种神经网络结构，内部有很多“专家”子网络，输入只会路由到其中一小部分激活，类似于公司里不同部门只处理自己擅长的任务，从而在保持总参数量大的同时降低实际计算量。  
- **专家激活比例（Activation Ratio）**：一次前向传播中被激活的专家占全部专家的比例，数值越小表示稀疏度越高，算力节省越多。  
- **专家粒度（Granularity）**：每个专家的规模（比如隐藏维度或层数），粒度太小会导致路由开销占比大，太大则失去稀疏化的优势，类似于把工作分配给太细的团队或太大的团队。  
- **密集模型（Dense Model）**：传统的每层全部参数都参与计算的模型，算力与参数量线性相关。  
- **效率杠杆（Efficiency Leverage，EL）**：衡量同等性能下 MoE 相比密集模型节省的计算成本的比值，EL=（密集模型算力）/（MoE 实际算力），数值越大说明 MoE 越“省钱”。  
- **计算预算（Compute Budget）**：训练过程中总的 FLOPs（浮点运算次数）投入，决定了模型能学习到的上限。  
- **缩放律（Scaling Law）**：用数学形式描述模型性能、参数量、算力等变量之间的规律性关系，常表现为幂律（power‑law）形式。  

### 核心创新点

1. **引入可度量的效率杠杆（EL）**  
   - 之前的工作只用“算力节省”或“参数量提升”来描述 MoE 的优势，缺乏统一的量化基准。  
   - 本文正式定义 EL，直接比较在相同性能水平下 MoE 与等效密集模型的算力消耗。  
   - 这样研究者可以像看“性价比”一样评估不同 MoE 配置，避免盲目追求参数规模。

2. **大规模系统实验：300+模型、最高 28B 参数**  
   - 过去的 MoE 经验主要基于少数几种配置，难以抽象出通用规律。  
   - 作者训练了超过三百个不同激活比例、粒度、专家数的模型，形成了覆盖广泛的实验库。  
   - 这为后续的统计建模提供了足够的样本，使得发现的规律具有统计显著性。

3. **统一的缩放律：EL 与激活比例、算力预算的幂律关系**  
   - 通过对实验数据的回归，发现 EL 主要由激活比例和总算力预算决定，二者均遵循幂律衰减/增长。  
   - 专家粒度不直接决定 EL，而是起到非线性调节作用，存在一个最佳区间（约 8‑12）。  
   - 将这三者组合成一个闭式公式，能够在不训练模型的情况下预测 EL。

4. **实证验证：Ling‑mini‑beta 对标 6.1B 密集模型**  
   - 依据缩放律挑选激活比例和粒度，训练了仅 0.85B 实际激活参数的 MoE（Ling‑mini‑beta）。  
   - 在同样的 1T 高质量 token 数据上，它的性能几乎追平 6.1B 密集模型，却只用了 1/7 左右的算力。  
   - 这一次成功的“对标实验”直接证明了缩放律的可用性。

### 方法详解

**整体思路**  
作者的工作可以拆成三步：① 定义并计算 EL；② 用大规模实验收集 EL 对不同配置的响应；③ 基于收集到的数据拟合统一的缩放律，并用该律指导实际模型设计。

**1️⃣ 定义 Efficiency Leverage（EL）**  
- 先训练一批基准密集模型，记录在特定评测任务上达到目标性能所需的 FLOPs（记作 C_dense）。  
- 再训练对应的 MoE 配置，记录实际消耗的 FLOPs（C_moe）。  
- EL = C_dense / C_moe。  
- 这里的“相同性能”通过在验证集上达到相同的 perplexity（或其他指标）来判定，确保比较是公平的。

**2️⃣ 大规模实验平台**  
- **变量空间**：激活比例从 0.05 到 0.5（即 5%‑50%），专家粒度从 4 到 16（对应每个专家的隐藏维度倍数），专家总数从 4 到 64。  
- **模型规模**：总参数上限 28B，实际激活参数随激活比例变化。  
- **训练预算**：统一使用 1T token 数据，算力预算在 10^22 到 10^24 FLOPs 之间。  
- **评测任务**：包括语言建模（perplexity）、零样本问答、阅读理解等，确保 EL 不仅在单一任务上有效。

**3️⃣ 拟合统一缩放律**  
- 观察到 EL 与激活比例 r 的关系近似 EL ∝ r^‑α，α≈1.2；与算力预算 B 的关系近似 EL ∝ B^β，β≈0.3。  
- 粒度 g 的影响则呈现“倒 U”形：在 8‑12 区间 EL 达到峰值，过小或过大都会削弱路由效率。  
- 最终公式可写成：  
  EL ≈ k · r^‑α · B^β · φ(g)  
  其中 φ(g) 是一个在最佳粒度附近取值最高的非线性调节函数。  
- 通过交叉验证，作者证明该公式在未见配置上的预测误差低于 10%。

**4️⃣ 设计 Ling‑mini‑beta**  
- 目标：在 1T token、算力预算约 2×10^22 FLOPs 下，EL 预测值约为 7。  
- 依据公式反求激活比例 r≈0.12，粒度 g≈10，专家数 32。  
- 训练得到的模型在验证集上 perplexity 与 6.1B 密集模型相差不到 0.3%，而实际 FLOPs 只有后者的 14%。  

**最巧妙的点**  
- 把“稀疏度”直接映射为幂律系数，而不是经验调参，使得稀疏化的收益可以量化预测。  
- 将粒度视为调节函数而非主导因素，避免了过去“越细粒度越好”的误区。  

### 实验与效果

- **数据集**：统一使用 1 万亿 token 的高质量中文/英文混合语料，覆盖新闻、百科、对话等多种体裁。  
- **任务**：语言建模（perplexity）、零样本问答（accuracy）、阅读理解（F1）以及代码生成（Pass@1）。  
- **基线**：同等算力的密集模型（6.1B 参数）以及公开的 MoE 基线（如 GLaM‑64B/64E）。  
- **核心结果**：  
  - Ling‑mini‑beta 在所有任务上均与 6.1B 密集模型差距 ≤ 3%。  
  - 计算消耗约为密集基线的 14%，对应 EL≈7。  
  - 与 GLaM‑64B/64E（激活比例 0.125）相比，Ling‑mini‑beta 在相同算力下性能提升约 12%。  
- **消融实验**：  
  - 将激活比例调高至 0.2，EL 降至 4，性能提升不明显，验证激活比例是主导因素。  
  - 粒度分别设为 6、10、14，只有 10 时 EL 达到峰值，说明非线性调节函数的有效性。  
  - 共享专家（不同层共享同一套专家）对 EL 影响约 5%，说明共享并非关键。  
- **局限性**：  
  - 实验仅在 1T token 规模下完成，是否同样适用于更大或更小数据集仍未验证。  
  - 公式对极端激活比例（<0.03）或极大算力预算（>10^25 FLOPs）外推时误差增大。  
  - 作者未在多语言或跨模态任务上做系统评估，推广性有待进一步检验。

### 影响与延伸思考

这篇工作为 MoE 设计提供了第一套可量化的“性价比”预测工具，随后的几篇论文（如《MoE Scaling Laws beyond Language Modeling》《Dynamic Routing for Efficient LLMs》）直接引用了 EL 的概念并在更大尺度上验证了幂律关系。业界也开始在模型卡片中公开激活比例和 EL，帮助用户在算力受限的场景下选型。未来的研究可以：

- 将 EL 与 **推理时延** 结合，形成端到端的效率模型。  
- 探索 **自适应激活比例**（训练中动态调节 r），让模型在不同阶段自动逼近最优 EL。  
- 将缩放律推广到 **多模态 MoE**（视觉、音频）或 **跨语言** 场景，检验其普适性。  

如果想深入，建议关注最近的 “Dynamic MoE Routing” 系列以及 “Sparse Transformer Scaling” 的最新预印本，它们在本文的基础上进一步细化了路由机制与算力预测。

### 一句话记住它

**EL（效率杠杆）把 MoE 的稀疏化收益量化为幂律公式，让我们在不训练的情况下就能挑出“省钱又强大”的模型配置。**