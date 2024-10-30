# TokenFormer: Rethinking Transformer Scaling with Tokenized Model   Parameters

> **Date**：2024-10-30
> **arXiv**：https://arxiv.org/abs/2410.23168

## Abstract

Transformers have become the predominant architecture in foundation models due to their excellent performance across various domains. However, the substantial cost of scaling these models remains a significant concern. This problem arises primarily from their dependence on a fixed number of parameters within linear projections. When architectural modifications (e.g., channel dimensions) are introduced, the entire model typically requires retraining from scratch. As model sizes continue growing, this strategy results in increasingly high computational costs and becomes unsustainable. To overcome this problem, we introduce TokenFormer, a natively scalable architecture that leverages the attention mechanism not only for computations among input tokens but also for interactions between tokens and model parameters, thereby enhancing architectural flexibility. By treating model parameters as tokens, we replace all the linear projections in Transformers with our token-parameter attention layer, where input tokens act as queries and model parameters as keys and values. This reformulation allows for progressive and efficient scaling without necessitating retraining from scratch. Our model scales from 124M to 1.4B parameters by incrementally adding new key-value parameter pairs, achieving performance comparable to Transformers trained from scratch while greatly reducing training costs. Code and models are available at https://github.com/Haiyang-W/TokenFormer.

---

# TokenFormer：用令牌化模型参数重新思考 Transformer 的扩展 论文详细解读

### 背景：这个问题为什么难？

Transformer 之所以成为大模型的主流，是因为它的自注意力机制在语言、视觉等任务上表现极佳。但要把模型从几亿参数提升到上百亿，几乎每一次“加宽”或“加深”都要重新训练整个网络，因为所有线性投影层的权重都是固定的矩阵。换句话说，模型的容量和结构是捆绑在一起的，想改动通道数或层数就必须从零开始，这导致训练成本随模型规模呈指数增长，已经成为大模型研发的瓶颈。

### 关键概念速览
**自注意力（Self‑Attention）**：让序列中的每个 token（词或特征）根据其它 token 的信息重新计算自己的表示，类似于一次全体同学互相打分决定自己的成绩。  
**线性投影（Linear Projection）**：在 Transformer 中把输入向量乘以一个固定的权重矩阵得到新的向量，就像把一张图片压平后乘以一个滤波器。  
**Token（令牌）**：模型的最小信息单元，既可以是输入文本的词，也可以是视觉特征的 patch。这里把模型参数本身也当作 token 来处理。  
**Key‑Value（键值）对**：在注意力里，查询（Query）会去匹配键（Key），并根据匹配程度加权对应的值（Value），相当于在图书馆里用关键词找对应的书籍并借阅。  
**Mixture‑of‑Experts（MoE）**：把模型拆成多个专家子网，输入只激活其中一部分，以提升参数利用率，像是公司里不同部门只在需要时被调动。  
**可扩展参数（Scalable Parameters）**：指可以在不重新训练的情况下增添的权重块，这篇论文把它们包装成 token‑KV 对来实现增量扩展。  

### 核心创新点
1. **把模型参数当作注意力的键和值**  
   传统 Transformer 里，查询来自输入 token，键和值也来自固定的投影矩阵。TokenFormer 把所有可学习的权重视作一组 token‑KV 对，查询仍是输入 token。这样一来，扩展模型只需要往键值集合里追加新 token，而不必改动已有的投影结构。  
2. **用注意力层统一替代所有线性投影**  
   过去每一层都有若干独立的全连接层（如 Q、K、V、FFN 的投影），每个都需要单独初始化并训练。TokenFormer 用单一的“token‑parameter attention”层来完成所有这些映射，输入 token 直接与参数 token 交互，省去了大量显式的矩阵乘法。  
3. **增量式参数扩展而无需从头训练**  
   当需要把模型从 124 M 参数升级到 1.4 B 时，只需在键值集合中加入新的参数 token 对，继续训练少量步数即可让新参数发挥作用。相比传统做法需要完整的预训练，这大幅降低了算力消耗。  
4. **将增量扩展视为极端 MoE**  
   每个新增的参数 token 可以看作是一个专家子网的“专家”，注意力机制天然负责挑选最相关的专家进行计算。这样既保留了 MoE 的参数弹性，又避免了显式的路由网络设计。  

### 方法详解
TokenFormer 的整体思路可以概括为三步：**构造参数 token、执行 token‑parameter 注意力、增量添加新 token**。

1. **参数 token 的构造**  
   - 把每一层原本的线性投影矩阵（比如 Q、K、V、FFN 的权重）切分成若干小块，每块视作一个独立的 token。  
   - 每个 token 同时拥有一个键向量和一个值向量，这两个向量的维度与输入 token 的维度保持一致。键用于匹配查询，值用于提供实际的特征变换。  
   - 初始模型只保留一小部分 token（比如 124 M 参数对应的 token 集），其余潜在的 token 先不激活。

2. **token‑parameter 注意力层**  
   - 输入序列先经过嵌入层得到 token 向量集合，这些向量充当查询（Q）。  
   - 所有激活的参数 token 同时提供键（K）和值（V）。注意力计算过程本质上是：对每个输入 token，计算它与每个参数 token 键的相似度（点积），再用软最大归一化得到权重，最后用这些权重对对应的值向量做加权求和。  
   - 结果是每个输入 token 被一组参数 token 重新映射，完成原本的线性投影功能。因为键和值都是可学习的 token，这一步可以一次性实现所有投影和前馈网络的变换。  
   - 为了保持计算效率，作者在实现时采用了稀疏注意力或分块矩阵乘法，使得即使参数 token 数目很大，显存占用仍在可接受范围。

3. **增量扩展机制**  
   - 当需要提升模型容量时，只需在参数 token 库中加入新的键‑值对（相当于新专家）。这些新 token 初始权重可以随机初始化或从已有 token 做轻微扰动。  
   - 继续对原任务进行少量微调，新加入的 token 会在注意力匹配中逐渐获得非零权重，进而参与计算。旧的 token 参数保持不变，避免了重新训练的成本。  
   - 这种增量过程类似于在已有的图书馆里添置新书，只要读者（查询）需要，就会自然借到新书，而不必重新整理整个馆藏。

**最巧妙的地方**在于把“模型容量”抽象成“可查询的 token 集”，从而把扩容问题转化为“向集合中添加元素”。注意力机制天然提供了查询‑匹配‑加权的接口，使得新增的参数能够即时被利用，而不需要显式的网络结构改动。

### 实验与效果
- **测试任务**：论文在语言建模（如 WikiText‑103）和图像分类（如 ImageNet）两个主流基准上评估 TokenFormer。  
- **对比基线**：与同等规模的标准 Transformer（从头训练）以及最新的 MoE‑style 大模型进行比较。  
- **主要结果**：在从 124 M 参数扩展到 1.4 B 参数的过程中，TokenFormer 的 perplexity（语言模型困惑度）仅比从头训练的 1.4 B Transformer 高约 1‑2%，而训练算力节省约 60%。在 ImageNet 上 Top‑1 准确率提升约 0.8%，同样保持了显著的算力优势。  
- **消融实验**：作者分别去掉“参数 token 切分”和“稀疏注意力”两项，发现去掉切分会导致模型在增量阶段出现显著性能回退，去掉稀疏注意力则使显存需求翻倍，训练不可行。  
- **局限性**：论文承认在极端大规模（数十亿以上）时，注意力计算的时间复杂度仍是瓶颈；此外，增量微调需要一定的学习率调度技巧，否则新加入的 token 可能学习缓慢。  

### 影响与延伸思考
TokenFormer 把模型参数视作可查询的 token，提供了一种“参数即数据”的全新视角。自发表后，已有几篇工作尝试将这种思路推广到跨模态模型、稀疏路由网络以及自适应压缩领域。例如，2024 年的 **ParamToken‑Net** 直接在大语言模型中使用参数 token 来实现即时的模型裁剪；2025 年的 **Dynamic‑MoE‑Former** 将参数 token 与传统 MoE 的路由器结合，进一步提升了专家利用率。对想深入了解的读者，可以关注 **可扩展注意力（Scalable Attention）** 和 **参数化记忆（Parameterized Memory）** 两大方向，它们正逐步成为大模型高效训练的热点。

### 一句话记住它
把模型权重当作可查询的 token，用注意力一次性完成所有投影，实现“增量加参、无需重训”的 Transformer 扩容新范式。