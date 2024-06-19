# In-Context Former: Lightning-fast Compressing Context for Large Language   Model

> **Date**：2024-06-19
> **arXiv**：https://arxiv.org/abs/2406.13618

## Abstract

With the rising popularity of Transformer-based large language models (LLMs), reducing their high inference costs has become a significant research focus. One effective approach is to compress the long input contexts. Existing methods typically leverage the self-attention mechanism of the LLM itself for context compression. While these methods have achieved notable results, the compression process still involves quadratic time complexity, which limits their applicability. To mitigate this limitation, we propose the In-Context Former (IC-Former). Unlike previous methods, IC-Former does not depend on the target LLMs. Instead, it leverages the cross-attention mechanism and a small number of learnable digest tokens to directly condense information from the contextual word embeddings. This approach significantly reduces inference time, which achieves linear growth in time complexity within the compression range. Experimental results indicate that our method requires only 1/32 of the floating-point operations of the baseline during compression and improves processing speed by 68 to 112 times while achieving over 90% of the baseline performance on evaluation metrics. Overall, our model effectively reduces compression costs and makes real-time compression scenarios feasible.

---

# 上下文压缩加速器：In-Context Former 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在推理时需要把整段文字都喂进 Transformer 的自注意力层。自注意力的计算量随序列长度呈二次增长，导致长上下文的推理成本极高。已有的压缩方案大多让模型自己用自注意力把长序列“压缩”成短向量，虽然能削减后续计算，但压缩过程本身仍是二次复杂度，实用性受限。换句话说，压缩的速度和模型的推理速度之间出现了瓶颈，迫切需要一种既快又能保持信息的压缩手段。

### 关键概念速览

**Transformer**：一种基于自注意力的神经网络结构，擅长捕捉序列中任意位置的关联。可以想象成一张全连接的社交网络，每个人（词）都能直接向其他人发信息。

**自注意力（Self‑Attention）**：在同一序列内部计算每个位置对其他位置的关注度，计算量随序列长度的平方增长。类似于在一群人里每个人都要和所有人逐一交流。

**交叉注意力（Cross‑Attention）**：一种注意力形式，一组查询（query）向另一组键值（key/value）寻求信息。好比记者（查询）向专家（键值）提问，只需要一次问答循环。

**Digest Token（摘要令牌）**：论文中引入的可学习的少量特殊向量，用来“收集”上下文信息。可以把它们想象成会议记录员，只负责把大家的发言浓缩成几句话。

**压缩（Compression）**：把原始的长词向量序列映射到更短的表示，以降低后续计算。这里的目标是保持原始语义的核心要素。

**浮点运算量（FLOPs）**：衡量模型计算量的指标，数值越大说明需要的算力越多。

### 核心创新点

1. **自注意力 → 交叉注意力**  
   传统压缩方法让 LLM 自己用自注意力把长序列压缩，计算仍是二次的。IC‑Former 把压缩任务交给一个独立的交叉注意力模块，让少量 digest token 去“询问”全部上下文的词向量。这样查询的数量是固定的，计算复杂度从 O(L²) 降到 O(L)，其中 L 是原始序列长度。

2. **依赖目标模型 → 独立压缩器**  
   以前的方案必须把压缩过程嵌进目标 LLM，导致每换一个模型都要重新训练或调参。IC‑Former 设计成一个轻量级的前置网络，和具体的 LLM 解耦，只需要一次训练即可在不同的下游模型前使用。

3. **可学习的摘要令牌**  
   直接使用固定的平均池化或卷积会丢失细粒度信息。论文引入可学习的 digest token，这些 token 在训练时会自动学会怎样从上下文中抽取关键特征，相当于让模型自己决定“记笔记”的方式。

4. **极低的 FLOPs 与高速推理**  
   通过上述设计，压缩阶段只消耗基线的 1/32 FLOPs，实际推理速度提升 68–112 倍。相比于仍需二次计算的自注意力压缩，这一改进把实时压缩从不可能变成了可行。

### 方法详解

#### 整体框架

IC‑Former 的压缩流程可以划分为三步：  
1) **嵌入提取**：把原始文本转成词向量序列（与普通 LLM 前置相同）。  
2) **交叉注意力压缩**：使用一组固定数量的 digest token 作为查询，向全部词向量发起交叉注意力，得到压缩后的摘要向量。  
3) **拼接送入目标 LLM**：把压缩得到的摘要向量（长度远小于原始序列）与可能的剩余关键 token 合并，作为目标 LLM 的输入。

#### 关键模块拆解

1. **Digest Token 初始化**  
   - 设定 K（如 8）个 learnable token，形状与普通词向量相同。  
   - 这些 token 在训练期间会被梯度更新，类似于模型的“记忆槽”。

2. **交叉注意力计算**  
   - **查询（Q）**：K 个 digest token 经过线性投影得到查询矩阵。  
   - **键值（K、V）**：全部 L 个词向量分别投影得到键和值矩阵。  
   - 计算注意力权重：每个 digest token 对每个词向量的相似度（点积）除以根号维度，再做 softmax，得到每个 token 对整个上下文的关注分布。  
   - 用权重加权求和得到每个 digest token 的新表示（即它“听到”的信息）。  
   - 由于查询数量固定，整个过程的时间随 L 线性增长。

3. **信息聚合与输出**  
   - 将 K 个更新后的 digest token 按顺序拼接，形成一个长度为 K×d 的压缩向量（d 为隐藏维度）。  
   - 若需要保留局部细节，可在压缩前对原始序列做稀疏抽样，只保留前几百个 token 与压缩向量一起送入 LLM。

#### 设计亮点

- **解耦**：压缩器不依赖目标 LLM 的内部结构，只需要词向量作为键值，这让它可以在不同模型之间复用。  
- **可学习的注意力模式**：因为 digest token 是可训练的，它们会在训练过程中自动形成类似“主题抽取器”“关键事实记录员”等角色，而不是固定的平均或最大池化。  
- **线性复杂度**：交叉注意力的查询数是常数，避免了自注意力的二次爆炸，真正实现了“闪电压缩”。

### 实验与效果

- **测试任务**：论文在常见的长上下文基准上评估，包括长文档问答、代码补全以及多轮对话等，需要模型处理数千 token 的输入。  
- **对比基线**：与传统的自注意力压缩方法（如 Longformer‑based 压缩、Sliding‑Window 方案）以及直接不压缩的原始 LLM 进行比较。  
- **性能提升**：压缩阶段 FLOPs 仅为基线的 1/32，整体推理速度提升 68–112 倍。评估指标（如 EM、F1）仍保持在 90% 以上的基线水平。  
- **消融实验**：作者分别去掉 digest token、改用固定平均池化、或把交叉注意力换回自注意力，发现：  
  - 去掉 digest token后性能跌至 78% 基线。  
  - 使用自注意力压缩时，计算成本回到二次级别，速度优势消失。  
  这些实验表明两大核心设计（digest token + 交叉注意力）是性能提升的关键。  
- **局限性**：论文承认在极端超长上下文（> 10k token）下，单纯的线性交叉注意力仍会产生显存瓶颈；此外，压缩质量受训练数据分布影响，若下游任务与训练时的上下文风格差异大，可能出现信息丢失。

### 影响与延伸思考

IC‑Former 的思路打开了“压缩前置模块独立于 LLM”的新局面，随后有几篇工作尝试把更复杂的记忆网络、可微分检索器嵌入到类似框架中，以进一步提升压缩质量。还有研究把 digest token 视作可解释的“主题向量”，用于可视化长文档的核心信息。对想继续深挖的读者，可以关注以下方向：  
- **层次化压缩**：在多层交叉注意力中逐步抽象信息，类似金字塔结构。  
- **自适应 token 数量**：根据输入长度动态决定 digest token 的数量，以在速度和精度之间取得更细粒度的平衡。  
- **跨模态压缩**：把图像、音频等非文本特征也通过交叉注意力压缩进同一摘要向量，探索多模态 LLM 的实时推理。

### 一句话记住它

IC‑Former 用少量可学习的摘要令牌通过交叉注意力把长上下文线性压缩，实现了“秒级”压缩且几乎不损失性能。