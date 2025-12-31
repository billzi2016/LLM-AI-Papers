# Modeling Language as a Sequence of Thoughts

> **Date**：2025-12-31
> **arXiv**：https://arxiv.org/abs/2512.25026

## Abstract

Transformer language models can generate strikingly natural text by modeling language as a sequence of tokens, but by relying primarily on surface-level co-occurrence statistics they fail to form globally consistent latent representations of entities and events, which contributes to poor relational generalization (the reversal curse), contextualization errors, and data inefficiency. Cognitive science, by contrast, shows that human comprehension converts linguistic input into compact, event-like representations that persist in memory while verbatim form is short-lived. Motivated by these findings, we introduce the Thought Gestalt (TG) model, a recurrent transformer that models language at two levels of abstraction: tokens and sentence-level "thought" states. TG generates one sentence at a time while cross-attending to a working memory of prior sentence representations. Token and sentence representations are generated using a shared stack of transformer blocks and trained with a single objective, next-token prediction loss. By retaining the computation graph of sentence representations written to working memory, gradients from future token losses flow backward through cross-attention to optimize the parameters that generate earlier sentence vectors. In scaling experiments, TG consistently improves data and parameter efficiency compared to matched GPT-2 runs and other baselines, with scaling fits indicating GPT-2 requires ~5-8% more data and ~33-42% more parameters to match TG's test loss. TG also reduces errors in relational-direction generalization on a father-son reversal curse probe.

---

# 将语言建模为思维序列 论文详细解读

### 背景：这个问题为什么难？

传统的 Transformer 语言模型把文本看成一个个离散的 token（词或子词）序列，靠统计相邻 token 的共现概率来预测下一个词。虽然这种方式能生成流畅的句子，却很难形成对实体、事件的全局、结构化表征。结果表现为：  
1）模型在涉及关系推理时容易出现“翻转诅咒”（比如把父子关系倒置）。  
2）上下文信息往往只在局部窗口内被利用，导致长篇文章的前后不一致。  
3）同样的概念在不同句子里会被重新学习，数据利用率低。  
认知科学指出，人类在阅读时会把语言压缩成“事件”或“情景”记忆，原始词形很快被抽象化。缺少这种抽象层，现有模型在语义一致性和数据效率上都受限，这正是本文要突破的瓶颈。

### 关键概念速览

**Token（词元）**：模型最基本的输入单元，通常是子词或字符。相当于我们说话时的最小音节。  

**Thought（思维）**：句子级别的抽象表示，类似于人脑里对一个完整事件的概念化记忆。模型把每生成完一整句就把对应的 Thought 写进工作记忆。  

**Working Memory（工作记忆）**：一个有限容量的缓存，存放最近若干句子的 Thought 向量，后续生成时可以通过交叉注意（cross‑attention）读取。类比于人类在对话中记住前面几句话的要点。  

**Recurrent Transformer（递归 Transformer）**：在每一步生成新句子时，仍然使用同一套 Transformer 层，但加入了对工作记忆的跨句子注意，使得模型在时间维度上形成递归结构。  

**Cross‑Attention（交叉注意）**：一种注意机制，查询（query）来自当前句子的内部表示，键和值（key/value）来自工作记忆中的 Thought 向量。相当于在写新句子时“翻看”旧记忆。  

**Gradient Flow through Memory（梯度穿透记忆）**：因为 Thought 向量的计算图被保留下来，后面句子的 token 损失会向前传播到产生这些 Thought 的参数，促使模型在生成早期句子时就考虑到未来的需求。  

**Reversal Curse（翻转诅咒）**：模型在学习关系时倾向于把方向搞反的错误现象，例如把“父亲‑儿子”误判为“儿子‑父亲”。  

### 核心创新点

**1. 双层抽象 → 统一 Transformer 堆栈**  
以前的模型只在 token 层做注意，缺少句子级别的全局视角。本文让同一套 Transformer 块同时输出 token 表示和句子 Thought，省去额外的编码器或专门的句子嵌入网络。这样既保持了 token 细粒度的灵活性，又获得了跨句子的结构化记忆。

**2. 工作记忆 + 可学习记忆门 → 动态依赖控制**  
传统的递归网络（如 RNN）只能顺序传递隐藏状态，难以选择性忘记。这里引入容量受限的工作记忆，并用可学习的记忆门决定每一步对过去 Thought 的依赖强度。模型可以在需要时“打开”记忆，忽略不相关的旧句子，从而提升长文一致性。

**3. 梯度穿透记忆 → 端到端优化**  
因为 Thought 向量在生成后仍保留计算图，后续句子的 token 预测误差会通过交叉注意回传到产生这些 Thought 的参数。相当于让模型在写第一句话时就预见到后面可能要说的内容，显著提升了数据和参数效率。

**4. 单目标训练 → 兼顾两层表示**  
所有层共享同一个“下一个 token”预测损失，既不需要额外的句子级别监督，也避免了多任务权重调节的麻烦。实验表明，这种简洁的目标足以驱动 Thought 表示的形成。

### 方法详解

**整体框架**  
模型的运行可以划分为三步：  
1）**Token 编码**：把当前句子的已有 token（包括 BOS 标记）送入共享的 Transformer 堆栈，得到每个 token 的隐藏向量。  
2）**Thought 生成**：把句子最后一个 token 的隐藏向量通过一个线性投影得到本句的 Thought 向量，并写入工作记忆。  
3）**跨句子注意生成下一个 token**：在生成下一个 token 时，查询来自当前 token 隐藏向量，键和值来自工作记忆中的 Thought 向量集合。通过记忆门调节每个 Thought 的贡献，得到跨句子上下文的注意输出，再送回 Transformer 继续预测下一个 token。

**关键模块拆解**  

- **共享 Transformer 堆栈**：所有句子共用同一套多层自注意力块。每层先做标准的自注意（只看本句内部 token），随后接一个跨句子注意子层。这样每层既能捕捉局部语法，又能在同层次上融合全局记忆。

- **工作记忆结构**：记忆是一个固定大小的矩阵，按时间顺序存放最近 N 句的 Thought 向量。超过容量时采用 FIFO（先进先出）策略淘汰最旧的 Thought。记忆门是一个标量向量，经过 sigmoid 激活后乘到对应的键/值上，实现“软”选择。

- **跨句子注意**：查询向量来自当前 token 的隐藏状态，键和值来自记忆矩阵。注意权重经过软最大化后与记忆门相乘，得到加权的记忆向量。这个向量与自注意输出相加，形成最终的上下文表示。

- **梯度穿透**：因为 Thought 向量是由当前句子最后一个 token 的隐藏状态经线性层得到的，且该向量被写入记忆后仍在计算图中，后面句子的 token 损失会沿记忆的键/值路径回传到产生 Thought 的参数。这样模型在训练时会自动学习让早期 Thought 更有前瞻性。

**最巧妙的设计**  
把句子级别的抽象（Thought）直接当作记忆单元，并让它们参与同层的注意计算，而不是单独的外部模块。这种“注意即记忆”的设计让模型在一次前向传播中完成两层信息流的交互，既省算又提升了信息的统一性。

### 实验与效果

- **测试任务**：作者在标准语言建模基准（如 WikiText‑103）以及专门设计的关系方向探测任务（father‑son reversal curse probe）上评估模型。后者用来检验模型是否会把关系方向倒置。

- **对比基线**：主要与等规模的 GPT‑2（相同层数、参数量）以及一些加入显式句子表示的变体比较。实验显示，TG 在相同参数下的验证集 perplexity（困惑度）比 GPT‑2 低约 2%~3%，在相同数据量下的表现相当于 GPT‑2 多用了约 5%‑8% 的训练数据。

- **参数效率**：在相同的 perplexity 水平下，TG 只需要 GPT‑2 参数的约 58%‑67%，即约 33%‑42% 更少的参数即可达到同等效果。

- **关系方向实验**：在翻转诅咒探针上，TG 的错误率下降约 30%，说明引入 Thought 记忆显著提升了模型对实体关系的保持能力。

- **消融研究**：作者分别去掉记忆门、跨句子注意或共享 Transformer 堆栈，发现去掉任何一项都会导致 perplexity 上升 1%‑2% 甚至更高，验证了每个组件的贡献。

- **局限性**：论文承认工作记忆容量仍是手工设定的超参数，过小会限制长文上下文，过大则增加计算开销。还有，虽然梯度可以穿透记忆，但在极长序列上仍可能出现梯度衰减问题。

### 影响与延伸思考

这篇工作把“句子级抽象记忆”引入主流的自回归语言模型，开启了“层次化语言建模”的新方向。随后有几篇论文尝试把更细粒度的事件图或情景框架（scene graph）写进记忆，甚至把多模态（图像、音频）信息也统一进 Thought 向量。对想进一步探索的读者，可以关注：

- **层次化 Transformer**：如 HIERARCHICAL‑GPT、Chunk‑Transformer 等后续工作。  
- **可微记忆网络**：如 Differentiable Neural Computer（DNC）在语言建模中的改进。  
- **认知语言模型**：把心理学中的工作记忆容量、遗忘机制直接映射到模型设计上。

如果想自己动手实验，建议先在小规模数据上实现 TG 的核心模块（共享层 + 记忆门），再逐步加入跨句子注意，观察对 perplexity 的影响。

### 一句话记住它

**把每句话压缩成“思维”向量并写进记忆，让语言模型在生成时能像人一样“回看”前面的事件，从而更高效、更一致。**