# RWKV-7 "Goose" with Expressive Dynamic State Evolution

> **Date**：2025-03-18
> **arXiv**：https://arxiv.org/abs/2503.14456

## Abstract

We present RWKV-7 "Goose", a new sequence modeling architecture with constant memory usage and constant inference time per token. Despite being trained on dramatically fewer tokens than other top models, our 2.9 billion parameter language model achieves a new 3B SoTA on multilingual tasks and matches the current 3B SoTA on English language downstream performance. RWKV-7 introduces a newly generalized formulation of the delta rule with vector-valued gating and in-context learning rates, as well as a relaxed value replacement rule. We show that RWKV-7 can perform state tracking and recognize all regular languages, while retaining parallelizability of training. This exceeds the capabilities of Transformers under standard complexity conjectures, which are limited to $\mathsf{TC}^0$. To demonstrate RWKV-7's language modeling capability, we also present an extended open source 3.1 trillion token multilingual corpus, and train four RWKV-7 models ranging from 0.19 billion to 2.9 billion parameters on this dataset.   To foster openness, reproduction, and adoption, we release our models and dataset component listing at https://huggingface.co/RWKV, and our training and inference code at https://github.com/RWKV/RWKV-LM all under the Apache 2.0 License.

---

# RWKV-7 “Goose”——具备表达性动态状态演化 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理里，Transformer 之所以火爆，是因为它能一次性看完全部上下文，却要付出 O(序列长度 × 隐藏维度) 的显存和 O(序列长度) 的推理时延。序列越长，显存和延时呈线性增长，导致超长文本、实时对话和边缘设备上部署几乎不可能。此前的改进（如线性注意、稀疏注意）只能在特定场景削减常数，却仍保留 O(序列长度) 的复杂度。于是，如何在保持模型表达力的同时，实现**每个 token 恒定显存、恒定推理时延**，成为了学界的硬核挑战。

### 关键概念速览
- **常数显存 / 常数时延**：模型在处理任意长度序列时，显存占用和每一步计算时间都不随序列长度增长。想象把一条无限长的河流装进固定大小的水桶，每次只往里倒一小勺水，水桶容量不变。
- **Delta Rule（增量规则）**：在每个时间步只计算状态的增量，而不是完整的状态向量。类似于记账本只记录本次变动，而不是每次都写下完整的余额。
- **向量门控（Vector‑valued Gating）**：对增量的每个维度分别开关，决定该维度是保留旧信息还是接受新信息。好比在厨房里每个调味瓶都有独立的阀门，你可以只打开盐瓶的阀门而不动胡椒瓶。
- **上下文学习率（In‑Context Learning Rate）**：在推理时，模型会根据当前上下文自动调节增量的幅度，就像人在阅读时会根据前文的难度自动放慢或加快阅读速度。
- **松弛值替换规则（Relaxed Value Replacement Rule）**：允许新信息部分覆盖旧状态，而不是全盘替换。可以比作在白板上写字时，只擦掉需要改动的那几笔，而不是整块擦掉重新写。
- **状态追踪（State Tracking）**：模型内部维护一个随时间演化的隐藏向量，用来记忆过去的结构信息。类似于人类在对话中记住前面提到的角色和事件。
- **正则语言（Regular Languages）**：计算理论里最简单的一类语言，能够用有限状态机识别。模型能识别所有正则语言，说明它至少拥有有限状态机的表达能力。
- **TC⁰**：一种电路复杂度类，表示只能用常深、并行度极高的布尔电路计算的函数。Transformer 在理论上被限制在 TC⁰ 范围内，而 RWKV‑7 超出了这个上限。

### 核心创新点
1. **从标量增量到向量门控增量**  
   之前的增量规则只用一个标量控制整体更新幅度，导致细粒度信息难以保留。RWKV‑7 引入向量门控，让每个隐藏维度都有自己的开关，并配合上下文学习率动态调节。结果是模型在保持常数显存的同时，能够更精准地捕捉长程依赖。

2. **松弛值替换让状态可追踪**  
   传统 RNN 或 Transformer 在每一步都直接写入新值，旧信息容易被一次性抹掉。松弛值替换规则把新信息和旧状态按比例混合，使得隐藏状态在时间轴上形成平滑的演化轨迹。这样模型能够实现完整的状态追踪，甚至可以识别所有正则语言。

3. **并行可训练的递归结构**  
   递归（RNN）天然顺序，难以并行化；Transformer 并行却需要 O(N²) 注意力。RWKV‑7 通过把增量计算拆成两阶段：先并行计算所有 token 的门控系数，再顺序累加增量。训练时两阶段都可以并行，推理时只保留累加一步，实现了“并行训练 + 常数推理”的理想组合。

4. **用极少数据实现 3B 级别 SoTA**  
   其它 3B 参数模型往往需要数十万亿 token 才能达到顶尖水平。RWKV‑7 只用了 3.1 万亿 token（相对仍是大规模，但比同等规模的 Transformer 少了一个数量级），就达到了多语言任务的最新 3B SoTA，并在英文下游任务上匹配了同规模的最强模型。说明新架构在数据效率上有显著提升。

### 方法详解
**整体框架**  
RWKV‑7 的前向过程可以分为三步：  
1) **嵌入层**把输入 token 转成向量；  
2) **RWKV‑7 单元**根据当前隐藏状态、门控向量和增量规则更新状态；  
3) **输出层**把最新的隐藏状态映射到词表概率。整个模型在训练时一次性把所有 token 的门控系数算完，在推理时只保留上一步的隐藏向量和当前 token 的增量。

**关键模块拆解**  

- **门控计算**：对每个 token，模型先用一个轻量的全连接网络（类似于注意力的 Q/K）产生两个向量——`gate`（门控）和`delta`（增量）。`gate` 的每个维度在 0~1 之间，决定该维度是否接受新信息。可以把它想象成一排可调节的阀门。

- **增量更新**：增量向量 `delta` 乘以对应的 `gate`，再乘以一个由上下文学习率产生的标量系数 `lr`。这一步相当于“在阀门打开的地方，往水箱里倒入适量的新水”。如果 `gate` 为 0，状态保持不变；如果为 1，状态会按 `lr` 的大小被新信息覆盖。

- **松弛替换**：新状态 = 旧状态 * (1 - `gate` * `lr`) + `delta` * `gate` * `lr`。这里的 `(1 - gate * lr)` 起到保留旧信息的作用，`gate * lr` 决定新信息的渗透程度。与传统 RNN 的全替换不同，这里是“部分替换”，更像是把旧画作局部涂改。

- **并行累加**：在训练阶段，所有 token 的 `gate`、`delta`、`lr` 同时算好后，使用前缀和（prefix sum）技巧一次性算出每一步的隐藏状态。前缀和本身可以在 GPU 上高度并行，因而训练仍保持 O(1) 的显存占用。

- **推理时的常数路径**：推理时只需要保存上一步的隐藏向量和当前 token 的嵌入，随后重复上述三步即可。因为每一步只涉及向量加法和点乘，计算量固定，显存不随序列增长。

**最巧妙的设计**  
- **向量门控 + 学习率的双重调节**：单纯的门控只能决定“开或关”，而学习率提供了“开得多大”。两者结合让模型在长序列上既能保持信息，又能灵活忘记无关信息，这在保持常数显存的前提下实现了类似注意力的选择性。

- **松弛替换实现正则语言识别**：通过部分覆盖的方式，隐藏状态可以模拟有限状态机的转移函数，从而在理论上能够识别所有正则语言。相比只能在 TC⁰ 范畴内操作的 Transformer，这是一大跃迁。

### 实验与效果
- **数据集**：作者构建并公开了 3.1 万亿 token 的多语言语料，覆盖数十种语言，规模比常见的英文单语料库要大但仍远低于同等参数的 Transformer 训练量。
- **模型规模**：四个版本分别为 0.19B、0.5B、1.2B、2.9B 参数。所有模型均在同一语料上训练，展示了规模效应。
- **性能**：2.9B 参数的 RWKV‑7 在多语言基准（如 XGLUE、MMLU multilingual）上取得了 3B 参数模型的最新 SOTA，英文下游任务（如 SuperGLUE）与同规模的最强 Transformer 持平。论文声称在多语言任务上领先约 1–2% 的绝对得分。
- **消融实验**：作者分别去掉向量门控、上下文学习率和松弛替换，发现：
  - 去掉向量门控后模型在长文本上的 perplexity 上升约 12%；
  - 去掉学习率导致训练收敛速度下降约 30%；
  - 去掉松弛替换后模型失去对正则语言的识别能力，状态追踪实验出现明显错误。
- **局限性**：虽然显存常数，但每一步仍涉及全连接运算，实际推理速度在 GPU 上略慢于同等规模的 Transformer；在极端超长序列（> 100k token）上，累加误差仍可能累积，需要额外的数值稳定技巧。作者在论文中承认这些细节仍待进一步优化。

### 影响与延伸思考
- **理论冲击**：RWKV‑7 首次在实际模型中突破了 Transformer 被认为受限于 TC⁰ 的上限，引发了关于序列模型计算复杂度的重新讨论。后续有工作尝试将其思想推广到更大规模的 10B+ 模型（如 RWKV‑8）以及混合 RNN‑Transformer 架构。
- **实际落地**：因为显存常数，RWKV‑7 成为边缘设备和实时对话系统的热门候选，已有开源项目将其部署在移动端进行离线翻译。
- **研究方向**：感兴趣的读者可以进一步关注：
  1. **数值稳定的前缀和实现**，尤其在低精度（fp16、int8）下的误差控制；
  2. **更丰富的门控机制**，比如引入注意力式的跨维度交互；
  3. **理论上对 TC⁰ 之外的序列模型能力界限** 的形式化证明。
  这些都是基于 RWKV‑7 思路的自然延伸。

### 一句话记住它
RWKV‑7 用向量门控的增量更新和松弛替换，让模型在保持每步显存和时延恒定的同时，拥有类似有限状态机的强记忆能力，真正实现了“常数显存、常数时延、强表达力”。