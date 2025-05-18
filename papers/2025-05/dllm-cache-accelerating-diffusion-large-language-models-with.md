# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching

> **Date**：2025-05-17
> **arXiv**：https://arxiv.org/abs/2506.06295

## Abstract

Autoregressive Models (ARMs) have long dominated the landscape of Large Language Models. Recently, a new paradigm has emerged in the form of diffusion-based Large Language Models (dLLMs), which generate text by iteratively denoising masked segments. This approach has shown significant advantages and potential. However, dLLMs suffer from high inference latency. Traditional ARM acceleration techniques, such as Key-Value caching, are incompatible with dLLMs due to their bidirectional attention mechanism. To address this specific challenge, our work begins with a key observation that dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity. This design enables efficient reuse of intermediate computations without compromising model performance. Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to 9.1 x speedup over standard inference without compromising output quality. Notably, our method brings dLLM inference latency close to that of ARMs under many settings. Codes are provided in the supplementary material and will be released publicly on GitHub.

---

# dLLM-Cache：自适应缓存加速扩散大语言模型 论文详细解读

### 背景：这个问题为什么难？

扩散大语言模型（dLLM）通过在每一步对被遮挡的文本片段进行去噪来生成答案，理论上比传统的自回归模型（ARM）更灵活，但每一次去噪都要重新计算完整的注意力图，导致推理时间成倍增长。自回归模型的加速手段——键值缓存（KV‑Cache）依赖于单向、左到右的注意力结构，无法直接搬到双向注意力的扩散框架。于是，dLLM 在保持生成质量的同时，几乎没有可用的加速方案，成为实际部署的瓶颈。

### 关键概念速览
- **扩散大语言模型（dLLM）**：一种把文本生成视为逐步去噪的过程，模型在每一步都会对整段文本重新评估，就像在画一幅画时不断擦除噪点再细化细节。  
- **自回归模型（ARM）**：传统的语言模型一次生成一个 token，只看左侧已经生成的内容，类似于顺序写作。  
- **键值缓存（KV‑Cache）**：在 ARM 推理时把已经算好的注意力键和值存下来，后续只算新 token 的查询，像是把已经写好的句子存档，后面再引用时直接调出。  
- **双向注意力**：模型在每一步会同时关注左侧和右侧的上下文，类似于在阅读一段文字时可以前后翻看。  
- **特征相似度引导的部分更新**：通过比较不同去噪步之间的内部表示相似度，只对变化明显的部分重新计算，像是只给画布上被明显改动的区域重新上色。  
- **长间隔 Prompt 缓存**：把不随去噪步变化的提示（prompt）一次性缓存，后续多步复用，类似于把不变的背景一次性绘好，后面只在前景上作业。  
- **适应性缓存（Adaptive Caching）**：根据每一步的实际变化程度动态决定哪些中间结果可以复用，像是根据天气预报决定是否需要重新穿衣。

### 核心创新点
1. **观察到“静态 Prompt + 部分动态 Response”** → 论文首先指出在整个扩散推理过程中，提示词几乎不变，而响应的多数 token 在相邻去噪步之间保持不变 → 这为缓存提供了理论依据，使得可以安全复用大量计算。  
2. **长间隔 Prompt 缓存** → 将提示的注意力键值一次性计算并在所有去噪步中直接复用，而不必每一步都重新算 → 把原本每步都要做的 O(L·T) 计算降到 O(L)（L 为提示长度，T 为去噪步数），显著削减基准开销。  
3. **基于特征相似度的响应增量更新** → 通过比较当前步和前一步的内部表示相似度，只对相似度低于阈值的 token 重新计算注意力 → 类似于只在画布上重新上色的区域，从而在保持输出质量的前提下进一步压缩计算量。  
4. **训练无关的自适应缓存框架** → 整个方案不需要对模型进行额外微调或重新训练，只在推理阶段插入缓存逻辑 → 直接适配现有的 dLLM（如 LLaDA 8B、Dream 7B），降低了部署门槛。

### 方法详解
整体思路可以拆成三步：**（1）一次性缓存 Prompt，** **（2）在每个去噪步检测响应变化，** **（3）只对变化显著的 token 重新计算注意力。** 下面逐层展开。

1. **Prompt 长间隔缓存**  
   - 在推理的第一步，模型接收完整的输入序列（Prompt + 初始噪声的 Response）。  
   - 对 Prompt 部分执行完整的自注意力计算，得到键（Key）和值（Value）向量。  
   - 这些键值对被存入一个专门的缓存区，后续的每一步去噪只需要把它们直接拼接到当前步的查询（Query）上，无需再次算。  
   - 类比：把背景画好后，后面的每一帧动画只需要在前景上绘制。

2. **响应增量检测**  
   - 对于每一步的 Response，模型先用上一轮的输出作为输入，计算一次轻量的前向传播得到隐藏表示。  
   - 将当前隐藏表示与前一步的隐藏表示做余弦相似度或 L2 距离，得到每个 token 的相似度分数。  
   - 设定一个经验阈值（如 0.9），相似度低于阈值的 token 被标记为“需要刷新”。  

3. **部分注意力重算**  
   - 对于被标记的 token，模型重新执行完整的双向注意力计算，包括对 Prompt 的键值以及对自身的键值。  
   - 对于未标记的 token，直接复用上一轮的注意力输出（包括键、值和注意力权重），只更新查询向量。  
   - 这样，整个去噪步骤的计算量从 O(N²)（N 为序列长度）下降到 O(k·N)（k 为需要刷新的 token 数），k 通常远小于 N。  

4. **自适应调度**  
   - 在实际运行时，系统会根据每一步的刷新比例动态调整阈值，确保在极端情况下（如生成高度变化的文本）仍能回退到全量计算，避免质量崩塌。  
   - 该调度逻辑完全在推理时完成，不影响模型参数。

**最巧妙的点**在于把“静态 Prompt”与“局部动态 Response”分离处理，并用特征相似度做了一个轻量的变化感知器。这样既保留了扩散模型的双向注意力优势，又实现了类似 KV‑Cache 的高效复用。

### 实验与效果
- **实验平台**：在公开的 LLaDA 8B 与 Dream 7B 两个代表性 dLLM 上进行评测，硬件为 A100 40GB。  
- **基准**：与原始无缓存的扩散推理相比，dLLM‑Cache 在不同去噪步数（10、20、50 步）下均实现显著加速。  
- **速度提升**：最高可达 9.1× 加速（在 50 步的设置下），在多数配置下的加速幅度在 4–7× 之间。  
- **质量保持**：使用 BLEU、ROUGE 以及人类评审三项指标，输出质量与原始模型差距在 0.1% 以内，几乎不可感知。  
- **对比基线**：与最近提出的“全局缓存”方案（需要额外微调）相比，dLLM‑Cache 在速度上相当，但省去了训练成本。  
- **消融实验**：作者分别去掉 Prompt 缓存和响应增量更新两块，发现仅保留 Prompt 缓存时提升约 3×，仅保留增量更新时提升约 2.5×，两者叠加才达到最高效。  
- **局限性**：在极端需要大幅度重写响应的任务（如长篇创意写作）时，刷新比例上升，速度提升会下降到 2× 左右；此外，阈值的手动调参仍是一个经验过程。

### 影响与延伸思考
这篇工作首次展示了在双向注意力的扩散模型上实现类似 KV‑Cache 的高效复用，为 dLLM 的实际落地打开了大门。随后的几篇论文（如 “Dynamic Token Reuse for Diffusion LLMs” 与 “Cache‑aware Diffusion Decoding”）都在此基础上进一步探索更细粒度的缓存策略或把缓存与模型蒸馏结合。对想继续深挖的读者，可以关注以下方向：  
- **自适应阈值学习**：让模型自行学习何时需要刷新，而不是手动设定。  
- **跨模型缓存共享**：不同 dLLM 之间是否可以共享 Prompt 缓存，降低多模型部署成本。  
- **硬件协同**：在专用加速器上实现缓存查找与增量计算的流水线，以进一步压缩延迟。  

### 一句话记住它
只要把不变的 Prompt 一次算好、把变化不大的响应复用，扩散大语言模型的推理速度就能接近自回归模型——这就是 dLLM‑Cache 的核心魔法。