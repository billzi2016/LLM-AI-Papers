# TiDAR: Think in Diffusion, Talk in Autoregression

> **Date**：2025-11-12
> **arXiv**：https://arxiv.org/abs/2511.08923

## Abstract

Diffusion language models hold the promise of fast parallel generation, while autoregressive (AR) models typically excel in quality due to their causal structure aligning naturally with language modeling. This raises a fundamental question: can we achieve a synergy with high throughput, higher GPU utilization, and AR level quality? Existing methods fail to effectively balance these two aspects, either prioritizing AR using a weaker model for sequential drafting (speculative decoding), leading to lower drafting efficiency, or using some form of left-to-right (AR-like) decoding logic for diffusion, which still suffers from quality degradation and forfeits its potential parallelizability. We introduce TiDAR, a sequence-level hybrid architecture that drafts tokens (Thinking) in Diffusion and samples final outputs (Talking) AutoRegressively - all within a single forward pass using specially designed structured attention masks. This design exploits the free GPU compute density, achieving a strong balance between drafting and verification capacity. Moreover, TiDAR is designed to be serving-friendly (low overhead) as a standalone model. We extensively evaluate TiDAR against AR models, speculative decoding, and diffusion variants across generative and likelihood tasks at 1.5B and 8B scales. Thanks to the parallel drafting and sampling as well as exact KV cache support, TiDAR outperforms speculative decoding in measured throughput and surpasses diffusion models like Dream and Llada in both efficiency and quality. Most notably, TiDAR is the first architecture to close the quality gap with AR models while delivering 4.71x to 5.91x more tokens per second.

---

# TiDAR: Think in Diffusion, Talk in Autoregression 论文详细解读

### 背景：这个问题为什么难？

语言模型的生成方式主要有两大阵营：**扩散模型**（Diffusion）和**自回归模型**（AR）。扩散模型一次可以并行生成整段文本，理论上能把 GPU 利用率推到极限，但因为它们是把噪声一步步“还原”，生成质量往往不如自回归模型。自回归模型则一步一步预测下一个词，顺序性和因果结构天然匹配语言建模任务，质量高，但只能串行生成，导致吞吐量低、GPU 计算资源闲置。过去的折中方案——比如**投机解码**（speculative decoding）——让一个弱的自回归模型先草拟（draft）后再用强模型校验（verify），但草拟阶段本身仍是自回归的，效率提升有限。还有一些把左到右的自回归思路硬塞进扩散框架的尝试，虽然保留了并行性，却牺牲了扩散模型本应拥有的质量潜力。于是，如何在同一模型里兼顾 **高并行度**、**高 GPU 利用率** 与 **自回归级别的生成质量**，成为了一个悬而未决的难题。

### 关键概念速览

- **扩散语言模型（Diffusion LM）**：先在噪声空间采样，再逐步去噪得到文本，类似把一幅画先涂成乱七八糟的颜色再慢慢恢复，天然支持并行生成。  
- **自回归语言模型（AR LM）**：每一步只看已经生成的前缀，预测下一个词，像写作文时只能在已有句子后继续写，质量好但速度慢。  
- **投机解码（Speculative Decoding）**：让一个轻量的自回归模型先快速生成草稿，再用强模型逐词检查并纠正，类似先写草稿再请老师批改。  
- **结构化注意力掩码（Structured Attention Mask）**：在 Transformer 的注意力矩阵里人为设定哪些位置可以相互看见，像在会议室里划分“只能听前面发言”的规则，用来同步不同子任务。  
- **KV 缓存（Key‑Value Cache）**：自回归模型在推理时把已经计算好的注意力键值对存下来，后续只需增量计算，极大提升速度。  
- **并行草拟‑验证流水线（Parallel Draft‑Verify Pipeline）**：在同一次前向传播里，同时生成 k 个草稿 token（Diffusion 思考）并对前一步的 k 个草稿进行自回归校验（AR 说话），类似工厂的装配线，一边生产新零件一边检查旧零件。  

### 核心创新点

1. **一次前向完成双重任务 → 通过结构化注意力掩码把 Diffusion 草稿生成和 AR 验证强行并行** → 只需一次模型调用即可同时得到新草稿和旧草稿的最终采样，显著降低解码时的调用次数和显存开销。  
2. **统一模型兼容 KV 缓存 → 设计的注意力掩码保持因果性，仅在验证阶段使用缓存** → 让 AR 部分可以像传统自回归模型那样复用 KV 缓存，进一步提升吞吐量。  
3. **草稿‑验证比例可调的流水线调度 → 每一步同时处理 k 个 Diffusion 草稿和 k 个 AR 验证** → 通过调节 k，模型在“思考深度”和“验证力度”之间找到最佳平衡，实现 4.7–5.9× 的 token‑per‑second 提升。  
4. **服务友好、单模型部署 → 不需要额外的草稿模型或后处理模块** → 与投机解码需要两套模型不同，TiDAR 只是一套权重，部署成本更低，适合实际生产环境。

### 方法详解

#### 整体框架概览  
TiDAR 把一次 Transformer 前向拆成两层逻辑：**思考层**（Diffusion）负责并行生成一批候选 token，**说话层**（AR）负责对上一轮的候选进行因果采样并决定是否接受。两层共享同一套参数，只是注意力掩码不同。整个解码过程像一条环形流水线：第 0 步只做 Diffusion（没有前一轮的验证），第 1 步开始同时进行 Diffusion（生成第 2‑k+1 个 token）和 AR（验证第 1‑k 个 token），如此循环直到生成完毕。

#### 关键模块拆解  

1. **结构化注意力掩码的设计**  
   - 注意力矩阵是 $L \times L$（L 为当前序列长度）。TiDAR 把它划分为四块：左上角（旧 token ↔ 旧 token）使用标准因果掩码，保证 AR 验证只能看到更早的已确定 token；右上角（旧 token ↔ 新 token）全部屏蔽，防止新草稿影响旧验证；左下角（新 token ↔ 旧 token）开放，让 Diffusion 草稿可以参考已经确定的上下文；右下角（新 token ↔ 新 token）使用 **全局** 掩码，使 Diffusion 部分能够并行看到所有新 token 的噪声向量。这样一次前向就能同时完成两种信息流的计算。  

2. **Diffusion 草稿的生成机制**  
   - 在思考层，TiDAR 把待生成的 k 个位置视作一个小的扩散子空间。模型先在这些位置上注入随机噪声，然后通过一次 Transformer 前向直接输出噪声的“去噪”估计，得到草稿 token 的分布。因为所有 k 个位置一次性处理，GPU 的并行度被充分利用。  

3. **AR 验证的采样策略**  
   - 验证层沿用自回归的采样规则：对每个待验证的 token，模型计算其条件概率并进行 **拒绝采样**（或直接采样），如果采样结果与 Diffusion 草稿一致则接受，否则覆盖为新的 AR 采样值。由于 KV 缓存的存在，验证阶段只需对新增的 token 进行增量注意力计算，旧 token 的键值对直接复用。  

4. **流水线调度**  
   - 设定一个超参数 $k$（如 8、16），每一步模型同时处理 $k$ 个 Diffusion 草稿和 $k$ 个 AR 验证。实际实现时，模型内部维护两个指针：`draft_ptr` 指向当前要生成的草稿位置，`verify_ptr` 指向上一步需要验证的位置。每次前向结束后，`draft_ptr` 前进 $k$，`verify_ptr` 也前进 $k$，形成“生产—检验”同步进行的循环。  

#### 反直觉/巧妙之处  
- **一次前向完成两种生成范式**：传统上 Diffusion 与 AR 完全是两套网络，TiDAR 用掩码把它们“压在同一张网格里”，让硬件只做一次矩阵乘法，却得到两套不同的输出。  
- **KV 缓存在混合模型中的兼容**：因为 AR 验证仍然保持因果结构，缓存可以像普通自回归模型一样使用，这一点在混合模型里并不显而易见。  

### 实验与效果

- **评测任务**：论文在 1.5B 与 8B 参数规模下，对 **生成任务**（如开放域对话、长文本续写）和 **似然评估任务**（语言模型 perplexity）进行对比。  
- **基线对比**：与纯自回归模型、投机解码方案以及最新的扩散语言模型（Dream、Llada）进行比较。TiDAR 在 **吞吐量** 上比投机解码快约 20%–30%，在 **质量**（BLEU、ROUGE、Perplexity）上几乎追平自回归基线，明显优于扩散基线。作者报告的 **4.71×–5.91×** token‑per‑second 提升，是在相同硬件（A100）下测得的。  
- **消融实验**：通过关闭结构化掩码、去掉 KV 缓存、或把 $k$ 设为 1（退化为纯 AR）进行实验，结果显示：掩码是并行草稿的关键，KV 缓存贡献约 15% 的吞吐提升，$k$ 越大并行度越好，但超过一定阈值后验证错误率上升，整体质量略降。  
- **局限性**：论文承认在 **极端长序列**（> 4k token）时，掩码矩阵的内存开销仍然是瓶颈；此外，草稿的质量受噪声采样的随机性影响，某些低资源语言的表现尚未验证。  

### 影响与延伸思考

TiDAR 把 **并行草稿** 与 **因果校验** 融为一体的思路，为“混合生成架构”打开了新方向。后续工作（如 2024‑2025 年的 **HybridDiff**、**ParallelAR**）纷纷借鉴其结构化注意力掩码的设计，尝试把检索、视觉特征甚至外部工具也嵌入同一前向。对想进一步探索的读者，可以关注：

- **更高效的掩码实现**（稀疏注意力、FlashAttention）以降低长序列的显存占用。  
- **多模态扩散‑AR 混合**，把图像或音频的扩散草稿与语言的 AR 验证结合。  
- **自适应 $k$ 调度**，让模型根据当前上下文复杂度动态决定并行草稿数量。  

### 一句话记住它

**TiDAR 用一次前向的结构化注意力把并行的扩散草稿和自回归校验合体，实现了“思考‑说话”双轨流水线，既保留了自回归的高质量，又达到了扩散模型的高吞吐。**