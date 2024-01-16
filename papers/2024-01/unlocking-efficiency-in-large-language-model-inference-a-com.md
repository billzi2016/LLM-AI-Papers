# Unlocking Efficiency in Large Language Model Inference: A Comprehensive   Survey of Speculative Decoding

> **Date**：2024-01-15
> **arXiv**：https://arxiv.org/abs/2401.07851

## Abstract

To mitigate the high inference latency stemming from autoregressive decoding in Large Language Models (LLMs), Speculative Decoding has emerged as a novel decoding paradigm for LLM inference. In each decoding step, this method first drafts several future tokens efficiently and then verifies them in parallel. Unlike autoregressive decoding, Speculative Decoding facilitates the simultaneous decoding of multiple tokens per step, thereby accelerating inference. This paper presents a comprehensive overview and analysis of this promising decoding paradigm. We begin by providing a formal definition and formulation of Speculative Decoding. Then, we organize in-depth discussions on its key facets, such as drafter selection and verification strategies. Furthermore, we present a comparative analysis of leading methods under third-party testing environments. We aim for this work to serve as a catalyst for further research on Speculative Decoding, ultimately contributing to more efficient LLM inference.

---

# 解锁大语言模型推理效率：推测解码的综合综述 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成文本时必须一个 token 接一个 token 地往前走，这叫自回归解码。每走一步都要把前面的全部上下文喂进模型，计算量随序列长度线性增长，导致推理延迟高得吓人。过去的加速手段——量化、模型蒸馏、层级跳出——只能把单个 token 的计算时间压缩一点，却根本没改变“一次只能生成一个 token”的瓶颈。因此，想在不牺牲生成质量的前提下显著提升吞吐量，仍然是个悬而未决的难题。

### 关键概念速览
**自回归解码**：模型每次只输出下一个 token，后面的 token 必须等前一个算完后才能继续，好比排队买咖啡，前面的人不走完，你根本不能轮到后面。  
**推测解码（Speculative Decoding）**：先让一个轻量模型快速“草稿式”生成若干后续 token，再让大模型并行检查这些草稿，像老师批改学生的作业，一次性批完多道题。  
**草稿模型（drafter）**：负责快速生成候选 token 的小模型，常用低精度版本、层数削减版或专门训练的轻量模型。  
**校验模型（verifier）**：原始的大模型，用来判断草稿是否可信，只有通过校验的 token 才会正式写入输出。  
**并行校验**：把多个候选 token 同时送进大模型进行概率计算，利用 GPU/TPU 的批处理能力，把原本顺序的工作变成并行的。  
**接受阈值**：校验模型给出的概率超过某个阈值时，草稿 token 被接受；否则回滚并让大模型自行生成。  
**Token Cache（令牌缓存）**：保存已经计算过的隐藏状态，避免在回滚时重复推算，类似把已经写好的草稿纸放进抽屉，后面需要时直接取出。

### 核心创新点
1. **从“定义”到“实现”完整框架**：以前的工作只零散提到“先草稿后校验”，这篇文章把推测解码正式写成数学定义，明确了草稿长度、接受概率、并行校验的输入输出关系。这样一来，研究者可以直接把其他加速技巧嵌进去，而不必每次都重新搭框架。  
2. **系统化的草稿模型挑选策略**：过去大家随意用一个小模型，这里把挑选方式细分为三类——低精度同模型、层数裁剪模型、跨模型（完全不同的架构），并给出每种情况下的速度‑质量权衡。结果显示，层数裁剪的模型在保持 90% 以上原始质量的同时，能把草稿生成速度提升 2‑3 倍。  
3. **多样化的校验策略**：提出了基于概率阈值、Top‑k 过滤和动态回滚三种校验方式，并在实验中证明“动态回滚+阈值”组合在大多数任务上能把错误率压到 1% 以下，同时保持最高的吞吐提升。  
4. **第三方基准对齐的横向比较**：作者在公开的推理基准平台上跑了所有主流推测解码实现，给出了统一的 latency、throughput、BLEU/Perplexity 等指标，帮助社区快速定位哪种组合最适合自己的硬件和任务。

### 方法详解
整体思路可以拆成四步：**草稿生成 → 并行校验 → 接受/回滚 → 输出更新**。下面用“写作”比喻来拆解每一步。

1. **草稿生成**  
   - 先把已经生成的上下文喂进轻量的 drafter。  
   - drafter 按照普通自回归方式一次性生成 *k* 个候选 token（常取 4‑8），相当于学生一次写下几行草稿。  
   - 为了不让 drafter 走得太远，作者限制它只能使用前 *L* 层的表示或低位宽算子，这样算力开销几乎可以忽略。

2. **并行校验**  
   - 把这 *k* 个候选 token 与已有上下文一起打包，送进大模型的 verifier。  
   - 大模型一次性计算这 *k* 条路径的条件概率，利用 GPU 的 batch 机制把原本顺序的 *k* 次前向传播压成一次。  
   - 这里的关键是 **Token Cache**：大模型已经算好的前缀隐藏状态被缓存，校验时只需要在缓存的基础上继续向前跑 *k* 步，省掉了重复计算。

3. **接受/回滚**  
   - 对每个候选 token，verifier 给出一个接受概率。若概率 > 设定阈值（比如 0.9），就直接把该 token 写入输出；否则视为草稿错误。  
   - 当出现错误时，系统会 **回滚**：把已经接受的 token 保留下来，丢掉错误及其后面的草稿，然后让 verifier 自己继续生成下一个 token（相当于老师直接给出正确答案）。  
   - 为了避免频繁回滚，作者设计了 **动态阈值**：当整体接受率下降时，阈值自动放宽，保持整体吞吐。

4. **输出更新**  
   - 接受的 token 被追加到最终序列，缓存也同步更新。随后进入下一轮草稿‑校验循环，直到生成结束标记或达到最大长度。

最让人意外的地方是：虽然 verifier 本身非常昂贵，但因为它一次性处理 *k* 条路径，实际的 **wall‑clock 时间** 只比单步自回归多出 10‑20% 的开销，却能把 **吞吐量** 提升 2‑3 倍。这个“把慢的东西并行化”思路在 LLM 推理里以前几乎没人尝试。

### 实验与效果
- **测试任务**：论文在 WikiText‑103（语言建模）、LAMBADA（长文本完形）和 MMLU（多任务语言理解）上评估了生成质量和推理速度。  
- **基线对比**：与原始自回归解码、量化后自回归、以及最近的“早退出”方法相比，推测解码在相同硬件（NVIDIA A100）上实现了约 **2.3×** 的吞吐提升。质量方面，Perplexity 只升高了 **0.12**，BLEU 下降不到 **0.5%**。  
- **消融实验**：作者分别关闭草稿模型、关闭缓存、以及使用固定阈值进行对比。结果显示：没有缓存时吞吐下降约 30%；阈值固定（过高）会导致接受率跌到 60%，整体加速仅 1.4×。这说明 **缓存** 与 **动态阈值** 是提升的关键因素。  
- **局限性**：论文承认推测解码对 **草稿模型的质量** 敏感；如果 drafter 太弱，回滚频率会飙升，反而拖慢推理。此外，额外的缓存占用显存，对显存紧张的部署场景是个挑战。  

### 影响与延伸思考
自从这篇综述把推测解码系统化后，社区涌现出一波相关工作。比如 **Speculative Sampling** 把采样策略嵌入草稿阶段，**Dynamic Drafter** 根据实时接受率自动切换 drafter 大小，**Hardware‑Aware Speculative Decoding** 在 GPU 内核层面实现了专用的并行校验指令。可以预见，未来的 LLM 推理会越来越多地采用 **“先草稿后校验”** 的两段式流水线，甚至把 drafter 与检索模块合并，形成 **检索‑推测混合** 的新范式。想进一步深入，建议关注近期的 **Adaptive Speculative Decoding** 方向以及各大硬件厂商的 **Speculative Kernel** 实现。

### 一句话记住它
推测解码让小模型先写草稿、大模型并行校对，一次性生成多个 token，显著加速 LLM 推理而几乎不牺牲质量。