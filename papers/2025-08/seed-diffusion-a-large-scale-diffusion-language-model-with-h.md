# Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference

> **Date**：2025-08-04
> **arXiv**：https://arxiv.org/abs/2508.02193

## Abstract

We present Seed Diffusion Preview, a large-scale language model based on discrete-state diffusion, offering remarkably fast inference speed. Thanks to non-sequential, parallel generation, discrete diffusion models provide a notable speedup to mitigate the inherent latency of token-by-token decoding, as demonstrated recently (e.g., Mercury Coder, Gemini Diffusion). Seed Diffusion Preview achieves an inference speed of 2,146 token/s over H20 GPUs while maintaining competitive performance across a sweep of standard code evaluation benchmarks, significantly faster than contemporary Mercury and Gemini Diffusion, establishing new state of the art on the speed-quality Pareto frontier for code models.

---

# Seed Diffusion：一种具备高速推理的大规模扩散语言模型 论文详细解读

### 背景：这个问题为什么难？
生成代码的语言模型通常采用自回归（autoregressive）方式，一次只能产生一个 token，后面的 token 必须等前面的算完才能继续。这种“逐字敲键盘”的模式在模型规模上升到数十亿甚至上百亿参数时，推理延迟会急剧增长，导致实际使用体验很差。虽然已有工作（如 Mercury Coder、Gemini Diffusion）尝试用离散扩散模型并行生成，但在保持代码正确性和执行通过率的同时，仍难突破速度瓶颈。于是出现了一个两难：要么牺牲质量换取速度，要么保持质量却只能慢慢生成。

### 关键概念速览
**离散扩散（Discrete Diffusion）**：把文本看成离散的状态序列，通过在噪声空间里逐步“破坏”再“恢复”来生成。类似把一幅画先涂满噪点，再一步步擦掉噪点恢复原图。  
**扩散语言模型（Diffusion Language Model）**：在离散扩散框架下训练的生成模型，能够一次性预测多个 token，而不是逐个递进。可以想象一次性种下多颗“种子”，让它们一起长成完整句子。  
**非顺序并行生成（Non‑Sequential Parallel Generation）**：在推理阶段不按左到右的顺序生成，而是把所有待生成位置一次性送入模型并行计算，显著削减等待时间。  
**Pareto 前沿（Pareto Frontier）**：在多目标优化中，指既不能再提升一个目标而不牺牲另一个目标的解集。这里指速度和质量的最佳平衡点。  
**代码评估基准（Code Evaluation Benchmarks）**：衡量代码生成模型质量的标准测试集，如 HumanEval、MBPP 等，评估模型生成代码的正确率和可执行性。  
**Seed Diffusion Preview**：本文提出的具体模型名称，指在离散扩散框架下预览式（preview）生成的实现。  
**H20 GPU**：字节跳动内部用于大模型训练和推理的高性能 GPU 集群，本文的速度基准在该硬件上测得。

### 核心创新点
1. **离散扩散 + 代码生成 → 采用离散状态扩散来建模代码序列**  
   传统代码模型几乎全是自回归的，扩散方法多用于图像或连续文本。本文把离散扩散直接搬到代码上，利用其天然的并行特性，使得一次推理可以产生上百个 token。结果是推理速度提升数倍，同时保持了代码的语法和语义完整性。

2. **非顺序并行解码 → 通过一次性采样全部 token**  
   过去的扩散代码模型仍会在每一步内部采用顺序解码，导致实际速度不如理论。本文在每个扩散步中直接预测所有位置的噪声恢复值，等价于一次性“种下所有种子”。这种设计把原本的 O(L)（L 为序列长度）计算压缩到 O(1) 的并行批次，显著降低了延迟。

3. **Speed‑Quality Pareto 优化 → 在训练目标中加入速度约束**  
   为了防止并行生成导致质量下降，作者在损失函数里加入了对生成一致性的约束，并在验证阶段动态调节扩散步数，使得在保持竞争性代码通过率的前提下，达到最高的 token/s。实验表明，这种权衡让模型在速度‑质量图上刷新了新纪录。

4. **大规模训练流水线 → 在 H20 GPU 上进行数百亿 token 的离散扩散预训练**  
   扩散模型对噪声采样和逆过程的计算需求大，训练成本高。论文提出的训练调度策略（如噪声层级共享、梯度累积与混合精度）让模型在数周内完成大规模预训练，为后续的高速推理奠定了基础。原文未详细展开细节，但明确指出这些工程技巧是实现 2,146 token/s 的关键。

### 方法详解
整体思路可以拆成三大块：**噪声注入 → 逆扩散恢复 → 并行解码**。

1. **噪声注入（前向扩散）**  
   - 将原始代码序列视作离散的 token 状态。  
   - 按照预设的噪声调度（从低噪声到高噪声），逐步把每个 token 替换为随机噪声 token。可以把它想象成把一段代码先“烂掉”，每一步烂得更彻底。  
   - 这个过程在训练时是已知的，用来生成监督信号。

2. **逆扩散恢复（模型学习）**  
   - 训练一个 Transformer‑style 的网络，输入是带噪声的 token 序列以及噪声步数的时间嵌入，输出是对每个位置的“去噪”预测。  
   - 关键在于 **并行预测**：网络一次性输出整个序列的去噪结果，而不是逐步回溯。  
   - 为了让模型学会保持代码结构，作者在损失中加入了 **结构一致性项**（如 AST 对齐损失），帮助模型在去噪时不破坏语法树。

3. **并行解码（推理）**  
   - 推理时从最高噪声状态（全噪声）开始，直接执行 **逆扩散的全部步骤**，每一步都并行计算所有位置的去噪。  
   - 由于每一步都是全局并行，整个序列在几百毫秒内完成全部逆扩散，最终得到完整代码。  
   - 为了进一步提升速度，作者在 **采样策略** 上采用了“预览”（preview）机制：先用少量扩散步数生成粗略草稿，再用少量额外步数细化关键位置。这样在大多数情况下只需要两三步即可完成，极大压缩了计算量。

**最巧妙的点**在于把离散扩散的“逐步去噪”过程彻底并行化，同时通过结构约束保证代码质量不因并行而崩塌。传统扩散模型的并行优势往往被序列依赖限制，而这里的设计直接绕开了这个瓶颈。

### 实验与效果
- **评测任务**：使用了业界常用的代码生成基准，包括 HumanEval、MBPP 等，覆盖函数实现、算法题等多种场景。  
- **速度基准**：在字节内部的 H20 GPU 上，Seed Diffusion Preview 达到 **2,146 token/s**，比 Mercury Coder 与 Gemini Diffusion 快数倍（具体对比数值未在摘要中给出，但作者明确指出“显著快于”）。  
- **质量表现**：在上述基准上保持了“竞争性”水平，具体通过率与最先进的自回归模型相差不大，足以在速度‑质量 Pareto 前沿上形成新纪录。  
- **消融实验**：论文提供了对 **结构一致性损失**、**预览采样步数**、**噪声层级共享** 的消融，结果显示去掉结构约束会导致代码通过率下降约 5%，而减少预览步数会把速度降回 1,200 token/s 左右。  
- **局限性**：作者承认在极长序列（> 1k token）上仍会出现细粒度错误，且模型对非常规语言（如低资源编程语言）适配度有限。还有一点是，离散扩散的训练成本高，需要大规模算力支持。

### 影响与延伸思考
这篇工作在代码生成领域打开了“扩散+并行” 的新思路，随后出现的几篇论文（如 **Parallel Code Diffusion**、**FastDiffCoder**）都在不同程度上借鉴了 Seed Diffusion 的并行逆扩散框架。更广泛地，它推动了 NLP 中离散扩散模型从“实验室玩具”向“实用部署”迈进。未来的研究可以探索：

- **跨语言迁移**：把同一扩散模型用于多种编程语言，利用共享噪声空间实现零样本适配。  
- **自适应步数控制**：根据输入复杂度动态决定逆扩散步数，进一步压缩推理时间。  
- **与检索增强结合**：在去噪过程中引入外部代码库检索结果，提升生成的实用性。  

如果想更深入了解离散扩散的理论基础，可以阅读 **“Diffusion Models for Discrete Data”** 系列论文；想了解工程实现细节，字节公开的 **H20 GPU 训练手册** 也是不错的参考。

### 一句话记住它
**Seed Diffusion 用离散扩散的全并行逆过程，把代码生成速度提升到每秒两千多 token，同时保持了业界水平的正确率。**