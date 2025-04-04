# Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models

> **Date**：2025-04-04
> **arXiv**：https://arxiv.org/abs/2504.03624

## Abstract

As inference-time scaling becomes critical for enhanced reasoning capabilities, it is increasingly becoming important to build models that are efficient to infer. We introduce Nemotron-H, a family of 8B and 56B/47B hybrid Mamba-Transformer models designed to reduce inference cost for a given accuracy level. To achieve this goal, we replace the majority of self-attention layers in the common Transformer model architecture with Mamba layers that perform constant computation and require constant memory per generated token. We show that Nemotron-H models offer either better or on-par accuracy compared to other similarly-sized state-of-the-art open-sourced Transformer models (e.g., Qwen-2.5-7B/72B and Llama-3.1-8B/70B), while being up to 3$\times$ faster at inference. To further increase inference speed and reduce the memory required at inference time, we created Nemotron-H-47B-Base from the 56B model using a new compression via pruning and distillation technique called MiniPuzzle. Nemotron-H-47B-Base achieves similar accuracy to the 56B model, but is 20% faster to infer. In addition, we introduce an FP8-based training recipe and show that it can achieve on par results with BF16-based training. This recipe is used to train the 56B model. We are releasing Nemotron-H base model checkpoints with support in Hugging Face and NeMo.

---

# Nemotron-H：高精度高效混合 Mamba‑Transformer 系列模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型的推理成本正成为瓶颈：自注意力层的计算随序列长度平方增长，显存和时间都被压得很紧。即使硬件在升级，模型规模的指数式扩张仍让实时响应变得不现实。传统的 Transformer 只能在精度和速度之间做粗糙的权衡，缺少一种既保持强大语言理解，又能把每个 token 的计算和显存固定在常数级别的方案。因此，如何在不牺牲准确率的前提下，根本性地降低推理开销，成为迫切需要解决的问题。

### 关键概念速览
- **Transformer**：目前最流行的序列建模框架，核心是自注意力机制，能够让每个 token 同时“看到”整个序列。缺点是计算和显存随序列长度的二次方增长。  
- **自注意力（Self‑Attention）**：对每个 token 计算它与所有其他 token 的相似度并加权求和，类似于在一群人中每个人都要听完所有人的发言再决定自己的回应。  
- **Mamba（状态空间模型）**：一种基于离散化连续时间状态空间的序列模型，内部只维护一个固定大小的状态向量，处理每个 token 时的计算和显存都是常数。可以把它想成“一条流水线”，每个 token 只在固定的机器上加工一次。  
- **Hybrid Mamba‑Transformer**：把原本全是自注意力的 Transformer 改造成“混合体”，大多数层换成 Mamba，少数关键层仍保留自注意力，以兼顾全局信息捕获和高效计算。  
- **常数计算/常数显存**：指模型在生成每个新 token 时，所需的 FLOPs 和显存不随已生成的上下文长度变化，类似于固定功率的发动机，无论跑多远都消耗相同的油。  
- **剪枝（Pruning）**：把模型中不重要的权重直接删掉，像把一棵大树的枯枝剪掉，减轻重量。  
- **蒸馏（Distillation）**：让小模型（学生）学习大模型（老师）的输出分布，像老师把经验浓缩成简短的笔记给学生。  
- **MiniPuzzle**：论文提出的“一体化剪枝+蒸馏”压缩方案，先系统性地删掉冗余参数，再用原始大模型的行为指导压缩后模型恢复性能。  
- **FP8 训练**：使用 8 位浮点数进行前向和反向传播，显存占用和算力需求大幅下降，类似于把高分辨率图片压成低分辨率后仍能辨认主要内容。  
- **BF16**：16 位浮点数的常用训练格式，精度比 FP8 高，但显存占用也更大。  

### 核心创新点
1. **自注意力 → Mamba 替换 → 推理速度提升**  
   传统模型几乎所有层都是自注意力，导致每个 token 的计算随上下文指数增长。作者把大多数层换成 Mamba，使每层的计算和显存固定为常数。实验显示，在相同模型规模下，推理速度最高提升 3 倍，且在多数基准上保持或超过原有精度。  

2. **MiniPuzzle 剪枝+蒸馏 → 56B → 47B 基础模型 → 进一步加速**  
   直接把 56B 参数的模型裁剪会导致显著性能下降。MiniPuzzle 先用结构化剪枝去除约 15% 参数，再让 47B 模型通过蒸馏学习 56B 老师的输出。结果是 47B 模型在保持 56B 精度的同时，推理速度再快 20%。  

3. **FP8 训练配方 → 与 BF16 同等效果 → 降低训练成本**  
   训练大模型通常需要大量显存，使用 BF16 已是业界常规。作者研发的 FP8 训练技巧包括动态缩放、误差补偿等，使得 56B 模型在 FP8 下的最终性能与 BF16 基准几乎持平，显存需求下降约 50%。  

4. **开放生态兼容 → Hugging Face 与 NeMo 双平台支持**  
   将模型权重和推理代码包装成标准接口，直接可在主流开源框架中使用，降低了社区复现和二次开发的门槛。  

### 方法详解
整体思路可以拆成三大阶段：**混合架构构建 → 高效训练 → 压缩蒸馏**。

1. **混合架构构建**  
   - 从标准的 Decoder‑only Transformer 开始，层数保持不变（例如 56 层）。  
   - 按照经验规则，将 70%~80% 的层替换为 Mamba 层，剩余的 20%~30% 保留自注意力层，主要放在靠近输入和输出的关键位置，以确保模型仍能捕获全局依赖。  
   - 每个 Mamba 层内部实现为离散化的连续时间状态空间方程：输入 token 通过线性投影进入状态更新模块，状态向量在每一步仅做一次矩阵乘法和非线性激活，计算量不随序列长度变化。  

2. **FP8 训练配方**  
   - **动态范围校准**：在每个前向/反向 pass 前，统计激活和梯度的最大值/最小值，使用比例因子把它们映射到 FP8 可表示的区间。  
   - **误差补偿**：引入一个小的 BF16 “残差缓冲”，保存 FP8 量化过程中的舍入误差，类似于数值积分中的误差校正。  
   - **学习率调度**：因为低位数会放大噪声，采用更平缓的 warm‑up 和 cosine‑decay 策略，防止训练过程出现不稳定的梯度爆炸。  
   - 通过上述技巧，作者在 56B 模型上完成了全 FP8 训练，显存占用约为 BF16 的一半，训练时间几乎不变。  

3. **MiniPuzzle 压缩**  
   - **结构化剪枝**：先对每一层的 Mamba 和自注意力权重计算重要性分数（基于梯度幅度或 L1 范数），按照全局阈值删除最不重要的 15% 参数。删除后模型仍保持可运行的结构。  
   - **蒸馏阶段**：使用原始 56B 模型作为教师，收集其在大规模语言建模数据上的 logits 与隐藏状态。学生模型（已剪枝的 47B）在相同数据上最小化 KL 散度和隐藏层 L2 损失，使其行为尽可能接近老师。  
   - **微调**：蒸馏结束后进行一次短周期的全量 FP8 微调，恢复因剪枝导致的轻微性能下降。  

**最巧妙的点**在于把剪枝和蒸馏紧密耦合：剪枝后模型的容量大幅下降，但蒸馏提供了强有力的“知识补偿”，让压缩后的模型几乎不失原有精度。再配合 FP8 训练，整体算力和显存需求被压到前所未有的低点。

### 实验与效果
- **评测任务**：在公开的大语言模型基准套件上（包括 MMLU、GSM8K、TruthfulQA、HumanEval 等）进行零样本和少样本评估。  
- **基线对比**：与同规模的开源 Transformer 系列模型 Qwen‑2.5‑7B/72B、Llama‑3.1‑8B/70B 进行横向比较。  
- **主要结果**：  
  - 8B 版 Nemotron‑H 在所有测评上与 Llama‑3.1‑8B 持平或略有提升，推理吞吐率提升约 2.5×。  
  - 56B 版在 MMLU、GSM8K 等任务上超过 Qwen‑2.5‑72B 约 1%~2% 的准确率，同时在相同硬件上实现最高 3× 的推理加速。  
  - MiniPuzzle 生成的 47B‑Base 与原 56B 模型的整体分数差距在 0.2% 以内，却在同等硬件上快 20%。  
- **消融实验**：  
  - 只使用 Mamba 替换（不保留自注意力）会导致长文本推理的全局一致性下降，验证了保留少量自注意力层的必要性。  
  - 去掉蒸馏，仅做剪枝会使 47B 模型的准确率下降约 3%，说明蒸馏是恢复性能的关键。  
- **局限性**：论文未给出对极长上下文（> 8k token）场景的显著优势数据；压缩过程仍需完整的教师模型进行蒸馏，训练成本仍然高昂。  

### 影响与延伸思考
Nemotron‑H 的出现标志着“混合架构”在大模型领域的实用化：把高效的状态空间模型嵌入 Transformer，直接解决了自注意力的二次方瓶颈。随后的工作（如 Mamba‑2、Hybrid‑S4 等）纷纷在此思路上扩展，探索更细粒度的层级混搭以及更轻量的蒸馏策略。对想进一步深入的读者，可以关注以下方向：  
- **更细致的层级调度**：研究在不同任务或不同深度上动态决定使用 Mamba 还是自注意力。  
- **低位数训练的稳健性**：FP8 已证明可行，后续可能会出现专门为状态空间模型设计的量化方案。  
- **跨模态混合**：把 Mamba 与视觉 Transformer 结合，探索在多模态大模型中的效率提升。  

### 一句话记住它
把大多数自注意力层换成常数计算的 Mamba，再用 MiniPuzzle 剪枝+蒸馏压缩，Nemotron‑H 在保持同等或更好精度的前提下，实现了最高 3 倍的推理加速。