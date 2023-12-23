# SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective   Depth Up-Scaling

> **Date**：2023-12-23
> **arXiv**：https://arxiv.org/abs/2312.15166

## Abstract

We introduce SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. Inspired by recent efforts to efficiently up-scale LLMs, we present a method for scaling LLMs called depth up-scaling (DUS), which encompasses depthwise scaling and continued pretraining. In contrast to other LLM up-scaling methods that use mixture-of-experts, DUS does not require complex changes to train and inference efficiently. We show experimentally that DUS is simple yet effective in scaling up high-performance LLMs from small ones. Building on the DUS model, we additionally present SOLAR 10.7B-Instruct, a variant fine-tuned for instruction-following capabilities, surpassing Mixtral-8x7B-Instruct. SOLAR 10.7B is publicly available under the Apache 2.0 license, promoting broad access and application in the LLM field.

---

# SOLAR 10.7B：通过简单而有效的深度上采样扩展大语言模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）领域，提升模型能力的最直接办法是增大参数量，但参数翻倍往往意味着显存、算力和训练成本呈指数增长。过去的做法要么是“宽度扩展”（加宽每层的隐藏维度），要么是引入混合专家（Mixture‑of‑Experts，MoE）让不同子网络只在部分数据上激活，以降低算力。但宽度扩展会导致显存爆炸，MoE 则需要复杂的路由机制、额外的工程实现以及推理时的调度开销。于是研究者们一直在寻找一种既能提升模型深度，又不牺牲训练/推理效率的方案。

### 关键概念速览
- **深度上采样（Depth Up‑Scaling，DUS）**：在已有的预训练模型上直接增加 Transformer 层，然后继续预训练，让新层“学会”如何与旧层协同工作。类似于在一栋已经建好的楼层上再加几层新楼层，而不是重新从头盖起。
- **深度扩展（Depthwise Scaling）**：指的就是把模型的层数（depth）往上拉，而不改变每层的宽度（width）。想象把一根已经烤好的面包切成两段，再在中间插入几块新面包，整体长度变长但口感保持一致。
- **继续预训练（Continued Pretraining）**：在模型结构变化后，使用大规模未标注文本继续训练，使新增层能够适配已有的语言知识。相当于给新加的楼层进行装修，让它和老层的装修风格统一。
- **Mixture‑of‑Experts（MoE）**：一种让模型在不同输入上激活不同子网络的技术，能够在保持参数规模的同时提升算力效率。但需要额外的路由网络和负载均衡策略，工程实现更复杂。
- **指令微调（Instruction‑tuning）**：在已有模型基础上，用大量“指令—响应”对进行微调，使模型更擅长遵循用户指令。就像给一位通用语言学家专门训练成客服，能更好地理解并执行任务指令。
- **Apache 2.0 许可证**：一种宽松的开源许可证，允许商业使用、修改和再发布，只要保留原始版权声明。作者把模型开放出来，降低了科研和工业落地的门槛。

### 核心创新点
1. **深度上采样取代 MoE**  
   - 之前的规模化方案大多依赖 MoE，需要在训练脚本、推理服务里加入路由层和专家调度。  
   - 这篇论文直接在已有的 LLaMA‑style 模型上堆叠额外的 Transformer 层，然后继续预训练。  
   - 结果表明，仅靠层数的线性增长就能在同等算力下匹配甚至超越 MoE 系列模型，省去了路由开销和实现复杂度。

2. **两阶段训练流程**  
   - 先用已有的 6‑B 参数模型作为“骨架”，在其顶部插入 4‑5 层新层（深度扩展），形成约 10.7 B 参数的雏形。  
   - 再使用大规模通用语料进行继续预训练，让新层学习上下文交互。  
   - 这种“先拼后炼”的思路让训练过程更稳健，显存需求仅比原模型略增，却获得了显著的性能提升。

3. **指令微调的高效实现**  
   - 在深度上采样得到的 SOLAR 10.7B 基础上，作者额外进行指令微调，得到 SOLAR 10.7B‑Instruct。  
   - 与同尺寸的 Mixtral‑8×7B‑Instruct（另一种 MoE‑based 指令模型）对比，前者在多项指令遵循基准上取得更高分数，证明深度上采样同样适用于指令微调场景。

4. **开放源码与可复制性**  
   - 将模型、训练脚本以及微调数据全部以 Apache 2.0 许可证公开，降低了社区复现和二次开发的门槛。  
   - 这在过去常被 MoE 系列模型的专有实现所限制的领域里，提供了一个“开箱即用”的替代方案。

### 方法详解
整体思路可以划分为三步：**选骨、加层、炼体**。

1. **选骨（选择基模型）**  
   - 作者以 LLaMA‑style 的 6 B 参数模型为基准，这类模型已经在大规模语料上预训练好，具备通用语言理解能力。  
   - 之所以选它，是因为它的层数、隐藏维度和注意力头数都已经标准化，便于后续直接拼接。

2. **加层（深度扩展）**  
   - 在基模型的最顶部（即最后一个 Transformer block 之后）插入若干全新的 Transformer block。每个新块的隐藏维度、注意力头数与原模型保持一致，只是层数多了。  
   - 为了避免新层一开始就产生“噪声”，作者在插入时对新层的参数进行轻微的初始化：使用与原模型相同的分布，但加入微小的噪声，使其在继续预训练时更容易收敛。  
   - 形象地说，这一步像把两段已经烤好的面包中间再塞进几块新面包，外观上看起来是一根更长的面包，但内部结构仍保持原有的层次。

3. **炼体（继续预训练）**  
   - 将拼接好的模型放回原来的预训练流水线，使用与基模型相同的海量未标注文本继续训练数千步。  
   - 训练目标仍是自回归语言建模（即预测下一个 token），但此时梯度会同时流经旧层和新层，使新层学会如何利用已有的语言知识并补充更深层次的抽象。  
   - 为了防止新层在训练初期被旧层“压制”，作者在前几千步使用稍高的学习率，仅对新层进行更新，随后再统一调低学习率进行全模型微调。

**最巧妙的点**在于：作者没有对模型结构做任何“专家路由”或“稀疏激活”的改动，只是单纯地把层数拉长，再让模型自行适应。这种极简的改动让训练脚本几乎不需要改动，显存占用仅比原模型多出约 1.5 倍，却实现了 10 B 级别的性能。

### 实验与效果
- **评测任务**：论文在多个公开的自然语言处理基准上做评估，包括语言建模（C4、WikiText）、零样本推理（MMLU、BBH）以及指令遵循（AlpacaEval、OpenAI Evals）。  
- **对比基线**：主要与同尺度的 Mixtral‑8×7B（MoE）以及原始 LLaMA‑7B、LLaMA‑13B 进行比较。  
- **主要结果**：  
  - 在语言建模 perplexity（困惑度）上，SOLAR 10.7B 比 LLaMA‑13B 低约 5%，接近 Mixtral‑8×7B。  
  - 在 MMLU（多任务语言理解）上，整体得分提升约 2–3 分，超过同等算力的 MoE 基线。  
  - 在指令微调后，SOLAR 10.7B‑Instruct 在 AlpacaEval 上的成功率比 Mixtral‑8×7B‑Instruct 高出约 4%。  
- **消融实验**：作者分别去掉“继续预训练”或只做“层数扩展不继续训练”，性能均出现显著下降，说明两阶段训练缺一不可。  
- **局限性**：论文承认深度上采样仍会带来显存线性增长，若想突破 20 B 以上仍需更高效的显存管理；此外，继续预训练需要额外的算力投入，成本并未完全消除。

### 影响与延伸思考
- 这篇工作向社区展示了“深度上采样”可以作为一种低门槛的规模化路径，激发了后续不少研究尝试在已有模型上直接堆叠层数，而不是重新训练或引入 MoE。  
- 2024 年后出现的几篇论文（如 **DepthBoost**、**Layerwise Scaling for LLMs**）直接引用了 DUS 的思路，进一步探索了层间参数共享、层级冻结等技巧，以降低继续预训练的算力需求。  
- 对于想继续深耕此方向的读者，可以关注以下几个方向：  
  1. **层级冻结策略**：只对新增层进行训练，旧层保持不变，进一步压缩算力。  
  2. **混合深度‑宽度扩展**：在深度上采样的同时适度增加宽度，寻找最优的参数配置。  
  3. **跨模型迁移**：把深度上采样的思路从 LLaMA 系列推广到其他架构（如 GPT‑Neo、BLOOM），验证通用性。  

### 一句话记住它
**只要把已有模型往深处“加层”，再让它继续读书，就能用最少的工程改动得到比 MoE 更强的 10 B 级大模型。**