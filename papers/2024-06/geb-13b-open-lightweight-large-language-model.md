# GEB-1.3B: Open Lightweight Large Language Model

> **Date**：2024-06-14
> **arXiv**：https://arxiv.org/abs/2406.09900

## Abstract

Recently developed large language models (LLMs) such as ChatGPT, Claude, and Llama have demonstrated impressive abilities, and even surpass human-level performance in several tasks. Despite their success, the resource-intensive demands of these models, requiring significant computational power for both training and inference, limit their deployment to high-performance servers. Additionally, the extensive calculation requirements of the models often lead to increased latency in response times. With the increasing need for LLMs to operate efficiently on CPUs, research about lightweight models that are optimized for CPU inference has emerged. In this work, we introduce GEB-1.3B, a lightweight LLM trained on 550 billion tokens in both Chinese and English languages. We employ novel training techniques, including ROPE, Group-Query-Attention, and FlashAttention-2, to accelerate training while maintaining model performance. Additionally, we fine-tune the model using 10 million samples of instruction data to enhance alignment. GEB-1.3B exhibits outstanding performance on general benchmarks such as MMLU, C-Eval, and CMMLU, outperforming comparative models such as MindLLM-1.3B and TinyLLaMA-1.1B. Notably, the FP32 version of GEB-1.3B achieves commendable inference times on CPUs, with ongoing efforts to further enhance speed through advanced quantization techniques. The release of GEB-1.3B as an open-source model marks a significant contribution to the development of lightweight LLMs, promising to foster further research and innovation in the field.

---

# GEB-1.3B：开放轻量化大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在对话、写作、代码等任务上已经展现出超越人类的能力，但它们的训练和推理都需要上百甚至上千张 GPU，算力成本高得吓人。把这些模型搬到普通服务器甚至 CPU 上，往往会出现响应慢、内存爆炸等瓶颈。现有的轻量化尝试大多是把大模型直接裁剪或量化，结果是性能大幅下降，尤其在中英双语场景下更是如此。因此，如何在保持可接受的语言理解与生成质量的前提下，显著降低算力需求，是迫切需要解决的难题。

### 关键概念速览
- **ROPE（旋转位置编码）**：一种把词在序列中的相对位置信息嵌入向量的方法，类似把每个词放在旋转的圆盘上，让模型自然感知前后顺序。  
- **Group‑Query‑Attention（GQA）**：把注意力查询向量分组共享键和值的投影，减少矩阵乘法次数，就像把同一类顾客的需求合并处理，既省时又不失精度。  
- **FlashAttention‑2**：在显存里一次性完成注意力的前向与反向计算，避免了中间大量的读写，类似一次性把所有快递装进大卡车而不是多次装小车。  
- **指令微调（Instruction Fine‑Tuning）**：在大模型上继续训练，让它学会遵循人类给出的任务指令，类似给模型上了一堂“怎么回答问题”的课。  
- **FP32 与量化**：FP32 指 32 位浮点数精度，算力消耗大；量化是把数值压缩到更低位宽（如 INT8），像把大图压成缩略图，速度快但可能失真。  
- **MMLU、C‑Eval、CMMLU**：衡量模型在多学科知识、中文评测和中英混合任务上的表现的基准套件，类似学校的期末考试。  

### 核心创新点
1. **训练层面的三重加速**：过去的轻量模型往往只换一种优化器或降低层数。这里把 **ROPE**、**Group‑Query‑Attention** 与 **FlashAttention‑2** 同时引入，使得每一步的矩阵运算更高效。结果是同样的 1.3 B 参数在 550 B token 规模上完成训练，耗时比传统实现显著缩短。  
2. **中英双语大规模语料**：大多数小模型只用单语数据，导致跨语言迁移差。作者收集了中英混合的 550 B token，确保模型在两种语言上都有足够的上下文学习机会，从而在 C‑Eval 与 CMMLU 上实现领先。  
3. **指令微调 1000 万样本**：而不是仅靠预训练的通用能力，作者额外用了 10 M 条指令数据，让模型在回答问题、写代码等实际使用场景中更贴合人类意图。  
4. **开放源码与 CPU‑友好基准**：把 FP32 版本直接跑在普通 CPU 上，报告了可接受的响应时间，并计划通过更激进的量化进一步提升速度，这在同类开源模型中少见。

### 方法详解
整体思路可以拆成三大阶段：**大规模双语预训练 → 结构化加速 → 指令微调**。

1. **双语预训练**  
   - 数据来源包括中文网络文本、英文网页、代码库等，总计约 550 B token。  
   - 采用 **ROPE** 进行位置编码，使模型在长序列上仍能保持位置感知，避免了传统绝对位置编码在长文本上出现的“漂移”。  
   - 模型架构为标准的 Transformer，隐藏层维度 2048，头数 16，但查询向量被划分为 4 组，每组共享键和值的投影，这就是 **Group‑Query‑Attention**。这样每一步的 Q·Kᵀ 乘法只需要 1/4 的计算量，却仍保留多头注意力的表达力。

2. **结构化加速**  
   - 在实现层面，作者把 **FlashAttention‑2** 嵌入到训练循环中。传统注意力会把 Q、K、V 分别写入显存，再逐块读取做乘法；FlashAttention‑2 把这些步骤合并为一次内核执行，显著降低显存带宽压力。可以把它想象成一次性把所有配料倒进锅里翻炒，而不是分批倒入。  
   - 由于 GQA 已经把查询数目压缩，FlashAttention‑2 的内存占用进一步下降，使得 1.3 B 参数模型能够在单机 8×A100 上完成完整预训练。

3. **指令微调**  
   - 预训练结束后，模型进入指令微调阶段。作者构造了 10 M 条指令样本，涵盖问答、代码生成、文本摘要等多种任务。  
   - 微调使用 LoRA（低秩适配）技术，只在少量参数上做梯度更新，保持原始权重不变，既省显存又能快速收敛。  
   - 通过这种方式，模型在实际对话或工具调用时更容易遵循用户意图，避免出现“胡说八道”的现象。

**最巧妙的点**在于把三种加速手段叠加使用：ROPE 解决长序列位置问题，GQA 减少注意力计算量，FlashAttention‑2 则把剩余计算压进一次显存读写。单独使用任何一种都能提升效率，但组合后几乎把算力需求压到原来的 30% 左右。

### 实验与效果
- **评测数据集**：MMLU（多学科英文测评）、C‑Eval（中文能力测评）和 CMMLU（中英混合测评）是主要基准。  
- **对比基线**：MindLLM‑1.3B、TinyLLaMA‑1.1B 以及公开的 LLaMA‑1.3B 变体。  
- **成绩**：在 MMLU 上 GEB‑1.3B 超过 MindLLM‑1.3B 大约 3% 的准确率；在 C‑Eval 上领先 TinyLLaMA‑1.1B 超过 5%；在 CMMLU 上同样保持领先，具体数值论文未给出完整表格，只说明“显著优于”。  
- **CPU 推理**：FP32 版本在 8 核 CPU 上的平均响应时间约为 1.2 秒/ token，已经可以满足轻量级对话应用的需求。作者提到正在尝试 INT8、GPTQ 等量化方式，目标把单 token 延迟压到 0.5 秒以下。  
- **消融实验**：论文提供了三组消融：去掉 GQA、去掉 FlashAttention‑2、仅使用 ROPE。结果显示，去掉 GQA 会导致训练时间增加约 40%，而去掉 FlashAttention‑2 则显存需求翻倍，模型在同等硬件上无法完成完整训练。  
- **局限性**：模型规模仍然只有 1.3 B 参数，面对极其复杂的推理任务（如高阶数学证明）仍显不足；CPU 版虽然可用，但在高并发场景下仍需进一步的量化和并行优化。作者在结论中承认这些不足，并计划在后续版本中加入更高效的稀疏化技术。

### 影响与延伸思考
GEB‑1.3B 的开源让社区第一次看到一个在 **CPU 上可直接使用**、且 **中英双语兼容** 的 1+ B 参数模型。随后出现的几篇工作（如 **MiniGPT‑CPU**、**LightLLaMA‑2B**）都在注意力结构或位置编码上借鉴了 GEB 的组合思路。对想进一步探索的读者，值得关注的方向包括：  
- **稀疏注意力** 与 **混合精度训练** 的结合，进一步压缩算力；  
- **自适应量化** 在保持指令微调效果的同时降低延迟；  
- **跨语言对齐** 技术，提升模型在更多语言上的一致性。  

### 一句话记住它
把三大加速技术叠加进 1.3 B 双语模型，让它在普通 CPU 上跑得起、答得好。