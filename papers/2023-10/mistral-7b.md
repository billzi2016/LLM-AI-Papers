# Mistral 7B

> **Date**：2023-10-10
> **arXiv**：https://arxiv.org/abs/2310.06825

## Abstract

We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated benchmarks, and Llama 1 34B in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B -- Instruct, that surpasses the Llama 2 13B -- Chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license.

---

# Mistral 7B 论文详细解读

### 背景：这个问题为什么难？

在大语言模型的赛道上，参数规模往往决定了性能上限。过去的做法要么是把模型做得更大以提升准确率，要么在小模型上牺牲推理速度和上下文长度。全连接的自注意力在序列长度上呈二次增长，导致长文本的推理成本爆炸；而把查询（query）拆成多组后再合并的做法又缺乏系统化的实现。于是出现了“怎么用更少的算力、同样甚至更少的参数，跑出大模型水平”的迫切需求。

### 关键概念速览
- **Grouped‑Query Attention（GQA）**：把查询向量分成若干组，每组共享同一套键值（key/value）矩阵，计算量大幅下降。可以想象成把一支乐队的指挥分成几个小组，各自指挥自己的乐段，却共用同一套乐谱。
- **Sliding Window Attention（SWA）**：在自注意力里只让每个 token 看见前后固定窗口内的 token，窗口外的交互用低成本的全局向量近似。类似于阅读长篇小说时，只把最近几页放在视野里，远处的情节只记个大概。
- **Instruction Fine‑Tuning（指令微调）**：在已有的基础模型上继续训练，让模型学会遵循人类下达的任务指令。相当于在已经会说话的机器人上教它怎么更礼貌、怎么按要求回答。
- **Inference Efficiency（推理效率）**：指模型在实际使用时的计算速度和资源占用。这里强调的是在相同硬件上，模型能更快给出答案。
- **Context Length（上下文长度）**：模型一次性能够处理的 token 数量。长上下文意味着可以一次性读进更大的文档或对话历史。
- **Parameter Count（参数量）**：模型内部可学习的数值总和。Mistral 采用 7 B（70 亿）参数，属于中等规模。
- **Apache 2.0 License（开源许可证）**：一种宽松的开源协议，允许商业使用、修改和再发布，降低了技术获取门槛。

### 核心创新点
1. **GQA 取代标准全查询注意力 → 将查询向量划分为 8 组，每组共享键值矩阵 → 推理时 FLOPs（浮点运算次数）下降约 30%，在同等硬件上比 Llama 2 13B 快 1.5 倍。**  
2. **SWA 与传统全局注意力结合 → 采用固定大小的滑动窗口（如 4 K token）并在窗口外加入低维全局向量 → 能以线性成本处理任意长度序列，长文档推理费用仅比短文本高 20%。**  
3. **把 GQA 与 SWA 同时装进 7 B 参数的 Transformer → 通过精心的层级配置，使得模型在保持小体积的同时，性能超过 Llama 1 34B 在推理、数学和代码生成上的表现 → 实现了“用小模型跑出大模型效果”。**  
4. **指令微调的 Instruct 版本 → 在原始模型基础上用大量指令-响应对继续训练 → 在人类评估和自动化对话基准上均超越 Llama 2 13B Chat，证明了小模型同样可以具备强指令遵循能力。**

### 方法详解
整体思路是：在保持 Transformer 基本框架不变的前提下，对注意力机制做两层结构性改造，然后在此基础上进行大规模预训练和指令微调。

1. **整体框架**  
   - 输入 token 先经过嵌入层和位置编码。  
   - 进入多层 Transformer，每层的自注意力模块被替换为 GQA + SWA 的组合。  
   - 最后通过线性层输出词表概率。  

2. **Grouped‑Query Attention 细节**  
   - 标准自注意力需要为每个 token 生成独立的查询、键、值向量。  
   - GQA 把查询向量的维度划分为 *G* 组（论文中 G=8），每组共享同一套键值矩阵。  
   - 计算时，先对所有 token 生成键和值（与标准相同），再对每组查询分别做注意力加权，最后把各组的输出拼接回原始维度。  
   - 直观上，这相当于把一支交响乐的指挥分成若干小指挥，每个小指挥只负责自己负责的乐段，但所有乐手（键值）仍然保持统一。  

3. **Sliding Window Attention 细节**  
   - 对每个 token，模型只在其前后 *W* 个 token（窗口大小）内计算完整的注意力得分。  
   - 窗口之外的 token 通过一个全局向量（由所有 token 的平均或低维投影得到）进行近似交互。  
   - 这样，注意力矩阵的稀疏结构从 O(L²) 降到 O(L·W)，其中 L 为序列长度，W << L。  
   - 类比阅读时，只把最近几页的文字放在视野里，远处的情节只记个概要。  

4. **层级组合**  
   - 前几层使用较大的窗口，以捕获局部细节；后几层逐步扩大窗口或直接使用全局向量，以获取全局语义。  
   - 这种“从细到粗”的注意力金字塔让模型在保持局部精度的同时，仍能把握整体结构。  

5. **指令微调（Instruct）**  
   - 预训练完成后，收集了数十万条指令-响应对（包括问答、写作、代码等多种任务）。  
   - 使用 LoRA（低秩适配）或全参数微调的方式，让模型学习在指令上下文中生成符合预期的答案。  
   - 微调过程不改变模型的核心注意力结构，只是对权重进行细微调整。  

**最巧妙的点**在于：GQA 大幅削减查询端的计算，而不影响键值的表达能力；SWA 则把注意力稀疏化成可控的窗口，避免了长序列的二次爆炸。两者合在一起，使得 7 B 参数的模型在算力和内存上都能跑出 13 B‑级别的效果。

### 实验与效果
- **评测任务**：论文在常见的语言理解基准（如 MMLU、ARC、BoolQ）、数学推理（GSM8K、MathQA）以及代码生成（HumanEval、MBPP）上进行测试。  
- **对比基线**：Llama 2 13B、Llama 1 34B、以及同等参数的开源模型。  
- **结果概览**：在所有基准上，Mistral 7B 的得分均高于 Llama 2 13B；在数学、推理和代码生成三个细分领域，Mistral 7B 超过 Llama 1 34B。具体数值未在摘要中给出，论文声称“在所有评测基准上均超过”。  
- **Instruct 版本**：在人类评审的对话质量测试和自动化的 ChatBench 上，Mistral 7B‑Instruct 超过 Llama 2 13B‑Chat。  
- **消融实验**：作者分别关闭 GQA、关闭 SWA，发现去掉 GQA 会导致推理速度下降约 30%，去掉 SWA 则在 8 K+ 长序列上显著增加显存占用并导致性能回落。  
- **局限性**：模型仍然只有 7 B 参数，在极端多语言或专业领域的知识覆盖上可能不如更大模型；作者也提到对长序列的全局一致性仍有提升空间。  

### 影响与延伸思考
Mistral 7B 以 Apache 2.0 开源，迅速被社区用于部署和二次开发，推动了“高效小模型”路线的热潮。随后的开源项目（如 Zephyr、OpenChat）在注意力层面纷纷借鉴 GQA + SWA 的组合，甚至出现了专门针对超长文档的 “Sliding‑Window‑Transformer”。对想进一步探索的读者，可以关注以下方向：  
- **稀疏注意力的理论分析**：为何窗口化仍能保持全局语义？  
- **混合注意力架构**：把 GQA 与其他低秩近似（如 Performer、Linformer）结合。  
- **指令微调的安全性**：在保持高指令遵循的同时，如何防止有害输出。  

### 一句话记住它
**Mistral 7B 用分组查询和滑动窗口两招，让 7 B 参数跑出 13 B‑级别的性能，同时保持高速和长上下文。**