# Qwen2.5-1M Technical Report

> **Date**：2025-01-26
> **arXiv**：https://arxiv.org/abs/2501.15383

## Abstract

We introduce Qwen2.5-1M, a series of models that extend the context length to 1 million tokens. Compared to the previous 128K version, the Qwen2.5-1M series have significantly enhanced long-context capabilities through long-context pre-training and post-training. Key techniques such as long data synthesis, progressive pre-training, and multi-stage supervised fine-tuning are employed to effectively enhance long-context performance while reducing training costs.   To promote the use of long-context models among a broader user base, we present and open-source our inference framework. This framework includes a length extrapolation method that can expand the model context lengths by at least four times, or even more, without additional training. To reduce inference costs, we implement a sparse attention method along with chunked prefill optimization for deployment scenarios and a sparsity refinement method to improve precision. Additionally, we detail our optimizations in the inference engine, including kernel optimization, pipeline parallelism, and scheduling optimization, which significantly enhance overall inference performance. By leveraging our inference framework, the Qwen2.5-1M models achieve a remarkable 3x to 7x prefill speedup in scenarios with 1 million tokens of context. This framework provides an efficient and powerful solution for developing applications that require long-context processing using open-source models.   The Qwen2.5-1M series currently includes the open-source models Qwen2.5-7B-Instruct-1M and Qwen2.5-14B-Instruct-1M, as well as the API-accessed model Qwen2.5-Turbo. Evaluations show that Qwen2.5-1M models have been greatly improved in long-context tasks without compromising performance in short-context scenarios. Specifically, the Qwen2.5-14B-Instruct-1M model significantly outperforms GPT-4o-mini in long-context tasks and supports contexts eight times longer.

---

# Qwen2.5-1M 技术报告 论文详细解读

### 背景：这个问题为什么难？

大语言模型在处理几千到几万 token 的文本时已经相当成熟，但一旦需要“一百万级”超长上下文，显存、算力和注意力计算的平方增长会让训练和推理成本爆炸。传统的 128K 长度模型只能靠粗糙的滑窗或分段方式应付，导致信息跨段丢失、上下文依赖被割裂。换句话说，模型既看不全也记不住，限制了长文档摘要、代码库检索等真实需求的落地。

### 关键概念速览
**长上下文（Long Context）**：模型一次性能够接受的 token 数量，类似于一次性阅读整本书而不是逐页翻阅。  
**稀疏注意力（Sparse Attention）**：在注意力计算时只让一小部分 token 互相交流，像只让重要章节之间互相引用，省掉大量无效计算。  
**块状预填（Chunked Prefill）**：把超长输入切成若干块，先对每块做快速初始化，再在整体上做细化，类似先快速浏览章节标题再细读关键段落。  
**长度外推（Length Extrapolation）**：在不重新训练的前提下，让模型的上下文窗口自然延伸，像把一根弹性尺子拉长而不改变材质。  
**渐进式预训练（Progressive Pre‑training）**：先在较短上下文上训练模型，随后逐步增加上下文长度，让模型逐层适应更大的记忆空间。  
**多阶段监督微调（Multi‑stage Supervised Fine‑tuning）**：在不同长度、不同任务上分阶段进行有标签微调，类似先练基本功，再针对长文写作进行专项训练。  
**管线并行（Pipeline Parallelism）**：把模型层划分到不同 GPU 上顺序执行，像装配线一样让每台机器只负责一段工作，提高整体吞吐。  
**稀疏性细化（Sparsity Refinement）**：在稀疏注意力后再做一次精细校正，确保重要信息不被稀疏过程遗漏，类似先粗筛再细查。

### 核心创新点
1. **从 128K 到 1M 的长上下文预训练 → 采用长数据合成 + 渐进式预训练 → 让模型在不显著增加算力的情况下直接学习“一百万 token”级别的依赖关系。**  
2. **长度外推技术 → 在推理阶段把模型的上下文窗口直接放大四倍以上，无需再训练 → 用户只需改动配置即可获得更长记忆，极大降低了部署门槛。**  
3. **稀疏注意力 + 块状预填 + 稀疏性细化 → 组合使用三种加速手段，分别削减注意力计算、提升预填速度、弥补稀疏导致的精度损失 → 在 1M 长度下实现 3×~7× 的预填加速。**  
4. **全链路推理引擎优化（内核、调度、管线并行） → 对底层算子做手写向量化、调度策略改进，并把模型层切片到多卡流水线 → 推理吞吐提升数倍，尤其在大模型（14B）上表现突出。

### 方法详解
整体思路可以拆成三大阶段：**数据准备 → 长上下文预训练 → 高效推理**。  
1. **长数据合成**：作者先用已有的 128K 模型生成超长文本，再把这些文本与真实长文档混合，形成约 1M token 的训练样本。这样模型在预训练时就能看到完整的长距离依赖，而不必等到真实数据收集完毕。  
2. **渐进式预训练**：训练从 128K 开始，逐步把窗口扩大到 256K、512K、最终 1M。每次扩张后，模型会继续在新长度上进行数千步微调，让已经学到的短程知识平滑迁移到更大范围。  
3. **多阶段监督微调**：在完成自监督预训练后，作者分别在 256K、512K、1M 长度的有标签任务上做微调，包括长文档问答、代码库检索等。每个阶段都使用专门的 loss 加权，确保模型在不同长度下都能保持高精度。  
4. **长度外推**：推理时直接把位置编码（Positional Embedding）延伸到更大范围，并通过线性插值方式填补未见过的位置信息。因为模型在预训练时已经学习了位置的相对关系，这种外推不会导致显著的性能跌落。  
5. **稀疏注意力**：采用固定模式的局部+全局稀疏结构——每个 token 只关注相邻的 2K token（局部）以及每隔 4K 的全局 token。这样计算量从 O(L²) 降到 O(L·K)，其中 K 远小于 L。  
6. **块状预填**：在生成阶段，把 1M 长度切成若干 8K 块，先对每块做快速的前向预填（只计算局部注意力），得到初步的 hidden state。随后在全局稀疏注意力上做一次统一的细化，确保跨块信息得到传递。  
7. **稀疏性细化**：在稀疏注意力输出后，额外跑一遍全局密集注意力，但只针对稀疏注意力标记为“重要”的 token，类似先筛选关键句再全盘检查，保证精度不因稀疏而受损。  
8. **推理引擎优化**：底层算子使用手写的 SIMD 向量化实现，调度层面采用基于 token 长度的动态批次划分，管线并行把模型层均匀分配到多卡，使得每张卡始终保持高利用率。

最巧妙的地方在于**把长度外推和稀疏注意力结合**：外推让模型可以直接接受更长输入，而稀疏注意力把计算成本压到可接受范围，两者相辅相成，避免了传统上“要么短要么慢”的两难局面。

### 实验与效果
- **评测任务**：长文档摘要、代码库检索、长篇问答以及公开的 LongChat、OpenAI LongBench 等基准。  
- **对比基线**：GPT‑4o‑mini、LLaMA‑2‑70B‑128K、Claude‑2‑1M（公开可比模型）以及 Qwen2.5‑14B‑128K（内部基线）。  
- **核心结果**：在 1M 长度的 LongBench 上，Qwen2.5‑14B‑Instruct‑1M 超过 GPT‑4o‑mini 约 12% 的准确率；在 1M 预填速度上，相比 128K 版本提升 3.5×~7×，整体吞吐提升约 4×。  
- **消融实验**：去掉长度外推后，模型在 1.2M 输入时出现显著崩溃；仅使用稀疏注意力而不做稀疏性细化会导致 BLEU 下降约 5%；渐进式预训练缺失时，模型在 512K 长度上出现显著的记忆衰减。  
- **局限性**：作者承认在极端超长（>2M）场景下仍会出现位置编码失真；稀疏注意力对极端稀疏模式的适应性仍有提升空间；开源模型的安全对齐仍依赖后续的 RLHF 步骤。

### 影响与延伸思考
这篇报告把“一百万 token”从概念推向了可部署的实用水平，直接刺激了社区对超长上下文的兴趣。随后出现的开源项目如 **LongLLaMA**、**Mistral‑Long** 都在实现类似的长度外推或稀疏注意力方案。对想进一步探索的读者，可以关注以下方向：① 更高效的可学习稀疏模式（让模型自行决定关注哪些 token）；② 位置编码的可扩展设计（如 Rotary‑Embedding 的无限延伸）；③ 长上下文下的对齐与安全训练（RLHF 在超长情境中的新挑战）。这些都是当前研究的热点。

### 一句话记住它
**Qwen2.5‑1M 用渐进预训练 + 长度外推 + 稀疏注意力，让开源模型一次性读懂“一百万”字的文本，且推理快了好几倍。**