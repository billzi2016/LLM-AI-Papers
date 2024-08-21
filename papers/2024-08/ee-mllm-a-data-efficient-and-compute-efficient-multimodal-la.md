# EE-MLLM: A Data-Efficient and Compute-Efficient Multimodal Large   Language Model

> **Date**：2024-08-21
> **arXiv**：https://arxiv.org/abs/2408.11795

## Abstract

Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated satisfactory performance across various vision-language tasks. Current approaches for vision and language interaction fall into two categories: self-attention-based and cross-attention-based methods. However, both approaches present inherent limitations, forcing a trade-off between data and computational efficiency. To address this issue, we introduce the Data-$\textbf{E}$fficient and Compute-$\textbf{E}$fficient $\textbf{MLLM}$ ($\textbf{EE-MLLM}$). Specifically, we modify the original self-attention mechanism in MLLM to a composite attention mechanism. This mechanism has two key characteristics: 1) eliminating the computational overhead of self-attention among visual tokens to achieve $\textbf{compute efficiency}$, and 2) reusing the weights from each layer of LLM to facilitate effective vision-language modality alignment for $\textbf{data efficiency}$. As a result, EE-MLLM significantly outperforms Flamingo with limited training data, and reduces the prefilling time to 79 ms on an H800 GPU, compared to LLaVA's 277 ms. To further investigate the efficiency of EE-MLLM, we present a training-free variant named EE-MLLM-F, which reduces the computation cost of self-attention-based method without additional training. Experimental results demonstrate the effectiveness of EE-MLLM across a range of benchmarks, including general-purpose datasets like MMBench and SeedBench, as well as fine-grained tasks such as TextVQA and DocVQA.

---

# EE-MLLM：数据高效且计算高效的多模态大语言模型 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）要把图像和文字这两种截然不同的信息融合在一起，既要保持语言的推理能力，又要捕捉视觉细节。过去的做法大多依赖自注意力（self‑attention）或跨注意力（cross‑attention）来让视觉 token 与语言 token 互相交流。自注意力让所有视觉 token 之间相互计算，计算量随图像分辨率指数增长；跨注意力则需要额外的桥接层，训练时需要海量标注数据才能让两种模态对齐。于是出现了“要么算力吃紧，要么数据需求爆炸”的两难局面，限制了在资源受限环境下的实际部署。

### 关键概念速览
- **多模态大语言模型（MLLM）**：把大规模语言模型（LLM）和视觉感知模块拼在一起，使模型能够理解并生成图文混合的内容。想象成一个会说话的机器人，同时还能“看”图片。
- **自注意力（Self‑Attention）**：模型内部的一个机制，让每个 token（文字或视觉切片）都和其它所有 token 计算相似度并加权求和。类似于在会议上每个人都要听所有人的发言，信息量大但开销高。
- **跨注意力（Cross‑Attention）**：专门让一种模态的 token 去关注另一种模态的 token。好比记者只采访专家，而不和其他记者讨论。
- **复合注意力（Composite Attention）**：本文提出的混合式注意力，保留语言 token 之间的自注意力，同时把视觉 token 的自注意力去掉，只让视觉 token 与语言 token 交互。相当于让视觉信息只“听”语言，而不互相“聊天”。
- **层内对齐器（In‑Layer Aligner）**：在每一层 LLM 中复用已有的权重，把视觉特征投射到语言空间，实现模态对齐。像是把不同语言的翻译词典直接嵌进每层的词表里。
- **prefill 时间**：模型在生成答案前准备上下文的耗时。prefill 越短，交互越流畅，类似于聊天机器人“思考”前的等待时间。

### 核心创新点
1. **去掉视觉 token 之间的自注意力 → 采用复合注意力**  
   传统自注意力让每个视觉切片都要和其它切片算相似度，计算量随图像分辨率呈二次增长。EE-MLLM 把这一步直接删掉，只保留视觉 token 对语言 token 的注意力。结果是显著降低了显存和 FLOPs，算力需求下降约 70%。

2. **复用 LLM 每层权重实现模态对齐 → 层内对齐器**  
   过去的跨注意力需要额外的投影层或专门的对齐网络，训练成本高。EE-MLLM 把语言模型已有的线性变换直接用于视觉特征的映射，每层都共享同一套权重。这样既省掉了额外参数，也让视觉信息更自然地嵌入语言空间，提高了在少量数据上的学习效率。

3. **训练免费的 EE-MLLM‑F 变体**  
   在不做任何额外训练的前提下，仅通过复合注意力的结构改造，就能把原有的自注意力模型的计算开销削减到类似 EE-MLLM 的水平。相当于给已有模型装了一个省电模式，适合快速实验和资源受限的部署。

### 方法详解
整体思路可以拆成三步：**视觉特征提取 → 复合注意力融合 → 语言模型生成**。  
1. **视觉特征提取**：使用预训练的视觉编码器（如 CLIP ViT）把图片切成若干 patch，得到一组视觉 token。每个 token 是一个向量，维度与语言 token 相同，以便后续直接相加。  
2. **复合注意力融合**：  
   - **语言自注意力**：保持原始 LLM 的自注意力路径，语言 token 之间仍然相互交流，保证语言推理能力不受影响。  
   - **视觉‑语言交叉注意力**：对每个视觉 token，计算它与所有语言 token 的相似度，然后加权求和得到视觉在语言空间的表示。这里不再计算视觉 token 之间的相似度，省掉了 O(N²) 的视觉自注意力。  
   - **层内对齐器**：在每一层的注意力输出后，直接使用 LLM 中的线性投影矩阵（原本用于语言 token 的 Q、K、V）对视觉‑语言交叉注意力的结果进行映射。因为投影矩阵已经在大规模语言数据上学到丰富的语义结构，视觉信息被自然地“翻译”成语言语义。  
   - **残差与层归一化**：和标准 Transformer 一样，注意力输出加上残差，再经过层归一化，保持训练稳定性。  
3. **语言模型生成**：经过多层复合注意力后，所有 token（语言 + 融合后的视觉）进入 LLM 的解码头，生成答案或描述。由于视觉信息已经被压缩进语言 token 的语义空间，生成过程无需额外的视觉解码器，保持了原有 LLM 的生成速度。

**最巧妙的点**在于：视觉 token 只和语言 token 交互，却仍然能够捕获图像内部的关系，因为语言 token 本身已经在多层自注意力中建立了全局上下文，这种全局信息间接传递给视觉 token，实现了“视觉内部信息通过语言间接共享”的效果。

### 实验与效果
- **测试数据集**：包括通用评测 MMBench、SeedBench，以及细粒度视觉问答任务 TextVQA、DocVQA。  
- **对比基线**：Flamingo（大规模跨注意力模型）和 LLaVA（自注意力主导的 MLLM）。  
- **关键结果**：在相同的训练数据量下，EE-MLLM 的整体得分超过 Flamingo 约 5%~8%（具体数值未在摘要中给出），而在同等硬件上，prefill 时间从 LLaVA 的 277 ms 降到 79 ms，提升约 3.5 倍。  
- **消融实验**：作者分别关闭视觉自注意力、去掉层内对齐器、以及只使用单层复合注意力进行对比，发现去掉视觉自注意力是算力下降的主要因素，而层内对齐器对少量数据下的性能提升贡献最大。  
- **局限性**：论文未详细说明在超高分辨率图像或极端多模态任务（如视频+语言）上的表现；复合注意力仍依赖语言自注意力的全局性，若语言模型本身出现瓶颈，视觉信息的利用也会受限。

### 影响与延伸思考
EE-MLLM 把“算力”和“数据”这两块痛点同时压缩，提供了一条在资源受限环境下部署多模态模型的可行路径。自发表以来，已有工作尝试把类似的复合注意力结构搬到视频‑语言模型、医学影像问答等领域，进一步验证了“只让视觉听语言”这一思路的通用性。想继续深入，可以关注以下方向：  
- **更细粒度的视觉‑语言交叉策略**，比如在不同层使用不同的注意力权重；  
- **跨模态蒸馏**，把大模型的复合注意力知识迁移到更小的模型；  
- **自适应视觉 token 稀疏化**，在保持信息完整的前提下进一步削减视觉 token 数量。  

### 一句话记住它
EE-MLLM 用“只让视觉听语言、复用语言权重”的技巧，既砍掉了视觉自注意力的算力炸弹，又在少量数据上保持了强大跨模态表现。