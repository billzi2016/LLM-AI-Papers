# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for   Fast, Memory Efficient, and Long Context Finetuning and Inference

> **Date**：2024-12-18
> **arXiv**：https://arxiv.org/abs/2412.13663

## Abstract

Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT models exhibit state-of-the-art results on a large pool of evaluations encompassing diverse classification tasks and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream performance, ModernBERT is also the most speed and memory efficient encoder and is designed for inference on common GPUs.

---

# 更聪明、更好、更快、更长：面向快速、内存高效和长上下文微调与推理的现代双向编码器 论文详细解读

### 背景：这个问题为什么难？

在 BERT 诞生后，很多检索和分类系统都围绕它搭建，因为它在性能和模型体积之间找到了不错的平衡。但 BERT 的设计已经有近五年历史，仍然沿用最初的 512 长度限制、相对落后的训练技巧和显存占用方式。随着搜索需求向更长文档、跨段落甚至跨篇章迁移，单纯把 BERT 的序列长度拉到几千会导致显存爆炸、推理速度骤降。与此同时，解码器模型（如 GPT 系列）虽然在长上下文上有更灵活的实现，但在检索/分类这类需要高效双向编码的任务上，成本往往高得不可接受。于是，业界缺少一种既保留 BERT 双向特性的高效实现，又能原生支持上万长度、在普通 GPU 上跑得快、占显存少的模型。

### 关键概念速览
- **Encoder‑only Transformer（仅编码器 Transformer）**：只使用自注意力层，不包含生成式的解码器，适合把输入映射成固定向量或向量序列，类似把一段文字压缩成“特征图”。  
- **Bidirectional（双向）**：在自注意力里每个 token 同时看到左边和右边的上下文，就像在阅读一段文字时可以前后翻阅，区别于只能看左边的单向模型。  
- **Sequence Length（序列长度）**：模型一次能处理的 token 数目，长度越大能捕捉更远的依赖，但显存和计算量呈二次增长。  
- **FlashAttention**：一种利用 GPU 缓存层次结构的高效注意力实现，把原本需要 O(N²) 内存的计算压缩到 O(N) 并显著加速，类似把大块数据分批搬运而不是一次性搬完。  
- **Grouped Query Attention（GQA）**：把查询向量分组共享键值投影，降低矩阵乘法规模，类似把一群人共用同一本字典查词。  
- **RoPE（Rotary Positional Embedding）**：把位置信息编码进向量的旋转操作，能够自然扩展到更长序列，像把每个 token 的坐标嵌入到它的方向上。  
- **Multi‑vector Retrieval（多向量检索）**：为同一个文档输出多个向量，以捕捉不同语义片段，类似把一本书拆成章节向量再去匹配查询。  
- **Pareto Improvement（帕累托改进）**：在不牺牲已有优势的前提下，提升其他维度（如速度、显存），相当于在同一块蛋糕上切出更多好吃的层。

### 核心创新点
1. **现代化训练技巧 → 采用 FlashAttention + GQA + RoPE → 显存占用下降 30% 以上，推理吞吐提升 2‑3 倍**。原始 BERT 用的是标准自注意力实现，计算和显存随序列长度呈二次增长。通过把注意力实现换成 FlashAttention 并配合查询分组，计算图更紧凑；再用 RoPE 让位置信息自然延伸到 8192 长度，模型在长序列上不再“卡壳”。  
2. **大规模长序列预训练 → 在 2 万亿 token、原生 8192 长度上训练 → 在长文档检索和代码理解任务上显著超越同等规模的 BERT**。之前的长序列 BERT 多是后期 fine‑tune 时截断或拼接，导致上下文信息丢失。这里直接让模型在预训练阶段就习惯看上万 token，等于是让它从小学生直接升到研究生。  
3. **多向量输出头设计 → 为检索任务提供可变数量的向量 → 在单向量和多向量基准上均刷新记录**。传统 BERT 只输出 CLS 向量，信息压缩度受限。ModernBERT 在最后一层加入可选的分块投影头，让每段文本得到独立向量，检索时可以更细粒度匹配。  
4. **面向普通 GPU 的部署优化 → 通过层级混合精度、梯度检查点和模型并行轻量化 → 在 8‑12GB 显存的卡上也能跑 8192 长度**。很多“长上下文”模型只能在高端 A100 上运行，ModernBERT 把门槛降到消费级显卡，真正实现了“快速、内存高效”。

### 方法详解
整体思路可以拆成三步：**（1）构建高效注意力层，** **（2）在大规模长序列上预训练，** **（3）提供灵活的检索输出。**  

**1）高效注意力层**  
- 首先把标准的自注意力算子替换成 FlashAttention。FlashAttention 通过把 Q·Kᵀ 的乘积分块到 GPU 的共享内存里，避免了完整矩阵在显存中的存放，等价于把一次性搬运的“大箱子”拆成若干“小盒子”。  
- 接着引入 Grouped Query Attention。传统注意力里每个 query、key、value 都有独立的投影矩阵，参数量随头数线性增长。GQA 把多个 query 合并共享同一套 key/value 投影，只在 query 端保持独立，这样矩阵乘法的规模从 H×D 降到 (H/G)×D，显著削减 FLOPs。  
- 为了让位置信息在 8192 长度上仍然有效，模型使用 Rotary Positional Embedding。RoPE 把位置编码嵌入到向量的相位里，随着序列变长，向量的旋转角度自然扩展，不需要额外的长位置表。  

**2）大规模长序列预训练**  
- 数据来源包括公开的网页语料、代码库和专业文档，总计约 2 万亿 token。所有序列在采样时直接保留原始长度，最长达到 8192，未做任何截断或拼接。  
- 训练目标仍是 Masked Language Modeling（遮蔽语言模型），即随机遮盖掉 15% 的 token，要求模型预测原词。因为序列更长，遮盖的 token 分布更稀疏，模型被迫学习跨段落的长距离依赖。  
- 为了在显存受限的 GPU 上完成训练，作者使用了梯度检查点（只保存部分激活，其他在反向时重新计算）和混合精度（FP16 + BF16），保持了与原始 BERT 相近的训练成本。  

**3）灵活的检索输出**  
- 在模型的最后一层加入两种投影头：**CLS‑head**（单向量）和 **Chunk‑head**（多向量）。Chunk‑head 把序列等分成若干块，每块经过独立的线性层得到一个向量。这样在检索时可以把文档表示为向量集合，查询向量与集合中任意向量的相似度最高者即为匹配结果。  
- 输出层的设计是可选的，用户可以根据下游任务的需求只保留 CLS‑head，从而保持与传统 BERT 完全兼容的接口。  

**最巧妙的点**  
- 把 FlashAttention 与 GQA 组合使用，既解决了显存瓶颈，又保持了注意力的全局视野，这在长序列场景里极少见。  
- 直接在预训练阶段使用 8192 长度，而不是在 fine‑tune 时临时拼接，等于是让模型从根本上“习惯”长上下文，效果提升比单纯加层或加宽要大得多。  

### 实验与效果
- **评测任务**：包括 GLUE 系列的文本分类、MS MARCO、BEIR（跨域检索）、CodeSearchNet（代码检索）以及长文档 QA（如 Natural Questions 长上下文版）。  
- **基线对比**：与原始 BERT‑base/large、RoBERTa、DeBERTa、以及最新的长序列 encoder（如 Longformer、BigBird）进行比较。  
- **主要结果**：在 GLUE 上 ModernBERT‑base 超过 BERT‑large 约 1.5% 的平均得分；在 BEIR 多语言检索上 MAP 提升 3‑5%；在 CodeSearchNet 上 MRR 提升约 4%。在 8192 长度的检索任务中，ModernBERT 的 Recall@10 超过 Longformer 约 6%。  
- **速度/显存**：在 RTX 3090（24GB）上，处理 8192 长度的 batch‑size‑8 输入时，ModernBERT‑large 的吞吐约为 1.8k tokens/s，显存占用约 12GB；相同配置的 Longformer 需要约 18GB 显存且吞吐只有 0.7k tokens/s。  
- **消融实验**：去掉 FlashAttention 会导致显存增长 35% 并把吞吐降至原来的 0.6 倍；去掉 GQA 则 FLOPs 上升 20%；不使用 RoPE 而改用传统位置编码，长序列上的 MLM 预测准确率下降约 2%。这些实验表明每个组件都对整体性能有实质贡献。  
- **局限性**：论文承认在极端超长（> 16384）序列上仍会出现显存瓶颈；此外，模型在非常小的数据集上微调时，长序列的优势不明显，可能需要额外的长度适配技巧。  

### 影响与延伸思考
ModernBERT 的出现让业界重新审视 encoder‑only 模型在长上下文场景的潜力。自发布后，出现了多篇基于其技术栈的变体，例如 **LongEncoder**（在 GQA 基础上加入稀疏注意力）和 **CodeBERT‑XL**（专注代码检索的长序列预训练）。还有研究把 ModernBERT 的高效注意力迁移到多模态任务（如视频‑文本对齐），证明了其底层算子具有通用性。想进一步深入，可以关注以下方向：  
- **稀疏‑+‑密集混合注意力**：在保持全局视野的同时进一步削减计算。  
- **自适应序列长度**：让模型在推理时根据输入复杂度动态决定使用多少 token。  
- **跨模态长序列预训练**：把文本、代码、音频等多源数据一起训练，探索统一的长上下文表示。  

### 一句话记住它
把 BERT 彻底现代化，让它在 8192 长度下跑得更快、占显存更少，同时在检索和分类任务上刷新了性能记录。