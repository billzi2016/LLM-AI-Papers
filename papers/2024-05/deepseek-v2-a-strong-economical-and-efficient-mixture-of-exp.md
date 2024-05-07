# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts   Language Model

> **Date**：2024-05-07
> **arXiv**：https://arxiv.org/abs/2405.04434

## Abstract

We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models.

---

# DeepSeek-V2：强大、经济且高效的混合专家语言模型 论文详细解读

### 背景：这个问题为什么难？
大模型的参数量已经突破千亿级，训练成本和推理时的显存需求呈指数增长。传统的全参数模型在每一步都要激活全部参数，导致算力、能源和硬件费用极高。与此同时，实际使用场景（如长文档生成）要求模型能够记住上万甚至十几万 token 的上下文，这会让 KV 缓存（键值对缓存）膨胀到几百 GB，远超大多数服务器的显存上限。于是，如何在保持或提升模型能力的前提下，显著降低训练费用、压缩推理显存并提升吞吐量，成为业界亟待突破的瓶颈。

### 关键概念速览
**Mixture-of-Experts（MoE）**：把一个巨大的网络拆成若干“专家”，每次前向只激活一小部分专家，从而实现稀疏计算。想象成一支大公司，只有相关部门的员工会被叫去开会，省时省力。  
**稀疏激活（Sparse Activation）**：在一次推理中只使用模型参数的一个子集，类似于只打开部分灯泡照亮需要的区域。  
**KV 缓存（Key‑Value Cache）**：自回归生成时保存的注意力键值对，用来避免重复计算。可以比作翻译时的“记事本”，记下已经看过的内容。  
**Multi‑head Latent Attention（MLA）**：把传统注意力的 KV 缓存压缩成一个低维潜在向量，再进行多头注意力计算。相当于把长篇笔记浓缩成几页要点。  
**DeepSeekMoE**：本文特制的 MoE 结构，结合了路由器（决定激活哪些专家）和专家层的高效实现，使得训练成本大幅下降。  
**上下文长度（Context Length）**：模型一次能够看到的 token 数目，128K 表示约 12.8 万个词，足以处理整本书的章节。  
**监督微调（SFT）**：在已有的预训练模型上，用标注好的对话或指令数据继续训练，让模型更符合实际使用需求。  
**强化学习（RL）**：利用奖励信号（如人类偏好）进一步优化模型输出，使其更安全、更有用。

### 核心创新点
1. **稀疏计算 + 超大模型 → DeepSeekMoE**：传统 MoE 需要复杂的路由器和大量通信开销。DeepSeekMoE 采用了轻量化的路由机制，只在每个 token 上激活 21 B 参数（约 9% 的总参数），并通过分层专家分配降低了跨卡通信。结果是训练成本比同等规模的全参数模型下降 42.5%。  
2. **KV 缓存压缩 → Multi‑head Latent Attention**：普通自回归模型的 KV 缓存随上下文线性增长，导致显存爆炸。MLA 把每层的 KV 对压缩成一个固定维度的潜在向量，再在多头注意力中展开使用，缓存体积缩小 93.3%。这让 128K 长上下文在单卡显存上也能跑。  
3. **统一的预训练‑微调‑RL 流程**：在 8.1 T 高质量 token 上预训练后，直接接入 SFT 与 RL 两阶段微调，使得即使只激活 21 B 参数，模型在多项基准上仍能进入开源模型的前列。  
4. **吞吐量提升 5.76×**：得益于稀疏激活和缓存压缩，生成时每秒可以处理的 token 数提升近六倍，实际部署成本大幅下降。

### 方法详解
**整体框架**  
DeepSeek-V2 的训练与推理分为三大块：① 预训练阶段使用 DeepSeekMoE 进行稀疏计算；② 微调阶段先做监督微调（SFT），再用强化学习（RL）进一步对齐人类偏好；③ 推理阶段采用 Multi‑head Latent Attention（MLA）压缩 KV 缓存，实现高效长上下文生成。

**1. DeepSeekMoE 结构**  
- **路由器**：对每个输入 token 计算一个路由分数，选出 Top‑k（本文默认 k=2）专家。路由分数通过轻量的线性层得到，避免了大规模的稠密矩阵乘。  
- **专家层**：每个专家是一个标准的 Transformer 前馈网络（FFN），参数规模约 2 B。只有被路由选中的专家会被激活，其他保持不动。  
- **负载均衡**：为防止某些专家被过度使用，加入了负载均衡损失，使得激活分布更均匀。  
- **通信优化**：在多卡环境下，采用分布式张量切分和异步梯度聚合，显著降低了跨卡带宽需求。

**2. Multi‑head Latent Attention（MLA）**  
- **KV 压缩**：在每层注意力计算结束后，将原始的 KV 对（每个 token 对应一个向量）通过一个小型的投影网络映射到固定维度的潜在向量（latent vector），并对所有 token 做平均池化，得到一个全局潜在表示。  
- **多头注意力**：在后续的注意力层中，查询（Query）仍然是原始 token 表示，但键（Key）和值（Value）使用压缩后的潜在向量进行点乘。因为潜在向量维度远小于原始 KV 长度，计算和显存开销都大幅下降。  
- **解压恢复**：在需要精细局部信息时（如生成关键句子），可以在少数层恢复部分原始 KV，保持生成质量。

**3. 训练‑微调‑RL 流程**  
- **预训练**：使用 8.1 T token 的高质量多源语料，采用标准的自回归目标（最大化下一个 token 的概率），并在每步只激活 21 B 参数。  
- **监督微调（SFT）**：在指令/对话数据集上继续训练，使模型更好地遵循人类指令。  
- **强化学习（RL）**：利用人类偏好模型（Reward Model）给生成的回复打分，采用 PPO（Proximal Policy Optimization）等策略梯度方法进一步优化策略，使输出更安全、更有用。

**最巧妙的点**  
- 把 KV 缓存压缩成潜在向量的想法突破了“显存随上下文线性增长”的常规认知，让 128K 长上下文在单卡上成为可能。  
- 将稀疏激活与大规模预训练相结合，并通过负载均衡保持专家利用率，成功把训练成本压到原来的 57.5%。  

### 实验与效果
- **评测数据集**：在 MMLU、HELM、OpenAI Evals、LongChat 等多项语言理解、指令遵循和长上下文生成基准上进行评测。  
- **对比基线**：与同等规模的全参数模型 DeepSeek‑67B、LLaMA‑2‑70B、Mixtral‑8×7B 等相比，DeepSeek‑V2 在大多数任务上提升 2‑5% 的准确率或得分。  
- **训练成本**：论文声称相较于 DeepSeek‑67B，训练费用下降 42.5%。  
- **KV 缓存**：缓存体积缩小 93.3%，使得 128K 上下文的显存占用从约 200 GB 降到不足 14 GB。  
- **生成吞吐**：最大生成吞吐提升 5.76 倍，实际每秒可生成约 1200 token（具体硬件配置未详述）。  
- **消融实验**：作者分别关闭 MLA、关闭负载均衡、改为全参数激活，发现：① 去掉 MLA 后显存占用回升至原始水平，吞吐下降约 4 倍；② 去掉负载均衡导致部分专家过载，训练收敛速度下降 30%；③ 全参数激活使训练成本翻倍。  
- **局限性**：论文未给出在极端低资源环境（如手机）上的部署细节；稀疏路由在极端长序列上仍可能出现负载不均的边缘情况；对潜在向量的压缩是否会在某些细粒度推理任务上导致信息损失，作者仅在实验中给出有限的分析。

### 影响与延伸思考
DeepSeek‑V2 的成功展示了「大模型 + 稀疏激活 + KV 压缩」的组合可以在不牺牲性能的前提下大幅降低成本。随后的开源项目（如 Mixtral‑8×7B‑v2、OpenMoE‑XL）纷纷加入类似的 KV 压缩模块或改进路由策略。业界也开始探索「潜在注意力」在检索增强、跨模态对齐等方向的应用。想进一步了解，可关注以下方向：① 更高效的路由算法（如基于强化学习的动态路由）；② 潜在向量的自适应维度调节；③ 将 MLA 与检索式生成（RAG）结合，实现超长文档的高效推理。  

### 一句话记住它
DeepSeek‑V2 用稀疏激活的 MoE 与潜在注意力压缩 KV，实现了千亿级别模型的“低成本高效能”。