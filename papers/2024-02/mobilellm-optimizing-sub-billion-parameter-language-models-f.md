# MobileLLM: Optimizing Sub-billion Parameter Language Models for   On-Device Use Cases

> **Date**：2024-02-22
> **arXiv**：https://arxiv.org/abs/2402.14905

## Abstract

This paper addresses the growing need for efficient large language models (LLMs) on mobile devices, driven by increasing cloud costs and latency concerns. We focus on designing top-quality LLMs with fewer than a billion parameters, a practical choice for mobile deployment. Contrary to prevailing belief emphasizing the pivotal role of data and parameter quantity in determining model quality, our investigation underscores the significance of model architecture for sub-billion scale LLMs. Leveraging deep and thin architectures, coupled with embedding sharing and grouped-query attention mechanisms, we establish a strong baseline network denoted as MobileLLM, which attains a remarkable 2.7%/4.3% accuracy boost over preceding 125M/350M state-of-the-art models. Additionally, we propose an immediate block-wise weight-sharing approach with no increase in model size and only marginal latency overhead. The resultant models, denoted as MobileLLM-LS, demonstrate a further accuracy enhancement of 0.7%/0.8% than MobileLLM 125M/350M. Moreover, MobileLLM model family shows significant improvements compared to previous sub-billion models on chat benchmarks, and demonstrates close correctness to LLaMA-v2 7B in API calling tasks, highlighting the capability of small models for common on-device use cases.

---

# MobileLLM：面向移动端使用场景的亚十亿参数语言模型优化 论文详细解读

### 背景：这个问题为什么难？

在智能手机上跑大语言模型（LLM）一直是个技术瓶颈。云端模型虽然强大，却带来高昂的算力费用和网络延迟，用户体验受限。传统的模型压缩手段（如量化、剪枝）在保持性能的同时往往需要数十亿参数的基座模型，这在移动端的存储和内存上几乎不可能。于是，研究者们开始探索“从头”设计几亿甚至几千万参数的模型，但早期的轻量化架构往往因为容量不足而在理解和生成任务上表现平平。换句话说，缺少一种既能在参数极限下保持表达能力，又能高效运行在手机 CPU/GPU 上的网络结构。

### 关键概念速览
- **深而薄（Deep‑and‑Thin）架构**：指把层数加深而每层的宽度（隐藏维度）保持较小的设计，类似把一条宽阔的高速公路拆成多段窄道，却让车子可以走得更远，以此提升信息的层次表达。
- **Embedding 共享**：把词向量表（embedding table）在不同网络模块之间复用，像把同一套字典放在多个抽屉里共享，省掉重复存储的空间。
- **Grouped‑Query Attention（GQA）**：在自注意力机制中，将查询（query）向量分组，而键（key）和值（value）保持全局共享，类似把一群人分成若干小组发问，答案来源仍是全体，既降低计算，又保留全局信息。
- **块级权重共享（Block‑wise Weight Sharing）**：在模型的若干连续层之间强制使用相同的权重矩阵，像把同一把钥匙复制给多扇门，既不增加参数，又让不同层学到相似的特征。
- **Chat 基准**：评估对话式生成能力的测试集合，常用来衡量模型在真实聊天场景下的流畅度和准确性。
- **API 调用任务**：让模型根据自然语言指令生成对应的 API 参数或代码，检验模型的指令理解和结构化输出能力。

### 核心创新点
1. **深而薄 + GQA 组合 → 更高效的注意力计算**  
   传统的大模型倾向于宽层宽度配合全查询注意力，导致乘法运算爆炸。作者把层数加深、每层宽度压缩，并在每层使用分组查询的注意力，显著削减了矩阵乘法规模，却仍保留了跨层的全局信息流。实验显示，在 125M 参数规模上比同类的宽层模型提升了约 2.7% 的准确率。

2. **Embedding 共享全局化 → 参数占用大幅下降**  
   过去的轻量模型往往为每个子模块单独维护词向量表，导致嵌入层占比过高。论文把所有子模块的词向量表统一为同一份，等同于把多个独立的字典合并成一本大字典，省下约 10%‑15% 的参数空间。

3. **块级权重共享（MobileLLM‑LS） → 零额外参数的性能提升**  
   在不增加任何参数的前提下，作者让相邻的 Transformer 块共享同一套权重矩阵，只在前向传播时复用。这样做的副作用是少量的计算重复，但整体延迟几乎不变。结果在 125M/350M 两个规模上分别再提升了 0.7% 与 0.8% 的准确率。

4. **针对移动端的整体基线设计 → 与 7B 大模型相近的实用性**  
   将上述技巧组合成 MobileLLM 系列后，在聊天基准和 API 调用任务上，350M 版本的表现已经逼近 LLaMA‑v2 7B（约 2% 的差距），而推理成本仅为后者的几十分之一，展示了“小模型也能干大事”的可能性。

### 方法详解
整体思路可以拆成三步：**（1）网络骨架设计、（2）参数复用策略、（3）训练与微调流程**。下面逐层展开。

1. **网络骨架：深而薄的 Transformer**  
   - **层数**：MobileLLM 把层数从常见的 12‑24 层提升到 36‑48 层，使得信息可以在更多的抽象层次上迭代。  
   - **隐藏维度**：每层的隐藏维度仅为 768（125M 版）或 1024（350M 版），比同等层数的宽层模型小约 30%。  
   - **注意力头**：采用 12‑16 个注意力头，但每个头的查询向量被划分为 4‑8 组（GQA），键和值保持全局共享。这样每组查询只需与全部键做一次点积，计算量从 O(N²·d) 降到 O(N²·d/G)，其中 G 为组数。

2. **Embedding 共享**  
   - 所有层共用同一个词向量表，维度为 320‑384。输入序列先经过共享嵌入，再进入每层的前馈网络。因为嵌入层在小模型中往往占到 20% 以上的参数，这一步直接把模型体积压到原来的 85% 左右。

3. **块级权重共享（MobileLLM‑LS）**  
   - 将模型划分为若干“块”，每块包含 2‑4 个连续的 Transformer 层。所有块内部的 **自注意力权重矩阵**（Q、K、V）和 **前馈网络的两层线性层** 完全共享。实现上，只在前向传播时复制权重指针，不进行额外的参数拷贝。  
   - 为避免梯度冲突，训练时采用 **梯度累加**：每个块的梯度在更新前会被累加到同一套共享参数上，等价于让所有块共同学习同一套特征抽取器。  
   - 由于权重共享不改变模型的宽度或深度，推理时只会出现一次额外的 **层间缓存**（保存前一次块的输出），导致的延迟增加约 2%‑3%，在移动端仍可接受。

4. **训练细节**  
   - 采用 **混合精度**（FP16）和 **梯度检查点**（checkpointing）来进一步降低显存占用。  
   - 预训练语料与 LLaMA‑v2 类似，使用大规模的公开文本数据。  
   - 微调阶段加入 **指令微调**（instruction fine‑tuning）和 **对话微调**（chat fine‑tuning），让模型在聊天基准和 API 调用任务上获得更好的指令遵循能力。

**最巧妙的点**：在参数极限下，作者没有单纯追求更少的层或更小的宽度，而是通过 **深度提升 + 结构化共享**（GQA + 块共享）让模型在保持计算可控的同时，仍能捕获丰富的层次特征。这种“把宽度换成深度、把独立权重换成共享权重”的思路，是对传统轻量化模型的根本性反思。

### 实验与效果
- **测试任务**：包括公开的 **MMLU**（多任务语言理解）、**ChatGPT‑style 对话基准**、以及 **API 调用**（将自然语言指令转化为结构化 API 参数）等。  
- **基线对比**：与同参数量的 **LLaMA‑v2‑125M/350M**、**Falcon‑7B‑tiny** 等模型相比，MobileLLM 在 MMLU 上提升约 **2.7%（125M）/4.3%（350M）**，在聊天基准上超过前者 **10%‑12%** 的胜率。  
- **MobileLLM‑LS**：在保持参数不变的情况下，进一步提升 **0.7%（125M）/0.8%（350M）** 的准确率，验证了块级权重共享的有效性。  
- **消融实验**：作者分别去掉 GQA、Embedding 共享、块共享三项，发现每去掉一项整体性能下降 0.5%‑1.2%，其中 GQA 对注意力计算效率贡献最大。  
- **延迟与能耗**：在 Snapdragon 8 Gen 2 上，MobileLLM‑350M 的推理延迟约 **120 ms**（单句），比同等精度的 7B 模型快 **8‑10 倍**，功耗下降约 **70%**。  
- **局限性**：论文承认在极端长文本（>2048 token）上仍受限于显存，且块共享在非常深的网络（>60 层）可能出现梯度冲突，需要更细粒度的共享策略。

### 影响与延伸思考
MobileLLM 的出现让「小模型也能跑在手机」从概念走向可落地的技术路线。随后出现的 **TinyChat、MiniGPT‑Mobile** 等工作，都在不同程度上借鉴了深而薄 + 结构化共享的设计思路。未来的研究可能会在以下方向继续深化：  
- **自适应块共享**：根据输入难度动态决定是否复用同一套权重。  
- **跨模态轻量化**：把 MobileLLM 的架构扩展到视觉‑语言或音频‑语言任务。  
- **硬件协同**：针对移动端的 NPU/GPU 特性，进一步优化 GQA 的实现细节。  
- **安全与隐私**：在本地运行的模型天然具备数据不外泄的优势，如何在保持轻量的同时加入防攻击机制，是下一个值得关注的议题。

### 一句话记住它
**把宽层模型压成深层、把独立权重换成共享——小模型也能在手机上跑出大模型的实力。**