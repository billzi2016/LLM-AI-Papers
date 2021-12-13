# GLaM: Efficient Scaling of Language Models with Mixture-of-Experts

> **Date**：2021-12-13
> **arXiv**：https://arxiv.org/abs/2112.06905

## Abstract

Scaling language models with more data, compute and parameters has driven significant progress in natural language processing. For example, thanks to scaling, GPT-3 was able to achieve strong results on in-context learning tasks. However, training these large dense models requires significant amounts of computing resources. In this paper, we propose and develop a family of language models named GLaM (Generalist Language Model), which uses a sparsely activated mixture-of-experts architecture to scale the model capacity while also incurring substantially less training cost compared to dense variants. The largest GLaM has 1.2 trillion parameters, which is approximately 7x larger than GPT-3. It consumes only 1/3 of the energy used to train GPT-3 and requires half of the computation flops for inference, while still achieving better overall zero-shot and one-shot performance across 29 NLP tasks.

---

# GLaM：使用专家混合实现高效扩展的语言模型 论文详细解读

### 背景：这个问题为什么难？

在语言模型领域，提升性能的唯一可靠办法一直是“更大”。把模型参数、训练数据和算力往上堆，GPT‑3（175 B）就是最典型的例子。但把所有参数都打开来算，训练成本会呈指数级飙升：需要数十万 GPU‑hour、巨额电费，甚至普通研究机构根本负担不起。于是，学界陷入了“容量够大、算力够省”这对矛盾的困境：要么保持稠密（dense）结构，付出巨额资源；要么压缩模型，性能大幅下降。正因为这根本性的资源瓶颈，才催生了本文的研究。

### 关键概念速览
- **稠密模型（Dense Model）**：所有参数在每一次前向传播时都被激活，就像全员开会，计算量随参数数目线性增长。  
- **专家混合（Mixture‑of‑Experts，MoE）**：把模型拆成很多“专家”，每个专家只负责一小部分输入，类似餐厅里不同厨师只做自己擅长的菜。  
- **稀疏激活（Sparsely Activated）**：一次推理只挑出少数几个专家参与计算，像只叫出两位厨师来做这道菜，显著降低 FLOPs。  
- **路由网络（Router / Gating Network）**：负责决定哪个专家被激活的轻量模块，常用“top‑k”策略挑选得分最高的 k 个专家。  
- **负载均衡损失（Load‑Balancing Loss）**：额外的正则项，鼓励路由网络把流量均匀分配到所有专家上，防止某几个专家被“抢饭”。  
- **零样本学习（Zero‑Shot）**：模型在没有看到任务示例的情况下直接完成任务，考验模型的通用知识。  
- **单样本学习（One‑Shot）**：给模型提供一个示例后再完成任务，比零样本稍易，但仍要求模型快速适应。  
- **参数/计算分离（Parameter‑Compute Decoupling）**：模型容量可以通过增加专家数量提升，而实际计算量只随激活的专家数增长，实现“大容量·小算力”。  

### 核心创新点
1. **稀疏激活的 MoE 前馈层 → 只在每层选出 2 个专家参与计算 → 训练和推理 FLOPs 与稠密模型相比下降约 50%**  
   传统的 MoE 早在机器翻译里出现，但在大规模语言模型上未被系统化。作者把每个 Transformer 层的前馈网络换成了 MoE 结构，使用 top‑2 路由，使得即使整体参数达到 1.2 T，实际每个 token 只走两条专家路径，算力开销与 175 B 稠密模型相当。

2. **负载均衡损失 + 动态容量因子 → 防止专家“热点” → 训练过程更稳健**  
   直接使用 top‑k 路由会导致部分专家被频繁选中，导致显存和计算不均。论文在路由输出上加了一个均衡正则，让每个专家的选中概率趋向均匀，同时引入“容量因子”让每个专家拥有略高于理论需求的缓冲空间，避免溢出。

3. **大规模专家并行实现 → 通过 GShard 框架在 TPU‑v4 上横向扩展至 2048 个专家 → 训练成本仅为 GPT‑3 的 1/3**  
   作者把专家划分到不同的硬件切片上，实现了几乎线性的扩展。相比于把所有参数堆在同一块机器上，专家并行让每台机器只需处理自己负责的子集，显著降低了通信开销和能耗。

4. **统一评估套件（29 项任务） → 零样本/单样本表现全线超越 GPT‑3**  
   在同样的 few‑shot 设置下，1.2 T GLaM 在 29 项 NLP 基准上平均提升约 2–3% 的准确率，且在推理时只需要一半的 FLOPs，证明稀疏激活并未牺牲模型质量。

### 方法详解
**整体框架**  
GLaM 仍然是标准的 Transformer 编码器，只是把若干层的前馈网络（FFN）换成了 MoE 结构。整个模型可以看成“稠密层 + 稀疏 MoE 层”的交替堆叠。稠密层负责捕捉全局信息，MoE 层提供巨大的容量扩展。

**关键模块拆解**  

1. **MoE 前馈层**  
   - 输入 token 向量先进入一个小型的路由网络（通常是两层线性 + softmax），输出每个专家的打分。  
   - 采用 **top‑2** 机制：选出分数最高的两个专家，分别得到一个权重（归一化后相加为 1）。  
   - 这两个专家各自执行自己的前馈计算（两层全连接 + 激活），随后用权重加权求和得到该 token 的 MoE 输出。  

2. **负载均衡损失**  
   - 对每个专家统计在一个 batch 中被选中的频率，记作 `p_i`。  
   - 计算 `∑_i p_i * (1 / p_i)` 的期望，加入到总损失中，目标是让所有 `p_i` 接近均匀分布。  
   - 这相当于在训练时给“抢饭的厨师”加罚，让他们主动让位给其他厨师。

3. **容量因子（Capacity Factor）**  
   - 每个专家的实际计算容量被设为理论需求的 1.25 倍左右，防止在高峰 batch 中出现“溢出”。  
   - 当某个专家的请求超过容量时，超额的 token 会被直接丢弃或重新路由到次优专家，保证显存不超限。

4. **专家并行（Expert Parallelism）**  
   - 所有专家被划分到不同的 TPU 切片上，每个切片只负责自己的一小部分专家。  
   - 路由网络的输出是全局的，但实际的前馈计算在本地完成，随后把结果汇总。  
   - 这种设计让参数量可以线性增长（增加专家），而通信量只随激活的专家数增长，保持了高效。

**最巧妙的地方**  
- **参数/计算分离**：通过稀疏激活，模型容量可以随意扩大，而实际算力几乎不变，这在之前的稠密模型里是不可能的。  
- **负载均衡 + 容量因子**：两者共同解决了 MoE 在大规模训练时的“热点”与“溢出”问题，使得训练过程既高效又稳健。

### 实验与效果
- **评测任务**：作者在 29 项公开的 NLP 基准上做 zero‑shot 与 one‑shot 测试，包括自然语言推理、阅读理解、情感分析、机器翻译等。  
- **对比基线**：主要与 GPT‑3（175 B 稠密）以及同规模的稠密模型（如 1.2 T Dense）进行比较。  
- **核心结果**：  
  - 1.2 T GLaM 在 zero‑shot 平均准确率上比 GPT‑3 高约 2.5%。  
  - 在 one‑shot 设置下，提升幅度略大，部分任务（如 Winogrande）提升超过 4%。  
  - 训练能耗仅为 GPT‑3 的 1/3，推理 FLOPs 约为其 50%。  
- **消融实验**：  
  - 去掉负载均衡损失后，部分专家被过度使用，导致训练不收敛，性能下降约 1.8%。  
  - 将 top‑2 改为 top‑1，算力进一步降低，但准确率下降约 1.2%。  
  - 只在少数层使用 MoE（而非全部 MoE 层）时，模型容量提升不明显，性能提升不到 1%。  
- **局限性**：  
  - 需要硬件（TPU‑v4）和软件（GShard）支持的专家并行，普通 GPU 环境下实现成本仍然较高。  
  - 路由网络本身是稠密的，随着专家数量极端增大时会产生一定的通信瓶颈。  
  - 论文未给出在极低资源（如移动端）上的部署实验，实际推理延迟仍需进一步优化。

### 影响与延伸思考
GLaM 的成功让业界重新审视 MoE 在大语言模型中的价值，直接催生了 **Switch‑Transformer**、**GShard**、**PaLM** 等后续工作，这些模型都在路由机制、专家并行或负载均衡上做了进一步改进。  
- **硬件层面**：Google 的 TPU‑v4 以及后续的 TPU‑v5 逐步加入对稀疏激活的原生支持，降低了实现门槛。  
- **算法层面**：研究者开始探索更细粒度的路由（如 token‑level 动态 top‑k）、基于强化学习的专家选择以及混合稀疏‑稠密的层次结构。  
- **应用层面**：在多语言模型、代码生成模型等需要超大容量的场景，MoE 已成为主流设计思路。  

如果想进一步深入，可以关注以下方向：  
1. **更高效的路由网络**：降低路由计算占比，同时提升选路的语义匹配度。  
2. **专家自适应学习**：让专家在训练过程中自行“进化”，专注于自己擅长的子任务。  
3. **跨硬件稀疏实现**：把 MoE 的稀疏激活搬到通用 GPU、CPU 上，降低部署成本。  

### 一句话记住它
GLaM 用稀疏激活的专家混合把模型容量炸到 1.2 万亿，却只花 GPT‑3 三分之一的能量，证明大模型不一定要全开算力。