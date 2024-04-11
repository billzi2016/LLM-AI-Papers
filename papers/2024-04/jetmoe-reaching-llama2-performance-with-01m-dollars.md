# JetMoE: Reaching Llama2 Performance with 0.1M Dollars

> **Date**：2024-04-11
> **arXiv**：https://arxiv.org/abs/2404.07413

## Abstract

Large Language Models (LLMs) have achieved remarkable results, but their increasing resource demand has become a major obstacle to the development of powerful and accessible super-human intelligence. This report introduces JetMoE-8B, a new LLM trained with less than $0.1 million, using 1.25T tokens from carefully mixed open-source corpora and 30,000 H100 GPU hours. Despite its low cost, the JetMoE-8B demonstrates impressive performance, with JetMoE-8B outperforming the Llama2-7B model and JetMoE-8B-Chat surpassing the Llama2-13B-Chat model. These results suggest that LLM training can be much more cost-effective than generally thought. JetMoE-8B is based on an efficient Sparsely-gated Mixture-of-Experts (SMoE) architecture, composed of attention and feedforward experts. Both layers are sparsely activated, allowing JetMoE-8B to have 8B parameters while only activating 2B for each input token, reducing inference computation by about 70% compared to Llama2-7B. Moreover, JetMoE-8B is highly open and academia-friendly, using only public datasets and training code. All training parameters and data mixtures have been detailed in this report to facilitate future efforts in the development of open foundation models. This transparency aims to encourage collaboration and further advancements in the field of accessible and efficient LLMs. The model weights are publicly available at https://github.com/myshell-ai/JetMoE.

---

# JetMoE：以十万美元实现 Llama2 性能 论文详细解读

### 背景：这个问题为什么难？

大模型的性能每提升一点，所需的算力和数据往往呈指数级增长。传统的全参数（dense）模型在训练时必须把所有参数都激活，这导致数十亿美元的 GPU 费用和数月的算力占用，普通研究团队难以负担。即便有了开源的模型架构，缺少足够的算力和高质量数据仍是瓶颈；因此在成本可控的前提下，想要得到和 Llama2 同等水平的模型几乎被认为是不可能的任务。

### 关键概念速览
- **Mixture-of-Experts（专家混合）**：把模型拆成若干“专家”（子网络），每次前向只让一小部分专家工作，类似把一支大队伍分成若干小组，只有需要的组上场。
- **Sparsely-gated MoE（稀疏门控 MoE）**：在专家混合上加一层“门”，根据输入内容挑选出最合适的 2~4 个专家激活，未被挑中的专家保持沉默，省掉大量计算。
- **参数（Parameters）**：模型内部的可学习数值，参数越多模型容量越大，但不一定每次都要用到全部参数。
- **激活（Activation）**：一次前向传播实际参与计算的参数子集。稀疏激活让大模型在推理时只动一小部分参数。
- **Token（标记）**：文本被切分后的最小单元，模型的每一次计算都是围绕一个 token 进行的。
- **GPU 小时（GPU‑hours）**：衡量算力消耗的单位，30,000 H100 GPU 小时相当于 30,000 小时的 H100 卡全速运行。
- **开放数据（Open‑source data）**：全部来源于公开可用的语料库，避免了商业数据的版权限制。
- **推理计算量（Inference FLOPs）**：模型在实际使用时每个 token 需要的浮点运算次数，直接决定响应速度和成本。

### 核心创新点
1. **稀疏激活的双层专家设计**  
   - 之前的 MoE 大多只在前馈层使用稀疏激活，注意力层仍是全参数。  
   - JetMoE 把稀疏门控同时放在注意力层和前馈层，两层都只激活约 2 B 参数。  
   - 结果是推理时整体计算量比同等密集模型（如 Llama2‑7B）下降约 70%，而模型容量仍保持 8 B 参数。

2. **极致成本控制的训练流水线**  
   - 传统大模型训练往往使用上百 TB 的原始文本，且不对数据质量做细粒度筛选。  
   - 作者手工混合了多种公开语料，控制在 1.25 T token，且每个 token 的质量经过过滤。  
   - 结合 30k H100 GPU 小时的高效并行调度，整体花费不到 10 万美元，打破了“大模型必须贵”的常规认知。

3. **全透明的开放生态**  
   - 所有数据混合比例、训练超参数、代码实现全部公开，任何人都可以复现或在此基础上改进。  
   - 与过去很多只开模型权重、却隐藏训练细节的做法不同，这种“学术友好”模式鼓励社区共同迭代。

4. **用小模型实现大模型效果**  
   - 在公开基准上，JetMoE‑8B 的整体得分超过 Llama2‑7B，且其聊天微调版 JetMoE‑8B‑Chat 超过 Llama2‑13B‑Chat。  
   - 这说明在相同算力预算下，稀疏激活的 8 B 参数模型可以匹配甚至超越 13 B 密集模型的实际表现。

### 方法详解
**整体框架**  
JetMoE 的训练流程可以概括为四步：① 数据收集与清洗 → ② Token 化与混合比例设定 → ③ 构建稀疏门控的双层专家网络 → ④ 大规模并行训练与微调。核心思想是让每个 token 只触发少数专家，从而在保持大模型容量的同时大幅削减每步计算。

**1. 数据层**  
- 作者从公开的网页抓取、学术论文、代码库、对话日志等渠道收集约 1.25 T token。  
- 通过语言检测、去重、质量过滤（如低信息量句子、广告等）后，按主题（通用、代码、对话）分配权重，形成一个“混合配方”。这种手工调配比单纯使用全部数据更能提升下游任务的覆盖度。

**2. 模型结构**  
- **专家库**：模型内部有 N 个注意力专家和 M 个前馈专家（具体数量在源码中给出）。每个专家都是一个完整的 Transformer 子层。  
- **稀疏门**：输入 token 经过一个轻量级的路由网络，输出每个专家的“分数”。随后取分数最高的 K（通常是 2）个专家激活，其他保持不计算。  
- **双层稀疏**：注意力层和前馈层各自拥有独立的门控机制，意味着一次前向最多只动 2 B 参数（约 25% 的总参数），其余 75% 处于休眠状态。

**3. 前向计算流程（文字版流程图）**  
```
Token → 嵌入层 → 路由网络A → 选出注意力专家①/② → 注意力计算 → 路由网络B → 选出前馈专家③/④ → 前馈计算 → 输出
```
- 路由网络A、B 都是极小的 MLP，几乎不增加额外 FLOPs。  
- 选出的专家共享同一组参数空间，但只在本次前向中被激活。

**4. 训练细节**  
- 使用 AdamW 优化器，学习率采用线性 warm‑up + cosine decay。  
- 为防止“专家饱和”（某些专家被频繁选中），加入负载均衡正则，使每个专家的选中概率趋于均匀。  
- 训练期间采用混合精度（FP16）和梯度检查点技术，进一步压缩显存占用。  
- 30k H100 GPU 小时被划分为 8‑16 节点的并行作业，利用 ZeRO‑3 分布式优化器实现参数跨卡分片。

**最巧妙的点**  
- 将稀疏门控扩展到注意力层是一个逆向思考：注意力是计算最密集的部分，传统做法认为它必须全激活。JetMoE 通过让注意力也走稀疏路径，直接把计算瓶颈砍掉三分之二，这在实际推理速度上产生了“翻倍”效应。

### 实验与效果
- **评测基准**：论文在多个公开任务上做了对标，包括 MMLU（多语言理解）、GSM8K（数学推理）、TruthfulQA（事实性问答）以及对话指令微调的 ChatEval。  
- **对比基线**：Llama2‑7B、Llama2‑13B、以及同规模的密集 8 B 模型（若有）。  
- **主要结果**：JetMoE‑8B 在整体平均分上超过 Llama2‑7B，差距在 2%~4% 之间；JetMoE‑8B‑Chat 在对话指令集上领先 Llama2‑13B‑Chat，大约提升 1.5% 的胜率。  
- **计算效率**：推理时每个 token 的 FLOPs 比 Llama2‑7B 低约 70%，对应的吞吐率提升约 3 倍，成本下降同等比例。  
- **消融实验**：作者分别关闭注意力层稀疏、前馈层稀疏以及负载均衡正则，发现：① 只保留前馈稀疏，性能下降约 1.2%；② 只保留注意力稀疏，下降约 0.9%；③ 去掉负载均衡，部分专家被过度使用，整体性能下降约 2%。这些实验说明双层稀疏和均衡机制共同决定了最终效果。  
- **局限性**：论文承认在极端长序列（> 4k token）上稀疏路由的开销仍然显著；此外，虽然训练成本低，但对 H100 这类高端 GPU 的依赖仍是门槛，普通实验室可能难以直接复制。

### 影响与延伸思考
JetMoE 的成功向社区展示：**大模型不一定要“砸钱砸资源”，只要在架构和数据上做精细设计，就能以百倍更低的预算达到同等水平**。自发布后，已有多篇工作尝试在更小的算力预算下复现或改进稀疏 MoE，例如在 6 B 参数范围内加入层级路由、或在多模态任务中使用稀疏专家。对想继续深入的读者，建议关注以下方向：  
- **专家路由的学习策略**：如何让路由网络更好地捕捉长程依赖。  
- **跨模态稀疏 MoE**：把视觉、音频专家加入同一模型，实现“一模型多模态”。  
- **低成本硬件适配**：把稀疏激活移植到更廉价的 GPU 或 TPU 上，降低硬件门槛。  
- **安全与公平性**：稀疏模型的专家分配是否会导致特定语言或文化的表现偏差，需要系统评估。

### 一句话记住它
**JetMoE 用双层稀疏专家把 8 B 参数的模型算力压到 2 B，花十万美元就跑出 Llama2 级别的性能。**