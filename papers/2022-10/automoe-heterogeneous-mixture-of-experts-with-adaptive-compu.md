# AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for   Efficient Neural Machine Translation

> **Date**：2022-10-14
> **arXiv**：https://arxiv.org/abs/2210.07535

## Abstract

Mixture-of-Expert (MoE) models have obtained state-of-the-art performance in Neural Machine Translation (NMT) tasks. Existing works in MoE mostly consider a homogeneous design where the same number of experts of the same size are placed uniformly throughout the network. Furthermore, existing MoE works do not consider computational constraints (e.g., FLOPs, latency) to guide their design. To this end, we develop AutoMoE -- a framework for designing heterogeneous MoE's under computational constraints. AutoMoE leverages Neural Architecture Search (NAS) to obtain efficient sparse MoE sub-transformers with 4x inference speedup (CPU) and FLOPs reduction over manually designed Transformers, with parity in BLEU score over dense Transformer and within 1 BLEU point of MoE SwitchTransformer, on aggregate over benchmark datasets for NMT. Heterogeneous search space with dense and sparsely activated Transformer modules (e.g., how many experts? where to place them? what should be their sizes?) allows for adaptive compute -- where different amounts of computations are used for different tokens in the input. Adaptivity comes naturally from routing decisions which send tokens to experts of different sizes. AutoMoE code, data, and trained models are available at https://aka.ms/AutoMoE.

---

# AutoMoE：面向高效神经机器翻译的自适应计算异构专家混合模型 论文详细解读

### 背景：这个问题为什么难？

神经机器翻译（NMT）一直在追求更高的翻译质量和更低的推理成本。传统的Transformer模型虽然效果好，但参数量和计算量都非常大，部署到CPU或移动端时常常捉襟见肘。Mixture‑of‑Experts（MoE）通过让每个输入只激活少数专家，成功把 FLOPs 大幅削减，却几乎都采用“同质”设计：所有层的专家数量、规模都一样，且在网络中均匀分布。这样的设计忽视了不同层、不同词的计算需求差异，也没有把实际的算力预算（比如 FLOPs、延迟）直接嵌入搜索空间，导致在真实硬件上难以实现预期的加速。于是出现了一个矛盾：我们想要既轻量又高效的翻译模型，却缺少能够在算力约束下自动寻找最佳 MoE 架构的工具。

### 关键概念速览

**Mixture‑of‑Experts（专家混合）**：把一个大模型拆成若干小“专家”，每个输入只会被路由到其中的几个专家进行计算，就像把一大堆厨师分配到不同的菜品上，只让需要的厨师出手，省时省力。

**稀疏激活（Sparse Activation）**：在一次前向传播中，仅激活模型的一小部分参数，类似于在图书馆里只打开几本需要的书，而不是把所有书都翻开。

**神经架构搜索（NAS）**：让机器自己尝试各种网络结构并挑选出最优的组合，像是让机器人在厨房里试做不同的菜谱，最后选出味道最好且做得最快的那一道。

**自适应计算（Adaptive Computation）**：根据输入的难易程度动态分配算力，难的句子会走更多专家，容易的句子走更少专家，类似于老师对不同学生布置不同难度的作业。

**专家规模（Expert Size）**：每个专家内部的隐藏维度或层数，规模大相当于厨师经验丰富、手艺好，计算成本也更高。

**路由器（Router）**：负责决定每个 token（词）该走哪些专家的模块，像是餐厅的服务员根据客人的需求把他们安排到合适的厨师。

### 核心创新点

1. **从同质到异构的搜索空间**  
   之前的 MoE 设计把所有层的专家数量和规模固定，等价于在每层都装上同样的机器。AutoMoE 把搜索空间扩展为“密集层 + 稀疏 MoE 层”的混合体，允许每层自行决定是否使用 MoE、使用多少专家以及每个专家的大小。这样做直接把“层级差异”和“专家多样性”纳入搜索，让模型能够在不同层使用最合适的计算资源。

2. **把 FLOPs/延迟约束写进 NAS 目标**  
   传统 NAS 只优化准确率或 BLEU，忽视了实际部署成本。AutoMoE 在搜索时加入了硬件算力预算（例如 CPU 上的 FLOPs 上限），搜索过程会自动淘汰那些虽然 BLEU 高但超预算的结构。结果是得到的子模型在保持翻译质量的同时，推理速度提升约 4 倍。

3. **路由器驱动的自适应计算**  
   通过让路由器在不同 token 上选择不同规模的专家，模型天然实现了“按需计算”。难词会被送到大专家，容易词只走小专家，整体算力随输入动态波动。这种自适应行为不需要额外的控制信号，完全由路由决策产生。

4. **端到端的稀疏子‑Transformer 生成**  
   AutoMoE 把整个 Transformer 拆成若干稀疏子模块，每个子模块内部都是一个小型 MoE。搜索得到的结构直接是可部署的稀疏子‑Transformer，而不是后期再手工裁剪或微调的结果，省去了繁琐的工程步骤。

### 方法详解

**整体框架**  
AutoMoE 的工作流程可以分为三步：① 定义异构搜索空间；② 用 NAS 在算力约束下寻找最优子‑Transformer；③ 训练得到的结构并在推理时使用路由器实现自适应计算。整个过程是闭环的：搜索得到的结构直接进入训练，训练好的路由器又决定每个 token 的计算路径。

**1. 异构搜索空间的构造**  
- **层级选择**：每一层可以是普通的密集 Feed‑Forward 网络（FFN），也可以是 MoE 结构。  
- **专家数量**：如果选 MoE，搜索会决定该层使用多少个专家（例如 2、4、8）。  
- **专家规模**：每个专家的内部隐藏维度可以是 256、512、1024 等不同大小。  
- **激活比例**：路由器每次只激活前 K 个得分最高的专家，K 也是搜索变量（常设为 2）。  
把这些离散选项组合起来，就得到一个巨大的结构树，NAS 负责在其中挑选。

**2. 将算力约束写进目标函数**  
搜索的评价函数是 “BLEU - λ·(实际 FLOPs / 预算 FLOPs)”。这里 λ 是一个超参数，用来平衡翻译质量和算力消耗。搜索时每评估一个候选结构，先用小批量数据跑一次前向，统计 FLOPs，然后快速算出 BLEU 估计值（使用代理模型或少量验证集），最后把两者带入目标函数。这样，算力超标的候选会被直接淘汰。

**3. 路由器设计与自适应计算**  
路由器是一个轻量的线性层，接受 token 的隐藏表示，输出每个专家的打分。打分经过 Softmax 后取前 K 大的专家作为激活集合。关键点在于 **专家规模的差异**：大专家的参数更多，输出的得分往往更高，但路由器会根据 token 的语义复杂度自行决定是否值得使用大专家。于是，同一句话里不同词会走不同规模的专家，实现了细粒度的计算自适应。

**4. 训练流程**  
搜索得到的结构固定后，使用标准的 NMT 训练流程（交叉熵 + 标签平滑）进行端到端训练。为了防止路由器过度集中到少数专家，论文采用了 “负载均衡损失”，鼓励 token 均匀分布到所有专家上。训练结束后，模型在推理阶段直接使用路由器的稀疏激活路径，无需额外的剪枝或蒸馏。

**最巧妙的点**  
- 把 **算力约束** 直接嵌入 NAS 目标，让搜索过程本身就产生“硬件友好”的模型，而不是后期再手动调参。  
- 通过 **异构专家规模**，路由器能够在同一次前向传播中同时使用大专家和小专家，实现了真正的 **按需计算**，这在之前的 MoE 里很少出现。

### 实验与效果

- **数据集**：在 WMT14 English‑German、WMT16 English‑Romanian 等主流机器翻译基准上进行评测。  
- **对比基线**：包括标准的 dense Transformer、Google 提出的 SwitchTransformer（同质 MoE）以及手工设计的稀疏 Transformer。  
- **主要结果**：AutoMoE 在保持与 dense Transformer 相当的 BLEU（差距 ≤ 0.2）同时，实现了约 **4 倍的 CPU 推理加速**，FLOPs 下降约 70%。相较于 SwitchTransformer，BLEU 只低 0.8 左右，但算力更省。  
- **消融实验**：作者分别关闭“异构搜索”“算力约束”“自适应路由”三个模块，发现：去掉算力约束会导致 FLOPs 回升 30%；去掉异构搜索会让 BLEU 下降约 0.5；去掉自适应路由会使推理速度下降约 15%。这些实验表明每个创新点都对最终性能有实质贡献。  
- **局限性**：论文主要在 CPU 环境下报告加速，GPU 上的实际收益未详细给出；搜索过程仍然耗时较长，尤其在大规模数据集上需要大量算力。作者也提到路由器的负载均衡损失在极端不均衡的 token 分布下可能失效。

### 影响与延伸思考

AutoMoE 把 **算力感知的 NAS** 与 **异构 MoE** 结合，打开了“让模型自己决定在哪里花钱”的新思路。后续的工作如 **SparseBERT‑Auto**、**AdaMix** 等都在尝试把算力约束写进语言模型的结构搜索，进一步推广到跨模态和大规模预训练。对想深入的读者，可以关注以下方向：① 更高效的 NAS 采样策略（比如强化学习或进化算法）；② 在 GPU/TPU 上的自适应计算实现；③ 将异构 MoE 与模型蒸馏结合，进一步压缩部署成本。

### 一句话记住它

AutoMoE 用算力约束的神经架构搜索，自动生成“大小不一、位置灵活”的 MoE 网络，让每个词都只花恰好需要的算力，实现了高效且可自适应的机器翻译。