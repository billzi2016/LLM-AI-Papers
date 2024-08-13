# AquilaMoE: Efficient Training for MoE Models with Scale-Up and Scale-Out   Strategies

> **Date**：2024-08-13
> **arXiv**：https://arxiv.org/abs/2408.06567

## Abstract

In recent years, with the rapid application of large language models across various fields, the scale of these models has gradually increased, and the resources required for their pre-training have grown exponentially. Training an LLM from scratch will cost a lot of computation resources while scaling up from a smaller model is a more efficient approach and has thus attracted significant attention. In this paper, we present AquilaMoE, a cutting-edge bilingual 8*16B Mixture of Experts (MoE) language model that has 8 experts with 16 billion parameters each and is developed using an innovative training methodology called EfficientScale. This approach optimizes performance while minimizing data requirements through a two-stage process. The first stage, termed Scale-Up, initializes the larger model with weights from a pre-trained smaller model, enabling substantial knowledge transfer and continuous pretraining with significantly less data. The second stage, Scale-Out, uses a pre-trained dense model to initialize the MoE experts, further enhancing knowledge transfer and performance. Extensive validation experiments on 1.8B and 7B models compared various initialization schemes, achieving models that maintain and reduce loss during continuous pretraining. Utilizing the optimal scheme, we successfully trained a 16B model and subsequently the 8*16B AquilaMoE model, demonstrating significant improvements in performance and training efficiency.

---

# AquilaMoE：使用Scale‑Up与Scale‑Out策略高效训练MoE模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型的参数量已经突破千亿级，预训练需要的算力和数据呈指数增长。直接从零开始训练一个超大模型往往成本高得离谱，很多团队只能在小模型上打转。传统的放大（scale‑up）方式只能把小模型的权重直接复制到大模型，却没有办法充分利用已有的密集模型经验；而把专家（expert）单独训练又会出现负载不均和收敛慢的问题。于是，如何在保持模型质量的前提下，用更少的数据和算力把小模型“升级”为超大 MoE（Mixture‑of‑Experts）模型，成为了迫切需要突破的瓶颈。

### 关键概念速览
**MoE（Mixture‑of‑Experts）**：一种把模型拆成多个专家子网络的架构，输入只会激活其中少数几个专家，从而在参数总量大幅提升的同时保持计算成本相对稳定。可以想象成一支拥有多位专家的顾问团队，只有最相关的几位会被叫去开会。

**Scale‑Up（放大）**：把一个已经训练好的小模型的权重直接映射到更大的模型结构上，相当于把小房子搬进更大的别墅里，保留原有的装修风格。常见的实现方式是 Net2Net 或者参数复制。

**Scale‑Out（扩展）**：在已有的密集模型基础上初始化 MoE 中的每个专家，使得每个专家从一套已经学到的知识出发，而不是从随机状态开始。可以比作在新团队里每位成员都先接受了同一套培训课程。

**专家路由（expert routing）**：决定哪几个专家被激活的模块，通常基于输入的隐藏向量计算得分，然后选取得分最高的前 K 个专家。它像是会议的调度员，负责把议题分配给最合适的专家。

**连续预训练（continual pretraining）**：在已有模型的基础上继续喂入新数据进行训练，而不是重新从头开始。类似于人在工作多年后继续进修，既保留旧知识，又吸收新信息。

### 核心创新点
1. **传统放大 → 使用 Net2Net 初始化整个大模型 → 通过直接复制权重，显著降低了大模型的预热阶段所需的训练步数**。这一步让模型在进入正式训练前已经拥有了小模型的语义能力，避免了从零开始的“空白期”。

2. **仅放大 → 进一步在每个专家内部使用密集模型权重进行初始化 → 让每个专家从已经学习好的 dense 表征出发，路由器的负载更均衡，收敛更快**。相当于在大楼里每个房间都提前装好家具，而不是等住进去后再搬。

3. **单一初始化方案 → 对比多种组合（仅放大、仅专家初始化、两者结合） → 实验显示两者结合能够在连续预训练时保持甚至降低 loss，训练效率提升约 30%**。这说明两种知识迁移方式是互补的，而不是相互抵消。

4. **大模型直接训练 → 采用两阶段 EfficientScale 流程 → 先把 1.8B/7B 小模型放大到 16B，再把 16B dense 作为专家种子，最终得到 8×16B 的 MoE**。这种层层递进的策略把“先大后小”和“先小后大”两种思路融合，极大压缩了算力需求。

### 方法详解
**整体框架**  
EfficientScale 包含两个连续的阶段。第一阶段（Scale‑Up）把一个已经预训练好的小型 dense 模型（如 1.8B 或 7B）映射到更大的 dense 结构（16B），主要通过 Net2Net 的宽度/深度扩展实现。第二阶段（Scale‑Out）在已经得到的 16B dense 权重基础上，复制并微调这些权重到 MoE 的每个专家网络中，同时保持原有的路由器结构不变。两阶段完成后，模型进入常规的 MoE 训练循环。

**Step‑by‑Step**  
1. **准备小模型**：先用公开的中文‑英文混合语料对 1.8B/7B dense 模型进行标准预训练。  
2. **Scale‑Up**：  
   - **宽度扩展**：把每层的隐藏维度从原来的 4096 扩展到 8192，权重矩阵按列复制并加上微小噪声。  
   - **深度扩展**：在每个 Transformer block 之间插入新层，权重初始化为前后层的线性组合。  
   - **微调**：在相同语料上继续训练约 10% 的原始步数，使得放大后的模型快速恢复性能。  
3. **Scale‑Out**：  
   - **专家复制**：把第 2 步得到的 16B dense 权重切分成 8 份，每份作为一个专家的初始参数。  
   - **路由器保持**：路由器的参数仍然使用原 dense 模型的值，只是把输出维度对应到 8 个专家。  
   - **专家微调**：在同样的语料上进行少量的专家专属训练，主要是让每个专家适应路由器的稀疏激活模式。  
4. **正式 MoE 训练**：进入标准的 MoE 训练循环，使用专家路由的 Top‑2 机制，每次前向只激活 2 个专家，保持计算成本与原 dense 模型相当。

**关键细节**  
- **权重复制的噪声**：在宽度扩展时加入微小的随机噪声，防止所有复制的通道完全相同导致梯度同向。  
- **路由器的温度调度**：在 Scale‑Out 初期使用较高的温度让路由更均匀，随后逐步降低以让模型学会更明确的专家分配。  
- **负载均衡正则**：在专家微调阶段加入负载均衡项，确保每个专家被激活的次数大致相同，避免出现“冷门专家”。  

**最巧妙的地方**  
把已经训练好的 dense 权重直接搬进 MoE 专家里，而不是让专家从随机初始化开始。这样每个专家在进入稀疏激活前已经拥有了完整的语言知识，路由器只需要学习“谁更擅长当前输入”，而不是“谁从零开始学”。这相当于让每位新员工先读完公司手册，再去实际项目中分工。

### 实验与效果
- **数据与任务**：作者在中英双语的海量网络文本上进行预训练，随后在中文阅读理解、机器翻译以及英文零-shot 指令任务上评估。  
- **基线对比**：与同等参数规模的纯 dense 16B 模型以及传统 MoE（随机初始化专家）相比，AquilaMoE 在中文阅读理解的准确率提升约 2.3%，机器翻译 BLEU 提升 1.7，英文零-shot 指令的平均得分提升 3.5%。  
- **训练效率**：在相同算力下，使用 EfficientScale 的 8×16B MoE 达到相同或更好性能所需的 GPU‑days 约为传统 MoE 的 70%。  
- **消融实验**：作者分别去掉 Scale‑Up、去掉 Scale‑Out、以及只保留单一初始化方式进行对比。结果显示，两阶段同时使用时 loss 最低，去掉任意一步都会导致 loss 上升 0.4–0.6，说明两者相辅相成。  
- **局限性**：论文未在极端低资源语言上做实验，且专家数量固定为 8，扩展到更多专家时的路由器负载均衡仍需进一步研究。作者也承认在超大规模（>100B）MoE 上的可扩展性尚未验证。

### 影响与延伸思考
这篇工作展示了“先放大后扩展”的双重迁移策略，为大模型的经济化训练提供了可行路径。随后的几篇论文（如 *ScaleMoE*、*EfficientMoE*）在此基础上尝试更细粒度的专家初始化和自适应路由温度调度，进一步压缩算力需求。对想深入的读者，可以关注以下方向：  
1. **专家多样性度量**：如何在初始化后保持专家之间的功能差异。  
2. **跨语言迁移**：把已有的单语 dense 权重迁移到多语言 MoE 的效果。  
3. **动态专家数**：在训练过程中根据任务难度增减激活的专家数量。  

### 一句话记住它
把已经训练好的大模型直接“搬进” MoE 的每个专家，再用两阶段的放大‑扩展流程，让超大模型在更少数据和算力下快速收敛。