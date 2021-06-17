# LoRA: Low-Rank Adaptation of Large Language Models

> **Date**：2021-06-17
> **arXiv**：https://arxiv.org/abs/2106.09685

## Abstract

An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA.

---

# LoRA：大语言模型的低秩适配 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理里，先用海量通用语料把模型预训练好，再针对具体任务微调已经成了标准流程。随着模型规模从几亿参数飙升到上千亿，完整微调——把所有参数都重新训练——的成本几乎不可承受。以 GPT‑3 175 B 为例，想要为每个下游任务部署一个完整微调的模型，硬件、存储和算力开销都让人望而却步。于是，业界急需一种既能保留大模型强大知识，又能用极少的可训练参数完成任务适配的方法。

### 关键概念速览
**Transformer**：当前主流的语言模型骨架，由自注意力层和前馈网络交替堆叠构成，类似于多层的“信息过滤器”。  
**全参数微调**：把模型里所有权重都当作可学习的变量，像把整辆车的发动机、轮胎、座椅全部重新调校。  
**Adapter**：在每层网络里插入小型子网络，只训练这些子网络的参数，保持主干不动，类似在原车上加装可拆卸的外挂。  
**低秩矩阵**：矩阵的秩（rank）表示其内部独立信息的数量，低秩矩阵只保留少数关键方向，就像用几根主干枝条概括整棵树的形状。  
**LoRA（Low‑Rank Adaptation）**：在每层的权重上叠加两个低秩矩阵（一个上升，一个下降），冻结原始权重，只训练这两个小矩阵，实现高效适配。  
**梯度累加**：把多批次的梯度加在一起再更新，能够在显存受限的情况下使用更大的等效批量。  
**推理延迟**：模型在实际使用时产生答案的时间，低延迟意味着用户几乎感受不到等待。

### 核心创新点
1. **冻结原始权重 + 注入低秩分解** → 传统做法要么全参数微调，要么在每层加大块 Adapter，导致显存或推理开销。LoRA 只在每层的线性投影上加上两个可训练的低秩矩阵（U、V），保持原始权重不动。这样训练参数从数十亿降到几千甚至几百，显存占用也随之下降约 3 倍。  
2. **等价的矩阵相加实现** → 在前向传播时，LoRA 把原始权重与低秩矩阵的乘积相加，等价于对原始权重做一次小幅度的“微调”。因为低秩矩阵的乘法可以提前合并到权重矩阵的乘法里，推理时不需要额外的算子，保持了原模型的速度。  
3. **对不同模型结构的统一适配** → 作者在 RoBERTa、DeBERTa、GPT‑2、GPT‑3 四类主流模型上都直接套用同一套 LoRA 插入方式，证明了方法的通用性。  
4. **经验性秩选择分析** → 通过实验观察到在多数任务上 1~8 的秩已经足够捕获任务特有的信号，进一步说明模型适配本身是高度低秩的现象，为后续理论研究提供了线索。

### 方法详解
整体思路可以概括为三步：  
1) **冻结** 预训练好的 Transformer 参数；  
2) **在每个线性层**（自注意力的 Q、K、V 投影以及前馈网络的两层）**插入** 两个低秩矩阵 U（维度 d×r）和 V（维度 r×d），其中 d 是原层的输入/输出维度，r 是人为设定的秩；  
3) **只对 U、V 进行梯度更新**，其余参数保持不变。

**插入细节**  
- 对于一个原始权重矩阵 W（大小 d×d），LoRA 用 ΔW = α·U·V 替代微调的增量，其中 α 是放大系数（常取 1），U、V 初始为随机小值。  
- 前向时，计算公式变为 (W + ΔW)·x = W·x + α·U·(V·x)。因为 V·x 的维度是 r，计算量远小于完整的 d 维乘法。  
- 反向时，只需要对 U、V 求梯度，显存只保存这两个小矩阵的梯度，省去对 W 的梯度存储。

**实现技巧**  
- 将 V·x 的结果缓存为 “低秩激活”，再与 U 做矩阵乘法，类似于把大矩阵乘法拆成两步小矩阵乘法。  
- 使用梯度累加可以在显存紧张的 GPU 上模拟更大批量，从而提升训练稳定性。  
- 在推理阶段，直接把 ΔW 合并到 W（一次性相加），得到一个完整的权重矩阵，模型结构不变，推理速度与原模型相同。

**最巧妙的点**  
把微调的“增量”限制在低秩空间，使得模型只在少数关键方向上调整，而不需要在高维空间里搜索；同时通过矩阵相加的等价变换，保证了推理时零额外开销，这在之前的 Adapter 方法里是做不到的。

### 实验与效果
- **任务与数据集**：在 GLUE（自然语言理解）、SQuAD（阅读理解）以及语言生成任务（OpenAI WebText）上分别对 RoBERTa、DeBERTa、GPT‑2、GPT‑3 进行微调。  
- **Baseline**：全参数微调、Adapter、Prefix‑Tuning 等。  
- **结果**：在 RoBERTa 上的 MNLI 任务，LoRA 与全参数微调的准确率相差不到 0.2%；在 GPT‑3 175 B 的 few‑shot 任务上，使用 LoRA 只训练约 0.01% 参数（约 1.5 M）即可达到与完整微调相当的性能。显存占用比 Adam 微调降低约 3 倍，训练吞吐提升 2–3 倍。  
- **消融实验**：作者分别关闭 U、V、α 或改变秩 r，发现 r=4~8 时性能基本收敛，进一步降低秩会导致显著下降；α 的放大作用对收敛速度有帮助。  
- **局限**：论文主要在英文大模型上验证，中文或多语言模型的秩需求尚未系统评估；极端低秩（r=1）在某些高复杂度任务上仍表现不佳。

### 影响与延伸思考
LoRA 发表后迅速成为大模型微调的“标配”，许多后续工作（如 IA³、AdaLoRA、DeltaTuning）都在其低秩思想上进行扩展，尝试自适应秩、稀疏化或结合 LoRA 与 Prompt‑Tuning。工业界也把 LoRA 融入模型服务平台，实现“一模型多任务”部署，显著降低算力成本。想进一步深入，可以关注 **秩自适应**（让模型在训练过程中自动决定每层的 r）以及 **跨模态 LoRA**（在视觉、语音模型上复制同样的低秩适配思路）的最新研究。

### 一句话记住它
只冻结大模型，给每层加上两个小的低秩矩阵，就能用几千个参数完成高质量微调，既省显存又不增推理延迟。