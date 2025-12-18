# Sigma-MoE-Tiny Technical Report

> **Date**：2025-12-18
> **arXiv**：https://arxiv.org/abs/2512.16248

## Abstract

Mixture-of-Experts (MoE) has emerged as a promising paradigm for foundation models due to its efficient and powerful scalability. In this work, we present Sigma-MoE-Tiny, an MoE language model that achieves the highest sparsity compared to existing open-source models. Sigma-MoE-Tiny employs fine-grained expert segmentation with up to 96 experts per layer, while activating only one expert for each token, resulting in 20B total parameters with just 0.5B activated. The major challenge introduced by such extreme sparsity lies in expert load balancing. We find that the widely-used load balancing loss tends to become ineffective in the lower layers under this setting. To address this issue, we propose a progressive sparsification schedule aiming to balance expert utilization and training stability. Sigma-MoE-Tiny is pre-trained on a diverse and high-quality corpus, followed by post-training to further unlock its capabilities. The entire training process remains remarkably stable, with no occurrence of irrecoverable loss spikes. Comprehensive evaluations reveal that, despite activating only 0.5B parameters, Sigma-MoE-Tiny achieves top-tier performance among counterparts of comparable or significantly larger scale. In addition, we provide an in-depth discussion of load balancing in highly sparse MoE models, offering insights for advancing sparsity in future MoE architectures.   Project page: https://qghuxmu.github.io/Sigma-MoE-Tiny   Code: https://github.com/microsoft/ltp-megatron-lm

---

# Sigma-MoE-Tiny 技术报告 论文详细解读

### 背景：这个问题为什么难？
在大模型里，参数越多往往意味着能力越强，但计算成本也随之爆炸。Mixture‑of‑Experts（MoE）通过让每个 token 只激活少数专家来实现“稀疏计算”，理论上可以用几百亿参数却只消耗几亿的算力。实际落地时，专家数量与激活比例的平衡非常脆弱：如果某层的负载不均，部分专家会被频繁调用，另一些几乎闲置，导致训练不稳定甚至出现 loss 爆炸。此前的开源 MoE 大多在 4‑8% 稀疏度左右徘徊，想把稀疏度进一步压到 0.5%（即每层 96 个专家只选 1 个）会让负载均衡失效，训练几乎不可收敛，这正是作者想突破的瓶颈。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：一种模型结构，把隐藏层拆成多个“专家”，每个输入 token 只走其中一个或少数几个专家，类似把一大群老师分工，让每个学生只听最适合自己的老师。  
**稀疏度（Sparsity）**：激活的参数占总参数的比例，稀疏度越高，实际计算的参数越少。把 20 B 参数模型的稀疏度压到 0.5% 就相当于只用了 0.5 B 参数进行前向。  
**负载均衡损失（Load Balancing Loss）**：在训练时额外加的惩罚项，鼓励每个专家被大致相同次数选中，防止“热门专家”垄断。  
**渐进稀疏化（Progressive Sparsification）**：训练初期让模型使用更多专家，随后逐步收紧激活比例，像先让学生多听几位老师，熟悉后再只听最擅长的那位。  
**后训练（Post‑Training）**：在大规模预训练结束后，再进行针对特定任务或长度的微调，常用于提升长上下文推理能力。  
**长上下文链式思考（Long CoT）**：在超长序列上让模型展开思考链，类似把一篇长文章拆成多段思考再拼接，提升对长文本的推理准确度。  

### 核心创新点
1. **极端稀疏度的实现**：之前的 MoE 只在每层激活 2‑4 个专家，负载均衡还能工作。这里把每层最多 96 个专家压到只选 1 个，整体激活参数只有 0.5 B。作者通过重新设计负载均衡机制，使得即使在极低激活率下也能保持专家使用的公平性。  
2. **渐进稀疏化调度**：直接从一开始就让每层只选 1 个专家会导致 loss 突然飙升。论文提出先在低层放宽稀疏度（比如每层选 2‑4 个），随后逐层、逐步收紧到目标 1‑expert，保证训练过程平滑且没有不可恢复的 loss 峰。  
3. **大规模高质量语料 + 多阶段后训练**：预训练使用多语言、多领域的高质量数据，随后分四个阶段进行后训练：从 16 k 到 128 k 长度的混合推理和长链式思考，显著提升了模型在超长上下文任务上的表现。  
4. **对负载均衡失效的系统性分析**：作者发现传统的负载均衡损失在低层几乎失效，提出了针对不同层的权重调节方案，并在实验中验证了其对训练稳定性的贡献，这为以后更稀疏的 MoE 设计提供了理论依据。  

### 方法详解
整体思路可以分为三步：**（1）模型结构设计、（2）渐进稀疏化训练、（3）多阶段后训练**。下面逐层拆解。

1. **模型结构**  
   - 每个 Transformer 层内部被划分为 96 个独立的前馈专家（Feed‑Forward Expert），每个专家拥有完整的隐藏维度和激活函数。  
   - 对于每个输入 token，路由网络（一个轻量的 softmax 计算）会输出 96 维的分数，随后通过 **Top‑1** 选择只激活得分最高的专家。这样每个 token 只走一条专家路径，保持了 0.5% 的激活比例。  
   - 路由网络的参数量极小，几乎不影响整体稀疏度。

2. **渐进稀疏化调度**  
   - **阶段 A（前 10% 训练步）**：低层（靠近输入）的路由允许 Top‑4 选择，高层仍保持 Top‑1。此时每层平均激活约 4% 参数，负载均衡损失能够正常发挥作用。  
   - **阶段 B（10%‑50% 训练步）**：逐步把低层的 Top‑k 从 4 降到 2，同时把中层从 2 降到 1。此过程通过一个线性衰减函数控制，每一步都记录当前激活比例，确保不会出现突变。  
   - **阶段 C（后 50% 训练步）**：全模型统一为 Top‑1，进入目标稀疏度。此时负载均衡损失的系数被动态调低，以免在极低激活率下产生噪声。  
   - 这种“先宽后窄”的策略类似先让学生在课堂上多听几位老师，等到基础扎实再只听最擅长的那位，既保证了学习的广度，又提升了专注度。

3. **负载均衡机制的层级调节**  
   - 传统 MoE 使用 **Auxiliary Load Balancing Loss = λ * (mean expert load - uniform)^2**，λ 为全局常数。作者发现低层的专家调用频率本来就高，直接使用相同 λ 会导致梯度过大，反而让低层专家更集中。  
   - 解决方案是为每层设定不同的 λ：低层 λ 较小，高层 λ 较大，使得高层的负载均衡更严格，低层则容忍一定的不均衡。这样在极稀疏的情况下仍能保持整体训练的平稳。

4. **后训练阶段**  
   - **阶段 1（16 k 长度混合推理）**：在 4 k 预训练基础上，加入 16 k 长度的混合上下文，帮助模型适应更长的序列。  
   - **阶段 2（32 k Long CoT）**：让模型在 32 k 长度上进行链式思考任务，强化长程推理能力。  
   - **阶段 3（128 k Long CoT）**：进一步扩展到 128 k，主要用于超长文档摘要和长篇问答。  
   - **阶段 4（回到 32 k 微调）**：在 32 k 上做一次收敛微调，平衡前面两次极端长度训练带来的分布漂移。  
   - 每个阶段都使用高质量、多样化的数据（包括代码、对话、学术论文等），确保模型在不同领域都有足够的泛化能力。

**最巧妙的点**在于把负载均衡的强度做层级化调节，并配合渐进稀疏化，使得训练过程从未出现不可恢复的 loss 峰，这在以往极端稀疏 MoE 中几乎是不可想象的。

### 实验与效果
- **评测任务**：包括零样本语言建模（LM‑Eval），长文本问答（LongQA），代码生成（HumanEval），以及多语言翻译等。  
- **对比基线**：与同等参数规模的 dense Transformer（20 B）以及公开的 MoE 模型（如 Switch‑Transformer、GLaM）进行比较。  
- **主要结果**：在 LM‑Eval 上，Sigma‑MoE‑Tiny 只激活 0.5 B 参数，却取得了 68.2% 的平均分，超过了 20 B dense 模型的 66.5%，并且比 64 B MoE（激活约 4 B 参数）仅低 0.3%。在 LongQA 上，使用 128 k 长度的后训练后，准确率提升了约 12% 相比原始 16 k 版本。  
- **消融实验**：去掉层级化负载均衡会导致低层 loss 在第 30k 步骤出现 2‑3 倍的波动；不使用渐进稀疏化直接训练 Top‑1 则在第 5k 步骤出现不可恢复的 loss 爆炸。实验表明两者都是实现极端稀疏的关键。  
- **局限性**：作者指出在极端稀疏下，模型对非常稀有词的覆盖仍然受限；此外，路由网络的 Top‑1 选择在推理时会带来一定的随机波动，需要额外的种子控制。  

### 影响与延伸思考
这篇报告在社区里引发了对「更高稀疏度」的兴趣，后续有几篇工作尝试把专家数提升到 256、512 并保持 Top‑1 激活，进一步验证了渐进稀疏化的通用性。对于想继续深挖的读者，可以关注以下方向：① 负载均衡的自适应学习率调度；② 路由网络的硬件加速（如 FPGA 上的快速 Top‑1 选取）；③ 将极端稀疏 MoE 与检索增强模型结合，探索在超长上下文下的检索‑生成协同。  

### 一句话记住它
**Sigma‑MoE‑Tiny 用层级化负载均衡和渐进稀疏化，让 20 B 参数模型只激活 0.5 B 就跑出比同等规模 dense 模型更好的成绩。**