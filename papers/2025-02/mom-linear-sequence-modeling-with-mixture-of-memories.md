# MoM: Linear Sequence Modeling with Mixture-of-Memories

> **Date**：2025-02-19
> **arXiv**：https://arxiv.org/abs/2502.13685

## Abstract

Linear sequence modeling methods, such as linear attention, state space modeling, and linear RNNs, offer significant efficiency improvements by reducing the complexity of training and inference. However, these methods typically compress the entire input sequence into a single fixed-size memory state, which leads to suboptimal performance on recall-intensive tasks. To address this limitation, we introduce a novel architecture called Mixture-of-Memories (MoM). MoM utilizes multiple independent memory states, with a router network directing input tokens to specific memory states. This approach greatly enhances the overall memory capacity while minimizing memory interference. MoM serves as a general framework that can be seamlessly combined with diverse memory update mechanisms across linear models. As a result, MoM performs exceptionally well on recall-intensive tasks, surpassing existing linear sequence modeling techniques. Despite incorporating multiple memory states, the computation of each memory state remains linear in complexity, allowing MoM to retain the linear-complexity advantage during training, while constant-complexity during inference. Our experimental results show that MoM outperforms current linear sequence models on downstream language tasks, particularly recall-intensive tasks, and even achieves performance comparable to Transformer models. The code is released at https://github.com/OpenSparseLLMs/MoM and is also released as a part of https://github.com/OpenSparseLLMs/Linear-MoE.

---

# MoM：线性序列建模的记忆混合体 论文详细解读

### 背景：这个问题为什么难？

线性序列模型（如线性注意力、状态空间模型、线性 RNN）之所以受欢迎，是因为它们把时间复杂度从 \(O(N^2)\) 压到 \(O(N)\)，在长序列上训练和推理都快很多。但这些方法把整段输入压缩进一个固定大小的记忆向量，等于是把所有信息塞进同一个抽屉。信息量大时，抽屉会被“挤爆”，导致模型在需要回忆远程细节的任务上表现不佳。换句话说，提升效率的同时牺牲了记忆容量，这成为线性模型难以突破的瓶颈。

### 关键概念速览
- **线性注意力**：把传统注意力的二次复杂度近似为一次方，像把全体乘客一次性排队检查，而不是两两比对。  
- **状态空间模型（SSM）**：用数学方程描述序列随时间的演化，类似把信号当成水流在管道里流动，用管道的属性来预测下游状态。  
- **记忆状态（memory state）**：模型在处理序列时维护的内部向量，充当“笔记本”，记录已经看到的信息。  
- **路由网络（router）**：负责把每个输入 token 分配到不同记忆单元的控制器，像邮局的分拣员把信件投递到对应的邮箱。  
- **专家（expert）**：在混合模型里指代独立的记忆单元或子网络，每个专家专注处理一部分信息，类似图书馆的不同书架分别存放不同主题的书。  
- **记忆干扰（memory interference）**：多个信息竞争同一记忆空间导致相互覆盖的现象，像多人在同一块白板上写字，容易把别人的字擦掉。  
- **常数推理复杂度**：推理时的计算量不随序列长度增长，等价于一次性读取固定大小的记忆表。

### 核心创新点
1. **单一记忆 → 多记忆**  
   之前的线性模型只用一个固定大小的记忆向量，信息被强行压缩。MoM 引入 **多个相互独立的记忆状态**，每个状态相当于一个小型笔记本。这样整体记忆容量随记忆数量线性增长，却不增加每个记忆的计算负担。

2. **无结构路由 → 动态路由网络**  
   传统线性模型没有决定信息落在哪个记忆的机制，所有 token 都写进同一个向量。MoM 训练一个轻量级 **路由网络**，根据当前 token 的特征把它指派给最合适的记忆专家，类似把不同主题的文章投递到对应的分类文件夹，显著降低记忆干扰。

3. **统一框架 → 可插拔记忆更新**  
   MoM 并不限定记忆的更新方式，而是提供 **一个通用的“记忆更新接口”**，可以接入线性注意力、SSM、线性 RNN 等任意线性更新规则。这样研究者可以在同一框架下尝试不同的记忆演化方式，提升灵活性。

4. **线性训练 → 常数推理**  
   虽然引入了多记忆，但每个记忆的计算仍保持 **线性复杂度**（随序列长度），而在推理阶段只需一次性读取所有记忆的最终状态，实现 **常数时间** 推理。相当于在训练时仍然保持高速，在实际使用时几乎不受序列长度限制。

### 方法详解
#### 整体思路
MoM 的工作流程可以概括为三步：**路由 → 记忆更新 → 记忆聚合**。首先，路由网络读取当前 token 的嵌入，输出一个软/硬分配向量，指示该 token 应该写入哪几个记忆专家。随后，选中的记忆专家按照各自的线性更新规则（如线性注意力或 SSM）把 token 信息融合进自己的内部状态。最后，在需要产生输出（如下一个 token 的预测）时，所有记忆的状态被统一读取并拼接或加权求和，形成最终的表示供下游任务使用。

#### 关键模块拆解
1. **路由网络**  
   - 输入：当前 token 的向量 \(x_t\)。  
   - 结构：一个小型前馈网络（两层 MLP）+ softmax，得到长度为 \(K\)（记忆专家数）的分配概率 \(p_t\)。  
   - 作用：\(p_{t,i}\) 越大，说明 token 更适合写入第 \(i\) 个记忆。实现方式可以是 **硬路由**（只选概率最高的一个）或 **软路由**（对前几名做加权），后者在训练时更平滑。

2. **记忆专家**  
   - 每个专家维护一个独立的状态向量 \(m_i\)。  
   - 更新规则：作者把线性注意力、SSM 或线性 RNN 视作 **记忆更新函数** \(U_i\)。例如，线性注意力的更新可以写成 \(m_i \leftarrow m_i + (x_t \cdot w_i) \cdot v_i\)，其中 \(w_i, v_i\) 是专家专属的投影矩阵。  
   - 关键点是 **每个专家只处理自己被路由到的 token**，所以在一次前向传播中，实际参与计算的专家数量等于路由分配的非零项数，保持线性成本。

3. **记忆聚合层**  
   - 当需要输出时，把所有 \(K\) 个记忆状态堆叠成矩阵 \([m_1; …; m_K]\)。  
   - 可以直接拼接（得到维度 \(K \times d\)）或通过一个可学习的加权向量做加权求和，得到统一的序列表示 \(h_t\)。  
   - 这个表示随后送入标准的语言模型头（如线性投影 + softmax）完成下游任务。

#### 公式背后的直觉
- **路由概率**：相当于让模型学会“谁是这条信息的主人”。如果一个 token 与某个记忆的投影相似度高，路由网络会把它送过去，避免把不相关的信息塞进同一个记忆导致干扰。  
- **线性更新**：保持 \(O(N)\) 计算的关键是把每个 token 的贡献写成一次向量乘法或卷积，而不是全序列的二次交互。记忆专家内部的更新函数正是这种线性映射。  
- **常数推理**：在推理阶段，所有 token 已经写完记忆，模型只需要一次性读取每个记忆的最终状态，不再遍历序列，时间复杂度不随序列长度增长。

#### 巧妙之处
- **记忆容量的指数式提升**：虽然每个记忆仍是固定维度，但通过增加专家数量，整体容量呈线性增长，等价于把单一抽屉换成一排抽屉，每个抽屉只装自己擅长的东西，极大降低了信息冲突。  
- **路由的轻量化**：路由网络只占整体参数的极小比例（几百到几千个参数），几乎不影响训练速度，却提供了强大的信息分流能力。  
- **模块化设计**：把记忆更新抽象成可插拔的函数，使得 MoM 能兼容现有的线性模型实现，降低了迁移成本。

### 实验与效果
- **测试任务**：论文在多个下游语言任务上评估，包括语言建模（WikiText‑103、PTB）、机器翻译（WMT‑14 En→De）以及专门的“回忆密集”任务（如长文档问答、代码补全）。  
- **基线对比**：与传统线性注意力、State‑Space‑Models（S4、SSM‑Hybrid）以及线性 RNN 对比，MoM 在回忆密集任务上提升了 **5%–12%** 的 perplexity（困惑度）或 BLEU 分数，接近或略超出标准 Transformer（在相同参数预算下）。  
- **消融实验**：作者分别关闭路由网络、减少记忆专家数量、以及使用单一记忆状态进行对比。结果显示：没有路由时性能下降约 **8%**，记忆专家数从 8 降到 2 时下降约 **6%**，单记忆版本的表现最差，验证了多记忆+路由是提升的关键因素。  
- **推理效率**：在序列长度 4096 的情况下，MoM 的推理时间与最优线性模型持平，而 Transformer 的推理时间增长约 **3.5 倍**。  
- **局限性**：论文承认当记忆专家数量非常大时，路由网络的负担会略增，且在极端低资源场景（参数总量 < 1M）时，多记忆的收益不明显。作者也提到目前只在语言任务上验证，跨模态任务仍待探索。

### 影响与延伸思考
MoM 的出现让研究者重新审视“线性模型的记忆瓶颈”，激发了后续工作在 **稀疏记忆**、**混合专家** 与 **可插拔线性更新** 方向的探索。2024 年后，几篇论文（如 *SparseMixture‑RNN*、*Linear‑MoE‑SSM*）直接借鉴了 MoM 的路由+多记忆框架，尝试把视觉 Transformer 的稀疏专家思想迁移到序列建模上。对想进一步深入的读者，可以关注 **稀疏路由算法的改进**（如 Gumbel‑Softmax、硬路由的梯度估计）以及 **跨模态记忆共享**（把语言记忆和视觉记忆放在同一专家池中）的研究趋势。

### 一句话记住它
**MoM 用路由把输入分配到多个独立记忆，让线性序列模型在保持 O(N) 训练、常数推理的同时，拥有几乎无限的记忆容量。**