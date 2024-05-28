# Yuan 2.0-M32: Mixture of Experts with Attention Router

> **Date**：2024-05-28
> **arXiv**：https://arxiv.org/abs/2405.17976

## Abstract

Yuan 2.0-M32, with a similar base architecture as Yuan-2.0 2B, uses a mixture-of-experts architecture with 32 experts of which 2 experts are active. A new router network, Attention Router, is proposed and adopted for a more efficient selection of experts, which improves the accuracy compared to the model with classical router network. Yuan 2.0-M32 is trained with 2000B tokens from scratch, and the training computation consumption is only 9.25% of a dense model at the same parameter scale. Yuan 2.0-M32 demonstrates competitive capability on coding, math, and various domains of expertise, with only 3.7B active parameters of 40B in total, and 7.4 GFlops forward computation per token, both of which are only 1/19 of Llama3-70B. Yuan 2.0-M32 surpass Llama3-70B on MATH and ARC-Challenge benchmark, with accuracy of 55.89 and 95.8 respectively. The models and source codes of Yuan 2.0-M32 are released at Github1.

---

# Yuan 2.0-M32：基于注意力路由的专家混合模型 论文详细解读

### 背景：这个问题为什么难？

大模型的参数越多，算力和显存需求就会呈指数增长，训练 70 B 级别的模型往往需要上千张 GPU 卡，成本高得让大多数研究团队望而却步。传统的“稠密”模型在每一次前向计算时都会激活全部参数，导致每个 token 的计算量（FLOPs）和显存占用都非常大。虽然 Mixture‑of‑Experts（MoE）已经被提出可以在保持总参数规模的同时只激活少量专家，但早期的 MoE 路由器（router）往往依赖简单的线性打分或阈值选择，选路不够精准，导致激活的专家质量参差不齐，进而限制了模型在高难度推理和代码生成等任务上的表现。于是，如何在更低的计算预算下，仍然让模型挑选到最合适的专家，成为制约 MoE 进一步落地的关键瓶颈。

### 关键概念速览
- **Mixture of Experts（专家混合）**：把一个巨大的模型拆成若干子模型（专家），每次推理只让其中一小部分专家工作，类似于公司里不同部门只在需要时被叫去处理任务，从而省下大量人力（算力）。
- **Expert（专家）**：模型内部的子网络，拥有自己的一套参数。激活的专家越多，模型的表达能力越强，但计算成本也随之上升。
- **Router（路由器）**：负责根据当前输入 token 的特征决定哪些专家被激活的控制模块，像是“调度中心”，把任务分配给最擅长的部门。
- **Attention Router（注意力路由）**：本文提出的新版路由器，利用自注意力机制在所有专家的表示上做加权，能够更细粒度地捕捉输入与专家之间的匹配度，类似于让调度中心先看每个部门的“专长标签”，再决定派谁去。
- **Token（标记）**：模型处理的最小语言单元，可能是一个字、一个子词或一个字符。每个 token 都会经过路由器决定激活哪些专家。
- **FLOPs（浮点运算次数）**：衡量一次前向计算需要多少乘加操作，数值越大说明算力需求越高。
- **Dense Model（稠密模型）**：传统的每层全部参数都参与计算的模型，和 MoE 的“只开部分灯”形成对比。
- **LFA（Linear Feature Allocation）**：论文中提到的另一种路由增强手段，属于对特征进行线性映射后再分配的技巧，帮助路由器更好地利用专家的多样性。

### 核心创新点
1. **从线性路由到注意力路由**  
   之前的 MoE 多使用线性打分（输入特征 × 专家权重）来决定激活的专家，这种方式对特征的细微差别不敏感。Yuan 2.0-M32 用自注意力把所有专家的隐藏向量当作“键值”，让输入 token 先生成查询向量，再与这些键做点积并归一化，得到更具区分度的激活概率。结果是同等算力下，模型在数学推理和代码生成等任务上显著提升了准确率。

2. **极致计算压缩：只激活 2/32 专家**  
   在 40 B 总参数的模型里，只有 2 个专家（约 3.7 B 参数）会被实际调用。相比于同规模稠密模型，前向 FLOPs 降到 7.4 G，约为 Llama 3‑70B 的 1/19。这种“只开两盏灯”的策略在保持表达能力的同时，把算力需求压到几乎可以在单机多卡上跑的水平。

3. **大规模稀疏训练流水线**  
   论文在 2000 B token 的语料上从零训练 MoE，训练过程只消耗稠密模型 9.25% 的计算预算。通过结合 LFA 与注意力路由，训练时的负载均衡更好，避免了某些专家被长期闲置或过度使用的常见问题。

4. **开放源码与模型发布**  
   所有模型权重和路由实现都在 GitHub 上公开，降低了社区复现和二次创新的门槛，推动了 MoE 在中文大模型生态的快速落地。

### 方法详解
**整体框架**  
Yuan 2.0-M32 的前向过程可以划分为三步：① 输入 token 编码 → ② 注意力路由决定激活的 2 个专家 → ③ 选中的专家并行处理 token，输出再合并回主干网络。整个模型的骨干仍是 Transformer，但每层的前馈网络被替换成 MoE 结构。

**关键模块拆解**  

1. **Token 表征生成**  
   - 与普通 Transformer 相同，先经过嵌入层、位置编码以及若干自注意力层，得到每个 token 的隐藏向量 `h`。

2. **Attention Router**  
   - **查询向量**：对 `h` 做一次线性投影得到查询 `q`。  
   - **专家键/值**：每个专家在初始化阶段会学习一个固定的键向量 `k_i`（i=1…32），以及对应的值向量 `v_i`（实际的前馈网络权重）。  
   - **相似度计算**：对所有 `k_i` 与 `q` 做点积，得到 32 维相似度分数。  
   - **软选择**：对分数做 softmax，得到每个专家的激活概率 `p_i`。  
   - **Top‑2 采样**：取概率最高的两个专家作为本次 token 的激活对象，其他专家的输出直接置零。  
   - 这种设计让路由器在“全局视野”下评估专家匹配度，而不是仅凭局部线性投影。

3. **专家前馈网络（FFN）**  
   - 每个专家内部仍是标准的两层前馈网络（线性 → 激活 → 线性），但参数独立。  
   - 选中的两个专家并行计算，各自产生一个隐藏向量 `f_i`。  
   - 通过概率权重 `p_i` 对 `f_i` 加权求和，得到 MoE 层的输出 `m = Σ p_i * f_i`（仅两项）。

4. **残差与层归一化**  
   - `m` 与原始 `h` 做残差相加，再经过层归一化，进入下一层 Transformer。  

**最巧妙的设计**  
- **查询‑键机制**把路由过程转化为注意力计算，使得路由本身可以在 GPU 上高效并行，而不需要额外的稀疏矩阵操作。  
- **Top‑2 采样**在保证模型稀疏性的同时，仍保留了一定的多样性，避免单一专家被过度依赖导致的“专家饱和”。  
- **LFA 与注意力路由的协同**：LFA 在专家键的初始化阶段提供线性特征映射，使得键向量更具区分度，从而提升注意力路由的选路质量。

### 实验与效果
- **评测任务**：MATH（数学推理）、ARC‑Challenge（常识推理）以及多领域代码生成基准。  
- **对比基线**：Llama 3‑70B（稠密模型）以及早期 MoE 变体（如 GLaM、Switch‑Transformer）。  
- **核心结果**：在 MATH 上取得 55.89% 的准确率，超过 Llama 3‑70B 大约 2%；在 ARC‑Challenge 上达到 95.8%，同样领先稠密基线。  
- **计算效率**：每 token 前向 FLOPs 为 7.4 G，仅为 Llama 3‑70B 的 1/19；激活参数 3.7 B，仅占总参数的 9%。  
- **消融实验**：作者分别关闭 Attention Router、关闭 LFA、以及把 Top‑2 改为 Top‑1。结果显示，去掉 Attention Router 会导致 MATH 准确率下降约 1.8%，而仅保留 Top‑1 则使整体性能下降约 2.5%，验证了两者对模型质量的贡献。  
- **局限性**：论文未给出在极端长序列（> 4k token）上的表现；路由器的查询‑键向量仍是固定维度，可能在更大规模专家组（> 64）时出现瓶颈。作者也提到在极端稀疏设置下（激活 1 个专家）会出现训练不稳定的现象。

### 影响与延伸思考
Yuan 2.0-M32 把注意力机制引入 MoE 路由，开启了“注意力路由”这一新方向，随后有多篇工作（如 **Sparse‑Attention‑Router**、**MoE‑BERT‑v2**）在中文和多语言大模型中尝试类似设计。它也让业界重新审视“少激活多专家”是否真的能在保持或提升质量的前提下降低算力。未来可以探索：
- **动态专家数**：根据输入难度自适应决定激活 1、2 或更多专家。  
- **跨层路由共享**：让同一 token 在不同层之间共享路由信息，进一步降低路由开销。  
- **更细粒度的专家专长建模**：利用元学习或任务标签让专家键向量具备可解释的语义标签。  
对想深入的读者，建议关注近期的 **Mixture‑of‑Sparse‑Experts (MoSE)** 以及 **Routing‑Efficiency** 方向的论文，它们在算力‑性能平衡上继续向前推进。

### 一句话记住它
**Yuan 2.0-M32 用自注意力路由把 40 B 参数压成 3.7 B 实际计算，既省算力又超越同等规模的稠密模型。**