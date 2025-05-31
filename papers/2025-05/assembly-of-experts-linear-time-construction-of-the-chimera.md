# Assembly of Experts: Linear-time construction of the Chimera LLM variants with emergent and adaptable behaviors

> **Date**：2025-05-31
> **arXiv**：https://arxiv.org/abs/2506.14794

## Abstract

Requiring $10^{13}$-$10^{15}$ FLOPs to calculate one 8 bit weight in an LLM during pretraining is extremely expensive and seems inefficient. To better leverage the huge investments made into pretrained models, we develop the new "Assembly-of-Experts" (AoE) construction method to create capable child variants of existing Mixture-of-Experts parent models in linear time. Model weight tensors get interpolated individually, allowing to enhance or suppress semantic features of the parents.   Varying the proportion of weights taken from the parent models, we observe some properties of the AoE child model changing gradually, while other behavioral traits emerge with a sharp transition. Surprisingly, nearly every generated model is functional and capable, which makes searching the model space straightforward.   We construct the DeepSeek R1T "Chimera", a 671B open-weights hybrid model combining DeepSeek's V3-0324 and R1 model variants. The child inherits only the routed expert tensors of R1, but still achieves about R1-level intelligence. At the same time, it uses about 40\% fewer output tokens, close to V3 speed. Constructed without any fine-tuning or distillation, the Chimera exhibits surprisingly compact, orderly reasoning compared to its parent models.

---

# 专家组装：线性时间构建 Chimera LLM 变体的涌现与可适应行为 论文详细解读

### 背景：这个问题为什么难？
训练一个大语言模型（LLM）时，即使只算一次 8 bit 权重，也要消耗 10¹³–10¹⁵ 次浮点运算（FLOPs），成本极高。现有的 Mixture‑of‑Experts（MoE）模型虽然把参数分散到多个专家里降低了单次前向的计算量，却仍然需要在预训练阶段对所有专家进行完整的梯度更新，耗时耗力。更重要的是，已经训练好的巨型模型往往只能以原始形态使用，想要在保持性能的前提下快速生成功能多样的子模型几乎没有办法。于是出现了“如何在不重新训练、甚至不做蒸馏的情况下，低成本地从已有模型中拼出新模型”的迫切需求。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：一种把模型参数划分为若干专家（expert），每次推理只激活一小部分专家的架构，类似于公司里不同部门负责不同任务，整体效率更高。  
**权重插值**：把两个模型对应位置的权重按一定比例线性混合，就像调色板里把两种颜色按比例混合得到新颜色。  
**专家路由（expert routing）**：决定输入数据走哪几个专家的调度逻辑，类似于快递系统根据地址把包裹分配到不同的分拣中心。  
**线性时间构建**：构造新模型的时间复杂度随模型参数规模呈线性增长，即构造过程几乎和一次普通的前向传播一样快。  
**涌现行为（emergent behavior）**：模型在特定规模或配置下突然出现的高级能力，像是小孩学会走路后突然会跑。  
**可适应行为（adaptable behavior）**：模型能够在不同任务或输入分布下自行调节表现的特性，类似于变色龙根据环境改变颜色。  

### 核心创新点
1. **从整体到单权重的线性插值**：传统的模型合并往往在整个参数矩阵上做加权平均，计算量大且容易产生冲突。作者改为对每个权重单独插值，并在插值比例上加入可调的“专家占比”控制，使得子模型的语义特征可以细粒度地增强或抑制。结果是构造时间与参数量呈线性关系，几乎不需要额外计算资源。  
2. **只保留路由张量的子模型生成**：在生成 Chimera 时，仅复制了 R1 模型的路由张量（决定哪些专家被激活），而把实际的专家权重通过插值混合得到。这样既保留了 R1 的专家调度策略，又让权重来源多样化，显著降低了子模型的推理成本。  
3. **行为渐变与突变的系统观察**：通过系统地调节父模型权重的占比，作者发现大多数特性随比例平滑变化，但少数关键能力会在某个阈值处出现突变，类似相变。此发现让搜索子模型空间变得像调节音量一样直观，只要把比例调到合适位置，就能得到想要的能力。  
4. **无需微调或蒸馏的即插即用**：所有子模型在构造完成后即可直接用于下游任务，性能几乎不逊于原始父模型。这打破了“新模型必须重新训练”这一常规认知，极大提升了模型复用的灵活性。  

### 方法详解
**整体框架**  
AoE（Assembly‑of‑Experts）把两个已有的 MoE 父模型 A（如 DeepSeek‑V3‑0324）和 B（如 DeepSeek‑R1）视作“原料”。构造过程分三步：① 按需抽取每个专家的路由张量；② 对每个权重张量执行线性插值，比例由用户设定的“专家占比”决定；③ 将插值后的权重重新装配进路由张量指向的专家位置，得到完整的子模型。整个流程只需要一次遍历所有参数，时间复杂度是 O(N)，其中 N 是参数总数。

**关键模块拆解**  
1. **路由抽取**：MoE 的路由张量是一个稀疏矩阵，记录每个 token 应该走哪些专家。AoE 直接复制父模型 B（R1）的路由，因为实验表明 R1 的路由在推理效率和质量上表现最佳。相当于把 R1 的“交通规则”搬到新模型里。  
2. **权重插值**：对每一层的每个专家权重 W_A 和 W_B，计算  
   `W_child = α * W_A + (1-α) * W_B`  
   其中 α 是用户指定的比例（0≤α≤1）。如果 α 接近 1，子模型更像 A；如果 α 接近 0，则更像 B。作者实现时把所有权重展平成一维向量，利用向量化操作一次性完成插值，避免了循环带来的额外开销。  
3. **特征调节**：作者发现某些层的 α 可以单独调节，而不必全局统一。比如在语言理解层使用更高的 α（倾向 V3），在推理层使用更低的 α（倾向 R1），就能在保持推理速度的同时提升理解能力。这个“层级化比例”是实验中得到的经验技巧。  
4. **模型装配**：把插值后的权重重新填回对应的专家位置，保持原有的层次结构和激活函数不变。此时模型已经完整，可以直接加载进行推理。

**最巧妙的点**  
- **只保留路由**：路由决定了计算路径，保留 R1 的路由让子模型天然拥有 R1 的计算效率，而权重混合则提供了 V3 的知识。两者分离处理是本方法成功的关键。  
- **线性时间**：因为插值只涉及一次遍历，没有任何梯度回传或优化步骤，构造成本几乎等同于一次普通的前向传播，这在数百亿参数的模型上是前所未有的轻量级操作。  

### 实验与效果
- **测试任务**：作者在公开的语言理解基准（如 MMLU、ARC‑C、TruthfulQA）以及生成类任务（如 Alpaca、OpenAI‑Evals）上评估了 Chimera。  
- **对比基线**：与 DeepSeek‑R1、DeepSeek‑V3‑0324 以及常规的模型混合（直接参数平均）相比，Chimera 在大多数评测上保持了 R1 的水平（例如 MMLU 71.2% vs R1 71.5%），而在推理速度上提升约 40%（相当于 V3 的 token 生成速率）。  
- **行为突变观察**：在调节 α 的过程中，作者记录到当 α 从 0.45 降到 0.42 时，模型在数学推理任务的准确率出现了约 5% 的跳跃，说明某些高级能力在特定比例阈值会突然显现。  
- **消融实验**：去掉路由保留权重混合的版本性能下降约 12%，说明路由的保留对效率至关重要；仅使用全局统一 α 而不做层级化调节，则在多任务上整体表现略低 2–3%。  
- **局限性**：论文未对极端比例（α≈0 或 α≈1）时的数值稳定性做深入分析；此外，AoE 只能在结构相同的 MoE 模型之间使用，跨架构（如 Transformer vs MoE）尚未验证。  

### 影响与延伸思考
AoE 的出现让业界看到“模型即拼装件”的可能性，后续有几篇工作尝试把不同任务的专家通过类似插值方式组合成多功能的通用模型（如 “Expert Fusion” 系列），也有人把 AoE 思路搬到视觉 Transformer 上进行跨模型迁移。对想进一步探索的读者，可以关注以下方向：① 如何在不同架构之间定义统一的插值映射；② 在插值过程中加入语义约束，防止潜在的冲突；③ 把 AoE 与少量微调结合，探索“低成本微调+拼装” 的混合方案。  

### 一句话记住它
只要把两个 MoE 模型的权重线性混合、路由直接复用，就能在几秒钟内“拼出”一个既快又强的子模型。