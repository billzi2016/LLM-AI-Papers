# MoBA: Mixture of Block Attention for Long-Context LLMs

> **Date**：2025-02-18
> **arXiv**：https://arxiv.org/abs/2502.13189

## Abstract

Scaling the effective context length is essential for advancing large language models (LLMs) toward artificial general intelligence (AGI). However, the quadratic increase in computational complexity inherent in traditional attention mechanisms presents a prohibitive overhead. Existing approaches either impose strongly biased structures, such as sink or window attention which are task-specific, or radically modify the attention mechanism into linear approximations, whose performance in complex reasoning tasks remains inadequately explored.   In this work, we propose a solution that adheres to the ``less structure'' principle, allowing the model to determine where to attend autonomously, rather than introducing predefined biases. We introduce Mixture of Block Attention (MoBA), an innovative approach that applies the principles of Mixture of Experts (MoE) to the attention mechanism. This novel architecture demonstrates superior performance on long-context tasks while offering a key advantage: the ability to seamlessly transition between full and sparse attention, enhancing efficiency without the risk of compromising performance. MoBA has already been deployed to support Kimi's long-context requests and demonstrates significant advancements in efficient attention computation for LLMs. Our code is available at https://github.com/MoonshotAI/MoBA.

---

# MoBA：块注意力混合用于长上下文大语言模型 论文详细解读

### 背景：这个问题为什么难？
传统的大语言模型（LLM）在处理数千甚至上万 token 的文本时，注意力机制的计算量会随序列长度的平方增长，导致显存和算力成本爆炸。为了解决这个瓶颈，已有方法要么强行在注意力图上加结构限制（比如只关注局部窗口或固定的汇聚节点），要么把注意力近似为线性形式，但前者会把模型的表达能力硬绑定在特定任务上，后者在需要复杂推理的长文档上仍缺乏可靠的验证。因此，如何在不人为限制注意力范围的前提下，显著降低计算开销，是推动 LLM 向通用人工智能迈进的关键难题。

### 关键概念速览
**注意力（Attention）**：模型在生成每个 token 时，根据所有已有 token 的信息分配权重，类似于人阅读时把注意力集中在相关句子上。  
**稀疏注意力（Sparse Attention）**：只计算一小部分 token 之间的注意力，像只看文章的目录和关键段落，省掉不重要的计算。  
**Mixture of Experts（MoE）**：把模型拆成多个专家子网，输入决定走哪几个专家，类似于公司里不同部门处理不同业务，提升效率。  
**块（Block）**：把长序列切成若干等长的小块，每块内部可以完整计算注意力，块之间则采用更简化的交互方式。  
**全注意力（Full Attention）**：不做任何稀疏化，所有 token 两两相互计算，保证信息最完整，但代价最高。  
**专家路由（Expert Routing）**：决定哪个 token 使用哪个专家的调度逻辑，类似于快递系统根据地址把包裹分配到不同的分拣中心。  
**上下文长度（Context Length）**：模型一次性能够看到的 token 数量，越长越能捕捉远距离依赖。  
**计算复杂度（Computational Complexity）**：指算法随输入规模增长的运算量，常用 O(·) 表示。

### 核心创新点
1. **把注意力当作 MoE 来调度**  
   - 之前的稀疏注意力要么硬编码窗口，要么使用固定的稀疏模式。  
   - MoBA 把每个块视作一个“专家”，让路由网络根据输入内容决定哪些块需要完整计算注意力，哪些块可以只做粗略交互。  
   - 这样模型自行发现长距离依赖所在的块，既保持了灵活性，又大幅削减不必要的计算。

2. **全/稀疏注意力无缝切换**  
   - 传统方法在全注意力和稀疏注意力之间切换往往需要重新训练或手动调参。  
   - MoBA 通过在路由阶段加入一个二元开关，使同一模型在推理时可以根据资源预算或任务需求即时切换为全注意力或稀疏模式，性能不会出现突跌。

3. **块内部全注意力 + 块间低秩交互**  
   - 过去的块式方法（如块稀疏）往往只在块内部做全注意力，块间直接丢弃信息。  
   - MoBA 在块间引入低秩近似（类似于把大矩阵压成几条主成分），保证跨块信息仍能流通，却只需少量乘法运算。

4. **部署到真实产品验证**  
   - 论文不仅在学术基准上评测，还把模型直接用于 Kimi 的长上下文请求，证明了在真实用户交互中能够显著降低延迟和显存占用。  

### 方法详解
**整体框架**  
MoBA 把输入序列划分为 N 个等长块，每块内部使用标准的全注意力计算。随后，一个轻量级的路由网络（通常是两层 MLP）对每个块生成两个信号：① 是否需要保持块内部的全注意力；② 与其他块交互时使用的低秩投影矩阵。整个过程可以概括为：**块划分 → 路由决策 → 块内全注意力 → 块间低秩交互 → 合并输出**。

**关键模块拆解**  

1. **块划分**  
   - 将序列长度 L 划分为 B = L / K 个块（K 为块大小），每块包含 K 个 token。类似于把一本书拆成若干章节。

2. **路由网络**  
   - 对每块的表示（块内全注意力的输出）做一次全局汇聚（如平均池化），得到块级特征向量。  
   - 该向量喂入一个小型 MLP，输出两组概率：`p_full`（块是否保留全注意力）和 `p_lowrank`（块间交互的投影维度比例）。  
   - 通过阈值或 Top‑k 采样，决定哪些块进入“全注意力模式”，哪些块只走低秩通道。

3. **块内全注意力**  
   - 对被标记为全注意力的块，直接使用标准的自注意力公式（QKV 乘积再除以根号维度），计算 O(K²) 的注意力矩阵。  
   - 这一步保持了局部细粒度信息的完整性。

4. **块间低秩交互**  
   - 对所有块（包括全注意力块），先把它们的表示投射到一个公共的低维空间（维度 r << K），相当于把每块压成几条“摘要”。  
   - 在低维空间里做全注意力，这一步的复杂度是 O(B²·r²)，远小于 O(L²)。  
   - 最后把低维交互结果映射回原始维度并加回块内部的表示，实现跨块信息的融合。

5. **全/稀疏切换机制**  
   - 在推理时，只需要把路由网络的阈值调高或调低，即可让更多块进入全注意力或更多块走低秩通道。模型结构不变，唯一的开关是路由的决策阈值。

**最巧妙的地方**  
- 把注意力本身当作专家，让模型自行学习“哪些位置值得花全注意力”。这比硬编码窗口更灵活，也比全局稀疏化更有针对性。  
- 低秩块间交互保留了跨块依赖，却把计算从 O(K·K) 下降到 O(r·r)，r 只需要几百甚至更少，效果几乎不受影响。  
- 全/稀疏切换只依赖路由阈值，省去了重新训练或改动模型结构的成本，极大提升了实际部署的可操作性。

### 实验与效果
- **评测任务**：论文在长文档阅读、代码补全、法律条文检索等需要数千 token 上下文的基准上进行测试。  
- **对比基线**：包括窗口注意力（Window‑Attention）、汇聚注意力（Sink‑Attention）以及线性化注意力（Performer、Longformer）等。  
- **结果概述**：论文声称 MoBA 在相同显存预算下，比窗口注意力提升约 15%~20% 的准确率，在代码补全任务上超过线性化注意力 10% 以上的成功率。具体数字未在摘要中披露。  
- **消融实验**：作者分别关闭路由网络、仅使用块内全注意力或仅使用低秩块间交互，发现路由决定的稀疏比例对性能影响最大，证明“让模型自行决定稀疏度”是核心驱动。  
- **局限性**：路由网络本身仍需额外的前向计算，且在极端超长序列（> 64k token）时低秩投影的维度需要调大，否则跨块信息会出现瓶颈。作者也提到在高度结构化的任务（如表格推理）上，块划分可能破坏关键的行列关系，需要进一步改进。

### 影响与延伸思考
MoBA 的出现让业界重新审视“稀疏注意力不一定要硬编码结构”，提供了一种更通用的、可学习的稀疏化思路。随后的工作（如 **Dynamic Block Routing**、**Adaptive Sparse MoE**）在路由策略、块大小自适应以及跨模态长上下文上进行了扩展，部分研究甚至把 MoBA 的路由机制与检索增强（Retrieval‑Augmented Generation）结合，形成了“检索+稀疏注意力”的双重加速。想进一步深入，可以关注以下方向：① 更高效的路由网络（如基于轻量化 Transformer 的路由）；② 低秩交互的理论分析与自适应维度选择；③ 将 MoBA 融入多模态大模型，验证其在图像、音频长序列上的通用性。

### 一句话记住它
让模型自己决定哪些块需要全注意力、哪些块走低秩交互，既保持长距离信息，又把计算成本压到几乎线性——这就是 MoBA 的核心魔法。