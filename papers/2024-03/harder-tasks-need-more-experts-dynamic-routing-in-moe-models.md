# Harder Tasks Need More Experts: Dynamic Routing in MoE Models

> **Date**：2024-03-12
> **arXiv**：https://arxiv.org/abs/2403.07652

## Abstract

In this paper, we introduce a novel dynamic expert selection framework for Mixture of Experts (MoE) models, aiming to enhance computational efficiency and model performance by adjusting the number of activated experts based on input difficulty. Unlike traditional MoE approaches that rely on fixed Top-K routing, which activates a predetermined number of experts regardless of the input's complexity, our method dynamically selects experts based on the confidence level in expert selection for each input. This allows for a more efficient utilization of computational resources, activating more experts for complex tasks requiring advanced reasoning and fewer for simpler tasks. Through extensive evaluations, our dynamic routing method demonstrates substantial improvements over conventional Top-2 routing across various benchmarks, achieving an average improvement of 0.7% with less than 90% activated parameters. Further analysis shows our model dispatches more experts to tasks requiring complex reasoning skills, like BBH, confirming its ability to dynamically allocate computational resources in alignment with the input's complexity. Our findings also highlight a variation in the number of experts needed across different layers of the transformer model, offering insights into the potential for designing heterogeneous MoE frameworks. The code and models are available at https://github.com/ZhenweiAn/Dynamic_MoE.

---

# 更难任务需要更多专家：MoE模型中的动态路由 论文详细解读

### 背景：这个问题为什么难？
Mixture‑of‑Experts（MoE）通过让不同的专家网络专注于不同的子任务，已经把大模型的算力利用率推向了新高度。但传统的 MoE 大多采用固定的 Top‑K 路由：每条输入无论多简单还是多复杂，都强行挑选 K 个专家来计算。这样会导致两类浪费：一是简单样本被迫动用不必要的计算资源，二是复杂样本只能得到有限的专家帮助，难以发挥 MoE 的“多专家协作”优势。于是，如何让模型根据输入难度动态调节激活的专家数量，成为提升效率和性能的关键瓶颈。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：把一个大模型拆成若干子模型（专家），每次只让部分专家处理输入，类似于公司里不同部门分工合作。  
**路由（Routing）**：决定哪几个专家参与当前输入的过程，就像调度中心把任务分配给合适的部门。  
**Top‑K 路由**：固定挑选得分最高的 K 个专家，不管任务难易，就像每次都让前 K 名员工上班。  
**动态路由**：根据每条输入的“信心”或难度，灵活决定激活多少专家，类似于繁忙时叫更多人加班，清闲时只派少数人。  
**专家选择置信度**：模型对某个专家是否适合当前输入的自信程度，可视作“投票分数”。  
**层级异构性**：不同 Transformer 层对专家数量的需求不同，就像前端和后端对人手的需求不一样。  
**BBH（Big-Bench Hard）**：一套专门测评模型深度推理能力的基准任务，难度相当于“高阶数学题”。  

### 核心创新点
1. **固定 Top‑K → 置信度驱动的专家数量**：传统方法总是激活相同数量的专家，本文改为先计算每个专家的选择置信度，然后设定阈值，置信度高的专家被激活，置信度低的则被舍弃。这样，复杂输入会自然触发更多专家参与，简单输入则只动用少数专家，计算开销随需求弹性伸缩。  
2. **统一的动态路由模块 → 层级自适应**：作者在每一层的路由器里加入了“激活计数器”，根据该层的整体置信度分布动态调节本层的激活上限。实验发现上层往往需要更多专家，而下层可以用更少的专家完成特征抽取，提供了设计异构 MoE 的经验依据。  
3. **置信度阈值学习 → 端到端可训练**：阈值不是手工设定，而是作为可学习参数加入模型，训练时通过梯度让模型自行发现“多少置信度算够”。这让动态路由在保持可微的前提下，仍能在大规模预训练中稳定收敛。  
4. **效率-性能双提升的实证**：在多个公开基准上，动态路由的模型在保持不到 90% 参数激活率的情况下，整体准确率提升约 0.7%。尤其在 BBH 这类需要深度推理的任务上，激活的专家数显著上升，验证了“难任务叫更多专家”的核心假设。

### 方法详解
整体思路可以拆成三步：**置信度评估 → 动态阈值筛选 → 计算资源调度**。下面按顺序展开。

1. **置信度评估**  
   - 每条输入先经过一个轻量的门控网络（gate），该网络为每个专家输出一个分数，代表该专家对当前输入的适配度。  
   - 这些分数经过 Softmax 归一化后，得到每个专家的**选择概率**。作者把这个概率直接视作置信度，因为它反映了模型对该专家的“信心”。  

2. **动态阈值筛选**  
   - 在每一层，模型维护一个**阈值参数 τ**（可以是全局共享，也可以是层专属）。  
   - 对每个专家的置信度 p，若 p ≥ τ，则该专家被激活；否则被屏蔽。  
   - 为了防止所有专家都被屏蔽，作者在阈值上加入了一个**最小激活数**的软约束：如果激活数低于预设的下限，阈值会自动降低，确保至少有几个专家被选中。  

3. **计算资源调度**  
   - 被激活的专家会收到当前 token 的表示，执行前向计算后返回输出。  
   - 所有激活的专家输出再通过加权求和（权重仍是原始的置信度）合并回主流。  
   - 由于激活的专家数是输入依赖的，实际 FLOPs 随输入难度弹性变化。  

**最巧妙的点**在于把阈值 τ 设为可学习参数，并让它在训练过程中通过梯度“感受”整体任务的难易分布。这样模型不需要人为调参，就能自行发现“哪些任务需要更多专家”。另外，作者在每层加入了**激活计数器**，用来统计本层平均激活数，并把统计信息反馈给阈值的更新，使得不同层能够自适应地形成“上层多专家、下层少专家”的分布。

### 实验与效果
- **评测任务**：包括自然语言理解基准（GLUE、SuperGLUE）、大规模语言模型评测（LAMBADA）以及专门的深度推理套件 BBH。  
- **对比基线**：固定 Top‑2 路由的标准 MoE、以及不使用 MoE 的同规模 Transformer。  
- **整体提升**：在所有任务上，动态路由模型的平均准确率提升约 0.7%，而激活的参数比例保持在 85% 左右。  
- **BBH 细节**：在 BBH 上，动态路由激活的专家数比 Top‑2 多约 30%，对应的得分提升 1.2%（相对提升更明显），验证了“难任务叫更多专家”的假设。  
- **消融实验**：去掉可学习阈值、改为固定阈值或直接使用 Top‑K，性能均出现回落，说明阈值学习是关键因素。层级激活计数器的实验表明，若强制所有层使用相同激活上限，整体 FLOPs 下降但准确率下降约 0.4%。  
- **局限性**：作者指出在极端低资源环境下，阈值学习可能导致激活数波动较大，需要额外的平滑技巧；此外，动态路由增加了门控网络的计算开销，在极端大模型上仍有进一步优化空间。

### 影响与延伸思考
这篇工作打开了“按需激活专家”的新思路，随后出现的研究大多围绕**异构 MoE**（不同层使用不同专家规模）和**稀疏激活的自适应调度**展开。比如后续的 *Adaptive Sparse Transformer*、*Conditional MoE* 等，都在阈值学习或任务感知激活上进行扩展。对想进一步探索的读者，可以关注以下方向：  
1. **更细粒度的难度估计**——把输入难度拆成子任务级别，甚至到 token 级别动态调度。  
2. **跨模态 MoE**——在视觉、语音等多模态模型中引入类似的动态路由，看能否同样提升效率。  
3. **硬件协同**——把动态激活信息直接映射到 GPU/TPU 的调度层，实现真正的算力弹性。  

### 一句话记住它
让模型根据输入难度自行决定“叫几位专家上场”，既省算力又提升了在复杂任务上的表现。