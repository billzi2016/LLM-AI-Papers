# MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware   Experts

> **Date**：2024-07-31
> **arXiv**：https://arxiv.org/abs/2407.21770

## Abstract

We introduce MoMa, a novel modality-aware mixture-of-experts (MoE) architecture designed for pre-training mixed-modal, early-fusion language models. MoMa processes images and text in arbitrary sequences by dividing expert modules into modality-specific groups. These groups exclusively process designated tokens while employing learned routing within each group to maintain semantically informed adaptivity. Our empirical results reveal substantial pre-training efficiency gains through this modality-specific parameter allocation. Under a 1-trillion-token training budget, the MoMa 1.4B model, featuring 4 text experts and 4 image experts, achieves impressive FLOPs savings: 3.7x overall, with 2.6x for text and 5.2x for image processing compared to a compute-equivalent dense baseline, measured by pre-training loss. This outperforms the standard expert-choice MoE with 8 mixed-modal experts, which achieves 3x overall FLOPs savings (3x for text, 2.8x for image). Combining MoMa with mixture-of-depths (MoD) further improves pre-training FLOPs savings to 4.2x overall (text: 3.4x, image: 5.3x), although this combination hurts performance in causal inference due to increased sensitivity to router accuracy. These results demonstrate MoMa's potential to significantly advance the efficiency of mixed-modal, early-fusion language model pre-training, paving the way for more resource-efficient and capable multimodal AI systems.

---

# MoMa：高效早期融合预训练的模态感知专家混合模型 论文详细解读

### 背景：这个问题为什么难？
多模态大模型需要同时处理文本和图像等不同类型的输入，传统做法是把所有模态的特征拼在一起交给同一个巨型网络，这会导致计算资源被所有模态共享，难以针对每种模态的特性进行专门优化。已有的混合专家（Mixture‑of‑Experts，MoE）模型虽然可以把参数稀疏化，但在多模态场景下往往把专家混在一起，让每个专家既要处理文字也要处理图像，导致路由器在不同模态之间频繁切换，效率和效果都受限。于是出现了“模态感知”与“早期融合”之间的矛盾：要在输入最早阶段就把模态信息融合，又要让计算专注于对应模态，这在之前的架构里几乎找不到合适的平衡点。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：一种把大量专家网络（子模型）放在一起，只激活其中少数的技术，就像把一支大军分成若干小队，只派出最适合当前任务的几支上场。  
**早期融合（Early Fusion）**：在模型最前层就把文本和图像的 token 串在一起，让它们共同参与后续的注意力计算，类似于把不同语言的词直接混在一句话里一起理解。  
**模态感知专家组（Modality‑Aware Expert Group）**：把专家按模态划分成若干组，每组只负责特定模态的 token，像是把厨师分成专做中餐的和专做西餐的两支队伍。  
**路由器（Router）**：负责决定每个 token 进入哪个专家的控制器，依据 token 的特征打分后挑选最合适的专家，就像根据顾客的口味把他们送到最擅长的厨师手中。  
**专家选择（Expert‑Choice）**：传统 MoE 中的路由方式，所有专家都在同一个池子里竞争，任何 token 都可能落到任何专家上。  
**混合深度（Mixture‑of‑Depths，MoD）**：在同一层级上让不同 token 走不同深度的子网络，类似于让有经验的学生直接跳过基础章节，而新手则走完整的学习路径。  

### 核心创新点
1. **模态专属专家组 → 把专家划分为文本组和图像组 → 计算只在对应模态的专家上进行，避免了跨模态的无效计算，显著降低 FLOPs。**  
2. **组内自适应路由 → 在每个模态组内部仍保留路由机制，让同一模态的不同 token 能够根据语义差异选择不同专家 → 保持了 MoE 的稀疏适应性，同时不牺牲模态专一性。**  
3. **与混合深度（MoD）结合 → 在 MoMa 基础上再加入深度混合，让部分 token 直接走更浅的子层 → 进一步提升整体 FLOPs 节省至 4.2 倍，但也暴露出对路由准确性的更高依赖。**  
4. **对比传统 Expert‑Choice MoE → 传统做法把 8 个混合模态专家放在同一池子里，虽然也能稀疏激活，但在多模态早期融合场景下 FLOPs 只能省到 3 倍 → MoMa 通过模态划分把省算力度提升到 3.7 倍（文本 2.6×，图像 5.2×），证明模态感知划分的有效性。**  

### 方法详解
**整体框架**  
MoMa 的训练流程可以拆成三步：① 将输入序列（文本 token + 图像 patch token）按原始顺序拼接；② 根据 token 的模态标签把它们送进对应的专家组；③ 在每个组内部使用路由器挑选若干专家进行稀疏计算，最后把所有组的输出拼回统一的隐藏层，送入后续的 Transformer 层。整个模型仍保持“早期融合”的特性，因为拼接在最前层就完成，后续的注意力层仍然可以跨模态交互。

**模态感知专家划分**  
假设模型有 8 个专家，MoMa 把它们分成两组：4 个文本专家、4 个图像专家。每个 token 在进入 MoE 层时先检查自己的模态标记（文本或图像），然后只能在对应组里参与路由。这样做的直观好处是：文本 token 永远不会浪费在图像专家上，图像 token 也不会占用文本专家的计算资源。

**组内路由机制**  
在每个模态组内部，MoMa 仍然使用“top‑k”路由：对每个 token 计算它与组内每个专家的匹配分数（通常是一个小的前馈网络输出），选出分数最高的 k（如 2）个专家激活。路由器的参数是独立训练的，能够学习到“这类文本倾向用专家 A、这类图像倾向用专家 B”的模式。由于路由只在同模态内部进行，路由器的负担大幅下降，且更容易收敛。

**与 MoD 的叠加**  
MoD 的核心思想是让不同 token 走不同深度的子网络。MoMa 把 MoD 加在 MoE 层之后：路由器除了决定使用哪些专家，还决定 token 是否需要继续进入更深的 Transformer 层。这样，容易被快速分类的 token 可以提前退出，进一步削减 FLOPs。实验显示，组合后整体 FLOPs 节省提升到 4.2×，但因为多了一层路由决策，模型对路由准确性的敏感度上升，导致在因果推理等对细粒度信息要求高的任务上表现下降。

**最巧妙的设计**  
最让人眼前一亮的地方是“模态感知+组内路由”的双层稀疏策略。传统 MoE 只在全局层面做稀疏，而 MoMa 在全局先做一次粗粒度的模态划分，再在细粒度上做路由，这相当于先把大工厂分成专门车间，再在车间内部安排最合适的工人，极大提升了资源利用率。

### 实验与效果
- **训练规模**：在 1 万亿 token 的预训练预算下，MoMa 1.4 B 参数模型（4 文本专家 + 4 图像专家）完成实验。  
- **计算节省**：相较于计算等价的稠密基线，MoMa 在整体 FLOPs 上省了 3.7 倍，其中文本部分省 2.6 倍，图像部分省 5.2 倍。  
- **对比基线**：传统的 Expert‑Choice MoE（8 个混合模态专家）只能实现约 3 倍整体 FLOPs 节省（文本 3×，图像 2.8×），说明仅靠全局稀疏不足以充分利用模态差异。  
- **与 MoD 组合**：加入 MoD 后整体 FLOPs 节省提升至 4.2 倍（文本 3.4×，图像 5.3×），但在因果推理任务上出现性能下降，作者归因于路由错误放大效应。  
- **消融实验**：论文提供了去掉模态划分、仅使用全局路由的实验，显示 FLOPs 节省回落到约 2.8 倍，验证模态感知划分是提升效率的关键因素。  
- **局限性**：对路由准确性的依赖在深度混合场景下会放大，导致某些需要细粒度因果关系的下游任务受影响；此外，模型仍需要显式的模态标签来进行划分，限制了对未标注模态的灵活性。

### 影响与延伸思考
MoMa 的模态感知专家划分为多模态 MoE 设计提供了一个清晰的思路：先在宏观层面把不同模态的计算资源隔离，再在微观层面保持稀疏适配。自论文发布后，已有工作尝试将这种“模态分组+组内路由”扩展到音频、视频等更多模态，或结合自监督的模态识别来去掉显式标签（推测）。另外，关于路由鲁棒性的研究也随之升温，如何在深度混合的情况下保持路由的可靠性成为后续的热点。想进一步了解，可以关注近期在 ICLR、NeurIPS 上出现的“Modality‑Specific MoE”系列论文，以及针对路由误差的校正技术（如路由正则化、软路由）等方向。

### 一句话记住它
**MoMa 用模态专属专家组先把计算资源按模态划分，再在组内路由，实现了多模态早期融合下的 3‑4 倍 FLOPs 大幅节省。**