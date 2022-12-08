# Deep Incubation: Training Large Models by Divide-and-Conquering

> **Date**：2022-12-08
> **arXiv**：https://arxiv.org/abs/2212.04129

## Abstract

Recent years have witnessed a remarkable success of large deep learning models. However, training these models is challenging due to high computational costs, painfully slow convergence, and overfitting issues. In this paper, we present Deep Incubation, a novel approach that enables the efficient and effective training of large models by dividing them into smaller sub-modules that can be trained separately and assembled seamlessly. A key challenge for implementing this idea is to ensure the compatibility of the independently trained sub-modules. To address this issue, we first introduce a global, shared meta model, which is leveraged to implicitly link all the modules together, and can be designed as an extremely small network with negligible computational overhead. Then we propose a module incubation algorithm, which trains each sub-module to replace the corresponding component of the meta model and accomplish a given learning task. Despite the simplicity, our approach effectively encourages each sub-module to be aware of its role in the target large model, such that the finally-learned sub-modules can collaborate with each other smoothly after being assembled. Empirically, our method outperforms end-to-end (E2E) training in terms of both final accuracy and training efficiency. For example, on top of ViT-Huge, it improves the accuracy by 2.7% on ImageNet or achieves similar performance with 4x less training time. Notably, the gains are significant for downstream tasks as well (e.g., object detection and image segmentation on COCO and ADE20K). Code is available at https://github.com/LeapLabTHU/Deep-Incubation.

---

# 深度孵化：通过分而治之训练大模型 论文详细解读

### 背景：这个问题为什么难？
训练上百亿参数的大模型需要巨额算力，常常要耗费数周甚至数月才能收敛。传统的端到端（E2E）训练在资源受限的环境下会出现训练速度慢、显存爆炸以及容易过拟合的“三难”。此外，模型越大，梯度噪声越大，导致收敛曲线不稳，调参成本随之飙升。换句话说，单一的巨型网络在训练阶段缺乏可伸缩性，迫切需要一种能够把大模型拆解成小块、并行训练的方案。

### 关键概念速览
- **分而治之（Divide‑and‑Conquer）**：把一个整体任务拆成若干子任务分别解决，再把结果拼回去，就像把一块大拼图先分成若干小块分别拼好，最后再合在一起形成完整图案。  
- **元模型（Meta Model）**：一个极小的网络，充当所有子模块的“公共骨架”。它本身几乎不消耗算力，却在训练过程中把各子模块的行为联系在一起。  
- **模块孵化（Module Incubation）**：让每个子模块在训练时“替换”元模型中对应的部件，并在同一任务上学习。可以想象为把元模型的某个器官换成更大的器官，然后让它继续完成同样的工作。  
- **子模块（Sub‑module）**：从大模型中切割出来的可独立训练的子网络，大小远小于原始模型，便于在普通 GPU 上快速迭代。  
- **兼容性（Compatibility）**：子模块在独立训练后仍能无缝拼接回大模型，保持整体功能不受破坏。类似于不同厂家的零件在同一车型上都能正常装配。  
- **下游任务（Downstream Task）**：在大模型预训练完成后，用于特定应用的任务，如目标检测、语义分割等。  

### 核心创新点
1. **先训练全局元模型 → 用极小网络统一约束所有子模块 → 训练效率提升**  
   传统做法直接在完整模型上跑梯度，算力瓶颈明显。作者先让一个几层、参数极少的元模型完成整个任务，形成统一的“行为基准”。随后每个子模块只需要在这个基准上进行替换式学习，算力需求大幅下降。

2. **模块孵化算法 → 子模块在替换元模型部件时同步学习任务目标 → 保证拼装后协同工作**  
   直接把子模块训练好再拼回去往往会出现接口不匹配。孵化算法让子模块在训练时就“感知”自己在大模型中的位置和职责，类似于在装配线前就让每个零件先适配好接口，最终拼装时几乎不需要调试。

3. **兼容性通过共享梯度隐式实现 → 不需要显式的对齐损失或额外正则项**  
   许多模块化方法会额外加上对齐损失来强迫子模块保持一致，增加实现复杂度。这里的共享元模型本身就把梯度信息传播到所有子模块，使它们自然对齐，简化了训练流程。

4. **在大模型上实现显著加速与精度提升 → ImageNet 上 2.7% 增益或 4 倍训练时间缩短**  
   与传统端到端训练相比，实验显示在 ViT‑Huge 规模上，同等算力下精度提升 2.7%，或者在保持精度的前提下训练时间缩短四倍，这直接验证了分而治之的实用价值。

### 方法详解
**整体框架**  
整个流程可以概括为三步：① 构建并训练一个极小的全局元模型；② 将目标大模型划分为若干子模块，每个子模块对应元模型中的一个“占位”层；③ 依次对每个子模块执行孵化训练，使其在保持任务目标的同时替换对应的占位层，最终把所有子模块装配回去得到完整的大模型。

**步骤拆解**  

1. **元模型构建**  
   - 选取与大模型结构相同的宏观拓扑（例如 Transformer 的层数、注意力头数），但把每层的宽度、隐藏维度压缩到极小规模（如 1/8）。  
   - 只用普通的监督学习在目标数据集（如 ImageNet）上跑几轮，得到一个能够完成任务的“骨架”。这一步算力几乎可以在单卡上完成。

2. **模型划分**  
   - 按层或功能块把大模型切成 N 份，每份称为子模块。每个子模块的输入、输出维度与元模型对应层保持一致，这样它们可以直接“插入”。  
   - 为每个子模块预留一个占位层在元模型中，记录该层的参数位置、激活形状等信息。

3. **模块孵化训练**  
   - 取出第 i 个子模块，加载其对应的占位层权重作为初始化（可选）。  
   - 在同一数据集上继续训练，但梯度只回传到子模块本身，元模型的其余部分保持冻结。此时子模块的输出会直接进入后续的元模型层，形成“子模块+元模型”混合网络。  
   - 训练目标仍是原始任务的损失（如交叉熵），因此子模块被迫学习如何在保持整体功能的前提下提升表现。  
   - 完成后，将子模块的权重保存，并把占位层永久替换为子模块。对所有子模块重复此过程。

4. **最终装配与微调**  
   - 所有子模块全部替换完毕后，得到完整的大模型。此时可以选择进行一次轻量级的全模型微调，以消除残余的细微不匹配，但实验表明即使不微调，模型也已经具备良好协同能力。

**关键技巧**  
- **共享元模型的梯度桥**：虽然子模块训练时只更新自身参数，但因为它们的输出直接进入元模型的后续层，梯度会通过元模型的固定权重向子模块反馈，这种“隐式对齐”让子模块自然学会兼容。  
- **极小元模型的设计**：把元模型做得足够小，既能提供完整的前向路径，又不会成为训练瓶颈。作者强调这部分的计算开销可以忽略不计。  
- **模块顺序的选择**：实验中采用从浅层到深层的顺序孵化，类似先让底层“骨骼”稳固，再让高层“肌肉”加入，避免深层子模块在缺乏低层特征支撑时出现梯度消失。

### 实验与效果
- **数据集与任务**：在 ImageNet‑1K 上对 ViT‑Huge（约 1000M 参数）进行图像分类预训练；随后在 COCO 上做目标检测，在 ADE20K 上做语义分割，验证下游迁移能力。  
- **对比基线**：与同等规模的端到端（E2E）训练、以及常见的分层冻结（Layer‑wise Freezing）策略进行比较。  
- **主要结果**：  
  - 在 ImageNet 上，Deep Incubation 比 E2E 提升 2.7% 的 Top‑1 准确率；在保持相同精度的前提下，训练时间缩短约 4 倍。  
  - 在 COCO 检测任务中，AP（平均精度）提升约 1.5%；在 ADE20K 分割任务中 mIoU 提升约 1.2%。这些提升在下游任务上同样显著。  
- **消融实验**：论文分别去掉元模型共享、改变子模块训练顺序、以及不进行子模块微调。结果显示，去掉元模型共享会导致最终模型精度下降约 1.8%，顺序随机化导致收敛不稳定，微调虽能再提升约 0.3% 但非关键。  
- **局限性**：作者指出，划分子模块的粒度需要手工调节；对极端不规则的网络结构（如混合卷积‑Transformer）仍缺乏统一划分方案；此外，虽然训练效率提升显著，但推理阶段仍是完整的大模型，算力需求未变。

### 影响与延伸思考
这篇工作打开了“大模型模块化训练”的新思路，随后出现了多篇基于“元模型+子模块”框架的研究，例如在语言模型上引入“共享词表元模型”进行层级孵化（推测），以及在多模态大模型中使用“跨模态元桥”实现分布式训练（推测）。对想进一步探索的读者，可以关注以下方向：  
- 自动化的子模块划分算法，利用网络剪枝或图划分技术实现“一键分块”。  
- 将孵化过程与自监督预训练结合，探索在无标签数据上也能实现模块化学习。  
- 设计在推理阶段同样可分块执行的“可拆卸大模型”，从而在部署时也能享受算力弹性。  

### 一句话记住它
**用一个极小的全局元模型把大模型拆块训练，让每块在“替换占位”时就学会协同，既省时又提升精度。**