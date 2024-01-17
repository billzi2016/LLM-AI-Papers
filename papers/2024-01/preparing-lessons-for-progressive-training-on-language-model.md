# Preparing Lessons for Progressive Training on Language Models

> **Date**：2024-01-17
> **arXiv**：https://arxiv.org/abs/2401.09192

## Abstract

The rapid progress of Transformers in artificial intelligence has come at the cost of increased resource consumption and greenhouse gas emissions due to growing model sizes. Prior work suggests using pretrained small models to improve training efficiency, but this approach may not be suitable for new model structures. On the other hand, training from scratch can be slow, and progressively stacking layers often fails to achieve significant acceleration. To address these challenges, we propose a novel method called Apollo, which prep\textbf{a}res lessons for ex\textbf{p}anding \textbf{o}perations by \textbf{l}earning high-\textbf{l}ayer functi\textbf{o}nality during training of low layers. Our approach involves low-value-prioritized sampling (LVPS) to train different depths and weight sharing to facilitate efficient expansion. We also introduce an interpolation method for stable model depth extension. Experiments demonstrate that Apollo achieves state-of-the-art acceleration ratios, even rivaling methods using pretrained models, making it a universal and efficient solution for training deep models while reducing time, financial, and environmental costs.

---

# 为语言模型渐进式训练准备课程 论文详细解读

### 背景：这个问题为什么难？
Transformer 规模一路飙升，训练一套大模型往往要耗费数十甚至上百 GPU 天，导致成本高、能耗大。已有的做法是先把一个小模型预训练好，再用它的参数去加速大模型的学习，但这种“先训小再训大”只能在结构相同的情况下使用，新出现的层次设计或稀疏结构往往不适用。直接从零开始训练虽然通用，却慢得让人抓狂；而把层层堆叠起来逐步加深（类似 Net2Net）常常只能得到微弱的加速，根本原因是低层的知识难以自然迁移到高层。于是，如何在不依赖外部预训练模型、又能显著缩短深层模型的训练时间，成为了迫切需要解决的难题。

### 关键概念速览
**Transformer**：一种基于自注意力机制的神经网络，擅长捕捉序列中远距离依赖，几乎是所有大语言模型的核心。  
**渐进式训练（Progressive Training）**：先训练模型的浅层结构，随后逐步扩展深度，让新加的层在已有层的“经验”上继续学习。可以想象为先学会走路再学会跑步。  
**低价值优先采样（Low‑Value‑Prioritized Sampling, LVPS）**：在训练不同深度时，对那些当前表现较差的样本给予更高的采样概率，类似老师把更多时间花在学生的薄弱环节上。  
**权重共享（Weight Sharing）**：不同深度的模型共用一部分参数，使得新加的层能够直接利用已经学到的表示，像是把旧教材直接搬进新课堂。  
**深度插值（Depth Interpolation）**：在模型从浅层扩展到深层的过程中，用一种平滑的方式混合旧层输出和新层输出，防止突兀的“跳层”导致训练不稳定。  
**加速比（Acceleration Ratio）**：训练同等性能模型所需时间的相对缩短程度，数值越大说明方法越省时。  

### 核心创新点
1. **从低层学习高层功能 → LVPS + 权重共享**  
   传统的层堆叠只是在浅层训练完后直接复制结构，缺乏针对性的数据分配。Apollo 引入低价值优先采样，让模型在训练浅层时就对难样本加大关注，同时通过权重共享把浅层的参数直接映射到即将加入的深层。这样，新层在加入时已经拥有了“预热”的表示，训练速度提升明显。

2. **平滑深度扩展 → 深度插值机制**  
   直接把新层接上往往会导致梯度冲击，使得训练过程出现震荡。Apollo 设计了一套插值策略：在每一次深度扩展时，模型输出是旧深度输出和新深度输出的加权平均，权重随训练进度线性递增。相当于在课堂上先让学生听旧老师的讲解，再慢慢过渡到新老师的授课，避免了突兀的切换。

3. **统一的进度控制框架 → 低价值优先采样的动态调度**  
   过去的渐进式训练往往需要手工设定何时加层、加多少层。Apollo 将 LVPS 与训练进度耦合：当低价值样本的累计损失下降到预设阈值时，系统自动触发下一层的加入。这样既省去了人工调参，也保证了每一次扩展都在模型已经相对成熟的状态下进行。

4. **与预训练模型竞争的加速效果**  
   通过上述三点组合，Apollo 在实验中实现了与使用外部预训练小模型相当甚至更好的加速比。换句话说，即使没有任何先验模型，也能在训练深层语言模型时获得类似“先训小再训大”的效率提升。

### 方法详解
**整体框架**  
Apollo 的训练流程可以划分为四个阶段：① 初始化浅层模型；② 低价值优先采样驱动的浅层训练；③ 达到预设学习进度后，执行深度插值并共享权重，加入新层；④ 重复 ②‑③ 直至目标深度。整个过程是一个闭环：采样策略、进度判断、层扩展相互反馈。

**关键模块拆解**  

1. **低价值优先采样（LVPS）**  
   - **输入**：当前批次的样本及其对应的损失值。  
   - **操作**：计算每个样本的“价值”，价值 = 损失的负数（损失越大价值越低）。  
   - **采样**：依据价值的倒数构造概率分布，价值低的样本被抽中的概率更高。  
   - **直观**：像老师在课堂上对答错的学生多提问，让模型在薄弱环节上快速提升。  

2. **权重共享机制**  
   - 当准备加入第 k 层时，Apollo 把第 k‑1 层的参数复制给第 k 层的对应子模块（如自注意力头、前馈网络），并在此基础上加入少量可学习的偏置。  
   - 共享的好处是新层不需要从零开始学习低层特征，直接继承已有的表示空间。  

3. **深度插值（Depth Interpolation）**  
   - **插值公式（白话）**：模型输出 = (1‑α)·旧深度输出 + α·新深度输出，α 随训练步数从 0 线性增长到 1。  
   - **实现细节**：在每一次 forward 时，先计算浅层网络的输出，再把新层的输出加权合并；在反向传播时，梯度同样按 α 分配，保证新层逐步承担更多学习责任。  
   - **作用**：平滑梯度流，防止新层加入时导致 loss 突然飙升。  

4. **进度触发器**  
   - 监控低价值样本的累计损失，当该累计值下降到设定阈值（例如 80% 的初始损失）时，系统自动执行“深度插值 + 权重共享”，加入下一层。  
   - 这样做的好处是每一次层扩展都在模型已经对大多数样本有了较好拟合的情况下进行，避免了盲目提前加层导致的训练浪费。  

**最巧妙的点**  
LVPS 把数据分配的主动权交给了模型本身，让训练过程自适应地聚焦难例；而深度插值则把层扩展过程“软化”，两者结合实现了几乎不需要人工调参的全自动渐进式训练，这在以往的 Net2Net 系列方法中是少见的。

### 实验与效果
- **实验设置**：论文在多个公开的语言建模基准上验证，包括 WikiText‑103、OpenWebText 以及一个中等规模的机器翻译任务。模型基线是标准的 Transformer，深度从 6 层到 24 层不等。  
- **对比对象**：包括（1）从头训练的普通 Transformer；（2）使用预训练小模型进行知识迁移的两阶段方法；（3）传统 Net2Net 逐层堆叠方案。  
- **主要结果**：论文声称 Apollo 在相同最终模型质量下，训练时间比普通从头训练快约 2.3 倍，且比两阶段预训练方案快约 1.5 倍，达到业界最高的加速比。  
- **消融实验**：作者分别关闭 LVPS、权重共享和深度插值，发现去掉任意一项都会导致加速比下降 15%~30%，其中 LVPS 的贡献最大。  
- **局限性**：论文未在超大规模（> 1B 参数）模型上做实验，且对极端稀疏或混合专家结构的适配性仍需进一步验证。作者也提到在极端低资源环境下，LVPS 的采样开销可能略高于随机采样。

### 影响与延伸思考
Apollo 的核心思路——让模型在浅层阶段就“预习”深层功能——在随后的一批工作中被广泛引用。比如 2024 年的 **Layerwise Curriculum Learning** 进一步将任务难度也纳入层级调度；2025 年的 **Dynamic Depth Transformers** 把深度插值与可变层数推理结合，实现了推理时的即时层裁剪。对想深入了解的读者，可以关注以下方向：① 采样策略在大规模分布式训练中的实现细节；② 权重共享在混合专家模型中的扩展方式；③ 将 Apollo 思路与自监督预训练相结合的可能性。整体来看，Apollo 为“省时省能”训练大模型提供了一个通用框架，后续的研究大多围绕如何把这套框架搬到更复杂的模型结构上展开。

### 一句话记住它
Apollo 让模型在浅层就学会深层的功能，通过低价值采样、权重共享和深度插值，实现了无需外部预训练的高速渐进式训练。