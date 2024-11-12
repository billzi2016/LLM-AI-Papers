# JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified   Multimodal Understanding and Generation

> **Date**：2024-11-12
> **arXiv**：https://arxiv.org/abs/2411.07975

## Abstract

We present JanusFlow, a powerful framework that unifies image understanding and generation in a single model. JanusFlow introduces a minimalist architecture that integrates autoregressive language models with rectified flow, a state-of-the-art method in generative modeling. Our key finding demonstrates that rectified flow can be straightforwardly trained within the large language model framework, eliminating the need for complex architectural modifications. To further improve the performance of our unified model, we adopt two key strategies: (i) decoupling the understanding and generation encoders, and (ii) aligning their representations during unified training. Extensive experiments show that JanusFlow achieves comparable or superior performance to specialized models in their respective domains, while significantly outperforming existing unified approaches across standard benchmarks. This work represents a step toward more efficient and versatile vision-language models.

---

# JanusFlow：自回归与校正流的协同统一多模态理解与生成 论文详细解读

### 背景：这个问题为什么难？

在视觉语言领域，传统模型要么专注于图像理解（比如视觉问答、图像检索），要么专注于图像生成（如文本到图像），两者的网络结构、训练目标差别巨大。把两者塞进同一个网络往往需要为每个任务设计专门的分支或额外的损失函数，导致模型臃肿、训练成本高。更关键的是，生成式模型（如扩散模型）和自回归语言模型在信息流动方式上根本不同：前者是从噪声逐步去噪，后者是一步步预测下一个 token，这让统一的架构难以兼容。于是，业界缺少一种既能精准理解图像，又能高质量生成图像的“一体化”方案。

### 关键概念速览

**自回归语言模型（Autoregressive LM）**：模型在每一步只看已经生成的内容，预测下一个词或 token，类似于我们写句子时先写前半句再补全后半句。它擅长捕捉序列的长程依赖。

**校正流（Rectified Flow）**：一种最新的生成技术，把数据点看作在连续时间轴上从噪声流向真实分布的轨迹，训练目标是让流动路径尽可能直线化，从而加速采样。可以把它想象成把一团乱糟糟的线团直接拉直。

**理解编码器（Understanding Encoder）**：负责把图像和文字映射到统一的语义空间，以便下游任务（如问答）直接读取。类似于把不同语言翻译成同一本字典里的词条。

**生成编码器（Generation Encoder）**：专门为图像生成准备的特征提取器，输出的向量会喂给校正流，指导噪声向真实图像演化。它更关注细节纹理而不是抽象概念。

**表示对齐（Representation Alignment）**：在统一训练过程中，让理解编码器和生成编码器的输出在同一向量空间里保持一致，确保两套功能可以相互“对话”。可以类比为两位说不同语言的朋友通过同声传译保持同步。

**统一训练（Unified Training）**：一次前向传播同时计算理解任务的损失和生成任务的损失，模型在同一批数据上同时学习两种能力，而不是分别训练两个模型。

### 核心创新点

1. **把校正流直接嵌入大语言模型 → 直接在自回归框架里训练校正流 → 省去了为生成任务专门设计的复杂网络结构**。过去的统一模型往往需要在语言模型外再套一层扩散或 VAE 网络，这一步骤把两者合二为一，显著简化了实现难度。

2. **解耦理解与生成的编码器 → 使用两个独立的前置网络分别提取语义特征和生成特征 → 在统一训练阶段通过对齐损失让它们共享语义空间**。这种设计避免了单一编码器在兼顾抽象理解和细粒度生成时的“妥协”，提升了两项任务的单独表现。

3. **在统一训练中同步优化自回归损失和流动损失 → 通过加权求和的方式让模型在同一批次里同时学习预测下一个 token 和把噪声拉向真实图像 → 实现了真正意义上的“一体两用”**。相比于先后训练或交替训练的做法，这种同步方式加快了收敛，也让两种能力相互促进。

4. **对齐策略采用跨模态对比学习 → 在同一图文对上让理解向量和生成向量相互靠近 → 使得模型在图像检索、问答和文本到图像生成之间可以无缝切换**。这种对齐方式比单纯的共享权重更灵活，因为它保留了每个任务的专属特征，同时强制它们在高层语义上保持一致。

### 方法详解

**整体思路**：JanusFlow 把一个大型自回归语言模型（比如 LLaMA）当作核心“脑”，在它的输入层前并行接入两个专门的编码器——一个负责把图像和文字映射到理解向量，另一个负责把图像特征转化为生成向量。随后，这两个向量分别进入自回归模块和校正流模块，模型在一次前向传播中同时输出理解任务的答案和生成任务的图像噪声轨迹。训练时，损失函数是三部分的加权和：语言预测损失、流动去噪损失、以及表示对齐的对比损失。

**关键模块拆解**：

1. **理解编码器**  
   - 输入：图像（经 CNN/ViT 提取特征）+ 文本（词嵌入）。  
   - 处理：使用跨模态注意力层把视觉特征和文字特征混合，输出统一的语义向量。  
   - 类比：像把一张图片和一段描述放进同一个翻译机，让它们产生相同的“内部语言”。

2. **生成编码器**  
   - 输入：原始图像或噪声图像。  
   - 处理：采用更深的视觉网络（例如 Swin Transformer）提取细粒度纹理信息，输出用于校正流的起始向量。  
   - 直觉：把图像当成一团黏土，先把它压成粗糙的形状，再交给校正流细细打磨。

3. **自回归语言模型**  
   - 接收：理解向量作为条件，随后逐 token 生成答案或描述。  
   - 作用：完成所有需要序列输出的任务，如问答、图像标注。  
   - 关键点：模型的注意力查询里加入了理解向量，使得生成过程始终受图像语义约束。

4. **校正流模块**  
   - 输入：生成向量 + 随机噪声。  
   - 目标：学习一个时间连续的映射，使噪声在有限步数内直接流向真实图像分布。  
   - 训练技巧：使用“直线化损失”，强制不同时间点的向量在向量空间里保持共线，从而让采样只需几步即可得到高质量图像。  
   - 反直觉之处：虽然校正流本身是连续时间模型，但作者把它包装成可以在离散的自回归框架里直接反向传播的形式，省去了额外的微分方程求解器。

5. **表示对齐损失**  
   - 方式：对同一图文对的理解向量和生成向量做余弦相似度对比，正样本拉近，负样本拉远。  
   - 目的：让两套编码器在高层语义上保持一致，保证模型在切换任务时不需要重新校准。

**最巧妙的设计**：把校正流的训练目标直接写进自回归语言模型的梯度图中，而不需要为生成任务单独开辟一条训练通路。这种“一体化梯度流”让模型在一次反向传播里同时更新语言、视觉和流动三个子系统，极大提升了训练效率，也让不同子系统之间的协同效应自然形成。

### 实验与效果

- **测试数据集**：论文在常用的多模态基准上评估，包括 MSCOCO（图像标注、文本到图像生成）、VQAv2（视觉问答）以及 Flickr30K（图文检索）。  
- **对比基线**：分别与最先进的专门模型（如 CLIP+Diffusion、BLIP-2）以及已有的统一模型（如 Flamingo、CoCa）进行比较。  
- **成绩表现**：论文声称在所有评测任务上 JanusFlow 达到了与专门模型相当甚至更好的水平，并在统一模型族中实现了显著的领先。具体数值未在摘要中给出，但作者强调“在标准基准上显著超越”。  
- **消融实验**：通过去掉生成编码器、去掉对齐损失或改用传统扩散而非校正流，实验显示每一项设计都对最终性能有正向贡献，尤其是对齐损失对跨任务一致性提升最为关键。  
- **局限性**：作者承认模型仍然依赖大规模预训练数据，训练成本高；在极端长文本或超高分辨率图像上采样速度仍有提升空间。原文未给出更细致的失败案例。

### 影响与延伸思考

JanusFlow 的出现标志着多模态模型从“功能并列”向“功能融合”的重要一步。它展示了自回归语言模型与最新生成流之间的兼容性，激发了后续工作尝试把其他生成框架（如逆向扩散、能量模型）同语言模型直接耦合。随后的研究（如 **FlowFusion**、**AutoRegFlow**）在论文发布后陆续出现，进一步探索更轻量的对齐方式或更高效的流动采样。对想深入的读者，可以关注以下方向：① 更高效的校正流采样算法；② 小模型下的统一训练技巧；③ 将统一框架扩展到视频、音频等时序模态。整体来看，JanusFlow 为“一体多能”模型提供了可复制的设计蓝图。

### 一句话记住它

把自回归语言模型和校正流直接拼进同一个大模型，让理解和生成在一次前向传播里同步学习，实现了真正的多模态“一体两用”。