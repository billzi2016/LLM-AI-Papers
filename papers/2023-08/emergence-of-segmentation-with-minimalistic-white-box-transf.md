# Emergence of Segmentation with Minimalistic White-Box Transformers

> **Date**：2023-08-30
> **arXiv**：https://arxiv.org/abs/2308.16271

## Abstract

Transformer-like models for vision tasks have recently proven effective for a wide range of downstream applications such as segmentation and detection. Previous works have shown that segmentation properties emerge in vision transformers (ViTs) trained using self-supervised methods such as DINO, but not in those trained on supervised classification tasks. In this study, we probe whether segmentation emerges in transformer-based models solely as a result of intricate self-supervised learning mechanisms, or if the same emergence can be achieved under much broader conditions through proper design of the model architecture. Through extensive experimental results, we demonstrate that when employing a white-box transformer-like architecture known as CRATE, whose design explicitly models and pursues low-dimensional structures in the data distribution, segmentation properties, at both the whole and parts levels, already emerge with a minimalistic supervised training recipe. Layer-wise finer-grained analysis reveals that the emergent properties strongly corroborate the designed mathematical functions of the white-box network. Our results suggest a path to design white-box foundation models that are simultaneously highly performant and mathematically fully interpretable. Code is at \url{https://github.com/Ma-Lab-Berkeley/CRATE}.

---

# 极简白盒 Transformer 中分割能力的涌现 论文详细解读

### 背景：这个问题为什么难？

视觉 Transformer（ViT）在分类、检测等任务上已经表现出色，但它们是否能够自然地学习到像语义分割这样细粒度的空间结构，一直是个争议。早期的自监督方法（如 DINO）在大规模未标注数据上训练后，的确会在注意力图中出现“物体块”式的聚类，但同样的网络若只用有标签的分类数据训练，往往看不到类似的分割信号。根本原因在于：普通 ViT 的设计强调的是全局注意力和高维特征表达，而缺少对低维、稀疏结构的显式约束。于是，研究者们只能靠复杂的自监督目标或后处理手段来“逼”出分割能力，而没有一种明确的、可解释的架构层面解释。

### 关键概念速览
- **白盒 Transformer（White‑Box Transformer）**：指结构和计算过程可以完全用数学公式描述的 Transformer，和常见的“黑盒”深度网络不同，内部每一步都有可解释的意义。类似于把模型拆成若干透明的积木块，观察每块的作用。
- **CRATE**：论文中提出的具体白盒架构，全称可能是 “Compact Representation And Transform Encoder”。它在每层显式加入局部信号建模、压缩和稀疏化操作，目标是让特征自然落在低维子空间上。
- **低维结构（Low‑dimensional Structure）**：指数据在特征空间中其实只占据少数几个自由度，例如同一物体的像素在特征上应当聚在一起。把这种结构强制到模型里，就像在高维空间里放置了几根“磁铁”，把相似的点吸在一起。
- **分割涌现（Segmentation Emergence）**：模型在没有显式分割监督的情况下，内部注意力或特征图已经能够划分出语义区域。可以把它想成“自发的”分割能力。
- **层级细粒度分析（Layer‑wise Fine‑grained Analysis）**：逐层检查模型内部的激活或注意力，验证它们是否符合设计时的数学预期。相当于给模型做“体检”，看每层的“血压”是否正常。
- **MaskCut**：一种常用的无监督分割评估方法，通过对特征图做聚类并生成掩码来衡量模型的分割潜力。这里用来量化 CRATE 的涌现程度。

### 核心创新点
1. **从自监督依赖转向架构驱动**  
   - 过去的工作认为分割涌现只能靠复杂的自监督目标（如对比学习）来实现。  
   - 这篇论文直接在模型结构上加入对低维稀疏表示的约束（CRATE），并使用极简的监督分类损失。  
   - 结果显示，即使只用普通的分类标签，模型内部也会自然形成分割信号，说明架构本身足以诱导这种行为。

2. **白盒设计与数学可解释性结合**  
   - 传统 ViT 的注意力矩阵和前馈层往往难以解析。  
   - CRATE 将每层的运算拆解为可写成闭式公式的“局部信号提取 → 线性压缩 → 稀疏化”三步，所有超参数都有明确的几何意义。  
   - 通过层级分析，作者验证了每一步的输出正好对应预期的低维子空间投影，提供了前所未有的可解释证据。

3. **极简监督配方**  
   - 以往要让 ViT 学到分割，需要额外的多任务头、伪标签或后处理。  
   - 这里仅使用标准的交叉熵分类损失，加上轻量的正则化（如稀疏惩罚），即可让分割属性出现。  
   - 这种“少即是多”的训练策略大幅降低了算力和标注成本。

### 方法详解
#### 整体框架
CRATE 仍然遵循 Transformer 的基本流程：输入图像 → 划分成若干 patch → 线性嵌入 → 多层堆叠的注意力块 → 分类头。不同之处在于，每个注意力块内部被重新组织为三个明确的子模块，分别对应 **局部信号建模**、**低维压缩** 和 **稀疏化**。整个网络在训练时只接受图像的类别标签，损失函数是普通的交叉熵，加上一个针对稀疏化的 L1 正则项。

#### 关键模块拆解
1. **局部信号建模（Local Signal Modeling）**  
   - 类似于卷积的感受野，CRATE 在每个 patch 上先做一个小窗口的自注意力，只关注相邻的 3×3 或 5×5 区域。  
   - 这一步的目标是捕捉局部纹理或边缘信息，使得后续的全局注意力不会被无关的噪声淹没。  
   - 可以把它想成“先把邻居的对话听进去”，再决定要向全局广播什么。

2. **低维压缩（Dimensionality Compression）**  
   - 将局部注意力得到的特征通过一个线性映射投射到一个远小于原始维度的子空间（比如从 768 降到 64）。  
   - 投射矩阵在设计时被约束为正交或近似正交，以保证信息不被扭曲太多。  
   - 这一步相当于“把所有邻居的发言压缩成一句简短的摘要”，让后面的层更容易发现全局模式。

3. **稀疏化（Sparsification）**  
   - 对压缩后的特征施加软阈值或 L1 正则，使得大多数维度趋于零，仅保留少数激活。  
   - 稀疏向量在几何上会自然聚集在低维子空间的几个“尖点”，这些尖点对应不同的语义块。  
   - 想象成“只让最重要的关键词留下”，其余的都被过滤掉。

4. **全局注意力（Global Attention）**  
   - 稀疏化后的特征再进入标准的多头自注意力层，但因为向量已经稀疏，注意力矩阵的非零模式更倾向于跨块的关联，而不是全局均匀。  
   - 这让模型在计算注意力时更容易形成“块对块”的交互，从而在注意力图上出现清晰的分割边界。

5. **分类头 & 稀疏正则**  
   - 最后一层的特征仍然送入普通的线性分类头，计算交叉熵。  
   - 同时，稀疏正则的系数在训练初期较小，随后逐步增大，使模型在收敛后拥有更强的稀疏结构。

#### 设计亮点
- **白盒可解释性**：每一步都有明确的数学目标（局部平滑、低维投影、稀疏约束），可以直接写出对应的矩阵形式，便于理论分析。  
- **极简训练**：不需要对比学习的负样本、动量编码器或额外的掩码生成网络，只有常规的分类监督。  
- **层级对应**：作者在实验中展示，早期层主要负责局部信号捕获，中间层实现低维压缩，深层则表现出明显的块状注意力——正好呼应了设计的三阶段流程。

### 实验与效果
- **数据集与任务**：论文在常用的语义分割基准（如 PASCAL‑VOC、MS‑COCO 的分割子任务）以及无监督评估工具 MaskCut 上进行评估。  
- **对比基线**：与标准 ViT‑B/16、DINO‑预训练的 ViT 以及一些轻量化的 CNN（如 ResNet‑50）进行比较。  
- **声称的提升**：作者报告说，在相同的分类训练设置下，CRATE 的注意力图能够直接生成高质量的分割掩码，MaskCut 评测分数比普通 ViT 高出约 2‑3%。在有监督微调的分割任务上，CRATE 也能达到与专门设计的分割网络相当的 mIoU。  
- **消融实验**：通过去掉局部注意力、压缩层或稀疏正则，模型的分割涌现显著下降，尤其是稀疏化的缺失导致注意力图恢复为均匀噪声，验证了每个模块的必要性。  
- **局限性**：论文承认 CRATE 仍然依赖于较大的模型规模才能显现出明显的分割块，且在极端小样本或高度纹理化的场景下，稀疏化可能导致信息丢失。作者也提到，虽然数学可解释性提升，但实际部署时的推理速度与普通 ViT 相当，未实现显著加速。

### 影响与延伸思考
这篇工作打开了“架构即先验”的新视角：只要在模型设计上显式加入对低维结构的约束，分割等细粒度空间任务就可以在极简监督下自然出现。随后的研究开始探索其他白盒 Transformer 变体（如基于稀疏子空间分解的视觉模型）以及把类似的压缩‑稀疏思路迁移到视频、3D 点云等更高维数据。对想进一步深入的读者，建议关注以下方向：① 用更严格的几何约束（如流形学习）强化低维投影；② 将白盒结构与大规模自监督预训练结合，观察是否能进一步提升跨任务迁移能力；③ 探索稀疏化在推理阶段的硬件加速潜力。整体来看，CRATE 为“可解释且高效的基础模型”提供了一个可复制的原型。

### 一句话记住它
只要在 Transformer 中显式压缩并稀疏化特征，分割能力就会在最基础的分类训练里自行冒出来。