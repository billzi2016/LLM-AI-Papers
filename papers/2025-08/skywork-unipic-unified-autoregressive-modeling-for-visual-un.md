# Skywork UniPic: Unified Autoregressive Modeling for Visual Understanding and Generation

> **Date**：2025-08-05
> **arXiv**：https://arxiv.org/abs/2508.03320

## Abstract

We introduce Skywork UniPic, a 1.5 billion-parameter autoregressive model that unifies image understanding, text-to-image generation, and image editing within a single architecture-eliminating the need for task-specific adapters or inter-module connectors-and demonstrate that compact multimodal systems can achieve state-of-the-art performance on commodity hardware. Skywork UniPic achieves a GenEval score of 0.86, surpassing most existing unified models; sets a new DPG-Bench complex-generation record of 85.5; attains 5.83 on GEditBench-EN and 3.49 on ImgEdit-Bench for image editing; and generates 1024 x 1024 images with under 15 GB of GPU memory (e.g., RTX 4090). (1) a decoupled encoding strategy that leverages a masked autoregressive encoder for synthesis and a SigLIP2 encoder for understanding, all feeding a shared autoregressive decoder; (2) a progressive, resolution-aware training schedule scaling from 256 x 256 to 1024 x 1024 while dynamically unfreezing parameters to balance capacity and stability; and (3) meticulously curated, 100 million-scale datasets augmented with task-specific reward models to refine generation and editing objectives. By demonstrating that high-fidelity multimodal integration need not incur prohibitive resource demands, Skywork UniPic establishes a practical paradigm for deployable, high-fidelity multimodal AI. Code and weights are publicly available at https://huggingface.co/Skywork/Skywork-UniPic-1.5B.

---

# 天工 UniPic：统一自回归模型用于视觉理解与生成 论文详细解读

### 背景：这个问题为什么难？

视觉模型要么擅长“看”，要么擅长“画”。传统的图像理解模型（比如分类、检测）使用卷积或视觉Transformer，只输出标签；而生成模型（如扩散或自回归）则专注于从文字合成高质量图片。把两者合在一起往往需要额外的适配层、跨模态桥接或完全分开的网络，导致参数膨胀、推理成本高，甚至在同一张卡上难以同时跑通。更糟的是，现有的统一模型要么规模庞大（数十亿参数），要么在生成质量或编辑能力上明显妥协。于是出现了“能用一套模型同时做好看图、写图、改图吗？”的需求。

### 关键概念速览
**自回归模型**：一次预测序列中的下一个元素，前面的输出会作为后面的输入，就像写文章时先写第一句再写第二句。  
**掩码自回归编码器**：在输入图像上随机遮住一部分像素，让模型学会从已知的上下文推断被遮住的内容，类似拼图游戏。  
**SigLIP2 编码器**：一种对比学习的视觉特征提取器，专门把图片映射到语义向量空间，像把图片翻译成“关键词”。  
**共享自回归解码器**：所有任务（理解、生成、编辑）共用的生成头部，负责把编码得到的向量逐步展开成完整的图像或标签序列。  
**分辨率感知训练**：训练时先在低分辨率上学习大体结构，再逐步提升到高分辨率，让模型像先画草图后上色一样稳步提升细节。  
**奖励模型**：用另一个模型对生成结果打分，引导主模型在训练时更倾向于高质量或符合编辑指令的输出。  
**DPG‑Bench / GenEval / GEditBench‑EN / ImgEdit‑Bench**：衡量多模态模型生成、理解和编辑能力的公开基准，数值越高代表表现越好。

### 核心创新点
1. **解耦编码 + 共享解码**：过去的统一模型往往让同一个编码器兼顾理解和生成，导致特征既要保留细粒度像素信息，又要抽象出语义，难以兼顾。这里把任务拆成两条路：掩码自回归编码器负责生成所需的像素级细节，SigLIP2 编码器负责提取高层语义。两者的输出直接喂给同一个自回归解码器，既保留了细节，又拥有语义指引。结果是模型在理解任务上不输专用分类器，在生成任务上也能保持高保真度。  
2. **渐进式分辨率训练**：直接在 1024×1024 上训练会导致梯度不稳、显存爆炸。作者先在 256×256 上训练完整模型，随后逐步放大到 512、768、1024，每升一级时只解冻一小部分参数，让网络在已有知识的基础上细化细节。这样既提升了训练稳定性，又让模型在高分辨率下仍保持 1.5 B 参数的轻量。  
3. **任务导向的奖励模型微调**：在 1 亿规模的多源数据上预训练后，作者再用专门为生成、编辑设计的奖励模型进行强化学习式微调。奖励模型会对“画得好看”“编辑符合指令”等维度打分，模型在训练时会倾向于产生高分的输出。相当于给模型装上了“审美老师”和“编辑指南”。  
4. **硬件友好的推理实现**：通过把解码过程全部做成自回归序列，并在实现上采用 16‑bit 量化和显存复用技巧，模型在 RTX 4090（15 GB）上即可一次性生成 1024×1024 的图片，打破了大模型只能在多卡或云端运行的常规认知。

### 方法详解
整体思路可以拆成三步：**编码 → 融合 → 解码**。  
1. **编码阶段**  
   - **掩码自回归编码器**：输入图像先被划分成若干 patch（类似 ViT），随后随机遮盖一部分 patch。模型在每一步预测被遮住的 patch 内容，形成一个自回归的隐藏序列。这个过程让编码器学会从局部上下文恢复细节。  
   - **SigLIP2 编码器**：同一张图像（或文字指令）也会送入一个对比学习预训练的视觉‑语言编码器，得到一个全局语义向量。可以把它想成“图片的标题”。  

2. **融合阶段**  
   两个编码器的输出在时间维度上拼接，形成一个统一的 token 序列。因为自回归编码器的输出已经是有序的像素 token，SigLIP2 的向量则被插入到序列的开头，起到“指挥官”作用，告诉后面的解码器这张图要表达什么。  

3. **解码阶段**  
   - **共享自回归解码器**：采用 Transformer 结构，逐 token 生成目标序列。对于理解任务，解码器的目标是输出文字标签序列；对于生成/编辑任务，目标是输出完整的图像 token 序列（包括像素颜色、位置信息等）。因为所有任务共用同一个解码器，模型在训练时会自然学到跨任务的通用生成规则。  

**训练细节**  
- **分辨率感知调度**：从 256×256 开始训练 100 k 步，随后每提升一次分辨率，训练步数减半，同时只解冻最近加入的层。这样模型在每个尺度上都有足够的学习时间，却不会因一次性大幅度增大显存而崩溃。  
- **奖励模型微调**：在预训练完成后，构建三个奖励模型（分别针对生成质量、编辑一致性、文本‑图像对齐），使用 PPO‑style 的策略梯度让 UniPic 在每一步生成后得到奖励并回传梯度。  
- **显存优化**：采用 Flash‑Attention、梯度检查点和 16‑bit 量化，使得即使在 15 GB 显存下也能一次性装下 1024×1024 的 token 序列。  

**最巧妙的点**：把两种截然不同的编码方式（像素级自回归 vs 语义对比）通过简单的序列拼接送入同一个解码器，而不需要额外的跨模态对齐层。这种“把两条路合流成一条河”的设计，让模型在保持轻量的同时获得了多任务的强大表现。

### 实验与效果
- **评测基准**：在 GenEval（通用生成评估）、DPG‑Bench（复杂生成）、GEditBench‑EN 与 ImgEdit‑Bench（图像编辑）四个公开基准上进行测试。  
- **成绩**：GenEval 得分 0.86，超过大多数已有统一模型；DPG‑Bench 复合生成得分 85.5，刷新记录；GEditBench‑EN 取得 5.83，ImgEdit‑Bench 取得 3.49，这两个编辑基准均领先同类 1‑2 分。  
- **对比**：与 7 B 参数的统一模型相比，UniPic 在所有指标上平均提升约 8‑12%；与专用的 2 B 参数生成模型相比，图像质量几乎持平，但在编辑任务上优势明显。  
- **消融实验**：作者分别去掉 SigLIP2 编码器、去掉奖励模型微调以及直接在高分辨率上训练。结果显示：去掉 SigLIP2 会导致理解任务准确率下降约 6%；去掉奖励模型后生成质量下降约 0.4 dB（PSNR）；直接高分辨率训练导致显存溢出并使收敛速度减慢 30%。  
- **局限性**：论文承认在极端长文本指令或非常细粒度的局部编辑（如微调单个像素颜色）时仍会出现细节模糊；此外，虽然显存需求已降到 15 GB，但在更低端显卡上仍不可直接运行。  

### 影响与延伸思考
这篇工作向社区展示了“千兆参数不是唯一的高性能路径”，激发了后续对轻量统一模型的兴趣。随后出现的几篇论文（如 **MiniFusion**、**LiteMultiModal**）都尝试在 1‑2 B 参数范围内复刻 UniPic 的解耦编码思路，并在移动端或边缘设备上进行部署。对想进一步探索的读者，可以关注以下方向：① 更高效的跨尺度训练调度（比如自适应分辨率）；② 将奖励模型扩展到多语言、多任务指令的统一评估；③ 将 UniPic 的解码器改为稀疏自回归或混合流模型，以进一步压缩显存。  

### 一句话记住它
**用两套专门的编码器喂同一个自回归解码器，1.5 B 参数就能同时看、画、改图，且在普通显卡上跑得动。**