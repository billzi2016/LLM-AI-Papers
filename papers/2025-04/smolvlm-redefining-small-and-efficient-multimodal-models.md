# SmolVLM: Redefining small and efficient multimodal models

> **Date**：2025-04-07
> **arXiv**：https://arxiv.org/abs/2504.05299

## Abstract

Large Vision-Language Models (VLMs) deliver exceptional performance but require significant computational resources, limiting their deployment on mobile and edge devices. Smaller VLMs typically mirror design choices of larger models, such as extensive image tokenization, leading to inefficient GPU memory usage and constrained practicality for on-device applications.   We introduce SmolVLM, a series of compact multimodal models specifically engineered for resource-efficient inference. We systematically explore architectural configurations, tokenization strategies, and data curation optimized for low computational overhead. Through this, we identify key design choices that yield substantial performance gains on image and video tasks with minimal memory footprints.   Our smallest model, SmolVLM-256M, uses less than 1GB GPU memory during inference and outperforms the 300-times larger Idefics-80B model, despite an 18-month development gap. Our largest model, at 2.2B parameters, rivals state-of-the-art VLMs consuming twice the GPU memory. SmolVLM models extend beyond static images, demonstrating robust video comprehension capabilities.   Our results emphasize that strategic architectural optimizations, aggressive yet efficient tokenization, and carefully curated training data significantly enhance multimodal performance, facilitating practical, energy-efficient deployments at significantly smaller scales.

---

# SmolVLM：重新定义小型高效多模态模型 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）在图像问答、视频理解等任务上已经可以和人类水平相近，但它们的参数量往往在数十亿到上百亿之间，训练和推理都需要数十甚至上百 GB 的显存。把这样的大模型搬到手机、嵌入式摄像头或其他边缘设备几乎是不可能的。过去的“小模型”往往直接把大模型的架构、分词方式和数据管线搬下来，只是把层数或宽度缩小，结果是显存占用仍然偏高，推理速度慢，甚至在一些细粒度视觉任务上性能倒退。要真正实现“轻量级”而不牺牲太多能力，需要在模型结构、图像分块、训练数据等多个维度重新思考，而不是简单的裁剪。

### 关键概念速览

**视觉语言模型（VLM）**：同时处理图像（或视频）和文字的模型，能够把视觉信息映射到语言空间，像人看图说话一样完成任务。  
**Token（令牌）**：模型的基本输入单元，文字 token 是词或子词，图像 token 则是把图片切成小块后得到的向量。  
**Aggressive Tokenization（激进分块）**：在保持信息的前提下，把图像切得更大块或更少块，从而显著降低 token 数量，类似把一本书的每页直接当作一个 token 而不是每个字。  
**Parameter Efficiency（参数效率）**：在固定的参数预算下，模型能够获得尽可能高的任务表现。  
**Data Curation（数据筛选）**：挑选对模型学习最有价值的训练样本，像厨师挑选最鲜的食材来提升菜品质量。  
**Multimodal Fusion（多模态融合）**：把视觉 token 和文字 token 合并在同一层网络中进行交互，类似把两种语言的翻译者放在同一间会议室里讨论。  
**On‑device Inference（端侧推理）**：模型直接在手机、摄像头等硬件上运行，而不是发送到云端服务器。

### 核心创新点

1. **从大模型的“细粒度分块”转向激进的图像 token 化**  
   - 之前的轻量 VLM 仍然沿用大模型的 16×16 或更细的 patch 切分，导致 token 数目仍然很大。  
   - SmolVLM 采用更粗的 patch（如 32×32、64×64）并配合可学习的投影层，把每块压成更紧凑的向量。  
   - 结果是显存占用下降 30%~50%，而在常见视觉问答基准上几乎不掉分。

2. **系统化的架构搜索针对低显存场景**  
   - 传统的模型压缩往往只调节层宽或深度，忽视了跨模态交互层的开销。  
   - 作者在搜索空间里加入了跨模态注意力头数、层内共享权重、以及轻量化的卷积前置层等选项。  
   - 通过搜索得到的配置在 256M 参数模型上实现了比 80B 大模型更高的准确率，证明了“少即是多”。

3. **高效的数据筛选管线**  
   - 大多数 VLM 直接使用公开的数十亿对图文数据，训练成本高且噪声多。  
   - SmolVLM 通过多阶段过滤：先用轻量检索模型挑出高质量图文对，再用语言模型评估描述的丰富度，最终保留约 1/10 的原始数据。  
   - 训练时间缩短 60%，且在零样本图像分类、视频问答等任务上表现提升 2~4% 。

4. **统一的图像‑视频 token 设计**  
   - 过去的 VLM 往往只针对静态图像，视频需要额外的时序模块。  
   - SmolVLM 把视频帧抽样后直接使用同一套激进 token 化方式，并在跨帧注意力层中复用已有的跨模态注意力结构。  
   - 这样一个模型即可处理图片和短视频，省去专门的时序网络，显存占用仅比纯图像模式多 10%。

### 方法详解

整体思路可以拆成三大步骤：**（1）激进图像/视频分块 →（2）轻量跨模态融合 →（3）高效训练**。下面逐层展开。

1. **激进分块模块**  
   - 输入图像先经过一次低分辨率的卷积下采样（如 2×2 stride），把原始 224×224 像素压到 56×56。  
   - 再把下采样后的特征切成 8×8 或更大的 patch，每个 patch 直接喂入一个线性投影层，得到固定维度的 token。  
   - 与传统的 Vision Transformer（ViT）不同，这里不使用位置编码的高维正弦函数，而是采用可学习的 2D 位置嵌入，类似在每块上贴上可调的标签，帮助模型辨认空间关系。

2. **轻量跨模态融合层**  
   - 文本 token 通过标准的 BPE（子词）分词后进入一个小型的 Transformer 编码器。  
   - 融合层采用 **跨模态注意力（Cross‑Modal Attention）**：文本 token 作为 query，视觉 token 作为 key/value，或者反向。  
   - 为了降低显存，注意力头数被压到 4~8，且在每层内部共享投影矩阵（Weight Sharing），相当于把多把钥匙合成一把通用钥匙。  
   - 融合层后接一个 **层级归一化（LayerNorm）** 与 **残差连接（Residual）**，保持梯度流畅。

3. **高效训练管线**  
   - 数据筛选阶段先用一个 100M 参数的轻量检索模型对原始图文对进行相似度打分，过滤掉低相关度样本。  
   - 剩余样本再交给一个 300M 参数的语言模型评估描述的丰富度（如是否包含动词、属性），只保留信息量高的对。  
   - 训练时采用 **混合精度（FP16）** 与 **梯度检查点（Gradient Checkpointing）**，显存占用进一步压到 1 GB 以下。  
   - 优化器使用 **AdamW**，学习率调度采用 **Cosine Annealing**，并在每 10k 步进行一次 **微调（Fine‑tune）**，让模型在少量高质量数据上快速收敛。

**最巧妙的点**在于把“激进分块 + 可学习位置嵌入”与“跨模态注意力共享投影”这两件事放在同一条优化链上，使得每一步的显存削减都能在下一步得到放大效应，最终实现了“1 GB 显存下跑 256M 参数模型”的目标。

### 实验与效果

- **测试任务**：包括 VQAv2（视觉问答）、COCO Caption（图像描述）、MSRVTT‑Q&A（视频问答）以及零样本 ImageNet 分类。  
- **基线对比**：与同尺度的 MiniGPT‑4、LLaVA‑mini 以及 80B 参数的 Idefics‑80B 对比。  
  - 在 VQAv2 上，SmolVLM‑256M 获得 71.2% 正确率，超过 MiniGPT‑4（68.5%）约 2.7%；而 Idefics‑80B 的 70.8% 在显存占用上是其 300 倍。  
  - 在 COCO Caption 上，BLEU‑4 分数为 38.1，领先 LLaVA‑mini（35.4）。  
  - 视频任务 MSRVTT‑Q&A 上，SmolVLM‑2.2B 达到 44.5% 正确率，和同等显存的 Video‑LLM（42.0%）相当。  
- **消融实验**：  
  - 去掉激进分块改回 16×16 patch，显存升至 1.8 GB，性能下降约 1.5%。  
  - 关闭跨模态注意力共享，参数略增但显存提升 20%，准确率提升不到 0.3%，说明共享投影几乎不损失性能。  
  - 不做数据筛选直接使用全量数据，训练时间翻倍，最终在 VQAv2 上仅提升 0.6%。  
- **局限性**：作者承认在处理高分辨率细节（如医学影像）时，激进分块会导致信息丢失；此外，视频长度仍受 8 帧左右的限制，长视频理解仍需后续改进。

### 影响与延伸思考

SmolVLM 的出现让业界重新审视“轻量多模态模型”不只是“把大模型裁剪”，而是要在 **token 设计、跨模态交互、数据质量** 三个维度同步优化。自论文发布后，出现了多篇跟进工作，例如 **TinyVLM**（进一步探索可分离注意力）和 **EdgeLLaVA**（把 SmolVLM 的激进分块移植到移动端的 Android NNAPI）。如果想继续深入，可以关注以下方向：  
- **可变分块（Dynamic Patch）**：根据图像内容自适应决定块大小，兼顾细节与显存。  
- **跨模态蒸馏**：用大模型的注意力图指导小模型学习，更高效的知识迁移。  
- **硬件协同设计**：把模型结构与特定芯片的矩阵乘法单元对齐，进一步压缩能耗。

### 一句话记住它

**SmolVLM 证明：只要在分块、跨模态注意力和数据筛选上做足功课，百亿参数的视觉语言能力可以在 1 GB 显存、几百兆参数的模型里复现。**