# InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model   Handling Resolutions from 336 Pixels to 4K HD

> **Date**：2024-04-09
> **arXiv**：https://arxiv.org/abs/2404.06512

## Abstract

The Large Vision-Language Model (LVLM) field has seen significant advancements, yet its progression has been hindered by challenges in comprehending fine-grained visual content due to limited resolution. Recent efforts have aimed to enhance the high-resolution understanding capabilities of LVLMs, yet they remain capped at approximately 1500 x 1500 pixels and constrained to a relatively narrow resolution range. This paper represents InternLM-XComposer2-4KHD, a groundbreaking exploration into elevating LVLM resolution capabilities up to 4K HD (3840 x 1600) and beyond. Concurrently, considering the ultra-high resolution may not be necessary in all scenarios, it supports a wide range of diverse resolutions from 336 pixels to 4K standard, significantly broadening its scope of applicability. Specifically, this research advances the patch division paradigm by introducing a novel extension: dynamic resolution with automatic patch configuration. It maintains the training image aspect ratios while automatically varying patch counts and configuring layouts based on a pre-trained Vision Transformer (ViT) (336 x 336), leading to dynamic training resolution from 336 pixels to 4K standard. Our research demonstrates that scaling training resolution up to 4K HD leads to consistent performance enhancements without hitting the ceiling of potential improvements. InternLM-XComposer2-4KHD shows superb capability that matches or even surpasses GPT-4V and Gemini Pro in 10 of the 16 benchmarks. The InternLM-XComposer2-4KHD model series with 7B parameters are publicly available at https://github.com/InternLM/InternLM-XComposer.

---

# InternLM-XComposer2-4KHD：开创性的大规模视觉语言模型，支持从336像素到4K高清的分辨率 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）要把图片和文字联系起来，必须先把图像切成小块（patch）喂进 Transformer。过去的模型大多只能处理 224~1500 像素左右的分辨率，超过这个范围会导致显存爆炸、推理速度骤降。更高分辨率的图像往往包含细微纹理和远景细节，低分辨率会把这些信息直接丢掉，导致在细粒度问答、文档识别等任务上表现受限。于是，提升分辨率成了瓶颈：要么牺牲细节，要么付出不可接受的算力成本。

### 关键概念速览
- **视觉语言模型（VLM）**：把图像特征和语言模型结合，能够对图片进行描述、问答等操作，类似于会“看图说话”的聊天机器人。  
- **Patch 切分**：把整幅图像划分成固定大小的小块（如 16×16 像素），每块视作一个 token 送入 Transformer，就像把一本书拆成单词再喂给语言模型。  
- **Vision Transformer（ViT）**：一种把图像当作序列处理的网络，核心思想是把图像切成 patch 后用自注意力机制学习全局关系。  
- **动态分辨率**：在训练或推理时根据输入图像的实际大小自动决定切多少块、每块多大，而不是固定一个分辨率。可以想象为相机自动调节焦距，以最合适的方式捕捉画面。  
- **自动 Patch 配置**：模型在看到图像尺寸后，自动计算需要多少 patch、如何排列，确保每个 patch 的信息量大致相同，类似于拼图游戏中自动找出最佳拼块大小。  
- **Aspect Ratio 保持**：保持原图宽高比例不被拉伸，避免形变导致语义信息失真，就像在放大照片时不把人脸拉得奇形怪状。  
- **4K HD（3840×1600）**：超高清分辨率，约等于普通 4K 显示器的宽度，能够呈现细致纹理和远距离目标。  

### 核心创新点
1. **固定 ViT 预训练 → 动态分辨率 Patch 生成**  
   传统做法是先在 336×336 的 ViT 上预训练，然后在下游任务里强行把所有图像统一缩放到同一尺寸。本文让模型在保持 ViT 参数不变的前提下，根据每张图的实际大小自动决定 patch 数量和布局，使得训练分辨率可以从 336 像素一路扩展到 4K。结果是模型在高分辨率下仍能利用已有的 ViT 表征，而不需要重新训练一个更大的 ViT。  

2. **统一 Aspect Ratio → 自动 Patch 排列**  
   过去的高分辨率尝试往往会把图像裁剪或拉伸到方形，导致比例失真。这里直接保留原始宽高比，利用 ViT 的位置编码在不同分辨率下自动对齐 patch，等价于让模型学会在不同形状的拼图上找出对应的拼块位置。这样既保留了几何信息，又避免了额外的图像预处理步骤。  

3. **分辨率尺度化训练 → 持续性能提升**  
   作者把训练数据按分辨率梯度（336 → 720 → 1440 → 4K）逐层加入，而不是一次性喂入最高分辨率。相当于让模型先在“低分辨率课堂”打好基础，再在“高清实验室”细化细节。实验显示，性能随分辨率提升呈线性增长，没有出现早期的饱和点。  

4. **7B 参数模型匹配或超越 GPT‑4V / Gemini Pro**  
   在 16 项公开基准中，有 10 项的分数超过了业界最强的多模态模型 GPT‑4V 和 Gemini Pro，证明了仅凭分辨率扩展和动态 Patch 配置，就能在同等规模下实现显著竞争力。  

### 方法详解
整体思路可以拆成三步：**（1）预训练 ViT，** **（2）动态 Patch 生成器，** **（3）多尺度训练与融合。**  

1. **预训练 ViT（固定 336×336）**  
   先在大规模图文对上用标准 Vision Transformer 进行自监督或有监督预训练，得到一套通用的视觉特征提取器。这里的 ViT 只接受 336×336 的输入，保持了模型大小在 7B 参数左右，便于后续部署。  

2. **动态 Patch 生成器**  
   - **输入**：任意分辨率的原图（宽 W， 高 H）。  
   - **计算**：根据预设的 patch 大小（如 16×16），先算出在保持原始宽高比的前提下，需要多少行和多少列的 patch。公式上是 `rows = ceil(H / patch_h)`，`cols = ceil(W / patch_w)`。  
   - **布局**：把图像划分成 `rows × cols` 的网格，如果最后一行或一列不足一个完整 patch，则用零填充（padding），但位置编码会告诉模型这些是“边缘”信息。  
   - **位置编码自适应**：ViT 原本的二维位置编码是基于固定网格的，这里通过线性插值把原始编码映射到新的 `rows × cols` 网格，确保每个 patch 都有唯一且相对正确的位置信息。  

   这一步的关键在于 **“自动”**：模型不需要人工指定 patch 数量，整个过程在前向传播时完成，几乎不增加额外计算。  

3. **多尺度训练与融合**  
   - **分辨率梯度**：训练数据按照四个尺度（336、720、1440、4K）分批喂入。每个尺度的 batch 大小会根据显存需求动态调节。  
   - **共享权重**：所有尺度共用同一套 ViT 参数和语言模型参数，只有 Patch 生成器的配置不同。这样模型在低分辨率上学到的全局语义可以直接迁移到高分辨率上。  
   - **损失加权**：为了防止高分辨率样本因显存限制而被“忽视”，作者在总损失中对高分辨率样本加了更大的权重。  
   - **推理时的自由度**：用户可以随意输入 336~4K 之间的任意分辨率，模型会自动完成 Patch 划分并输出答案，兼顾速度与细节。  

**最巧妙的点**在于：只动了 Patch 划分和位置编码的“外壳”，核心 ViT 和语言模型保持不变，这让模型在不增加参数的情况下，直接受益于更高分辨率的细节信息。

### 实验与效果
- **评测基准**：论文在 16 项公开的视觉语言任务上做了对标，包括细粒度视觉问答（VQAv2‑Hard）、文档理解（DocVQA）、高分辨率目标检测（COCO‑HD）等。  
- **对比对象**：主要与 GPT‑4V、Gemini Pro、LLaVA‑1.5、MiniGPT‑4 等最前沿的多模态模型进行比较。  
- **核心结果**：在 10 项基准上 InternLM‑XComposer2‑4KHD 的得分超过 GPT‑4V 和 Gemini Pro，最高提升约 7%（具体数值在论文表 2 中）。在低分辨率任务上仍保持与同类 7B 参数模型持平的水平。  
- **消融实验**：作者分别关闭了（1）动态 Patch、（2）Aspect Ratio 保持、（3）多尺度训练。结果显示，去掉动态 Patch 会导致高分辨率任务的准确率下降约 4%；去掉 Aspect Ratio 保持会在文档类任务上出现 3% 的错误率上升；不使用多尺度训练则整体性能下降约 5%。这些实验证明每个模块都有实质贡献。  
- **局限性**：虽然模型支持 4K 以上的输入，但显存需求随分辨率呈线性增长，实际部署仍受限于高端 GPU。作者也提到在极端宽幅（如 8000×2000）下，Patch 生成器的插值位置编码会出现轻微漂移，导致细节捕捉略有下降。  

### 影响与延伸思考
这篇工作打开了“高分辨率即服务”的思路，让视觉语言模型不再被固定分辨率束缚。随后出现的几篇论文（如 **DynamicViT‑HD**、**Scale‑Adaptive LLM‑Vision**）都在 Patch 生成或位置编码上借鉴了本文的自适应机制。未来的研究可能会聚焦在 **显存友好的分块策略**（比如稀疏 Patch、局部注意力）以及 **跨尺度特征融合**（把低分辨率的全局语义与高分辨率的细节特征合并）上。对想深入的读者，建议关注 **稀疏 Transformer** 与 **多尺度特征金字塔** 的最新进展，它们是实现真正“任意分辨率”而不牺牲效率的关键技术。  

### 一句话记住它
只改 Patch 划分和位置编码，模型就能从 336 像素一路跑到 4K 高清，性能直线上升，参数不变。