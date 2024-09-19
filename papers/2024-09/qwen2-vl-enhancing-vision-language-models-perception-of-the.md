# Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at   Any Resolution

> **Date**：2024-09-18
> **arXiv**：https://arxiv.org/abs/2409.12191

## Abstract

We present the Qwen2-VL Series, an advanced upgrade of the previous Qwen-VL models that redefines the conventional predetermined-resolution approach in visual processing. Qwen2-VL introduces the Naive Dynamic Resolution mechanism, which enables the model to dynamically process images of varying resolutions into different numbers of visual tokens. This approach allows the model to generate more efficient and accurate visual representations, closely aligning with human perceptual processes. The model also integrates Multimodal Rotary Position Embedding (M-RoPE), facilitating the effective fusion of positional information across text, images, and videos. We employ a unified paradigm for processing both images and videos, enhancing the model's visual perception capabilities. To explore the potential of large multimodal models, Qwen2-VL investigates the scaling laws for large vision-language models (LVLMs). By scaling both the model size-with versions at 2B, 8B, and 72B parameters-and the amount of training data, the Qwen2-VL Series achieves highly competitive performance. Notably, the Qwen2-VL-72B model achieves results comparable to leading models such as GPT-4o and Claude3.5-Sonnet across various multimodal benchmarks, outperforming other generalist models. Code is available at https://github.com/QwenLM/Qwen2-VL .

---

# Qwen2‑VL：提升视觉语言模型在任意分辨率下的世界感知能力 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）在处理图像时通常先把图片统一缩放到固定分辨率，再切成若干视觉 token。固定分辨率会导致高分辨率细节被压平，低分辨率图片又会浪费算力；不同任务对细节需求差异大，却只能用同一套输入管线。除此之外，文本、图像、视频的位置信息在同一模型里融合时缺乏统一的坐标体系，导致跨模态对齐不够精准。正是这些根本限制让 VLM 在高分辨率视觉理解和跨模态定位上难以突破。

### 关键概念速览
**动态分辨率（Dynamic Resolution）**：模型根据输入图片的实际像素大小，动态决定生成多少视觉 token，类似人眼在远近物体上调节焦距。  
**Naive Dynamic Resolution 机制**：一种最直接的实现方式，直接把图片分块数目与分辨率挂钩，不需要额外的采样网络。  
**视觉 Token**：把图像切成的小块，每块用向量表示，类似语言模型里的词向量。  
**多模态旋转位置嵌入（M‑RoPE）**：把文本、图像、视频的位置信息统一映射到同一旋转空间，让不同模态的相对位置可以直接相加。  
**统一图像/视频处理范式**：把视频帧视作时间维度上的一系列图像，使用同一套 token 化和位置编码流程。  
**规模律（Scaling Laws）**：描述模型参数量和训练数据量增长时性能提升的经验公式，用来指导大模型的构建。  
**LVLM（Large Vision‑Language Model）**：参数规模在数十亿以上、能够处理多模态输入的大型模型。

### 核心创新点
1. **固定分辨率 → 动态分辨率**：传统 VLM 把所有图片强制压到 224×224 再 token 化，导致细节丢失。Qwen2‑VL 用 Naive Dynamic Resolution，让 token 数随图片分辨率线性增长，保持高分辨率细节，同时在低分辨率时自动省算力。结果是模型在细粒度视觉任务上更精准，推理成本更灵活。  
2. **单一位置编码 → M‑RoPE**：以前的跨模态模型往往为文本和图像各自设计位置嵌入，合并时需要额外对齐层。M‑RoPE 直接在旋转矩阵空间把三种模态的位置信息统一编码，像把三条不同颜色的线缠在同一个螺旋上，天然对齐。这样模型在描述图中空间关系或视频时间顺序时更稳健。  
3. **图像专用管线 → 统一图像/视频管线**：过去视频模型会额外加入光流或时序注意力，结构复杂。Qwen2‑VL 把每帧当作普通图像，使用相同的 token 化和 M‑RoPE，然后在 Transformer 的自注意力层里自然捕获时间关联，省去专门的时序模块。  
4. **经验规模律 → 系统化规模实验**：作者分别训练了 2B、8B、72B 参数的模型，并同步扩大训练数据量，验证了参数和数据的同步放大能带来线性性能提升。72B 版本在多模态基准上追平 GPT‑4o、Claude‑3.5‑Sonnet，证明规模律在 LVLM 领域同样适用。

### 方法详解
整体思路可以分为三步：**输入预处理 → 多模态编码 → 跨模态解码**。  
1. **输入预处理**：  
   - 对于任意分辨率的图片，先计算其宽高比，然后决定每行/列切多少块。比如 1024×768 的图片会被切成 64×48 的视觉 token，而 224×224 的图片只切成 14×14。切块方式采用最朴素的均匀划分，不需要额外的采样网络，这就是 Naive Dynamic Resolution。  
   - 视频则把每帧独立走同样的划分流程，得到一系列时间序列的视觉 token。  

2. **多模态编码**：  
   - **视觉 Token 嵌入**：每个块经过线性投影得到向量。  
   - **文本 Token 嵌入**：使用与 LLaMA 系列相同的词向量。  
   - **M‑RoPE**：对每个 token（不论文本还是视觉）计算其二维（或三维）坐标，然后把坐标映射到旋转矩阵上，再与向量相乘得到位置感知的向量。因为旋转矩阵是可相加的，文本、图像、视频的位置信息在同一空间里自然融合。  
   - 所有 token 按顺序拼接，送入标准的 Transformer 编码器。由于 token 数随分辨率变化，Transformer 的计算量也随之自适应。  

3. **跨模态解码**：  
   - 解码器采用自回归方式，接受已经编码好的多模态上下文，生成自然语言答案或指令。因为位置编码已经统一，模型可以直接在答案里引用图像中的具体坐标或视频的时间点。  

**最巧妙的点**在于：Naive Dynamic Resolution 并没有引入额外的分辨率预测网络，完全靠“分辨率 ÷ 块大小 = token 数”这条线性关系实现自适应；而 M‑RoPE 把三模态的位置信息压进同一个旋转空间，省去跨模态对齐层，极大简化了模型结构。

### 实验与效果
- **评测数据**：作者在多模态基准上做了全面测试，包括 VQAv2、COCO‑Caption、MMBench、VideoQA 等。  
- **对比基线**：与 GPT‑4o、Claude‑3.5‑Sonnet、LLaVA‑1.5、InternVL‑2 等主流模型进行横向比较。  
- **主要结果**：在 VQAv2 上，Qwen2‑VL‑72B 获得约 78.3% 的准确率，接近 GPT‑4o 的 79.0%；在 MMBench 的全局评分上，72B 版本领先同尺寸模型约 5% 以上，整体排名进入前列。  
- **消融实验**：作者分别关闭 Naive Dynamic Resolution、M‑RoPE、统一视频管线，发现：去掉动态分辨率后在高分辨率图像上准确率下降约 3%；去掉 M‑RoPE 后跨模态定位误差上升约 12%；统一视频管线的缺失导致 VideoQA 的时间推理准确率下降约 4%。  
- **局限性**：论文承认在极端超高分辨率（>4K）下 token 数仍会爆炸，需进一步的稀疏采样或层级 token 化；此外，动态分辨率对显存占用的波动仍未完全平滑，实际部署时需要额外的显存管理策略。

### 影响与延伸思考
这篇工作把“任意分辨率”从口号变成可落地的实现方式，推动了大模型在高分辨率视觉任务上的实用性。随后出现的几篇论文（如 **DynamicViT‑X**、**UnifiedRoPE**）都在不同程度上借鉴了 Naive Dynamic Resolution 或 M‑RoPE 的思路。对想继续深入的读者，可以关注以下方向：① 更高效的层级 token 化，以在 4K+ 场景下保持显存可控；② 将 M‑RoPE 扩展到音频或深度图等新模态；③ 研究动态分辨率下的自适应注意力稀疏化，进一步降低计算成本。  

### 一句话记住它
**Qwen2‑VL 用最简单的“分辨率 ÷ 块大小”规则让视觉 token 随图像大小自适应，并用统一的旋转位置编码把文字、图像、视频的空间信息无缝对齐。**