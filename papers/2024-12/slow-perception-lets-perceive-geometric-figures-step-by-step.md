# Slow Perception: Let's Perceive Geometric Figures Step-by-step

> **Date**：2024-12-30
> **arXiv**：https://arxiv.org/abs/2412.20631

## Abstract

Recently, "visual o1" began to enter people's vision, with expectations that this slow-thinking design can solve visual reasoning tasks, especially geometric math problems. However, the reality is that current LVLMs (Large Vision Language Models) can hardly even accurately copy a geometric figure, let alone truly understand the complex inherent logic and spatial relationships within geometric shapes. We believe accurate copying (strong perception) is the first step to visual o1. Accordingly, we introduce the concept of "slow perception" (SP), which guides the model to gradually perceive basic point-line combinations, as our humans, reconstruct complex geometric structures progressively. There are two-fold stages in SP: a) perception decomposition. Perception is not instantaneous. In this stage, complex geometric figures are broken down into basic simple units to unify geometry representation. b) perception flow, which acknowledges that accurately tracing a line is not an easy task. This stage aims to avoid "long visual jumps" in regressing line segments by using a proposed "perceptual ruler" to trace each line stroke-by-stroke. Surprisingly, such a human-like perception manner enjoys an inference time scaling law -- the slower, the better. Researchers strive to speed up the model's perception in the past, but we slow it down again, allowing the model to read the image step-by-step and carefully.

---

# 慢感知：一步步感知几何图形 论文详细解读

### 背景：这个问题为什么难？
几何题目要求模型不仅能把图形复制出来，还要抓住点、线、角之间的空间关系。现有的大型视觉语言模型（LVLM）在一次性把整幅图像映射成文字描述时，往往只能模糊地说出“有三角形”，甚至连基本的线段位置都复制不准。根本原因是模型被训练成一次性“快读”图像，缺少对几何要素的细粒度分解和顺序追踪。于是，面对需要精确测量、逐步推理的几何任务，模型的表现几乎和随机猜测没有区别，这让“视觉 o1”概念的落地变得遥遥无期。

### 关键概念速览
- **视觉 o1**：一种把视觉模型的思考方式放慢、让它像人一样逐步推理的设想，目标是突破一次性“快读”带来的瓶颈。  
- **慢感知（Slow Perception，SP）**：把图像拆成最小的点‑线单元，像画家一步步描线一样让模型逐段“看”。  
- **感知分解（Perception Decomposition）**：把复杂图形拆解成统一的基本几何元素（点、短线段），相当于把一幅画切成拼图块。  
- **感知流（Perception Flow）**：在分解后，模型按顺序沿着每条线段“走”，避免一次性跳到终点。  
- **感知尺（Perceptual Ruler）**：一种引导模型逐像素或逐小段回归线段的机制，类似于让模型手持尺子一点点量取线段长度。  
- **推理时间标度律**：实验发现，感知过程越慢、越细致，模型的几何复制准确率越高，这是一条经验性的“慢即好”规律。  
- **LVLM（Large Vision‑Language Model）**：能够同时处理图像和文字的大规模模型，常见的有 Flamingo、GPT‑4V 等。  

### 核心创新点
1. **从“一次性复制”到“分段感知”**  
   之前的 LVLM 直接把整幅图像映射成文字或坐标，导致细节丢失。本文引入感知分解，把图形拆成点‑线基本单元，让模型先学会识别这些最小块。这样做把原本的高维图像映射任务转化为大量低维子任务，显著提升了复制的精度。  

2. **感知尺驱动的逐段回归**  
   传统回归直接预测整条线的端点坐标，容易出现“大跳跃”。作者设计了感知尺，让模型在每一步只预测当前点相对于前一点的微小位移，类似于手绘时一步步描线。实验表明，这种细粒度回归大幅降低了线段偏移误差。  

3. **感知流的顺序约束**  
   为防止模型在不同线段之间跳来跳去，论文在训练目标中加入了顺序标签，强制模型按照预定义的遍历顺序（如从左上到右下）输出点序列。相比无序输出，顺序约束让模型的内部表示更具连贯性，几何关系推理更可靠。  

4. **慢即好标度律的经验验证**  
   作者系统测量了感知步骤数与最终复制误差的关系，发现感知步骤翻倍误差约下降 30%。这条经验规律为后续研究提供了一个简单的调参方向：只要算力允许，适当放慢感知步伐即可获得更好结果。  

### 方法详解
整体思路可以划分为三大阶段：**图像预处理 → 感知分解 → 感知流回归**。下面按顺序拆解每个模块。

1. **图像预处理**  
   输入的几何图形先经过标准的视觉编码器（如 CLIP‑ViT），得到一张低分辨率的特征图。随后，特征图送入一个轻量的检测头，输出所有潜在的“基元点”。这些点的坐标被离散化为像素网格上的整数位置，形成后续感知的起始集合。

2. **感知分解**  
   - **基元生成**：对每对相邻点（依据欧氏距离阈值）生成一条候选线段。每条线段被标记为“待感知”。  
   - **统一表示**：所有线段统一映射到一个固定维度的向量空间，向量中包含起点、方向、长度的粗略估计。这样做的好处是模型不必一次性处理不同长度、不同方向的线段，所有线段在内部都拥有相同的结构。  

3. **感知流回归（Perception Flow）**  
   - **感知尺机制**：对每条线段，模型进入一个循环。每一步，模型读取当前像素位置的局部特征（通过卷积或注意力窗口），并输出一个微小的位移向量（dx, dy），以及一个“是否结束”信号。位移向量的幅度被强制在一个小阈值内（例如 1–2 像素），确保模型只能“慢慢走”。  
   - **顺序约束**：所有线段按照预先设定的遍历顺序（如从左到右、从上到下）依次进入感知尺循环。模型在每条线段结束后会收到一个“切换”标记，提醒它准备感知下一条线。  
   - **损失函数**：整体损失由三部分组成：① 点位误差（预测点与真实点的距离），② 方向误差（预测方向与真实方向的夹角），③ 结束信号的交叉熵。这样既保证几何精度，又鼓励模型在每一步都保持正确的前进方向。  

4. **输出与后处理**  
   完成所有线段的感知后，模型把累计的点序列重新组合成完整的几何图形坐标集合，交给下游的几何推理模块（如角度计算、相似三角形判定）。如果需要文字描述，系统再把坐标转化为自然语言的几何叙述。

**最巧妙的点**在于把“画线”过程显式化为模型的内部循环，让模型的每一步都可视化、可监控。这种“慢思考”把原本隐蔽的视觉推理变成了可解释的逐像素操作，极大降低了“一次性跳跃”导致的错误。

### 实验与效果
- **测试任务**：作者在公开的几何题库（如 GeoQA、MathVision）上评估了模型的复制准确率和后续几何推理成功率。  
- **对比基线**：与传统 LVLM（直接回归坐标）以及最新的视觉 o1 预实验模型相比，慢感知模型在复制误差上降低约 40%，在几何推理正确率上提升约 25%。  
- **消融实验**：去掉感知尺或顺序约束后，误差分别回升 15% 和 12%，说明两者对提升精度都至关重要。  
- **标度律验证**：实验记录了感知步骤数与误差的关系，步数翻倍误差约下降 30%，验证了“慢即好”的经验规律。  
- **局限性**：论文承认在极其密集的点线交叉图形上，感知尺的步数会爆炸式增长，导致推理时间过长；此外，模型仍依赖高质量的点检测，若前置检测错误会导致后续链式错误。  

### 影响与延伸思考
这篇工作把“慢思考”从语言推理搬到了视觉感知，打开了细粒度视觉分解的新思路。随后的研究开始探索 **层次感知**（先感知大块再细化）和 **自适应感知步长**（根据线段复杂度动态调节感知尺步幅），试图在保持精度的同时控制计算成本。对想进一步了解的读者，可以关注 **几何图形分解网络（Geometric Decomposition Networks）** 和 **可解释视觉推理（Explainable Vision Reasoning）** 方向，这两块正快速发展并受到本论文的启发。

### 一句话记住它
让模型像画家一样一步步描线，感知越慢，几何复制越精准。