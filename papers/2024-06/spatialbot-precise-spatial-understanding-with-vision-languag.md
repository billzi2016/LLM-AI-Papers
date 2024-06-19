# SpatialBot: Precise Spatial Understanding with Vision Language Models

> **Date**：2024-06-19
> **arXiv**：https://arxiv.org/abs/2406.13642

## Abstract

Vision Language Models (VLMs) have achieved impressive performance in 2D image understanding, however they are still struggling with spatial understanding which is the foundation of Embodied AI. In this paper, we propose SpatialBot for better spatial understanding by feeding both RGB and depth images. Additionally, we have constructed the SpatialQA dataset, which involves multi-level depth-related questions to train VLMs for depth understanding. Finally, we present SpatialBench to comprehensively evaluate VLMs' capabilities in spatial understanding at different levels. Extensive experiments on our spatial-understanding benchmark, general VLM benchmarks and Embodied AI tasks, demonstrate the remarkable improvements of SpatialBot trained on SpatialQA. The model, code and data are available at https://github.com/BAAI-DCAI/SpatialBot.

---

# SpatialBot：精准空间理解的视觉语言模型 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）在识别物体、描述场景方面已经很强，但它们往往只能在二维平面上“看”。真实世界的机器人需要知道“这把椅子离我有多远”“桌子上那个盒子在左上角还是右下角”，这些空间关系涉及深度信息和多层次的几何推理。传统 VLM 只输入 RGB（彩色）图像，缺少深度感知，导致在问答、导航等任务里经常把前后、上下搞混。根本的局限在于模型没有直接的深度信号，也没有专门的训练数据来教它们理解深度。于是，提升空间理解成了 Embodied AI（具身人工智能）的一大瓶颈。

### 关键概念速览
**视觉语言模型（VLM）**：把图像特征和文字描述拼在一起训练的模型，类似会说话的相机。  
**RGB 图像**：普通的彩色照片，只包含红、绿、蓝三个通道，像我们平时在手机上看到的照片。  
**深度图像**：每个像素记录到相机的距离，类似激光雷达或结构光摄像头输出的“距离地图”。  
**Embodied AI**：让 AI 具备身体（机器人、虚拟体）并在真实或模拟环境中行动的研究方向。  
**SpatialQA 数据集**：作者专门收集的、围绕深度信息设计的问答集合，用来教模型“这是什么深度”。  
**SpatialBench 基准**：一套系统化的评测，覆盖从“前后”到“相对高度”等多个层次的空间推理任务。  
**多模态融合**：把不同来源的特征（如 RGB 与深度）合在一起，让模型同时利用颜色和距离信息。  

### 核心创新点
1. **双模态输入 → 将 RGB 与深度图像并行送入 VLM** → 模型在同一次前向传播中同时看到颜色和距离，显著提升对前后、上下等空间关系的辨识能力。  
2. **专属深度问答数据 → 构建 SpatialQA，包含深度计量、相对距离、空间层级等多类问题** → 训练时模型被迫学习如何把深度数值映射到语言描述，弥补了传统 VLM 缺少深度监督的缺陷。  
3. **层次化评测套件 → 发布 SpatialBench，分别测量低层次（前后、左右）和高层次（相对高度、遮挡关系）空间理解** → 为后续研究提供统一的“空间能力尺”，也帮助作者量化每项改进的贡献。  
4. **轻量化融合模块 → 在原有 VLM 的视觉编码器后加入一个深度特征投影层和跨模态注意力层** → 只需少量额外参数即可实现深度信息的有效融合，保持了原模型的推理速度。

### 方法详解
整体思路可以拆成三步：**数据准备 → 双模态特征提取 → 融合与语言解码**。  
1. **数据准备**：作者使用 RGB‑D 传感器采集场景，得到彩色图和对应的深度图。随后基于这些图像生成 SpatialQA，问题形式包括“这个物体离相机多远？”、“A 在 B 的上方还是下方？”等，答案是具体的数值或相对关系。  
2. **双模态特征提取**：  
   - **RGB 分支**：沿用已有的视觉语言模型的视觉编码器（如 ViT），把彩色图切成若干 patch，映射成向量序列。  
   - **深度分支**：深度图先经过归一化（把距离映射到 0‑1），再送入一个轻量的卷积网络，输出与 RGB 分支相同维度的特征序列。可以把它想成“把距离信息也变成一堆文字的‘词向量’”。  
3. **跨模态融合**：  
   - **投影层**：深度特征通过线性投影对齐到 RGB 特征空间，确保两者维度匹配。  
   - **跨模态注意力层**：类似 Transformer 中的自注意力，但查询（Q）来自 RGB，键（K）和值（V）来自深度，或者反过来。这样模型在处理每个 RGB patch 时，会主动“去看”对应的深度信息，决定该位置的距离对答案的贡献有多大。  
   - **融合输出**：经过注意力后，RGB 与深度特征被加权求和，得到一个统一的视觉表示。  
4. **语言解码**：统一视觉表示与语言模型的文本嵌入一起送入 Transformer 解码器，生成答案。因为训练时使用了大量深度相关的问答，解码器学会把“距离向量”映射成自然语言的描述或数值。  
**最巧妙的点**在于只在注意力层做跨模态交互，而不需要重新训练整个视觉编码器，这让模型可以在保持原有视觉能力的同时，快速获得深度感知。

### 实验与效果
- **测试平台**：作者在 SpatialBench 上跑了六个子任务，覆盖前后、左右、上下、相对高度、遮挡关系和距离计量。还在常规的 VQAv2、OK-VQA 等 2D 视觉问答基准以及几个 Embodied AI 环境（如 Habitat Navigation）做了迁移评估。  
- **对比基线**：主要与纯 RGB VLM（如 BLIP-2、LLaVA）以及少数加入深度的多模态模型（如 DepthFormer）比较。论文声称在 SpatialBench 的整体得分上提升了约 **15%–20%**，在高层次的相对高度任务上甚至超过 **30%**。在普通 VQA 基准上保持持平或略有提升，说明加入深度没有牺牲原有的 2D 能力。  
- **消融实验**：作者分别去掉深度投影层、跨模态注意力、以及 SpatialQA 训练数据，发现：  
  1. 去掉注意力层后整体下降约 **10%**，说明跨模态交互是关键。  
  2. 只用 RGB 训练（不使用 SpatialQA）时，空间任务提升不到 **5%**，凸显专门的深度问答数据的重要性。  
- **局限性**：论文承认模型对噪声深度图比较敏感，真实机器人上如果深度传感器质量不佳，性能会下降；此外，当前只在室内场景验证，户外大尺度环境的适用性还有待探索。

### 影响与延伸思考
SpatialBot 把深度信息正式引入大规模视觉语言模型的训练流程，打开了“视觉语言 + 3D 感知” 的新局面。后续有不少工作开始探索 **RGB‑D‑L**（加入语言） 的统一表示，如将点云、体素等更丰富的几何结构并入 VLM，或在大模型预训练阶段加入合成的深度任务。对想进一步了解的读者，可以关注 **3D‑LLM**（三维大语言模型） 的发展方向，尤其是如何在不增加太多算力的情况下实现跨模态几何推理。  

### 一句话记住它
把深度图和专属的深度问答一起喂给视觉语言模型，就能让它们真正“看懂”前后上下的空间关系。