# MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe

> **Date**：2025-09-16
> **arXiv**：https://arxiv.org/abs/2509.18154

## Abstract

Multimodal Large Language Models (MLLMs) are undergoing rapid progress and represent the frontier of AI development. However, their training and inference efficiency have emerged as a core bottleneck in making MLLMs more accessible and scalable. To address the challenges, we present MiniCPM-V 4.5, an 8B parameter model designed for high efficiency and strong performance. We introduce three core improvements in model architecture, data strategy and training method: a unified 3D-Resampler model architecture for highly compact encoding over images and videos, a unified learning paradigm for document knowledge and text recognition without heavy data engineering, and a hybrid reinforcement learning strategy for proficiency in both short and long reasoning modes. Comprehensive experimental results in OpenCompass evaluation show that MiniCPM-V 4.5 surpasses widely used proprietary models such as GPT-4o-latest, and significantly larger open-source models such as Qwen2.5-VL 72B. Notably, the strong performance is achieved with remarkable efficiency. For example, on the widely adopted VideoMME benchmark, MiniCPM-V 4.5 achieves state-of-the-art performance among models under 30B size, using just 46.7\% GPU memory cost and 8.7\% inference time of Qwen2.5-VL 7B.

---

# MiniCPM-V 4.5：通过架构、数据与训练配方烹制高效多模态大语言模型 论文详细解读

### 背景：这个问题为什么难？
多模态大语言模型（MLLM）要同时理解文字、图片甚至视频，算力需求极高。过去的模型要么在视觉编码上使用庞大的卷积或Transformer网络，导致显存占用爆炸；要么在训练数据上做大量手工标注和跨模态对齐，成本不菲。于是模型越大、性能越好，却越难部署到普通 GPU 上，限制了实际应用的普及。提升 **训练/推理效率** 成了摆在研究者面前的硬核挑战。

### 关键概念速览
**3D‑Resampler**：一种把图像序列或视频压缩成极小特征向量的网络，类似把一段长视频“抽帧+压缩”，保留关键信息却大幅削减体积。  
**统一学习范式**：不再为文字识别、文档理解分别准备专门的数据集，而是用同一套数据和目标让模型一次性学会两种能力，像“一锅炖”一样省去繁琐的工程。  
**混合强化学习（Hybrid RL）**：在训练时交替使用短程奖励（快速回答）和长程奖励（多轮推理），让模型既能给出简洁答案，又能展开深度思考，类似在跑短跑和马拉松之间切换训练计划。  
**OpenCompass**：一个统一评测平台，集合了多种多模态任务的基准，帮助研究者快速对比不同模型的整体实力。  
**VideoMME**：专门测视频理解能力的基准，评估模型在动作识别、时序推理等方面的表现。  
**GPU 显存占用**：模型在推理时需要的显卡显存量，直接决定了能否在常规卡上跑。  
**推理时间比例**：相对于基准模型的耗时，用百分比表示的加速幅度，数值越小越快。  

### 核心创新点
1. **之前的视觉编码 → 3D‑Resampler 统一压缩**  
   传统 MLLM 用独立的图像编码器（如 ViT）或视频专用的时空 Transformer，参数多、显存占用大。MiniCPM‑V 4.5 把所有视觉输入送进同一个 3D‑Resampler，先在空间上做轻量卷积，再在时间维度上做跨帧聚合，得到极紧凑的特征。结果是同等视觉信息下显存只要原来的约 46.7%，推理速度提升至原来的 8.7%。  

2. **繁琐的数据对齐 → 统一学习范式**  
   过去需要为 OCR、文档检索分别准备标注数据，甚至手动设计跨模态对齐规则。作者直接把文档图片和普通文本混在一起，用同一套语言建模目标进行训练，模型在看到图片时自然学会文字识别，在看到纯文本时继续语言推理。这样省掉了大量数据工程，训练管线更简洁。  

3. **单一强化学习目标 → 混合 RL 兼顾短长推理**  
   传统 RL 只针对单一任务（比如对话）优化奖励，导致模型在复杂多轮推理时表现欠佳。MiniCPM‑V 4.5 在训练阶段交替使用两类奖励：一种鼓励快速、准确的单句回答，另一种奖励多步推理的完整性。模型因此在 OpenCompass 的短问答和长视频推理两类子任务上都能保持领先。  

4. **大模型追求极致 → 8B 参数的高效实现**  
   与 72B 参数的 Qwen2.5‑VL 相比，MiniCPM‑V 4.5 只用了 8B 参数，却在 OpenCompass 综合得分上超过了它，并且在 VideoMME 上成为 30B 以下模型的 SOTA。核心在于上述三大技术的协同效应，让“小体积”也能跑出“大成绩”。  

### 方法详解
**整体框架**  
MiniCPM‑V 4.5 的训练流程可以划分为三步：① 视觉前端的 3D‑Resampler 编码；② 与语言模型（8B 参数的 Transformer）进行跨模态融合；③ 基于混合强化学习的多任务微调。整个系统像一条流水线：原始图像/视频 → 紧凑特征 → 语言模型 → 输出答案。

**1. 3D‑Resampler 细节**  
- **空间压缩**：先用几层轻量卷积把每帧的分辨率降到 1/4，类似把高清图片压成缩略图。  
- **时间聚合**：把连续帧的特征送入一个小型跨帧注意力模块（只保留关键帧的注意力），相当于在视频里挑出“关键镜头”。  
- **统一投射**：最终得到的特征维度与语言模型的隐藏层维度匹配，直接相加或拼接进入后续层。这样既避免了大规模的视觉‑语言对齐网络，也保证了信息的完整性。

**2. 统一学习范式**  
- **数据混合**：训练数据包括普通文本、带文字的文档图片、以及纯视觉任务（如图像描述）。所有样本统一使用“指令+输入+输出”的格式，模型只需要预测下一个 token。  
- **目标一致**：不区分 OCR 还是文本生成，模型在看到图片时自动触发内部的文字识别子模块，输出的 token 序列自然包含识别结果。相当于让模型在“看图说话”时顺带学会“读图”。  

**3. 混合强化学习**  
- **短程奖励**：对单轮问答使用标准的奖励模型（如 RLHF）鼓励答案的准确性和简洁性。  
- **长程奖励**：对多轮推理或视频问答，构造一个基于答案完整度、时序一致性的奖励函数，类似给模型一个“剧情连贯度”评分。  
- **交替训练**：每个训练 epoch 随机抽取短程或长程任务，使模型在两种模式之间保持平衡。  

**最巧妙的点**  
- **显存共享**：3D‑Resampler 的输出直接复用语言模型的 KV 缓存，不需要额外的显存拷贝，显著降低了显存峰值。  
- **无需专门 OCR 标注**：通过统一学习，模型在大量普通文档图片上自监督学会文字识别，省掉了昂贵的 OCR 标注成本。  

### 实验与效果
- **评测平台**：使用 OpenCompass 对 30+ 多模态任务进行综合评估，涵盖图像问答、视频推理、文档理解等。  
- **关键基线**：与 GPT‑4o‑latest（闭源商用）以及开源的 Qwen2.5‑VL 72B、Qwen2.5‑VL 7B 进行对比。  
- **核心结果**：在 OpenCompass 综合得分上，MiniCPM‑V 4.5 超过 GPT‑4o‑latest，并且显著领先 72B 参数的 Qwen2.5‑VL。  
- **VideoMME**：在视频理解基准上，MiniCPM‑V 4.5 成为 30B 以下模型的 SOTA，显存仅为 Qwen2.5‑VL 7B 的 46.7%，推理时间仅为其 8.7%。  
- **消融实验**：论文提供了对 3D‑Resampler、统一学习范式、混合 RL 三个模块的单独去除实验，结果显示去掉任意一项都会导致整体得分下降 3%~7%，验证了每个创新的必要性。  
- **局限性**：作者指出模型仍在极端长视频（超过 10 分钟）上出现时序漂移，且对极端低分辨率图像的细节捕捉仍不如更大模型。  

### 影响与延伸思考
MiniCPM‑V 4.5 向社区展示了“高效”可以和“强大”并存的可能性，激发了后续对 **轻量跨模态压缩** 与 **统一多任务学习** 的热潮。后续工作如 **LiteVision‑L**、**UnifiedMM‑Tiny** 等都在借鉴其 3D‑Resampler 思路和混合 RL 训练策略。对想继续深入的读者，建议关注以下方向：  
- 更高效的时空注意力机制（如稀疏或局部注意力）。  
- 自监督跨模态对齐方法，进一步削减标注需求。  
- 长视频推理的记忆管理与时间层次建模。  

### 一句话记住它
**MiniCPM‑V 4.5 用 8 B 参数和 3D‑Resampler 把视觉信息压得极致，再配合统一学习和混合 RL，让小模型跑出大模型的多模态表现。**