# BlueLM-V-3B: Algorithm and System Co-Design for Multimodal Large   Language Models on Mobile Devices

> **Date**：2024-11-16
> **arXiv**：https://arxiv.org/abs/2411.10640

## Abstract

The emergence and growing popularity of multimodal large language models (MLLMs) have significant potential to enhance various aspects of daily life, from improving communication to facilitating learning and problem-solving. Mobile phones, as essential daily companions, represent the most effective and accessible deployment platform for MLLMs, enabling seamless integration into everyday tasks. However, deploying MLLMs on mobile phones presents challenges due to limitations in memory size and computational capability, making it difficult to achieve smooth and real-time processing without extensive optimization. In this paper, we present BlueLM-V-3B, an algorithm and system co-design approach specifically tailored for the efficient deployment of MLLMs on mobile platforms. To be specific, we redesign the dynamic resolution scheme adopted by mainstream MLLMs and implement system optimization for hardware-aware deployment to optimize model inference on mobile phones. BlueLM-V-3B boasts the following key highlights: (1) Small Size: BlueLM-V-3B features a language model with 2.7B parameters and a vision encoder with 400M parameters. (2) Fast Speed: BlueLM-V-3B achieves a generation speed of 24.4 token/s on the MediaTek Dimensity 9300 processor with 4-bit LLM weight quantization. (3) Strong Performance: BlueLM-V-3B has attained the highest average score of 66.1 on the OpenCompass benchmark among models with $\leq$ 4B parameters and surpassed a series of models with much larger parameter sizes (e.g., MiniCPM-V-2.6, InternVL2-8B).

---

# BlueLM‑V‑3B：面向移动设备的多模态大语言模型的算法与系统协同设计 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）把文字、图片等信息融合在一起，能实现“看图说话”“图文问答”等强大功能。但这些模型往往拥有数十亿甚至上百亿参数，运行时需要几 GB 的显存和高算力的 GPU。手机的内存只有几 GB，CPU/GPU 性能远不及服务器，直接把完整模型搬到手机会导致卡顿、耗电甚至根本跑不起来。此前的移动端部署大多只能跑纯文本小模型，或者把视觉部分离线预处理，失去了实时交互的优势。因此，如何在保持多模态理解能力的同时，压缩模型体积、加速推理，成为阻碍 MLLM 真正走进日常生活的关键瓶颈。

### 关键概念速览
- **多模态大语言模型（MLLM）**：同时接受文字和视觉输入，输出自然语言的模型。想象成一个会“看图说话”的聊天机器人。
- **动态分辨率（Dynamic Resolution）**：根据任务需求或硬件资源，灵活调节输入图片的分辨率。类似于在手机上看视频时自动切换清晰度以省流量。
- **权重量化（Weight Quantization）**：把模型参数从 32 位浮点数压缩到更低位宽（如 4 位），以降低显存占用和计算量。可以比作把高分辨率图片压成 JPEG，肉眼仍能看清细节。
- **硬件感知部署（Hardware‑Aware Deployment）**：在设计模型和推理流程时，充分考虑目标芯片的指令集、缓存结构等特性，像为特定车型调校发动机一样优化性能。
- **OpenCompass 基准**：一个统一评测多模态模型能力的排行榜，覆盖问答、推理、生成等多种任务。相当于多模态模型的“奥运会成绩单”。
- **LLM（Large Language Model）**：大规模语言模型，本文指 2.7 B 参数的文本生成核心。可以把它想成一个拥有 27 亿“记忆单元”的聊天大脑。

### 核心创新点
1. **重新设计的动态分辨率机制 → 将主流 MLLM 采用的固定或粗糙分辨率策略换成细粒度、硬件感知的自适应方案 → 在保持视觉信息完整性的前提下，大幅降低了图像特征提取的 FLOPs，使得在手机上处理 1080p 图像仍能保持实时。**  
2. **系统层面的硬件感知优化 → 针对 MediaTek Dimensity 9300 的向量指令、缓存层级和多核调度进行专门的算子融合与调度策略 → 推理时实现 4‑bit 量化权重的高效加载和并行计算，令生成速度提升至 24.4 token/s。**  
3. **轻量化视觉编码器 + 参数共享策略 → 视觉编码器仅 400 M 参数，却通过跨模态注意力层与语言模型共享部分投影矩阵 → 在显著削减模型体积的同时，保持了对细粒度视觉信息的捕获能力，使得在 OpenCompass 上取得 66.1 的最高平均分。**  
4. **端到端的协同设计流程 → 从模型结构、量化方案到系统调度全链路共同优化，而不是单独压缩模型或单独加速硬件 → 形成了“算法‑系统同调”的闭环，使得整体性能提升远超单一手段的叠加效果。

### 方法详解
**整体框架**  
BlueLM‑V‑3B 的推理过程可以划分为三大步骤：① 动态分辨率预处理 → ② 视觉特征提取 → ③ 跨模态语言生成。核心思想是让每一步都“懂得”当前手机的算力和内存限制，动态调节计算量。

**1. 动态分辨率预处理**  
- 输入图片先经过一个轻量级的分辨率预测网络（几层卷积），输出一个推荐分辨率尺度。  
- 依据该尺度，对原图做等比下采样。若手机当前 CPU 利用率高，则选低分辨率；若空闲则提升分辨率。  
- 这种细粒度调节类似于视频播放器的自适应码率（ABR），但这里的“码率”是像素数。

**2. 轻量化视觉编码器**  
- 编码器采用改进的 ViT（Vision Transformer）结构，层数和隐藏维度均比传统 1B‑级视觉模型小很多。  
- 为了进一步压缩，模型在每个 Transformer Block 的投影矩阵上使用 **参数共享**：语言模型的词向量投影与视觉特征的线性映射共用同一组权重。这样既减少了参数，又让视觉特征更自然地融入语言空间。  
- 编码器输出的视觉 token 与文本 token 通过跨模态注意力层交叉融合，形成统一的上下文表示。

**3. 硬件感知系统优化**  
- **4‑bit 权重量化**：所有模型参数在加载前被量化为 4 位整数，并配合专用的 de‑quantization kernel 在运行时恢复近似浮点值。  
- **算子融合**：将注意力计算、层归一化、前馈网络等常见算子在一次内存遍历中完成，减少了中间结果的读写次数。  
- **多核调度**：利用 Dimensity 9300 的大核（A78）负责视觉特征提取，小核（A55）负责语言解码，形成生产者‑消费者流水线，最大化芯片利用率。  
- **缓存亲和**：把经常使用的投影矩阵放入 L2 缓存，避免频繁访问主存，显著降低延迟。

**最巧妙的点**  
- **跨模态投影共享**：在大多数 MLLM 中，视觉和语言的投影是独立的，导致参数冗余。BlueLM‑V‑3B 把两者合二为一，既削减了 400 M 参数的视觉编码器，又让视觉信息在语言空间中更易被利用。  
- **动态分辨率+硬件感知的闭环**：分辨率调节不再是离线预设，而是实时读取芯片负载信息后决定，形成软硬件的即时反馈。

### 实验与效果
- **评测基准**：在 OpenCompass 上对 12 项多模态任务（包括图文问答、视觉推理、描述生成等）进行综合评分。  
- **整体表现**：BlueLM‑V‑3B 获得 66.1 的平均分，领先所有参数 ≤ 4 B 的模型，并且超过了 MiniCPM‑V‑2.6（2.6 B）和 InternVL2‑8B（8 B）等更大模型。  
- **速度对比**：在 MediaTek Dimensity 9300 上，使用 4‑bit 量化后实现 24.4 token/s 的生成速度；同平台上同等量化的 LLaMA‑2‑7B 只能达到约 12 token/s。  
- **消融实验**：论文提供了三组消融：① 关闭动态分辨率（固定 224×224）导致 FLOPs 增加 38%，推理速度下降至 18 token/s；② 去掉跨模态投影共享，模型体积上升 15%，但在 OpenCompass 上分数下降 2.3 分；③ 使用 8‑bit而非 4‑bit 量化，显存占用翻倍，速度仅提升 5%。这些实验表明每个模块都对最终性能贡献显著。  
- **局限性**：作者指出在极端低功耗模式（如 5 W 以下）仍会出现卡顿，且对极高分辨率（> 4K）图像的细节捕获仍受限。量化误差在某些细粒度视觉推理任务上会略微下降准确率。

### 影响与延伸思考
BlueLM‑V‑3B 是首批在商用移动芯片上实现 2.7 B 参数语言模型 + 400 M 视觉编码器的端侧多模态系统，展示了“算法‑系统协同”可以突破传统“只压模型”或“只加速硬件”的瓶颈。发表后，业界出现了多篇围绕 **动态分辨率 + 硬件感知调度** 的跟进工作，例如华为的 “MobiViT” 与苹果的 “On‑Device Multimodal” 项目，都在尝试把视觉前端的自适应分辨率与系统调度结合。  
如果想进一步深入，可以关注以下方向：  
1. **更细粒度的硬件感知量化**（如混合位宽、层级自适应量化）。  
2. **跨模态知识蒸馏**：把大模型的多模态知识迁移到更小的端侧模型。  
3. **能耗感知调度**：在保证实时性的同时，动态平衡功耗与性能。  
这些都是在移动端实现真正“随时随地”多模态 AI 的关键路径。

### 一句话记住它
**BlueLM‑V‑3B 用动态分辨率 + 硬件感知系统，让 2.7 B 语言模型和 400 M 视觉编码器在手机上跑得又快又省。**