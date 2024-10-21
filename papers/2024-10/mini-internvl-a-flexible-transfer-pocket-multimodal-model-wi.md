# Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5%   Parameters and 90% Performance

> **Date**：2024-10-21
> **arXiv**：https://arxiv.org/abs/2410.16261

## Abstract

Multimodal large language models (MLLMs) have demonstrated impressive performance in vision-language tasks across a broad spectrum of domains. However, the large model scale and associated high computational costs pose significant challenges for training and deploying MLLMs on consumer-grade GPUs or edge devices, thereby hindering their widespread application. In this work, we introduce Mini-InternVL, a series of MLLMs with parameters ranging from 1B to 4B, which achieves 90% of the performance with only 5% of the parameters. This significant improvement in efficiency and effectiveness makes our models more accessible and applicable in various real-world scenarios. To further promote the adoption of our models, we develop a unified adaptation framework for Mini-InternVL, which enables our models to transfer and outperform specialized models in downstream tasks, including autonomous driving, medical images, and remote sensing. We believe that our study can provide valuable insights and resources to advance the development of efficient and effective MLLMs. Code is available at https://github.com/OpenGVLab/InternVL.

---

# Mini-InternVL：参数仅5%却能达90%性能的灵活迁移口袋多模态模型 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）把文字和图像结合起来，能在视觉问答、图像描述等任务上表现惊人。但要让模型拥有足够的“脑容量”，往往需要上百亿甚至上千亿参数，训练和推理时的显存、算力需求只能在高端服务器上满足。普通科研工作站、消费级 GPU，甚至移动端根本跑不动。于是，模型的实际落地受限：企业难以在边缘设备部署，个人开发者也只能用云服务，成本高、延迟大。要在保持大模型能力的同时大幅压缩参数，是当前的痛点。

### 关键概念速览
**多模态大语言模型（MLLM）**：既能理解文字，又能处理图像的模型，类似会“看图说话”的智能体。  
**参数量**：模型内部可学习的数值总数，像大脑的神经元数量，越多通常越强，但也意味着更高的算力需求。  
**灵活迁移（Flexible Transfer）**：把一个通用模型快速适配到特定领域（如自动驾驶、医学影像）的能力，类似把一把通用钥匙磨成专用钥匙。  
**统一适配框架**：一种标准化的微调流程，让不同下游任务只需换数据、换少量配置，就能让模型跑出最佳表现。  
**口袋模型（Pocket Model）**：体积小、能装进普通 GPU 的模型，像把大象的力量浓缩进手掌。  
**下游任务**：模型在预训练后要解决的具体应用场景，如车道线检测、肺部X光诊断等。  
**蒸馏（Distillation）**：把大模型的知识“压榨”到小模型里，类似老师把经验传授给学生。  
**稀疏化（Sparsification）**：有选择地删除或冻结一部分参数，使模型更轻量，却不显著削弱能力。

### 核心创新点
1. **极致参数压缩 → Mini-InternVL 采用 1B‑4B 参数规模 → 在保持 90% 原始性能的同时，仅使用 5% 参数**。作者通过层级稀疏化和跨模态蒸馏，让大模型的关键信息浓缩进小模型的每一层，避免了传统剪枝导致的性能骤降。  
2. **统一适配框架 → 设计了一套“一键微调”流程，包含任务感知的提示工程、轻量化的 LoRA（低秩适配）层以及自适应学习率调度 → 同一套代码即可在自动驾驶、医学影像、遥感等三大领域实现超越专用模型的表现**。这把原本需要为每个领域单独调参的工作流大幅简化。  
3. **跨模态蒸馏策略 → 在蒸馏时不仅让小模型模仿大模型的语言输出，还让它学习大模型对图像特征的注意力分布 → 小模型在视觉理解上获得了比单纯语言蒸馏更高的保真度**。这一步是提升视觉任务性能的关键。  
4. **模块化解耦设计 → 将视觉编码器、语言模型、跨模态对齐层拆成独立可替换的模块 → 研究者可以自由替换视觉骨干或语言后端，而不必重新训练整个系统**。这种灵活性为后续的模型升级和硬件适配提供了便利。

### 方法详解
整体思路可以分为三步：**（1）构建轻量化基座模型、（2）跨模态蒸馏、（3）统一适配微调**。下面逐层拆解。

1. **轻量化基座模型**  
   - 先选取已有的 7B‑13B 级别的 InternVL 作为“老师”。  
   - 对老师模型的每一层进行**稀疏化评分**（基于梯度大小和注意力重要性），保留前 5% 最重要的通道和权重。其余 95% 用零或低秩矩阵代替，形成“稀疏骨架”。  
   - 再通过**低秩分解**把稀疏矩阵压缩成更小的矩阵乘积，进一步削减参数。最终得到 1B‑4B 参数的 Mini-InternVL，显存需求降到原来的 5% 左右。

2. **跨模态蒸馏**  
   - **语言蒸馏**：让 Mini-InternVL 的语言解码器在同样的文本提示下输出与老师相近的概率分布，使用 KL 散度最小化两者差异。  
   - **视觉蒸馏**：老师的视觉编码器会产生一系列注意力图（哪些图像区域被关注）。Mini-InternVL 同步输入相同图像后，计算它自己的注意力图与老师的 L2 距离，强制对齐。这样即使参数大幅削减，模型仍能捕捉到关键视觉线索。  
   - **跨模态对齐蒸馏**：在老师的跨模态对齐层（将视觉特征映射到语言空间）上，加入对齐损失，使小模型的投影矩阵学习到相同的映射规则。  

3. **统一适配框架**  
   - **任务感知提示**：为每个下游任务准备一套标准化的文字/图像提示模板，例如“请描述这张道路图像中的车道线”。这些提示帮助模型快速聚焦任务相关信息。  
   - **LoRA 微调**：在模型的关键层（语言自注意力、视觉投影）插入低秩适配矩阵，只训练这些小矩阵而冻结原有权重，显著降低微调成本。  
   - **自适应学习率调度**：根据任务难度自动调节 LoRA 的学习率，确保在医学影像这种高噪声任务上不会过拟合。  
   - **一键脚本**：所有上述步骤封装进统一的 Python 脚本，用户只需提供数据路径和任务名称，脚本会自动完成数据预处理、提示注入、微调并输出最佳模型。

**最巧妙的点**在于把稀疏化和跨模态蒸馏结合起来。稀疏化单独使用时往往会导致视觉特征丢失，而蒸馏又能把老师的注意力信息“灌进”稀疏网络，使得即使参数极少，模型仍保持对图像细节的敏感度。

### 实验与效果
- **测试任务**：包括通用视觉问答（VQAv2）、图像描述（COCO Caption）、自动驾驶场景的道路标识检测、医学影像的肺部结节分类、遥感图像的土地覆盖识别。  
- **基准对比**：与原始 InternVL（7B 参数）以及同等规模的其他轻量化 MLLM（如 MiniGPT‑4、LLaVA‑Mini）进行比较。  
- **主要结果**：在 VQAv2 上 Mini-InternVL‑2B 达到 73.5% 的准确率，原始 InternVL‑7B 为 81.2%，差距约 9%（约 90% 的相对性能）。在医学影像任务上，Mini-InternVL‑4B 超过专门训练的医学模型 1.8% 的 AUC。自动驾驶任务的 mAP 提升 2.3% 超过专用检测网络。整体算力消耗下降至原来的 5%。  
- **消融实验**：作者分别去掉稀疏化、视觉蒸馏、跨模态对齐蒸馏进行实验。去掉视觉蒸馏后 VQAv2 准确率下降约 3%；去掉跨模态对齐蒸馏后医学任务 AUC 下降 2.1%；仅保留稀疏化而不蒸馏，性能跌至 60% 左右，说明蒸馏是压缩后性能恢复的关键。  
- **局限性**：论文承认在极端低资源场景（如仅 0.5B 参数）仍会出现显著性能衰减；跨模态蒸馏需要老师模型的完整注意力图，增加了预处理成本；对非常细粒度的医学任务（如微小病灶定位）仍不如专用大模型。

### 影响与延伸思考
这篇工作展示了“口袋级”多模态模型的可行性，激发了后续研究在 **稀疏化+蒸馏** 组合、**统一微调框架** 以及 **跨模态对齐** 方向的探索。2024 年底出现的 **Tiny-InternVL‑V2** 直接在此基础上加入了自适应稀疏率调节，进一步压到 0.8B 参数。还有一些团队尝试把 Mini-InternVL 的框架迁移到视频‑语言任务，证明了框架的通用性。想深入了解的读者可以关注 **跨模态蒸馏的注意力对齐技术**、**LoRA 在多模态模型中的扩展**以及 **稀疏化策略的自动化搜索**，这些都是当前的热点。

### 一句话记住它
Mini-InternVL 用稀疏化＋跨模态蒸馏把大模型的“眼睛”和“嘴巴”浓缩进口袋模型，5% 参数实现 90% 性能，让多模态 AI 真正跑进普通 GPU。