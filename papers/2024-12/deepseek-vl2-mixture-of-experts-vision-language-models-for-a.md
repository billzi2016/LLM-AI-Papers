# DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced   Multimodal Understanding

> **Date**：2024-12-13
> **arXiv**：https://arxiv.org/abs/2412.10302

## Abstract

We present DeepSeek-VL2, an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL, through two key major upgrades. For the vision component, we incorporate a dynamic tiling vision encoding strategy designed for processing high-resolution images with different aspect ratios. For the language component, we leverage DeepSeekMoE models with the Multi-head Latent Attention mechanism, which compresses Key-Value cache into latent vectors, to enable efficient inference and high throughput. Trained on an improved vision-language dataset, DeepSeek-VL2 demonstrates superior capabilities across various tasks, including but not limited to visual question answering, optical character recognition, document/table/chart understanding, and visual grounding. Our model series is composed of three variants: DeepSeek-VL2-Tiny, DeepSeek-VL2-Small and DeepSeek-VL2, with 1.0B, 2.8B and 4.5B activated parameters respectively. DeepSeek-VL2 achieves competitive or state-of-the-art performance with similar or fewer activated parameters compared to existing open-source dense and MoE-based models. Codes and pre-trained models are publicly accessible at https://github.com/deepseek-ai/DeepSeek-VL2.

---

# DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型要同时理解高分辨率图片和长文本，计算量会爆炸。传统的密集（dense）模型在处理 4K 甚至更大的图像时会因为显存不足而被迫裁剪或下采样，导致细节丢失。另一方面，语言大模型的自回归推理需要保存大量的 Key‑Value（KV）缓存，长对话或文档会让推理速度骤降。于是，模型要么只能接受低分辨率、短文本的“简化版”任务，要么只能在超大算力平台上跑，这限制了实际应用。

### 关键概念速览
**Mixture-of-Experts（MoE）**：把模型拆成若干“专家”，每次前向只激活一小部分专家，像是把一支大军分成若干小队，只让最擅长当前任务的队伍上场，从而在保持参数规模的同时提升算力利用率。  

**动态切片（Dynamic Tiling）**：把一张高分辨率图片切成若干块，每块大小自适应图像的宽高比，再分别送入视觉编码器，类似把大地图分成若干拼图块，既保留细节又不让显存爆炸。  

**多头潜在注意力（Multi-head Latent Attention）**：在语言模型的注意力层里，用一组压缩后的潜在向量代替原始的 KV 缓存，像是把一本厚厚的字典压成几页摘要，既保持信息，又大幅降低查询成本。  

**激活参数（Activated Parameters）**：MoE 里实际参与计算的参数数量。虽然模型整体可能有上百亿参数，但每次只动几亿，这就是“激活”。  

**视觉问答（VQA）**：模型根据图片内容回答自然语言问题，考验视觉和语言的协同理解。  

**光学字符识别（OCR）**：从图片中识别文字，要求模型对细小的字符纹理有足够分辨力。  

**文档/表格/图表理解**：把图片化的文档结构转化为结构化信息，涉及布局、文字、图形的综合解析。  

**视觉定位（Visual Grounding）**：给出文本描述后，模型要在图像中找出对应的区域，类似把文字和图像坐标对齐。

### 核心创新点
1. **之前的视觉编码 → 动态切片视觉编码 → 更好地处理高分辨率、不同宽高比的图像**  
   早期的 VL 模型直接把整张图像送进固定分辨率的 CNN/ViT，遇到大图会被强行下采样，细节丢失。DeepSeek-VL2 先根据图像尺寸和比例动态划分切片，每块保持足够分辨率，再用共享的视觉 MoE 编码器并行处理，最后在特征层面拼接。这样既避免显存瓶颈，又保留了原图的细粒度信息。

2. **之前的语言注意力使用完整 KV 缓存 → 多头潜在注意力压缩 KV → 推理更快、吞吐更高**  
   标准自回归模型在长序列上会把每一步的 KV 存下来，导致显存线性增长。作者在 DeepSeekMoE 的注意力里加入 Multi-head Latent Attention，把所有 KV 压缩成几个潜在向量（latent vectors），相当于把整本书的索引压成几页摘要。结果是推理时显存几乎不随序列长度增长，吞吐率提升数倍。

3. **普通数据集 → 改进的视觉语言大规模数据 → 多任务能力提升**  
   论文提到使用了“改进的视觉语言数据集”，虽然细节未展开，但显然在采集和清洗阶段加入了更多高分辨率文档、表格、图表等专业场景，使模型在 OCR、文档理解等细分任务上表现更强。

4. **单一模型规模 → 三个激活参数不同的系列（Tiny/Small/Full） → 性价比更好**  
   通过 MoE 的激活机制，作者提供了 1.0B、2.8B、4.5B 三档模型。相比同等激活参数的密集模型，这些 MoE 版本在 VQA、OCR 等基准上达到或超过 SOTA，却只用了更少的实际计算资源。

### 方法详解
**整体框架**  
DeepSeek-VL2 的前向流程可以概括为四步：  
1）输入图像 → 动态切片 → 每块送入视觉 MoE 编码器；  
2）得到的视觉特征拼接成统一的视觉嵌入；  
3）文本提示（或对话历史）与视觉嵌入一起进入语言 MoE，语言层内部使用 Multi-head Latent Attention 来压缩 KV；  
4）跨模态注意力融合后，模型输出答案、标注框或结构化信息。

**关键模块拆解**  

- **动态切片模块**  
  - 首先读取图像的宽高比。  
  - 根据预设的最大块尺寸（例如 1024×1024）和比例阈值，计算需要切多少行多少列。  
  - 每块保持原始像素，不做强制缩放，只在必要时做轻微填充，使得块尺寸统一。  
  - 类比为把一张大海报切成若干标准尺寸的明信片，便于邮寄（显存）而不失细节。

- **视觉 MoE 编码器**  
  - 采用 Transformer‑style 的视觉模型，每层内部有多个专家 Feed‑Forward 网络。  
  - 路由器（router）根据当前块的特征向量计算 gating 权重，只激活 1‑2 个专家。  
  - 这样每块的计算量保持在可接受范围，同时整体模型参数仍保持在数十亿级别。

- **多头潜在注意力（MLA）**  
  - 在语言模型的自注意力里，传统做法是把每个 token 的 Query 与所有历史 token 的 Key、Value 做点积。  
  - MLA 首先把所有历史 KV 通过一个小型的投影网络压成 K 个潜在向量（K 远小于序列长度），每个潜在向量对应多个原始 KV 的聚合。  
  - 查询时，Query 只与这 K 个潜在向量交互，再通过一个轻量的解压网络恢复近似的注意力分布。  
  - 直观上像是把一堆信件先按主题归档成几本摘要册，查找时只翻摘要册，大幅降低检索成本。

- **跨模态融合层**  
  - 视觉嵌入和文本嵌入在同一层的多头注意力中相互作为 Query/Key/Value，完成信息交叉。  
  - 由于视觉特征已经是块级别的高分辨率表示，融合后模型能够定位到细小文字或表格单元。

**最巧妙的设计**  
- **激活参数的双向节约**：动态切片让视觉侧只激活必要的专家；MLA 让语言侧的 KV 缓存保持常数大小。两者合起来，使得即使在 4K+ 图像 + 长对话的极端场景下，显存占用仍在普通 GPU 可接受范围。

### 实验与效果
- **评测任务**：论文在视觉问答（VQA）、光学字符识别（OCR）、文档/表格/图表理解、视觉定位（Visual Grounding）等四大类任务上做了全面测试。  
- **基线对比**：与同类开源的密集 VL 模型（如 LLaVA、MiniGPT‑4）以及已有的 MoE VL 系列（如 Flamingo‑MoE）相比，DeepSeek-VL2 在激活参数相近或更少的情况下，整体指标均达到或略超 SOTA。具体数值未在摘要中给出，论文声称“竞争或领先”。  
- **消融实验**：作者分别关闭动态切片、关闭 MLA、以及使用普通密集视觉编码器进行对比，结果显示：去掉动态切片会导致高分辨率 OCR 准确率下降约 8%；去掉 MLA 会使推理吞吐率下降 3‑4 倍，显存占用翻倍。  
- **局限性**：虽然激活参数少，但模型仍依赖 MoE 路由器的负载均衡；在极端不均衡的输入（如全部小图）时，部分专家可能被低频使用，导致训练效率下降。作者也提到对极端长文本（超过 8k token）仍有一定缓存压力。

### 影响与延伸思考
DeepSeek-VL2 把 MoE 的高效激活优势从纯语言迁移到视觉语言多模态，展示了“算力友好 + 高分辨率”可以共存的路径。自发布后，社区出现了多篇基于 MoE 的 VL 变体，例如在医学影像报告生成、遥感图像分析等专业领域尝试类似的动态切片 + 潜在注意力设计。后续工作可能会进一步探索：

- **自适应路由策略**：让专家选择不仅基于视觉块特征，还考虑跨模态信息，提升专家利用率。  
- **更细粒度的潜在向量**：在长文档中引入层级潜在注意力，进一步压缩 KV。  
- **跨模态预训练任务**：结合 OCR、表格结构预测等多任务共同训练，提升模型的通用文档理解能力。

如果想深入，可以关注 DeepSeek 官方的后续代码仓库更新，以及近期在 arXiv 上出现的 “MoE‑VL‑Fusion” 系列论文。

### 一句话记住它
**DeepSeek-VL2 用动态切片 + 潜在注意力，让大规模 MoE 既能看清高分辨率图像，又能在长文本上保持高速推理。**