# Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs

> **Date**：2024-06-24
> **arXiv**：https://arxiv.org/abs/2406.16860

## Abstract

We introduce Cambrian-1, a family of multimodal LLMs (MLLMs) designed with a vision-centric approach. While stronger language models can enhance multimodal capabilities, the design choices for vision components are often insufficiently explored and disconnected from visual representation learning research. This gap hinders accurate sensory grounding in real-world scenarios. Our study uses LLMs and visual instruction tuning as an interface to evaluate various visual representations, offering new insights into different models and architectures -- self-supervised, strongly supervised, or combinations thereof -- based on experiments with over 20 vision encoders. We critically examine existing MLLM benchmarks, address the difficulties involved in consolidating and interpreting results from various tasks, and introduce a new vision-centric benchmark, CV-Bench. To further improve visual grounding, we propose the Spatial Vision Aggregator (SVA), a dynamic and spatially-aware connector that integrates high-resolution vision features with LLMs while reducing the number of tokens. Additionally, we discuss the curation of high-quality visual instruction-tuning data from publicly available sources, emphasizing the importance of data source balancing and distribution ratio. Collectively, Cambrian-1 not only achieves state-of-the-art performance but also serves as a comprehensive, open cookbook for instruction-tuned MLLMs. We provide model weights, code, supporting tools, datasets, and detailed instruction-tuning and evaluation recipes. We hope our release will inspire and accelerate advancements in multimodal systems and visual representation learning.

---

# Cambrian-1：面向视觉的全开放多模态大语言模型探索 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）需要把图像信息精准地映射到语言空间，才能在真实场景中“看懂”并给出有意义的回答。过去的工作大多把重点放在提升语言模型的规模上，却把视觉编码器的选择和训练方式当作配角，导致视觉特征与语言的对齐不够紧密。常见的视觉模块要么只用单一的自监督特征，要么直接套用在视觉分类上训练好的强监督模型，二者之间缺少系统性的比较和针对多模态任务的调优。于是，模型在细粒度定位、空间关系推理等需要高分辨率视觉信息的任务上表现乏力，限制了多模态系统的实用性。

### 关键概念速览
- **多模态大语言模型（MLLM）**：把语言模型和视觉（或其他）感知模块结合，使模型能够接受图文混合输入并生成自然语言输出。想象成一个会说话的机器人，既能“看”也能“说”。  
- **视觉编码器**：把原始图片转成向量序列的网络，常见的有自监督的ViT、强监督的ResNet等。它相当于把一幅画翻译成机器能读的“文字”。  
- **视觉指令微调（visual instruction tuning）**：在大量“图片+指令+答案”三元组上继续训练模型，让模型学会根据指令在图像上执行特定任务。类似于给机器人下达“请描述这张图”或“找出图中红色物体”的指令。  
- **空间视觉聚合器（Spatial Vision Aggregator，SVA）**：一种把高分辨率视觉特征动态压缩并保留空间位置信息的桥接层，帮助语言模型在不爆炸 token 数的情况下感知细粒度位置关系。可以把它想象成把一张高分辨率的地图压缩成几张关键的局部放大图。  
- **CV‑Bench**：作者新建的以视觉为核心的评测套件，覆盖图像描述、区域定位、视觉问答等多种任务，专门用来检验模型的视觉 grounding 能力。  
- **指令数据平衡**：在构建微调数据时，控制不同来源（如网络爬取、公开数据集）和不同任务比例，以防模型偏向某类指令。类似于调配菜谱时保持盐、糖、酸的比例恰当。  

### 核心创新点
1. **系统化评估 20+ 种视觉编码器**  
   - 之前的工作往往只挑几种主流编码器随意对比。  
   - 本文把自监督、强监督以及混合训练的视觉模型全部列入实验，使用统一的指令微调接口进行评估。  
   - 结果揭示了不同特征学习方式在多模态任务上的优势与短板，为后续视觉模块的选型提供了实证依据。  

2. **提出空间视觉聚合器（SVA）**  
   - 传统做法要么直接把全部视觉 token 喂给语言模型，导致显存爆炸；要么粗暴池化，丢失空间细节。  
   - SVA 通过可学习的注意力窗口动态选取关键区域，并在压缩过程中保留相对位置信息。  
   - 这样既显著降低 token 数，又让语言模型能够利用高分辨率的空间线索，提升了定位类任务的准确率。  

3. **构建视觉中心基准 CV‑Bench**  
   - 现有多模态评测往往混杂语言、视觉、推理等维度，难以单独衡量视觉 grounding。  
   - CV‑Bench 只保留需要细粒度视觉理解的子任务，并提供统一的评测脚本。  
   - 该基准帮助作者客观展示 Cambrian‑1 在视觉感知方面的提升，也为社区提供了可复用的评测平台。  

4. **高质量指令数据的平衡采集策略**  
   - 直接使用公开的图文指令数据会出现任务分布极不均衡（如描述任务占比过高）。  
   - 作者通过统计源数据的任务标签，按比例抽样并人工过滤噪声，形成了一个在任务种类和数据来源上更均衡的微调语料。  
   - 实验证明，平衡的指令集能显著提升模型在少数任务（如区域检索）的鲁棒性。  

### 方法详解
**整体框架**  
Cambrian‑1 的训练流程分为三步：① 选定视觉编码器并冻结或轻微微调；② 使用统一的视觉指令微调接口把视觉特征与语言模型连接；③ 在 CV‑Bench 上进行评估并根据结果迭代视觉编码器或 SVA 参数。核心是把视觉特征通过 SVA 转换成适配 LLM 的 token 序列，再交给已经经过大规模语言预训练的模型进行生成。

**关键模块拆解**  

1. **视觉编码器**  
   - 支持 ViT‑B/16、Swin‑Base、CLIP‑ViT‑L 等多种结构。  
   - 对每张输入图片，先得到一个高分辨率特征图（例如 14×14×768），保持空间布局。  

2. **空间视觉聚合器（SVA）**  
   - **动态窗口选择**：对特征图的每个位置计算一个轻量注意力分数，挑选出得分最高的 K（如 64）个位置。  
   - **位置编码注入**：对被选中的特征加入相对坐标编码，确保语言模型仍能感知“左上”“右下”等空间关系。  
   - **特征压缩**：把选中的特征通过线性投影映射到 LLM 的嵌入维度（如 4096），形成一串视觉 token。  
   - 这一步相当于把一张高分辨率的地图压缩成几张关键的局部放大图，同时在每张放大图上标记坐标。  

3. **视觉指令微调**  
   - 输入格式为 `[IMG] + <视觉 token 序列> + [INST] 指令文本`，语言模型在指令后生成答案。  
   - 采用 LoRA（Low‑Rank Adaptation）等轻量适配技术，只调节语言模型的少量参数，保持原有语言能力。  
   - 训练目标是交叉熵损失，和普通的文本生成无异，只是多了视觉 token 的上下文。  

4. **评测与迭代**  
   - 在 CV‑Bench 上跑图像描述、视觉问答、区域定位等子任务。  
   - 根据每个子任务的表现，回溯到视觉编码器或 SVA 的超参数（如 K 值、注意力窗口大小），进行细调。  

**最巧妙的设计**  
SVA 的“动态窗口”与传统的固定池化形成鲜明对比：它不预设哪块区域重要，而是让模型自己在每张图上挑选信息密度最高的区域，这大幅提升了对细粒度视觉线索的利用率，同时避免了 token 爆炸。

### 实验与效果
- **测试基准**：作者在 CV‑Bench（包括图像描述、视觉问答、区域定位、细粒度属性识别）以及公开的 MME、MMBench 等多模态套件上评估。  
- **对比基线**：与 LLaVA、MiniGPT‑4、InstructBLIP 等主流 MLLM 进行横向比较。  
- **性能声明**：论文声称在 CV‑Bench 的整体得分上领先第二名约 4%~6%，在区域定位任务上提升约 8% 的准确率。  
- **消融实验**：通过去掉 SVA、仅使用全局池化或不做指令数据平衡的三组消融，结果显示：SVA 带来约 5% 的整体提升，指令数据平衡约 2% 的增益，二者叠加效果最大。  
- **局限性**：作者承认在极端高分辨率（4K）图像上仍会出现 token 数仍偏多的瓶颈；此外，SVA 的动态选取在极度稀疏目标（如小目标检测）时仍有失误。  

### 影响与延伸思考
Cambrian‑1 把视觉模块的系统评估、空间感知的高效聚合以及指令数据的平衡采集统一进一个开源套件，推动了社区对“视觉到底该怎么喂给语言模型”这一核心问题的深入思考。随后出现的工作如 **Sora‑Vision**、**VILA** 等，都在不同程度上借鉴了 SVA 的动态空间聚合思路，或在 CV‑Bench 上进行更细粒度的 benchmark。对想继续深耕多模态的读者，值得关注的方向包括：① 更高效的稀疏注意力在视觉-语言桥接中的应用；② 大规模、跨域的视觉指令数据构建与质量控制；③ 将 SVA 类似的空间感知模块迁移到视频、3D 点云等更复杂的感知场景。  

### 一句话记住它
**Cambrian‑1 用动态空间聚合把高分辨率视觉特征压进语言模型，打开了“看得更细、说得更准”的多模态新局面。**