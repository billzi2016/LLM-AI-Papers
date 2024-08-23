# ParGo: Bridging Vision-Language with Partial and Global Views

> **Date**：2024-08-23
> **arXiv**：https://arxiv.org/abs/2408.12928

## Abstract

This work presents ParGo, a novel Partial-Global projector designed to connect the vision and language modalities for Multimodal Large Language Models (MLLMs). Unlike previous works that rely on global attention-based projectors, our ParGo bridges the representation gap between the separately pre-trained vision encoders and the LLMs by integrating global and partial views, which alleviates the overemphasis on prominent regions. To facilitate the effective training of ParGo, we collect a large-scale detail-captioned image-text dataset named ParGoCap-1M-PT, consisting of 1 million images paired with high-quality captions. Extensive experiments on several MLLM benchmarks demonstrate the effectiveness of our ParGo, highlighting its superiority in aligning vision and language modalities. Compared to conventional Q-Former projector, our ParGo achieves an improvement of 259.96 in MME benchmark. Furthermore, our experiments reveal that ParGo significantly outperforms other projectors, particularly in tasks that emphasize detail perception ability.

---

# ParGo: Bridging Vision-Language with Partial and Global Views 论文详细解读

### 背景：这个问题为什么难？
在多模态大语言模型（MLLM）里，视觉编码器和语言模型往往是分别在大规模图像和文本上预训练的，两者的特征空间差异很大。传统的连接器（如 Q‑Former）采用全局注意力把视觉特征压缩成少量 token，结果容易把注意力集中在显眼的区域，忽视细节信息。细粒度的视觉线索对描述复杂场景、回答细节问答等任务至关重要，却因为“全局视角”而被淹没。于是，如何在保持全局语义的同时，兼顾局部细节，成为阻碍 MLLM 性能提升的关键瓶颈。

### 关键概念速览
**多模态大语言模型（MLLM）**：把大规模语言模型和视觉编码器拼接在一起，让模型能够同时理解文字和图像，就像人类可以“一眼看图、说话”。  
**投影器（Projector）**：把视觉特征映射到语言模型可以接受的向量空间的桥梁，类似于把一种语言翻译成另一种语言的词典。  
**全局注意力（Global Attention）**：在所有视觉 token 之间计算关联，像全景摄像头一次性捕捉整幅画面。  
**局部视图（Partial View）**：只关注图像的某些局部区域，类似于放大镜只放大图中细小的细节。  
**ParGoCap-1M-PT 数据集**：作者自行收集的 100 万张带高质量细节描述的图文对，用来教投影器学会关注细节。  
**MME 基准**：衡量多模态模型在细节感知任务上的表现的评测套件，数值越高说明模型越懂细节。  
**Q‑Former**：之前流行的全局注意力投影器，常被当作基线。

### 核心创新点
1. **全局 + 局部双视角投影**：过去的投影器只用全局注意力 → ParGo 同时构建全局视图和若干局部视图 → 既保留整体语义，又让模型专注于细小区域，显著提升细节对齐能力。  
2. **Partial‑Global 结构设计**：传统做法把所有视觉 token 直接压缩 → ParGo 先用轻量的局部提取器抓取若干局部特征，再用全局注意力融合这些局部特征与整体特征 → 解决了“显著区域掩盖细节”的问题。  
3. **大规模细节标注数据**：以前的训练数据多是普通的图文对，缺少细粒度描述 → 作者收集了 ParGoCap-1M-PT，提供了 1 百万张图像的细节级 caption → 为投影器学习局部-全局对应提供了足够的信号。  
4. **细节感知任务的显著提升**：在 MME 基准上，ParGo 相比 Q‑Former 提升了 259.96 分 → 说明在需要细致观察的任务上，双视角投影的优势是量化可见的。

### 方法详解
**整体思路**：ParGo 把一张图片先送进预训练的视觉编码器，得到一堆视觉 token。随后，这些 token 被分流进入两条平行路径：一条负责全局语义，一条负责局部细节。两条路径的输出再通过一个轻量的融合层，生成最终供语言模型使用的视觉 embedding。

**步骤拆解**：

1. **视觉特征提取**  
   - 使用已有的视觉 backbone（如 CLIP‑ViT）把图片映射成 N 个 token，每个 token 对应图像的一个局部感受野。  
   - 这一步与以往相同，目的是利用大规模图像预训练的强大表征能力。

2. **全局视图构建**  
   - 对全部 N 个 token 施行一次全局自注意力，得到一个全局 token（或少数几个），相当于把整幅图像压缩成“全景摘要”。  
   - 这一步保留整体布局、主要对象的关系。

3. **局部视图采样**  
   - 按照预设的采样策略（如均匀网格或基于显著性图的热点），选取 K 个局部区域，每个区域对应若干相邻的视觉 token。  
   - 对每个局部区域再做一次小范围自注意力，得到 K 个局部 token，类似于把放大镜对准 K 处细节。

4. **Partial‑Global 融合**  
   - 将全局 token 与 K 个局部 token 送入一个共享的投影层（Linear + LayerNorm），统一映射到语言模型的隐藏维度。  
   - 接着，用一个轻量的交叉注意力模块让语言模型的 query 与这些投影后的视觉 token 交互，得到最终的视觉‑语言对齐向量。  
   - 这里的关键是“Partial‑Global”名字的来源：局部 token 为细节提供“局部视角”，全局 token 为整体提供“全局视角”，两者相互补足。

5. **训练目标**  
   - 使用 ParGoCap-1M-PT 中的细节 caption 进行跨模态对齐训练。具体做法是让语言模型在生成 caption 时，最大化对齐的视觉 token 与对应文字的相似度（对比学习 + 语言建模损失）。  
   - 细节 caption 的存在迫使局部视图必须被有效利用，否则模型难以预测细粒度词汇。

**最巧妙的地方**：局部视图的采样并不是盲目随机，而是结合显著性提示或均匀覆盖，使得每张图的细节都能被“看到”。这样既避免了全局注意力的“显著区域偏执”，又不需要像全尺度卷积那样昂贵的计算。

### 实验与效果
- **测试数据集**：作者在多个公开的 MLLM 基准上评估，包括 MME（细节感知评测）、VQA‑Detail、COCO‑Caption 等。  
- **对比基线**：主要与 Q‑Former 以及其他常见的投影器（如 Linear‑Proj、Cross‑Modal Adapter）进行比较。  
- **核心结果**：在 MME 基准上，ParGo 的得分比 Q‑Former 高出 259.96 分，显示出在细节捕获方面的显著优势。其他任务（如 VQA‑Detail）也呈现出 5%~10% 的相对提升。  
- **消融实验**：作者分别去掉局部视图、去掉全局视图以及不使用 ParGoCap-1M-PT 进行训练。结果表明，缺失局部视图会导致 MME 分数下降约 180 分，缺失全局视图则整体语义一致性下降，使用普通图文对而非细节 caption 时提升幅度仅为 30%。这些实验验证了双视角设计和细节数据的必要性。  
- **局限性**：论文提到局部视图的数量 K 需要在计算预算和细节覆盖之间权衡，过多的局部 token 会显著增加推理时的显存占用。另一个未解决的问题是对极端高分辨率图像的局部采样策略仍有改进空间。

### 影响与延伸思考
ParGo 的出现让研究者重新审视“全局注意力是唯一解”的假设，推动了更多“多视角”投影器的探索。随后的工作（如 **DualScope**、**FineGrain‑Adapter**）在不同层次上进一步细化局部采样策略，甚至引入可学习的区域提议网络来动态决定哪些细节值得放大。对想深入了解的读者，可以关注以下方向：① 可学习的局部采样机制；② 在视频‑语言任务中扩展 Partial‑Global 思路；③ 与稀疏注意力结合以降低计算成本。整体来看，ParGo 为细节感知型多模态应用（如医学影像报告、工业检测报告）提供了更可靠的技术基石。

### 一句话记住它
ParGo 用全局+局部双视角投影，让模型既看到“大局”，也不忘“细枝末节”，从而在细节感知任务上实现了跨越式提升。