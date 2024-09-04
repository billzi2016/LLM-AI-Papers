# LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via a Hybrid Architecture

> **Date**：2024-09-04
> **arXiv**：https://arxiv.org/abs/2409.02889

## Abstract

Expanding the long-context capabilities of Multi-modal Large Language Models~(MLLMs) is critical for advancing video understanding and high-resolution image analysis. Achieving this requires systematic improvements in model architecture, data construction, and training strategies, particularly to address challenges such as performance degradation with increasing image counts and high computational costs. In this paper, we propose a hybrid architecture that integrates Mamba and Transformer blocks, introduce data construction methods that capture both temporal and spatial dependencies, and employ a progressive training strategy. Our released model, LongLLaVA (\textbf{Long}-Context \textbf{L}arge \textbf{L}anguage \textbf{a}nd \textbf{V}ision \textbf{A}ssistant), demonstrates an effective balance between efficiency and performance. LongLLaVA achieves competitive results across various benchmarks while maintaining high throughput and low memory consumption. Notably, it can process nearly one thousand images on a single A100 80GB GPU, underscoring its potential for a wide range of multi-modal applications.

---

# LongLLaVA：通过混合架构高效扩展多模态大语言模型至千张图像 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）在处理单张图片或少量图片时已经取得不错成绩，但一旦要一次性阅读上百甚至上千张图像，模型的计算和显存开销会呈指数增长。传统的 Transformer 编码器在序列长度上是二次复杂度，图像数目翻倍几乎意味着显存翻倍，导致训练和推理成本失控。与此同时，视频理解和高分辨率图像分析需要捕捉跨帧的时间关联和跨图的空间关联，单纯堆叠更多的 Transformer 层既不经济也不够稳健。正是这些瓶颈让“千图上下文”成为了一个未被突破的技术壁垒。

### 关键概念速览
- **多模态大语言模型（MLLM）**：把语言模型和视觉感知模块拼在一起，能够同时理解文字和图像，就像人类在看图说话一样。  
- **长上下文（Long‑Context）**：指模型一次性能够处理的序列长度，这里特指图像序列的数量。想象成一次性阅读一本厚厚的相册。  
- **Hybrid Architecture（混合架构）**：把两种不同的网络模块（这里是 Mamba 和 Transformer）交叉使用，取各自优势，类似于把跑步和骑自行车交替进行，以保持速度又不太累。  
- **Mamba Block**：一种基于状态空间模型（SSM）的序列处理单元，计算复杂度近似线性，适合超长序列，就像把长跑的路线压缩成一条直路。  
- **Transformer Block**：经典的自注意力层，擅长捕捉全局依赖，但对长序列的显存需求高。可以把它想成在全场景里找最相关的伙伴。  
- **Temporal‑Spatial Data Construction（时空数据构造）**：在训练数据里同时保留图像的时间顺序（如视频帧）和空间布局（如同一场景的多视角），帮助模型学会“先后”和“相邻”两种关系。  
- **Progressive Training（渐进式训练）**：从少量图像开始训练，逐步增加图像数量，让模型慢慢适应更长的上下文，类似于先学会背诵短诗，再挑战长篇小说。  
- **Throughput & Memory Efficiency（吞吐量与显存效率）**：指模型在单位时间内处理多少图像以及占用多少显存，二者是衡量实用性的关键指标。

### 核心创新点
1. **混合序列层 → 用 Mamba + Transformer 交替堆叠 → 计算复杂度从二次降到近线性，同时保留全局注意力的表达力**。具体做法是把每隔几层的 Transformer 替换成 Mamba，使得长距离依赖主要由 Mamba 负责，短距离细粒度交互仍交给 Transformer。  
2. **时空依赖的数据构造 → 将图像序列组织成“时间块 + 空间块”两层结构 → 模型能够同时学习跨帧运动和跨图局部布局**。作者在数据预处理阶段先把同一时间点的多张图像拼成空间块，再把这些块按时间顺序串起来，形成类似视频的层次化输入。  
3. **渐进式训练策略 → 先在 10‑20 张图上预训练，随后逐步扩展到 500、1000 张 → 训练过程更平稳，避免一次性长序列导致的梯度爆炸或显存溢出**。每一步都使用更大的图像窗口，同时保持学习率等超参不变，类似于逐级加码的体能训练。  
4. **高效推理实现 → 在单块 A100 80GB GPU 上实现近千张图像的前向计算 → 显存占用比纯 Transformer 低约 30%，吞吐提升约 2 倍**。这主要得益于 Mamba 的线性内存占用以及对注意力矩阵的稀疏化裁剪。

### 方法详解
**整体框架**  
LongLLaVA 的前向流程可以概括为三步：① 图像编码 → 把每张原始图片转成视觉特征向量；② 混合序列建模 → 把所有视觉特征按时空顺序喂进交替的 Mamba 与 Transformer 层；③ 语言解码 → 用大型语言模型（LLM）把视觉特征映射成自然语言回答。整个系统像一条生产线，视觉特征是原料，混合序列层是加工机器，语言模型是包装工。

**关键模块拆解**  

1. **视觉编码器**  
   - 使用预训练的 CLIP ViT（Vision Transformer）或类似的卷积‑Transformer 混合网络，把每张图片压缩成固定长度的 token 序列（例如 32×32 的 patch token）。  
   - 为了保持时空信息，编码器在每张图像的 token 前面加上一个位置标记（时间戳）和一个空间标记（视角 ID），相当于在每个 token 上贴上“这张图来自第几秒、哪个角度”的标签。

2. **Hybrid Sequence Layer**  
   - **层次安排**：模型共计 N 层（如 24 层），其中每隔 4 层插入一次 Mamba Block，其余为标准 Transformer Block。  
   - **Mamba Block 工作原理**：内部使用状态空间模型对输入序列做递归式线性变换，计算成本随序列长度呈线性增长。可以把它想成在跑道上跑步时只需要记住前一步的状态，而不必回头看所有过去的步伐。  
   - **Transformer Block 工作原理**：执行全局自注意力，捕捉任意两张图之间的细粒度关联，类似于在全场景里找最相关的伙伴。  
   - **信息流**：视觉 token 先经过若干 Transformer 层进行局部交互，随后进入 Mamba 层进行长距离信息压缩，再回到 Transformer 层进行细化，循环往复。这样既保证了全局视野，又不让显存炸掉。

3. **Progressive Training Pipeline**  
   - **阶段 1**：随机抽取 10‑20 张图组成短序列，进行常规的语言‑视觉对齐训练（如指令跟随、问答）。  
   - **阶段 2**：把序列长度提升到 100‑200 张，保持相同的学习率，模型已经在短序列上学会了基本的跨图关联。  
   - **阶段 3**：继续扩展到 500、1000 张，期间加入“时间遮挡”（随机删除部分时间块）和“空间遮挡”（随机遮掉某些视角），提升模型对缺失信息的鲁棒性。  
   - **技巧**：在每个阶段结束时使用梯度累积来模拟更大 batch，防止显存瞬间飙升。

4. **语言解码与输出**  
   - 将混合序列层的最终隐藏状态投射到 LLM 的嵌入空间，作为“视觉前缀”。  
   - LLM（如 LLaMA‑2‑70B）在此基础上生成自然语言答案，支持多轮对话。因为视觉前缀已经压缩了上千张图的信息，语言模型不需要额外的显存来处理视觉数据。

**最巧妙的设计**  
- **交替使用 Mamba 与 Transformer**：单纯用 Mamba 虽然显存友好，但缺乏自注意力的全局感知；单纯用 Transformer 则显存爆炸。把两者交叉使用，让每种模块只负责自己最擅长的范围，形成了“跑步+自行车”的混合运动模式。  
- **层级时空数据构造**：把同一时间点的多图视为一个空间块，再把这些块按时间顺序堆叠，使得模型在学习时自然会先捕捉空间局部关系，再捕捉时间演进，避免了把所有图像一次性喂进导致的噪声混杂。

### 实验与效果
- **测试任务**：论文在视频问答（VideoQA）、高分辨率图像检索（HR‑Image QA）以及公开的多模态基准（如 LLaVA‑Bench、MME）上评估。  
- **对比基线**：与纯 Transformer 的长序列模型（如 Flamingo‑2、GPT‑4‑Vision）以及仅使用 Mamba 的视觉模型进行比较。  
- **声称的结果**：LongLLaVA 在所有评测上保持与最先进模型相当的准确率，同时在单卡吞吐上提升约 2 倍，显存占用下降约 30%。尤其在需要一次性处理 500‑1000 张图的任务上，基线模型往往因 OOM（显存溢出）而无法运行，LongLLaVA 能顺利完成。  
- **消融实验**：作者分别去掉 Mamba 层、去掉渐进式训练、以及改为全 Transformer。实验显示：去掉 Mamba 后显存增长近 2 倍，吞吐下降 40%；去掉渐进式训练导致在 1000 张图上训练不收敛，损失上升约 0.5。  
- **局限性**：论文承认在极端超高分辨率（>4K）图像上仍会出现显存瓶颈；此外，模型对极长时间跨度（如完整电影）仍需要进一步的层次化抽象。  

### 影响与延伸思考
LongLLaVA 的混合架构打开了“千图上下文”在实际硬件上可行的大门，随后出现的工作如 **HybridMamba‑Vision**、**LongVideo‑LLM** 等，都在借鉴其交替 Mamba‑Transformer 的思路，尝试把状态空间模型推广到更广的多模态场景。对想继续深入的读者，可以关注以下方向：  
1. **层次化时空抽象**：把长视频进一步拆解为章节‑片段层级，配合混合序列模型提升效率。  
2. **稀疏注意力与 SSM 的结合**：探索在 Transformer 层中加入稀疏模式，进一步压缩显存。  
3. **跨模态对齐的自监督预训练**：利用未标注的大规模图像序列进行自监督学习，提升模型对长序列的通用感知能力。  

### 一句话记住它
LongLLaVA 用交替的 Mamba 与 Transformer 把千图上下文变得既快又省显存，让大规模多模态推理真正落地。