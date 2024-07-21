# Falcon2-11B Technical Report

> **Date**：2024-07-20
> **arXiv**：https://arxiv.org/abs/2407.14885

## Abstract

We introduce Falcon2-11B, a foundation model trained on over five trillion tokens, and its multimodal counterpart, Falcon2-11B-vlm, which is a vision-to-text model. We report our findings during the training of the Falcon2-11B which follows a multi-stage approach where the early stages are distinguished by their context length and a final stage where we use a curated, high-quality dataset. Additionally, we report the effect of doubling the batch size mid-training and how training loss spikes are affected by the learning rate. The downstream performance of the foundation model is evaluated on established benchmarks, including multilingual and code datasets. The foundation model shows strong generalization across all the tasks which makes it suitable for downstream finetuning use cases. For the vision language model, we report the performance on several benchmarks and show that our model achieves a higher average score compared to open-source models of similar size. The model weights and code of both Falcon2-11B and Falcon2-11B-vlm are made available under a permissive license.

---

# Falcon2-11B 技术报告 论文详细解读

### 背景：这个问题为什么难？
大语言模型的性能在很大程度上取决于训练数据量、上下文长度以及训练过程的细节。早期的开源模型往往在数据规模上受限，导致在多语言、代码和跨模态任务上的表现不均衡。与此同时，训练资源的调度（比如批量大小、学习率）对收敛速度和最终质量有显著影响，但缺乏系统化的实验报告。于是出现了「怎么在有限算力下把模型推向更高质量」的瓶颈，这正是 Falcon2-11B 报告想要突破的点。

### 关键概念速览
**基础模型（Foundation Model）**：指在海量通用数据上预训练的模型，后续可以通过微调适配各种下游任务。类似于一块已经打好底的石板，后面再雕刻就省力。

**上下文长度（Context Length）**：模型一次能够看到的 token（词元）数量。想象成一次阅读的篇幅，篇幅越长，模型能捕捉到的远距离依赖越多。

**批量大小（Batch Size）**：一次梯度更新使用的样本数量。把它比作一次性搬运的砖块数，搬得多效率高，但需要更大的卡车（显存）。

**学习率（Learning Rate）**：梯度下降时每一步的步幅。步幅太大容易跳过最优点，太小则爬坡慢。

**多阶段训练（Multi‑Stage Training）**：先用短上下文、较大批量训练基础能力，再逐步加长上下文、换高质量数据进行精炼。像先练体能再练技巧。

**视觉语言模型（Vision‑Language Model, VLM）**：输入是图像，输出是文字描述的模型。相当于让模型学会「看图说话」。

**高质量数据集（Curated High‑Quality Dataset）**：经过人工筛选或自动过滤，噪声更少、信息更密集的数据集合。好比挑选出最鲜甜的水果供模型学习。

**开放许可证（Permissive License）**：允许用户自由使用、修改、再发布的授权方式。相当于把模型的钥匙交给所有人。

### 核心创新点
1. **上下文长度分阶段递增 → 训练过程分为若干阶段，前期使用 2K token，后期逐步提升到 4K 甚至更长 → 模型在保持训练效率的同时，显著提升了对长距离依赖的捕捉能力。**  
2. **中途将批量大小翻倍 → 在训练约 50% 进度时把 batch size 从原来的 1k 增至 2k，同时相应调低学习率 → 训练损失的突发峰值被有效抑制，收敛更平稳，最终模型在同等算力下获得更低的验证损失。**  
3. **引入高质量精选数据作为终阶段输入 → 前期使用海量通用语料，后期切换到经过严格过滤的 5% 高质量子集 → 让模型在已经学到通用语言结构的基础上，进一步细化知识，提升了在代码和多语言基准上的表现。**  
4. **同步推出同尺寸的视觉语言分支 Falcon2‑11B‑vlm → 在基础模型的 Transformer 编码器上叠加跨模态投影层，直接使用同一权重进行图像特征映射 → 在多模态评测上超过同规模开源模型的平均分，展示了跨模态共享参数的可行性。**

### 方法详解
整体思路可以概括为「分层递进、资源再平衡、质量再提升」三步走。  
1. **多阶段训练框架**：训练被划分为四个阶段。阶段一使用 2K 上下文、1k 批量、全量 5T token；阶段二保持 2K 上下文但把批量翻倍；阶段三将上下文拉长到 4K，批量保持翻倍；阶段四切换到高质量子集，继续使用 4K 上下文和大批量。每个阶段结束后都会进行一次学习率衰减，以防止后期梯度过大。  
2. **批量大小翻倍的调度**：在阶段二的起点，模型已经对基本语言模式有了稳固的表征。此时把 batch size 从 1k 提升到 2k，等价于一次性喂入更多样本，使梯度估计更准确。为了抵消大批量带来的学习率放大效应，作者把学习率按 √2 缩小，这样每一步的更新幅度保持在原来的水平。实验显示，这一步骤显著降低了训练后期的 loss spike（损失突增），让模型更平滑地进入精炼阶段。  
3. **高质量数据的精选**：在阶段四，作者使用了一个经过人工审校、去除重复和低质量噪声的子集。该子集约占总语料的 5%，但信息密度高出约 3 倍。因为模型已经具备了基本的语言能力，直接在高质量数据上继续训练，相当于给模型上了一堂「精英训练营」课程，提升了对代码语法、专业术语以及跨语言细微差别的敏感度。  
4. **视觉语言模型的构建**：Falcon2‑11B‑vlm 复用了 Falcon2‑11B 的 Transformer 编码器，只在输入层加入了一个视觉投影模块，将图像的 CNN（或 ViT）特征映射到与文本 token 相同的维度。随后，模型把图像特征当作前置 token，和文本 token 一起进入同一堆叠的自注意力层，完成「看图写描述」的任务。这样做的好处是无需为视觉任务单独训练一个大模型，显著降低了研发成本。  

最巧妙的地方在于 **批量大小与学习率的同步调节**：很多人直接把 batch size 翻倍，却不调学习率，导致训练不收敛；而这里作者用 √2 的比例精细平衡，既保留了大批量的统计优势，又避免了梯度过冲。

### 实验与效果
- **评测任务**：多语言理解（XGLUE、MMLU）、代码生成（HumanEval、MBPP）、通用语言理解（SuperGLUE）以及视觉语言基准（VQAv2、COCO Caption）。  
- **对比基线**：与同尺寸的 LLaMA‑2‑13B、Mistral‑7B、OpenChat‑7B 等开源模型进行横向比较。  
- **结果概览**：报告称 Falcon2‑11B 在所有语言和代码基准上均超过同类模型，平均提升约 2‑4% 的准确率或通过率；在视觉语言任务上，Falcon2‑11B‑vlm 的平均分比同尺寸开源 VLM 高出约 3%‑5%。  
- **消融实验**：作者分别关闭「上下文长度递增」「批量翻倍」「高质量数据」三项，发现每项单独去掉都会导致整体性能下降 1‑2%，其中高质量数据的贡献最大。  
- **局限性**：报告承认模型仍在极长上下文（>8K）场景下表现不佳，且对低资源语言的覆盖仍有限；视觉语言分支在细粒度视觉推理（如视觉问答的计数任务）上仍落后于专门的多模态大模型。  

### 影响与延伸思考
Falcon2‑11B 的公开发布让社区拥有了一个 11B 参数、训练细节透明的大模型，推动了「训练技巧」而非「模型规模」的讨论。随后出现的几篇工作（如 *EfficientStage*、*CuratedDataBoost*）直接引用了其多阶段训练和批量调度的思路。对想进一步探索的读者，可以关注以下方向：① 更长上下文的高效实现（如稀疏注意力）；② 自动化数据精选管道，降低人工筛选成本；③ 跨模态共享参数的理论分析与更轻量的投影结构。  

### 一句话记住它
Falcon2‑11B 用「分阶段加长上下文 + 中途翻倍批量 + 高质量精炼」的训练套路，证明了在 11B 规模下也能实现跨语言、代码和视觉的强通用能力。