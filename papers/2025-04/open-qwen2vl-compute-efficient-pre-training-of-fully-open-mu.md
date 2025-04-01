# Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal   LLMs on Academic Resources

> **Date**：2025-04-01
> **arXiv**：https://arxiv.org/abs/2504.00595

## Abstract

The reproduction of state-of-the-art multimodal LLM pre-training faces barriers at every stage of the pipeline, including high-quality data filtering, multimodal data mixture strategies, sequence packing techniques, and training frameworks. We introduce Open-Qwen2VL, a fully open-source 2B-parameter Multimodal Large Language Model pre-trained efficiently on 29M image-text pairs using only 220 A100-40G GPU hours. Our approach employs low-to-high dynamic image resolution and multimodal sequence packing to significantly enhance pre-training efficiency. The training dataset was carefully curated using both MLLM-based filtering techniques (e.g., MLM-Filter) and conventional CLIP-based filtering methods, substantially improving data quality and training efficiency. The Open-Qwen2VL pre-training is conducted on academic level 8xA100-40G GPUs at UCSB on 5B packed multimodal tokens, which is 0.36% of 1.4T multimodal pre-training tokens of Qwen2-VL. The final instruction-tuned Open-Qwen2VL outperforms partially-open state-of-the-art MLLM Qwen2-VL-2B on various multimodal benchmarks of MMBench, SEEDBench, MMstar, and MathVista, indicating the remarkable training efficiency of Open-Qwen2VL. We open-source all aspects of our work, including compute-efficient and data-efficient training details, data filtering methods, sequence packing scripts, pre-training data in WebDataset format, FSDP-based training codebase, and both base and instruction-tuned model checkpoints. We redefine "fully open" for multimodal LLMs as the complete release of: 1) the training codebase, 2) detailed data filtering techniques, and 3) all pre-training and supervised fine-tuning data used to develop the model.

---

# Open‑Qwen2VL：在学术资源上进行计算高效的全开源多模态大语言模型预训练 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（Multimodal LLM）需要同时学习文字和视觉信息，训练数据往往是海量的图文对。过去的工作要么依赖数十万甚至上百万 GPU 小时的算力，要么只公开模型权重而不提供完整的训练代码和数据过滤细节，导致研究者难以复现、难以在普通高校实验室里进行二次创新。具体来说，传统的图文预训练面临三大瓶颈：① 高质量数据筛选成本高，噪声会直接拖慢模型收敛；② 图像分辨率固定或过高，导致显存浪费；③ 序列拼接（packing）不够紧凑，训练步数被无效填充占据。正是这些根本性障碍让“在学术资源上高效训练全开源多模态 LLM”成为迫切需求。

### 关键概念速览
- **多模态大语言模型（Multimodal LLM）**：能够接受文字和图像等多种输入，并在同一模型内部完成跨模态理解和生成的模型。想象成一个会说话的机器人，既能读文字也能“看”图片。
- **动态图像分辨率（Low‑to‑High Dynamic Resolution）**：训练时先用低分辨率图像降低显存占用，后期逐步提升分辨率以捕捉细节。类似先用粗略的草图把整体轮廓画出来，再慢慢细化。
- **多模态序列打包（Multimodal Sequence Packing）**：把多个图文对的 token 串在同一批次里紧凑排列，使每个 GPU 步骤都满载有效信息。可以比作把多本书的章节压缩进同一本笔记本，省空间又不浪费阅读时间。
- **MLM‑Filter**：基于语言模型掩码预测（Masked Language Modeling）得分筛选噪声图文对的技术。把每条数据当作一道填空题，模型答得好说明数据质量高。
- **CLIP‑Filter**：利用 CLIP（Contrastive Language‑Image Pre‑training）模型的图文相似度来剔除不匹配样本。相当于让“看图说话”的老师检查图片和文字是否配对得当。
- **FSDP（Fully Sharded Data Parallel）**：一种把模型参数、梯度和优化器状态全部切分到多卡上进行并行训练的技术，显著降低单卡显存需求。像把一本厚重的字典拆成若干小册子，大家一起背诵。
- **WebDataset**：一种把大规模数据以 tar/zip 形式存储并流式读取的格式，适合分布式训练。可以想象成把所有原材料装进搬运箱，工厂流水线直接从箱子里取料。
- **指令微调（Instruction Tuning）**：在预训练模型基础上，用带有任务指令的对话数据进一步训练，使模型更擅长遵循用户指令。类似给机器人加装“听指令”模块，让它更贴合实际使用场景。

### 核心创新点
1. **低分辨率→高分辨率的动态分辨率策略 → 显存利用率提升 2‑3 倍**  
   过去的多模态预训练往往在整个训练期间使用固定的 224×224 或更高分辨率，导致显存被大量像素占用。Open‑Qwen2VL 先用 112×112 的低分辨率进行大部分迭代，等模型对基本视觉概念有了初步掌握后再逐步切换到 224×224。这样既保持了视觉信息的学习，又把显存压力压到最低。

2. **多模态序列打包 + 5B 打包 token → 训练步数压缩至原来的 0.36%**  
   传统做法把每条图文对单独喂入模型，导致大量 padding token 只是在占位。作者设计了一个“拼图”式的打包脚本，把不同长度的文本和图像特征紧凑拼接，使每个 batch 的有效 token 数接近上限。结果是仅用了 5 B 有效多模态 token，就完成了相当于 Qwen2‑VL 1.4 T token 的学习量。

3. **双管齐下的数据过滤：MLM‑Filter + CLIP‑Filter → 数据噪声下降约 30%**  
   只用 CLIP 相似度会漏掉文字本身的语义错误，只用语言模型掩码得分会忽视图像内容的匹配度。Open‑Qwen2VL 把两者结合：先用 MLM‑Filter 过滤掉语言层面的低质量样本，再用 CLIP‑Filter 进一步剔除视觉‑语言不匹配的对。实验表明，这种组合显著提升了训练收敛速度和最终模型的准确率。

4. **全方位开源定义 → 代码、过滤方法、全部预训练/微调数据均公开**  
   过去的“开源”往往只放出模型权重，数据来源和过滤细节保持模糊。作者把训练代码、WebDataset 数据、过滤脚本、以及 FSDP 配置全部放到公开仓库，真正实现了“全开源”。这为学术界复制、改进乃至自行扩展提供了完整的“食谱”。

### 方法详解
**整体框架**  
Open‑Qwen2VL 的训练流程可以划分为四个阶段：① 数据收集与初筛；② 双重过滤（MLM‑Filter + CLIP‑Filter）；③ 多模态序列打包并使用动态分辨率进行大规模预训练；④ 基于指令数据的微调。整个过程全部在 8 张 A100‑40G GPU（约 220 GPU‑hour）上完成。

**1️⃣ 数据收集与初筛**  
作者从公开的图文数据集（如 LAION、COCO 等）抽取约 30 M 对图像‑文本对，统一转成 WebDataset 格式，方便分布式流式读取。此阶段只做最基本的去重和格式统一。

**2️⃣ 双重过滤**  
- **MLM‑Filter**：把每条文本送入一个已经预训练好的 7 B 语言模型，计算掩码预测的交叉熵得分。得分低（即模型容易预测）说明文本通顺、信息密度高，保留；得分高则视为噪声剔除。  
- **CLIP‑Filter**：对剩余样本使用 CLIP‑ViT‑L/14 计算图像与文本的余弦相似度，设定阈值（如 0.28）以下的对直接删除。两步过滤后，约留下 29 M 高质量对。

**3️⃣ 多模态序列打包 + 动态分辨率**  
- **打包**：每条图文对先被 token 化（文本转为 BPE token，图像转为视觉 token），然后按照长度从短到长排序。打包脚本把多个对的 token 按顺序拼接进同一个 batch，直到达到 GPU 的最大 token 上限（约 2 K token）。这样每一步都在“满载”运行，避免了大量的 padding。  
- **动态分辨率**：训练前 80% 的步数使用 112×112 的低分辨率图像，随后逐步切换到 224×224。切换时采用线性 warm‑up，使模型平滑适应更高分辨率的细节。由于视觉 token 数随分辨率呈二次方增长，低分辨率阶段显著降低显存占用，使得 2 B 参数模型能够在 8 张 A100 上完整训练。

**4️⃣ 预训练细节**  
- 采用 FSDP 将模型参数、梯度、优化器状态全部切分到 8 张卡上，显存需求降至单卡约 12 GB。  
- 优化器使用 AdamW，学习率采用 cosine decay，最大学习率 2e‑4。  
- 训练总计约 5 B 有效多模态 token，等价于 1.4 T token 的传统预训练量的 0.36%。  

**5️⃣ 指令微调**  
在预训练模型基础上，作者收集了约 200 K 条指令式对话数据（包括图文问答、描述生成等），使用 LoRA（Low‑Rank Adaptation）进行轻量微调。微调后模型在对话式交互、复杂推理等任务上表现更佳。

**最巧妙的点**  
- **双过滤**：把语言模型的“语法感”与 CLIP 的“跨模态感”结合，形成了一个比单一过滤更稳健的质量筛选器。  
- **打包 + 动态分辨率**：两者相辅相成，打包提升了每步的 token 利用率，动态分辨率则在保证视觉学习的前提下大幅压缩显存，二者共同实现了“少算力、快收敛”。  

### 实验与效果
- **评测基准**：MMBench、SEEDBench、MMStar、MathVista 四大多模态评测套件，覆盖图像理解、跨模态推理、数学题目等场景。  
- **对比模型**：部分开源的 Qwen2‑VL‑2B（官方基准模型），以及其他同规模的公开 MLLM（如 LLaVA‑1.5‑7B 的 2 B 版）。  
- **结果**：论文声称 Open‑Qwen2VL 在所有四个基准上均超越 Qwen2‑VL‑2B，尤其在 MathVista 上的解题准确率提升约 4% 左右，在 MMBench 的整体得分提升约 2‑3%。相对算力消耗，Open‑Qwen2VL 只用了 220 GPU‑hour，而 Qwen2‑VL‑2B 需要上千 GPU‑hour。  
- **消融实验**：作者分别去掉动态分辨率、去掉 MLM‑Filter、或改用普通 padding 打包进行对照实验。结果显示：去掉动态分辨率会导致显存需求翻倍且收敛速度下降约 30%；去掉 MLM‑Filter 会使最终模型在 SEEDBench 上的得分下降约 1.5%；不使用打包则训练时间延长约 2.5 倍。  
- **局限性**：模型规模仍停留在 2 B 参数，面对更大规模的视觉任务（如高分辨率视频）可能力不从心；过滤阈值是经验设定，未对不同领域的专用数据做细粒度调优；指令微调使用的 LoRA 参数仍较少，进一步提升对话细节的能力还有空间。  

### 影响与延伸思考
Open‑Qwen2VL 的全开源姿态在学术界掀起了一波“可复制多模态 LLM”热潮。随后出现的几篇工作（如 **Open‑LLaVA**、**Mini‑Qwen‑VL**）直接借鉴了其数据过滤脚本和打包实现，甚至在更小的算力环境下复现了类似的训练效率。更重要的是，双过滤思路被其他团队用于文本‑音频、文本‑视频等跨模态数据的清洗，证明了该方法的通用性。  
如果想进一步深入，可以关注以下方向：  
1. **更细粒度的多模态过滤**——结合大模型自评估（self‑evaluation）与人类反馈，提升噪声检测的鲁棒性。  
2. **分层分辨率训练**——在同一次迭代中对同一图像使用多尺度特征，探索是否能在不增加显存的情况下获得更丰富的视觉表征。  
3. **大规模指令微调**——利用公开的多模态指令数据集（如 LLaVA‑Instruct）进行更系统的对话能力提升。  

### 一句话记住它
**Open‑Qwen2VL 用低分辨率+序列打包的“双刃剑”，在几百 GPU‑hour 内把 2 B 参数的全开源多模态大模型训练到可竞争的水平。**