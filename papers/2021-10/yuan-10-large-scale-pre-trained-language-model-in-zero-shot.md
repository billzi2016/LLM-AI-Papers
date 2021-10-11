# Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and   Few-Shot Learning

> **Date**：2021-10-10
> **arXiv**：https://arxiv.org/abs/2110.04725

## Abstract

Recent work like GPT-3 has demonstrated excellent performance of Zero-Shot and Few-Shot learning on many natural language processing (NLP) tasks by scaling up model size, dataset size and the amount of computation. However, training a model like GPT-3 requires huge amount of computational resources which makes it challengeable to researchers. In this work, we propose a method that incorporates large-scale distributed training performance into model architecture design. With this method, Yuan 1.0, the current largest singleton language model with 245B parameters, achieves excellent performance on thousands GPUs during training, and the state-of-the-art results on NLP tasks. A data processing method is designed to efficiently filter massive amount of raw data. The current largest high-quality Chinese corpus with 5TB high quality texts is built based on this method. In addition, a calibration and label expansion method is proposed to improve the Zero-Shot and Few-Shot performance, and steady improvement is observed on the accuracy of various tasks. Yuan 1.0 presents strong capacity of natural language generation, and the generated articles are difficult to distinguish from the human-written ones.

---

# Yuan 1.0：大规模预训练语言模型在零样本与少样本学习中的应用 论文详细解读

### 背景：这个问题为什么难？

在零样本（Zero‑Shot）和少样本（Few‑Shot）任务里，模型必须在几乎没有标注数据的情况下直接完成下游任务。早期的语言模型规模有限，参数只有几亿到十几亿，导致它们的通用知识不足，面对新任务时往往只能靠微调才能取得可用的效果。GPT‑3 通过把参数提升到上百亿甚至千亿级，展示了规模本身可以带来强大的跨任务迁移能力，但训练这样的大模型需要上万块 GPU‑day 的算力，普通研究团队难以负担。于是，如何在保持极大模型容量的同时，让训练成本更可控、数据质量更高，成为了当时的瓶颈。

### 关键概念速览
- **零样本学习（Zero‑Shot Learning）**：模型在没有看到任何该任务的标注样本的情况下直接给出答案，类似于人类凭直觉完成陌生任务。  
- **少样本学习（Few‑Shot Learning）**：模型只看到极少量（通常 < 10）示例后完成任务，像是给模型几道例题后让它自行解答。  
- **单例模型（Singleton Model）**：整个模型是一个完整的网络，没有像 Mixture‑of‑Experts 那样把参数切分成多个专家子网。所有参数在每一次前向传播中都被使用。  
- **分布式训练性能（Distributed Training Efficiency）**：指在上千甚至上万块 GPU 上并行训练时，通信、负载均衡和内存利用的整体效率。  
- **高质量语料过滤**：从海量原始文本中挑选出噪声少、语言规范、信息丰富的片段，类似于在矿山里筛选出金子。  
- **校准（Calibration）**：对模型输出的概率进行后处理，使其更符合真实分布，避免模型对某些答案过于自信。  
- **标签扩展（Label Expansion）**：在零样本/少样本提示中加入同义词、别名或上下文描述，让模型更容易捕捉任务意图。  

### 核心创新点
1. **把分布式训练效率嵌入模型架构**  
   之前的超大模型往往在架构上只追求性能，忽视了在数千块 GPU 上的通信瓶颈。Yuan 1.0 在设计层面采用了更深的层级并行和张量切分策略，使得每块 GPU 只需要处理模型参数的一个小子块，同时通过优化 All‑Reduce 通信图降低网络开销。结果是同等算力下训练时间显著缩短，245 B 参数模型能够在几千块 GPU 上顺利收敛。  

2. **大规模高质量中文语料构建**  
   直接使用爬取的原始网页文本会带来大量噪声（广告、乱码、重复等），导致模型学习到错误的语言规律。论文提出了一套自动化过滤流水线：先用轻量级语言模型做粗筛，再用规则过滤去除低质量段落，最后用高置信度的语义相似度模型剔除重复内容。这样得到的 5 TB 中文语料被称为当时最大的高质量中文语料库，为模型的语言理解和生成奠定了坚实基础。  

3. **校准+标签扩展提升零/少样本表现**  
   在零样本/少样本提示中，直接使用任务名称往往会导致模型输出偏差。作者先对模型的 logits 进行温度校准，使得概率分布更平滑；随后在提示里加入任务的同义词、常见别名以及简短的任务描述（标签扩展），相当于给模型提供了多角度的任务解释。实验显示，这一组合在多项基准上平均提升了 2‑4% 的准确率。  

### 方法详解
整体思路可以拆成三大步骤：**（1）高效分布式训练架构设计 → (2) 大规模语料过滤与构建 → (3) 零/少样本提示校准与标签扩展**。下面逐层展开。

1. **分布式训练架构**  
   - **层级并行**：模型的前向和反向计算被划分为两层并行。第一层是 **张量切分**（Tensor Parallelism），把每层的权重矩阵沿列或行切成若干块，分别放在不同 GPU 上计算局部矩阵乘法。第二层是 **流水线并行**（Pipeline Parallelism），把模型的层序划分为若干阶段，每个阶段由一组 GPU 负责。这样既降低了单卡显存需求，又让计算保持高占用率。  
   - **通信优化**：传统 All‑Reduce 在数千卡时会出现网络拥塞。Yuan 1.0 采用 **环形拓扑**（Ring）结合 **梯度压缩**（Gradient Compression），只在必要时传输梯度的低位信息，显著削减带宽占用。  
   - **负载均衡**：通过预先统计每层参数量和计算量，动态调整每个 GPU 所承担的张量切分比例，避免出现“卡点”导致的训练停滞。  

2. **语料过滤流水线**  
   - **粗筛阶段**：使用一个 2 B 参数的轻量模型对原始文本进行语言检测和基本质量评分（如字符比例、重复率）。低于阈值的直接丢弃。  
   - **规则过滤**：对剩余文本执行正则表达式检查，剔除包含广告链接、HTML 标签、乱码等的行。  
   - **语义去重**：将每段文本转成向量，用余弦相似度计算相似度，若相似度超过 0.95 则认为是重复内容，只保留一份。  
   - **质量再评估**：最后用一个经过微调的中文 BERT 对每段文本打分，保留前 80% 的高分段落。整个流水线在分布式集群上并行运行，日均处理数十 TB 原始数据。  

3. **Zero/Few‑Shot 提示校准与标签扩展**  
   - **温度校准**：在推理阶段，对模型输出的 logits 除以一个温度系数 τ（经验值 0.7），使得概率分布更平滑，降低极端置信度。  
   - **标签扩展**：对每个任务，构造一个包含任务名称、同义词、常见别名以及简短描述的提示模板。例如情感分类任务的提示从 “情感分析：” 扩展为 “情感分析（正面/负面）：”。这些额外的文字帮助模型在没有显式微调的情况下更好地对齐任务语义。  
   - **多提示融合**：对同一输入生成多个不同标签扩展的提示，分别得到预测后再做多数投票或概率加权，进一步提升鲁棒性。  

**最巧妙的点**在于把硬件层面的分布式优化直接写进模型设计，而不是事后再去调参。这样模型规模可以随算力线性扩展，而不出现“训练不收敛”或“显存爆炸”的常见问题。

### 实验与效果
- **评测任务**：在中文自然语言处理的主流基准上进行零样本和少样本评估，包括机器阅读理解（CMRC），情感分析（ChnSentiCorp），文本分类（THUCNews），以及生成任务（中文新闻写作）。  
- **基线对比**：与同规模的 GPT‑3（175 B）以及国内的 ERNIE‑3.0（260 B，MoE 结构）进行比较。论文报告在大多数任务上，Yuan 1.0 的零样本准确率比 GPT‑3 提高约 3%~5%，在少样本（k=5）设置下提升约 2%~4%。在生成任务上，人类评审的可辨识率下降到 45%（即 55% 的生成文本被误认为是人写的），接近人类水平。  
- **消融实验**：分别去掉（1）张量切分优化、（2）语料过滤、（3）校准+标签扩展。结果显示，缺少语料过滤会导致整体性能下降约 6%；去掉校准+标签扩展在零样本任务上下降约 2.5%；不使用张量切分导致训练时间翻倍，实际可训练规模受限。  
- **局限性**：论文承认模型仍然是单例结构，参数量巨大导致推理成本高，部署到边缘设备仍不可行；此外，虽然中文语料质量提升显著，但对多语言或低资源语言的适应性未作系统评估。  

### 影响与延伸思考
Yuan 1.0 的发布在 2021 年标志着“单例大模型”在中文 NLP 领域的里程碑，直接推动了后续的 **Yuan 2.0**、**ChatGLM** 等模型在架构上更注重分布式效率。它的语料过滤流水线被多家企业采纳，成为构建大规模中文语料库的标准做法。校准+标签扩展的思路也被后来的 **Prompt‑Tuning** 研究所引用，形成了“提示工程”与模型本身协同提升的趋势。想进一步深入，可以关注 **混合并行（Hybrid Parallelism）**、**高效稀疏激活（Sparse Activation）** 以及 **多语言大模型的跨语言迁移** 等方向，这些都是在 Yuan 1.0 基础上继续突破的热点。  

### 一句话记住它
把分布式训练效率写进模型结构，让 245 B 参数的中文大模型在千卡级算力下也能训练，并通过高质量语料和提示校准实现强零/少样本能力。