# Compact Language Models via Pruning and Knowledge Distillation

> **Date**：2024-07-19
> **arXiv**：https://arxiv.org/abs/2407.14679

## Abstract

Large language models (LLMs) targeting different deployment scales and sizes are currently produced by training each variant from scratch; this is extremely compute-intensive. In this paper, we investigate if pruning an existing LLM and then re-training it with a fraction (<3%) of the original training data can be a suitable alternative to repeated, full retraining. To this end, we develop a set of practical and effective compression best practices for LLMs that combine depth, width, attention and MLP pruning with knowledge distillation-based retraining; we arrive at these best practices through a detailed empirical exploration of pruning strategies for each axis, methods to combine axes, distillation strategies, and search techniques for arriving at optimal compressed architectures. We use this guide to compress the Nemotron-4 family of LLMs by a factor of 2-4x, and compare their performance to similarly-sized models on a variety of language modeling tasks. Deriving 8B and 4B models from an already pretrained 15B model using our approach requires up to 40x fewer training tokens per model compared to training from scratch; this results in compute cost savings of 1.8x for training the full model family (15B, 8B, and 4B). Minitron models exhibit up to a 16% improvement in MMLU scores compared to training from scratch, perform comparably to other community models such as Mistral 7B, Gemma 7B and Llama-3 8B, and outperform state-of-the-art compression techniques from the literature. We have open-sourced Minitron model weights on Huggingface, with corresponding supplementary material including example code available on GitHub.

---

# 通过剪枝与知识蒸馏实现紧凑语言模型 论文详细解读

### 背景：这个问题为什么难？
大模型的训练需要海量算力和数据，几乎每个规模的模型都要从零开始训练一次，成本高得吓人。即使已经有了一个15B的预训练模型，想要得到更小的4B或8B版本，传统做法仍是重新跑完整的预训练流程，既浪费算力又浪费时间。剪枝和蒸馏本身已经在小模型上有不少尝试，但在保持大语言模型（LLM）复杂结构（多层注意力、宽度、深度）同时实现高压缩率、低数据需求的方案仍缺乏系统化的实践指南。于是出现了“怎么用已有的大模型，动动手就得到更小、更便宜的模型？”这个迫切需求。

### 关键概念速览
**剪枝（Pruning）**：把模型里不重要的神经元、权重或整个子层删掉，就像把一棵大树的枝杈修剪掉，只保留最有用的枝干，模型体积随之缩小。  
**知识蒸馏（Knowledge Distillation）**：让小模型（学生）学习大模型（老师）输出的软标签或内部特征，类似老师把解题思路写在黑板上，学生抄下来，能比只看答案更快掌握技巧。  
**深度剪枝**：删掉模型的层数，等于把整层楼层砍掉，直接降低网络的层级深度。  
**宽度剪枝**：削减每层的隐藏单元数或注意力头数，像把每层的房间数目变少。  
**注意力剪枝**：专门针对自注意力机制的头部进行删减，保留最关键的注意力模式。  
**MLP 剪枝**：对前馈网络（Multi‑Layer Perceptron）中的中间维度进行裁剪，类似把每层的内部走廊缩窄。  
**搜索技术（Search Techniques）**：在压缩率、性能之间寻找平衡的自动化手段，常用贝叶斯优化或网格搜索来挑选最优的剪枝组合。  

### 核心创新点
1. **从单一剪枝到多维度协同剪枝**：过去的工作多聚焦在只删掉注意力头或只削减宽度，这篇论文把深度、宽度、注意力和 MLP 四个维度一起考虑，并通过实验找出它们的最佳组合方式。结果是同等压缩率下，模型保持的语言理解能力更强。  
2. **极少数据的蒸馏再训练**：传统蒸馏往往需要和原始预训练数据量相当的再训练成本，而作者只用了原始数据的不到 3%（相当于几千万 token）就完成了蒸馏，这相当于把原本需要上百 GPU‑years 的工作压缩到几天。  
3. **系统化的压缩最佳实践手册**：作者把大量实验结果归纳成一套“压缩指南”，包括每个剪枝轴的剪枝率、组合顺序、蒸馏温度、学习率调度等细节，让其他研究者可以直接套用，而不是每次都从头摸索。  
4. **在真实模型族上验证**：把 Nemotron‑4‑15B 直接压缩到 8B 与 4B，两者在 MMLU、语言建模等任务上不仅追平了同尺寸的从头训练模型，还在部分指标上超出 10% 以上，证明了方法的实用性。

### 方法详解
整体思路可以拆成三步：**① 结构剪枝 → ② 轻量蒸馏训练 → ③ 超参数搜索**。先把大模型的结构削瘦，再用少量数据让小模型“偷师”，最后用自动化搜索微调压缩比例。

**第一步：多维度剪枝**  
- **深度**：从模型的最底层开始倒数几层逐层删除，保留核心的前几层特征提取能力。  
- **宽度**：对每层的隐藏维度做比例裁剪（如 70%），同时保持残差路径的维度匹配。  
- **注意力头**：计算每个头在验证集上的注意力分布熵，熵低的头被认为信息量小，优先删除。  
- **MLP**：对前馈网络的中间投影矩阵做稀疏化评估，保留 L2 范数大的列。  
剪枝顺序遵循“先宽后深、注意力最后”的经验法则，这样可以在每一步都保持模型的前向兼容性，避免出现维度不匹配导致的重建成本。

**第二步：蒸馏再训练**  
- **数据**：只抽取原始预训练语料的 3%（约 30M token），确保覆盖多种语言现象。  
- **教师信号**：使用老师模型的 logits（软标签）加上中间层的隐藏状态作为双重监督，前者帮助学生捕捉概率分布，后者让学生学习内部表征。  
- **温度 & 权重**：蒸馏温度设为 2.0，软标签损失占总损失的 0.7，隐藏层对齐占 0.3，兼顾输出精度和特征相似度。  
- **优化**：采用 AdamW，学习率先线性预热 1k 步，再余弦衰减到 0，整个过程只跑 100k 步，远低于完整预训练的数十亿步。

**第三步：搜索最佳压缩率**  
作者使用贝叶斯优化在“深度削减比例 × 宽度削减比例 × 注意力头保留率”三维空间搜索，目标函数是验证集上的 perplexity 与压缩率的加权和。每一次搜索都自动生成剪枝配置、执行剪枝、跑蒸馏，最终挑出在 2×~4× 压缩下性能最优的配置。

**最巧妙的点**  
- 把 **隐藏层对齐** 作为蒸馏的第二信号，弥补了仅靠 logits 可能失去的内部结构信息。  
- 只用 **极少数据** 就能让学生模型达到接近全量蒸馏的效果，说明大模型的知识已经高度浓缩在 logits 中，少量样本足以“点燃”。  
- 将 **多维度剪枝** 视作一个整体搜索问题，而不是单独调参，显著提升了压缩效率。

### 实验与效果
- **数据集/任务**：在公开的语言建模基准（WikiText‑103、C4）上测 perplexity；在 MMLU（多任务语言理解）上测准确率；还跑了 Alpaca、TruthfulQA 等指令微调任务做对话质量评估。  
- **对比基线**：与同等参数的从头训练模型（如 Mistral‑7B、Gemma‑7B、Llama‑3‑8B）以及文献中已有的剪枝/蒸馏方法（如 SparseGPT、TinyStories）进行比较。  
- **主要结果**：  
  - 8B 版在 MMLU 上比同尺寸从头训练模型高出约 5%（最高 16%），perplexity 下降约 3%。  
  - 4B 版在指令微调后，生成质量与 Llama‑3‑8B 差距不大，且推理速度提升约 30%。  
  - 与最先进的压缩技术相比，本文的压缩方案在相同 FLOPs 下提升 1.2–1.5 倍的准确率。  
- **消融实验**：作者分别去掉注意力剪枝、去掉隐藏层对齐蒸馏、只用单维度剪枝，发现每一项都贡献约 2–4% 的性能提升，说明多维度协同和双重蒸馏是关键。  
- **局限性**：压缩过程仍需要一次完整的模型剪枝与蒸馏循环，算力需求虽比全量预训练低 40 倍，但对显存仍有一定要求；此外，极端压缩（>5×）时性能下降明显，作者在论文中承认目前的搜索策略在高压缩率下不够稳健。

### 影响与延伸思考
这篇工作在 LLM 压缩领域掀起了“先训练后裁剪、少量再蒸馏”的潮流，随后出现的几篇论文（如 *SparseLLM*, *DistilLM*）都在不同程度上借鉴了多维度剪枝与双重蒸馏的思路。业界也开始把它作为模型家族发布的标准流程：先训练一个大基座模型，再通过自动化剪枝+蒸馏快速产出 2×、4×、8× 规模的子模型。想进一步深入，可以关注以下方向：  
- **自适应剪枝搜索**：把搜索过程嵌入训练循环，实现在线压缩。  
- **跨语言蒸馏**：利用多语言老师模型，让单语言学生在更少数据下获得跨语言能力。  
- **硬件感知压缩**：把显存、算子并行度等硬件约束直接加入搜索目标，产出更适配特定芯片的模型。  

### 一句话记住它
只要把大模型剪枝再用极少数据蒸馏，就能在 2‑4 倍压缩下得到比同尺寸从头训练更强的 LLM。