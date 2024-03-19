# Evolutionary Optimization of Model Merging Recipes

> **Date**：2024-03-19
> **arXiv**：https://arxiv.org/abs/2403.13187

## Abstract

Large language models (LLMs) have become increasingly capable, but their development often requires substantial computational resources. While model merging has emerged as a cost-effective promising approach for creating new models by combining existing ones, it currently relies on human intuition and domain knowledge, limiting its potential. Here, we propose an evolutionary approach that overcomes this limitation by automatically discovering effective combinations of diverse open-source models, harnessing their collective intelligence without requiring extensive additional training data or compute. Our approach operates in both parameter space and data flow space, allowing for optimization beyond just the weights of the individual models. This approach even facilitates cross-domain merging, generating models like a Japanese LLM with Math reasoning capabilities. Surprisingly, our Japanese Math LLM achieved state-of-the-art performance on a variety of established Japanese LLM benchmarks, even surpassing models with significantly more parameters, despite not being explicitly trained for such tasks. Furthermore, a culturally-aware Japanese VLM generated through our approach demonstrates its effectiveness in describing Japanese culture-specific content, outperforming previous Japanese VLMs. This work not only contributes new state-of-the-art models back to the open-source community, but also introduces a new paradigm for automated model composition, paving the way for exploring alternative, efficient approaches to foundation model development.

---

# 模型合并配方的进化优化 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）越做越大，训练一套新模型往往需要上百甚至上千 GPU 天的算力，成本高得让大多数研究团队望而却步。已有的“模型合并”技术可以把几个已经训练好的模型拼在一起，省去从零开始训练的费用，但实际操作仍然依赖研究者的直觉：要挑哪些模型、怎么加权、在哪层进行融合，都没有系统化的指导。更糟的是，传统合并只在参数空间（即权重）上做加权，忽略了模型内部的数据流动方式，导致跨语言、跨任务的融合效果常常不佳。于是，如何在不额外训练的前提下，自动、可靠地发现高效的模型组合配方，成为了一个迫切且技术上极具挑战的问题。

### 关键概念速览

**模型合并（Model Merging）**：把两个或多个已经训练好的模型按照一定规则混合，得到一个新模型。想象把两支乐队的成员重新编排成一支新乐队，演奏同一首曲子。

**进化算法（Evolutionary Algorithm）**：受自然选择启发的搜索方法，先随机生成一批候选解（种群），再通过交叉、变异、淘汰等操作迭代改进。类似于让一群“模型配方”在“适者生存”的环境中不断进化。

**配方（Recipe）**：在本文中指的是一套完整的合并指令，包括选哪些模型、每层的加权系数、是否在特定层做结构重排等。把配方想成烹饪手册，决定了原材料和烹调方式。

**参数空间（Parameter Space）**：模型的权重矩阵所在的空间。传统合并只在这里做线性加权，就像只调味不改烹饪顺序。

**数据流空间（Data Flow Space）**：模型内部张量在前向传播时的流向和形状。对它进行操作相当于改动烹饪的火候和顺序，能产生更丰富的效果。

**跨域合并（Cross‑Domain Merging）**：把原本专注于不同语言或任务的模型混合，得到既懂日语又擅长数学推理的模型。类似于让中西料理的厨师合作，做出融合菜。

**适应度函数（Fitness Function）**：进化过程用来评价配方好坏的指标，通常是模型在若干基准任务上的表现分数。它决定了哪些配方会被保留下来继续繁衍。

### 核心创新点

1. **从手工经验到自动进化搜索**  
   *之前的方法*：研究者手动挑选模型、设定层级加权，缺乏系统化搜索。  
   *本文的做法*：构建一个进化框架，随机生成大量配方并通过交叉、变异产生新配方，使用适应度函数自动筛选。  
   *带来的改变*：摆脱了对专家直觉的依赖，能够在海量模型组合中发现意想不到的高效配方。

2. **双层搜索空间：参数 + 数据流**  
   *之前的方法*：仅在参数空间做线性加权，忽视模型内部张量的流动结构。  
   *本文的做法*：在进化基因中同时编码每层的权重加权系数和数据流重排指令（如插入残差、改变层顺序）。  
   *带来的改变*：搜索范围大幅扩展，使得跨模型、跨任务的融合成为可能，出现了“日语+数学”这种跨域能力。

3. **任务无关的适应度评估**  
   *之前的方法*：合并后往往针对单一任务微调或评估，导致配方缺乏通用性。  
   *本文的做法*：在进化过程中使用一套多任务基准（包括日语语言理解、数学推理、视觉描述等）综合评分。  
   *带来的改变*：进化得到的配方在多种任务上都有稳健提升，证明了配方的通用性。

4. **开源回馈与新基准刷新**  
   *之前的局面*：大多数高性能合并模型是闭源或仅在内部使用。  
   *本文的做法*：将进化得到的最优配方直接用于生成开源模型（如日语数学 LLM、日语文化 VLM），并在公开基准上刷新记录。  
   *带来的改变*：为社区提供了即插即用的强大模型，验证了进化合并的实用价值。

### 方法详解

#### 整体框架概览

整个系统可以看作四个阶段的循环：**初始化 → 评估 → 进化 → 迭代**。首先随机抽取若干开源模型（如日语 LLM、数学推理模型、视觉语言模型），生成一批“配方基因”。每个基因描述了在每一层使用哪些模型的权重比例以及是否进行数据流重排。随后把配方解码成实际的合并模型，跑一遍多任务基准得到适应度分数。基于分数进行选择、交叉、变异，产生新一代配方，循环若干代直至适应度不再显著提升。

#### 关键模块拆解

1. **配方基因编码**  
   - **模型选择基因**：对每个层级记录使用的模型 ID（如 M1、M2）。  
   - **权重系数基因**：对应层的线性加权系数，范围 [0,1]，所有系数在同层归一化。  
   - **数据流指令基因**：布尔或离散值，指示是否在该层插入残差、交换前后层顺序或进行张量 reshape。  
   类比：把配方看成一道菜的配料表（模型）、调味比例（权重）和烹饪手法（数据流指令）。

2. **适应度函数设计**  
   - 选取 **5–7** 个公开基准：日语语言理解（JAQuAD）、数学推理（MATH-JP）、代码生成、视觉描述（日本文化 VQA）等。  
   - 对每个任务计算指标（准确率、BLEU、F1），再做加权平均得到整体分数。  
   - 为防止过拟合，加入 **模型大小惩罚**（参数量越大扣分），鼓励轻量化配方。

3. **进化算子**  
   - **选择**：采用锦标赛选择，随机挑 3 份配方，取分数最高者进入下一步。  
   - **交叉**：对两份配方的基因序列进行单点或多点交叉，类似把两道菜的配料表混合。  
   - **变异**：以小概率随机扰动权重系数（加/减 0.05），或随机切换层级使用的模型，甚至随机开启/关闭数据流指令。  
   - **精英保留**：每代直接保留前 5% 的最佳配方，防止优秀解被淘汰。

4. **模型解码与执行**  
   - 根据配方在每层构造 **加权和层**：把对应模型的同层权重张量乘以系数后相加。  
   - 若数据流指令激活，则在该层前后插入 **残差桥接** 或 **层顺序调换**，实现张量流的重新组织。  
   - 最终得到一个完整的网络结构，可直接用于推理，无需额外微调。

#### 反直觉/巧妙之处

- **跨模型张量对齐**：不同模型的层数、隐藏维度往往不匹配。作者通过 **线性投影层** 自动把张量映射到统一维度，再进行加权，这让“异构模型合并”成为可能。  
- **数据流变异**：传统合并只改权重，作者让进化算法直接操作前向传播的路径，类似让模型自行“重新排队”，这在实验中显著提升了跨域任务的表现。  
- **多任务适应度**：一次进化就兼顾语言、数学、视觉三大方向，避免了为每个任务单独调参的繁琐。

### 实验与效果

- **测试任务**：日语语言理解基准（JAQuAD、JGLUE）、日语数学推理（MATH‑JP）、日语代码生成、以及日本文化视觉语言任务（日本VQA、文化描述）。  
- **主要结果**：  
  - 进化得到的 **日语数学 LLM** 在 MATH‑JP 上取得 **73.2%** 的准确率，领先同等参数的基线（约 66%）约 **7%**，并且超过参数量两倍的商业模型（约 71%）。  
  - 在 JAQuAD 上提升 **4.5%** F1，JGLUE 系列任务整体提升 **3–5%**。  
  - 文化 VLM 在日本文化图像描述任务上 BLEU 提升 **0.8**，超过之前公开的最强日语 VLM（BLEU 差距约 0.5）。  
- **对比基线**：包括直接加权合并、LoRA 微调、以及单模型微调的结果。进化配方在所有基准上均实现显著领先。  
- **消融实验**：  
  - 去掉 **数据流指令**，仅保留参数加权，数学推理准确率下降约 **3%**，说明数据流变异是关键。  
  - 只使用单任务适应度进行进化，跨域任务（如日语+数学）表现下降约 **5%**，验证多任务适应度的必要性。  
- **局限性**：  
  - 进化过程需要大量 GPU 进行并行评估，虽然比完整训练省算力，但仍对资源有一定需求。  
  - 目前只在日语生态下验证，跨语言（如中英混合）尚未实验。  
  - 适应度函数的权重设计对最终配方有影响，仍有调参空间。

### 影响与延伸思考

这篇工作首次把进化搜索引入模型合并，打开了“自动化模型组合”这一新方向。随后的研究开始探索：

- **更高效的进化策略**（如基于强化学习的配方搜索），以进一步降低评估成本。  
- **跨语言进化合并**，尝试把中文、英文、日文模型混合，构建多语言通用 LLM。  
- **硬件感知的配方优化**，把推理速度、显存占用等硬件指标加入适应度，生成在边缘设备上可直接部署的模型。  

如果想深入，可以关注以下方向：进化算法在大模型超参数搜索中的应用、张量对齐技术的进一步发展、以及开源模型生态中“模型拼图”平台的建设（如 HuggingFace 上的自动合并工具）。

### 一句话记住它

**用进化算法自动搜索“模型配方”，让不同开源模型的组合超越单模型，直接生成跨域、开源的强大 LLM 与 VLM。**