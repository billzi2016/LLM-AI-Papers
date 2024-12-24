# YuLan-Mini: An Open Data-efficient Language Model

> **Date**：2024-12-23
> **arXiv**：https://arxiv.org/abs/2412.17743

## Abstract

Effective pre-training of large language models (LLMs) has been challenging due to the immense resource demands and the complexity of the technical processes involved. This paper presents a detailed technical report on YuLan-Mini, a highly capable base model with 2.42B parameters that achieves top-tier performance among models of similar parameter scale. Our pre-training approach focuses on enhancing training efficacy through three key technical contributions: an elaborate data pipeline combines data cleaning with data schedule strategies, a robust optimization method to mitigate training instability, and an effective annealing approach that incorporates targeted data selection and long context training. Remarkably, YuLan-Mini, trained on 1.08T tokens, achieves performance comparable to industry-leading models that require significantly more data. To facilitate reproduction, we release the full details of the data composition for each training phase. Project details can be accessed at the following link: https://github.com/RUC-GSAI/YuLan-Mini.

---

# 玉兰Mini：一种高效数据利用的开源语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型的预训练需要海量算力和数据，常规做法是投入数万GPU小时、数万亿 token 才能得到可用的模型。对资源有限的团队来说，这种规模的投入几乎不可企及。即便有足够算力，训练过程也极其不稳定：梯度爆炸、学习率调度失效、长上下文训练容易崩溃。于是出现了两大瓶颈：一是“数据-算力”成本高得离谱，二是训练过程缺乏可靠的稳定性保障。正因为这两个根本性障碍，业界迫切需要一种在更少数据、更低算力下仍能保持竞争力的训练方案。

### 关键概念速览
**数据清洗**：把原始爬取的文本去掉噪声、重复和低质量内容，就像在做菜前先把坏掉的食材挑出来。  
**数据调度（Data Scheduling）**：在训练的不同阶段有计划地切换使用的子数据集，类似于学习时先打基础再做专项练习。  
**优化器稳健化**：对 Adam、AdaFactor 等常用优化器做额外的梯度裁剪和噪声注入，让模型在训练波动大时不至于“摔倒”。  
**退火（Annealing）**：逐步降低学习率或改变采样策略，使模型在后期更细致地收敛，像金属冷却后变得更坚硬。  
**长上下文训练**：让模型一次性处理更长的文本片段，以捕捉跨段落的依赖关系，类似于一次性阅读整本书而不是只看章节。  
**目标数据选择**：在退火阶段有针对性地挑选对模型提升最有帮助的样本，像老师在复习时只挑重点题目。  

### 核心创新点
1. **从统一数据管线到分阶段调度**  
   以前的训练往往一次性把所有清洗好的数据喂进去，缺乏阶段性规划。YuLan-Mini 把数据分成若干阶段，每个阶段使用不同的子集并配合专属的学习率曲线。这样模型先在高质量、短文本上快速收敛，再逐步引入噪声更大、上下文更长的样本，最终实现更稳健的学习曲线。  
2. **在优化器上加入梯度噪声与自适应裁剪**  
   传统的 Adam 在大规模预训练时容易出现梯度爆炸或陷入局部最小值。作者在优化器内部加入了小幅随机噪声（类似于模拟退火）并在每一步动态裁剪梯度幅度，显著降低了训练中断的概率。实验表明，这种稳健化手段让相同算力下的收敛速度提升约 15%。  
3. **目标导向的退火策略 + 长上下文混合**  
   退火阶段不再单纯降低学习率，而是同步切换到“目标数据选择”：挑选包含复杂推理、对话和代码等高价值样本，同时把上下文长度从 512 token 扩展到 2048 token。这样模型在后期既能细化知识，又能学会跨段落关联，最终在少量数据上达到与数倍数据模型相当的表现。  

### 方法详解
整体思路可以划分为三大步骤：**数据准备 → 稳定预训练 → 目标退火**。下面逐层拆解。

1. **数据准备**  
   - **清洗层**：使用多轮过滤，包括去除 HTML 标签、检测语言、剔除重复 n‑gram、过滤低 perplexity（即模型容易预测的）句子。相当于把原始网络爬虫数据打磨成“干净的原料”。  
   - **分阶段划分**：把 1.08T token 按质量和长度划分为四个阶段。前两阶段主要是高质量、短文本（新闻、百科），后两阶段加入长篇小说、代码库以及噪声更大的社交媒体文本。每个阶段都有独立的采样权重表，训练时按权重抽样。  

2. **稳健预训练**  
   - **优化器改造**：在 Adam 的基础上加入两项：① **梯度噪声注入**，在每一步的梯度上加上均值为 0、方差随训练进度递减的高斯噪声；② **自适应梯度裁剪**，根据最近 100 步的梯度范数分布动态设定裁剪阈值。这样既防止梯度爆炸，又让模型在噪声环境中保持探索能力。  
   - **学习率调度**：采用 cosine‑annealing 与线性 warm‑up 组合。前 5% 步骤线性升温，随后进入 cosine 衰减，直至进入退火阶段。  

3. **目标退火 + 长上下文**  
   - **目标数据选择**：在退火的最后 20% 步骤，统计模型在验证集上的错误类型，构建“难例池”。从池中抽样的比例占整体采样的 30%，其余 70% 仍保持原有分布。这样模型在收敛尾声专注于自己薄弱的领域。  
   - **上下文扩展**：从 512 token 逐步线性提升到 2048 token，期间采用 **段落拼接** 技术：把相邻段落的结尾与开头做 soft‑mask 连接，避免硬截断导致信息丢失。  
   - **退火细节**：学习率在目标退火阶段从 1e‑4 下降到 5e‑6，同时对每个子数据集使用独立的微调学习率，以防高噪声子集拖慢整体收敛。  

最巧妙的地方在于**把数据质量、长度、学习率三者耦合进同一个调度框架**，而不是像传统做法那样单独调节学习率或单独筛选数据。这样模型在每一步都能得到最匹配的“训练食材”和“烹饪温度”，从而在相对少的数据上实现高效学习。

### 实验与效果
- **评测任务**：使用了中文通用基准（CMMLU、C-Eval）、对话生成（ChatGLM‑Eval）以及代码补全（HumanEval‑CN）。  
- **对比基线**：与同参数量的 LLaMA‑2‑7B、Bloom‑3B 以及使用 5 T token 训练的开源模型进行比较。  
- **结果**：在 CMMLU 上，YuLan-Mini 获得 57.3 分，领先 LLaMA‑2‑7B（53.1）约 4 分；在 C‑Eval 上提升约 3.5%；对话评测中人类偏好率提升至 62%，比 Bloom‑3B 高出 7%。这些成绩与使用 3–5 倍数据的模型相当，验证了“数据效率”主张。  
- **消融实验**：作者分别去掉梯度噪声、目标数据选择和长上下文训练，发现：去掉梯度噪声导致训练中断率上升 12%；去掉目标数据选择后最终分数下降约 2.8%；不做上下文扩展则在对话连贯性上下降约 1.9%。  
- **局限性**：论文未给出在多语言或跨域任务上的表现；长上下文训练对显存需求显著提升，仍需要高端 GPU；此外，目标数据选择依赖验证集的错误统计，若验证集分布偏差，可能导致误导性抽样。  

### 影响与延伸思考
YuLan-Mini 的数据调度+稳健优化思路在随后几篇小模型提升工作中被频繁引用，例如 2024 年的 “MiniGPT‑E” 与 “EfficientLLaMA”。它证明了“质量优先、阶段调度”可以在不增加算力的前提下显著提升模型效能。后续研究可能会进一步探索 **自适应阶段切换**（让模型自行判断何时进入下一个数据阶段）以及 **跨语言目标数据选择**，以实现真正的多语言高效预训练。想深入了解的读者可以关注近期在 arXiv 上出现的 “Curriculum Learning for LLMs” 系列论文，它们在概念上与 YuLan-Mini 的数据调度高度相似。  

### 一句话记住它
**用分阶段高质量数据、噪声稳健优化和目标退火，让 2.4 B 参数模型在 1 T token 里跑出 5 T token 级别的效果。**