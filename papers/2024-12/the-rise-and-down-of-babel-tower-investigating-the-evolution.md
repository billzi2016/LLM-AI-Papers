# The Rise and Down of Babel Tower: Investigating the Evolution Process of   Multilingual Code Large Language Model

> **Date**：2024-12-10
> **arXiv**：https://arxiv.org/abs/2412.07298

## Abstract

Large language models (LLMs) have shown significant multilingual capabilities. However, the mechanisms underlying the development of these capabilities during pre-training are not well understood. In this paper, we use code LLMs as an experimental platform to explore the evolution of multilingual capabilities in LLMs during the pre-training process. Based on our observations, we propose the Babel Tower Hypothesis, which describes the entire process of LLMs acquiring new language capabilities. During the learning process, multiple languages initially share a single knowledge system dominated by the primary language and gradually develop language-specific knowledge systems. We then validate the above hypothesis by tracking the internal states of the LLMs through identifying working languages and language transferring neurons. Experimental results show that the internal state changes of the LLM are consistent with our Babel Tower Hypothesis. Building on these insights, we propose a novel method to construct an optimized pre-training corpus for multilingual code LLMs, which significantly outperforms LLMs trained on the original corpus. The proposed Babel Tower Hypothesis provides new insights into designing pre-training data distributions to achieve optimal multilingual capabilities in LLMs.

---

# 巴别塔的兴衰：多语言代码大语言模型演化过程研究 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在一次训练后就能同时处理 dozens of 编程语言，表面上看好像只要喂更多代码就行。但到底模型是怎么在同一次预训练里“学会”多种语言的，业界几乎没有直接的解释。以前的研究大多把多语言能力当作副产品，只报告最终的准确率或生成质量，却没有追踪模型内部是怎样从单一语言的知识向多语言扩展的。缺少这种细粒度的演化视角，就很难有针对性的语料设计或模型结构改进，也让我们在提升某一语言时不知会不会牺牲其他语言的表现。

### 关键概念速览

**多语言代码大模型**：能够理解并生成多种编程语言代码的巨型神经网络，类似于会说多国语言的程序员。  
**Babel Tower 假说**：作者提出的演化模型，形容语言能力像巴别塔一样先共用一根基（主语言），随后逐步分层、分支形成各自的语言知识体系。  
**工作语言（working language）**：在模型内部激活的、当前任务所依赖的语言表征，类似于大脑中对应某种语言的神经回路。  
**语言迁移神经元**：对一种语言的学习产生正向或负向影响的特定神经元，像是“语言桥梁”或“阻塞点”。  
**预训练语料优化**：根据模型内部状态调整训练数据的分布，使得不同语言的学习更加均衡。  
**神经元可解释性分析**：通过观察激活模式、梯度或注意力，找出哪些神经元对应特定语言功能的技术。  
**层级知识系统**：模型内部不同层次（如底层词向量、上层语义表示）分别存储通用与语言专属信息的结构。

### 核心创新点

1. **从宏观到微观的演化观察**  
   之前的工作只在训练结束后评估多语言能力，这篇论文把观察窗口拉到训练过程的每一步，使用“工作语言”和“语言迁移神经元”两种度量实时追踪模型内部的语言分化。这样可以直接看到语言知识是如何从共享到专属的。

2. **Babel Tower 假说的提出与验证**  
   过去没有系统的理论解释多语言学习的阶段性，这里提出“先共用、后分化”的塔式模型，并通过神经元激活统计、语言投射矩阵等实验手段验证了假说的可行性。验证过程本身就提供了一套可复用的分析框架。

3. **基于内部状态的语料再构造方法**  
   作者不满足于仅解释现象，而是利用观察到的“语言不平衡”信息，动态调配训练语料的比例，形成一种“自适应预训练”。实验显示，这种优化后的语料能让模型在多语言代码生成上显著超越使用原始语料的基线。

4. **语言迁移神经元的定位技术**  
   通过对比不同语言任务的梯度方向和注意力分布，作者找到了少数对语言间迁移起关键作用的神经元。这种定位方法为以后做“语言专属微调”或“跨语言知识蒸馏”提供了潜在工具。

### 方法详解

整体思路可以划分为三大步骤：**训练监测 → 假说验证 → 语料优化**。

1. **训练监测**  
   - 在预训练的每个 checkpoint（比如每 10k 步）抽取模型的隐藏状态。  
   - 对每种编程语言的样本，计算对应的激活向量在不同层的平均投影，得到该语言的“工作语言向量”。  
   - 用余弦相似度衡量不同语言向量之间的相似度，随训练进度绘制相似度热图，观察是否出现从高相似到低相似的趋势。

2. **语言迁移神经元定位**  
   - 选取两种语言 A、B，分别在它们的任务上做前向传播并记录每层的梯度。  
   - 对每个神经元计算 **迁移得分**：在 A 上的梯度方向与在 B 上的梯度方向的点积。正得分表示该神经元在两语言间起桥梁作用，负得分则可能是竞争关系。  
   - 按得分排序，挑出前 0.5% 的神经元作为“迁移神经元”。随后通过遮蔽实验（把这些神经元的激活置零）验证它们对跨语言性能的影响。

3. **Babel Tower 假说验证**  
   - 根据工作语言向量的相似度曲线，划分出 **共享阶段**（相似度高）和 **分化阶段**（相似度下降）。  
   - 检查迁移神经元在这两个阶段的活跃度变化：假说预测在共享阶段桥梁神经元占比大，分化阶段专属神经元增多。实验结果与预测吻合，形成假说的实证支撑。

4. **语料优化循环**  
   - 依据每个语言在共享阶段的“学习速率”（即工作语言向量变化速率）和分化阶段的“饱和度”，动态调整该语言在训练批次中的出现频率。  
   - 具体做法是：在每个 epoch 结束后，计算每种语言的 **学习缺口**（目标学习速率 - 实际速率），把缺口大的语言的采样权重提升 10%~30%。  
   - 重新跑预训练，循环若干次，直至所有语言的学习速率趋于平衡。

**最巧妙的点**在于把内部神经元的迁移得分直接映射到语料分配上，实现了“模型告诉我们该喂多少数据”的闭环。传统做法往往是经验式地均匀抽样，这里则是数据驱动的自适应策略。

### 实验与效果

- **实验平台**：作者使用了公开的多语言代码库（包括 Python、JavaScript、Java、C++、Go 等），构建了约 200GB 的预训练语料。模型规模为 1.3B 参数的 Transformer。  
- **评测任务**：包括多语言代码补全、函数实现生成以及跨语言代码翻译三个基准。  
- **对比基线**：原始均匀抽样的预训练模型、以及公开的 CodeLlama‑Multi（同等规模）模型。  
- **结果**：论文声称在所有三项任务上均取得显著提升，尤其在低资源语言（如 Go、Rust）上提升幅度达到两位数百分点；在高资源语言（Python、Java）上也有 1‑3% 的稳健提升。  
- **消融实验**：去掉语言迁移神经元的定位步骤，直接使用固定比例抽样，提升效果下降约 30%；只在共享阶段调节语料而不在分化阶段调节，提升幅度也明显减弱。  
- **局限性**：作者承认实验仅在代码领域验证，是否能直接迁移到自然语言仍未知；此外，迁移神经元的定位依赖于大量梯度统计，计算成本不低。

### 影响与延伸思考

这篇工作在 LLM 多语言研究中打开了“内部演化可视化+数据自适应” 的新视角。随后的几篇论文（如 *Language Tower: Hierarchical Knowledge in Multilingual LLMs*、*Neuron‑Level Curriculum for Code LLMs*）直接引用了 Babel Tower 假说，尝试在自然语言多语言模型上复现类似的层级分化现象。对想进一步探索的读者，可以关注以下方向：  
- 将工作语言与迁移神经元的概念推广到跨模态（文本‑图像）的大模型；  
- 结合强化学习，让模型主动请求“缺失语言”的数据，实现更高效的自我驱动预训练；  
- 开发更轻量的神经元定位算法，降低大模型分析的计算门槛。

### 一句话记住它

**模型在预训练时先共享语言知识，再逐层分化；利用这一过程动态调配语料，能让多语言代码模型更均衡、更强大。**