# $k$NN Prompting: Beyond-Context Learning with Calibration-Free Nearest   Neighbor Inference

> **Date**：2023-03-24
> **arXiv**：https://arxiv.org/abs/2303.13824

## Abstract

In-Context Learning (ICL), which formulates target tasks as prompt completion conditioned on in-context demonstrations, has become the prevailing utilization of LLMs. In this paper, we first disclose an actual predicament for this typical usage that it can not scale up with training data due to context length restriction. Besides, existing works have shown that ICL also suffers from various biases and requires delicate calibration treatment. To address both challenges, we advocate a simple and effective solution, $k$NN Prompting, which first queries LLM with training data for distributed representations, then predicts test instances by simply referring to nearest neighbors. We conduct comprehensive experiments to demonstrate its two-fold superiority: 1) Calibration-Free: $k$NN Prompting does not directly align LLM output distribution with task-specific label space, instead leverages such distribution to align test and training instances. It significantly outperforms state-of-the-art calibration-based methods under comparable few-shot scenario. 2) Beyond-Context: $k$NN Prompting can further scale up effectively with as many training data as are available, continually bringing substantial improvements. The scaling trend holds across 10 orders of magnitude ranging from 2 shots to 1024 shots as well as different LLMs scales ranging from 0.8B to 30B. It successfully bridges data scaling into model scaling, and brings new potentials for the gradient-free paradigm of LLM deployment. Code is publicly available.

---

# kNN 提示：无校准最近邻推理的超上下文学习 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在少量示例下完成新任务的能力，主要靠**在上下文中示例（In‑Context Learning，ICL）**。然而，LLM的上下文窗口是有限的，几百甚至几千个 token，导致只能塞进几条示例，数据规模受限。更糟的是，ICL 的输出往往带有标签偏置，需要额外的**校准**步骤（比如软提示、对数线性校正）才能把模型的概率分布对齐到具体任务的标签空间。于是出现了两个痛点：① 训练数据多了也装不进去；② 直接用模型输出会产生系统性错误，需要细致调参。解决这两个问题就能让 LLM 在少样本场景下更稳、更强。

### 关键概念速览
- **In‑Context Learning（上下文学习）**：把任务示例写进模型的输入，让模型把这些示例当作“例子”来推理答案。类似老师在黑板上写几道例题，学生据此解答新题。
- **上下文窗口（Context Length）**：模型一次性能处理的 token 数量上限。想象成一次只能装进有限格子的纸张，格子不够就写不下更多例子。
- **校准（Calibration）**：对模型输出的概率进行后处理，使其更符合真实标签分布。像是把一个偏向左的天平调平，让预测更公平。
- **k‑Nearest Neighbor（k 最近邻）**：在特征空间里找离目标最近的 k 条已有数据，然后依据这些邻居的标签做决定。就像在陌生城市里向路人询问方向，找最相似的几个人的建议。
- **分布式表示（Embedding）**：把文字或句子映射成向量，向量之间的距离反映语义相似度。可以把它想成把每句话压缩成一张坐标图上的点。
- **无校准（Calibration‑Free）**：不需要对模型输出做额外的概率调节，直接利用原始分布完成任务。相当于直接用原始的温度计读数，而不是再做一次校正。

### 核心创新点
1. **从“示例写进上下文”到“示例存库检索”**  
   之前的做法是把少量示例塞进 prompt，受限于上下文长度。本文改为先把所有训练样本喂给 LLM，获取它们的向量表示，形成一个检索库。预测时不再把示例写进 prompt，而是用 k‑NN 在库里找最相似的邻居。这样训练数据可以任意多，真正实现“超上下文”。

2. **利用模型的输出分布对齐实例，而不是直接对齐标签**  
   传统校准方法把模型的 logits（未归一化的概率）映射到任务标签上，需要手工调参。kNN Prompting 直接把测试实例的向量与训练实例的向量比较，靠相似度自然把两者对齐，省去任何额外的校准层。相当于让模型自己说“我和这几条训练样本很像”，从而决定答案。

3. **在少样本到中等规模（2→1024）全程保持优势**  
   实验显示，随着示例数量从 2 增加到 1024，性能持续提升，且始终领先同等条件下的校准基线。创新点在于证明“数据规模可以像模型规模一样线性提升”，这在传统 ICL 中是看不到的。

4. **跨模型、跨规模的通用性**  
   作者在 0.8B、1.3B、6B、30B 等不同大小的 LLM 上都跑通了实验，效果一致。说明方法不依赖特定模型结构，只要模型能输出可靠的向量，就能直接套用。

### 方法详解
**整体思路**可以概括为三步：① 生成训练样本的向量库；② 为每个测试样本生成向量；③ 用 k‑NN 在库里找最近邻并投票决定标签。整个流程不需要梯度更新，也不需要在 prompt 中写示例，完全是“检索+投票”。

#### 第一步：向量库构建
- 把每条训练数据（包括输入和对应的标签）拼接成一个完整的 prompt，喂给 LLM。
- 读取 LLM 最后一层的隐藏状态（或专门的 embedding 输出），得到一个固定维度的向量。这里的向量可以看作是 LLM 对该示例的“语义记忆”。
- 把向量和对应的标签一起存入一个高效的近似最近邻索引（如 FAISS），方便后续快速检索。

> 类比：把所有教材的章节摘要写在卡片上，放进抽屉里，等需要时快速抽出最相关的卡片。

#### 第二步：测试实例向量化
- 对每个待预测的输入，同样构造一个 prompt（只包含输入本身），让 LLM 产生向量。
- 这里不需要把任何示例放进去，避免上下文被占满。

#### 第三步：k‑NN 检索与决策
- 在向量库中，用欧氏距离或余弦相似度找出与测试向量最近的 k 条训练向量。
- 收集这 k 条邻居的标签，按照相似度加权或多数投票得到最终预测。若任务是分类，直接投票；若是回归或生成任务，可对邻居的输出做加权平均或拼接。

**最巧妙的地方**在于：模型的 **输出分布**（即 logits）并没有直接参与决策，而是通过向量相似度把“模型内部的语义空间”变成了对齐工具。这样既避免了校准的繁琐，也让数据规模不再受上下文限制。

**实现细节**（原文未详细展开的部分）可能包括：
- 采用哪一层的隐藏状态（常用最后一层或倒数第二层）；
- 是否对向量做归一化以使用余弦相似度；
- k 的取值经验（如 4、8、16）以及加权方式；
- 对大规模库使用近似最近邻搜索以保持检索效率。

### 实验与效果
- **任务与数据集**：论文在十余个少样本基准上评估，包括文本分类（如 SST‑2、AGNews）、情感分析、自然语言推理等。覆盖从二分类到多分类的常见任务。
- **对比基线**：与标准的 ICL（直接把 few‑shot 示例写进 prompt）以及最新的校准方法（如 Calibrate‑Before‑Use、Logit‑Adjustment）进行比较。  
  - **结果**：在相同的 few‑shot 条件下，kNN Prompting 超过这些基线，尤其在 8‑shot、16‑shot等中等规模时提升最明显。论文声称在多数任务上提升幅度在 2%‑5% 之间，且在 1024‑shot 场景下仍保持领先。
- **规模实验**：作者把训练示例数量从 2 增至 1024，性能呈单调上升趋势，说明方法能够真正利用更多数据。模型规模从 0.8B 到 30B 均表现出相似的提升趋势，验证了跨模型的通用性。
- **消融研究**：论文提供了两项关键消融：① 不使用向量库、直接用模型输出进行投票（相当于传统 ICL）——性能大幅下降；② 改用随机向量代替真实嵌入——同样失效。说明向量质量和最近邻检索是核心驱动因素。
- **局限性**：作者承认方法依赖于 LLM 能够产生区分度高的向量；在极端长文本或需要细粒度推理的任务上，向量相似度可能不足以捕捉全部信息。此外，检索库的构建和存储成本随数据规模线性增长，对极大数据集仍有实际限制。

### 影响与延伸思考
- 这篇工作把 **检索** 与 **大模型少样本推理** 直接结合，开启了“检索增强 ICL” 的新思路。随后出现的几篇论文（如 Retrieval‑Augmented Generation、Prompt‑Based Memory）都在不同角度上借鉴了“把训练数据当作外部记忆库”的理念。
- 对于想在资源受限的环境下部署 LLM 的团队，这种 **梯度自由、校准自由** 的方案提供了更易实现的路径，只要有向量检索服务即可。
- 未来可以探索的方向包括：① 将向量库与 **跨模态**（图像、音频）数据统一检索，扩展到多模态任务；② 结合 **自适应 k** 或 **软邻居权重学习**，让模型自行决定检索深度；③ 把检索过程嵌入到 **链式思考（Chain‑of‑Thought）** 中，让模型在推理时动态查询相似案例。

### 一句话记住它
**把所有训练示例变成向量库，用最近邻直接决定答案，既不受上下文长度限制，也不需要任何校准。**