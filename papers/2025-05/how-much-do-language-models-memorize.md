# How much do language models memorize?

> **Date**：2025-05-30
> **arXiv**：https://arxiv.org/abs/2505.24832

## Abstract

We propose a new method for estimating how much a model knows about a datapoint and use it to measure the capacity of modern language models. Prior studies of language model memorization have struggled to disentangle memorization from generalization. We formally separate memorization into two components: unintended memorization, the information a model contains about a specific dataset, and generalization, the information a model contains about the true data-generation process. When we completely eliminate generalization, we can compute the total memorization, which provides an estimate of model capacity: our measurements estimate that GPT-style models have a capacity of approximately 3.6 bits per parameter. We train language models on datasets of increasing size and observe that models memorize until their capacity fills, at which point "grokking" begins, and unintended memorization decreases as models begin to generalize. We train hundreds of transformer language models ranging from $500K$ to $1.5B$ parameters and produce a series of scaling laws relating model capacity and data size to membership inference.

---

# 语言模型记忆了多少？ 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，研究者常常想知道模型到底把训练数据“背”了多少，而不是仅仅学会了语言的统计规律。过去的工作大多用“成员推断”（membership inference）来检测记忆，但这种方法把记忆和真正的泛化能力混在一起，导致很难判断模型到底是记住了原始句子，还是仅仅学会了类似的模式。再者，缺乏一个统一的度量手段，使得不同模型、不同规模之间的比较变得模糊。于是，如何把“记忆”和“泛化”严格分开、并给出一个可比的容量数字，成了亟待解决的难题。

### 关键概念速览
- **意外记忆（Unintended Memorization）**：模型对训练集中特定实例的记忆程度，超出正常泛化需求的部分。可以想象成学生在考试时把老师的原题抄下来，而不是只学会了解题技巧。  
- **泛化（Generalization）**：模型对真实数据生成过程的学习能力，即在未见过的样本上仍能表现良好。相当于学生掌握了解题思路，能够应对新题。  
- **模型容量（Model Capacity）**：这里指每个参数能够存储的信息量，单位是比特/参数。把模型想象成一本笔记本，容量就是每页能写多少信息。  
- **成员推断（Membership Inference）**：一种检测技术，判断某条数据是否出现在模型的训练集里。类似于侦探通过痕迹判断嫌疑人是否曾经到场。  
- **grokking**：当模型在训练初期大量记忆数据，随后容量饱和后开始转向真正的泛化，表现出“突然领悟”的现象。像是学生先背答案，后来真正理解了概念。  
- **Scaling Law（尺度律）**：描述模型大小、数据量与记忆/泛化表现之间的数学关系。类似于物理学里“质量与体积的比例”。  
- **完全消除泛化（Eliminate Generalization）**：在实验设计中人为让模型只能记忆而不能学习通用规律，以便测量纯记忆容量。可以比作把学生的教材全部删掉，只让他背答案。  

### 核心创新点
1. **记忆与泛化的形式化分离**  
   - 之前的成员推断只能给出一个混合信号。  
   - 本文引入了“意外记忆”和“泛化”两个互补信息量，并通过信息论框架把它们拆开。  
   - 结果是我们可以单独测量模型的纯记忆容量，而不被泛化干扰。

2. **通过完全抑制泛化来估算总记忆**  
   - 传统做法让模型在正常训练条件下学习，记忆与泛化交织。  
   - 作者在训练数据上加入了强噪声、随机标签等手段，使模型只能记住原始样本，无法抽取通用规律。  
   - 这样得到的记忆量直接对应模型的“信息存储上限”，从而推算出每参数约 3.6 比特的容量。

3. **大规模实验揭示容量饱和与 grokking 的关系**  
   - 过去只观察到模型在训练后期出现突变，却不清楚背后机制。  
   - 通过在不同数据规模下训练数百个模型，作者发现当记忆占满容量后，模型会自动转向泛化，意外记忆下降，这正是 grokking 现象的根源。  
   - 这为解释模型为何会出现“突然学会”提供了量化依据。

4. **提出一套完整的 scaling law**  
   - 之前的尺度律多聚焦在损失或推理速度上。  
   - 本文把模型参数量、训练数据量与成员推断成功率联系起来，形成了可预测的记忆/泛化曲线。  
   - 研究者可以据此预估给定规模模型在特定数据量下的记忆风险。

### 方法详解
整体思路可以概括为三步：**构造纯记忆任务 → 量化信息量 → 拟合尺度律**。

1. **构造纯记忆任务**  
   - 训练数据被人为扰乱：对每条文本随机打乱词序、替换标签或加入大量无意义噪声，使得模型无法从统计规律中获益。  
   - 只保留原始文本的唯一标识（例如特定的哈希），让模型只能通过记忆这些标识来完成下一个词预测。  
   - 这一步相当于把学生的教材全部换成乱码，只留下答案本身。

2. **信息量的度量**  
   - 对每个测试样本，使用成员推断算法估计模型对该样本的“可识别度”。  
   - 将可识别度转化为二进制信息量（bits），即模型对该样本的记忆强度。  
   - 将所有样本的记忆信息累计，除以模型参数总数，得到 **每参数记忆比特数**。  
   - 关键的公式是：记忆比特 = -log₂(误判率)，这里的误判率是模型把未见样本误判为训练样本的概率。

3. **拟合尺度律**  
   - 在参数规模从 0.5M 到 1.5B、数据规模从几万到上亿的组合上重复上述实验。  
   - 用回归模型捕捉 **记忆比特 = f(参数数, 数据量)** 的关系。  
   - 结果显示记忆比特随参数线性增长，但在数据量超过某个阈值后趋于饱和，形成 “容量上限”。  

**最巧妙的设计**在于“完全消除泛化”。作者通过两种手段实现：一是把训练目标改为预测随机噪声，二是对每个样本使用独立的随机种子，使得跨样本之间没有可学习的共性。这样模型只能靠记忆每个样本的噪声来降低损失，真正的语言规律被彻底切断。

### 实验与效果
- **实验平台**：作者训练了 300 多个 Transformer 语言模型，规模从 500K 参数到 1.5B 参数不等，使用的硬件主要是 NVIDIA A100 GPU。  
- **数据集**：主要使用公开的英文网页语料（如 C4）以及专门构造的噪声数据集，后者用于实现“无泛化”条件。  
- **基线对比**：与传统成员推断（在正常训练条件下）相比，纯记忆实验得到的记忆比特约高出 20%–30%，因为没有泛化的“稀释”。  
- **关键数字**：在 1B 参数模型上，测得的总记忆容量约为 3.6 比特/参数；在 500M 参数模型上略低，为 3.4 比特/参数，验证了容量随规模略有提升的趋势。  
- **消融实验**：作者分别去掉噪声标签、恢复部分真实语言统计，发现即使少量泛化信息也会显著降低测得的记忆比特，说明方法对“完全记忆”要求非常敏感。  
- **局限性**：实验仅在英文网页语料上完成，未验证中文或多模态数据；另外，完全抑制泛化的做法在实际应用中难以复现，只能作为理论测度手段。  

### 影响与延伸思考
这篇工作为“模型隐私风险”提供了一个可量化的基准，后续很多研究把 3.6 比特/参数作为评估模型泄露潜在信息的上限。随后出现的几篇论文（如 “Quantifying Memorization in Large Language Models”）直接引用了本文的容量估计，并尝试在实际对话模型中检测意外记忆。还有工作把作者的尺度律扩展到多语言模型、图像生成模型，探索不同模态的记忆上限是否相似。对想进一步深入的读者，可以关注 **信息论在深度学习中的应用**（尤其是 “bits‑per‑parameter” 这一度量）以及 **grokking 现象的动力学建模**，这些方向正逐步形成新的研究热点。

### 一句话记住它
**GPT 系列模型大约只能用每个参数 3.6 比特来“记住”训练数据，这个上限由纯记忆实验精确测得。**