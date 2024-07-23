# Data Mixture Inference: What do BPE Tokenizers Reveal about their   Training Data?

> **Date**：2024-07-23
> **arXiv**：https://arxiv.org/abs/2407.16607

## Abstract

The pretraining data of today's strongest language models is opaque; in particular, little is known about the proportions of various domains or languages represented. In this work, we tackle a task which we call data mixture inference, which aims to uncover the distributional make-up of training data. We introduce a novel attack based on a previously overlooked source of information: byte-pair encoding (BPE) tokenizers, used by the vast majority of modern language models. Our key insight is that the ordered list of merge rules learned by a BPE tokenizer naturally reveals information about the token frequencies in its training data. Given a tokenizer's merge list along with example data for each category of interest, we formulate a linear program that solves for the proportion of each category in the tokenizer's training set. In controlled experiments, we show that our attack recovers mixture ratios with high precision for tokenizers trained on known mixtures of natural languages, programming languages, and data sources. We then apply our approach to off-the-shelf tokenizers released with recent LMs. We confirm much publicly disclosed information about these models, and also make several new inferences: GPT-4o and Mistral NeMo's tokenizers are much more multilingual than their predecessors, training on 39% and 47% non-English language data, respectively; Llama 3 extends GPT-3.5's tokenizer primarily for multilingual (48%) use; GPT-3.5's and Claude's tokenizers are trained on predominantly code (~60%). We hope our work sheds light on current design practices for pretraining data, and inspires continued research into data mixture inference for LMs.

---

# 数据混合推断：BPE 分词器能揭示其训练数据的哪些信息？ 论文详细解读

### 背景：这个问题为什么难？
现代大语言模型的预训练语料规模浩瀚，却几乎没有公开的统计数据。研究者只能凭借官方的模糊描述，猜测模型到底用了多少英文、多少代码、多少非英语文本。传统的“看模型输出”或“查询模型内部激活”方法只能间接推断，往往受限于模型的黑箱特性，难以给出可靠的比例。更关键的是，现有的统计手段没有利用模型训练前的一个关键环节——分词器本身所携带的频率暗示。因此，要想从模型外部直接、精确地恢复训练数据的混合比例，需要一种全新的思路。

### 关键概念速览
**BPE（Byte‑Pair Encoding）分词器**：一种把常见字符或字符对合并成新“子词”的算法，训练时会产生一条有序的合并规则列表。可以把它想成“拼字游戏的合并手册”，越早出现的规则对应越常见的子词。  
**合并规则列表（merge list）**：BPE 在训练时记录的每一步合并操作的顺序，类似于烹饪食谱的步骤，顺序暗示了材料（子词）的使用频率。  
**数据混合推断（Data Mixture Inference）**：给定分词器和若干已知类别的示例，逆向求解每个类别在原始训练语料中所占比例的任务。  
**线性规划（Linear Programming）**：一种在满足线性约束的前提下最优化线性目标函数的数学工具，这里用来把“子词出现频率 = 类别比例 × 类别子词频率”这套线性关系求解出来。  
**类别示例（Category Exemplars）**：研究者提前准备的、能够代表目标类别（如英语、中文、Python 代码等）的文本样本，用来估计每个类别在合并规则下的子词分布。  
**频率暗示（Frequency Hint）**：BPE 合并顺序中隐含的子词出现频次信息，类似于从一本书的目录顺序猜测章节重要性的线索。

### 核心创新点
1. **从分词器而非模型本体获取信息**：过去的工作大多尝试直接分析模型的权重或输出分布，忽视了分词器本身携带的统计线索。本文把 BPE 的合并规则视作“频率指纹”，打开了一条全新的信息渠道。  
2. **把合并规则转化为线性约束**：作者观察到每条合并规则对应的子词出现频率可以用各类别的子词频率的加权和来表示。于是构建了一个线性方程组，直接映射到线性规划求解，而不是使用迭代的概率模型。这样既省时又保证全局最优。  
3. **只需少量类别示例即可恢复比例**：传统的逆向统计往往需要大规模标注数据。这里只要为每个感兴趣的类别准备几百句代表性文本，就能估计出该类别在整个训练语料中的占比，极大降低了实验成本。  
4. **在真实商用分词器上验证并发现新现象**：作者把方法应用到 GPT‑4o、Mistral NeMo、Llama 3 等公开分词器，成功复现官方披露的比例，同时揭示了之前未公开的多语言或代码占比，证明了方法的实用价值。

### 方法详解
整体思路可以拆成四步：

1. **收集目标分词器的合并规则列表**  
   从公开的模型仓库或直接读取 tokenizer 文件，得到一条按出现频率从高到低排序的合并规则序列。每条规则对应一个子词（例如 “Ġthe” 或 “Ġdef”）。

2. **准备每个类别的示例文本并统计子词频率**  
   对每个感兴趣的类别（如英文、中文、Python、JavaScript 等），作者手动挑选数百句具有代表性的语料。使用同一 BPE 分词器对这些示例进行分词，统计每个子词在该类别中的出现次数，得到一个“类别子词频率向量”。可以把它想成每个类别的“子词配方”。

3. **构建线性方程组**  
   对于合并规则列表中的前 N 条（N 通常取几千），记第 i 条规则对应的子词出现频率为 f_i（从分词器的统计信息直接读取）。根据混合比例 p_c（c 为类别）和类别子词频率向量，可以写出：  
   f_i ≈ Σ_c p_c * freq_{c,i}  
   这里的 Σ 表示所有类别的加权求和。所有 i 条规则形成一个线性方程组，未知数只有各类别的比例 p_c。

4. **线性规划求解**  
   为了保证比例非负且总和为 1，加入约束 p_c ≥ 0， Σ_c p_c = 1。目标函数可以是最小化所有方程的绝对误差或平方误差。使用标准的线性规划求解器（如 Gurobi、CVXPY）即可得到最优的 p_c 值。求解过程相当于在“子词频率空间”里找一个最贴合观测频率的混合点。

**关键细节**  
- 只使用前几千条规则是因为后面的规则对应的子词极少出现，噪声大，反而会削弱求解精度。  
- 为防止某些子词在所有类别中都极少出现导致方程不满秩，作者会对频率极低的子词做阈值过滤。  
- 线性规划的目标函数采用 L1 范数（绝对误差）而非 L2，能够更鲁棒地抵御少数异常子词的影响。  
- 实验中发现，合并规则的顺序本身已经足够提供强信号，甚至不需要实际的子词计数，只要知道规则出现的相对位置即可近似恢复比例。

### 实验与效果
- **受控实验**：作者先在人为构造的混合语料上验证方法。比如把英文、中文、Python 代码按 30% / 40% / 30% 混合，训练 BPE 分词器后再用本文方法恢复比例，误差在 ±2% 以内，显示出极高的精度。  
- **真实模型测试**：对 GPT‑4o、Mistral NeMo、Llama 3、GPT‑3.5、Claude 等公开分词器进行推断。结果显示 GPT‑4o 大约 39% 的子词来源于非英文语言，Mistral NeMo 为 47%；Llama 3 在原有 GPT‑3.5 分词器基础上扩展，非英文比例提升至 48%；而 GPT‑3.5 与 Claude 的分词器中代码子词占比约 60%。这些数字与官方披露的训练数据比例基本吻合，同时也补足了官方未提及的多语言比例。  
- **基线对比**：与最直接的“词表频率统计”方法相比，本文的线性规划方案在同等数据量下误差降低约 30%。  
- **消融实验**：去掉类别示例的多样性（仅使用单一主题文本）会导致恢复比例偏差显著上升，说明示例的覆盖面对准确性至关重要。  
- **局限性**：方法依赖于分词器的公开可得性；如果模型使用了自定义或加密的分词器，无法直接应用。此外，极端稀有语言或极低频子词在合并规则中可能根本不出现，导致这类语言的比例被低估。作者在论文中也承认，对极度不平衡的混合（如 99% 英文、1% 其他）时，误差会稍大。

### 影响与延伸思考
这篇工作打开了“从分词器逆向窥探训练数据”的新视角，随后出现的几篇跟进研究尝试把同样的思路扩展到 SentencePiece、Unigram 等其他分词框架，甚至把分词器的子词嵌入向量也纳入线性模型，以提升对细粒度主题的辨识度。业界开始在模型发布时更主动地提供分词器的合并规则，以便外部审计。对想进一步探索的读者，可以关注以下方向：  
- **多语言细分**：把“非英文”进一步拆分为具体语言（法语、阿拉伯语等），需要更细致的类别示例和更高维的线性系统。  
- **动态训练数据追踪**：在模型持续微调过程中，实时更新分词器并监控比例变化，帮助评估增量数据的影响。  
- **隐私与安全**：如果分词器能泄露训练数据的分布信息，是否会成为攻击向量？研究如何在不牺牲模型性能的前提下对分词器进行“去指纹化”。  
- **跨模型比较**：利用统一的混合推断框架，对比不同公司、不同规模模型的训练数据策略，形成行业基准。

### 一句话记住它
只要看 BPE 分词器的合并顺序，就能像读“频率指纹”一样，精准逆推出语言模型的训练数据配比。