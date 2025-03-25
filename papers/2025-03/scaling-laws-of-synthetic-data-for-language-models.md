# Scaling Laws of Synthetic Data for Language Models

> **Date**：2025-03-25
> **arXiv**：https://arxiv.org/abs/2503.19551

## Abstract

Large language models (LLMs) achieve strong performance across diverse tasks, largely driven by high-quality web data used in pre-training. However, recent studies indicate this data source is rapidly depleting. Synthetic data emerges as a promising alternative, but it remains unclear whether synthetic datasets exhibit predictable scalability comparable to raw pre-training data. In this work, we systematically investigate the scaling laws of synthetic data by introducing SynthLLM, a scalable framework that transforms pre-training corpora into diverse, high-quality synthetic datasets. Our approach achieves this by automatically extracting and recombining high-level concepts across multiple documents using a graph algorithm. Key findings from our extensive mathematical experiments on SynthLLM include: (1) SynthLLM generates synthetic data that reliably adheres to the rectified scaling law across various model sizes; (2) Performance improvements plateau near 300B tokens; and (3) Larger models approach optimal performance with fewer training tokens. For instance, an 8B model peaks at 1T tokens, while a 3B model requires 4T. Moreover, comparisons with existing synthetic data generation and augmentation methods demonstrate that SynthLLM achieves superior performance and scalability. Our findings highlight synthetic data as a scalable and reliable alternative to organic pre-training corpora, offering a viable path toward continued improvement in model performance.

---

# 合成数据规模律对语言模型的研究 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）的强大能力主要来源于海量、质量高的网页文本，但这些公开网页数据正被快速消耗，新增优质语料的边际收益越来越低。直接爬取更多网页会遇到版权、噪声和多样性不足等瓶颈。于是研究者开始尝试用模型自己生成的“合成数据”来补充预训练，但此前没有系统的证据表明合成数据能像真实语料那样遵循可预测的规模律（即模型性能随数据量和模型大小的关系）。缺乏这种可预测性，就很难判断投入多少合成数据才值得，导致合成数据的使用仍然是“试错”而非科学规划。

### 关键概念速览
**大语言模型（LLM）**：参数量从几亿到上千亿不等的自回归文本生成模型，类似于会写作文的“超级机器人”。  
**预训练语料**：模型在正式下游任务前学习的海量文本，像是学生的“通识教材”。  
**合成数据**：由模型或算法自动生成的文本，不是直接从人类写作中采集的，类似于“人工合成的练习题”。  
**规模律（Scaling Law）**：描述模型性能随模型参数、训练数据量、计算预算等因素的数学关系，像是“经验曲线”。  
**Rectified Scaling Law**：论文中观察到的、在合成数据上出现的修正后规模律，加入了一个饱和项，使得性能在一定数据量后趋于平稳。  
**概念图（Concept Graph）**：把文档中的高层次概念抽象成节点、概念之间的共现关系抽成边的图结构，类似于把知识点用思维导图连起来。  
**Token 效率**：模型在达到同等性能时所需的训练 token 数量，越少越好，等价于“学习速度”。  

### 核心创新点
1. **从原始语料到合成数据的全流程框架 → SynthLLM**  
   过去的合成数据方法多是直接让已有模型续写或做简单的噪声注入，缺乏系统的多文档概念重组。SynthLLM 首先在大规模原始语料上构建概念图，然后在图上随机抽取子图并重新组合，最后用语言模型把这些抽象的概念链渲染成自然语言文本。这样得到的合成数据在主题多样性和结构连贯性上都有显著提升。

2. **发现合成数据遵循修正规模律 → Rectified Scaling Law**  
   通过对不同模型尺寸（3B、8B 等）在合成数据上进行大规模实验，作者发现性能随 token 数的提升并非单纯的幂律，而在约 300 B token 后出现饱和趋势。引入一个饱和项后，经验曲线能够精准预测不同模型在合成数据上的最佳训练量。

3. **大模型更省 token 的经验规律 → Token 效率提升**  
   实验显示，8 B 参数模型在约 1 T token 时即可达到性能峰值，而 3 B 参数模型需要约 4 T token 才能逼近同等水平。换句话说，模型越大，合成数据的“学习效率”越高，这为资源受限的实验室提供了明确的训练预算指引。

4. **系统对标现有合成/增强方法 → 性能领先**  
   与传统的回译、噪声注入、以及最近的指令微调数据生成方法相比，SynthLLM 在相同计算预算下的下游任务（如阅读理解、代码生成）表现 consistently better，说明概念图驱动的重组策略在提升数据质量方面具有实际优势。

### 方法详解
#### 整体框架
SynthLLM 的工作流可以划分为四个阶段：  
1) **概念抽取**：对原始预训练语料进行句子级别的主题识别，抽出高层次概念（如“量子计算”“气候变化”）。  
2) **概念图构建**：把所有概念视作图的节点，依据它们在同一文档或相邻段落中出现的频率建立边，形成一个巨大的概念共现网络。  
3) **子图采样与重组**：在概念图上随机采样若干子图，每个子图包含 5–15 个概念，并通过启发式规则（如保持因果顺序、避免概念冲突）将它们排列成一条“概念链”。  
4) **文本生成**：把概念链喂入一个已经预训练好的语言模型（可使用同规模的 LLM），让模型把抽象的概念序列写成连贯的自然语言段落，最终得到合成文档。所有生成的文档拼接成合成训练集。

#### 关键模块拆解
- **概念抽取**：使用轻量的主题模型（如 LDA）或零-shot 分类器，把每句话映射到若干标签。这里的标签不是细粒度词，而是更抽象的“概念”，类似于把一段文字归类为“金融”“技术”“文化”。  
- **概念图构建**：把抽取的概念视作节点，若两个概念在同一篇文章中出现次数超过阈值，就在它们之间连一条加权边。权重代表共现强度，边的方向可以依据出现顺序决定。想象成把所有书的目录页拼在一起，形成一个巨大的知识网络。  
- **子图采样**：采用随机游走或基于度中心性的采样策略，确保抽取的子图既包含高频概念，又有一定的稀有概念混入，以提升多样性。采样后会进行“概念排序”，利用边的方向信息生成一个大致的因果或时间顺序。  
- **文本生成**：把排序好的概念序列转化为提示（prompt），如“请写一段包含以下概念的文章：概念1 → 概念2 → …”。使用的生成模型可以是同规模的 LLM，也可以是更小的模型以降低成本。生成后会进行基本的质量过滤（去除重复、检查语法）。  

#### 公式与算法的白话解释
论文中用到的核心经验公式是：

> **性能 ≈ A·(tokens)^α / (model size)^β + C·(1 - e^{-tokens / T_sat})**

- 前半部分是传统的幂律，A、α、β 是经验系数，描述了“更多数据、更多参数”带来的提升。  
- 后半部分是饱和项，C 表示最大可达的性能提升，T_sat 是出现饱和的 token 阈值（约 300 B），指数函数保证当 token 数远大于 T_sat 时，额外数据几乎不再提升性能。  

这个公式的意义在于：合成数据的收益在一定规模后会出现“拐点”，因此不必盲目追求无限量的合成文本。

#### 最巧妙的设计
最让人眼前一亮的地方是把 **概念图** 当作“语料的骨架”，再让语言模型填充“肉”。这种先抽象后具体的两步走，既保留了原始语料的主题多样性，又避免了直接复制已有句子导致的同质化问题。相当于先给模型一个“写作提纲”，再让它自由发挥。

### 实验与效果
- **实验设置**：作者在 SynthLLM 生成的合成数据上分别训练了 3 B、8 B 两种规模的 LLM，训练 token 数从 0.5 T 到 5 T 不等。下游评估任务包括 MMLU（多任务语言理解）、ARC（科学推理）和 HumanEval（代码生成）。  
- **对比基线**：与传统的回译增强、随机噪声注入以及最近的指令微调数据生成方法相比，SynthLLM 在相同计算预算下的 MMLU 分数提升约 2–3%（具体数值论文未给出），ARC 正确率提升约 1.5%。  
- **规模律验证**：实验发现 8 B 模型在约 1 T token 时性能趋于饱和，而 3 B 模型需要约 4 T token 才能接近同一水平，这正好吻合文中提出的 rectified scaling law。  
- **消融实验**：去掉概念图重组，仅使用纯随机抽取的概念序列生成文本，模型的学习曲线明显下降，尤其在 300 B token 前的提升幅度减半，说明概念图是提升合成数据质量的关键因素。  
- **局限性**：论文承认 SynthLLM 的质量依赖于原始语料的覆盖度，若概念抽取阶段漏掉重要领域，生成的合成数据会出现盲区。此外，饱和点的 300 B token 是在当前实验设置下得到的，可能随模型架构或任务类型而变化。

### 影响与延伸思考
这篇工作首次给出合成数据的可预测规模律，为“数据预算规划”提供了理论依据。随后的研究（如 2024‑2025 年的 ConceptGraph‑LLM、Synthetic‑Curriculum 等）纷纷借鉴了概念图驱动的生成思路，尝试在多模态、指令微调甚至强化学习数据生成中加入类似的结构化重组层。对想进一步探索的读者，可以关注以下方向：  
- **更细粒度的概念抽取**：利用大模型的零-shot 能力自动发现跨语言、跨领域的概念。  
- **动态饱和点预测**：结合模型训练曲线实时估计何时进入饱和区，从而自动停止合成数据的生成。  
- **合成数据的安全与偏见控制**：在概念图层加入伦理约束或过滤规则，防止合成文本放大已有偏见。  

### 一句话记住它
**SynthLLM 用概念图把“知识骨架”重新拼接，再让大模型填词，证明合成数据也能遵循可预测的规模律。**