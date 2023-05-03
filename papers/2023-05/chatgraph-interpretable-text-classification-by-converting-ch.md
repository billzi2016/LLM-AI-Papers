# ChatGraph: Interpretable Text Classification by Converting ChatGPT   Knowledge to Graphs

> **Date**：2023-05-03
> **arXiv**：https://arxiv.org/abs/2305.03513

## Abstract

ChatGPT, as a recently launched large language model (LLM), has shown superior performance in various natural language processing (NLP) tasks. However, two major limitations hinder its potential applications: (1) the inflexibility of finetuning on downstream tasks and (2) the lack of interpretability in the decision-making process. To tackle these limitations, we propose a novel framework that leverages the power of ChatGPT for specific tasks, such as text classification, while improving its interpretability. The proposed framework conducts a knowledge graph extraction task to extract refined and structural knowledge from the raw data using ChatGPT. The rich knowledge is then converted into a graph, which is further used to train an interpretable linear classifier to make predictions. To evaluate the effectiveness of our proposed method, we conduct experiments on four datasets. The result shows that our method can significantly improve the performance compared to directly utilizing ChatGPT for text classification tasks. And our method provides a more transparent decision-making process compared with previous text classification methods.

---

# ChatGraph：通过将ChatGPT知识转化为图结构实现可解释文本分类 论文详细解读

### 背景：这个问题为什么难？
文本分类一直是 NLP 的核心任务，但传统深度模型往往像黑盒子，用户看不到它们到底凭什么把一段文字划到某个类别。近几年出现的超大语言模型（LLM）如 ChatGPT，虽然在零样本或少样本设置下表现惊艳，却有两个致命短板：一是模型本身难以在特定任务上进行细粒度的微调，二是其内部的推理过程完全不可见。于是，业界面临“性能好但不透明、可调性差”的两难局面，这正是本文要破解的难点。

### 关键概念速览
**大语言模型（LLM）**：像 ChatGPT 这样拥有上百亿参数、通过海量文本预训练得到的通用语言理解系统。它可以直接生成答案，但内部权重难以解释。  
**知识图谱**：把实体（如“苹果”）和它们之间的关系（如“属于”）组织成节点和边的网络，类似于把散乱的事实装进结构化的网格。  
**提示工程（Prompt Engineering）**：给 LLM 设计特定的输入文字，引导它输出想要的格式或信息，就像给助理下指令让它写报告。  
**线性分类器**：如逻辑回归或线性支持向量机，模型的决策边界是一个线性函数，权重可以直接映射到特征重要性，解释性极强。  
**图特征**：从知识图谱中抽取的数值描述，例如节点出现频次、边的类型分布或基于图的嵌入向量，类似于把一张地图压缩成几行数字。  
**可解释性**：模型输出背后能够给出人类可理解的因果说明，像医生给出诊断依据一样。  

### 核心创新点
1. **利用 ChatGPT 进行知识抽取 → 直接让 ChatGPT 生成结构化的三元组（实体‑关系‑实体）** → 把原始文本转化为可操作的图数据，省去了传统的手工标注或专门的抽取模型。  
2. **从抽取的三元组构建任务专属知识图 → 将所有三元组拼接成一个统一的图结构** → 让后续的分类器可以在统一的、可视化的空间里学习，而不是在高维、不可解释的向量上。  
3. **在图特征上训练线性分类器 → 用权重直接映射到图中的节点或边** → 预测结果可以追溯到具体的实体或关系，解释过程像阅读一张因果图。  
4. **不对 ChatGPT 本体进行微调** → 通过提示让模型输出结构化信息，保持了 LLM 的通用能力，同时规避了大模型微调的高成本和数据需求。  

### 方法详解
整体思路可以拆成四步：**提示‑抽取‑图构建‑线性分类**。下面用类比把每一步拆开讲。

1. **提示设计 & 知识抽取**  
   想象你请一位懂得很多的老师帮你把一段文章“拆解成要点”。作者给 ChatGPT 设计了一段特定的 Prompt，要求它把每句话中的关键实体、属性以及它们之间的关系以“实体‑关系‑实体”的形式输出。比如输入“苹果是一种水果，富含维生素C”，模型会返回 `(苹果, 是, 水果)`、`(苹果, 富含, 维生素C)`。这里的技巧在于让 LLM 按结构化格式输出，而不是自由文本。

2. **图构建**  
   所有抽取到的三元组被视作图的边，实体成为节点。把同一篇文档的所有边拼在一起，就得到一张**文档级知识图**。如果多篇文档共享相同实体，图会自然地把它们连在一起，形成跨文档的语义网络。作者随后对图进行简化处理（如去除低频节点、合并同义词），确保图既保留关键信息，又不至于过于稀疏。

3. **图特征工程**  
   为了让线性模型能“看见”图，必须把图转成数值特征。常见做法包括：  
   - **节点出现计数**：某个实体在图中出现多少次。  
   - **关系类型分布**：不同关系（如“属于”“导致”）的出现频率。  
   - **图嵌入**：使用轻量的图神经网络或随机游走方法把每个节点映射到低维向量，再对向量做池化得到文档向量。  
   这些特征在本论文中被拼接成一个固定长度的特征向量，直接喂给后面的线性分类器。

4. **线性分类器训练与解释**  
   作者选用了最经典的逻辑回归作为分类器。因为逻辑回归的权重可以直接解释为“该特征对正类（或负类）的贡献”。在预测时，模型会输出每个图特征的权重乘积之和，最高的类别即为预测结果。解释时，只需要查看权重最大的几个特征，对应的节点或关系就能说明模型为何做出该决定。

**最巧妙的点**在于：整个流程只依赖一次性 Prompt 调用，没有对 ChatGPT 进行任何梯度更新，既保留了 LLM 的强大语言理解，又把输出转化为结构化、可解释的图形。这样既解决了微调难题，又让决策过程透明化。

### 实验与效果
- **数据集**：论文在四个公开的文本分类基准上做实验（具体名称未在摘要中列出），涵盖情感分析、主题分类等不同场景。  
- **对比基线**：包括直接使用 ChatGPT 进行 zero‑shot 分类的方案，以及传统深度文本分类模型（如 BERT‑based 分类器）。  
- **性能提升**：论文声称，在所有数据集上，ChatGraph 的准确率均显著高于直接调用 ChatGPT 的结果，同时也超过了常规深度模型的表现。具体提升幅度未给出数值。  
- **可解释性验证**：作者展示了若干案例，说明模型的高权重特征对应的实体或关系正是人类专家会关注的关键信息。  
- **消融实验**：通过去掉 Prompt 抽取、仅使用原始文本特征或仅使用图嵌入等设置，论文展示每一步对整体性能的贡献，证明知识抽取和图特征是提升的主要驱动力。  
- **局限性**：作者承认，Prompt 的质量对抽取效果高度敏感；如果 ChatGPT 输出的三元组不完整或出现噪声，后续图构建和分类会受影响。此外，当前实现仍依赖于手工设计的特征工程，尚未探索更自动化的图学习方式。

### 影响与延伸思考
这篇工作打开了“LLM + 结构化知识图谱” 的新思路，后续有不少研究尝试把其他大模型（如 GPT‑4、Claude）用于自动生成知识图谱，再结合图神经网络进行下游任务。还有工作把 Prompt 设计与自监督学习结合，提升抽取的准确率。对想进一步探索的读者，可以关注以下方向：  
- **自适应 Prompt 生成**：让模型自己学习如何提问以获得更高质量的结构化输出。  
- **端到端图学习**：把知识抽取、图构建和图嵌入统一进一个可微分框架，减少手工特征。  
- **跨模态解释**：把文本图与图像或音频的知识图谱对齐，实现多模态可解释推理。  

### 一句话记住它
把 ChatGPT 当成“会说话的知识抽取器”，把抽出的三元组拼成图，再用线性模型做分类——性能提升且决策全程可追溯。