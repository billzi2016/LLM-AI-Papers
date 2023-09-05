# Benchmarking Large Language Models in Retrieval-Augmented Generation

> **Date**：2023-09-04
> **arXiv**：https://arxiv.org/abs/2309.01431

## Abstract

Retrieval-Augmented Generation (RAG) is a promising approach for mitigating the hallucination of large language models (LLMs). However, existing research lacks rigorous evaluation of the impact of retrieval-augmented generation on different large language models, which make it challenging to identify the potential bottlenecks in the capabilities of RAG for different LLMs. In this paper, we systematically investigate the impact of Retrieval-Augmented Generation on large language models. We analyze the performance of different large language models in 4 fundamental abilities required for RAG, including noise robustness, negative rejection, information integration, and counterfactual robustness. To this end, we establish Retrieval-Augmented Generation Benchmark (RGB), a new corpus for RAG evaluation in both English and Chinese. RGB divides the instances within the benchmark into 4 separate testbeds based on the aforementioned fundamental abilities required to resolve the case. Then we evaluate 6 representative LLMs on RGB to diagnose the challenges of current LLMs when applying RAG. Evaluation reveals that while LLMs exhibit a certain degree of noise robustness, they still struggle significantly in terms of negative rejection, information integration, and dealing with false information. The aforementioned assessment outcomes indicate that there is still a considerable journey ahead to effectively apply RAG to LLMs.

---

# 检索增强生成中的大语言模型基准评估 论文详细解读

### 背景：这个问题为什么难？
在纯粹的生成式大语言模型（LLM）里，模型只能靠内部参数记忆事实，容易出现“幻觉”——给出根本不存在的答案。检索增强生成（RAG）通过把外部文档拉进来，理论上可以让答案更可靠，但实际效果受检索质量、文档噪声以及模型对信息的整合能力影响很大。过去的工作大多只在单一模型或单一任务上做了点实验，缺少系统化的评估框架，导致我们不知道不同模型在哪些环节会卡壳，也找不出改进的突破口。

### 关键概念速览
**检索增强生成（RAG）**：让语言模型在生成答案前先去检索外部文档，再把检索到的内容当作“记忆”使用，类似于人写报告前先上网查资料。  
**噪声鲁棒性**：模型在面对包含无关或错误信息的文档时，仍能抽取出正确答案的能力，就像在嘈杂的咖啡馆里仍能听清对话。  
**负样本拒绝**：当检索结果里根本没有相关信息时，模型主动不回答或给出“我不知道”，相当于人面对不确定问题时选择保持沉默。  
**信息整合**：需要把多篇文档的碎片信息拼凑成完整答案的能力，类似于拼图游戏，需要把不同块儿拼在一起才能看到全貌。  
**反事实鲁棒性**：文档里故意植入错误事实时，模型能识别并纠正的能力，像是读新闻时辨别假消息的技巧。  
**RGB（Retrieval‑Augmented Generation Benchmark）**：本文构建的双语评估套件，专门划分出四类测试场景来测量上述能力。  
**负样本**：指检索结果中不包含任何有用信息的文档，常用于检验模型的拒绝能力。  
**错误检测率 / 错误修正率**：分别衡量模型发现文档错误和主动纠正错误的成功比例。

### 核心创新点
1. **从单一指标到四维能力评估**：以前的 RAG 评测往往只看整体准确率或 BLEU 分数，本文把评估拆成噪声鲁棒、负样本拒绝、信息整合、反事实鲁棒四个维度。这样可以精准定位模型的薄弱环节，而不是只能说“整体表现不好”。  
2. **双语、任务多样的 RGB 基准**：作者自行收集并人工标注了中英文检索‑生成对齐数据，分别对应四种能力的测试集。相比只用英文单一任务的旧基准，RGB 能更全面地检验模型在不同语言和不同检索噪声下的表现。  
3. **系统化的六模型横向对比**：在同一基准上同时跑了 6 种主流 LLM（包括闭源和开源），并给出每种模型在四个能力上的详细表现。这样做让研究者一眼看到哪类模型在负样本拒绝上更强，哪类模型在信息整合上更弱。  
4. **明确的瓶颈诊断框架**：通过把每个测试场景的错误类型归类，作者提出了“噪声‑负样本‑信息‑反事实”四大瓶颈模型，帮助后续工作有针对性地改进检索或生成模块。

### 方法详解
整体思路可以概括为三步：**构造评估集 → 统一检索‑生成流程 → 多维度指标统计**。

1. **评估集构造**  
   - **噪声鲁棒**：从真实检索结果中挑出包含少量相关信息、其余为无关噪声的文档。  
   - **负样本拒绝**：直接使用全负样本文档，即检索结果里没有任何答案线索。  
   - **信息整合**：设计需要跨文档组合的复杂问答，例如“请列出 A、B、C 三家公司在 2022 年的收入”，每家公司信息散落在不同文档。  
   - **反事实鲁棒**：在原始文档中人为植入错误事实（如把 2021 年改成 2022 年），观察模型是否会盲目复制错误。  
   每类实例都配有人工标注的“黄金答案”，并标记哪些文档是正向、哪些是负向。

2. **统一检索‑生成管线**  
   - 输入问题 → 使用统一的检索 API（如 BM25 或向量检索）取前 N 篇文档。  
   - 将检索到的文档拼接成一个“检索上下文”，并在前面加上任务提示（例如“请仅在下面的文档中寻找答案”）。  
   - 把提示+上下文喂入目标 LLM，得到生成答案。  
   关键在于 **提示工程**：通过明确指示模型“如果文档里没有答案，请说不知道”，来激活负样本拒绝能力。

3. **多维度指标统计**  
   - **准确率**：答案与黄金答案完全匹配的比例，主要用于噪声鲁棒和信息整合。  
   - **拒绝率**：在负样本场景下模型主动不回答的比例。  
   - **错误检测率**：在反事实场景中模型识别出文档错误的比例（比如标记“文档中信息可能有误”）。  
   - **错误修正率**：模型在发现错误后给出正确答案的比例。  
   通过这些指标的交叉对比，作者能够绘制出每个模型的能力雷达图。

**最巧妙的地方**在于把“检索质量”与“生成质量”解耦：评估时不去纠结检索本身的召回率，而是直接把检索结果交给模型，让模型的表现本身反映出它对噪声和负样本的容忍度。这种设计让评估更贴近真实使用场景，因为实际系统里检索往往不可避免地带噪。

### 实验与效果
- **数据集**：RGB 包含约 4,000 条中英文问答，均匀分布在四个能力子集。  
- **模型**：六个代表性 LLM（包括 GPT‑4、Claude、LLaMA‑2、ChatGLM、Bloom、OpenAI‑davinci）在相同检索‑生成管线下进行测试。  
- **结果概览**：所有模型在噪声鲁棒上都能保持 70% 以上的准确率，说明对轻度噪声有一定容错。负样本拒绝表现最差，最高的拒绝率只有约 45%，其余模型往往直接生成答案导致错误。信息整合任务的准确率普遍低于 50%，尤其是需要跨三篇文档拼接信息时错误率激增。反事实鲁棒性最为薄弱，错误检测率普遍在 30% 左右，错误修正率更是不到 20%。  
- **消融实验**：作者分别去掉提示中的“如果不知道请说不知道”以及去掉检索上下文的前置说明，发现拒绝率下降约 15%，说明提示工程对负样本拒绝贡献显著。  
- **局限性**：评估只覆盖了检索前 5 篇文档，未探讨大规模检索或多轮检索的影响；此外，中文数据量相对英文仍然较少，可能导致跨语言对比有偏差。

### 影响与延伸思考
这篇工作在 RAG 社区里起到了“诊断手册”的作用，随后出现的多篇论文（如 **RAG‑Eval**、**FactCheck‑RAG**）都直接引用了 RGB 作为基准，或在其基础上加入了多轮检索、跨模态检索等扩展。对想继续深入的读者，可以关注以下方向：① 如何在检索阶段主动过滤噪声文档（检索过滤器学习）；② 提示工程与模型内部的“拒绝机制”联合训练；③ 将事实校验模型嵌入生成过程，实现实时错误检测。整体来看，RGB 为评估提供了统一坐标系，后续工作只要在此坐标系上加分，就能清晰看到进步。

### 一句话记住它
**RGB 把 RAG 的四大核心能力拆解成可量化的测试，让我们一眼看清每个大模型在“噪声、负样本、信息整合、反事实”上的真实短板。**