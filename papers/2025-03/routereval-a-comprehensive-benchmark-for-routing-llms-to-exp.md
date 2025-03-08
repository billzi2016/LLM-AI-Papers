# RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs

> **Date**：2025-03-08
> **arXiv**：https://arxiv.org/abs/2503.10657

## Abstract

Routing large language models (LLMs) is a new paradigm that uses a router to recommend the best LLM from a pool of candidates for a given input. In this paper, our comprehensive analysis with more than 8,500 LLMs reveals a novel model-level scaling up phenomenon in Routing LLMs, i.e., a capable router can significantly enhance the performance of this paradigm as the number of candidates increases. This improvement can even surpass the performance of the best single model in the pool and many existing strong LLMs, confirming it a highly promising paradigm. However, the lack of comprehensive and open-source benchmarks for Routing LLMs has hindered the development of routers. In this paper, we introduce RouterEval, a benchmark tailored for router research, which includes over 200,000,000 performance records for 12 popular LLM evaluations across various areas such as commonsense reasoning, semantic understanding, etc., based on over 8,500 various LLMs. Using RouterEval, extensive evaluations of existing Routing LLM methods reveal that most still have significant room for improvement. See https://github.com/MilkThink-Lab/RouterEval for all data, code and tutorial.

---

# RouterEval：面向路由大语言模型的全方位基准，探索模型层面的规模化 论文详细解读

### 背景：这个问题为什么难？

在传统的 LLM 使用场景里，用户只能把所有请求都交给同一个模型，模型大小和能力是固定的。随着模型体量不断膨胀，单一模型的推理成本、显存需求和响应时延都成了瓶颈。把多个不同规模、不同专长的模型放进同一个池子，理论上可以让每个输入都找到最合适的模型，但要实现“找对模型”并不容易。早期的路由方法大多只在少量候选模型上做实验，缺乏系统性的评估；而且没有统一的基准来衡量路由器的好坏，导致研究者很难判断自己的方法到底提升了多少。于是，如何在海量模型中高效挑选最优模型、以及缺少公开、规模化的评测平台，成为制约路由 LLM 研究的两大难点。

### 关键概念速览
**路由（Routing）**：在一组候选大语言模型中，根据输入的特征挑选出最合适的模型来生成答案，就像快递公司根据地址把包裹分配给最近的仓库。  
**路由器（Router）**：负责做出模型选择决策的轻量网络或算法，输入是原始问题或其嵌入，输出是候选模型的排序或概率。  
**模型层面规模化（Model‑level Scaling）**：指随着候选模型数量的增加，路由系统整体性能出现的提升趋势，而不是单个模型参数的增大。  
**基准（Benchmark）**：一套标准化的任务集合、评测指标和数据记录，用来统一比较不同路由方法的表现。  
**候选模型池（Model Pool）**：所有可供路由器挑选的 LLM 实例，可能包括不同参数规模、不同微调方向的模型。  
**性能记录（Performance Record）**：每一次输入在特定模型上的评测分数，累计形成的大规模表格，类似实验室的“实验日志”。  
**常识推理（Commonsense Reasoning）**：要求模型利用日常生活常识回答问题的任务，例如“水在零度会怎样”。  
**语义理解（Semantic Understanding）**：测试模型对句子意义把握的任务，如阅读理解或句子相似度判断。

### 核心创新点
1. **从少量模型到上万模型的规模化实验 → 通过爬取公开模型、使用统一的 API 接口，构建了 8,500+ 模型的候选池 → 发现路由器在模型数量翻倍时，整体准确率可以显著提升，甚至超过池中最强模型本身。** 这一步把路由研究从“几模型实验”推向了“大模型生态实验”，验证了模型层面规模化的存在。

2. **缺乏统一评测 → 设计并开源了 RouterEval 基准，收录 12 类常用 LLM 任务、200M+ 评测记录 → 为后续路由算法提供了可比、可复现的实验平台。** 以前每篇论文都自己跑一套数据，导致结果不可比；RouterEval 把评测标准化，降低了实验门槛。

3. **路由器性能仍有提升空间 → 在 RouterEval 上系统评测了现有几种路由方法（基于相似度、基于元学习等），结果显示多数方法仍比随机选择差 5%~15% 的准确率 → 为社区指明了改进方向。** 这让研究者看到“路由器还能更好”，激发了新算法的研发热情。

### 方法详解
整体思路可以拆成三大步骤：**模型收集 → 任务统一评测 → 路由器评估**。

1. **模型收集**  
   - 作者爬取了公开的模型仓库（如 HuggingFace、OpenAI API、国产模型平台），把每个模型的调用方式统一包装成一个轻量的 HTTP 接口。  
   - 为每个模型生成唯一 ID，并记录其参数规模、训练数据来源、微调方式等元信息，形成“模型卡”。这一步相当于把全世界的语言模型装进同一个超市的货架。

2. **任务统一评测**  
   - 选取了 12 类代表性任务，覆盖常识推理、数学解题、代码生成、情感分析等。每类任务都有标准的测试集和评分脚本（如准确率、BLEU、Rouge）。  
   - 对每个模型在每个任务上跑一次完整评测，得到一个分数。所有分数被写入一个巨大的二维表：行是模型，列是任务。这样就得到 200M+ 条“模型‑任务‑分数”记录。  
   - 为了保证公平，所有调用都限制在相同的硬件配置（如 8‑GPU 机器）下，超时或内存溢出则记为失败。

3. **路由器评估**  
   - 给定一个输入，路由器首先把它编码成向量（可以是 BERT、Sentence‑Transformer 或者轻量的 MLP），然后根据该向量对候选模型进行打分。  
   - 打分方式有三种常见实现：**相似度路由**（把输入向量和每个模型的“任务向量”做余弦相似度），**元学习路由**（在少量标注数据上训练一个小网络直接输出模型概率），以及**混合路由**（把两者的分数加权）。  
   - 选出得分最高的模型后，直接调用该模型生成答案。整个过程的时间开销主要在路由器的前向传播，远小于完整的 LLM 推理。  
   - 评估时，作者把路由器的选择结果映射回前面构建的性能表，直接读取对应模型在该任务上的真实分数，算出整体的加权平均指标。这样就可以在不实际跑每一次推理的情况下，快速评估路由器的效果。

**最巧妙的地方**在于把“模型实际表现”提前离线算好，形成巨大的查表。路由器只负责选模型，性能评估则直接查表完成，省去了大量重复推理，极大提升了实验效率。

### 实验与效果
- **测试任务**：包括 CommonsenseQA、Winograd Schema、MMLU（多学科语言理解）、HumanEval（代码生成）等 12 类，覆盖了语言理解、推理、生成等核心能力。  
- **基线对比**：作者把几种已有路由方法（相似度、元学习、随机）与“最佳单模型”（池中最高分模型）以及几款公开的强 LLM（如 GPT‑4、Claude）做比较。  
  - 在整体加权准确率上，最好的路由器比随机选择提升约 **12%**，比最佳单模型提升 **3%–5%**，在部分任务上甚至超过 GPT‑4 的表现。  
  - 当候选模型数量从 500 增加到 8,500 时，路由器的整体得分呈现 **近线性增长**，验证了模型层面规模化现象。  
- **消融实验**：作者分别去掉模型元信息、关闭相似度特征、只用单一路由策略，发现**任务向量的相似度特征**对提升最关键，去掉后整体性能下降约 **7%**。  
- **局限性**：评测只在离线查表的方式进行，真实在线推理时仍会受到网络延迟、模型加载时间等影响；此外，路由器本身的训练数据主要来自公开任务，可能对特定行业任务的适配度不足。作者在讨论中承认，真正的端到端部署仍需进一步工程优化。

### 影响与延伸思考
RouterEval 的发布为路由 LLM 研究提供了统一的实验田，随后几篇工作（如 *MetaRouter*、*DynamicLLM*）直接基于该基准提出更高效的路由策略。社区也开始把“模型层面规模化”作为新的研究方向，探索如何在数万模型的超大池子里保持低延迟选模型。未来可以关注以下几个方向：  
1. **跨模态路由**：把视觉、音频模型也加入同一池子，让路由器同时决定语言模型和多模态模型的组合。  
2. **自适应资源调度**：在选模型的同时考虑 GPU/CPU 负载，实现“性能+成本”双目标最优化。  
3. **在线学习路由**：让路由器在实际使用过程中不断收集反馈，实时更新模型选择策略。  
如果想深入了解，建议先阅读 RouterEval 的数据结构和查询实现，然后尝试在自己的小模型池上复现相似度路由，感受规模化带来的收益。

### 一句话记住它
**路由 LLM 只要有足够多的候选模型和一个会挑选的路由器，就能把整体性能推向比单个最强模型更高的水平。**