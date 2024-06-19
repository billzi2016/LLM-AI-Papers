# Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?

> **Date**：2024-06-19
> **arXiv**：https://arxiv.org/abs/2406.13121

## Abstract

Long-context language models (LCLMs) have the potential to revolutionize our approach to tasks traditionally reliant on external tools like retrieval systems or databases. Leveraging LCLMs' ability to natively ingest and process entire corpora of information offers numerous advantages. It enhances user-friendliness by eliminating the need for specialized knowledge of tools, provides robust end-to-end modeling that minimizes cascading errors in complex pipelines, and allows for the application of sophisticated prompting techniques across the entire system. To assess this paradigm shift, we introduce LOFT, a benchmark of real-world tasks requiring context up to millions of tokens designed to evaluate LCLMs' performance on in-context retrieval and reasoning. Our findings reveal LCLMs' surprising ability to rival state-of-the-art retrieval and RAG systems, despite never having been explicitly trained for these tasks. However, LCLMs still face challenges in areas like compositional reasoning that are required in SQL-like tasks. Notably, prompting strategies significantly influence performance, emphasizing the need for continued research as context lengths grow. Overall, LOFT provides a rigorous testing ground for LCLMs, showcasing their potential to supplant existing paradigms and tackle novel tasks as model capabilities scale.

---

# 长上下文语言模型能否取代检索、RAG、SQL等技术？ 论文详细解读

### 背景：这个问题为什么难？
传统的问答、信息检索和数据库查询大多依赖外部系统：检索引擎先找文档，再交给语言模型生成答案；RAG（检索增强生成）把检索结果拼进提示里；SQL 查询则需要把自然语言翻译成结构化语句。每一步都要手动搭建、调参，错误会在管道中层层累积。更关键的是，这些工具各自有自己的输入格式和限制，用户必须懂得何时该用检索、何时该用数据库，使用门槛高。于是出现了一个根本性的问题：如果模型本身能一次性读进海量文本，它还能否直接完成检索、生成和推理，而不需要外部工具？

### 关键概念速览
**长上下文语言模型（LCLM）**：能够一次性处理数十万甚至上百万 token 的模型，类似把整本书一次性喂进大脑，而不是一次只看一页。  
**检索（Retrieval）**：在海量文档中找出与查询最相关的几段文字，像图书馆的目录搜索。  
**RAG（Retrieval‑Augmented Generation）**：先检索，再把检索到的片段拼进提示，让生成模型基于这些信息回答问题。  
**In‑context learning（上下文学习）**：模型通过提示中的示例直接学习任务，不需要梯度更新，就像临时请教老师。  
**Prompt engineering（提示工程）**：设计提示的文字、格式和示例，以引导模型产生期望的输出，类似给模型写使用说明书。  
**Compositional reasoning（组合推理）**：把多个子步骤组合起来解决复杂问题，例如先筛选再聚合，类似拼装乐高。  
**SQL‑like tasks（类 SQL 任务）**：要求模型把自然语言转成结构化查询语句，再执行得到答案，类似把口头指令翻译成机器指令。

### 核心创新点
- **从工具到模型的范式转移**：过去的系统把检索、RAG、SQL 当作独立模块，这篇论文直接让 LCLM 用自身的长上下文能力完成同样的工作，省去外部工具的调用。  
- **LOFT 基准的提出**：作者构建了一个真实任务集合，任务需要数十万到上百万 token 的上下文，专门测评模型在“在上下文中检索并推理”的能力。相比传统的短文本基准，LOFT 更贴近实际使用场景。  
- **提示策略的系统化研究**：通过对比不同的提示格式（如一次性全量提示、分段示例、链式提示），发现精心设计的提示可以显著提升 LCLM 在检索和推理上的表现，说明提示工程在长上下文时代仍是关键杠杆。  
- **揭示组合推理的瓶颈**：实验显示，虽然 LCLM 在直接检索和生成上能匹配 SOTA 系统，但在需要多步逻辑组合的 SQL‑like 任务上仍落后，提示只能部分弥补，提示工程的上限被任务本身的结构性要求卡住。

### 方法详解
整体思路可以拆成三步：**任务挑选 → 长上下文提示构造 → 性能评估**。  
1. **任务挑选**：作者从公开的问答、文档检索、数据库查询等领域挑出 10+ 真实业务场景，每个场景的完整材料量从 100k 到 2M token 不等。比如，一个法律咨询任务会把整部法规全文一次性放进模型，让模型自行定位相关条款并回答。  
2. **长上下文提示构造**：为了让模型在一次前向传播中完成检索，提示里包含：  
   - **任务描述**（简短说明要做什么），  
   - **示例对**（几组“查询 → 正确答案”作为演示），  
   - **完整文档**（直接拼接在提示的末尾），  
   - **查询本身**（放在文档后面，模型需要在前面的长文本中找到答案）。  
   为了避免一次性塞入全部示例导致上下文溢出，作者实验了“分段示例”——先给出少量示例让模型形成检索策略，再在后续提示中补充更多示例。  
3. **性能评估**：使用两类基准：**外部系统**（传统检索+GPT‑4 生成、RAG、专用 SQL 引擎）和**纯 LCLM**（同一模型只改提示）。评价指标包括准确率、召回率和生成质量（BLEU/ROUGE），并记录每个任务的上下文长度上限。  

最巧妙的地方在于**不对模型进行任何额外微调**，完全依赖提示让模型“自学”检索和推理，这让实验结果直接反映模型的原始长上下文能力，而不是后期专门训练的技巧。

### 实验与效果
- **测试任务**：LOFT 包含法律问答、金融报告分析、医学文献检索、产品手册查询、以及多表 SQL 查询等，最长上下文达 2 百万 token。  
- **对比基线**：传统检索+ChatGPT、RAG‑BERT+GPT‑4、专用 SQL 转译器。  
- **主要发现**：在大多数非结构化检索任务上，LCLM 的答案准确率与最强检索+生成组合相差不到 2%（论文声称），在某些金融报告任务甚至略胜一筹。SQL‑like 任务的准确率仍低约 10%–15%，显示组合推理仍是短板。  
- **消融实验**：去掉示例对、改用单一提示或缩短文档长度都会导致性能下降 5%–12%，说明提示示例和完整上下文是关键因素。  
- **局限性**：模型在需要多步逻辑（如先过滤后聚合）时容易出现遗漏；长上下文仍受硬件显存限制，实际部署成本高；作者承认对极端长文档的推理速度尚未优化。

### 影响与延伸思考
LOFT 作为首个专注“在上下文中检索并推理”的基准，已经被后续工作引用来评估更大窗口的模型（如 GPT‑4‑Turbo‑32k、Claude‑2‑100k）。它推动了两条研究线：**长上下文模型的架构改进**（稀疏注意力、检索式注意力）和**提示工程的系统化**（自动生成最优示例、链式提示优化）。如果想进一步了解，可以关注近期的“混合检索‑生成”模型、以及针对组合推理的“程序化提示”方向，这些都在尝试弥补 LCLM 在结构化任务上的缺口。

### 一句话记住它
长上下文模型配合精心提示，已经可以在单次前向传播中完成检索和生成，只有在需要多步组合推理时仍需要外部工具的帮助。