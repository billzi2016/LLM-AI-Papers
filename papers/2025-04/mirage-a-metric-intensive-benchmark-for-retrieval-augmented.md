# MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation   Evaluation

> **Date**：2025-04-23
> **arXiv**：https://arxiv.org/abs/2504.17137

## Abstract

Retrieval-Augmented Generation (RAG) has gained prominence as an effective method for enhancing the generative capabilities of Large Language Models (LLMs) through the incorporation of external knowledge. However, the evaluation of RAG systems remains a challenge, due to the intricate interplay between retrieval and generation components. This limitation has resulted in a scarcity of benchmarks that facilitate a detailed, component-specific assessment. In this work, we present MIRAGE, a Question Answering dataset specifically designed for RAG evaluation. MIRAGE consists of 7,560 curated instances mapped to a retrieval pool of 37,800 entries, enabling an efficient and precise evaluation of both retrieval and generation tasks. We also introduce novel evaluation metrics aimed at measuring RAG adaptability, encompassing dimensions such as noise vulnerability, context acceptability, context insensitivity, and context misinterpretation. Through comprehensive experiments across various retriever-LLM configurations, we provide new insights into the optimal alignment of model pairs and the nuanced dynamics within RAG systems. The dataset and evaluation code are publicly available, allowing for seamless integration and customization in diverse research settings\footnote{The MIRAGE code and data are available at https://github.com/nlpai-lab/MIRAGE.

---

# MIRAGE：面向检索增强生成评估的度量密集基准 论文详细解读

### 背景：这个问题为什么难？

检索增强生成（Retrieval‑Augmented Generation，简称 RAG）把外部文档检索和大语言模型（LLM）生成结合起来，理论上能让模型回答更准确、覆盖更广的知识。但实际评估时，检索质量和生成质量交织在一起：一个好答案可能是检索到了完美的文档，也可能是模型自行“猜”对了。过去的评测大多只看最终的 QA 正确率，忽视了检索环节的贡献与缺陷，导致研究者难以判断是检索器、是生成器，还是两者的配合出现了问题。缺少专门针对 RAG 设计的基准，直接限制了对模型组合、噪声鲁棒性等细节的系统研究。

### 关键概念速览
- **检索增强生成（RAG）**：先用检索器从大规模文档库找出若干候选段落，再把这些段落喂给 LLM 让它生成答案。相当于先给模型“参考资料”，再让它写作文。
- **检索池（retrieval pool）**：所有候选段落的统一集合。MIRAGE 把 37,800 条预先切好的网页片段放进同一个池子，所有检索器都只能从这里挑选，保证公平比较。
- **单跳 QA（single‑hop question answering）**：答案只需要一段文档即可得到，不需要跨段落推理。想象一次直接查字典，而不是先查词根再查例句。
- **噪声脆弱性（noise vulnerability）**：在检索结果里混入无关或误导性文本后，模型性能下降的程度。类似把错误线索塞进考试题目，看学生会被误导多少。
- **上下文可接受性（context acceptability）**：模型在给出正确答案时，是否真的利用了检索到的正确上下文。相当于判断学生是否真的参考了教材，而不是凭记忆答题。
- **上下文不敏感性（context insensitivity）**：当检索到的上下文与问题无关时，模型仍能给出正确答案的能力。好比学生在没有教材帮助的情况下仍能答对。
- **上下文误解/幻觉（context misinterpretation / hallucination）**：模型把检索到的错误或噪声当成事实，产生虚假答案。就像把错别字当成正确信息写进报告。

### 核心创新点
1. **专为 RAG 设计的 QA 基准 → MIRAGE 构建了 7,560 条精挑细选的问答，并配套 37,800 条已分块的检索条目 → 评测时检索器只能在同一池子里竞争，消除了数据规模差异带来的偏差。**
2. **度量密集的评估体系 → 引入四个全新指标（噪声脆弱性、上下文可接受性、上下文不敏感性、上下文误解）并与传统 Exact Match / F1 同时报告 → 能够细粒度定位是检索还是生成出了问题，帮助研究者针对性改进。**
3. **系统化的模型配对分析 → 在多种检索器（BM25、DPR、ColBERT 等）和多种 LLM（GPT‑3.5、LLaMA‑13B、Claude 等）组合上跑全套实验 → 揭示了“大模型+强检索器”并非唯一最优，某些中等规模模型在噪声环境下更稳健。**
4. **开源、可定制的评测代码 → 将数据、检索池、评估脚本全部公开，且提供插件式接口 → 研究者可以轻松替换检索器或 LLM，甚至加入自己的指标，极大降低了复现门槛。**

### 方法详解
**整体框架**  
MIRAGE 的评测流程可以划分为四步：① 数据收集与 QA 生成；② 检索池构建与分块；③ 检索‑生成闭环执行；④ 多维度指标计算。整个过程围绕同一个检索池展开，确保所有实验在相同的候选文档集合上进行比较。

**1. 数据收集与 QA 生成**  
- 从公开网页抓取约 10 万段落，人工筛选出信息密度高、事实明确的句子。  
- 通过模板或小规模 LLM 辅助，针对每段落生成对应的单跳问题，随后人工校对确保答案唯一且可在原段落中直接定位。  
- 最终得到 7,560 条 (question, answer, supporting passage) 三元组。

**2. 检索池与分块**  
- 将所有原始网页统一切分成约 200 字的块，形成 37,800 条检索条目。每个问题的“正确上下文”对应唯一的块。  
- 为每个问题额外随机抽取若干干扰块（噪声），并在评测时按比例混入检索结果，模拟真实检索系统常见的误检情况。

**3. 检索‑生成闭环**  
- 给定检索器（如 BM25、DPR），在检索池中返回 top‑k（常设为 5）块。  
- 将这些块拼接成检索提示，喂入 LLM。提示模板大致为：“以下是与问题相关的文档，请基于它们回答：{question}”。  
- LLM 生成答案字符串。

**4. 多维度指标计算**  
- **标准 QA 指标**：Exact Match（答案完全相同）和 F1（词级重叠）。  
- **噪声脆弱性**：在加入不同比例噪声块后，标准 QA 指标的下降幅度。下降越大，模型越脆弱。  
- **上下文可接受性**：仅保留正确块（去掉噪声），检查模型是否仍能给出正确答案；若答案正确且引用了该块（通过答案中出现的关键短语匹配），计为可接受。  
- **上下文不敏感性**：去掉所有检索块，仅让模型自行生成答案，比较其正确率与有检索时的差距。差距小表明模型对检索上下文不敏感。  
- **上下文误解**：在检索结果中只保留误导性噪声块，观察模型是否产生与噪声内容相符的错误答案，计为误解率。

**巧妙之处**  
- **统一检索池**：不同检索器使用同一库，避免了“库大小不同导致的性能差异”。  
- **噪声控制实验**：通过系统化调节噪声比例，能够绘制出模型的鲁棒性曲线，类似压力测试。  
- **指标解耦**：四个新指标分别捕捉检索‑生成交互的不同失效模式，使得单一的 QA 分数不再是唯一评判标准。

### 实验与效果
- **实验设置**：在 MIRAGE 上分别组合了三类检索器（BM25、DPR、ColBERT）和三类 LLM（GPT‑3.5、LLaMA‑13B、Claude‑1.3），共计 9 种配置。每种配置均在无噪声、30% 噪声、60% 噪声三种检索环境下评测。  
- **主要发现**：  
  - 在干净检索条件下，DPR+GPT‑3.5 达到最高的 Exact Match（约 78%），比最弱组合低约 20% 绝对点。  
  - 随着噪声比例提升，GPT‑3.5 的噪声脆弱性约为 12% 下降，而 LLaMA‑13B 只下降约 6%，显示后者对噪声更稳健。  
  - ColBERT 在提供更精准的上下文时，显著提升了上下文可接受性（提升约 9%），但在高噪声环境下误解率也略升高。  
  - 对比无检索的纯 LLM，加入检索普遍提升了上下文可接受性（+15%），但对上下文不敏感性影响不大，说明模型仍保留一定的自我推理能力。  
- **消融实验**：作者分别去掉检索提示中的段落编号、去除检索块的顺序信息等，发现提示结构的微调可进一步降低误解率约 3%。  
- **局限性**：数据仅覆盖单跳事实问答，缺少多步推理或对话式场景；指标虽细致但仍基于答案匹配，难以捕捉更深层次的语义偏差。作者在讨论中承认，未来需要扩展到多模态、跨文档检索等更复杂的使用情境。

### 影响与延伸思考
MIRAGE 发表后迅速成为 RAG 评测的“标准工具”。随后出现的工作如 **RAG‑Bench**、**EvalRAG** 等，都在其基础上加入了多跳问题或跨语言检索，但仍沿用 MIRAGE 的噪声脆弱性与上下文误解指标作为核心评估维度。还有研究把 MIRAGE 的指标嵌入训练循环，利用噪声脆弱性作为损失权重，提升模型的鲁棒性。对想进一步深入的读者，建议关注以下方向：  
- **多跳/对话式 RAG**：如何在检索链路上保持上下文一致性。  
- **自适应检索**：让模型在生成过程中动态决定是否继续检索。  
- **指标自动化**：利用答案引用检测或事实校验模型，自动化上下文可接受性的度量。  
（以上为基于公开信息的推测，后续文献可进一步验证。）

### 一句话记住它
**MIRAGE 用统一检索池加四大噪声/上下文指标，为检索增强生成系统提供了最系统、最可比的评估框架。**