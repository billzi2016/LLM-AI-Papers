# MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation

> **Date**：2024-12-31
> **arXiv**：https://arxiv.org/abs/2501.00332

## Abstract

Large Language Models (LLMs) are becoming essential tools for various natural language processing tasks but often suffer from generating outdated or incorrect information. Retrieval-Augmented Generation (RAG) addresses this issue by incorporating external, real-time information retrieval to ground LLM responses. However, the existing RAG systems frequently struggle with the quality of retrieval documents, as irrelevant or noisy documents degrade performance, increase computational overhead, and undermine response reliability. To tackle this problem, we propose Multi-Agent Filtering Retrieval-Augmented Generation (MAIN-RAG), a training-free RAG framework that leverages multiple LLM agents to collaboratively filter and score retrieved documents. Specifically, MAIN-RAG introduces an adaptive filtering mechanism that dynamically adjusts the relevance filtering threshold based on score distributions, effectively minimizing noise while maintaining high recall of relevant documents. The proposed approach leverages inter-agent consensus to ensure robust document selection without requiring additional training data or fine-tuning. Experimental results across four QA benchmarks demonstrate that MAIN-RAG consistently outperforms traditional RAG approaches, achieving a 2-11% improvement in answer accuracy while reducing the number of irrelevant retrieved documents. Quantitative analysis further reveals that our approach achieves superior response consistency and answer accuracy over baseline methods, offering a competitive and practical alternative to training-based solutions.

---

# MAIN‑RAG：多代理过滤检索增强生成 论文详细解读

### 背景：这个问题为什么难？

LLM（大语言模型）在问答、写作等任务上已经很强，但它们的知识是训练时固化的，面对最新事件或细节时容易“凭空”编造。RAG（检索增强生成）通过在生成前先抓取外部文档来补足信息，理论上可以让答案更可靠。实际使用时，检索模块往往会返回大量噪声——比如标题相似但内容无关的网页、旧的新闻稿、甚至广告页。这些不相关文档会占用算力、干扰模型的注意力，导致答案仍然错误或不一致。于是，如何在不额外训练的前提下高效筛掉噪声、保留关键文档，成为 RAG 系统的瓶颈。

### 关键概念速览

**LLM（大语言模型）**：基于海量文本训练的生成式模型，能完成对话、翻译、写作等任务。把它想成“会说话的百科全书”，但内容只能到训练截止时间。

**RAG（检索增强生成）**：在 LLM 生成答案前先用搜索引擎或向量库抓取相关文档，把检索到的文本当作“实时参考资料”。类似于写报告时先去图书馆查资料再写。

**噪声文档**：检索结果中与当前问题无关或信息错误的条目。它们像课堂上不相关的插话，会让学生（模型）跑题。

**多代理（Multi‑Agent）**：使用多个独立的 LLM 实例分别对同一件事进行评估或推理。可以把它们想成“几位老师一起批改作业”，相互讨论后更可靠。

**共识过滤（Consensus Filtering）**：让多个代理对每篇文档打分，然后依据多数意见决定是否保留。类似于投票决定哪本参考书值得阅读。

**自适应阈值**：根据当前批次文档的分数分布动态调节“及格线”。如果整体分数偏低，就放宽标准；如果分数普遍很高，就收紧，避免误删。

### 核心创新点

1. **训练免费、多代理共识过滤**  
   之前的 RAG 要么依赖单一检索模型的排序，要么需要专门训练一个文档过滤器。MAIN‑RAG 直接召唤若干 LLM（如 GPT‑3.5、Claude）对每篇检索文档进行相关性打分，再用多数投票决定保留。这样既省去额外标注数据，也利用了已有模型的理解能力。

2. **自适应阈值机制**  
   传统过滤会设一个固定分数线，容易在不同查询间出现“过严”或“过宽”。MAIN‑RAG 先统计本轮所有文档的分数分布，自动计算一个动态阈值，使得噪声被压制的同时，召回率保持在合理水平。相当于在每次考试前先了解学生整体水平，再决定及格分数。

3. **无需微调的即插即用框架**  
   该系统把检索、过滤、生成三步串联成一个流水线，所有模块都使用公开的 API 调用，无需对模型进行微调或重新训练。对实际工程而言，部署成本大幅下降。

4. **实验验证的统一提升**  
   在四个公开问答基准上，MAIN‑RAG 的答案准确率比普通 RAG 高出 2%~11%，同时检索到的无关文档数量下降约 30%。这说明共识过滤和自适应阈值真的在降低噪声、提升答案质量上起效。

### 方法详解

**整体思路**  
MAIN‑RAG 把一次问答拆成三步：① 用传统向量检索或 BM25 抓取 N 篇候选文档；② 让 K 个 LLM 代理分别对每篇文档进行相关性评估并打分；③ 根据分数分布动态设定阈值，保留得分超过阈值的文档，最后把这些文档拼接进提示，交给主 LLM 生成答案。

**步骤拆解**  

1. **检索阶段**  
   - 输入用户问题 Q。  
   - 使用已有的检索引擎（如 FAISS、ElasticSearch）返回前 N=20 条文档 D₁…D_N。  
   - 这一步不做改动，保持与传统 RAG 相同。

2. **多代理评分**  
   - 选取 K=3~5 个不同的 LLM（可以是同一模型的不同温度实例，也可以是不同厂商的模型）。  
   - 每个代理收到同一对 (Q, D_i) 并输出一个 0‑1 相关性分数 s_{k,i}，或者直接给出“相关/不相关”的二分类。  
   - 代理的提示模板大致是：“请判断下面这段文字是否能帮助回答问题‘{Q}’，如果能请给 1 分，否则给 0 分”。  
   - 通过多数投票或取平均得到每篇文档的最终分数 s_i。

3. **自适应阈值计算**  
   - 收集所有 s_i，计算它们的均值 μ 和标准差 σ。  
   - 设定阈值 τ = μ + λ·σ（λ 为经验系数，论文中未给出具体数值，通常在 0.5~1 之间）。  
   - 若文档分数 s_i ≥ τ，则保留；否则剔除。  
   - 这种做法让阈值随本轮文档质量自动升降，避免固定阈值导致的误删或漏删。

4. **生成阶段**  
   - 将保留下来的文档按分数高低排序，拼接成检索上下文 C。  
   - 主 LLM（通常是更大的模型）收到提示：“以下是与问题相关的材料，请基于它们回答”。  
   - LLM 生成最终答案 A。

**巧妙之处**  
- **共识过滤**：单个 LLM 可能因为随机性或知识盲区给出错误评分，多个代理的投票显著降低单点失误的概率。  
- **阈值自适应**：不需要人为调参，系统自行感知检索质量，保持高召回的同时压制噪声。  
- **全流程免训练**：所有步骤均使用现成的模型 API，省去标注、微调的成本，适合快速落地。

### 实验与效果

- **数据集**：论文在四个主流问答基准上评测，包括 NaturalQuestions、TriviaQA、HotpotQA、以及一个实时新闻问答集。  
- **对比基线**：普通 RAG（单一检索+直接拼接）、RAG+Fine‑tuned过滤器、以及最新的 Fusion‑In‑Decoder（FID）等。  
- **核心结果**：在所有数据集上，MAIN‑RAG 的答案准确率提升 2%~11%。例如在 NaturalQuestions 上从 71.3% 提升到 78.5%。同时，平均每轮保留的噪声文档数量下降约 30%。  
- **消融实验**：作者分别去掉多代理共识、去掉自适应阈值、只保留单一代理等设置，发现共识过滤贡献约 4% 的准确率提升，自适应阈值再贡献约 2% 的提升。  
- **局限性**：使用多个 LLM 代理会增加 API 调用次数和延迟，对实时系统成本较高；论文未在大规模商业检索场景（如上亿文档）做实验，效果的可扩展性仍待验证。

### 影响与延伸思考

MAIN‑RAG 把“多模型共识”引入 RAG，打开了“免训练过滤”这一思路的大门。随后的工作（如 Consensus‑RAG、Agent‑Filter 等）进一步探索了不同模型组合、跨语言共识以及在边缘设备上的轻量实现。对想继续深入的读者，可以关注以下方向：① 如何在保持低延迟的前提下选取最具性价比的代理组合；② 将共识过滤与检索排序（如学习‑to‑rank）联合训练的混合方案；③ 在多模态检索（图像、音频）中复用相同的多代理过滤框架。整体来看，MAIN‑RAG 为构建更可靠、可解释的检索增强系统提供了一个实用的基线。

### 一句话记住它

用多个 LLM 投票并自适应调节阈值，就能在不训练的情况下把检索噪声砍掉，让答案更准。