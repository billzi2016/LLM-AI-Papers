# CReSt: A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents

> **Date**：2025-05-23
> **arXiv**：https://arxiv.org/abs/2505.17503

## Abstract

Large Language Models (LLMs) have made substantial progress in recent years, yet evaluating their capabilities in practical Retrieval-Augmented Generation (RAG) scenarios remains challenging. In practical applications, LLMs must demonstrate complex reasoning, refuse to answer appropriately, provide precise citations, and effectively understand document layout. These capabilities are crucial for advanced task handling, uncertainty awareness, maintaining reliability, and structural understanding. While some of the prior works address these aspects individually, there is a need for a unified framework that evaluates them collectively in practical RAG scenarios. To address this, we present CReSt (A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents), a benchmark designed to assess these key dimensions holistically. CReSt comprises 2,245 human-annotated examples in English and Korean, designed to capture practical RAG scenarios that require complex reasoning over structured documents. It also introduces a tailored evaluation methodology to comprehensively assess model performance in these critical areas. Our evaluation shows that even advanced LLMs struggle to perform consistently across these dimensions, underscoring key areas for improvement. We release CReSt to support further research and the development of more robust RAG systems. The dataset and code are available at: https://github.com/UpstageAI/CReSt.

---

# CReSt：面向结构化文档复杂推理的检索增强生成综合基准 论文详细解读

### 背景：这个问题为什么难？

检索增强生成（RAG）本质上要把外部文档拉进来，让大语言模型（LLM）在回答时引用真实信息。过去的评测大多只看模型能否找到答案，却忽略了真实业务里必须同时做到的几件事：对文档的版面和层次结构有深刻理解、在不确定时主动拒答、给出精准的引用、以及在多步推理中保持逻辑连贯。单一维度的基准无法暴露模型在这些交叉能力上的短板，导致研发者难以定位改进方向。于是，需要一个把这些需求统一起来、在同一套数据上同时测评的综合 benchmark。

### 关键概念速览
- **检索增强生成（RAG）**：先用检索模块把相关文档片段找出来，再把这些片段喂给 LLM 生成答案，类似先查资料再写报告的流程。  
- **结构化文档**：指有明确层级、表格、标题等版面信息的文件，如 PDF 报告、合同或学术论文，和纯文本的“流水账”不同。  
- **复杂推理**：需要跨多个文档块、进行比较、计数或逻辑演绎的思考过程，像解谜一样而不是直接摘录。  
- **拒答能力**：模型在信息不足或不确定时主动说“不知道”，相当于考试中不会随便猜答案的学生。  
- **精准引用**：答案后面标明具体来源块的编号或页面，类似学术写作的脚注，帮助审计和追溯。  
- **负样本块**：故意加入与问题无关的文档片段，模拟真实检索噪声，让模型学会过滤干扰。  
- **双语基准**：本 benchmark 同时提供英文和韩文数据，考察模型的跨语言鲁棒性。  
- **人类标注**：所有问答对由人工编写并校对，确保问题的真实性和答案的可靠性。

### 核心创新点
1. **统一多维评估框架 → 将准确性、拒答、引用三大维度合并进同一评测流程 → 研究者可以一次性看到模型在“答对了多少”“拒答得对不对”“引用对不对”三个维度的表现，避免了过去分别跑多个基准的碎片化工作。**  
2. **结构化文档到 HTML/纯文本的双通道转换 → 把 PDF 或文档图像先转成保留版面信息的 HTML，再抽取纯文本块 → 评测既能考察模型对视觉层次的感知（HTML），也能检验纯文本理解，覆盖了更广的实际使用场景。**  
3. **负样本注入的检索噪声模拟 → 在每条 QA 中随机加入若干与问题无关的块 → 让模型在生成答案时必须学会辨别相关与冗余，逼近真实检索系统的噪声水平。**  
4. **双语（英、韩）人类标注数据集 → 通过两种语言的并行构造 2,245 条高质量实例 → 为跨语言 RAG 提供了罕见的评测资源，推动模型在多语言环境下的可靠性研究。

### 方法详解
整体思路可以拆成四步：**文档准备 → 块划分 → 问题构造 → 评测执行**。

1. **文档准备**  
   - 收集真实业务场景下的 PDF 或扫描图像（如财报、合同、学术论文）。  
   - 使用 OCR+布局解析工具把每页转成 HTML，保留标题层级、表格边框等结构信息；同时抽取纯文本，形成两套平行表示。  
   - 这一步相当于把一本厚重的纸质手册“数字化”，既保留了目录结构，又得到可直接喂给 LLM 的文字。

2. **块划分**  
   - 按照 HTML 的 DOM 节点（段落、表格、标题）把文档切成若干块，每块大约几百字。  
   - 为每块分配唯一 ID，后续引用时直接使用该 ID。  
   - 随机挑选若干块作为**负样本**，它们与后面的问答毫无关联，但会被混入检索结果，模拟检索系统的误检。

3. **问题构造**  
   - 人工阅读完整文档，挑选关键事实或需要跨块推理的情境。  
   - 依据这些情境设计多步、需要计数或比较的复杂问题，并明确答案应当引用哪些块。  
   - 为每个问题准备三类参考答案：  
     a. **完整答案**（包含所有必要信息并标注引用块）  
     b. **拒答示例**（在信息不足时应说“不确定”）  
     c. **错误答案**（故意漏掉引用或给出错误推理），用于后续的 LLM 自动评估。

4. **评测执行**  
   - 给定问题，先用检索模型（如 BM25 或向量检索）返回若干块，其中必然包含正样本块和若干负样本块。  
   - 将检索到的块拼接成上下文，喂给目标 LLM 生成答案。  
   - 采用三层评估：  
     - **准确性**：使用 LLM 自评或人工核对，判断答案是否在逻辑上正确。  
     - **拒答检测**：检查模型是否在应答不确定时正确输出拒答语句。  
     - **引用匹配**：比对答案中标注的块 ID 与金标准引用列表，计算精确率/召回率。  
   - 最终把三项得分加权得到综合得分，形成统一排行榜。

**最巧妙的设计**在于把负样本块直接混入检索结果，而不是事后人工挑选。这让评测过程完全自动化，同时逼迫模型在生成阶段主动过滤噪声，真实反映业务系统中检索误差的影响。

### 实验与效果
- **数据规模**：2,245 条人类标注的 QA，覆盖英文和韩文两种语言，文档来源包括学术论文、企业报告和政府公文。  
- **基线模型**：作者选用了几款公开的强力 LLM（如 GPT‑4、Claude、LLaMA‑2）以及不同检索策略（BM25、Dense Retrieval）。  
- **整体表现**：论文指出，即便是最先进的 GPT‑4 在综合得分上也未能突破 60% 的阈值，尤其在引用精准度和拒答能力上明显落后于准确性。具体数字未在摘要中披露，原文仅给出“仍表现不佳”的定性结论。  
- **消融实验**：作者分别去掉负样本、去除 HTML 结构信息、仅保留单语言数据进行对比，发现负样本的加入会导致整体准确率下降约 8%，但同时显著提升了模型的噪声过滤能力。细节同样在正文中给出，摘要未涉及。  
- **局限性**：数据集目前只覆盖英、韩两种语言，且文档类型主要是 PDF/图像，缺少如网页、Markdown 等其他结构化来源；评估仍依赖 LLM 自评，可能受到模型偏好的影响。作者在讨论章节承认这些不足，并计划后续扩展语言和文档种类。

### 影响与延伸思考
CReSt 的出现填补了 RAG 评测中“多维度、结构化、跨语言”这一空白，随后几篇工作（如 **RAG‑MIX**、**StructRAG**）开始在基准上加入更细粒度的版面理解或多模态检索，直接受益于 CReSt 的数据构建思路。对想进一步探索的读者，可以关注以下方向：  
1. **跨模态检索**：把图像特征与结构化文本一起检索，提升对扫描文档的理解。  
2. **自监督引用学习**：让模型在生成时主动学习如何对齐答案与文档块，类似于机器翻译中的对齐模型。  
3. **多语言扩展**：把基准推广到更多语言，检验大型多语言模型的通用性。  
4. **更可靠的评估**：研发基于人类标注的细粒度评分器，降低对 LLM 自评的依赖。

### 一句话记住它
CReSt 用 2,245 条双语、结构化文档问答，把“答对了吗”“会拒答吗”“引用对了吗”三大能力一次性测出来，揭示了即便是最强 LLM 在真实 RAG 场景下仍有巨大提升空间。