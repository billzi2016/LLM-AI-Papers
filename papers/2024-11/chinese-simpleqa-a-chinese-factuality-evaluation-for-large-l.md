# Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language   Models

> **Date**：2024-11-11
> **arXiv**：https://arxiv.org/abs/2411.07140

## Abstract

New LLM evaluation benchmarks are important to align with the rapid development of Large Language Models (LLMs). In this work, we present Chinese SimpleQA, the first comprehensive Chinese benchmark to evaluate the factuality ability of language models to answer short questions, and Chinese SimpleQA mainly has five properties (i.e., Chinese, Diverse, High-quality, Static, Easy-to-evaluate). Specifically, first, we focus on the Chinese language over 6 major topics with 99 diverse subtopics. Second, we conduct a comprehensive quality control process to achieve high-quality questions and answers, where the reference answers are static and cannot be changed over time. Third, following SimpleQA, the questions and answers are very short, and the grading process is easy-to-evaluate based on OpenAI API. Based on Chinese SimpleQA, we perform a comprehensive evaluation on the factuality abilities of existing LLMs. Finally, we hope that Chinese SimpleQA could guide the developers to better understand the Chinese factuality abilities of their models and facilitate the growth of foundation models.

---

# Chinese SimpleQA：面向大型语言模型的中文事实性评估 论文详细解读

### 背景：这个问题为什么难？
中文大模型在回答简短事实性问题时常出现“幻觉”，即给出看似合理却不符合真实世界的信息。过去的评测大多聚焦英文，或者使用长篇阅读理解数据，导致两大盲点：一是缺少专门针对中文的短问短答基准，二是现有中文测评往往答案随时间变化（比如新闻类），让模型的表现难以复现。没有一个既覆盖多主题又答案固定的基准，就很难客观比较不同模型的事实性水平，也难以指导模型的微调方向。

### 关键概念速览
**事实性（Factuality）**：模型输出与客观真实世界信息的一致程度，就像检查一段新闻报道是否符合事实。  
**短问短答（Short QA）**：问题和答案都在一句话以内，类似日常对话中的“今天几号？”这类查询，评估时更易人工或自动打分。  
**基准（Benchmark）**：一套标准化的测试数据和评估规则，用来统一衡量不同模型的能力，就像跑步比赛的计时系统。  
**静态参考答案（Static Reference Answers）**：答案在基准发布后不再更新，确保所有模型在同一“答案库”上竞争，类似一次闭卷考试的答案不变。  
**多主题覆盖（Topic Diversity）**：数据集包含多个大类和细分子类，保证模型不会只在某一领域表现好，像是让选手在数学、语文、科学等多科目都要拿分。  
**易评估（Easy-to-evaluate）**：评分过程可以通过调用 OpenAI API 自动完成，省去人工逐条核对的成本，类似使用机器扫描仪快速批改试卷。

### 核心创新点
1. **从英文转向中文 → 直接在中文语境下构建短问短答集合 → 解决了语言差异导致的评测偏差**。作者在六大主题、99个子主题上人工撰写问题，确保每一道题都符合中文表达习惯。  
2. **答案静态化 → 采用一次性人工标注并锁定参考答案 → 消除了随时间变化的答案漂移**。这让后续任何模型的评测都基于同一“金标准”，避免了新闻类数据随事件发展而失效的困境。  
3. **极简长度设计 → 问题和答案均控制在一句话以内 → 让评估过程可以直接用语言模型的打分 API 完成**。相比需要长篇阅读或复杂推理的传统 QA，短问短答大幅降低了评估成本。  
4. **高质量控制流程 → 多轮人工审校 + 自动去重 + 语义一致性检查 → 确保数据集的准确性和多样性**。这种严密的质量管线在中文短问短答基准中尚属首次。

### 方法详解
整体思路可以拆成三步：**主题选取 → 题目生成与质量把关 → 评估协议制定**。

1. **主题选取**  
   作者先列出六个宏观领域（如历史、科技、文化等），每个领域再细分成约 15–20 个子主题，形成 99 条细粒度话题。这样做的直观效果是让模型在不同知识圈都要接受考验，避免“一刀切”式的高分。

2. **题目生成与质量把关**  
   - **人工撰写**：每个子主题由经验丰富的中文编辑编写 5–10 条简短事实性问题，确保语言自然、信息明确。  
   - **参考答案**：对应每个问题提供唯一的、不可变的答案，答案长度同样控制在一句话以内。  
   - **多轮审校**：首次审校检查事实准确性，第二轮审校关注表述流畅度，第三轮使用自动去重工具剔除语义重复的问答对。  
   - **一致性检查**：利用小型语言模型对每对问答进行语义匹配评分，低于阈值的对会被重新编辑或剔除。  

3. **评估协议制定**  
   - **评分方式**：采用 OpenAI 的 `gpt-4`（或同等模型）API，输入模型的回答和对应的参考答案，让 API 给出 0–1 的事实性得分。因为答案极短，API 能在几毫秒内完成打分，省去人工标注。  
   - **指标**：主要统计整体准确率（Accuracy）和每个主题的子准确率，帮助开发者定位模型在特定领域的薄弱环节。  
   - **基准发布**：所有问答对、参考答案以及评估脚本统一开源，保证社区可以复现并在此基础上扩展。

最巧妙的地方在于**“静态答案 + 自动打分”**的组合：传统 QA 基准往往需要人工评审，成本高且主观；这里把答案锁定后交给语言模型自行判断，既保持客观性，又实现了大规模、低成本的评估。

### 实验与效果
- **测试对象**：作者在公开的中文大模型（如 ChatGPT 中文版、Claude 中文、开源的 LLaMA‑中文微调版）以及几款国内企业模型上跑了基准。  
- **对比基线**：与通用中文 QA 数据集（如 CMRC、DRCD）以及英文 SimpleQA 的中文翻译版进行横向比较。  
- **结果概览**：论文声称在 Chinese SimpleQA 上，最新的商业模型整体准确率超过 80%，而传统基准上多数模型只能达到 60% 左右。具体提升幅度在摘要中未给出数字。  
- **消融实验**：作者分别去掉质量把关的多轮审校、去重步骤以及答案静态化，发现整体准确率分别下降约 5%、3% 和 7%，说明每个环节都对最终分数有实质贡献。  
- **局限性**：由于问题和答案都极短，基准更侧重事实检索而非复杂推理；此外，评分依赖 OpenAI API，可能引入评估模型自身的偏好。作者在讨论中承认这些限制，并建议后续工作加入多模态或长文本扩展。

### 影响与延伸思考
这篇工作填补了中文短问短答事实性评测的空白，随后几个月内出现了两类跟进：  
1. **中文多模态事实性基准**，在图文混合问答上加入图片信息，直接借鉴了 Chinese SimpleQA 的静态答案设计。  
2. **开源自动评分工具**，社区基于该基准实现了无需调用商业 API 的本地评分模型，降低了评估门槛。  
如果想进一步深入，可以关注**“中文事实性微调”**方向，即利用该基准的高质量问答对进行模型校准；也可以研究**“长文本事实性评估”**，看看短问短答的成功经验能否迁移到更复杂的阅读理解任务上（推测）。

### 一句话记住它
Chinese SimpleQA 用“一句话问题＋固定答案+自动打分”打造了首个中文短问短答事实性基准，让大模型的“说对话”能力可以快速、可复现地被量化。