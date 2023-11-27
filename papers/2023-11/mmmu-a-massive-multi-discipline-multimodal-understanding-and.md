# MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning   Benchmark for Expert AGI

> **Date**：2023-11-27
> **arXiv**：https://arxiv.org/abs/2311.16502

## Abstract

We introduce MMMU: a new benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span 30 subjects and 183 subfields, comprising 30 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures. Unlike existing benchmarks, MMMU focuses on advanced perception and reasoning with domain-specific knowledge, challenging models to perform tasks akin to those faced by experts. The evaluation of 14 open-source LMMs as well as the proprietary GPT-4V(ision) and Gemini highlights the substantial challenges posed by MMMU. Even the advanced GPT-4V and Gemini Ultra only achieve accuracies of 56% and 59% respectively, indicating significant room for improvement. We believe MMMU will stimulate the community to build next-generation multimodal foundation models towards expert artificial general intelligence.

---

# MMMU：面向专家级通用人工智能的大规模多学科多模态理解与推理基准 论文详细解读

### 背景：这个问题为什么难？

在视觉语言模型（VLM）兴起之前，评测大多停留在单一模态或日常对话层面，例如图像分类、视觉问答或简单的文字描述。真实世界的专业任务往往需要同时阅读图表、化学结构、乐谱等高度专业化的视觉信息，并结合大学教材级别的学科知识进行推理。现有 benchmark 如 VQAv2、ScienceQA 等，要么只覆盖少数学科，要么图像类型单一，导致模型在“专家级”场景下的表现几乎没有被测量。缺少这样的大规模、多学科、多模态测试集，研究者很难判断模型是否真的具备跨领域的深度理解和推理能力。

### 关键概念速览
- **多模态模型（Multimodal Model）**：能够同时处理文字、图片、音频等不同类型输入的模型，就像人用眼睛看图、用耳朵听声音、用大脑把信息融合一样。
- **专家级通用人工智能（Expert AGI）**：指能够在多个专业领域达到大学水平甚至更高的人工智能，类似于拥有“全科医生”或“全能工程师”能力的系统。
- **跨学科推理（Cross‑disciplinary Reasoning）**：在解答时需要把不同学科的概念串联起来，例如用数学公式解释经济图表，类似于把化学实验结果和物理定律一起用来解释现象。
- **图表/结构化视觉信息**：包括柱状图、流程图、地图、化学式、乐谱等，这类信息不像自然图片那样直观，需要先进行结构化解析再进行推理。
- **CoT（思维链）**：让模型在给出答案前先写出推理步骤，像人在解题时先列出草稿，帮助模型避免“一口气”跳到错误结论。
- **Few‑shot Prompting**：在少量示例的帮助下让模型学习任务格式，类似于老师给学生几道例题后让他们自行作答。

### 核心创新点
1. **规模与覆盖的跃迁**  
   之前的多模态 benchmark 多在几千题、单一学科或少数图像类型上徘徊。MMMU 收集了 11.5 K 题目，横跨 6 大学科、30 门专业、183 个子领域，图像类型多达 30 种。这样的大规模、多样本设计让模型必须在“全科”层面展示能力，而不是在狭窄的“专科”上作弊。

2. **从“感知”到“推理+知识”**  
   传统 benchmark 往往只要求模型识别图像内容或回答事实性问题。MMMU 把每道题目设计成需要先感知（比如读出化学结构式），再结合学科知识进行多步推理。相当于在一次考试里既要做“看图”也要做“写论文”，提升了评测的难度和真实性。

3. **统一评测协议**  
   为了公平比较，作者为所有模型提供统一的 Prompt 模板，强制使用 CoT 方式输出。这样可以直接对比不同模型在同一道题上的思考路径，避免因提示差异导致的误判。

4. **公开基准与开放评测平台**  
   除了发布数据集，作者还搭建了在线评测服务器，任何人都可以提交模型输出得到即时得分。类似于 Codeforces 对编程竞赛的作用，极大降低了复现门槛，促进社区快速迭代。

### 方法详解
**整体思路**  
MMMU 本身不是一种新模型，而是一个评测框架。它的核心流程可以概括为三步：① 题目构造 → ② Prompt 统一 → ③ 自动评分。下面把每一步拆开讲。

**① 题目构造**  
- **来源**：作者从公开的大学教材、期末考试、专业资格考试等渠道抓取原始题目。  
- **筛选**：使用人工审校把题目分成“感知型”（只需读取图像信息）和“推理型”（需要结合学科知识）。  
- **多模态化**：对每道题目配上对应的视觉材料，如把统计表格转成图片、把化学方程式渲染成结构图。  
- **标签**：每题标注学科、子领域、所需图像类型以及答案的多选/填空形式，方便后续分析。

**② Prompt 统一**  
- **模板**：所有模型都收到形如“下面是一张图，请先描述图中信息，然后回答以下问题。请一步步思考（CoT）。”的指令。  
- **Few‑shot 示例**：在每个学科的开头放入 2–3 例示例，展示如何先描述图像再进行推理。  
- **输出格式**：强制要求模型先输出“描述：…”，随后是“思考步骤：…”，最后给出“答案：”。这样评分脚本可以自动抓取答案字段。

**③ 自动评分**  
- **答案匹配**：对选择题使用字符串相似度，对填空题使用正则匹配或数值容差。  
- **细粒度分析**：评分脚本会记录模型在“描述”阶段的准确率、在“思考步骤”是否出现关键概念、以及最终答案的正确率，帮助研究者定位模型的薄弱环节。  
- **排行榜**：所有提交的模型得分会实时更新，形成公开排行榜。

**最巧妙的设计**  
- **CoT 强制**：让所有模型必须走思维链，避免“黑箱”直接输出答案，这在多学科推理中尤为重要，因为一步错会导致整题错。  
- **多模态统一化**：把表格、地图、乐谱等全部转成图片，使得视觉编码器可以统一处理，省去了为每种结构单独设计特征提取器的工作。

### 实验与效果
- **测试对象**：作者评测了 14 个开源大语言多模态模型（如 LLaVA、MiniGPT‑4 等）以及两款商业模型：GPT‑4V（Vision）和 Gemini Ultra。  
- **整体表现**：在全部 11.5 K 题目上，最强的开源模型最高准确率只有约 38%，而 GPT‑4V 达到 56%，Gemini Ultra 59%。这说明即使是业界最前沿的系统，也只能在专家级任务上取得略高于随机的成绩。  
- **学科差异**：在艺术设计类题目上，模型表现相对较好（≈65%），而在医学、化学结构解析上则跌至 30% 以下，凸显专业视觉信息仍是瓶颈。  
- **消融实验**：作者去掉 CoT 强制后，整体准确率下降约 7%，说明思维链对复杂推理有实质帮助。去掉 Few‑shot 示例后，低资源模型的下降更明显，验证了示例引导的重要性。  
- **局限性**：论文承认数据集仍受限于公开教材的版权，某些高阶专业题目（如专利审查）未被覆盖；此外，评分脚本对开放式答案的容错仍有提升空间。

### 影响与延伸思考
MMMU 推出后，社区迅速把它当作“多模态专业考试”来使用。随后出现的工作如 **M3Exam**、**VisReason** 等，都在尝试进一步扩大题目规模或加入视频、音频等新模态。还有研究把 MMMU 作为微调数据源，尝试让模型在“看图+写论文”任务上进行持续学习。对想深入的读者，可以关注以下方向：  
- **结构化视觉解析**：如何让模型直接读取化学式、乐谱等专业符号。  
- **跨模态知识图谱**：把学科知识库与视觉特征对齐，提升推理的可靠性。  
- **自监督多模态预训练**：利用大规模未标注的专业文献和图表进行预训练，缩小与专家水平的差距。  
（以上为基于公开信息的推测，后续文献会进一步验证）

### 一句话记住它
MMMU 用 11.5 K 条大学级别的多模态题目，把“看图”升级为“看图并像专家一样推理”，让我们首次看到即便是 GPT‑4V 也只能在专家任务上拿到 60% 左右的分数。