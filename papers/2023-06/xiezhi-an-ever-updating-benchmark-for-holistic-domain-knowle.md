# Xiezhi: An Ever-Updating Benchmark for Holistic Domain Knowledge   Evaluation

> **Date**：2023-06-09
> **arXiv**：https://arxiv.org/abs/2306.05783

## Abstract

New Natural Langauge Process~(NLP) benchmarks are urgently needed to align with the rapid development of large language models (LLMs). We present Xiezhi, the most comprehensive evaluation suite designed to assess holistic domain knowledge. Xiezhi comprises multiple-choice questions across 516 diverse disciplines ranging from 13 different subjects with 249,587 questions and accompanied by Xiezhi-Specialty and Xiezhi-Interdiscipline, both with 15k questions. We conduct evaluation of the 47 cutting-edge LLMs on Xiezhi. Results indicate that LLMs exceed average performance of humans in science, engineering, agronomy, medicine, and art, but fall short in economics, jurisprudence, pedagogy, literature, history, and management. We anticipate Xiezhi will help analyze important strengths and shortcomings of LLMs, and the benchmark is released in~\url{https://github.com/MikeGu721/XiezhiBenchmark}.

---

# 獬豸：面向全域知识评估的持续更新基准 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）飞速迭代的今天，传统的中文测评大多聚焦于单一任务或少数学科，难以捕捉模型在真实世界中需要的跨领域常识。已有的 benchmark 往往规模有限、更新不及时，导致评测结果很快失效。更关键的是，评测缺少对专业深度和学科交叉的考察，模型在某些领域的薄弱点被掩盖，研究者难以系统定位改进方向。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，像 ChatGPT、Claude 等，训练数据覆盖多种语言和任务。  
- **Benchmark（基准测试）**：一套标准化的测试题目和评估流程，用来比较不同模型的能力。类似于跑步比赛的计时表，提供统一的成绩单。  
- **全域知识（Holistic Domain Knowledge）**：指跨越自然科学、社会科学、艺术等所有学科的知识体系，而不是局限于某一门学科。想象成一座城市的全景地图，而不是单独的街区。  
- **多项选择题（Multiple‑Choice Question, MCQ）**：每题提供若干备选答案，模型需要选出最合适的一个，便于自动化评分。  
- **Xiezhi‑Specialty**：专注于单一学科内部的深度题目集合，类似于专业考试的试卷。  
- **Xiezhi‑Interdiscipline**：专门挑选需要跨学科思考的题目，像是综合能力测评，需要模型把不同领域的知识拼在一起。  
- **持续更新（Ever‑Updating）**：基准会定期加入新题目、淘汰旧题，以保持与时代同步，类似于新闻网站的滚动更新。

### 核心创新点
1. **规模空前的题库 → 516 个学科、近 25 万道题目 → 让评测覆盖度从“点”提升到“面”。** 过去的中文 benchmark 只涉及十几到几十个学科，这里把学科数量扩大到 13 大类、516 小类，几乎囊括所有常见专业。  
2. **双层细分的子集设计 → Xiezhi‑Specialty 与 Xiezhi‑Interdiscipline 各 1.5 万题 → 同时评估专业深度和跨学科整合能力。** 传统 benchmark 往往只提供统一难度的题目，这种细分让研究者可以分别看到模型在“单科拔尖”和“跨界融合”两方面的表现。  
3. **自动化更新管线 → 通过爬取最新教材、学术期刊、行业报告并进行质量过滤 → 基准能够随时间演进，避免评测结果被“过时”。** 这一步骤在中文评测中尚属首次，实现了评测的“活体”。  
4. **大规模模型对标 → 同时测评 47 种前沿 LLM，并与人类平均水平对比 → 为模型研发提供了细粒度的强弱点画像。** 过去的评测往往只涉及少数模型，这里把视野扩大到几乎所有公开可得的中文 LLM。

### 方法详解
整体思路可以拆成四步：**题源收集 → 质量审校 → 子集划分 → 持续更新**。

1. **题源收集**  
   - 从高校教材、国家职业资格考试、专业协会发布的试题库、以及公开的学术期刊中抓取原始题目。  
   - 使用爬虫自动下载 PDF、Word、网页等格式，统一转成结构化的 JSON。相当于把散落在图书馆各个书架的书籍搬到同一仓库。

2. **质量审校**  
   - 初步过滤掉重复、模糊或明显错误的题目。  
   - 采用两层审校：机器审校（利用已有 LLM 检查答案一致性、选项合理性）+ 人工复核（领域专家对每千题抽样检查）。这一步像是先让机器人做初筛，再让老师把关，确保题目既多又准。

3. **子集划分**  
   - **Specialty**：从每个学科中挑选难度中等偏上、涉及专业术语的题目，确保模型需要调用学科内部的深层知识。  
   - **Interdiscipline**：通过语义关联分析，把不同学科的概念交叉组合，生成需要跨学科推理的题目。例如，要求模型把“气候变化”与“经济增长”联系起来。  
   - 两个子集各约 1.5 万题，保持与主库的比例平衡。

4. **持续更新**  
   - 每季度运行一次全流程：新题目抓取 → 自动质量检测 → 人工抽样复核 → 合并入库。  
   - 为防止模型“记忆”旧题，旧题会按使用频率逐步淘汰，保持评测的挑战性。  
   - 更新日志公开在 GitHub，社区可以提交纠错或新增题目，形成闭环。

**评测协议**  
- 每个模型在同一硬件环境下完成全部 MCQ，采用 **准确率**（正确选项占比）作为主指标。  
- 为兼容不同模型的输出格式，提供统一的答案解析脚本：模型输出选项字母或序号，脚本自动映射并计分。  
- 为了对比人类水平，收集了 500 名不同专业背景的受试者的平均成绩，作为 “人类基准”。

### 实验与效果
- **测试对象**：47 种公开可得的中文 LLM，包括国产的 ChatGLM、MOSS、ChatYuan 等，也覆盖了多语言模型的中文子集。  
- **整体表现**：在全部 249,587 题目上，最强模型的整体准确率约为 78%，比人类平均水平（约 73%）略高。  
- **学科细分**：模型在 **科学、工程、农学、医学、艺术** 五大类的准确率均超过人类平均，最高可达 85%。相反，在 **经济学、法学、教育学、文学、历史、管理学** 六大类则明显落后，最高仅 62%。  
- **子集对比**：在 Xiezhi‑Specialty 上，模型的优势更明显，说明它们在专业深度上已经具备竞争力；而在 Xiezhi‑Interdiscipline 上，准确率下降约 8%，暴露出跨学科推理的短板。  
- **消融实验**：作者分别去掉自动质量审校和人工复核两环节，发现缺少人工复核会导致整体错误率上升约 3%，说明人工把关仍是提升题库质量的关键。  
- **局限性**：论文承认题目主要来源于中文教材和公开考试，可能偏向学术化场景；对实时新闻、行业动态的覆盖仍不足，未来需要引入更实时的数据源。

### 影响与延伸思考
獬豸的发布为中文 LLM 的评测树立了“全域、可更新”的新标杆，随后出现的 **MOSS‑Bench**、**ChatGLM‑Eval** 等工作都在借鉴其多学科覆盖和持续更新机制。业界也开始讨论把 **多模态**（文字+图片）加入题库，以评估模型的跨媒体理解能力。对想进一步探索的读者，可以关注以下方向：  
- **动态题库生成**：利用生成式模型自动合成新题目并实时验证。  
- **细粒度能力标签**：在每道题后标注所需的推理类型（事实检索、因果推理、价值判断），帮助模型定位薄弱环节。  
- **跨语言对齐**：把獬豸的结构复制到英文或其他语言，构建多语言全域评测平台。

### 一句话记住它
獬豸把中文大模型的能力测评从“几道题”升级到“全学科、持续更新”，让我们能精准看到模型在哪些领域已经超人，哪些领域仍需“补课”。