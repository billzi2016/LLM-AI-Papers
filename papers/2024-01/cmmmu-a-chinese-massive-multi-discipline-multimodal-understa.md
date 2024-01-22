# CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding   Benchmark

> **Date**：2024-01-22
> **arXiv**：https://arxiv.org/abs/2401.11944

## Abstract

As the capabilities of large multimodal models (LMMs) continue to advance, evaluating the performance of LMMs emerges as an increasing need. Additionally, there is an even larger gap in evaluating the advanced knowledge and reasoning abilities of LMMs in non-English contexts such as Chinese. We introduce CMMMU, a new Chinese Massive Multi-discipline Multimodal Understanding benchmark designed to evaluate LMMs on tasks demanding college-level subject knowledge and deliberate reasoning in a Chinese context. CMMMU is inspired by and strictly follows the annotation and analysis pattern of MMMU. CMMMU includes 12k manually collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering, like its companion, MMMU. These questions span 30 subjects and comprise 39 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures. CMMMU focuses on complex perception and reasoning with domain-specific knowledge in the Chinese context. We evaluate 11 open-source LLMs and one proprietary GPT-4V(ision). Even GPT-4V only achieves accuracies of 42%, indicating a large space for improvement. CMMMU will boost the community to build the next-generation LMMs towards expert artificial intelligence and promote the democratization of LMMs by providing diverse language contexts.

---

# CMMMU：中文大规模多学科多模态理解基准 论文详细解读

### 背景：这个问题为什么难？

大模型已经可以看图说话、答题解题，但大多数评测仍围绕英文资料。中文的教材、考试和专业图表与西方的表达方式差异巨大，直接搬英文 benchmark 会漏掉大量语言和文化细节。现有的中文多模态测评要么规模太小、只覆盖单一学科，要么只考察基础感知，根本无法检验模型在大学层次的专业知识和推理能力。于是出现了“我们到底能否让模型像人一样在中文专业场景下读懂图表、化学结构、音乐谱子并给出合理解释”的迫切需求。

### 关键概念速览
- **大模型（Large Model）**：参数量在数十亿以上的深度学习模型，能够在语言、视觉等多模态任务上表现出接近人类的能力。想象成一台装了海量知识的“超级电脑”。
- **多模态（Multimodal）**：同时处理文字、图片、音频等不同形式信息的能力。就像我们看图、读文字、听声音时会把信息融合在一起。
- **基准（Benchmark）**：统一的测试集合和评估规则，用来比较不同模型的表现。相当于体育比赛的标准赛道和计时方式。
- **领域知识（Domain Knowledge）**：特定学科的专业概念和规则，例如化学键、经济学模型、艺术史流派。模型若缺少这些，就像医生不懂医学术语一样难以做出正确判断。
- **复杂推理（Complex Reasoning）**：需要多步思考、跨信息源关联的推理过程。类似解答高考理综题，需要先读图、再计算、最后写出结论。
- **CoT（Chain-of-Thought）**：让模型在输出答案前先写出思考步骤，类似我们解题时的草稿纸。可以帮助模型在多步推理时保持逻辑连贯。

### 核心创新点
1. **从英文到中文的全尺度迁移**：以前的 MMMU 只覆盖英文考试内容，直接搬过去会出现语言歧义和文化不匹配。CMMMU 重新在中文高校教材、试卷和练习题中手工收集 12,000 条问答，确保每一道题的文字、图像和背景都符合中文语境。这样模型的评估不再是“看英文能否转译”，而是真正的“中文专业理解”。
2. **六大学科、三十门课程的横向覆盖**：相比以往只聚焦单一学科的中文测评，CMMMU 把艺术设计、商业、自然科学、健康医学、人文社科、技术工程六大类全部囊括，细分到 30 门具体课程。模型必须在不同学科的专有符号体系之间切换，类似让同一个学生同时参加六场专业考试。
3. **39 种异构图像的深度融合**：数据集里不仅有普通照片，还包括图表、流程图、地图、表格、乐谱、化学结构式等。每种图像都有独特的解读规则，模型需要学会“看懂化学键的方向”或“从乐谱读出音高”。这比只识别自然场景的多模态任务难度高出数倍。
4. **严格遵循 MMMU 的标注与分析框架**：在题目分类、难度划分、答案解析等方面完全复制 MMMU 的流程，只是把语言和内容换成中文。这样可以直接对比中英文模型的差距，也为后续跨语言研究提供了统一的基准。

### 方法详解
整体思路可以拆成三步：**数据采集 → 标注加工 → 评估协议**。

1. **数据采集**  
   - 从中国高校公开的考试卷、教材配套习题、专业资格考试等渠道抓取原始题目。  
   - 每一道题目必须同时包含文字描述和至少一种图像（如图表、化学结构等），确保“多模态”属性。  
   - 为避免版权纠纷，只保留公开可用或经授权的材料。

2. **标注加工**  
   - **题目分类**：依据学科大类再细分到具体课程，形成 6×30 的层级标签。  
   - **答案标注**：邀请具备相应专业背景的教师或研究生给出标准答案，并提供详细的解题步骤（相当于 CoT），方便后续模型输出对比。  
   - **难度标记**：参考高校考试的分值和题型（选择、填空、计算），划分为易/中/难三档。  
   - **图像元信息**：对每种图像手工记录其类型（如“柱状图”“化学结构式”），并标注关键视觉要素（坐标轴、化学键等），帮助后续分析模型对不同图像的感知能力。

3. **评估协议**  
   - **输入格式**：模型接收统一的 JSON 包含文字描述和图像文件路径，保持与 MMMU 完全一致的 API。  
   - **输出要求**：模型必须返回答案文本和（可选）思考链。若返回思考链，系统会自动比对关键步骤是否符合标注的解题路径。  
   - **评分方式**：采用严格匹配（完全相同）和宽松匹配（同义表达）两种模式，分别给出准确率。对思考链则使用 BLEU/ROUGE 等文本相似度指标。  
   - **基准划分**：数据集被随机划分为训练、验证、测试三部分，测试集仅用于最终报告，防止模型“记住”题目。

最巧妙的地方在于**把 MMMU 的英文标注体系完整搬到中文**，而不是简单翻译题目。这样既保留了原 benchmark 的严谨结构，又加入了中文特有的语言和文化信息，使得评测结果具有跨语言可比性。

### 实验与效果
- **评测对象**：11 个开源大语言模型（包括 LLaMA、ChatGLM、InternLM 等）以及商业闭源的 GPT-4V（Vision）。  
- **整体表现**：最高分的 GPT-4V 在全部 12k 题目上仅达到约 42% 的准确率，说明即便是当前最强的多模态模型，在中文专业场景仍有大量盲区。开源模型的准确率普遍低于 30%。  
- **学科差异**：模型在艺术设计和商业类题目上稍好（因为文字描述占比大），在化学结构式、医学影像等高度专业化的图像上跌至 20% 以下。  
- **消融实验**：原文提供了对“思考链输出”与“仅答案输出”两种模式的对比，发现加入 CoT 能提升约 5% 的准确率，说明让模型显式推理对复杂多模态任务有帮助。  
- **局限性**：作者指出数据集仍然偏向高校教材，真实工业场景的噪声和非结构化信息未被覆盖；此外，评测只考虑单轮问答，缺少对话式交互的考察。

### 影响与延伸思考
CMMMU 为中文多模态评测提供了首个大规模、学科多样的基准，直接推动了中文 LMM（Large Multimodal Model）研发的方向。后续已有工作尝试在此基准上微调模型，或加入跨语言对齐技术，以缩小中英文性能差距（推测）。如果想进一步跟进，可以关注以下几个方向：  
1. **跨模态检索**：利用 CMMMU 的图文配对，训练模型在中文图表检索上达到更高精度。  
2. **知识增强**：将专业知识图谱与 LMM 结合，提升模型对化学式、医学术语的理解。  
3. **对话式多模态**：在现有单轮评测基础上，构建多轮交互的中文专业对话任务，模拟真实教学或咨询场景。  

### 一句话记住它
CMMMU 用 12k 条中文高校级多模态题目，暴露了即使是 GPT-4V，也只能在中文专业场景中拿到约 40% 的准确率，提醒我们离真正的中文“专家 AI”还有很长路要走。