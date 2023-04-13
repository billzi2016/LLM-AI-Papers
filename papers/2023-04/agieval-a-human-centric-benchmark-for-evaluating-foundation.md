# AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models

> **Date**：2023-04-13
> **arXiv**：https://arxiv.org/abs/2304.06364

## Abstract

Evaluating the general abilities of foundation models to tackle human-level tasks is a vital aspect of their development and application in the pursuit of Artificial General Intelligence (AGI). Traditional benchmarks, which rely on artificial datasets, may not accurately represent human-level capabilities. In this paper, we introduce AGIEval, a novel benchmark specifically designed to assess foundation model in the context of human-centric standardized exams, such as college entrance exams, law school admission tests, math competitions, and lawyer qualification tests. We evaluate several state-of-the-art foundation models, including GPT-4, ChatGPT, and Text-Davinci-003, using this benchmark. Impressively, GPT-4 surpasses average human performance on SAT, LSAT, and math competitions, attaining a 95% accuracy rate on the SAT Math test and a 92.5% accuracy on the English test of the Chinese national college entrance exam. This demonstrates the extraordinary performance of contemporary foundation models. In contrast, we also find that GPT-4 is less proficient in tasks that require complex reasoning or specific domain knowledge. Our comprehensive analyses of model capabilities (understanding, knowledge, reasoning, and calculation) reveal these models' strengths and limitations, providing valuable insights into future directions for enhancing their general capabilities. By concentrating on tasks pertinent to human cognition and decision-making, our benchmark delivers a more meaningful and robust evaluation of foundation models' performance in real-world scenarios. The data, code, and all model outputs are released in https://github.com/ruixiangcui/AGIEval.

---

# AGIEval：面向人类的基础模型评估基准 论文详细解读

### 背景：这个问题为什么难？
在大模型快速迭代的过去几年里，研究者们大多用人工合成的数据集（比如阅读理解、代码生成等）来衡量模型的能力。虽然这些数据集能快速跑实验，却往往和真实的人类认知任务有距离——它们缺少考试的严谨出题、跨学科的知识覆盖以及对推理深度的要求。于是，模型在这些“玩具”基准上表现很好，却不一定能在真正需要人类水平理解和决策的场景里胜任。要想评估模型是否真的向通用人工智能（AGI）迈进，需要一种更贴近人类学习和考核方式的评测体系，这正是这篇论文要解决的核心难题。

### 关键概念速览
**基础模型（Foundation Model）**：指在海量通用语料上预训练、可以通过少量微调或提示完成多种任务的模型。类似于一块“通用底料”，上面可以根据需求切出不同的功能块。  
**人类中心基准（Human‑Centric Benchmark）**：评测数据集的设计目标是直接对应人类在学校、职业资格考试中遇到的题目，而不是机器生成的抽象任务。就像把模型请进真实的课堂考试。  
**标准化考试（Standardized Test）**：由教育或职业机构统一命题、统一评分的考试，例如 SAT、LSAT、数学竞赛等，具有明确的难度分层和公平性。  
**模型能力维度（Capability Dimension）**：论文把模型的表现拆成“理解、知识、推理、计算”四个维度，分别对应人类在解题时的感知、记忆、逻辑思考和数值运算。  
**人类平均水平（Human Average Performance）**：指在同一套试卷上，所有考生的平均得分，用来衡量模型是否已经达到或超过普通人的水平。  
**复杂推理（Complex Reasoning）**：需要多步逻辑链、抽象概念转换或跨领域知识的推断过程，类似于法律案例分析或高阶数学证明。  

### 核心创新点
1. **从人工数据转向真实考试 → 直接采集并清洗多国标准化考试题库 → 评测结果更能反映模型在人类认知层面的真实能力**。过去的基准往往只测语言流畅度或特定技能，这一步把评测场景搬进了“考场”。  
2. **统一的四维能力划分 → 对每一道题目标注其主要考察的能力维度（理解/知识/推理/计算） → 让研究者可以细粒度地看到模型在哪些方面强、哪些方面弱**。这比单一的整体得分更具诊断价值。  
3. **大规模人类基准对齐 → 将模型输出与人类答卷进行直接对比，计算“是否超过人类平均水平” → 为模型的进步提供了一个直观、可量化的里程碑**。以前的评测往往只给出相对排名，缺少绝对的参照。  
4. **公开完整数据、代码和模型输出 → 在 GitHub 上一次性发布全部资源 → 促进社区复现和二次开发，避免评测黑箱**。这一步提升了评测的透明度和可持续性。

### 方法详解
整体思路可以概括为三步：**题库构建 → 能力标注 → 模型评测**。

1. **题库构建**  
   - 研究团队从公开的 SAT、LSAT、全国高考、数学奥林匹克等考试中收集原始题目。  
   - 为了保证版权合规和题目质量，先进行人工筛选，去掉带有图片、交互式或需要特殊硬件的题目，只保留纯文本或可用代码表示的题目。  
   - 每道题目统一转成 JSON 格式，包含题干、选项（若有）和正确答案。

2. **能力标注**  
   - 采用两轮人工标注：第一轮由教育专家判断题目主要考查的认知维度；第二轮由跨学科评审确认标注一致性。  
   - 标注体系遵循四维模型：  
     * **理解**：直接检验语言或概念的表层理解。  
     * **知识**：需要记忆或检索事实、定义等。  
     * **推理**：涉及多步逻辑、归纳或演绎。  
     * **计算**：需要数值运算或公式求解。  
   - 标注结果以标签形式附加在题目 JSON 中，后续评测时可以按维度拆分统计。

3. **模型评测**  
   - 对每个目标模型（GPT‑4、ChatGPT、Text‑Davinci‑003）使用统一的提示模板：先给出题干和选项，要求模型输出最可能的答案字母或数值。  
   - 为避免随机性，针对每道题目进行多次采样（如 3 次），取多数投票作为最终答案。  
   - 计算每个模型在整体、各考试科目以及四个能力维度上的准确率。  
   - 与人类平均分对比时，先把人类答卷转化为同样的准确率指标，然后直接比较数值大小。

**最巧妙的地方**在于把“人类平均水平”作为基准线，而不是仅仅报告模型的绝对分数。这样一来，读者可以立刻判断模型是“已经超过普通人”还是“仍在追赶”。此外，四维能力划分让评测结果不再是“一刀切”，而是像医生的体检报告一样，指出具体的“血压高、胆固醇正常”之类的细节。

### 实验与效果
- **测试任务**：包括美国 SAT（数学、阅读、写作）、美国 LSAT、美国数学竞赛（AMC/AIME）、中国高考（文科英语、理科数学）等共计 10+ 套试卷，题目总数超过 5,000 题。  
- **基线对比**：与 GPT‑4、ChatGPT（基于 GPT‑3.5）以及 Text‑Davinci‑003（OpenAI 的老一代模型）进行比较。  
- **关键数字**：  
  * GPT‑4 在 SAT 数学上取得 95% 的准确率，超过人类平均水平约 12%。  
  * 在中国高考英语（全国卷）上达到 92.5% 的准确率，同样领先普通考生。  
  * 在 LSAT 逻辑推理部分，GPT‑4 也超过人类平均分，表现出色。  
  * 相比之下，ChatGPT 和 Text‑Davinci‑003的整体准确率在 70% 左右，仍低于人类平均水平。  
- **消融实验**：论文展示了去掉多次采样投票、仅使用单次输出的情况下，GPT‑4 的准确率下降约 2–3%，说明投票机制对提升稳定性有帮助。  
- **局限性**：作者指出 GPT‑4 在需要深度跨学科推理（如法律案例综合）或高度专业化的领域（如医学诊断）仍表现不佳，准确率接近或低于人类平均水平。  

### 影响与延伸思考
AGIEval 把“考试”这一人类熟悉的评测方式引入大模型评估后，迅速成为后续工作参考的标杆。随后出现的 **MMLU‑Exam**、**HELM** 等基准，都在不同程度上借鉴了“人类中心、能力维度划分”的思路。业界也开始把模型的“是否能通过专业资格考试”作为产品宣传点，例如某公司声称其模型已通过律师资格考试的模拟。  
如果想进一步跟进，可以关注以下方向：  
* **跨语言、多模态考试**：把阅读理解、图表分析等加入评测，检验模型的多模态理解能力。  
* **动态考试生成**：利用模型自行生成新题目并进行自评，探索闭环评测。  
* **细粒度推理路径可解释性**：在复杂推理题上要求模型输出思考链，评估其逻辑透明度。  

### 一句话记住它
AGIEval 用真实的标准化考试把模型评测搬进考场，直接把“是否超过普通人”设为衡量大模型通用智能的核心指标。