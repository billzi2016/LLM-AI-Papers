# C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for   Foundation Models

> **Date**：2023-05-15
> **arXiv**：https://arxiv.org/abs/2305.08322

## Abstract

New NLP benchmarks are urgently needed to align with the rapid development of large language models (LLMs). We present C-Eval, the first comprehensive Chinese evaluation suite designed to assess advanced knowledge and reasoning abilities of foundation models in a Chinese context. C-Eval comprises multiple-choice questions across four difficulty levels: middle school, high school, college, and professional. The questions span 52 diverse disciplines, ranging from humanities to science and engineering. C-Eval is accompanied by C-Eval Hard, a subset of very challenging subjects in C-Eval that requires advanced reasoning abilities to solve. We conduct a comprehensive evaluation of the most advanced LLMs on C-Eval, including both English- and Chinese-oriented models. Results indicate that only GPT-4 could achieve an average accuracy of over 60%, suggesting that there is still significant room for improvement for current LLMs. We anticipate C-Eval will help analyze important strengths and shortcomings of foundation models, and foster their development and growth for Chinese users.

---

# C‑Eval：面向基础模型的多层级多学科中文评估套件 论文详细解读

### 背景：这个问题为什么难？

中文大语言模型（LLM）在过去几年里突飞猛进，但缺乏系统、规模化的中文测评平台。现有的中文 benchmark 大多聚焦于单一任务（如阅读理解、情感分析），覆盖的学科和难度层次非常有限，难以检验模型在“高中、大学、专业”层面的深度知识和推理能力。于是，研究者们只能用碎片化的测试结果来判断模型的真实水平，这让模型的进步方向和瓶颈难以定位。C‑Eval 正是为了解决“中文评测不够全面、难度分层不明确、学科覆盖单薄”这三个痛点而诞生的。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度学习模型，像 GPT‑4、ChatGPT 那样通过海量文本学习语言规律。可以类比为“会说话的百科全书”。
- **基础模型**：指在大规模通用语料上预训练、随后可以通过少量微调适配各种下游任务的模型。相当于“一块通用的原材料”，不同任务只需要少量加工。
- **多选题评测**：每道题给出若干备选答案，模型需要选出唯一正确项。类似于学校的客观题考试，便于自动批分。
- **难度层级**：本套件把题目划分为中学、高中、大学、专业四个层次，模拟不同教育阶段的认知要求。可以想象为“从小学到博士的梯度”。
- **学科覆盖**：52 个学科从人文社科到理工工程全覆盖，确保模型不只在某一领域强，而是整体能力得到检验。像是“一张全科考试卷”。
- **C‑Eval Hard**：从全部题目中挑选出最具挑战性的子集，专门用来衡量模型的高阶推理和知识整合能力。相当于“奥林匹克竞赛题”。
- **零样本评估**：模型直接在测试集上作答，不进行任何针对性的微调。类似于让学生在没有复习的情况下参加考试，考察其“即学即用”能力。

### 核心创新点
1. **全中文、全学科的统一基准 → 设计并发布了覆盖 52 门学科、四个难度层级的多选题库** → 研究者和企业现在可以用同一套标准对比不同模型的中文知识深度，而不必自行拼凑碎片化数据。  
2. **难度层级化评测 → 将题目按中学、高中、大学、专业四层划分** → 让模型的表现可以细粒度地映射到教育阶段，帮助定位模型在“基础知识”还是“专业推理”上更薄弱。  
3. **高难度子集 C‑Eval Hard → 从全量题库中抽取需要高级推理的题目** → 为已经在普通层面表现不错的模型提供更严苛的挑战，推动模型向真正的“专家水平”迈进。  
4. **跨语言基准对比 → 同时评测英文主导的 LLM（如 GPT‑4）和中文本土模型** → 揭示语言偏好对中文任务表现的影响，为多语言模型的研发提供实证依据。

### 方法详解
整体思路可以概括为三步：**题库构建 → 难度/学科标注 → 零样本评测**。

1. **题库构建**  
   - 研究团队从公开教材、历年考试、专业资格试题等渠道收集了大量多选题。每道题都包含题干、四到五个选项以及唯一正确答案。  
   - 为保证质量，所有题目经过人工校对，确保答案唯一且表述清晰。这里的“人工校对”相当于老师批改试卷，防止出现歧义。

2. **难度与学科标注**  
   - 依据题目来源（如中学教材 vs. 专业资格考试）以及题目本身的概念深度，团队将每题划分到四个难度层级之一。  
   - 同时，根据学科分类体系（如《中华人民共和国学科分类标准》），为每题贴上学科标签。这样每道题都有“年级+学科”的双重坐标，类似于在地图上标记出每座山的海拔和所属山脉。

3. **零样本评测流程**  
   - 对每个待测 LLM，直接把题目文本（包括题干和所有选项）喂入模型，要求模型输出选项的字母或编号。  
   - 采用 **“直接选择”** 的提示模板：*“以下哪项是正确答案？A. … B. … C. … D. … 请直接给出选项字母。”* 这种方式避免了额外的后处理步骤。  
   - 计算每个层级、每个学科的准确率，再取平均得到整体得分。  
   - 对 C‑Eval Hard 子集重复同样的评测流程，得到模型在高阶推理上的表现。

**最巧妙的地方**在于把“难度层级”和“学科标签”两维信息同时引入评测，这让结果不仅是一个单一的准确率，而是一个多维度的能力画像。研究者可以直接看到模型在“高中物理” vs. “大学哲学”上的差异，类似于学生成绩单的科目分布。

### 实验与效果
- **测试对象**：论文评测了多款最前沿的 LLM，包括 GPT‑4、GPT‑3.5、Claude、ChatGLM、BLOOM‑Z 等，覆盖了英文主导模型和中文本土模型。  
- **整体表现**：只有 GPT‑4 在全部层级的平均准确率突破了 60%（论文声称），其余模型普遍低于此阈值。具体的数值分布（如 30%‑50%）在原文中未给出。  
- **层级差异**：GPT‑4 在中学层级的得分最高，大学和专业层级的下降幅度相对明显，说明即使是最强模型，在专业深度上仍有提升空间。  
- **C‑Eval Hard**：在高难度子集上，所有模型的准确率均出现显著下滑，GPT‑4 仍保持领先，但具体数字未披露。  
- **消融实验**：论文未提供针对题库构建或提示模板的消融实验，只是对比了不同模型的整体表现。  
- **局限性**：作者承认评测仅限于多选题形式，无法覆盖生成式、对话式或开放式任务；此外，零样本评估可能低估了通过少量微调能显著提升的模型潜力。

### 影响与延伸思考
C‑Eval 首次提供了系统化、层级化、学科全覆盖的中文基准，迅速成为中文 LLM 研发的“标配”。后续不少工作（如 **MOSS‑Eval**、**Chinese-MMLU**）在数据规模、题型多样性上进行扩展，直接受到了 C‑Eval 思路的启发。对想进一步深入的读者，可以关注以下方向：

- **生成式评测**：设计能够评估模型写作、代码生成等开放式能力的中文基准。  
- **少样本微调效应**：研究在 C‑Eval 上进行少量示例学习（few‑shot）是否能显著提升准确率。  
- **跨语言对齐**：把 C‑Eval 与英文基准（如 MMLU）对应起来，探索多语言模型的知识迁移规律。  
- **动态题库**：利用模型自我生成题目并闭环评测，形成持续迭代的评测生态。

### 一句话记住它
**C‑Eval 用 52 门学科、四层难度的中文多选题，给大语言模型提供了“全科考试”式的能力画像。**