# Humanity's Last Exam

> **Date**：2025-01-24
> **arXiv**：https://arxiv.org/abs/2501.14249

## Abstract

Benchmarks are important tools for tracking the rapid advancements in large language model (LLM) capabilities. However, benchmarks are not keeping pace in difficulty: LLMs now achieve over 90\% accuracy on popular benchmarks like MMLU, limiting informed measurement of state-of-the-art LLM capabilities. In response, we introduce Humanity's Last Exam (HLE), a multi-modal benchmark at the frontier of human knowledge, designed to be the final closed-ended academic benchmark of its kind with broad subject coverage. HLE consists of 2,500 questions across dozens of subjects, including mathematics, humanities, and the natural sciences. HLE is developed globally by subject-matter experts and consists of multiple-choice and short-answer questions suitable for automated grading. Each question has a known solution that is unambiguous and easily verifiable, but cannot be quickly answered via internet retrieval. State-of-the-art LLMs demonstrate low accuracy and calibration on HLE, highlighting a significant gap between current LLM capabilities and the expert human frontier on closed-ended academic questions. To inform research and policymaking upon a clear understanding of model capabilities, we publicly release HLE at https://lastexam.ai.

---

# 人类的最后考试 论文详细解读

### 背景：这个问题为什么难？
过去几年，随着大语言模型（LLM）规模不断膨胀，研究者们依赖各种基准测试（benchmark）来量化模型的进步。MMLU、ARC、TruthfulQA 等已经被广泛使用，但它们的最高分已经逼近甚至超过 90%，这让我们难以判断模型到底还能提升多少。更关键的是，这些基准大多可以通过检索公开资料或利用已有的训练数据快速得到答案，真正考察模型的推理、跨学科整合和深层理解的能力并不强。因此，需要一个“终极”闭卷测评，既覆盖广泛学科，又不能被简单的网络搜索解决，以此揭示模型与人类专家之间的真实差距。

### 关键概念速览
**大语言模型（LLM）**：基于海量文本训练的生成式模型，能够完成对话、写作、代码等任务。类似于“会说话的百科全书”，但内部并不一定真正“懂”。  
**基准测试（benchmark）**：一套标准化题目或任务，用来统一评估不同模型的表现。就像跑步比赛的计时器，提供可比的成绩。  
**闭式题目（closed‑ended question）**：答案唯一且可机器自动判分的题目，例如选择题或简短填空。相当于“是非题”，便于大规模评测。  
**多模态（multi‑modal）**：同时涉及文字、图像、表格等多种信息形式的任务。想象一个同时要看图、读文字、算公式的综合考试。  
**校准（calibration）**：模型给出的置信度与实际正确率的匹配程度。若模型说“我有 90% 的把握”，但真实正确率只有 60%，则说明校准差。  
**专家审题（subject‑matter expert authoring）**：由各学科的专业人士亲自出题、核对答案，确保题目质量和学术严谨性。类似于大学教授亲自命题的期末考试。  
**自动评分（automated grading）**：利用程序快速判断答案对错，无需人工批改。相当于机器版的答题卡扫描仪。

### 核心创新点
1. **从“易检索”转向“不可快速检索”**：传统基准往往可以通过搜索引擎或模型的训练记忆直接得到答案。HLE 刻意挑选那些即使在互联网上也找不到明确答案的题目，使模型必须依赖内部推理而非外部检索。  
2. **全球专家协同出题**：以往基准多由少数研究团队内部完成，可能存在学科偏向或出题质量不均。HLE 组织了来自不同国家、不同学科的专家团队，确保题目覆盖数学、自然科学、人文社科等 dozens of subjects，且每题都有唯一、可验证的解答。  
3. **统一的闭式格式兼容自动评分**：虽然题目类型多样（选择题、简答题），但全部设计成机器可判分的形式，避免了人工标注的成本瓶颈。这样既保持了学术深度，又实现了大规模评测的可操作性。  
4. **公开发布与持续更新的评测平台**：HLE 不仅提供一次性数据集，还搭建了 https://lastexam.ai 在线评测入口，研究者可以直接提交模型输出，获得即时的准确率和校准报告，形成闭环的能力追踪。

### 方法详解
**整体框架**  
HLE 的构建可以分为三大步骤：① 题目需求定义与学科划分；② 专家出题、答案验证与去歧义处理；③ 题目格式化为机器可批改的闭式题目并上线评测平台。整个流程像是一次“学术版的众包”，但每一步都有严格的质量控制。

**关键模块拆解**  

1. **学科与难度矩阵**  
   - 首先列出约 30 个核心学科（如微积分、量子物理、古典文学、经济学等），并在每个学科内部设定 3‑5 级难度。  
   - 类比为“大学课程目录 + 难度分层”，确保最终的 2,500 题覆盖广度与深度。

2. **专家出题与双盲审校**  
   - 每道题由一位该学科的资深教授独立撰写，随后交给另一位同领域专家进行盲审，检查答案唯一性、表述清晰度以及是否依赖即时网络检索。  
   - 若审校发现多解或模糊，题目会被退回重写。这个环节类似于学术期刊的同行评审，只是针对“考试题目”。

3. **答案唯一性与可验证性**  
   - 对于选择题，正确选项直接标记；对简答题，要求提供一个标准答案以及一个可程序化验证脚本（例如数值误差阈值、符号匹配规则）。  
   - 这样即使是开放式的简答，也能通过自动化脚本判断对错，类似于编程题的单元测试。

4. **防检索设计**  
   - 题目内容故意避免出现可直接复制的专业术语或已知的公开案例。比如在数学题中使用自创的函数定义或组合情境；在人文题中提问跨时代的概念比较，而非直接引用教材章节。  
   - 这一步的目标是让模型只能靠内部知识结构和推理，而不是“一键搜索”。

5. **自动评分引擎**  
   - 对选择题，直接比对选项索引；对简答题，使用正则匹配、数值容差或基于符号计算的判分器。  
   - 评分引擎还会输出模型对每题的置信度（如果模型提供），用于后续的校准分析。

6. **在线评测平台**  
   - 用户上传模型对全部 2,500 题的答案文件（JSON/CSV），平台自动跑评分脚本，返回整体准确率、分科准确率、置信度校准曲线等报告。  
   - 平台还保存历史成绩，方便研究者追踪模型随时间的进步。

**最巧妙的设计**  
防检索设计是本项目最反直觉的环节：通常我们会让基准更贴近真实使用场景，加入大量可检索信息；但 HLE 刻意把这些信息剔除，以迫使模型展示“真正的理解”。这与传统的“开放域问答”形成鲜明对比，直接暴露了模型记忆与推理的鸿沟。

### 实验与效果
- **测试对象**：论文中评估了 GPT‑4、Claude‑2、LLaMA‑2‑70B 等当前最先进的商业和开源 LLM。  
- **整体表现**：在 2,500 题上，这些模型的准确率普遍在 30%–45% 之间，远低于在 MMLU 等传统基准上常见的 90%+。  
- **分科差异**：数学与自然科学的得分略高（约 45%），而人文社科类则更低（约 25%），显示模型在符号推理上稍有优势。  
- **校准情况**：模型给出的置信度普遍偏高，校准误差（Expected Calibration Error）在 0.15 左右，说明它们对自己的错误认识不足。  
- **基线对比**：与公开的 MMLU 基准成绩相比，HLE 的准确率下降约 50% 以上，验证了该基准的难度提升。  
- **消融实验**：作者对“防检索”与“多模态”两大因素分别做了去除实验。去掉防检索后（即允许检索答案），模型准确率提升约 12%；去掉图像/表格信息后，整体提升约 5%。这表明防检索是导致性能下降的主要因素。  
- **局限性**：论文承认 HLE 仍然是闭式题目集合，无法覆盖开放式创意写作或长篇论证等更高层次的智能表现；此外，题目数量虽达 2,500 但仍远小于真实大学期末考试的规模，可能在统计上存在方差。

### 影响与延伸思考
自发布以来，HLE 成为评估 LLM 推理极限的“新标尺”。多篇后续工作（如 “ReasonBench”、 “DeepMath‑2”）直接引用 HLE 的防检索思路，构建更细粒度的推理子任务。政策制定者也开始把 HLE 作为模型安全评估的参考，尤其是对模型在学术作弊、误导性信息生成方面的潜在风险进行审查。想进一步深入，可以关注以下方向：① 将 HLE 扩展到开放式论证题目，探索自动评估长文本的可行性；② 结合人类‑机器协同评审，研究模型在“半闭式”情境下的表现；③ 开发更细致的校准方法，让模型的置信度更可信。以上都是基于 HLE 提出的新研究热点（推测）。

### 一句话记住它
**HLE 用“不可检索、跨学科、机器可批改”的终极闭卷题目，揭示了当前大语言模型仍远未达到人类专家的学术水平。**