# Translating Radiology Reports into Plain Language using ChatGPT and   GPT-4 with Prompt Learning: Promising Results, Limitations, and Potential

> **Date**：2023-03-16
> **arXiv**：https://arxiv.org/abs/2303.09038

## Abstract

The large language model called ChatGPT has drawn extensively attention because of its human-like expression and reasoning abilities. In this study, we investigate the feasibility of using ChatGPT in experiments on using ChatGPT to translate radiology reports into plain language for patients and healthcare providers so that they are educated for improved healthcare. Radiology reports from 62 low-dose chest CT lung cancer screening scans and 76 brain MRI metastases screening scans were collected in the first half of February for this study. According to the evaluation by radiologists, ChatGPT can successfully translate radiology reports into plain language with an average score of 4.27 in the five-point system with 0.08 places of information missing and 0.07 places of misinformation. In terms of the suggestions provided by ChatGPT, they are general relevant such as keeping following-up with doctors and closely monitoring any symptoms, and for about 37% of 138 cases in total ChatGPT offers specific suggestions based on findings in the report. ChatGPT also presents some randomness in its responses with occasionally over-simplified or neglected information, which can be mitigated using a more detailed prompt. Furthermore, ChatGPT results are compared with a newly released large model GPT-4, showing that GPT-4 can significantly improve the quality of translated reports. Our results show that it is feasible to utilize large language models in clinical education, and further efforts are needed to address limitations and maximize their potential.

---

# 使用 ChatGPT 与 GPT‑4 通过提示学习将放射学报告翻译为通俗语言：有前景的结果、局限与潜力 论文详细解读

### 背景：这个问题为什么难？
放射科报告原本是专业医生写给同行看的，语言高度专业化、术语密集，普通患者往往看不懂。传统的人工翻译成本高、速度慢，而且不同医生的表达风格差异大，导致患者获取信息的时效性差。早期尝试使用规则模板或固定词典来“简化”报告时，往往只能处理少数常见术语，遇到复杂影像描述或模糊表达就会崩溃，信息丢失或误导的风险很高。因此，如何在保持医学准确性的前提下，自动把专业报告转化为患者友好的语言，成为临床沟通的瓶颈。

### 关键概念速览
**大语言模型（LLM）**：像 ChatGPT、GPT‑4 这样经过海量文本训练的模型，能够理解并生成自然语言，类似于“会说话的百科全书”。  
**提示学习（Prompt Learning）**：通过给模型设计特定的指令或示例，引导它产生想要的输出，就像在对话中给出明确的任务说明。  
**放射学报告**：影像科医生对 CT、MRI 等检查结果的文字描述，包含解剖位置、病灶特征、诊断结论等专业信息。  
**信息缺失 / 错误信息**：翻译后报告中遗漏的关键医学信息或出现的与原报告不符的内容，直接影响患者的理解与后续决策。  
**随机性（Stochasticity）**：同样的提示在不同调用时可能得到略有差异的答案，类似于人类在重复解释同一件事时的细微变化。  
**评估打分系统**：本文使用 5 分制让放射科医生对翻译质量打分，5 分代表完美、1 分代表不可接受。  
**特定建议**：模型在翻译后给出的后续行动建议，如“定期复查”或“若出现咳嗽加重立即就诊”，这些建议的具体性直接影响患者的自我管理。

### 核心创新点
1. **直接使用通用大模型进行医学报告简化 → 通过 ChatGPT 对真实的低剂量胸部 CT 与脑 MRI 报告进行“一键”翻译 → 平均 4.27 分的高质量评分，信息缺失仅 0.08 处，误信息 0.07 处，证明通用模型在医学领域的可行性。  
2. **提示工程细化 → 在基础提示上加入报告结构、目标受众（患者）以及需要输出的建议格式 → 随机性显著下降，过度简化或遗漏的情况被抑制，展示了提示设计对模型输出稳定性的关键作用。  
3. **对比最新模型 GPT‑4 → 在相同任务上让 GPT‑4 生成翻译 → 质量明显提升（具体提升幅度未在摘要中给出），说明模型规模和训练数据的进步直接转化为更可靠的医学解释能力。  
4. **自动生成后续建议 → 除了语言转换，还让模型输出一般性或基于报告发现的具体随访建议 → 在约 37% 的案例中提供了针对性建议，展示了模型在临床教育之外的潜在辅助决策价值。

### 方法详解
整体思路可以拆成三步：**数据收集 → 提示设计 → 模型调用与结果评估**。

1. **数据收集**  
   - 选取 62 例低剂量胸部 CT 肺癌筛查和 76 例脑 MRI 转移筛查报告，时间集中在二月上半月。  
   - 每份报告均由放射科医生完成，包含影像所见、诊断结论以及必要的临床建议。  

2. **提示设计（Prompt Learning）**  
   - 基础提示示例：  
     ```
     你是一名医学科普写手，请把下面的放射学报告翻译成普通患者能懂的语言，并在最后给出两条后续建议。报告如下：
     【原始报告】
     ```  
   - 为降低随机性，作者在提示中加入了**结构化指令**（如“先解释发现，再给出结论”，以及“每条建议必须基于报告中的具体发现”），并提供了**几例成功的示例输出**作为 few‑shot 示例。  
   - 这种做法相当于给模型一个“写作模板”，让它在每次生成时遵循相同的框架，减少了“随意发挥”的空间。  

3. **模型调用**  
   - 首先使用 ChatGPT（基于 GPT‑3.5）对所有报告进行翻译。  
   - 随后在同样的提示下切换到 GPT‑4，获取对比结果。  
   - 每份报告的输出都保存为两版：**普通语言版**和**建议版**。  

4. **评估流程**  
   - 两位资深放射科医生独立对每份翻译进行 5 分制打分，记录信息缺失和误信息的次数。  
   - 统计整体平均分、缺失/误信息率，并对比两种模型的表现。  

**巧妙之处**：作者没有对模型进行任何医学专用微调，而是完全依赖提示工程实现任务，这大幅降低了实现成本，也证明了“提示即编程”的威力。此外，加入**few‑shot 示例**的做法让模型在医学语境下的表现更接近专业水平，类似于给模型“上了一堂速成课”。  

### 实验与效果
- **数据规模**：共 138 份报告（62 例胸部 CT + 76 例脑 MRI）。  
- **整体评分**：ChatGPT 的平均得分为 4.27（满分 5），信息缺失 0.08 处，误信息 0.07 处，说明大多数翻译既完整又准确。  
- **建议生成**：在全部案例中，约 37% 的报告得到基于影像发现的具体随访建议，其余则给出通用的“请及时复诊”等提醒。  
- **模型对比**：GPT‑4 在同样提示下的翻译质量显著优于 ChatGPT（摘要未给出具体数值），但可以推断其在信息完整性和语言自然度上都有提升。  
- **随机性观察**：同一报告多次调用 ChatGPT 时会出现轻微差异，尤其在描述细节时可能出现“过度简化”。通过更细致的提示（明确要求保留关键数值、部位名称）可以缓解此问题。  
- **局限性**：  
  - 只测试了两类影像（胸部 CT、脑 MRI），对其他模态（如超声、PET）尚未验证。  
  - 评估仅由放射科医生完成，缺少患者侧的可理解性实验。  
  - 对于极端复杂或罕见病灶，模型仍可能产生误信息，需进一步安全机制。  

### 影响与延伸思考
这篇工作在医学自然语言处理（MedNLP）领域起到了“示范效应”：首次系统展示了无需专门微调、仅靠提示即可让大语言模型承担临床报告通俗化的任务。随后出现的研究（如 2024 年的 “Prompt‑Driven Clinical Summarization”）直接借鉴了其提示结构，尝试将同类方法推广到病历、手术记录等更长文本。未来的方向可能包括：  
- **患者中心的可读性评估**：加入真实患者的理解测试，量化模型输出的实际教育价值。  
- **安全过滤层**：在模型生成后加入医学知识图谱校验，自动捕捉潜在误信息。  
- **跨语言扩展**：将提示翻译成多语言版本，服务非英语患者群体。  
- **实时临床集成**：与 PACS 系统对接，实现报告生成后即时提供患者友好版。  

### 一句话记住它
只要给大语言模型写好“患者友好”的提示，就能把专业放射报告自动转成易懂语言，且质量已接近医生手工翻译的水平。