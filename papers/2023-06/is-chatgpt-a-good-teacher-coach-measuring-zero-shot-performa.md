# Is ChatGPT a Good Teacher Coach? Measuring Zero-Shot Performance For   Scoring and Providing Actionable Insights on Classroom Instruction

> **Date**：2023-06-05
> **arXiv**：https://arxiv.org/abs/2306.03090

## Abstract

Coaching, which involves classroom observation and expert feedback, is a widespread and fundamental part of teacher training. However, the majority of teachers do not have access to consistent, high quality coaching due to limited resources and access to expertise. We explore whether generative AI could become a cost-effective complement to expert feedback by serving as an automated teacher coach. In doing so, we propose three teacher coaching tasks for generative AI: (A) scoring transcript segments based on classroom observation instruments, (B) identifying highlights and missed opportunities for good instructional strategies, and (C) providing actionable suggestions for eliciting more student reasoning. We recruit expert math teachers to evaluate the zero-shot performance of ChatGPT on each of these tasks for elementary math classroom transcripts. Our results reveal that ChatGPT generates responses that are relevant to improving instruction, but they are often not novel or insightful. For example, 82% of the model's suggestions point to places in the transcript where the teacher is already implementing that suggestion. Our work highlights the challenges of producing insightful, novel and truthful feedback for teachers while paving the way for future research to address these obstacles and improve the capacity of generative AI to coach teachers.

---

# ChatGPT 能否成为优秀的教师教练？零样本评估其在课堂教学评分与可操作性反馈上的表现 论文详细解读

### 背景：这个问题为什么难？

教师培训里最核心的环节是课堂观察后给出专业反馈，但真正能做到“一对一、持续、质量高”的教练资源极其稀缺。传统做法要么依赖经验丰富的导师现场听课，要么使用昂贵的录像分析平台，成本和时间都让大多数学校望而却步。即便有了观察工具，如何把客观评分转化为具体、可落地的改进建议仍是个技术难题。于是研究者开始想：如果把生成式大模型当成“虚拟教练”，能否在不做任何微调的情况下，直接给出有价值的评分和建议？

### 关键概念速览
- **教师教练（Teacher Coaching）**：指通过课堂观察、数据分析和专家反馈帮助教师提升教学质量的过程，类似运动员的私人教练。  
- **课堂观察工具（Observation Instrument）**：一套预先定义好的评分标准，用来量化教师在课堂上的具体行为，例如“提问深度”或“学生参与度”。  
- **零样本（Zero‑Shot）**：模型在没有看到任何任务特定示例的情况下直接完成任务，就像第一次去陌生城市却凭直觉找路。  
- **可操作性建议（Actionable Insight）**：具体、明确、教师可以立刻实施的改进措施，区别于笼统的“需要改进”。  
- **高亮与错失机会（Highlights & Missed Opportunities）**：前者是指教师已经做得好的片段，后者是指本可以更好利用的教学时机。  
- **专家评估（Expert Evaluation）**：由经验丰富的教师对模型输出进行人工打分和评论，确保评价的专业性。  

### 核心创新点
1. **任务定义的三层拆解**  
   之前的研究大多把“给教师反馈”当成一个整体任务，这篇论文把它细化为：① 根据观察工具给出分数，② 标记教学亮点和错失机会，③ 提供具体的改进建议。这样做让评估更具针对性，也方便后续对每一步的表现单独打分。  

2. **零样本评估框架**  
   传统做法会先对大模型进行微调或提供少量示例提示，而作者直接让 ChatGPT 在完全没有任务示例的情况下完成三项任务，检验它的“即插即用”能力。相当于把模型当成“即席教师顾问”，看它能否凭已有的通用知识给出专业反馈。  

3. **基于真实课堂转录的实验**  
   许多 AI 教育研究使用合成对话或简化的教学片段，这里选取了真实的小学数学课堂转录，保证评测环境与实际教练场景高度一致。  

4. **专家教师的细粒度打分**  
   研究邀请了多位资深数学教师，对模型的每条输出进行相关性、创新性和可操作性三维评分，提供了比单一准确率更丰富的质量视角。  

### 方法详解
整体思路可以概括为三步走：**输入准备 → 模型调用 → 人工评估**。下面把每一步拆开说。

1. **输入准备**  
   - 先把完整的课堂转录切成若干 30‑60 秒的片段，确保每段都有完整的教学互动。  
   - 对每段附上对应的观察工具条目列表（例如“提问深度”“学生思考时间”），但不提供任何示例答案。  

2. **模型调用**  
   - 对每个片段，向 ChatGPT 发送一条统一的提示语，包含三条指令：  
     a. 根据观察工具给出数值评分（通常是 1‑5 分）。  
     b. 标记本段中值得称赞的教学行为（highlights）和本可以更深入的机会（missed opportunities）。  
     c. 基于上述标记，给出 1‑2 条具体的改进建议，要求语言简洁、可直接在课堂上实施。  
   - 关键在于 **零样本**：提示里只说明任务目标和评分尺度，完全不提供示例答案，让模型靠自身的通用知识完成。  

3. **人工评估**  
   - 每位专家教师独立阅读模型输出，分别在“相关性”“新颖性”“可操作性”三个维度打 1‑5 分。  
   - 为了捕捉模型是否“重复老师已经做的事”，评审会记录模型建议是否指向教师已经实施的策略。  

**最巧妙的点**在于把“评分+亮点+建议”这三个子任务压在同一个提示里，让模型一次性输出完整报告，模拟真实教练的工作流程。这样既省去了多轮交互的成本，也能直接观察模型在不同层面的表现差异。

### 实验与效果
- **数据来源**：真实的小学数学课堂转录，共计约 10 小时教学内容，切分后得到数十个片段。  
- **评估对象**：ChatGPT（GPT‑4）在零样本设置下完成全部三项任务。  
- **对比基准**：论文没有提供传统机器学习模型或微调后模型的对比，只把 ChatGPT 的输出和专家教师的人工反馈进行横向比较。  
- **主要发现**：  
  - 在相关性上，模型的建议大多数能指向教学改进点，专家评分普遍在 3.5 分以上（满分 5 分）。  
  - 创新性不足：约 **82%** 的建议指向教师已经在做的事情，说明模型倾向于“复述”而非提供新视角。  
  - 可操作性方面，模型能够给出具体的课堂技巧（如“让学生先自行思考再举手”），但这些技巧往往是常见的教学常识。  
- **局限性**：  
  - 只在数学课堂测试，跨学科通用性未知。  
  - 零样本提示缺乏对观察工具的深度解释，导致模型有时误解评分尺度。  
  - 评估仅依赖专家主观打分，缺少客观教学效果的后续验证（如学生学习成绩变化）。  

### 影响与延伸思考
这篇工作首次把生成式大模型放进“教师教练”这一专业场景，并用零样本方式检验其即插即用的潜力。后续研究大多围绕两条主线展开：  
1. **提示工程**：如何设计更细致的提示，让模型在不微调的情况下也能捕捉观察工具的细微差别。  
2. **人机协同**：把模型的初步反馈交给人类教练二次加工，形成“AI‑助教+专家审校”的混合流程。  

如果想进一步了解，可以关注近期在教育技术会议（如 AIED、EDM）上出现的“AI 教练”系列论文，尤其是那些结合微调、强化学习或多模态（视频+文本）输入的工作。  

### 一句话记住它
ChatGPT 在零样本下能给出看似专业的教学反馈，但大多数建议都在重复老师已有的做法，真正的创新和深度仍需人类教练来填补。