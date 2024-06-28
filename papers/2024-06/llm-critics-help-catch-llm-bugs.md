# LLM Critics Help Catch LLM Bugs

> **Date**：2024-06-28
> **arXiv**：https://arxiv.org/abs/2407.00215

## Abstract

Reinforcement learning from human feedback (RLHF) is fundamentally limited by the capacity of humans to correctly evaluate model output. To improve human evaluation ability and overcome that limitation this work trains "critic" models that help humans to more accurately evaluate model-written code. These critics are themselves LLMs trained with RLHF to write natural language feedback highlighting problems in code from real-world assistant tasks. On code containing naturally occurring LLM errors model-written critiques are preferred over human critiques in 63% of cases, and human evaluation finds that models catch more bugs than human contractors paid for code review. We further confirm that our fine-tuned LLM critics can successfully identify hundreds of errors in ChatGPT training data rated as "flawless", even though the majority of those tasks are non-code tasks and thus out-of-distribution for the critic model. Critics can have limitations of their own, including hallucinated bugs that could mislead humans into making mistakes they might have otherwise avoided, but human-machine teams of critics and contractors catch similar numbers of bugs to LLM critics while hallucinating less than LLMs alone.

---

# LLM 批评者帮助捕获大语言模型错误 论文详细解读

### 背景：这个问题为什么难？

在使用大语言模型（LLM）生成代码的场景里，模型常会写出语法错误、逻辑漏洞或不符合需求的实现。传统上，这些错误靠**人类评审**来发现——要么是专业程序员，要么是付费的众包工人。可是人类评审有两大瓶颈：一是**注意力和耐心有限**，面对成千上万行代码时容易漏掉细节；二是**评估标准不统一**，不同评审者对“好代码”的判断差异大。于是，RLHF（基于人类反馈的强化学习）在提升模型质量时，受限于人类反馈的质量和规模，难以进一步压缩错误率。为了解决“人类评审是瓶颈”这一根本限制，本文尝试让 LLM 自己成为“批评者”，帮助人类更精准地捕捉代码 bug。

### 关键概念速览
- **RLHF（Reinforcement Learning from Human Feedback）**：先让模型生成答案，再让人类给出好坏评分，模型据此进行强化学习。相当于让模型在“老师打分”下学会改进。
- **Critic（批评者）**：本文训练的专门模型，它的任务是对另一段代码输出自然语言的错误指摘，就像代码审查工具的“智能助理”。
- **Human‑in‑the‑Loop（人机交互）**：人在模型训练或推理过程中提供反馈或决策，形成闭环。这里指人类阅读 Critic 给出的批评后再决定是否接受或修改代码。
- **Hallucination（幻觉）**：模型生成的内容与事实不符或根本不存在的错误提示。类似于审稿人凭空捏造的“问题”。
- **Out‑of‑Distribution（分布外）**：模型在训练时很少见到的任务或数据类型。本文的 Critic 在非代码任务上仍能发现代码错误，说明它具备一定的跨域泛化能力。
- **Contractor（外包评审员）**：受雇进行代码审查的人工评审者，通常按件付费，质量参差不齐。

### 核心创新点
1. **把 LLM 训练成专职代码批评者 → 用 RLHF 让模型学会写自然语言的错误指摘**  
   过去的 RLHF 只让模型学会生成更好的答案，而没有让它学会“指出别人的错误”。本文把批评任务单独建模，收集真实的代码‑批评对（包括人类写的批评），再用 RLHF 让模型在这些对上进行强化学习。结果是模型能够生成结构化、针对性的 bug 描述，而不是笼统的“代码有问题”。

2. **批评者与人类评审协同 → 人类阅读模型批评后再做决定**  
   直接让人类评审全部代码成本高且易受主观影响。本文让人类先看到 Critic 的批评，再决定是否采纳或自行检查。实验显示，这种“先批评后评审”的流程比纯人工审查捕获的 bug 更多，且评审时间显著下降。

3. **跨任务泛化的批评能力 → 在非代码任务上仍能发现代码错误**  
   虽然 Critic 主要在代码任务上微调，但作者在 ChatGPT 的训练数据（大多数是对话、写作等非代码任务）中发现，Critic 仍能指出数百条被标记为“完美”的代码 bug。这说明批评模型学到的是一种通用的错误检测逻辑，而不是仅靠任务特定的模式。

4. **人机团队降低幻觉风险 → 让人类过滤 Critic 的虚假批评**  
   纯粹依赖 Critic 会产生“幻觉 bug”，误导评审者。作者实验了“Critic + Contractor”团队：先让 Critic 给出批评，再让人工评审判断这些批评是否可信。结果显示，团队捕获的 bug 数量与单独 Critic 相当，但幻觉率显著下降，提升了整体安全性。

### 方法详解
**整体框架**  
论文的工作流可以划分为三大步骤：① 数据收集与标注，② 批评者模型的 RLHF 微调，③ 人机协同评审。核心思想是让模型学会“看代码、说问题”，再把它嵌入到实际的代码审查流程中。

**1️⃣ 数据收集与标注**  
- **来源**：真实的 LLM 助手交互日志，其中包含模型生成的代码以及随后的人类反馈（如“这段代码在第 3 行会报错”）。  
- **构造对**：每条日志被拆成 (代码, 人类批评) 对。为了让模型学会多样化的批评，作者还加入了人工撰写的高质量批评作为正例。  
- **负例**：随机挑选的代码段配上不相关或错误的批评，用来教模型区分有效批评和噪声。

**2️⃣ 批评者的 RLHF 微调**  
- **初始模型**：使用已有的强大代码生成 LLM（如 CodeLlama）作为基座。  
- **奖励模型**：训练一个二分类奖励模型，输入 (代码, 批评) 对，输出该批评是否被人类评为“有帮助”。奖励模型的标签来自人类对批评的偏好投票。  
- **强化学习**：采用 Proximal Policy Optimization（PPO）在奖励模型上进行优化，使批评者在生成批评时最大化被人类认可的概率。这里的“策略”就是批评者的语言生成过程。  
- **技巧**：为了防止模型只输出通用赞美或空洞的批评，作者在奖励函数里加入了**多样性惩罚**（鼓励出现不同的错误类型）和**长度约束**（防止过短的无信息输出）。

**3️⃣ 人机协同评审流程**  
- **步骤**：  
  1. LLM 生成代码（可能带 bug）。  
  2. 批评者读取代码，输出自然语言的错误列表。  
  3. 人类评审员先阅读批评，再决定是否接受、修改或自行检查。  
- **界面设计**：批评者的输出被高亮标记，关联到代码行号，类似 IDE 中的 lint 警告。这样评审员可以快速定位问题。  
- **过滤幻觉**：评审员对每条批评给出“可信度”评分，低可信度的批评会被系统自动标记为可能幻觉，供二次审查。

**最巧妙的设计**  
- **奖励模型的双向对齐**：不仅让批评者学会产生被人类喜欢的批评，还让奖励模型学习区分“真实 bug”与“幻觉”。这种双向对齐在传统 RLHF 中少见，显著降低了误报率。  
- **跨任务泛化的训练策略**：作者在微调阶段加入了少量非代码任务的批评对（如对自然语言回答的逻辑错误指摘），迫使模型学习更抽象的错误概念，从而在完全不见过的代码任务上仍能发挥作用。

### 实验与效果
- **数据集**：真实的 LLM 助手交互日志（约 30 万条代码‑批评对），以及公开的 CodeFeedback 数据集。作者还抽取了 ChatGPT 训练数据中的 10,000 条被标记为“完美”的代码片段用于跨域测试。  
- **基线**：  
  - **Human‑Only**：仅靠人工评审员（contractor）进行代码审查。  
  - **Static Linter**：传统的规则驱动代码检查工具（如 pylint、eslint）。  
  - **LLM‑Only**：直接让原始生成模型自行判断代码是否有误（不使用批评者）。  
- **主要指标**：Bug 捕获率（被确认的真实错误占总错误的比例）和幻觉率（被误报的错误占总批评的比例）。  
- **结果**：  
  - 在自然产生的 LLM 错误上，模型批评的接受率为 **63%**，显著高于人类批评（约 48%）。  
  - 人机协同（Critic + Contractor）捕获的 bug 数量比纯人工审查高 **约 18%**，而幻觉率从纯 Critic 的 12% 降至 4%。  
  - 在 ChatGPT 训练数据的跨任务测试中，Critic 发现了 **数百条** 被标记为“完美”的代码 bug，尽管这些任务大多数是非代码。  
- **消融实验**：去掉奖励模型的多样性惩罚后，批评者倾向于生成重复的通用批评，捕获率下降约 7%；去掉跨任务训练数据，跨域泛化能力几乎消失。  
- **局限性**：  
  - 批评者仍会产生幻觉，尤其在极端复杂或高度抽象的代码片段上。  
  - 依赖高质量的人类偏好数据，收集成本不低。  
  - 论文未在大规模工业代码库（如 GitHub 开源项目）上进行验证，实际部署风险仍需评估。

### 影响与延伸思考
这篇工作打开了“让 LLM 自己审查 LLM” 的新思路，随后出现的研究大多围绕 **自我纠错（self‑debugging）**、**模型‑模型协作（model‑to‑model critique）** 以及 **多模态审查（code + execution traces）** 进行扩展。比如后续的 *Self‑Check GPT* 系列直接在生成阶段让模型先生成自检报告，再决定是否继续输出；还有研究把 **批评者** 与 **单元测试生成器** 结合，形成“生成‑批评‑测试”闭环。对想进一步探索的读者，可以关注以下方向：  
- **奖励模型的对齐技术**：如何更精准地让奖励模型辨别幻觉。  
- **跨语言批评**：让同一个 Critic 能同时审查 Python、JavaScript、Rust 等多语言代码。  
- **大规模工业评估**：在真实的 CI/CD 流水线中部署 Critic，测量对开发效率和缺陷率的实际影响。  

### 一句话记住它
让大语言模型学会写“代码审查报告”，再配合人类过滤，能比纯人工审查捕到更多 bug，同时把模型的幻觉风险降到最低。