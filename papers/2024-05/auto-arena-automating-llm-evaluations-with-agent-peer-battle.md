# Auto-Arena: Automating LLM Evaluations with Agent Peer Battles and   Committee Discussions

> **Date**：2024-05-30
> **arXiv**：https://arxiv.org/abs/2405.20267

## Abstract

As LLMs continuously evolve, there is an urgent need for a reliable evaluation method that delivers trustworthy results promptly. Currently, static benchmarks suffer from inflexibility and unreliability, leading users to prefer human voting platforms like Chatbot Arena. However, human evaluations require significant manual effort. To address this, we propose the Auto-Arena, an innovative framework that automates the entire evaluation process using LLM-powered agents. Firstly, an LLM examiner generates questions. Then, two LLM candidates engage in a multi-round peer battle based on individual questions, aiming at revealing their true performance differences. Finally, a committee of LLM judges collaboratively discusses and decides the winner, reducing bias and enhancing fairness. During the peer battles, we observe intriguing scenarios where the LLM candidates display competitive behaviors and even learn from the opponents. In our extensive experiments involving 15 recent LLMs, Auto-Arena shows a 92.14% correlation with human preferences, surpassing all previous expert-annotated benchmarks without any manual efforts. As a result, Auto-Arena offers a promising alternative to current human evaluation platforms for evaluating LLMs automatically.

---

# Auto-Arena：使用代理对战与委员会讨论实现大语言模型自动评估 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）更新速度极快，传统的静态基准测试往往只能在模型发布前准备好题目，模型一旦升级就会出现“数据泄漏”或“过时”问题，导致评估结果失真。人类投票平台（如 Chatbot Arena）虽然能捕捉最新模型的表现，却需要大量人工标注、组织投票，成本高且难以规模化。于是，业界缺少一种既能快速生成评测，又能保持公平可靠的自动化方案。

### 关键概念速览
- **LLM 考官**：由大语言模型充当的出题者，负责生成多样化、针对性的评测问题。相当于让模型自己当老师出卷。
- **候选模型**：待评估的两款大语言模型，它们会在同一问题上展开“对决”。可以把它们想象成比赛中的两支队伍。
- **多轮对战（Peer Battle）**：候选模型轮流回答、反驳、补充，形成多轮交互，旨在让细微差别暴露出来，类似于辩论赛的来回论证。
- **委员会（Committee）**：由若干 LLM 评审组成的讨论小组，负责对对战过程进行集体评议，最终投票决定胜者。像是多位裁判一起商量判罚，降低单一模型的偏见。
- **偏差消减**：通过让多个评审模型共同讨论，削弱单个模型固有的倾向性，提升评估公平性。
- **人类偏好相关性**：评估结果与真实用户投票的一致程度，用百分比表示。数值越高说明自动评估越贴近人类感受。

### 核心创新点
1. **全链路自动化 → LLM 考官 → 省去人工出题**  
   过去的自动评测往往依赖预先准备好的题库，难以覆盖新出现的能力。Auto‑Arena 让一个大模型自行生成评测问题，使评测内容随模型进化而动态更新，解决了题库陈旧的问题。

2. **对手式多轮交互 → 两个候选模型互相“辩论” → 揭示细粒度差异**  
   传统的“一问一答”评测只能捕捉表层表现。这里引入了多轮对战，让模型在同一话题上不断追问、纠错、补充，类似于两位棋手在对弈中逐步显露实力，从而更精准地区分强弱。

3. **LLM 评审委员会 → 多模型共同讨论 → 降低单点偏差**  
   单一评审模型的判断容易受训练数据或提示词影响。Auto‑Arena 让若干评审模型一起讨论、投票，类似于学术会议的评审团，显著提升了评估的客观性和鲁棒性。

4. **高相关性验证 → 92.14% 与人类偏好吻合 → 超越所有已有人工标注基准**  
   通过在 15 种最新 LLM 上实验，作者展示了自动评估与真实用户投票的高度一致性，证明了上述设计在实际使用中的有效性。

### 方法详解
**整体框架**  
Auto‑Arena 把评估过程拆成三大阶段：出题、对战、裁决。每一步都由大语言模型自行完成，形成闭环的自动评估流水线。

**1️⃣ LLM 考官生成问题**  
- 输入：评估目标的描述（如“回答开放式问答”“进行代码生成”等）。  
- 过程：考官模型依据提示生成一组多样化的问题，每个问题都配有明确的评分维度（准确性、流畅度、创新性等）。  
- 类比：就像老师先出题，再让学生去答。

**2️⃣ 候选模型的多轮对战**  
- 第一步，模型 A 接收问题并给出初始答案。  
- 第二步，模型 B 读取 A 的答案，针对其可能的缺陷或补充点进行追问或直接给出自己的答案。  
- 随后交替进行数轮（默认 3‑5 轮），每轮都在前一轮的输出基础上继续深化。  
- 关键点在于让模型“看到”对手的思路，从而被迫展示更完整的能力或纠正自身错误。  
- 这一步的输出是一段对话日志，记录了两模型的交锋细节。

**3️⃣ 委员会讨论与裁决**  
- 召集若干评审模型（通常 3‑5 个），每个模型读取完整的对战日志。  
- 每个评审模型先独立给出评分，然后在“讨论阶段”共享各自的评分理由，进行多轮对话。  
- 最终，所有评审模型投票决定哪一方表现更好。投票采用多数制或加权平均，得到最终胜者。  
- 这种“集体讨论”机制类似于法官合议庭，能够相互纠正偏见，提升判决的公正性。

**最巧妙的设计**  
- **对战中的学习效应**：模型在看到对手的答案后会主动改进自己的表述，这种自我提升在传统单向评测里根本不存在。  
- **评审委员会的多模态对话**：评审模型之间的讨论不是简单的数值汇总，而是通过自然语言交流共享观点，类似于人类评审的辩论过程。

### 实验与效果
- **实验对象**：作者选取了 15 种近期发布的主流大语言模型，覆盖不同规模、不同训练数据的模型。  
- **评测任务**：包括开放式问答、代码生成、推理解释等多种场景，确保评估的广度。  
- **基线对比**：与传统人工投票平台、已有的专家标注基准以及静态数据集评测结果进行比较。  
- **核心结果**：Auto‑Arena 与人类偏好的相关性达到 **92.14%**，显著高于所有对比基准，且完全不需要人工干预。  
- **消融实验**：论文中对“多轮对战”和“评审委员会”分别做了去除实验，发现去掉任意一环相关性会下降数个百分点，验证了每个模块的贡献。  
- **局限性**：作者指出评审模型本身的能力上限仍会影响最终判决；在极端专业领域（如医学诊断）仍可能需要人工复核。  

### 影响与延伸思考
Auto‑Arena 的出现让业界看到了“全自动、可扩展的 LLM 评估”并非遥不可及。后续工作开始探索更细粒度的对战策略（如引入对手的“策略库”），以及将评审委员会扩展为跨模态模型（如视觉‑语言模型）共同讨论。还有研究尝试把对战过程嵌入模型训练循环，实现“评估即训练”的闭环优化。对想进一步了解的读者，可以关注 2024‑2025 年间出现的 “LLM 竞技评测平台” 以及 “自我对话式微调” 方向的论文，这些都在 Auto‑Arena 的思路上继续深化。

### 一句话记住它
Auto‑Arena 用 LLM 自己出题、互相对战、集体裁决，实现了无需人工干预、与人类偏好 92% 以上一致的全自动大模型评估。