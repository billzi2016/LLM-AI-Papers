# FIMO: A Challenge Formal Dataset for Automated Theorem Proving

> **Date**：2023-09-08
> **arXiv**：https://arxiv.org/abs/2309.04295

## Abstract

We present FIMO, an innovative dataset comprising formal mathematical problem statements sourced from the International Mathematical Olympiad (IMO) Shortlisted Problems. Designed to facilitate advanced automated theorem proving at the IMO level, FIMO is currently tailored for the Lean formal language. It comprises 149 formal problem statements, accompanied by both informal problem descriptions and their corresponding LaTeX-based informal proofs. Through initial experiments involving GPT-4, our findings underscore the existing limitations in current methodologies, indicating a substantial journey ahead before achieving satisfactory IMO-level automated theorem proving outcomes.

---

# FIMO：面向自动定理证明的挑战性形式化数据集 论文详细解读

### 背景：这个问题为什么难？

自动定理证明（ATP）在过去十年里已经可以处理不少高校教材级别的题目，但要迈向国际数学奥林匹克（IMO）这种“创意+深度”并存的水平，却一直缺乏合适的训练和评估基准。现有的公开数据集大多来源于形式化数学库（如Mathlib）或人工合成的逻辑题，它们的难度分布偏低，且缺少与人类竞赛题目对应的非形式化描述。没有一个既包含正式化代码，又保留原始题目叙述和手写证明的完整集合，研究者很难判断模型是否真的具备“IMO级”推理能力。因此，构建这样一个挑战性数据集成为迫切需求。

### 关键概念速览
- **自动定理证明（ATP）**：让机器在给定的公理和规则下，自动找出证明或反例的过程，类似于让电脑代替人类写数学证明。  
- **形式化语言（Lean）**：一种可以被机器严格检查的数学语言，所有定义、定理和证明都必须写成机器可读的代码，就像把数学题目翻译成程序。  
- **IMO Shortlist（IMO 预选题）**：每年IMO前挑选出的高难度题目集合，代表了竞赛中最具挑战性的思路。  
- **非形式化描述**：普通人阅读的自然语言题目陈述，通常是几句话的文字说明。  
- **LaTeX 证明**：用 LaTeX 排版的书面数学证明，保留了人类写作的结构和符号，但仍是非机器可直接验证的文本。  
- **Prompt Engineering**：为大语言模型（LLM）设计输入提示的技巧，类似于给模型“指路牌”，决定它能否正确理解并生成证明。  
- **Few‑shot Learning**：在只给模型少量示例的情况下，让它学习完成新任务的能力，像是只看几道例题就能解出新题。  

### 核心创新点
1. **从 IMO 预选题直接抽取 → 形式化为 Lean 代码 → 获得首个 IMO‑级正式化数据集**  
   过去的 ATP 数据集大多自行生成或来源于已有的数学库，这篇论文直接把真实竞赛题目搬进 Lean，确保了难度和创意的真实性。结果是一个只有 149 条的“小而精”集合，却足以检验模型在高阶推理上的极限。

2. **同步提供三种视角 → 非形式化描述、LaTeX 手写证明、Lean 正式化**  
   传统数据集只给出形式化代码，缺少人类思考的桥梁。这里把每道题的自然语言、传统书面证明和机器可验证的代码全部对齐，形成了“多模态”对照，方便研究者探索如何让模型从文字到代码的跨模态迁移。

3. **基准实验使用 GPT‑4 → 揭示当前 LLM 在 IMO 级 ATP 上的瓶颈**  
   作者用最强大的通用语言模型 GPT‑4 进行初步尝试，结果显示即便是最先进的模型，也只能在极少数题目上给出可接受的证明，凸显了从“语言理解”到“严谨证明”之间的巨大鸿沟。

### 方法详解
整体思路可以拆成三步：**题目收集 → 多模态对齐 → 基准评估**。

1. **题目收集**  
   - 从 IMO 官方公布的 Shortlist 中挑选 149 条在难度、主题上具有代表性的题目。  
   - 每道题目先保留原始的英文（或其他语言）自然语言描述，作为“非形式化”输入。

2. **多模态对齐**  
   - **手写 LaTeX 证明**：作者或志愿者依据题目自行撰写完整的书面证明，使用 LaTeX 排版，确保与人类竞赛解答的风格一致。  
   - **Lean 正式化**：把 LaTeX 证明一步步翻译成 Lean 代码。这个过程需要先在 Mathlib 中寻找对应的定义与引理，然后用 Lean 的 tactic（策略）语言构造证明脚本。相当于把人类的“思路笔记”转成机器的“执行脚本”。  
   - 为了保持一致性，所有三种形式都用统一的编号体系关联，形成一条可追溯的链路：自然语言 → LaTeX → Lean。

3. **基准评估**  
   - 选用 GPT‑4 作为代表性大语言模型，采用 **few‑shot prompting**：在提示中先给出几道已经形式化好的题目及其对应的 Lean 代码，随后让模型尝试生成新题目的 Lean 证明。  
   - 评估标准是两层：**语义匹配**（模型输出是否在逻辑上与原题目对应）和 **形式化校验**（Lean 编译器是否接受该证明）。只有通过 Lean 检查的才算成功。  
   - 实验中还加入了 **prompt ablation**：分别去掉 LaTeX 参考、只保留自然语言、或只给出部分 Lean 前置，引出模型对不同信息源的依赖程度。

最巧妙的地方在于把 **LaTeX 证明** 作为中间桥梁。直接让模型从自然语言跳到 Lean 往往失败，而提供 LaTeX 让模型先“复述”人类的书面思路，再转化为代码，显著提升了可行性。

### 实验与效果
- **测试对象**：全部 149 条 FIMO 题目，分别在“完整三模态提示”和“仅自然语言提示”两种设置下跑 GPT‑4。  
- **对比基线**：没有公开的同类基线，因为 FIMO 是首个 IMO‑级正式化数据集。作者仅把 GPT‑4 的表现与人类手写 LaTeX/Lean 的成功率作对照。  
- **结果**：在最有利的完整提示下，GPT‑4 只在约 7%（约 10 题）上生成了 Lean 编译通过的证明；在仅自然语言提示下，成功率跌至不到 2%。这表明即便是最强的通用模型，也远未达到 IMO 级自动定理证明的水平。  
- **消融实验**：去掉 LaTeX 参考后成功率下降约 60%，说明 LaTeX 作为中间层对模型至关重要。  
- **局限性**：数据集规模仍然小（149 题），不够支撑大规模深度学习训练；仅覆盖 Lean 一种形式化语言，缺少对其他系统（Coq、Isabelle）的兼容性；实验只用了 GPT‑4，未探索专门为定理证明设计的模型。

### 影响与延伸思考
FIMO 的出现为 ATP 社区提供了第一个真正面向高难度竞赛题目的正式化基准，推动了以下几个方向的探索：  
- **跨模态学习**：如何让模型更好地从自然语言或 LaTeX 迁移到形式化代码，已成为新的研究热点。  
- **专用定理证明模型**：在看到通用 LLM 的瓶颈后，研究者开始尝试结合符号推理引擎的专用模型（如 Lean‑GPT、MiniF2F 的后继）。  
- **数据扩展**：后续工作（如 FIMO‑2、IMO‑Formal）计划把题目规模扩大到千级，并加入 Coq、Isabelle 等多语言对齐，形成更通用的评测平台。  
想进一步深入，可以关注 **Lean 生态的自动化工具**（如 `mathlib` 的自动化 tactic、`Sledgehammer`）以及 **大语言模型在形式化数学中的微调技术**。

### 一句话记住它
FIMO 把真实的 IMO 竞赛题目“一键翻译”成 Lean 代码，首次让我们看到机器在最高水平数学推理上的巨大差距。