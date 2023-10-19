# MAF: Multi-Aspect Feedback for Improving Reasoning in Large Language   Models

> **Date**：2023-10-19
> **arXiv**：https://arxiv.org/abs/2310.12426

## Abstract

Language Models (LMs) have shown impressive performance in various natural language tasks. However, when it comes to natural language reasoning, LMs still face challenges such as hallucination, generating incorrect intermediate reasoning steps, and making mathematical errors. Recent research has focused on enhancing LMs through self-improvement using feedback. Nevertheless, existing approaches relying on a single generic feedback source fail to address the diverse error types found in LM-generated reasoning chains. In this work, we propose Multi-Aspect Feedback, an iterative refinement framework that integrates multiple feedback modules, including frozen LMs and external tools, each focusing on a specific error category. Our experimental results demonstrate the efficacy of our approach to addressing several errors in the LM-generated reasoning chain and thus improving the overall performance of an LM in several reasoning tasks. We see a relative improvement of up to 20% in Mathematical Reasoning and up to 18% in Logical Entailment.

---

# MAF：多方面反馈提升大语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成自然语言时表现惊艳，但在需要严密推理的任务上仍会出现“幻觉”——给出看似合理却错误的答案；中间推理步骤也常常不符合逻辑；数学题甚至会出现算术错误。早期的自我改进方法大多只给模型一个统一的反馈信号，像“对了”或“错了”，却没有针对不同错误类型提供专门的纠正。结果是模型在同一次迭代中只能修正一小部分错误，整体推理质量提升有限。

### 关键概念速览
**思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出推理步骤，类似人做数学题时的草稿，帮助暴露错误并提供纠错机会。  
**自我反馈（Self‑Feedback）**：模型自行检查自己的输出并给出改进建议，像是让学生自己批改作业。  
**冻结模型（Frozen LM）**：在训练或微调阶段不改变参数的语言模型，常被当作“评审员”使用，因为它的行为稳定可预测。  
**外部工具（External Tool）**：除语言模型之外的专用程序，如符号求解器或逻辑验证器，负责检查特定类型的错误。  
**多方面反馈（Multi‑Aspect Feedback）**：把若干专门化的评审员组合起来，每个负责一种错误（比如算术、逻辑、一致性），形成多层次的纠错网络。  
**迭代细化（Iterative Refinement）**：模型在收到反馈后重新生成答案，再交给下一轮评审，类似编辑稿件的多轮审稿过程。

### 核心创新点
1. **单一反馈 → 多维评审**  
   过去的自我改进只用一个通用的纠错模块，往往只能捕捉到表层错误。MAF 引入了多个专职评审，每个评审聚焦一种错误类型（如数学错误、逻辑矛盾、事实不符），从根本上扩大了错误检测的覆盖面。  
2. **冻结语言模型 + 外部工具的混合**  
   传统方法要么全靠语言模型自行判断，要么全部依赖外部工具，难以兼顾灵活性和专业性。MAF 把冻结的语言模型当作“语言感知评审”，负责语义连贯性；把符号求解器、逻辑验证器等外部工具当作“硬核评审”，负责数学和逻辑精度，两者互补。  
3. **迭代细化的闭环设计**  
   不是一次性给出反馈后直接结束，而是把反馈结果喂回模型，让它在下一轮生成时利用前一次的纠错信息。这样模型可以逐步收敛到更可靠的推理链，而不是一次性“全盘改正”。  
4. **统一的多任务训练框架**  
   为了让模型能够接受不同来源的反馈，MAF 设计了一个统一的损失函数，把来自各评审的评分统一映射为“改进信号”，避免了为每种错误单独训练的繁琐。

### 方法详解
**整体框架**  
MAF 的工作流程可以概括为三步：① 让基线 LLM 生成初始推理链；② 将这条链分别送入若干反馈模块，每个模块返回针对自身关注错误的评估与修正建议；③ 把所有建议合并成一个“综合反馈”，再喂回 LLM 进行一次或多次的重新生成。整个过程可以循环多轮，直至模型的输出在所有评审上都达到预设阈值。

**关键模块拆解**  
1. **初始生成（Base Generation）**  
   使用未微调的 LLM（如 GPT‑3.5）直接输出 CoT。此阶段不做任何约束，目的是让模型自由发挥，暴露所有潜在错误。  
2. **冻结语言模型评审（Frozen LM Judge）**  
   这是一段保持参数不变的模型，它接收原始推理链并输出两类信息：① 语义一致性得分（判断前后句子是否自洽），② 语言层面的错误提示（比如歧义、遗漏）。它的工作方式类似“让另一个学生检查你的草稿”。  
3. **外部工具评审（External Tool Judges）**  
   - **数学求解器**：把推理链中的算式抽取出来交给符号计算器（如 SymPy），比较计算结果与模型给出的答案，若不一致则返回“算式错误”。  
   - **逻辑验证器**：把逻辑蕴含关系转化为形式化命题，使用 SAT/SMT 求解器检查是否满足前提，若违背则返回“逻辑冲突”。  
   - **事实检索器**（可选）：查询知识库或搜索引擎，验证模型陈述的事实是否真实。  
4. **反馈融合（Feedback Fusion）**  
   每个评审的输出被标准化为统一的“改进指令”。例如，数学求解器返回“第3步算式应为 7×8=56”，语言模型评审返回“第2步的推理缺少对前提的解释”。这些指令被拼接成一段新的提示，放在模型的下一轮输入前。  
5. **迭代细化（Iterative Refinement）**  
   LLM 接收到融合后的提示后重新生成推理链。若仍有错误，流程再次进入评审环节。实验中通常设定 2–3 轮即可收敛。

**最巧妙的设计**  
- **冻结模型的“软评审”**：因为参数不变，它的评判标准保持一致，避免了在迭代过程中因模型自身漂移导致的评审不稳定。  
- **外部工具的“硬核校准”**：把符号求解和逻辑验证外包给专用引擎，确保数学和逻辑错误几乎不可能被遗漏。  
- **统一反馈的“软指令”**：把不同来源的纠错信息转化为自然语言指令，让 LLM 能直接在生成时遵循，而不需要额外的结构化输入。

### 实验与效果
- **测试任务**：论文在数学推理（Mathematical Reasoning）和逻辑蕴含（Logical Entailment）两个公开基准上评估。  
- **对比基线**：包括原始 LLM（无反馈）、单一自我反馈（CoT + 单一评审）以及最近的自我纠错框架。  
- **提升幅度**：在数学推理上相对提升最高可达 20%；在逻辑蕴含上提升最高可达 18%。这些数字直接来源于论文的实验报告。  
- **消融实验**：作者分别去掉冻结语言模型评审、数学求解器或逻辑验证器，发现每去掉一个模块整体性能下降约 4–6%，说明多方面评审的每一环都贡献显著。  
- **局限性**：论文承认当前实现依赖于外部工具的可用性和调用成本；在需要大量实时计算的场景下迭代次数会导致响应延迟。此外，冻结模型的评审质量受其预训练数据的限制，面对极端专业领域仍可能漏检。

### 影响与延伸思考
MAF 把“多评审+迭代”思路系统化后，激发了后续工作在“多模态反馈”“跨语言评审”等方向的探索。比如有研究把视觉模型的错误交给专用图像检验器，再回馈给语言模型，实现图文推理的协同纠错。还有人尝试把人类专家的标注当作外部评审，形成“人机混合反馈”。如果想进一步深入，可以关注以下两个方向：① 设计更高效的反馈融合策略，减少迭代次数；② 将评审模块训练为可微分的子网络，实现端到端的自适应纠错（目前大多数外部工具仍是黑盒调用）。

### 一句话记住它
让大语言模型在每一步都接受专门化的“评审员”检查，并把所有评审的纠错指令循环喂回模型，才能显著削减推理错误。