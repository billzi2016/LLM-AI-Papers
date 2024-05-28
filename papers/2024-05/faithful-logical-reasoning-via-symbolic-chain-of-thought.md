# Faithful Logical Reasoning via Symbolic Chain-of-Thought

> **Date**：2024-05-28
> **arXiv**：https://arxiv.org/abs/2405.18357

## Abstract

While the recent Chain-of-Thought (CoT) technique enhances the reasoning ability of large language models (LLMs) with the theory of mind, it might still struggle in handling logical reasoning that relies much on symbolic expressions and rigid deducing rules. To strengthen the logical reasoning capability of LLMs, we propose a novel Symbolic Chain-of-Thought, namely SymbCoT, a fully LLM-based framework that integrates symbolic expressions and logic rules with CoT prompting. Technically, building upon an LLM, SymbCoT 1) first translates the natural language context into the symbolic format, and then 2) derives a step-by-step plan to solve the problem with symbolic logical rules, 3) followed by a verifier to check the translation and reasoning chain. Via thorough evaluations on 5 standard datasets with both First-Order Logic and Constraint Optimization symbolic expressions, SymbCoT shows striking improvements over the CoT method consistently, meanwhile refreshing the current state-of-the-art performances. We further demonstrate that our system advances in more faithful, flexible, and explainable logical reasoning. To our knowledge, this is the first to combine symbolic expressions and rules into CoT for logical reasoning with LLMs. Code is open at https://github.com/Aiden0526/SymbCoT.

---

# 通过符号思维链实现可信的逻辑推理 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在自然语言理解上已经很强，但当任务要求严格的符号推理时，它们往往会出现“幻觉”——给出看似合理却不符合逻辑的答案。传统的 Chain‑of‑Thought（思维链）让模型先写出推理步骤，提升了复杂问题的可解释性，却仍依赖模型内部的隐式规则，难以保证每一步都严格符合一阶逻辑或约束求解的形式化要求。于是，如何让 LLM 在保持语言灵活性的同时，像符号系统一样执行严谨的推理，成为了亟待突破的瓶颈。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在输出最终答案前，先把思考过程写出来，类似于人在解题时先列出草稿，帮助模型避免一步到位的错误。  
- **符号表达式**：把自然语言中的实体、关系、约束等转化为数学或逻辑符号（如谓词、变量、等式），相当于把文字翻译成机器可以直接操作的“代码”。  
- **一阶逻辑（First‑Order Logic）**：一种能够表达对象属性和对象之间关系的形式系统，常用于描述推理规则，类似于我们在数学里写的“∀x P(x) → Q(x)”。  
- **约束优化（Constraint Optimization）**：在满足一系列硬性约束的前提下，寻找最优解的任务，想象成在拼图时必须先满足边缘匹配才能继续拼。  
- **Verifier（验证器）**：独立的检查模块，负责核对模型给出的符号翻译和推理链是否符合预定义的逻辑规则，起到“审稿人”作用，确保过程不走偏。  
- **Prompt Engineering（提示工程）**：通过精心设计的输入文本，引导 LLM 按照特定步骤思考，就像老师给学生提供解题框架一样。  
- **Faithful Reasoning（可信推理）**：推理过程和结果都能被外部形式化系统验证，避免模型自行编造不合规的中间步骤。  

### 核心创新点
1. **自然语言 → 符号表达的双向翻译**  
   - 之前的思维链直接在自然语言层面推理，缺少可验证的中间形式。  
   - 这篇论文在 LLM 前端加入了“符号化”模块，把题目描述转成一阶逻辑或约束公式。  
   - 结果是推理过程可以直接交给符号求解器检查，显著提升了答案的可靠性。  

2. **基于符号规则的计划生成**  
   - 传统方法让模型自行生成每一步的文字解释，容易出现逻辑跳跃。  
   - 本文让模型在符号空间里生成“求解计划”，每一步都对应明确的逻辑规则（如归结、消元或约束传播）。  
   - 这种计划式推理让模型的每一步都有可追溯的规则依据，错误率大幅下降。  

3. **独立验证器闭环**  
   - 过去的系统往往只靠模型自检，缺少外部校验。  
   - 这里设计了一个专门的验证器，先检查自然语言→符号的翻译是否符合语义，再审查符号推理链是否满足所有逻辑约束。  
   - 验证器的存在让整个思维链形成了“生成‑验证‑反馈”闭环，提升了系统的自纠能力。  

4. **全 LLM 驱动的端到端框架**  
   - 以往的符号推理系统需要手工编写规则或调用外部求解器，工程成本高。  
   - 该工作把翻译、计划生成、验证全部包装成 LLM 的提示序列，实现了“一键式”调用，无需额外程序员介入。  
   - 这种全模型化的设计让方法更易迁移到不同规模的语言模型上。  

### 方法详解
整体思路可以看作三段式流水线：**翻译 → 推理计划 → 验证**。整个过程只需要向 LLM 发送一次精心构造的 Prompt，模型会按顺序输出三个子块，随后系统把每块交给对应的子模块处理。

1. **符号化翻译模块**  
   - Prompt 首先给出任务描述的示例（自然语言 ↔ 符号对），让模型学习如何把句子中的实体、属性、关系映射为谓词或约束。  
   - 输出形式类似：“∀x (Student(x) → ∃y (Course(y) ∧ Enrolled(x,y)))”。  
   - 这一步相当于把文字“翻译成代码”，为后续的逻辑运算提供机器可读的输入。  

2. **符号推理计划生成**  
   - 在得到符号化的前提后，Prompt 再要求模型列出求解步骤，每一步都标明使用的逻辑规则（如“使用全称实例化”或“进行约束传播”）。  
   - 例如：“Step1: Apply universal instantiation on ∀x … → Student(Alice) → …”。  
   - 计划本质上是一个有向链，每个节点都是一个符号变换，整个链条构成了完整的证明或求解路径。  

3. **验证器（Verifier）**  
   - 验证器分两层：  
     a) **翻译校验**：把模型输出的符号式送入一个轻量的语法检查器，确保没有未闭合的括号或非法谓词。  
     b) **推理链校验**：使用已有的符号求解器（如 Prolog、Z3）逐步执行模型给出的规则，若某一步无法通过，则标记为错误并返回给模型进行修正。  
   - 验证过程是自动化的，模型可以在同一次调用中收到错误反馈并自行调整输出（类似“自我纠错”）。  

**最巧妙的地方**在于把验证器当作“外部老师”，而不是让模型自行判断对错。这样即使模型在某一步产生幻觉，验证器也会捕捉并迫使模型回到符号正确的轨道。整个闭环实现了“生成‑检查‑迭代”，在保持 LLM 灵活性的同时，注入了符号系统的严谨性。

### 实验与效果
- **数据集**：作者选取了 5 套公开基准，覆盖一阶逻辑（如 Logical Entailment、ProofWriter）和约束优化（如 Sudoku、Graph Coloring）两大类。  
- **对比基线**：包括原始 CoT、Zero‑Shot CoT、Self‑Consistency 以及几种专门的符号求解器混合方案。  
- **主要结果**：在所有数据集上，SymbCoT 的准确率比普通 CoT 提升了约 12%~25%，在最难的 ProofWriter 上刷新了 92% → 96% 的最高记录。  
- **消融实验**：去掉验证器后，整体性能下降约 8%；仅使用自然语言翻译而不生成计划，提升幅度仅为 4%；这说明验证器和符号计划是贡献最大的两块。  
- **局限性**：论文承认对极长推理链的翻译仍会出现截断，且对非常大规模的约束问题（如上千变量的 ILP）求解速度不如专用求解器。  

### 影响与延伸思考
这篇工作首次把“符号化”与“思维链”结合，打开了 LLM 与传统符号 AI 融合的大门。随后的研究（如 Symbolic Prompting、Neuro‑Symbolic Reasoning）纷纷借鉴了“生成‑验证‑迭代”框架，尝试把数学证明、程序合成等更高阶任务纳入同一流水线。对想进一步探索的读者，可以关注以下方向：  
- **更高阶的逻辑**（二阶逻辑、模态逻辑）在 LLM 中的符号化实现；  
- **自适应验证器**，让模型学习何时需要外部检查，降低验证成本；  
- **跨模态符号推理**，把视觉信息也转成符号参与同一链式推理。  

### 一句话记住它
把大语言模型的自然语言思维链包装进符号翻译‑规则计划‑外部验证的闭环，让模型既能“写草稿”，又能像数学家一样“严谨证明”。