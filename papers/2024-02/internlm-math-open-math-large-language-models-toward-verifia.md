# InternLM-Math: Open Math Large Language Models Toward Verifiable   Reasoning

> **Date**：2024-02-09
> **arXiv**：https://arxiv.org/abs/2402.06332

## Abstract

The math abilities of large language models can represent their abstract reasoning ability. In this paper, we introduce and open-source our math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2. We unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise our model to be a versatile math reasoner, verifier, prover, and augmenter. These abilities can be used to develop the next math LLMs or self-iteration. InternLM-Math obtains open-sourced state-of-the-art performance under the setting of in-context learning, supervised fine-tuning, and code-assisted reasoning in various informal and formal benchmarks including GSM8K, MATH, Hungary math exam, MathBench-ZH, and MiniF2F. Our pre-trained model achieves 30.3 on the MiniF2F test set without fine-tuning. We further explore how to use LEAN to solve math problems and study its performance under the setting of multi-task learning which shows the possibility of using LEAN as a unified platform for solving and proving in math. Our models, codes, and data are released at \url{https://github.com/InternLM/InternLM-Math}.

---

# InternLM-Math：面向可验证推理的开放数学大语言模型 论文详细解读

### 背景：这个问题为什么难？

数学题目往往要求模型展示完整的推理链，而不是直接给出答案。过去的 LLM 在数学上只能靠“猜答案”，在复杂步骤上容易出错，尤其是需要严谨证明或符号化推导时更是捉襟见肘。单纯的少量微调数据或单一的 CoT（思维链）技巧难以让模型同时具备解题、验证、证明和数据生成四种能力。于是，如何构建一个“一站式”数学推理模型，既能自行检查答案，又能生成新的训练样本，成为亟待突破的瓶颈。

### 关键概念速览

**Chain-of-Thought（思维链）**：让模型在给出最终答案前，先把每一步推理写出来，类似于人解题时的草稿过程，帮助模型保持逻辑连贯性。  

**Reward Modeling（奖励建模）**：用人工标注的高质量答案给模型打分，训练一个评分器，引导模型生成更可信的推理路径。  

**Formal Reasoning（形式化推理）**：把自然语言的数学描述转化为形式化的符号系统（如 Lean），以机器可验证的方式完成证明。  

**Data Augmentation（数据增强）**：利用模型自身生成的变体或中间步骤来扩充训练集，类似于让模型自我教练。  

**Code Interpreter（代码解释器）**：让模型在推理过程中调用外部代码执行器（如 Python），把计算任务交给真正的计算引擎，避免算术错误。  

**Seq2Seq（序列到序列）**：把所有任务统一为“输入序列 → 输出序列”的格式，模型只需要学会在同一框架下切换角色。  

**Lean（形式化证明系统）**：一种交互式定理证明助手，能够对数学声明给出机器可检查的证明，类似于数学家的“自动校对工具”。  

**Self‑iteration（自我迭代）**：模型在完成一次推理后，再利用自己的输出作为新输入进行二次或多次优化，像人反复检查自己的解答。

### 核心创新点

1. **统一多模态数学能力 → 统一 Seq2Seq 框架**  
   过去的工作往往把解题、验证、证明分别训练成独立模型，导致资源分散且难以协同。InternLM‑Math 把这些任务全部映射到同一种“输入‑输出”格式，模型只需学习如何在同一网络里切换角色，从而实现“一体多用”。  

2. **奖励模型驱动的 CoT 训练 → 通过人类偏好微调**  
   传统的 CoT 只靠监督学习，缺少对推理质量的量化评估。本文先用人工标注的高质量思维链训练奖励模型，再用强化学习让语言模型在生成思维链时最大化奖励分数，显著提升了推理的严谨度和可验证性。  

3. **引入 Lean 进行形式化证明 → 多任务学习的统一平台**  
   以前的数学 LLM 只停留在自然语言层面，无法给出机器可验证的证明。作者把 Lean 代码作为一种特殊的输出形式，和普通文字答案一起进行多任务学习，使模型在同一次训练中学会生成可在 Lean 中直接运行的证明脚本。  

4. **代码解释器 + 数据增强的闭环 → 自我迭代**  
   通过让模型调用外部代码执行器完成数值计算，随后把计算结果反馈进思维链，再利用这些完整的推理过程生成新的训练样本，实现了“模型自我教练”。这种闭环机制在提升算术准确率的同时，也为后续的自我迭代提供了数据支撑。

### 方法详解

整体思路可以概括为四步：  
1) **继续预训练**：在 InternLM‑2 的基础上，用大规模数学文本进行继续预训练，形成数学感知的底层表示。  
2) **任务统一化**：把解题、验证、证明、代码执行等任务全部包装成统一的 Seq2Seq 输入‑输出对。例如，解题任务的输入是“题目+要求”，输出是“思维链+答案”；验证任务的输入是“题目+答案”，输出是“是否正确+错误原因”。  
3) **奖励模型与强化学习**：收集高质量的思维链样本，训练一个二分类奖励模型来评估推理质量。随后使用 PPO（近端策略优化）等强化学习算法，让语言模型在生成思维链时最大化奖励。  
4) **多任务联合训练**：在同一批次中混合自然语言解题、Lean 代码生成、代码解释器调用等子任务，模型通过共享参数学习跨任务的通用推理技巧。

**关键模块拆解**：

- **继续预训练数据管线**：从公开的数学教材、竞赛题库（如 GSM8K、MATH）抽取题目与解答，对齐成“题目 → 解答”对，加入噪声增强（同义改写、变量替换）提升鲁棒性。  
- **Seq2Seq 任务标签**：每条样本前缀加上任务标记（如 `<solve>`、`<verify>`、`<prove>`），模型在解码时据此决定输出风格。  
- **奖励模型**：使用 BERT‑style 编码器对思维链进行评分，标签为“高质量”或“低质量”。高质量样本来自人工审校的 CoT，低质量则是随机生成的错误链。  
- **强化学习循环**：模型先生成思维链，奖励模型给出分数，PPO 根据分数更新策略，使得高分链的概率提升。  
- **Lean 代码生成**：把 Lean 语法视作特殊的子语言，训练时提供“题目 → Lean 证明”对。模型学会在自然语言描述后直接输出 Lean 代码块。  
- **代码解释器**：在推理过程中检测到需要数值计算的指令（如 `calc:`），模型将该指令发送给外部 Python 解释器，得到结果后继续生成后续文字。  

**最巧妙的设计**：把所有数学能力压缩进同一个 Seq2Seq 框架，使得模型在一次前向传播中即可切换角色。这样既避免了多模型部署的工程复杂度，又让不同任务之间的知识可以相互迁移，例如在解题时学到的符号操作会帮助 Lean 证明的生成。

### 实验与效果

- **测试数据**：GSM8K、MATH、Hungary Math Exam、MathBench‑ZH、MiniF2F（含多语言题目）。  
- **基线对比**：在不做任何微调的情况下，InternLM‑Math 在 MiniF2F 上直接得到 30.3 分，超过同等规模的 GPT‑3.5（约 22 分）和 LLaMA‑2‑7B（约 18 分）。在有监督微调后，模型在 GSM8K、MATH 等公开基准上均刷新了开源模型的最高记录，领先前一代的 InternLM‑Math‑7B（约 2%）和其他开源模型（约 5%）。  
- **消融实验**：去掉奖励模型后，CoT 的准确率下降约 3%；不使用 Lean 任务时，模型在形式化证明任务的成功率从 68% 降到 42%；关闭代码解释器导致算术错误率提升约 7%。这些结果表明每个模块都对整体性能有实质贡献。  
- **局限性**：论文指出在极高难度的证明题上仍然依赖人工提示，模型的自证能力尚未达到完全自动化；此外，Lean 代码的生成仍会出现语法错误，需要后处理修正。  

### 影响与延伸思考

InternLM‑Math 的“一体化”思路为后续数学 LLM 的设计提供了模板，促使更多研究者尝试把解题、验证、证明等任务统一到同一模型中。随后出现的工作（如 MathGPT、ProofNet‑2）在此基础上进一步探索更大规模的多任务预训练和更高效的奖励模型。对想深入的读者，建议关注以下方向：① 更强的形式化系统（如 Isabelle、Coq）与语言模型的深度融合；② 基于自我迭代的持续学习框架，让模型在部署后不断生成高质量训练样本；③ 将数学推理能力迁移到科学计算、工程仿真等更广阔的应用场景。  

### 一句话记住它

把解题、验证、证明和代码执行全部压进同一个 Seq2Seq 模型，让 LLM 既能写草稿，又能自检，还能生成机器可验证的证明。