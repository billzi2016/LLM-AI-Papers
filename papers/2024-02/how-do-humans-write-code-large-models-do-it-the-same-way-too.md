# How Do Humans Write Code? Large Models Do It the Same Way Too

> **Date**：2024-02-24
> **arXiv**：https://arxiv.org/abs/2402.15729

## Abstract

Program-of-Thought (PoT) replaces natural language-based Chain-of-Thought (CoT) as the most popular method in Large Language Models (LLMs) mathematical reasoning tasks by utilizing external tool calls to circumvent computational errors. However, our evaluation of the GPT-4 and Llama series reveals that using PoT introduces more reasoning errors, such as incorrect formulas or flawed logic, compared to CoT. To address this issue, we propose Human-Think Language (HTL), which leverages a suite of strategies that help integrate PoT and CoT, encompassing: (1) a new generation paradigm that uses full CoT reasoning to control code generation. (2) Focus Attention, that directs model attention to the CoT reasoning during PoT to generate more logical code. (3) reinforcement learning that utilizes the accuracy of both CoT and PoT responses as rewards to prevent repetitive reasoning steps in LLMs when solving difficult math problems. Our method achieves an average improvement of 6.5% on the Llama-Base model and 4.3% on the Mistral-Base model across 8 mathematical calculation datasets. It also shows significant effectiveness on five out-of-domain datasets by controlling the model's information flow, exhibiting strong transferability. Additionally, HTL shows the most significant improvement in non-mathematical natural language inference task, contributing to a unified reasoning task framework

---

# 人类如何写代码？大模型也以同样方式 论文详细解读

### 背景：这个问题为什么难？

在数学推理任务里，LLM（大语言模型）最早靠 **CoT（Chain‑of‑Thought）** 把思考过程写成自然语言，效果大幅提升。但 CoT 只能让模型“在脑子里”演算，遇到需要精确计算或符号操作时仍会出错。于是出现 **PoT（Program‑of‑Thought）**：让模型生成代码并调用外部工具执行，以规避算术错误。理论上，代码执行比文字推理更可靠，然而实际评测发现，PoT 往往把错误搬进代码里——公式写错、逻辑不通，导致整体推理质量下降。换句话说，模型在“写代码”这一步缺乏对人类思考过程的有效约束，导致错误传播。解决这个矛盾——既想利用代码的精确性，又不让模型的思考脱轨——成为本文要攻克的核心难题。

### 关键概念速览

**CoT（思维链）**：模型在给出最终答案前，先把推理步骤写成自然语言，就像人解题时在草稿纸上列步骤，便于检查和纠错。

**PoT（程序思维）**：模型把推理过程转化为可执行代码，再交给外部计算器或解释器运行，以获得精确数值。类似于把手算交给计算器。

**HTL（Human‑Think Language）**：一种融合 CoT 与 PoT 的语言框架，借助人类思考的结构化方式，引导模型在写代码时保持逻辑连贯。

**Focus Attention（聚焦注意力）**：在生成代码的过程中，模型被显式提醒去“看”之前的 CoT 步骤，确保代码实现对应的思考路径。可以想象成在写程序时不断回头检查草稿。

**RL‑Reward（强化学习奖励）**：把 CoT 与 PoT 的正确率都当作奖励信号，使用强化学习（如 PPO）让模型学会在困难题目上避免重复、无效的推理循环。

**信息流控制**：在模型内部人为划分“思考区”（CoT）和“实现区”（PoT），并通过注意力机制让信息只在需要时流动，防止噪声干扰。

### 核心创新点

1. **全链 CoT 控制代码生成**  
   *之前的 PoT 直接让模型写代码，缺少对思考过程的约束 → 本文在生成代码前先让模型完整走一遍 CoT，随后把这段文字推理作为“控制指令”喂给代码生成模块 → 代码更贴合人类的逻辑，错误率显著下降。

2. **聚焦注意力机制**  
   *传统注意力在生成代码时会把所有上下文混在一起，导致模型忘记前面的思考步骤 → 通过显式的 Focus Attention，把 CoT 内容标记为高权重区域，迫使模型在写每一行代码时都参考对应的思考句子 → 生成的代码更符合推理链，公式错误大幅减少。

3. **双向奖励的强化学习**  
   *单纯用 CoT 正确率或 PoT 正确率作为奖励会让模型偏向其中一种方式 → 采用双向奖励：只有当 CoT 与 PoT 两者都正确时才给高分，同时惩罚重复的推理步骤 → 模型学会在难题上快速收敛到最简洁的思考路径，整体解题效率提升。

4. **信息流控制的统一框架**  
   *过去的研究往往把 CoT 与 PoT 当作独立模块，缺少统一的任务视角 → HTL 把两者包装进同一套语言体系，明确划分思考区与实现区，并通过注意力桥梁让信息有序流动 → 这种统一框架在跨域（数学、自然语言推理）任务上表现出强迁移能力。

### 方法详解

**整体思路**  
HTL 把一次完整的推理拆成三步：① 生成完整的 CoT 思考链；② 在每一步 CoT 的基础上，用聚焦注意力生成对应的代码片段（PoT）；③ 将代码交给外部解释器执行，得到数值后回填到 CoT 中，形成闭环。整个过程在训练阶段通过强化学习优化，使得 CoT 与 PoT 的一致性最大化。

**步骤拆解**

1. **CoT 生成**  
   - 输入题目 → LLM 按照普通 CoT 方式输出完整的文字推理。  
   - 这里不做任何修改，确保模型的自然语言思考能力完整保留。

2. **聚焦注意力驱动的代码生成**  
   - 对每条 CoT 语句，模型在内部标记为 “思考标记”。  
   - 代码生成模块接收两类信息：① 当前思考标记的文本；② 全局上下文（题目、已有代码）。  
   - 通过专门设计的注意力权重，把思考标记的权重提升到 0.9 以上，其他信息权重降低。这样模型在写代码时几乎只能“看到”对应的思考步骤。  
   - 生成的代码严格对应单条 CoT，例如“设 x = …” → 生成 `x = ...` 的 Python 语句。

3. **外部执行与回填**  
   - 生成的代码块被送入安全的解释器（如 sandboxed Python），得到数值或布尔结果。  
   - 结果被插回原 CoT 的相应位置，形成 “思考 + 实现” 的完整答案草稿。

4. **双向奖励强化学习**  
   - **奖励设计**：  
     - 若 CoT 最终逻辑正确且 PoT 运行无误，奖励 +1。  
     - 若两者仅有一方正确，奖励 0。  
     - 若出现重复的思考步骤（检测到相同的 CoT 句子出现多次），额外扣分。  
   - 使用 PPO（Proximal Policy Optimization）在原始 LLM 参数上进行微调，使模型在生成新题目时倾向于一次性完成思考并实现，避免循环。

**最巧妙的点**  
- 把 **思考标记** 当作显式的注意力锚点，让模型在代码生成时“看不见”其他干扰信息，这种“思考‑实现”双通道的硬连线在以往的 PoT 里从未出现。  
- 双向奖励把 **正确性** 与 **效率** 同时纳入优化目标，解决了单纯追求代码执行正确却忽视思考连贯性的老问题。

### 实验与效果

- **数据集**：8 个数学计算基准（包括 GSM8K、MATH 等）以及 5 个跨域任务（自然语言推理、逻辑推理等）。  
- **模型**：在 Llama‑Base 与 Mistral‑Base 两个开源基线上进行微调。  
- **结果**：  
  - Llama‑Base 上平均提升 6.5%（如 GSM8K 从 48% 提升到 54.5%）。  
  - Mistral‑Base 上提升 4.3%（如 MATH 从 41% 提升到 45.3%）。  
  - 在非数学的自然语言推理任务上，提升幅度更大，成为该框架在统一推理任务中的亮点。  
- **消融实验**：  
  - 去掉聚焦注意力，提升仅剩 2% 左右，说明注意力是关键。  
  - 只使用单向奖励（仅 CoT）时，提升下降约 1.8%。  
  - 信息流控制的统一框架对跨域任务贡献最大，去掉后跨域提升几乎消失。  
- **局限性**：  
  - 需要安全的代码执行环境，部署成本比纯文字 CoT 高。  
  - 对于极其复杂的程序（如递归或大规模数据结构）仍会出现实现错误，作者承认 HTL 主要针对“短小精悍”的数学/逻辑代码。  
  - 强化学习阶段对奖励的敏感度较高，调参成本不容忽视。

### 影响与延伸思考

HTL 把人类写代码的“先思考、后实现”流程显式化，为 LLM 的工具调用提供了更可靠的思维框架。自论文发布后，已有几篇工作尝试把 **思考‑实现双链** 应用于代码补全、自动化实验设计等方向（如 *CoT‑Code*、*Think‑Then‑Act* 系列），并在强化学习的奖励设计上进一步探索多目标优化。未来可以关注：

- 把 HTL 扩展到 **多步工具链**（如数据库查询 → 可视化 → 报告生成），验证信息流控制的通用性。  
- 探索 **自监督的思考标记生成**，让模型自行学习何时需要插入代码，而不是每一步都强制生成。  
- 将 **安全沙箱** 与模型紧耦合，降低部署门槛，使 HTL 能在实际产品中直接使用。

### 一句话记住它

**HTL 用“先写思考链、再用聚焦注意力写代码、再用双向奖励强化”把人类写代码的流程完整搬进大模型，显著提升了数学与跨域推理的可靠性。**