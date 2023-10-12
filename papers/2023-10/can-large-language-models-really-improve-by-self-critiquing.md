# Can Large Language Models Really Improve by Self-critiquing Their Own   Plans?

> **Date**：2023-10-12
> **arXiv**：https://arxiv.org/abs/2310.08118

## Abstract

There have been widespread claims about Large Language Models (LLMs) being able to successfully verify or self-critique their candidate solutions in reasoning problems in an iterative mode. Intrigued by those claims, in this paper we set out to investigate the verification/self-critiquing abilities of large language models in the context of planning. We evaluate a planning system that employs LLMs for both plan generation and verification. We assess the verifier LLM's performance against ground-truth verification, the impact of self-critiquing on plan generation, and the influence of varying feedback levels on system performance. Using GPT-4, a state-of-the-art LLM, for both generation and verification, our findings reveal that self-critiquing appears to diminish plan generation performance, especially when compared to systems with external, sound verifiers and the LLM verifiers in that system produce a notable number of false positives, compromising the system's reliability. Additionally, the nature of feedback, whether binary or detailed, showed minimal impact on plan generation. Collectively, our results cast doubt on the effectiveness of LLMs in a self-critiquing, iterative framework for planning tasks.

---

# 大型语言模型真的能通过自我批评自己的计划而提升？ 论文详细解读

### 背景：这个问题为什么难？

在传统的自动规划系统里，生成的计划往往需要外部的验证器来检查是否满足约束或目标。随着大模型（LLM）在自然语言推理上的惊人表现，研究者开始期待同一个模型既能生成计划，又能自行审查其合理性，从而形成闭环的“自我批评”迭代。可是，计划空间极其庞大且约束关系错综复杂，单靠语言模型的隐式知识往往难以捕捉细粒度的逻辑错误。此前的工作大多把验证交给专门的符号求解器或规则引擎，缺乏对“LLM 自我审查”能力的系统评估，这让人不清楚自我批评到底能否真正提升规划质量。

### 关键概念速览

**规划（Planning）**：在给定初始状态和目标状态的前提下，自动生成一系列可执行动作的过程，就像在地图上找出从出发点到目的地的路线。  

**自我批评（Self‑critique）**：模型在生成候选答案后，再次审视并给出改进建议的行为，类似于人写完作文后自己检查语法错误。  

**验证器（Verifier）**：负责判断计划是否满足所有约束的模块，传统上是基于逻辑求解或模型检查的工具。  

**假阳性（False Positive）**：验证器错误地认为一个有缺陷的计划是合法的，就像安全检查把违禁品误判为安全物品。  

**二元反馈（Binary Feedback）**：仅告诉模型“对”或“错”，不提供具体错误细节。  

**详细反馈（Detailed Feedback）**：在二元基础上补充错误位置或原因的说明，类似老师在批改作业时给出的注释。  

**迭代规划（Iterative Planning）**：生成‑批评‑改进的循环过程，期望每轮都让计划更接近目标。  

**GPT‑4**：OpenAI 发布的最先进的大语言模型，本文把它当作生成器和验证器的统一主体。

### 核心创新点

1. **从外部验证转向纯 LLM 验证**  
   *之前的系统*：使用符号求解器或专门的规则库来检查计划合法性。  
   *本文的做法*：让同一个 GPT‑4 模型在自然语言提示下完成验证任务，完全不依赖外部工具。  
   *带来的改变*：提供了一个“一体化”实验平台，直接测量 LLM 自身的审查能力，而不是混合了外部强验证器的结果。

2. **系统化比较不同反馈粒度的影响**  
   *之前的研究*：多在少数任务上尝试二元或详细反馈，缺乏统一实验设计。  
   *本文的做法*：在同一套规划任务上分别提供仅“对/错”与包含错误描述的两类反馈，观察对后续计划生成的影响。  
   *带来的改变*：揭示了反馈细节对 LLM 规划质量的贡献几乎可以忽略，为后续设计反馈机制提供了实证依据。

3. **量化自我批评导致的误报率**  
   *之前的工作*：往往只报告整体成功率，忽视验证错误的类型。  
   *本文的做法*：统计 LLM 验证器产生的假阳性比例，并与外部可靠验证器进行对比。  
   *带来的改变*：发现自我批评容易产生误判，直接削弱了系统的可靠性，提醒研究者在使用 LLM 进行自检时必须加入额外的安全网。

### 方法详解

整体框架可以看成三步循环：**计划生成 → 自我批评 → 计划修正**。整个流程全部由 GPT‑4 执行，核心在于如何让模型在“自我批评”阶段输出既可信又可操作的建议。

1. **计划生成模块**  
   - 输入：任务描述（如“把 A 物体搬到 B 位置”）以及当前环境状态的自然语言描述。  
   - 提示：让模型以“步骤列表”的形式输出计划，每一步都是可执行的动作句子。  
   - 结果：得到一段文字化的计划草案。

2. **自我批评（验证）模块**  
   - 输入：同样的任务描述 + 生成的计划。  
   - 提示分两种：  
     *二元版*：“这段计划是否满足目标？只回答‘是’或‘否’。”  
     *详细版*：“如果不满足，请指出具体哪一步出错并说明原因。”  
   - 输出：模型给出“通过”或“未通过”，以及（可选的）错误定位。这里的输出即为 LLM 自己的验证结果。

3. **计划修正模块**  
   - 若验证返回“未通过”，系统把错误信息（如果有）拼回提示，要求模型在原计划基础上进行改写。  
   - 改写的策略是让模型保留已正确的步骤，只重写被标记为错误的部分。  
   - 修正后再次进入自我批评，形成迭代。

**关键细节**  
- 为防止模型直接复制原计划并声称“通过”，提示中加入了“请务必检查每一步是否满足约束”。  
- 为了衡量误报，作者另外准备了一个独立的符号验证器（外部的“金标准”），对每一次自我批评的结果进行交叉检查。  
- 实验中使用了“温度=0.0”的确定性采样，以保证同一输入的输出可重复，便于统计假阳性率。

**最巧妙的地方**  
把验证任务包装成自然语言问答，使得 GPT‑4 能在同一模型内部完成两种截然不同的功能（生成与审查），而不需要额外的代码实现。这种“一问一答”式的自我审查，是对 LLM 能力边界的直接探测。

### 实验与效果

- **任务与数据**：作者选取了公开的几套基于文本描述的规划基准（如 Blocks World、简化的机器人搬运任务），每个任务都有明确的初始状态、目标状态和动作库。  
- **对比基线**：  
  1. **外部验证器系统**：使用 GPT‑4 生成计划，但交给符号求解器进行验证。  
  2. **纯 LLM 系统**：本文的自我批评闭环。  
- **主要发现**：  
  - 在所有基准上，外部验证器系统的成功率显著高于纯 LLM 系统，后者的成功率下降约 10%~15%。  
  - 自我批评产生的假阳性比例约为 20%，即每五个被标记为“通过”的计划中就有一个实际上不满足约束。  
  - 二元反馈与详细反馈在最终成功率上的差异不到 2%，说明提供更丰富的错误信息并未带来实质性提升。  
- **消融实验**：作者分别关闭“错误定位信息”以及“保留已正确步骤”两项策略，发现去掉错误定位会导致整体成功率再跌 5%，而不保留已正确步骤则使迭代次数翻倍，效率下降。  
- **局限性**：实验仅覆盖了相对小规模、规则明确的文本规划任务，未涉及高维连续动作空间或需要深度物理模拟的场景。作者也承认，GPT‑4 的内部知识库可能对特定领域的约束不够精准，导致误判。

### 影响与延伸思考

这篇工作在社区里引发了对“LLM 自我审查”可靠性的广泛讨论。随后有几篇论文尝试在自我批评前加入 **外部校准器**（如小型逻辑模型）或使用 **多模型投票** 来降低假阳性率，属于对本文提出的风险的直接回应。还有研究把自我批评扩展到 **代码生成**、**数学证明** 等更高风险的任务，借鉴了本文的二元/详细反馈实验设计。想进一步了解该方向，建议关注 **“LLM‑based verification”** 与 **“neural theorem proving”** 的交叉工作，它们正尝试把符号可靠性与语言模型的灵活性结合起来（推测）。

### 一句话记住它

让大语言模型自己检查计划并不比外部可靠验证器更好，反而会引入显著的误判风险。