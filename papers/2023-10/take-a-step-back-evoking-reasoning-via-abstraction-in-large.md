# Take a Step Back: Evoking Reasoning via Abstraction in Large Language   Models

> **Date**：2023-10-09
> **arXiv**：https://arxiv.org/abs/2310.06117

## Abstract

We present Step-Back Prompting, a simple prompting technique that enables LLMs to do abstractions to derive high-level concepts and first principles from instances containing specific details. Using the concepts and principles to guide reasoning, LLMs significantly improve their abilities in following a correct reasoning path towards the solution. We conduct experiments of Step-Back Prompting with PaLM-2L, GPT-4 and Llama2-70B models, and observe substantial performance gains on various challenging reasoning-intensive tasks including STEM, Knowledge QA, and Multi-Hop Reasoning. For instance, Step-Back Prompting improves PaLM-2L performance on MMLU (Physics and Chemistry) by 7% and 11% respectively, TimeQA by 27%, and MuSiQue by 7%.

---

# 退一步：通过抽象激发大语言模型的推理 论文详细解读

### 背景：这个问题为什么难？

在传统的提示工程里，直接把题目喂给大语言模型（LLM），期望它一步到位给出答案。实际使用时，模型常常被细节信息牵着走，尤其是涉及多步推理、跨学科知识或需要抓住核心原理的任务，容易走偏或卡在表层信息上。已有的思维链（CoT）等方法通过让模型先写出推理步骤缓解了部分问题，但仍缺少一种机制帮助模型主动抽离出“高层概念”或“第一原理”，从而在更抽象的层面规划推理路径。正是这种抽象能力的缺失，让模型在复杂 STEM、知识问答和多跳推理任务上表现受限，迫切需要一种更高层次的提示手段。

### 关键概念速览
- **Step-Back Prompting（退一步提示）**：先让模型把具体问题抽象成概念或原理，再用这些抽象结果继续求解。相当于先让学生把题目翻译成“核心公式”，再去算答案。
- **抽象（Abstraction）**：把细节信息压缩成更通用的描述，例如把“氢原子在磁场中产生的能级分裂”抽象为“Zeeman 效应”。抽象帮助模型看到问题的本质。
- **高层概念（High‑level concepts）**：抽象后得到的关键词或概念，类似于解题时的“已知定律”。它们比原始文字更易于在推理中复用。
- **第一原理（First principles）**：最基本的物理或数学定律，像是“能量守恒”。在抽象阶段把问题映射到这些原理上，可让推理过程更稳固。
- **思维链（Chain‑of‑Thought, CoT）**：让模型在输出答案前写出逐步推理，就像在草稿纸上列步骤。它让过程透明，但不一定涉及抽象层次。
- **多跳推理（Multi‑Hop Reasoning）**：答案需要跨越多个事实或步骤才能得到，类似于在维基百科里跳转多次查找信息。
- **大语言模型（Large Language Model, LLM）**：拥有上百亿参数、通过海量文本训练的生成式模型，如 PaLM‑2L、GPT‑4、Llama2‑70B。

### 核心创新点
1. **从“直接求解”到“先抽象再求解”**  
   之前的提示大多让模型直接给出答案或写思维链；这篇论文在提示中加入了一个“退一步”阶段，先让模型输出抽象概念或第一原理。这样模型的注意力从细节转向本质，后续推理更有方向感。

2. **抽象结果作为显式的推理指南**  
   传统 CoT 只把步骤写出来，模型自行决定每一步的细节；本文把抽象得到的概念/原理直接喂回模型，作为第二轮提示的核心内容。相当于给模型提供了一张“解题路线图”，显著降低了走弯路的概率。

3. **跨模型、跨任务的通用性验证**  
   作者在 PaLM‑2L、GPT‑4、Llama2‑70B 三种不同规模、不同训练方式的模型上都实验，发现退一步提示均能带来提升。这表明该技巧不是针对特定模型的专属技巧，而是一种普适的提示策略。

4. **无需额外微调，仅靠两轮对话实现**  
   与需要专门微调或训练专用解码器的方案不同，Step‑Back Prompting 只在推理时多加一次交互，保持了使用的大语言模型的原始能力，降低了部署成本。

### 方法详解
**整体思路**：把一次完整的问答拆成两步。第一步，给模型一个“抽象”指令，让它把原始问题压缩成高层概念或第一原理；第二步，把这些抽象结果和原始问题一起重新提示模型，让它在抽象的指引下完成推理并输出答案。

**步骤拆解**  
1. **构造 Step‑Back 提示**  
   - 输入格式示例：  
     ```
     问题：<原始问题文本>  
     请先把这个问题抽象成核心概念或涉及的基本原理，用简短的句子描述，不要直接给出答案。
     ```
   - 这里的“抽象”指令明确告诉模型只输出概念，不做计算。

2. **模型生成抽象输出**  
   - LLM 返回一段文字，如“涉及能量守恒、动量守恒以及弹性碰撞的基本原理”。这一步相当于模型的“思考笔记”。

3. **构造二次提示**  
   - 将抽象结果拼回原始问题，形成新的提示：  
     ```
     问题：<原始问题>  
     关键概念：<模型在步骤1给出的抽象>  
     请基于上述关键概念一步步推理，给出最终答案。
     ```
   - 这里把抽象结果显式标记为“关键概念”，帮助模型把注意力聚焦在这些信息上。

4. **模型完成推理并输出答案**  
   - LLM 在第二轮提示下进行思维链式推理或直接给出答案，整体过程比一次性直接回答更稳健。

**类比**：想象你在做一道物理题，老师先让你写出涉及的定律（抽象），再让你用这些定律解题。学生如果直接跳到计算，往往会遗漏关键定律；先写定律再算，成功率更高。

**最巧妙的地方**：只增加一次“抽象”交互，却能把模型的注意力从噪声细节转向核心原理，几乎不增加计算成本。这个设计利用了 LLM 本身已经具备的抽象能力，却把它显式化为后续推理的输入。

### 实验与效果
- **测试任务**：包括 MMLU（Physics、Chemistry 两个子集）、TimeQA、MuSiQue，以及其他 STEM 与知识问答基准。MMLU 是衡量模型学科知识的标准测评，TimeQA 关注时间推理，MuSiQue 侧重多跳推理。
- **基线对比**：与直接提示（Zero‑Shot）和思维链（CoT）两种常用基线比较。  
  - 在 PaLM‑2L 上，Physics 子集提升 7%，Chemistry 提升 11%。  
  - TimeQA 上提升 27%。  
  - MuSiQue 上提升 7%。  
  这些数字表明退一步提示在不同任务上都有可观的增益，尤其是需要时间线推理的任务提升最为显著。
- **消融实验**：论文报告了去掉抽象阶段或不把抽象结果回填的实验，性能回落到接近普通 CoT 的水平，说明抽象本身和抽象结果的回馈都是关键因素。
- **局限性**：作者指出抽象质量依赖模型的内部知识储备，若模型本身对某领域了解不足，抽象可能产生误导；此外，当前实验只在单轮“抽象+求解”两步完成，未探索更深层次的多层抽象。原文未提供对极长上下文或极端噪声输入的评估。

### 影响与延伸思考
这篇工作在提示工程社区引发了对“层次化思考”的关注。随后出现的研究如 **Self‑Consistency**、**Self‑Reflection**、以及 **Decomposition Prompting** 都在不同程度上尝试让模型先拆解或抽象问题再求解。可以预见，未来会有更多自动化的抽象生成器，甚至把抽象阶段当作微调任务来进一步提升模型的概念提取能力。想深入了解的读者可以关注 **Hierarchical Prompting**、**Meta‑Reasoning** 方向的最新论文，或尝试在自己的任务上实验多层次的“抽象‑推理‑抽象”循环。

### 一句话记住它
让大模型先退一步抽象出核心概念，再用这些概念指引推理，显著提升复杂任务的正确率。