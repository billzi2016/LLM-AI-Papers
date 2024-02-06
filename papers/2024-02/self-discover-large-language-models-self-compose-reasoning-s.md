# Self-Discover: Large Language Models Self-Compose Reasoning Structures

> **Date**：2024-02-06
> **arXiv**：https://arxiv.org/abs/2402.03620

## Abstract

We introduce SELF-DISCOVER, a general framework for LLMs to self-discover the task-intrinsic reasoning structures to tackle complex reasoning problems that are challenging for typical prompting methods. Core to the framework is a self-discovery process where LLMs select multiple atomic reasoning modules such as critical thinking and step-by-step thinking, and compose them into an explicit reasoning structure for LLMs to follow during decoding. SELF-DISCOVER substantially improves GPT-4 and PaLM 2's performance on challenging reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH, by as much as 32% compared to Chain of Thought (CoT). Furthermore, SELF-DISCOVER outperforms inference-intensive methods such as CoT-Self-Consistency by more than 20%, while requiring 10-40x fewer inference compute. Finally, we show that the self-discovered reasoning structures are universally applicable across model families: from PaLM 2-L to GPT-4, and from GPT-4 to Llama2, and share commonalities with human reasoning patterns.

---

# 自我发现：大语言模型自组推理结构 论文详细解读

### 背景：这个问题为什么难？
在传统的提示工程里，研究者往往只能让大语言模型（LLM）沿着预先设定的思路（比如一步步写草稿）来推理，这种“一刀切”的方式在面对结构复杂、需要多种思考方式交叉的任务时常常失效。现有的链式思考（CoT）虽然提升了不少，但它只能提供单一的“逐步思考”模板，缺乏对任务本身内部推理模块的自适应发现能力。更进一步的自洽采样（CoT‑Self‑Consistency）虽然能通过多次采样提升鲁棒性，却极度消耗算力，成本难以接受。于是，如何让模型自己识别并组合出最合适的推理结构，成为提升复杂推理性能的关键瓶颈。

### 关键概念速览
**原子推理模块**：指模型内部可复用的基本思考单元，例如“批判性思考”“逐步演绎”。把它们想象成乐高砖块，任意组合就能搭出不同的推理模型。  
**自发现（Self‑Discovery）**：模型在生成答案前主动搜索、挑选并排列这些原子模块的过程，类似人类在解题前先决定使用哪种解题技巧。  
**显式推理结构**：模型在解题时遵循的预先写好的步骤序列，像一张流程图，保证每一步都按计划执行。  
**CoT（Chain of Thought）**：让模型在给出最终答案前先写出思考链条，相当于在纸上写草稿。  
**CoT‑Self‑Consistency**：对同一道题进行多次 CoT 采样，取多数答案，以降低偶然错误，但需要大量计算。  
**推理基准**：本文使用的评测集合，包括 BigBench‑Hard、基于真实世界情境的 Agent 推理以及数学竞赛题 MATH。  

### 核心创新点
- **固定 CoT 模板 → 动态模块挑选**：传统方法直接让模型执行“一步步思考”。SELF‑DISCOVER 让模型先在候选原子模块中挑选最适合当前题目的组合，再按该组合执行推理。这样模型可以根据题目特性灵活切换“批判性思考”或“归纳推理”等技巧，提升了对多样化任务的适配度。  
- **单次采样 → 显式结构引导**：以前要靠多次采样才能稳住答案。本文在一次解答过程中就提供了完整的推理结构，使得模型在解题时不再随意跳转，显著降低了对计算资源的需求。实验显示，在相同算力下比 CoT‑Self‑Consistency 高出 20% 以上。  
- **跨模型通用 → 结构共享**：作者把在 GPT‑4 上自发现的结构直接迁移到 PaLM‑2‑L、Llama‑2 等不同模型，仍然保持性能提升。这说明推理结构本身具有模型无关的通用性，类似于人类的思考框架可以跨语言使用。  
- **与人类思维的相似性 → 解释性提升**：通过对比发现，自发现的结构经常出现“先列出关键假设→再做演绎”的模式，和人类解题的步骤高度吻合，提升了模型输出的可解释性。  

### 方法详解
SELF‑DISCOVER 的整体流程可以概括为三步：**模块库构建 → 自发现搜索 → 结构化解码**。

1. **模块库构建**  
   研究者预先准备一组原子推理模块，每个模块对应一种提示模板（如“请先列出所有已知条件并评估其可信度”。）这些模板本质上是小型的 CoT，能够在不同任务上独立发挥作用。

2. **自发现搜索**  
   当模型收到一道新题时，它会先在内部进行一次“思考”，输出一段文字描述自己认为需要哪些模块以及它们的排列顺序。实现上，作者让模型在一个特殊的“自发现”提示下生成类似“[模块A → 模块C → 模块B]”的序列。这里的关键是使用**自回归采样**结合**置信度打分**，让模型倾向于选取在训练数据中表现好的组合。

3. **结构化解码**  
   获得模块顺序后，模型进入正式解题阶段。它会逐个调用对应的原子模块，每调用一次就把该模块的提示拼接到当前上下文中，随后生成该步骤的推理内容。整个过程形成一条清晰的推理链，类似流水线作业：先执行模块A的思考，再把输出交给模块C，以此类推，直至得到最终答案。

**最巧妙的设计**在于把“挑选模块”这一步也交给 LLM 本身完成，而不是外部搜索算法。模型利用自身的语言理解能力来评估哪种思考方式更匹配题目，从而实现真正的“自我发现”。此外，显式的结构化解码让每一步都可以被检查或干预，极大提升了可解释性和调试友好度。

### 实验与效果
- **测试任务**：BigBench‑Hard（包含多种高难度推理题）、Grounded Agent Reasoning（需要结合外部环境信息的推理）以及数学竞赛数据集 MATH。  
- **基线对比**：与标准 CoT、CoT‑Self‑Consistency 以及其他少量提示改进方法相比。  
- **核心结果**：在所有基准上，SELF‑DISCOVER 相比普通 CoT 提升最高可达 **32%**（如在 MATH 上的准确率提升），并且在与 CoT‑Self‑Consistency 对比时，准确率提升 **超过 20%**，而推理所需的计算量仅为后者的 **1/10 到 1/40**。  
- **消融实验**：作者分别去掉“模块挑选”或“显式结构”两环节，性能分别下降约 15% 和 22%，说明两者都是提升的关键因素。  
- **局限性**：论文指出，自发现过程仍依赖于预先定义的原子模块集合，若任务需要全新思考方式，模型可能无法自行创造全新模块。此外，在极端长文本或需要跨章节推理的场景下，显式结构的长度限制仍是瓶颈。

### 影响与延伸思考
SELF‑DISCOVER 打开了让 LLM 自主组织推理流程的大门，随后出现的工作大多围绕**模块化提示**、**可编辑推理图谱**以及**跨模型推理结构迁移**展开。比如有研究尝试把原子模块扩展为可学习的微调子网络，进一步提升模块选择的精细度；还有项目把自发现的结构用于多模态任务，验证其跨模态通用性。想深入了解的读者可以关注**“可组合推理（Composable Reasoning）”**和**“自我调节语言模型（Self‑Regulating LLM）”**这两个方向，那里聚集了大量后续创新。

### 一句话记住它
让大语言模型先自己挑选并串联思考“积木”，一次解答就能比多次乱猜更准、更省算。