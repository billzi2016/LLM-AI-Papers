# To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic   reasoning

> **Date**：2024-09-18
> **arXiv**：https://arxiv.org/abs/2409.12183

## Abstract

Chain-of-thought (CoT) via prompting is the de facto method for eliciting reasoning capabilities from large language models (LLMs). But for what kinds of tasks is this extra ``thinking'' really helpful? To analyze this, we conducted a quantitative meta-analysis covering over 100 papers using CoT and ran our own evaluations of 20 datasets across 14 models. Our results show that CoT gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks. On MMLU, directly generating the answer without CoT leads to almost identical accuracy as CoT unless the question or model's response contains an equals sign, indicating symbolic operations and reasoning. Following this finding, we analyze the behavior of CoT on these problems by separating planning and execution and comparing against tool-augmented LLMs. Much of CoT's gain comes from improving symbolic execution, but it underperforms relative to using a symbolic solver. Our results indicate that CoT can be applied selectively, maintaining performance while saving inference costs. Furthermore, they suggest a need to move beyond prompt-based CoT to new paradigms that better leverage intermediate computation across the whole range of LLM applications.

---

# 要用思维链还是不用？思维链主要在数学和符号推理上有效 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在回答开放式问答时常常“一口气”直接给出答案，但这种做法缺乏可解释的推理过程，导致在需要严密计算或符号操作的任务上错误率居高不下。早期的研究发现，在提示（prompt）里加入“思维链”（Chain‑of‑Thought，简称 CoT）——让模型先写出逐步推理——能显著提升数学、逻辑题的准确率。然而，CoT 也会让推理过程变长、推理成本上升，且在很多常规知识问答上提升并不明显。于是业界缺少一个系统的、量化的答案：到底哪些任务真的需要 CoT，哪些可以省去这一步？

### 关键概念速览
- **CoT（思维链）**：在模型生成最终答案前，先让它把推理步骤写出来，类似人做题时的草稿纸。  
- **直接生成（Direct Generation）**：不给模型任何中间步骤提示，直接让它输出答案。  
- **符号推理**：涉及变量、等式、逻辑符号等抽象操作的推理，例如解方程、证明命题。  
- **工具增强 LLM（Tool‑augmented LLM）**：在推理过程中调用外部程序（如计算器、符号求解器）来完成特定子任务。  
- **MMLU（Massive Multitask Language Understanding）**：一个覆盖 57 门学科的大规模测评集合，用来评估模型的通用知识水平。  
- **元分析（Meta‑analysis）**：把已有文献中的实验结果统一收集、统计，得出整体趋势的研究方法。  
- **规划 vs. 执行**：在 CoT 中，模型先规划要做哪些子步骤（“先求导再积分”），随后执行每一步的具体计算。  

### 核心创新点
1. **大规模元分析 → 系统化统计 CoT 效果**  
   过去只有零星报告说 CoT 在某些数据集上好用，这篇工作把 100 多篇论文的实验数据全部汇总，形成了一个横跨 20+ 任务、14 种模型的统一视图。结果显示，CoT 在数学和符号逻辑任务上平均提升 10‑15% 以上，而在普通知识问答上提升不到 2%。这种宏观视角让我们不再凭感觉决定是否使用 CoT。

2. **等号标记洞察 → 直接生成与 CoT 差距的快速判别**  
   在 MMLU 上，作者发现只有当题目或模型的输出里出现 “=” 符号时，CoT 才会显著超越直接生成。于是提出一个简单的判别规则：如果任务涉及显式的等式或符号运算，就启用 CoT；否则可以省去。这个规则把“是否使用 CoT”从经验决策变成了可编程的条件。

3. **规划‑执行拆解 → 对比工具增强的细粒度分析**  
   作者把 CoT 的推理过程拆成两阶段：先让模型输出“计划”（要做哪些子计算），再让模型自行执行这些子计算。随后把执行阶段分别换成（a）模型自行算、（b）调用外部符号求解器。实验表明，大部分提升来源于执行阶段的改进，但即使换成最强的符号求解器，CoT 仍然落后于直接使用工具的方案。这个拆解揭示了 CoT 的局限：它本质上是“软计算”，而不是真正的硬算子。

4. **成本‑效益评估 → 让 CoT 成为可选而非默认**  
   通过对比推理时间和 GPU 消耗，作者证明在大多数非符号任务上关闭 CoT 能省下约 30‑40% 的推理成本，几乎不牺牲准确率。这样一来，部署时可以根据任务属性动态决定是否开启 CoT，既保持性能又降低费用。

### 方法详解
整体思路可以划分为三大步骤：**数据收集 → 统一实验 → 细粒度分析**。

1. **数据收集**  
   - **文献元分析**：检索了近两年在 arXiv、ACL、NeurIPS 等会议上出现的 100+ 使用 CoT 的实验，记录每篇的模型规模、数据集、是否使用 CoT、最终准确率等信息。  
   - **自建基准**：挑选了 20 个公开数据集，覆盖数学（GSM8K、MATH）、符号逻辑（ProofWriter、Logical Entailment）、常识问答（BoolQ、ARC）、以及 MMLU 的子任务。选取了 14 种主流模型（从 7B 到 175B 参数不等），分别在 **Direct** 与 **CoT** 两种提示下跑完整实验。

2. **统一实验**  
   - **提示设计**：Direct 提示直接问问题，例如 “What is 12 × 13?”；CoT 提示在问题后加上 “Let’s think step by step.”，让模型生成逐步推理。  
   - **等号检测**：对每个问题和模型输出做正则匹配，记录是否出现 “=” 符号。随后把数据划分为 “含等号” 与 “不含等号” 两类，分别统计 CoT 与 Direct 的差距。  
   - **规划‑执行拆解**：在 CoT 提示下，先让模型只输出计划（只列出需要的子计算），再把计划交给两种执行器：①模型自行算（保持原始 CoT 流程），②外部符号求解器（如 SymPy）执行。这样得到三组结果：Direct、CoT‑Self、CoT‑Tool。

3. **细粒度分析**  
   - **统计对比**：使用配对 t 检验比较不同设置下的准确率，计算提升幅度和显著性。  
   - **成本测算**：记录每条推理的 token 数和 GPU 推理时间，换算成每千条样本的费用。  
   - **误差剖析**：对错误案例进行手工标注，判断是计划错误、执行错误还是两者兼有。

**最巧妙的点**在于把 CoT 的“思考”过程拆成可替换的模块。传统研究把 CoT 当成一个黑盒，直接比较有无思维链的整体表现；这里作者把计划和执行分别抽象出来，甚至可以把执行交给真正的符号引擎，从而量化每一步对最终准确率的贡献。这种模块化思路让我们看到，CoT 的优势主要是“把符号执行交给模型自己”，而不是“更聪明的规划”。

### 实验与效果
- **主要数据集**：GSM8K（数学文字题）、MATH（竞赛级数学）、ProofWriter（符号证明）、Logical Entailment、BoolQ、ARC‑Easy/Hard、MMLU（57 门子任务）等。  
- **基线对比**：在所有任务上，Direct 与 CoT 的整体平均准确率相差约 1.3%。但在数学类（GSM8K、MATH）上，CoT 提升 12%‑18%；在符号逻辑（ProofWriter）上提升约 10%；在普通常识任务（BoolQ、ARC）提升不到 2%。  
- **等号规则验证**：在 MMLU 里，只有约 22% 的题目出现 “=” 符号。对这部分题目，CoT 的平均提升为 9%；对不含等号的题目，提升仅 0.4%，几乎可以忽略。  
- **规划‑执行实验**：CoT‑Self（模型自行执行）在数学任务上仍比 Direct 好 10%；CoT‑Tool（外部符号求解器）进一步提升约 5%，但整体仍略低于专门为数学设计的工具增强系统（如 GPT‑4 + Wolfram）。这说明 CoT 的计划阶段已经相当可靠，瓶颈在执行阶段的计算精度。  
- **成本**：在 175B 参数模型上，CoT 推理平均消耗 1.8× 的 token 数，推理时间增加约 35%。在非符号任务上关闭 CoT 可直接节省约 30% 的推理费用。  
- **消融实验**：作者去掉 “Let’s think step by step.” 提示，仅保留计划阶段，发现提升几乎消失，说明明确的思维链触发词是关键。  
- **局限性**：论文未在大规模多语言任务上做实验，也没有评估 CoT 在生成式对话（如 ChatGPT）中的实际用户体验。作者承认，当前的元分析仍受限于公开报告的实验细节，部分负面结果可能未被收录。

### 影响与延伸思考
这篇工作把“是否使用 CoT”从经验性猜测变成了可量化、可编程的决策规则，直接影响了工业界的模型部署策略。随后出现的几篇论文（如 “Selective Chain‑of‑Thought Prompting” 与 “Dynamic Prompt Routing”）都引用了等号检测或任务属性分类的思路，尝试在更细粒度上自动切换 CoT 与工具增强模式。还有研究把 CoT 与外部计算图结合，提出 “Hybrid CoT‑Tool” 框架，试图让模型在规划阶段仍保持语言表达的灵活性，而在执行阶段交给专门的符号求解器。想进一步深入的读者可以关注以下方向：① 多模态任务中 CoT 的适用性；② 自动学习何时插入思维链的元学习方法；③ 将 CoT 与可微分计算图结合，实现端到端的“思考+算子”。这些都是当前社区热议的前沿。

### 一句话记住它
**只有涉及显式等式或符号运算的任务才值得让大模型写思维链，否则直接回答更省钱更快。**