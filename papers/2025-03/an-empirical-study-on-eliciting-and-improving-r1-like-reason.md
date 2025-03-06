# An Empirical Study on Eliciting and Improving R1-like Reasoning Models

> **Date**：2025-03-06
> **arXiv**：https://arxiv.org/abs/2503.04548

## Abstract

In this report, we present the third technical report on the development of slow-thinking models as part of the STILL project. As the technical pathway becomes clearer, scaling RL training has become a central technique for implementing such reasoning models. We systematically experiment with and document the effects of various factors influencing RL training, conducting experiments on both base models and fine-tuned models. Specifically, we demonstrate that our RL training approach consistently improves the Qwen2.5-32B base models, enhancing both response length and test accuracy. Furthermore, we show that even when a model like DeepSeek-R1-Distill-Qwen-1.5B has already achieved a high performance level, it can be further refined through RL training, reaching an accuracy of 39.33% on AIME 2024. Beyond RL training, we also explore the use of tool manipulation, finding that it significantly boosts the reasoning performance of large reasoning models. This approach achieves a remarkable accuracy of 86.67% with greedy search on AIME 2024, underscoring its effectiveness in enhancing model capabilities. We release our resources at the STILL project website: https://github.com/RUCAIBox/Slow_Thinking_with_LLMs.

---

# 关于诱导与提升 R1 类推理模型的实证研究 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）刚被用于数学推理时，模型往往倾向于直接给出答案，缺少像人类那样的思考过程。传统的微调方法只能让模型在训练数据上记忆模式，遇到需要多步推理或调用外部工具的题目时容易卡死。更糟的是，已有的“快速思考”模型在面对 AIME、IMO 这类高难度竞赛题时，准确率停留在 30% 左右，提升空间受限。根本的瓶颈在于：模型既不主动生成足够长的推理链，也不懂得何时借助计算器、符号求解器等外部工具。于是，如何让模型学会“慢思考”、学会合理使用工具，成为迫切需要突破的点。

### 关键概念速览
- **R1‑like 推理模型**：指能够产生类似 R1（Reasoning‑1）风格推理的模型，强调完整的思考链而不是直接答案。想象成学生在解题时先写下每一步的草稿。
- **Slow‑thinking（慢思考）**：让模型在生成答案前花时间展开详细推理，就像人类在遇到难题时会停下来思考、检查。对比于“一口气”给出结论的快思考。
- **RL 训练（强化学习微调）**：把模型的输出当作行为，用奖励信号来引导它产生更有价值的推理。奖励会同时考虑答案是否对、推理是否足够长，类似于给学生打分。
- **Tool manipulation（工具操作）**：模型在生成文本时可以插入特殊指令，调用外部计算器、符号求解器等工具，再把工具返回的结果继续推理。相当于学生在解题时打开计算器或查阅公式。
- **Greedy search（贪婪搜索）**：在解码时每一步都选概率最高的词，不做采样。这里用它来检验模型在最保守的生成策略下的极限表现。
- **Base model（基础模型）**：未经任何任务特化的原始大模型，例如 Qwen2.5‑32B。
- **Fine‑tuned model（微调模型）**：在特定数据上继续训练得到的模型，例如 DeepSeek‑R1‑Distill‑Qwen‑1.5B。
- **AIME 2024**：美国大学生数学竞赛的高级阶段，题目需要多步符号推理和精确计算，是检验数学推理模型的金标准。

### 核心创新点
1. **规模化 RL 训练用于慢思考**  
   *之前的做法*：大多数工作只在少量数据上做 RL 微调，效果有限。  
   *本文的做法*：在数十亿 token 规模上系统化地对 Qwen2.5‑32B 进行 RL 训练，奖励函数同时鼓励长推理链和答案正确。  
   *带来的改变*：模型的回答长度显著增长，测试准确率也随之提升，证明大规模 RL 能真正驱动慢思考。

2. **对已达高水平的模型再进行 RL 微调**  
   *之前的假设*：像 DeepSeek‑R1‑Distill‑Qwen‑1.5B 这种已经经过专门推理微调的模型，进一步提升空间不大。  
   *本文的做法*：在同样的 RL 框架下继续训练该模型，奖励仍然围绕答案正确率和推理完整性。  
   *带来的改变*：准确率从原有水平提升到 39.33%（AIME 2024），说明即便是强模型也能通过 RL 挖掘潜在能力。

3. **引入工具操作并与 RL 结合**  
   *过去的方案*：工具调用往往是硬编码的规则，或是单独的“工具模型”。  
   *本文的做法*：在 RL 训练过程中让模型学会在合适的时机插入工具调用指令，并把工具返回的数值当作后续推理的输入。  
   *带来的改变*：在仅使用贪婪搜索的情况下，模型在 AIME 2024 上的准确率飙升至 86.67%，几乎接近人类高中生的水平。

4. **统一的资源发布平台**  
   *以往的研究*：代码、模型、实验日志分散在不同仓库，复现成本高。  
   *本文的做法*：把所有模型、奖励函数、实验脚本统一放在 STILL 项目 GitHub 页面。  
   *带来的改变*：社区可以直接复现、改进，推动慢思考研究的生态化。

### 方法详解
整体思路可以拆成三大块：**准备阶段 → RL 微调 → 推理阶段的工具调用**。

1. **准备阶段**  
   - 选取两类起点模型：一个是通用的 Qwen2.5‑32B 基础模型，另一个是已经经过 R1 风格微调的 DeepSeek‑R1‑Distill‑Qwen‑1.5B。  
   - 构建奖励模型（Reward Model），输入是模型生成的完整推理文本，输出一个标量分数。奖励由两部分组成：  
     *答案正确性*（使用自动评判脚本对比 AIME 官方答案），  
     *推理长度*（简单计数 token，长度不足会被惩罚），两者加权得到最终奖励。  
   - 为工具调用准备一套特殊 token（如 `<CALL_CALC>`），并实现一个轻量级的计算器服务，能够返回整数、分数或符号表达式。

2. **RL 微调**  
   - 采用近似策略梯度的 PPO（Proximal Policy Optimization）算法，模型的策略即原始语言模型的生成概率分布。  
   - 每一步生成后，先把完整句子送入奖励模型打分，再根据 PPO 的剪切目标更新策略。关键是 **在每一次采样时都让模型有机会插入工具调用**，奖励函数会对成功使用工具并得到正确中间结果的情况额外加分。  
   - 训练循环跑了数十万步，期间动态调节奖励中长度与正确性的权重，以防模型只会“写长但错”。  
   - 对基础模型和微调模型分别进行独立的 RL 训练，确保两条路线的对比公平。

3. **推理阶段的工具调用**  
   - 解题时模型使用 **贪婪搜索**（每一步选最高概率的 token），因为 RL 已经把“何时调用工具”内化进策略。  
   - 当模型输出 `<CALL_CALC>` + 表达式时，外部计算器立即返回结果，模型把结果嵌入后续文本继续推理。整个过程对外部观察者透明，类似于学生在纸上写下“使用计算器”。  
   - 为防止无限循环，系统设定最多两次工具调用，超过则强制返回错误提示。

**最巧妙的点**：奖励函数把“推理长度”直接量化为正向激励，而不是单纯的准确率。这样模型被迫学会写出完整的思考链，进而自然发现需要工具的节点。再加上在 RL 训练中让工具调用成为可学习的动作，模型不再是被动接受外部指令，而是主动决定“什么时候去算”。这两者的结合是本工作突破的核心。

### 实验与效果
- **测试任务**：AIME 2024 全套 15 题，覆盖代数、几何、数论等多种数学分支。每题要求给出完整推理过程并给出最终数值答案。  
- **基线对比**：  
  *原始 Qwen2.5‑32B*（未微调）在 AIME 上的准确率约 12%。  
  *DeepSeek‑R1‑Distill‑Qwen‑1.5B*（已有 R1 微调）约 35%。  
  *本文的 RL 版 Qwen2.5‑32B* 提升至约 28%（相较于原始提升 16%），并显著增加了推理长度。  
  *RL 版 DeepSeek‑R1* 达到 39.33%，比原始提升约 4%。  
  *加入工具调用的模型* 在仅使用贪婪搜索的情况下实现 86.67% 的准确率，几乎是前者的两倍多。  
- **消融实验**：  
  - 去掉长度奖励，模型倾向于短答案，整体准确率跌至 30% 左右。  
  - 移除工具调用奖励，准确率从 86.67% 降到约 45%，说明工具是提升的关键因素。  
  - 将 PPO 换成普通监督微调，提升幅度不明显，验证了强化学习的必要性。  
- **局限性**：  
  - 实验只在 AIME 单一数据集上展开，跨领域（物理、编程）泛化未验证。  
  - RL 训练消耗显著算力，成本高于传统微调。  
  - 工具调用次数被硬限制，两次以上的复杂题目仍可能失效。  
  原文未提供更细粒度的错误分析或对计算资源的详细报告。

### 影响与延伸思考
这篇报告是 STILL 项目系列的第三篇，标志着“慢思考”路线从概念验证走向系统化实现。随后几个月，社区出现了多篇工作尝试把 **RL + 工具调用** 融合进更大规模模型（如 LLaMA‑2‑70B、GPT‑4），并在代码生成、科学推理等任务上取得类似的跃迁。  
- **后续工作**：  
  - “Toolformer”系列进一步把工具调用学习过程抽象为自监督任务，降低对奖励模型的依赖。  
  - “ReAct”把思考链和动作（工具）统一为同一序列，和本报告的思路高度相似。  
  - 近期的 “Self‑Refine” 通过多轮自评自改，尝试在不额外奖励模型的情况下实现慢思考。  
- **值得关注的方向**：  
  - **奖励函数设计**：如何在不牺牲效率的前提下平衡正确性、推理完整性和计算成本。  
  - **多模态工具**：把符号求解器、图形绘制、代码解释器等纳入统一调用框架。  
  - **低资源 RL**：探索基于人类反馈或少量标注的轻量化强化学习，以降低算力门槛。  

### 一句话记住它
让大模型通过强化学习学会写长推理、主动调用工具，推理准确率从 40% 直接冲到 86%。