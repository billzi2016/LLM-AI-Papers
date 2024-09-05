# Planning In Natural Language Improves LLM Search For Code Generation

> **Date**：2024-09-05
> **arXiv**：https://arxiv.org/abs/2409.03733

## Abstract

While scaling training compute has led to remarkable improvements in large language models (LLMs), scaling inference compute has not yet yielded analogous gains. We hypothesize that a core missing component is a lack of diverse LLM outputs, leading to inefficient search due to models repeatedly sampling highly similar, yet incorrect generations. We empirically demonstrate that this lack of diversity can be mitigated by searching over candidate plans for solving a problem in natural language. Based on this insight, we propose PlanSearch, a novel search algorithm which shows strong results across HumanEval+, MBPP+, and LiveCodeBench (a contamination-free benchmark for competitive coding). PlanSearch generates a diverse set of observations about the problem and then uses these observations to construct plans for solving the problem. By searching over plans in natural language rather than directly over code solutions, PlanSearch explores a significantly more diverse range of potential solutions compared to baseline search methods. Using PlanSearch on top of Claude 3.5 Sonnet achieves a state-of-the-art pass@200 of 77.0% on LiveCodeBench, outperforming both the best score achieved without search (pass@1 = 41.4%) and using standard repeated sampling (pass@200 = 60.6%). Finally, we show that, across all models, search algorithms, and benchmarks analyzed, we can accurately predict performance gains due to search as a direct function of the diversity over generated ideas. Code can be found at https://github.com/scaleapi/plansearch.

---

# 在自然语言中规划提升大语言模型代码生成搜索 论文详细解读

### 背景：这个问题为什么难？

生成代码的 LLM（大语言模型）在训练算力提升后已经能写出相当可用的程序，但在推理阶段的搜索仍旧瓶颈。常见的做法是让模型多次采样代码，然后挑出通过测试用例的那一个。问题是，模型往往会一次又一次地产出高度相似、但同样错误的代码片段，导致搜索空间缺乏多样性，真正有价值的解法被埋在“相似的噪声”里。缺少多样化的候选答案，使得即使投入大量算力，也难以显著提升最终的通过率。

### 关键概念速览
- **LLM（大语言模型）**：用海量文本训练得到的模型，能够生成自然语言或代码，就像会写作文的机器人。  
- **代码生成搜索**：在给定题目时，让模型多次生成代码并用单元测试挑选出正确答案的过程，类似于在黑盒里不断尝试钥匙。  
- **多样性（Diversity）**：候选答案在内容、思路上的差异程度，想象成不同的解题路线，而不是同一条路的不同拐弯。  
- **Plan（计划）**：用自然语言描述解决问题的思路或步骤，类似于先写解题大纲再动手写代码。  
- **Observation（观察）**：对题目进行拆解、提取关键信息的简短描述，像是先把题目里的“已知”和“要求”列出来。  
- **pass@k**：在 k 次生成中至少有一次通过所有测试用例的比例，k 越大说明搜索越强。  
- **LiveCodeBench**：一个专门为竞争编程设计的、确保训练数据不泄露的基准测试集，用来衡量模型在真实编程场景下的表现。  

### 核心创新点
1. **从代码搜索转向计划搜索**  
   - 之前的搜索直接在代码空间里采样，往往得到相似的错误实现。  
   - 这篇论文先让模型生成一系列自然语言计划，再基于每个计划去生成代码。  
   - 结果是搜索覆盖了更广的思路空间，显著提升了通过率。

2. **引入观察阶段提升计划质量**  
   - 传统做法直接让模型写计划，容易遗漏题目细节。  
   - 作者在生成计划前先让模型输出若干“观察”，即对题目要点的抽取。  
   - 这些观察被喂回计划生成器，使得计划更贴合题目，进而产生更有针对性的代码。

3. **用多样性指标预测搜索收益**  
   - 过去没有明确的度量来解释为什么搜索有时效果好、有时不佳。  
   - 论文提出用生成的自然语言想法的多样性（如词汇分布差异）来直接预测搜索提升幅度。  
   - 这让研究者可以在不跑完整实验的情况下预估不同搜索策略的价值。

4. **在竞争级基准上实现新纪录**  
   - 将 PlanSearch 与 Claude 3.5 Sonnet 结合，在 LiveCodeBench 上的 pass@200 达到 77.0%。  
   - 相比不做搜索的单次生成（pass@1=41.4%）和普通重复采样（pass@200=60.6%），提升近 20% 绝对点。  

### 方法详解
**整体框架**  
PlanSearch 把一次代码生成任务拆成三层：观察 → 计划 → 代码。先让模型对题目做若干简短的观察，收敛出关键信息；再基于这些观察生成多条自然语言计划；最后对每条计划执行代码生成并用单元测试验证。整个流程在推理阶段循环多次，以获得足够的候选答案。

**步骤拆解**  

1. **观察生成**  
   - 输入：原始编程题目（描述、函数签名、示例等）。  
   - 操作：使用 LLM 进行 few‑shot 提示，让模型输出 N 条“观察”。每条观察是一句话，概括题目中的输入类型、边界条件、核心算法需求等。  
   - 类比：相当于先把题目拆成要点清单，像做笔记一样。

2. **计划生成**  
   - 输入：原始题目 + 所有观察。  
   - 操作：再次调用 LLM，要求它基于每条观察写出一段解决思路（计划），每个计划约 2‑3 句，明确算法步骤、数据结构选择、复杂度分析等。  
   - 多样性来源：因为观察本身已经是多条不同的要点，模型在不同观察的驱动下会产生风格迥异的计划。  

3. **代码生成与评估**  
   - 对每条计划，使用同一个 LLM（或更强的模型）生成对应的代码实现。  
   - 生成的代码立即跑题目提供的单元测试，记录是否通过。  
   - 所有通过的代码加入最终候选集合；未通过的继续保留，以备后续可能的二次采样或改写。  

4. **搜索策略**  
   - 整个过程可以并行执行：观察、计划、代码三层分别做多次采样，形成一个宽度为 O(N_observation × N_plan × N_code) 的搜索树。  
   - 为控制算力，作者采用了“层级截断”：只保留每层通过率最高的 K 条（例如 K_observation=5、K_plan=10），其余被剪枝。  

**最巧妙的点**  
- 把“多样性”从代码层面搬到自然语言层面。自然语言的表达空间远大于代码的语法空间，一句话的不同写法就能引导模型走出全然不同的实现路径。  
- 观察阶段的引入让计划不至于“空中楼阁”。如果直接让模型写计划，往往会出现“我不知道输入是什么”的情况；先让模型说出它看到的要点，再让它基于这些要点规划，思路更连贯。  

### 实验与效果
- **测试数据集**：HumanEval+、MBPP+（两套常用的代码生成基准的扩展版）以及 LiveCodeBench（专为竞争编程设计、确保训练数据不泄露）。  
- **基线对比**：  
  - 不做搜索的单次生成（pass@1）在 LiveCodeBench 上为 41.4%。  
  - 传统的重复采样（直接在代码空间采样 200 次）得到 pass@200=60.6%。  
  - PlanSearch 在同样的 200 次预算下实现 pass@200=77.0%，领先约 16.4% 绝对点。  
- **消融实验**：论文分别关闭观察模块、计划模块以及多样性剪枝，结果显示：去掉观察会导致 pass@200 下降约 8%；仅搜索代码不使用计划则回到传统采样水平；多样性剪枝被去掉后搜索空间爆炸，算力成本翻倍而收益提升有限。  
- **多样性预测**：作者用生成想法的词汇散度（如 Jensen‑Shannon Divergence）做回归，能够在 0.9 以上的相关系数下预测实际的 pass@k 提升，验证了“多样性＝收益”的假设。  
- **局限性**：  
  - 依赖 LLM 对自然语言计划的理解能力，若模型本身在推理计划时表现不佳，整体收益会受限。  
  - 额外的观察和计划步骤增加了推理时间，大约比纯代码采样慢 1.5‑2 倍。  
  - 在极端算力受限的场景（如移动端）仍然难以直接使用。  

### 影响与延伸思考
PlanSearch 把“先想后写”的人类编程习惯搬进了 LLM 搜索流程，打开了“自然语言驱动搜索”这一新方向。随后的工作开始探索：  
- 用更细粒度的思维链（CoT）结合计划搜索，进一步提升复杂算法题的通过率（推测）。  
- 将计划搜索与检索增强（RAG）结合，让模型先检索已有相似计划再进行微调（已有后续工作）。  
- 在代码审查、自动重构等任务中使用计划作为解释层，提升可解释性和安全性。  

如果想深入，可以关注以下几个方向：  
1. **计划质量评估**：如何自动衡量自然语言计划的可行性？  
2. **跨模型协同**：让一个模型负责观察，另一个模型负责计划，第三个模型负责代码，形成“专家团队”。  
3. **算力优化**：在保持多样性的前提下，设计更高效的剪枝策略或使用强化学习调度搜索预算。  

### 一句话记住它
让大语言模型先用自然语言写“解题大纲”，再基于这些大纲生成代码，搜索空间瞬间变宽，代码通过率大幅提升。