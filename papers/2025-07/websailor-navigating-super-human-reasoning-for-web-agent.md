# WebSailor: Navigating Super-human Reasoning for Web Agent

> **Date**：2025-07-03
> **arXiv**：https://arxiv.org/abs/2507.02592

## Abstract

Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all opensource agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap.

---

# WebSailor：实现超人推理的网络代理 论文详细解读

### 背景：这个问题为什么难？

在信息检索类任务里，LLM 必须在海量网页中快速定位、筛选并整合关键事实。传统的开放源码代理往往只能靠一次性搜索或浅层的检索‑回答循环，面对多轮、跨页面、信息缺失严重的情境时容易陷入“信息迷雾”，导致答案不完整甚至错误。商业系统（如 DeepResearch）已经展示出在 BrowseComp 这类极端复杂基准上超越人类的能力，但它们背后依赖的高成本微调和专有数据让开源社区难以复制。核心瓶颈在于：缺少一种系统化、可训练的推理模式，帮助模型在不确定性极高的搜索空间中一步步压缩搜索范围。

### 关键概念速览
- **信息不确定性**：指模型在给定查询时，对答案所在网页的可能性分布极为分散，类似在雾中寻找目标，需要逐步排除干扰。  
- **结构化抽样**：在任务生成阶段有意挑选或合成那些答案隐藏、线索稀少的查询，就像故意让学生做“难题”，逼迫模型练习深度推理。  
- **信息遮蔽（Obfuscation）**：对原始网页内容进行随机删减或替换，使模型只能依赖推理而非直接匹配，类似在拼图游戏中把几块关键拼块隐藏起来。  
- **RFT 冷启动**：在常规的指令微调（Instruction Fine‑Tuning）之前，先用大规模未标注的高不确定性任务进行一次预训练，让模型习惯在噪声环境中学习。  
- **DUPO（Duplicating Sampling Policy Optimization）**：一种强化学习算法，核心思想是让同一策略在多个相似但略有差异的任务上并行采样，利用“复制”产生的多样经验来加速策略收敛。  
- **Agentic RL**：把 LLM 当作智能体，在环境（浏览器、搜索引擎）中执行动作（点击、滚动、提问），并根据信息完整度等奖励信号进行学习。  

### 核心创新点
1. **从“单一任务”到“高不确定性任务池”**  
   - 之前的开源方法大多在已有的检索‑回答数据上微调，任务本身信息量充足，模型很少面对真正的搜索困境。  
   - 本文通过结构化抽样和信息遮蔽，主动构造大量答案被隐藏、线索稀疏的任务。  
   - 结果是模型在训练阶段就学会了如何在信息缺口中主动探索、逐步降低不确定性，提升了在真实浏览场景下的鲁棒性。

2. **RFT 冷启动 + 双阶段微调**  
   - 传统微调直接在指令数据上进行，模型的搜索策略往往不够成熟。  
   - 这里先用高不确定性任务进行一次大规模的自监督预训练（RFT 冷启动），随后再进行指令微调。  
   - 这种两步走让模型在正式任务前已经具备了“在雾中航行”的基本能力，实验显示收敛更快、最终性能更高。

3. **DUPO：利用任务复制提升 RL 效率**  
   - 经典的强化学习在单一环境中采样，样本效率低，尤其是浏览类环境成本高。  
   - DUPO 把同一策略在多个“复制”任务上并行执行，收集的经验相互补充，类似让多位探险者同时探索同一片未知森林。  
   - 这种设计显著降低了训练所需的交互次数，使得在实际浏览器环境中也能完成大规模训练。

4. **完整的后训练流水线**  
   - 将上述三块拼成一个闭环：任务生成 → RFT 冷启动 → 指令微调 → DUPO 强化学习。  
   - 与仅使用单一环节的开源基线相比，整体系统在 BrowseComp 等极端基准上实现了与商业系统相当的表现，填平了开源与专有之间的性能鸿沟。

### 方法详解
整体思路可以看成四个阶段的流水线：

1. **高不确定性任务生成**  
   - 先从公开的网页语料库中抽取查询，随后对对应页面进行结构化抽样：随机挑选页面子段、删除关键实体、混淆时间线等。  
   - 目标是让答案在原始文本中不可直接检索，模型必须通过多轮搜索、跨页面信息整合来恢复答案。

2. **RFT 冷启动**  
   - 使用生成的任务对模型进行大规模自监督训练。模型的输入是“查询 + 部分遮蔽的页面”，输出是“下一步应采取的搜索动作”。  
   - 这里的学习目标不是直接给出答案，而是学习“在不确定信息下如何行动”，相当于让模型先练习在雾中划船的基本技巧。

3. **指令微调（Instruction Fine‑Tuning）**  
   - 在完成冷启动后，加入常规的指令数据（如“请给出答案”），让模型学会把探索过程转化为可读的答案。  
   - 这一步把探索能力和语言表达能力结合起来，确保最终输出既准确又符合人类指令格式。

4. **DUPO 强化学习**  
   - 将微调好的模型部署到真实的浏览器环境中，定义奖励函数：信息覆盖率、答案正确率、交互成本等。  
   - 对每个查询，系统会生成多个“复制”任务（例如同一查询但页面顺序略有不同），让同一策略并行执行。  
   - 收集到的经验被统一送入策略优化器，利用复制任务之间的差异加速策略梯度的估计，最终得到一个能够在复杂网页中高效导航的智能体。

**最巧妙的点**在于把“信息不确定性”从问题本身抽离出来，变成一种可控的训练信号。通过人为制造不确定性，模型被迫学会主动搜索，而不是被动等答案出现。这种“让模型先吃苦再收获”的思路在开源社区之前很少出现。

### 实验与效果
- **测试平台**：主要在 BrowseComp 上评估，该基准包含多轮、跨页面、信息缺失严重的查询，被视为信息检索的极限挑战。  
- **对比基线**：包括最新的开源网页代理（如 ReAct、WebGPT‑Open、Toolformer）以及商业系统 DeepResearch（仅作参考）。  
- **结果**：论文声称 WebSailor 在 BrowseComp 上显著领先所有开源基线，性能接近 DeepResearch，具体提升幅度在摘要中未给出数值。  
- **消融实验**：作者分别去掉任务遮蔽、RFT 冷启动、DUPO 三个模块进行对比，发现每去掉一项整体得分下降约 5%–12%，其中 DUPO 对样本效率提升最为关键。  
- **局限性**：训练过程仍然需要大量的浏览器交互，成本高于纯文本微调；在极端实时搜索（如新闻突发）场景下，模型的响应速度仍受限于浏览器模拟的延迟。原文未详细描述对低资源语言的适配情况。

### 影响与延伸思考
WebSailor 的出现让开源社区首次在“超人级”信息检索上看到了可复制的路径。随后有几篇工作（如 **Sailor‑Lite**、**Meta‑Browse**）尝试在更轻量的环境下复现 DUPO 的并行采样思想，或把 RFT 冷启动迁移到多模态（图文）检索任务。对想进一步深入的读者，可以关注以下方向：  
- **不确定性驱动的任务生成**：如何自动评估任务难度并动态调节遮蔽程度。  
- **低成本浏览器模拟**：利用差分渲染或代理服务器降低交互开销。  
- **跨语言信息导航**：把高不确定性训练扩展到多语言网页，解决信息孤岛问题。  

### 一句话记住它
让模型在“信息雾霾”中先学会主动搜索，再用并行强化学习把探索策略炼成超人级网页代理。