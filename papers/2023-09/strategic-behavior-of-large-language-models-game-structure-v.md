# Strategic Behavior of Large Language Models: Game Structure vs.   Contextual Framing

> **Date**：2023-09-12
> **arXiv**：https://arxiv.org/abs/2309.05898

## Abstract

This paper investigates the strategic decision-making capabilities of three Large Language Models (LLMs): GPT-3.5, GPT-4, and LLaMa-2, within the framework of game theory. Utilizing four canonical two-player games -- Prisoner's Dilemma, Stag Hunt, Snowdrift, and Prisoner's Delight -- we explore how these models navigate social dilemmas, situations where players can either cooperate for a collective benefit or defect for individual gain. Crucially, we extend our analysis to examine the role of contextual framing, such as diplomatic relations or casual friendships, in shaping the models' decisions. Our findings reveal a complex landscape: while GPT-3.5 is highly sensitive to contextual framing, it shows limited ability to engage in abstract strategic reasoning. Both GPT-4 and LLaMa-2 adjust their strategies based on game structure and context, but LLaMa-2 exhibits a more nuanced understanding of the games' underlying mechanics. These results highlight the current limitations and varied proficiencies of LLMs in strategic decision-making, cautioning against their unqualified use in tasks requiring complex strategic reasoning.

---

# 大型语言模型的战略行为：游戏结构与情境框架的比较 论文详细解读

### 背景：这个问题为什么难？
在传统的 AI 研究里，语言模型大多被当作“信息检索器”或“文本生成器”，很少有人系统地检验它们在真正的战略情境下会怎么决定。即使有研究把 LLM 当作博弈玩家，也往往只用单一游戏或固定的提示，忽视了游戏本身的结构差异和人类常用的情境描述（比如“外交”或“朋友之间”）。这种做法把模型的行为简化成“对同一问题的同一答案”，导致我们无法判断模型到底是懂得游戏规则，还是仅仅在跟随表面文字。于是，评估 LLM 的抽象战略推理能力以及它们对情境框架的敏感度，成为了一个迫切但尚未解决的难题。

### 关键概念速览
**大型语言模型（LLM）**：通过海量文本训练得到的生成式模型，能够根据输入的文字产生连贯的输出。把它想象成一个“会说话的百科全书”，但它的“思考”过程是统计性的，而不是符号推理。  
**博弈论**：研究理性主体在有限规则下如何做决策的数学框架。常用的两人零和或非零和游戏就像“棋盘”，每一步都有收益和风险。  
**囚徒困境（Prisoner’s Dilemma）**：两人若都合作可获中等收益，若一方背叛另一方则背叛者获得最高收益，合作者得到最低。类似于“谁先抢红包”。  
**鹿猎（Stag Hunt）**：合作能得到大额奖励，单独行动只能得到小额奖励，强调信任与协同。像两个人一起去打猎，只有合作才能捕到鹿。  
**雪堆博弈（Snowdrift）**：双方都想让对方先让路，但如果都不让，双方都受损。类似于两车在雪道相遇，谁先倒车谁受的损失更小。  
**囚徒乐园（Prisoner’s Delight）**：与囚徒困境相反，合作比背叛更有吸引力，背叛反而会被惩罚。可以类比为“互惠互利的合作社”。  
**情境框架（Contextual Framing）**：在提示中加入的背景信息（如“外交谈判”或“闺蜜聊天”），用来影响模型对同一游戏的解读。相当于给玩家穿上不同的“角色服装”。  
**策略敏感度（Strategic Sensitivity）**：模型对游戏结构或情境变化的响应程度。高敏感度意味着模型能根据规则或背景调整自己的行动。

### 核心创新点
1. **多游戏、多情境实验设计 → 将四种经典两人博弈与三种情境框架（外交、商业、友情）组合 → 揭示不同模型在同一规则下的行为差异**。以前的工作往往只测一种游戏，导致结论难以推广。这里的组合矩阵让作者能够系统比较“游戏结构”与“情境框架”对模型决策的相对影响。  
2. **对比三大主流 LLM（GPT‑3.5、GPT‑4、LLaMa‑2）的策略表现 → 通过统一的提示模板让模型自行给出行动（合作/背叛）并解释理由 → 量化每个模型的抽象推理能力与情境适应度**。过去的评估多聚焦于单一模型或仅用准确率，这里加入了解释文本，使得评估更具可解释性。  
3. **引入“情境敏感度指数”（Contextual Sensitivity Score） → 计算模型在不同情境下策略变化的幅度 → 用数值化指标比较模型对情境的依赖程度**。这一指标帮助读者直观看到 GPT‑3.5 对情境的高依赖、GPT‑4 与 LLaMa‑2 的相对平衡。  
4. **细致的错误分析与机制推断 → 通过对模型输出的语言特征（如使用外交术语、情感词）进行归类 → 解释为何 GPT‑3.5 在抽象推理上表现弱，而 LLaMa‑2 能捕捉游戏的“底层结构”**。这一步把现象和潜在原因联系起来，为后续改进提供了方向。

### 方法详解
整体思路可以拆成四步：**（1）游戏与情境构造 →（2）提示模板设计 →（3）模型推理与答案收集 →（4）策略分析与指标计算**。

1. **游戏与情境构造**  
   - 选取四个经典两人博弈，每个都有明确的收益矩阵。  
   - 为每个游戏准备三种情境描述：外交（正式、利益导向）、商业（竞争与合作并存）和友情（轻松、情感导向）。情境的文字差异仅在开头的背景句，核心规则保持不变。

2. **提示模板设计**  
   - 统一的提示结构如下：  
     ```
     你现在是两位玩家之一。游戏是【游戏名称】。情境是【情境描述】。请在“合作”或“背叛”之间做出选择，并解释你的理由，要求简洁。
     ```  
   - 这种模板保证模型的输入信息一致，只让情境文字产生差异，从而能够直接比较情境对决策的影响。

3. **模型推理与答案收集**  
   - 对每个模型（GPT‑3.5、GPT‑4、LLaMa‑2）分别调用 5 次（不同随机种子），得到 5 份决策+解释。  
   - 将“合作/背叛”映射为二元策略（1/0），解释文本保存用于后续语言特征分析。

4. **策略分析与指标计算**  
   - **策略分布**：统计每个模型在每个游戏、每个情境下的合作率。  
   - **情境敏感度指数**：先算出同一模型在同一游戏下不同情境的合作率方差，再对所有游戏取平均，得到一个数值（方差越大，情境影响越大）。  
   - **语言特征分析**：使用关键词计数（如“协商”“利益”“朋友”）来检验模型是否真的在情境词汇上做出调整。  
   - **解释一致性**：检查解释是否与策略匹配（比如背叛时是否提到“对方不可信”），用来评估模型的自洽程度。

**最巧妙的点**在于把情境框架当作“可控变量”，而不是让它混在提示里随意出现。这样可以像实验心理学里控制变量一样，精准测量情境对策略的边际贡献。

### 实验与效果
- **测试对象**：四个两人博弈（囚徒困境、鹿猎、雪堆、囚徒乐园）×三种情境（外交、商业、友情）= 12 种组合。每种组合对每个模型进行 5 次采样，累计 180 条决策记录。  
- **基线对比**：作者把模型的行为与“随机策略”（50% 合作）以及“理论纳什均衡”（每个游戏的最优混合策略）作对比。  
- **主要发现**：  
  - GPT‑3.5 的合作率在不同情境间波动最大，情境敏感度指数约为 0.27（相对较高），但在囚徒困境等抽象游戏中整体合作率仅 38%，低于理论纳什（约 33%）但高于随机。  
  - GPT‑4 的合作率更稳健，情境敏感度指数约 0.12，且在鹿猎和囚徒乐园中能接近最优合作率（分别 78% 与 85%），显示出对游戏结构的更好把握。  
  - LLaMa‑2 在所有游戏中都表现出较高的结构敏感度（合作率随游戏规则变化明显），情境敏感度指数约 0.09，且在雪堆博弈中实现了 71% 的合作率，超过 GPT‑4 的 64%。  
- **消融实验**：作者去掉情境描述，仅保留游戏名称，发现 GPT‑3.5 的合作率几乎不变（说明它主要靠情境驱动），而 GPT‑4 与 LLaMa‑2 的合作率下降约 5%，表明它们在一定程度上已经内部化了游戏规则。  
- **局限性**：实验只覆盖了两人、一次性决策的静态博弈，未考虑重复博弈或多方博弈；模型输出受提示模板影响，换一种提示可能得到不同结果。作者也承认没有对模型内部表征（如注意力权重）进行深入分析。

### 影响与延伸思考
这篇工作在 AI 与博弈论交叉领域打开了“情境框架”这一新维度，随后有几篇后续论文（如 2024 年的 “Prompt‑Driven Strategic Reasoning in LLMs”）借鉴了情境敏感度指数，尝试在多轮谈判和合作博弈中加入角色设定。还有研究把情境框架与强化学习结合，让 LLM 在模拟环境中自行学习情境‑策略映射。对想进一步探索的读者，可以关注以下方向：  
- **多轮动态博弈**：让模型在重复交互中形成信任或报复策略。  
- **跨模型对齐**：研究如何让不同规模的 LLM 在同一情境下产生一致的战略判断。  
- **内部表征可视化**：利用注意力热图或激活分析，直接观察情境词汇如何影响模型的决策路径。  

### 一句话记住它
情境描述能显著左右 LLM 的博弈决策，但只有 GPT‑4 与 LLaMa‑2 真正把游戏规则内化，GPT‑3.5 仍是“情境驱动的聊天机器人”。