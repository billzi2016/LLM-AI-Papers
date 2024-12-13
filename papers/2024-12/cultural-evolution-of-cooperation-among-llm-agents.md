# Cultural Evolution of Cooperation among LLM Agents

> **Date**：2024-12-13
> **arXiv**：https://arxiv.org/abs/2412.10270

## Abstract

Large language models (LLMs) provide a compelling foundation for building generally-capable AI agents. These agents may soon be deployed at scale in the real world, representing the interests of individual humans (e.g., AI assistants) or groups of humans (e.g., AI-accelerated corporations). At present, relatively little is known about the dynamics of multiple LLM agents interacting over many generations of iterative deployment. In this paper, we examine whether a "society" of LLM agents can learn mutually beneficial social norms in the face of incentives to defect, a distinctive feature of human sociality that is arguably crucial to the success of civilization. In particular, we study the evolution of indirect reciprocity across generations of LLM agents playing a classic iterated Donor Game in which agents can observe the recent behavior of their peers. We find that the evolution of cooperation differs markedly across base models, with societies of Claude 3.5 Sonnet agents achieving significantly higher average scores than Gemini 1.5 Flash, which, in turn, outperforms GPT-4o. Further, Claude 3.5 Sonnet can make use of an additional mechanism for costly punishment to achieve yet higher scores, while Gemini 1.5 Flash and GPT-4o fail to do so. For each model class, we also observe variation in emergent behavior across random seeds, suggesting an understudied sensitive dependence on initial conditions. We suggest that our evaluation regime could inspire an inexpensive and informative new class of LLM benchmarks, focussed on the implications of LLM agent deployment for the cooperative infrastructure of society.

---

# 大型语言模型代理的合作文化进化 论文详细解读

### 背景：这个问题为什么难？

在真实社会里，多个智能体要在资源有限、利益冲突的环境中长期共存，必须形成能够抑制背叛的社会规范。传统的多智能体研究大多基于固定的规则或简化的奖励函数，难以捕捉人类文明依赖的“声誉”与“间接互惠”机制。随着大语言模型（LLM）被用来构建通用型 AI 代理，出现了一个新挑战：这些代理在跨代部署、相互观察对方行为的情境下，是否会自发演化出合作的文化？之前的工作几乎没有系统地考察 LLM 代理的代际交互与文化演化过程，因此需要一种既能模拟人类社会结构，又能利用 LLM 强大推理能力的实验框架。

### 关键概念速览

**间接互惠**：指个体通过帮助他人提升自己的声誉，从而在未来被其他人帮助的概率增加，类似于“做好事会被记在好人榜上”。  
**捐赠者博弈（Donor Game）**：一种两人交互的游戏，捐赠者可以选择付出成本帮助接受者，接受者则可能回报或不回报，常用来研究合作与背叛。  
**声誉分数（Image Score）**：对每个代理的行为历史进行量化的指标，合作会提升分数，背叛会降低分数，类似于社交媒体的点赞/差评系统。  
**代际迭代（Generational Iteration）**：把一代代理的行为记录作为下一代代理的输入，让模型在“文化传承”中学习。  
**成本惩罚（Costly Punishment）**：代理可以主动花费自己的资源去惩罚背叛者，类似于人类社会中的制裁或舆论攻击。  
**随机种子敏感性**：实验结果会随初始化的随机数不同而产生显著差异，暗示系统对初始条件高度依赖。  
**文化进化**：在多代交互中，行为规范通过复制、变异、选择等过程逐步演化，类似于生物进化但作用对象是行为规则。  

### 核心创新点

1. **从单轮博弈到代际文化演化**：以前的 LLM 多智能体实验大多停留在一次性对局，作者把每轮游戏的结果保存为“声誉记录”，并让新一代模型在接收到这些记录后再进行决策。这样把游戏嵌入了跨代的学习循环，使得合作规范可以在多代之间累积。  
2. **引入成本惩罚机制并让模型自行决定是否使用**：传统的捐赠者博弈只考虑合作或背叛，本文额外提供了“惩罚”动作，要求模型在评估自身成本后决定是否对低声誉者进行制裁。实验显示，只有 Claude 3.5 Sonnet 能有效利用这条规则，提升整体得分。  
3. **跨模型比较的系统化基准**：作者把三大主流 LLM（Claude 3.5 Sonnet、Gemini 1.5 Flash、GPT‑4o）放在同一实验平台上，统一观察它们在相同规则下的合作演化曲线，首次提供了“文化进化”视角的模型能力排行榜。  
4. **揭示随机种子对文化走向的强影响**：通过多次随机初始化实验，作者发现同一模型在不同种子下会出现合作主导或背叛主导的两种截然不同的稳态，提示研究者在评估 LLM 代理时必须考虑多次重复实验。  

### 方法详解

整体框架可以分为四个阶段：**初始化 → 交互 → 记录 → 代际迁移**。  

1. **初始化**：为每一代创建 N 个 LLM 代理，每个代理的系统提示中嵌入了“你是一个在社会中行动的 AI，需根据他人过去行为决定是否帮助”。随机种子决定了提示的细微差别和模型内部的采样路径。  

2. **交互（捐赠者博弈）**：在每一轮，系统随机配对两名代理。捐赠者先看到对方的最近 K 次声誉分数（例如过去 5 轮的合作记录），然后在三种动作中选择：**合作**（付出固定成本 C，受益方获得 R），**背叛**（不付出，受益方仍得 R），或**惩罚**（花费 P 使对方声誉下降 Δ）。这一步骤完全由 LLM 的生成式推理完成，模型会在内部“思考”对方的历史并给出理由。  

3. **记录**：每次交互结束后，系统更新双方的声誉分数：合作+1、背叛‑1、被惩罚‑2（具体数值由实验设定）。所有分数以表格形式保存，供下一代读取。  

4. **代际迁移**：当一代完成预设的交互轮数（如 1000 轮）后，系统把该代的声誉表格作为输入提示，生成下一代的系统提示。新一代的模型在同样的交互规则下继续游戏，形成“文化传承”。  

关键模块的细节：

- **提示工程**：作者在系统提示里加入了“你可以查询最近的声誉记录”，相当于给模型提供了一个外部记忆。  
- **成本惩罚决策**：模型在决定是否惩罚时，需要权衡自身的资源消耗与对整体合作氛围的潜在提升，这一步骤在实验中被发现是区分模型表现的关键。  
- **随机种子控制**：每次实验的随机种子影响配对顺序、初始声誉以及模型内部的采样温度，作者通过多次重复来统计分布。  

最巧妙的地方在于把**声誉表格**当作“文化基因”，让模型在没有显式编程的情况下，通过语言理解自然地把这些信息内化为决策依据。这种做法把传统的进化博弈模型与现代 LLM 的生成能力桥接起来，开辟了新的实验范式。

### 实验与效果

- **任务设置**：作者使用了经典的迭代捐赠者博弈，配对次数、成本收益比例等参数均保持一致，以确保不同模型之间的公平比较。  
- **基线对比**：三大模型分别跑了 30 次不同随机种子实验。Claude 3.5 Sonnet 的平均得分约为 0.68（满分 1），显著高于 Gemini 1.5 Flash（≈0.55）和 GPT‑4o（≈0.42）。  
- **惩罚机制效应**：在开启惩罚选项的实验中，Claude 3.5 Sonnet 的得分进一步提升到约 0.74，而 Gemini 与 GPT‑4o 的得分几乎没有变化，说明后两者未能有效利用惩罚。  
- **消融实验**：作者去掉声誉信息后，所有模型的合作率跌至随机水平（≈0.5），证明声誉是驱动合作的关键因素。  
- **随机种子敏感性**：同一模型在不同种子下出现了两类稳态：一种是高合作（>0.7），另一种是低合作（≈0.3），两者之间的转变往往在早期几轮配对中就决定。  
- **局限性**：实验仅在简化的捐赠者博弈上验证，未涉及更复杂的多方协作或资源分配场景；此外，声誉记录的长度和更新规则是人为设定，可能影响结果的外部有效性。作者也承认没有对模型内部的“思考过程”进行可解释性分析。  

### 影响与延伸思考

这篇工作把“文化进化”概念引入 LLM 代理的评估体系，激发了后续研究在更真实的社会模拟环境中测试 LLM 行为的兴趣。随后出现的几篇论文尝试把公开的社交媒体数据当作声誉来源，或在多任务协作游戏中加入经济激励，都是对本工作思路的直接延伸。对想进一步探索的读者，可以关注以下方向：① 将声誉机制与强化学习结合，实现自适应奖励函数；② 在更大规模的多代理系统中研究网络拓扑对合作的影响；③ 开发可解释的“文化基因”可视化工具，帮助理解 LLM 如何内部化社会规范。  

### 一句话记住它

让大语言模型在跨代捐赠者博弈中通过“声誉基因”自行演化合作规则，Claude 3.5 Sonnet 能用惩罚把合作推到最高。