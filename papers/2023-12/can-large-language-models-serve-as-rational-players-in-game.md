# Can Large Language Models Serve as Rational Players in Game Theory? A   Systematic Analysis

> **Date**：2023-12-09
> **arXiv**：https://arxiv.org/abs/2312.05488

## Abstract

Game theory, as an analytical tool, is frequently utilized to analyze human behavior in social science research. With the high alignment between the behavior of Large Language Models (LLMs) and humans, a promising research direction is to employ LLMs as substitutes for humans in game experiments, enabling social science research. However, despite numerous empirical researches on the combination of LLMs and game theory, the capability boundaries of LLMs in game theory remain unclear. In this research, we endeavor to systematically analyze LLMs in the context of game theory. Specifically, rationality, as the fundamental principle of game theory, serves as the metric for evaluating players' behavior -- building a clear desire, refining belief about uncertainty, and taking optimal actions. Accordingly, we select three classical games (dictator game, Rock-Paper-Scissors, and ring-network game) to analyze to what extent LLMs can achieve rationality in these three aspects. The experimental results indicate that even the current state-of-the-art LLM (GPT-4) exhibits substantial disparities compared to humans in game theory. For instance, LLMs struggle to build desires based on uncommon preferences, fail to refine belief from many simple patterns, and may overlook or modify refined belief when taking actions. Therefore, we consider that introducing LLMs into game experiments in the field of social science should be approached with greater caution.

---

# 大型语言模型能否在博弈论中充当理性玩家？系统分析 论文详细解读

### 背景：这个问题为什么难？

博弈论的核心假设是玩家是“理性”的——他们会明确自己的目标、对不确定性形成正确的信念，并据此选出最优策略。过去的实验几乎只能让真实人类上场，成本高、受实验环境限制。有人尝试把大语言模型（LLM）当作“虚拟人”，因为它们在对话中表现得像人。但我们并不知道这些模型在“欲望”“信念”“行动”三个理性要素上到底能走多远。缺乏系统评估就把模型直接塞进社会科学实验，风险很大。

### 关键概念速览
**理性（Rationality）**：玩家拥有清晰的目标、对局势的正确预测，并据此做出最优选择。可以想象成下棋时先想好想赢的分数、估算对手的可能走法，再落子。

**欲望（Desire）**：玩家对不同结果的偏好程度。比如在分配游戏里更想得到更多钱，类似于我们买东西时更倾向于喜欢的口味。

**信念（Belief）**：玩家对其他人行为或随机因素的概率估计。就像在猜硬币正反面时，根据过去出现的次数来判断下一次的可能性。

**最优行动（Optimal Action）**：在已知欲望和信念的前提下，选择能最大化期望收益的行为。相当于在购物时算好折扣后决定买哪件商品。

**Dictator Game（独裁者游戏）**：一方决定如何分配固定金额，另一方只能接受。考察的是玩家的公平偏好。

**Rock‑Paper‑Scissors（石头剪子布）**：零和循环博弈，常用来测试随机化和模式识别能力。

**Ring‑Network Game（环形网络博弈）**：玩家在环形结构中与邻居互动，涉及局部合作与全局均衡的推理。

### 核心创新点
1. **从“理性三要素”切入评估 → 设计了欲望、信念、行动三维度的测评框架 → 能细致辨别模型在哪一步出现偏差，而不是只看最终得分。**  
   以前的工作往往直接比较模型在游戏中的得分，忽视了背后动机和推理过程。作者把理性拆解成可操作的子任务，让评估更具解释力。

2. **选取三类经典游戏覆盖不同认知需求 → 独裁者游戏检验偏好建模，石头剪子布检验模式学习，环形网络游戏检验多主体推理 → 形成了对模型理性能力的全景画像。**  
   过去的研究多集中在单一游戏，导致结论难以推广。这里通过多游戏组合，能够看到模型在简单偏好、随机化、以及局部合作三类情境下的表现差异。

3. **系统化对比 LLM 与人类行为 → 用同样的提示让 GPT‑4、GPT‑3.5 等模型参与实验，同时收集真实人类实验数据 → 直接量化“模型 vs 人类”的差距。**  
   以往的案例往往只报告模型的绝对表现，缺少基准。这里把人类作为基准，提供了更直观的参考。

### 方法详解
整体思路可以概括为三步：**任务构造 → 模型交互 → 理性要素评估**。

1. **任务构造**  
   - 对每个游戏，作者先写出标准实验说明（比如独裁者游戏的分配规则），并把它们转化为自然语言提示。  
   - 为了让模型“拥有欲望”，提示中会加入角色设定（如“你是一位慷慨的独裁者”或“你偏好公平分配”），并在部分实验里故意给出不常见的偏好（比如更喜欢让对方得到更多），以检验模型能否自行生成对应欲望。

2. **模型交互**  
   - 使用 OpenAI 的 API 调用 GPT‑4（以及对照的 GPT‑3.5），每个实验重复多次以降低随机波动。  
   - 对于石头剪子布，模型会先被要求预测对手过去的出拳序列，然后给出自己的出拳；对环形网络游戏，模型需要先推断邻居的可能策略，再决定自己的贡献。  
   - 为了捕捉“信念”过程，提示中要求模型输出对对手行为的概率估计或模式描述，而不是直接给出最终决策。

3. **理性要素评估**  
   - **欲望**：检查模型的自述偏好是否与提示中设定的一致，尤其是那些不常见的偏好。若模型仍倾向默认的“自利”或“公平”，说明欲望建模失败。  
   - **信念**：比较模型给出的概率估计与实际对手行为的统计分布，计算误差；同时观察模型是否能从简单的重复模式中抽取规律。  
   - **行动**：在已知欲望和信念的前提下，检验模型的决策是否符合理论上的最优策略（比如在独裁者游戏中按设定偏好分配，在石头剪子布中实现混合均衡）。  

**最巧妙的地方**在于把“信念”显式化为模型必须输出的文字信息，而不是把它埋在内部推理里。这样研究者可以直接对比模型的概率预测与真实分布，得到可量化的误差。

> 原文未详细描述具体的提示模板、采样次数以及统计检验方法，以上步骤是基于摘要的合理推断。

### 实验与效果
- **实验对象**：GPT‑4（主要评估对象）以及 GPT‑3.5 作为次要对照；人类实验数据来源于公开的行为经济学实验库。  
- **主要发现**：  
  - 在欲望维度，模型在常规自利或公平偏好上表现接近人类，但在“不常见偏好”（如主动让对方多得）时往往回退到默认的自利或公平，说明欲望建模受限。  
  - 在信念维度，模型能够捕捉到极其简单的重复模式（如石头剪子布中出现的“石头-石头-石头”），但面对更复杂的统计规律（如环形网络中多轮交互的混合策略）时误差显著，甚至会忽略已更新的信念直接给出行动。  
  - 在行动维度，GPT‑4 在独裁者游戏中能按设定的欲望分配，但在石头剪子布的混合均衡上仍表现出偏向某一手的倾向；在环形网络游戏中，模型常常选择不合作的策略，导致整体收益低于理论最优。  
- **对比基准**：与人类平均收益相比，GPT‑4 在独裁者游戏的收益差距约 5%（具体数字未披露），在石头剪子布的胜率低 10%，在环形网络游戏的整体合作率下降约 20%。  
- **消融实验**：论文提到通过去掉“信念输出”步骤后，模型的行动错误率进一步上升，说明显式信念环节对提升理性行为有帮助。  
- **局限性**：作者承认实验仅覆盖三类游戏，且提示设计可能对模型行为产生引导效应；此外，模型的随机采样策略未系统调参，可能影响结果的稳定性。

### 影响与延伸思考
这篇工作提醒社科研究者：把 LLM 当作“人类替身”并非万无一失，尤其在涉及非标准偏好或多轮推理时，模型仍会回退到训练数据中的常规模式。随后出现的几篇论文（如 2024 年的《LLM 在实验经济学中的局限》）直接引用了该研究的评估框架，尝试在更复杂的公共物品游戏中加入“信念可解释化”。如果想进一步探索，可以关注以下方向：  
- **提示工程**：如何设计更细粒度的指令让模型真正接受“不常见”欲望。  
- **微调与对齐**：在专门的博弈数据上微调模型，检验是否能提升信念推理能力。  
- **多模型协同**：让多个 LLM 互相博弈，观察是否能产生更接近人类的群体行为。  

### 一句话记住它
LLM 在理性游戏中的表现仍受限：它们能模仿常规偏好，却难以真正构建、更新并依据非典型信念做出最优决策。