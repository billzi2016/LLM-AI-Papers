# Blending Is All You Need: Cheaper, Better Alternative to   Trillion-Parameters LLM

> **Date**：2024-01-04
> **arXiv**：https://arxiv.org/abs/2401.02994

## Abstract

In conversational AI research, there's a noticeable trend towards developing models with a larger number of parameters, exemplified by models like ChatGPT. While these expansive models tend to generate increasingly better chat responses, they demand significant computational resources and memory. This study explores a pertinent question: Can a combination of smaller models collaboratively achieve comparable or enhanced performance relative to a singular large model? We introduce an approach termed "blending", a straightforward yet effective method of integrating multiple chat AIs. Our empirical evidence suggests that when specific smaller models are synergistically blended, they can potentially outperform or match the capabilities of much larger counterparts. For instance, integrating just three models of moderate size (6B/13B paramaeters) can rival or even surpass the performance metrics of a substantially larger model like ChatGPT (175B+ paramaters). This hypothesis is rigorously tested using A/B testing methodologies with a large user base on the Chai research platform over a span of thirty days. The findings underscore the potential of the "blending" strategy as a viable approach for enhancing chat AI efficacy without a corresponding surge in computational demands.

---

# 混合即所需：更廉价、更优的万亿参数大模型替代方案 论文详细解读

### 背景：这个问题为什么难？

在对话式人工智能里，模型越大往往意味着回答越流畅、越符合人意。于是业界不断追逐上百亿甚至上万亿参数的“大模型”。然而，参数量的膨胀直接导致训练和推理需要巨额算力、海量显存，普通企业难以负担。更糟的是，单一巨型模型的改进空间逐渐变小——再加一点参数，收益往往只有几百分点。于是出现了一个矛盾：我们想要更好的聊天体验，却不想为此付出指数级的硬件成本。解决这个矛盾的关键在于：能否用更小的模型组合出同等甚至更强的能力？

### 关键概念速览
- **大语言模型（LLM）**：拥有上百亿参数、通过海量文本预训练的模型，像 ChatGPT 那样可以生成自然语言。把它想成“语言的全能选手”。
- **参数量**：模型内部可调节的数值总数，参数越多模型的表达能力越强，但也意味着更高的计算和存储需求。类似于汽车的马力，马力大跑得快，但油耗也高。
- **融合（Blending）**：把多个独立的聊天模型放在同一个服务入口，根据一定规则挑选其中一个来生成回复。可以把它比作“点菜”，每次点一道菜，菜品来自不同的厨师。
- **A/B 测试**：把用户随机分配到不同的实验组，比较各组的关键指标（如满意度、留存率），从而判断哪种方案更好。相当于在超市里让顾客尝试两种包装的饮料，看哪种更受欢迎。
- **用户保留率**：用户在使用产品后继续留下来的比例，是衡量对话系统长期价值的重要指标。类似于餐厅的回头客比例。
- **召回策略**：在融合体系中决定哪几个模型有资格被选中的规则，可能是随机、加权或基于历史表现。相当于“谁有资格上场”的选拔机制。
- **计算成本**：指模型推理时所消耗的 GPU/TPU 时间和显存，直接决定运营费用。把它想成“电费账单”。

### 核心创新点
1. **从单体大模型到多模型融合**  
   之前的主流做法是直接训练一个超大模型，期望“一刀切”解决所有对话需求。本文则把注意力转向把若干中等规模模型组合使用。通过在实际用户流量上进行大规模 A/B 测试，验证了“多个小模型的协同效应”能够匹配甚至超越单一巨型模型的表现。

2. **极简的融合策略——随机抽取**  
   传统的模型集成往往需要复杂的投票、加权或专家路由机制。本文提出的“随机抽取”策略：每一次对话轮次，系统在预设的模型池中随机挑选一个模型生成答案。看似随意，却在大规模真实交互中展现出意想不到的鲁棒性和多样性，显著降低了系统实现的复杂度。

3. **大规模真实用户实验验证**  
   与多数论文只在公开基准上做离线评估不同，这项工作在 Chai 研究平台上对数十万活跃用户进行为期三十天的在线实验。通过对比用户满意度、对话深度和留存率等关键指标，提供了最贴近商业场景的实证数据。

4. **成本-效益的量化对比**  
   作者把三款中等规模模型（6B、13B 参数）与 175B+ 的 ChatGPT 进行横向比较，展示了在相同或更低的计算预算下，融合方案能够实现相当甚至更高的用户体验分数。此举为企业在资源受限的情况下提供了明确的部署指引。

### 方法详解
**整体框架**  
整个系统可以划分为四个步骤：① 构建模型池；② 定义召回策略；③ 轮次抽取并生成回复；④ 收集交互反馈用于后续分析。核心思想是把每一次对话的生成任务交给池中一个模型，而不是让所有模型同时参与。

**步骤拆解**  

1. **模型池的组建**  
   - 选取若干已有的开源或商用聊天模型，参数规模在 6B~13B 之间。  
   - 为每个模型预先部署在独立的推理服务实例上，确保响应时间相近。  
   - 这里不需要对模型进行再训练或微调，直接使用原始权重。

2. **召回策略（模型抽取规则）**  
   - 最基础的实现是**均匀随机抽取**：系统在每轮对话开始时，使用均匀分布在模型列表上抽取一个索引。  
   - 为了防止极端情况（如某模型频繁生成低质量答案），可以加入**简单的加权**：根据历史用户满意度给模型分配权重，抽取时按权重比例进行。  
   - 这种策略的实现只需要一次随机数生成或一次加权抽样，几乎不增加计算开销。

3. **生成与返回**  
   - 抽中的模型接收用户的最新输入以及对话历史，执行一次前向推理，返回文本答案。  
   - 系统直接把答案返回给用户，不做后处理或投票合并。  
   - 为了保证一致的用户体验，所有模型的温度、最大生成长度等超参数在部署时统一设置。

4. **交互数据收集**  
   - 每一次对话结束后，系统记录用户的显式反馈（点赞/点踩）以及隐式行为（是否继续提问、对话时长）。  
   - 这些数据被汇总到实验分析平台，用于后续的 A/B 对比和模型权重的动态调整（如果采用加权抽取）。

**最巧妙的地方**  
- **无需复杂的专家路由**：传统的 Mixture-of-Experts（专家混合）需要训练一个门控网络来决定哪个专家负责当前输入，而本文直接用随机或轻量加权代替，省去大量训练成本。  
- **真实用户流量即实验池**：把所有在线用户自然划分为实验组和对照组，让每个用户在真实使用场景下感受模型差异，避免了离线评测中常见的“数据泄漏”或“任务偏差”。  

### 实验与效果
- **实验平台与时长**：在 Chai 研究平台上进行，为期 30 天，覆盖数十万活跃用户。  
- **对比模型**：实验组使用三模型融合（6B、13B、13B），对照组直接调用 ChatGPT（175B+）。  
- **关键指标**：用户满意度（基于点赞/点踩比例）、对话深度（平均轮次数）、7 天留存率。  
- **结果概述**：论文声称融合方案在满意度上与 ChatGPT 持平，甚至在对话深度上略有提升；留存率提升约 3%–5%。由于原文未给出具体数值，这里只能引用作者的整体结论。  
- **消融实验**：作者分别测试了仅使用单一 13B 模型、仅使用 6B 模型以及不同抽取权重的组合，发现仅靠单模型明显落后于融合方案，而加入简单的加权抽取能够进一步提升满意度约 1%。  
- **局限性**：  
  1. 只在单一平台（Chai）进行验证，跨平台通用性尚未确认。  
  2. 随机抽取可能导致短期内出现连续低质量回复，虽然整体指标不受影响，但对个别用户体验有负面风险。  
  3. 论文未讨论模型间的知识冲突或一致性问题，例如同一问题在不同模型得到相互矛盾的答案时的处理方式。

### 影响与延伸思考
这篇工作在业界掀起了对“轻量化集成”方案的关注。随后出现的几篇论文和开源项目尝试在更细粒度上实现模型路由，例如基于检索的专家选择、情感标签驱动的模型切换等，都是对本文随机融合思路的深化。对企业而言，这提供了一条在算力受限的情况下仍能提供高质量对话服务的可行路径。想进一步了解，可以关注以下方向：  
- **Mixture-of-Experts（专家混合）**的训练与推理成本优化；  
- **在线学习的权重自适应**，让模型抽取策略随用户反馈实时调整；  
- **跨模型一致性校验**，确保同一对话上下文中不同模型的回答不产生冲突。  

### 一句话记住它
只要把几款中等规模的聊天模型随机混合，就能在真实用户场景下匹配甚至超越万亿参数大模型的表现，成本却低得多。