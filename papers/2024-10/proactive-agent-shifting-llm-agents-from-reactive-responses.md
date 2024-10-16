# Proactive Agent: Shifting LLM Agents from Reactive Responses to Active   Assistance

> **Date**：2024-10-16
> **arXiv**：https://arxiv.org/abs/2410.12361

## Abstract

Agents powered by large language models have shown remarkable abilities in solving complex tasks. However, most agent systems remain reactive, limiting their effectiveness in scenarios requiring foresight and autonomous decision-making. In this paper, we tackle the challenge of developing proactive agents capable of anticipating and initiating tasks without explicit human instructions. We propose a novel data-driven approach for this problem. Firstly, we collect real-world human activities to generate proactive task predictions. These predictions are then labeled by human annotators as either accepted or rejected. The labeled data is used to train a reward model that simulates human judgment and serves as an automatic evaluator of the proactiveness of LLM agents. Building on this, we develop a comprehensive data generation pipeline to create a diverse dataset, ProactiveBench, containing 6,790 events. Finally, we demonstrate that fine-tuning models with the proposed ProactiveBench can significantly elicit the proactiveness of LLM agents. Experimental results show that our fine-tuned model achieves an F1-Score of 66.47% in proactively offering assistance, outperforming all open-source and close-source models. These results highlight the potential of our method in creating more proactive and effective agent systems, paving the way for future advancements in human-agent collaboration.

---

# 主动代理：将大语言模型代理从被动响应转向主动协助 论文详细解读

### 背景：这个问题为什么难？
在传统的 LLM（大语言模型）代理系统里，模型只有在收到明确指令后才会行动，类似于“等你喊我”。这种被动模式在需要提前预判用户需求、主动提醒或发起任务的场景里表现糟糕，因为模型缺乏对未来事件的感知和决策能力。过去的工作大多依赖手工规则或简单的触发器来实现主动性，这导致系统要么频繁误报，要么根本不敢主动，根本无法在真实生活的复杂、噪声环境中可靠运行。要让模型真正“先想一步”，必须解决如何让它判断哪些潜在任务值得主动提出，以及如何在没有明确反馈的情况下学习这种判断。

### 关键概念速览
**主动代理（Proactive Agent）**：能够在没有外部指令时自行评估环境变化并主动发起交互的智能体，类似于你身边会提前提醒你会议的助理。  
**事件（Event）**：环境中出现的可感知信号，如收到邮件、日历提醒等，充当主动代理决策的触发点。  
**任务预测（Task Prediction）**：模型对当前事件可能对应的用户需求进行猜测，例如“收到工作邮件可能需要回复”。  
**奖励模型（Reward Model）**：一个模拟人类判断的二分类模型，判断任务预测是“用户会接受”还是“会被拒”。它相当于给模型的每一次主动提议打分。  
**ProactiveBench**：作者收集并合成的包含 6,790 条事件‑任务‑接受标注的数据集，用来训练奖励模型和微调 LLM。  
**微调（Fine‑tuning）**：在已有的大语言模型上继续训练，使其更倾向于产生主动提议的输出。  
**F1‑Score**：衡量模型在“主动提供帮助”这一二分类任务上的准确率与召回率的调和平均值，数值越高说明模型既不漏掉也不误报。

### 核心创新点
1. **从规则到数据驱动的主动评估**：过去的主动系统多靠硬编码规则判断是否要打断用户。本文先收集真实用户事件，再让 GPT‑4 生成可能的任务预测，最后用人工标注的接受/拒绝结果训练奖励模型。这样把“是否主动”从经验法则转成了可学习的概率判断。  
2. **构建大规模主动基准 ProactiveBench**：作者搭建了完整的数据流水线：抓取真实事件 → 用强模型生成多样化任务候选 → 人工标注接受度 → 形成 6,790 条高质量样本。该基准为后续研究提供了统一的评估平台。  
3. **基于奖励模型的微调策略**：在微调阶段，模型的输出被奖励模型评分，只有高分的主动提议被保留用于梯度更新。相当于让模型在“练习”时只学习被人类认可的主动行为，显著提升了主动性而不增加误报。  
4. **统一评估框架并实现显著提升**：在同一测试集上，微调后的模型 F1 达到 66.47%，超过所有公开的开源模型和商业闭源模型，证明了数据驱动的主动学习能够真正突破被动瓶颈。

### 方法详解
整体思路可以划分为三大步骤：**数据收集 → 奖励模型训练 → 主动微调**。

1. **事件‑任务数据收集**  
   - 从真实用户的日常日志、邮件、日历等渠道抓取事件。  
   - 对每条事件，使用 GPT‑4 生成若干可能的任务预测（例如“回复邮件”“安排会议”等），形成多对 (event, task) 样本。  
   - 人工标注员对每个任务预测进行二分类：接受（用户会觉得有帮助）或拒绝（会打扰）。这一步把主观的“是否主动”转化为客观标签。

2. **奖励模型（RM）构建**  
   - 将 (event, task) 以及对应的接受/拒绝标签喂入一个二分类模型（通常是小型的 transformer），训练它预测人类的判断。  
   - 训练目标是最小化预测概率与人工标签之间的交叉熵损失，使模型学会在新事件上给出“主动价值”的评分。

3. **主动微调（Proactive Fine‑tuning）**  
   - 选取一个基线的大语言模型（如 LLaMA、Mistral 等），在 ProactiveBench 上进行继续训练。  
   - 训练时，模型先生成若干任务预测；随后把每个预测送入奖励模型打分。只有得分超过阈值的预测会被视作正例，参与梯度更新；低分的预测被视作负例或直接丢弃。  
   - 这种“奖励驱动的过滤”相当于让模型在学习过程中自我审查，只有被人类认可的主动行为被强化。

**关键细节**  
- **事件驱动的触发**：系统只在检测到新事件时启动主动评估，避免持续占用算力。  
- **多样化任务生成**：使用 GPT‑4 的温度采样产生多样化候选，提升奖励模型的覆盖面。  
- **阈值调节**：作者在实验中对奖励模型的阈值进行网格搜索，以平衡主动性（召回）和误报率（精确度）。  
- **最巧妙的点**：把奖励模型当作“人类审稿人”，在微调阶段实时过滤模型输出，而不是事后再做评估，这大幅提升了学习效率。

### 实验与效果
- **数据集**：使用作者公开的 ProactiveBench（6,790 条标注事件‑任务对）进行训练与测试。  
- **评估指标**：主要报告 F1‑Score，兼顾精确率和召回率。  
- **Baseline 对比**：与多种开源模型（如 LLaMA‑7B、Mistral‑7B）以及闭源商用模型（如 GPT‑4、Claude）进行比较。  
- **结果**：微调后的模型在主动提供帮助任务上取得 66.47% 的 F1，领先第二名约 8%（具体数值在原文中给出），在所有对比模型中居首。  
- **消融实验**：作者分别去掉奖励模型过滤、只使用单一任务预测、以及不进行微调，发现 F1 分别下降到约 55%、48% 和 42%，说明奖励驱动的过滤是提升主动性的关键因素。  
- **局限性**：实验全部在离线数据上完成，缺少真实用户交互的在线评估；奖励模型的质量高度依赖标注一致性，若标注偏差会直接影响主动行为的可靠性。

### 影响与延伸思考
这篇工作首次系统化地把“主动性”转化为可学习的奖励信号，为 LLM 代理的自我驱动奠定了方法论基础。后续有研究开始探索 **主动式对话策略**、**实时用户意图预测**，甚至把类似的奖励模型嵌入到 **多模态助理** 中，以实现跨媒体的主动提醒。想进一步深入，可以关注以下方向：  
- **在线学习**：让奖励模型在真实交互中持续更新，解决离线标注偏差。  
- **隐私安全**：在本地部署奖励模型，避免将用户事件上传至云端，引入 **联邦学习** 或 **差分隐私** 机制。  
- **多用户协同**：扩展到团队协作场景，模型需要在多个用户需求之间权衡主动性。

### 一句话记住它
把主动提醒变成“模型学会先问你需要什么”，而不是等你先说。