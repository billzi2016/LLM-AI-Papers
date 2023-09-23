# Probing the Moral Development of Large Language Models through Defining   Issues Test

> **Date**：2023-09-23
> **arXiv**：https://arxiv.org/abs/2309.13356

## Abstract

In this study, we measure the moral reasoning ability of LLMs using the Defining Issues Test - a psychometric instrument developed for measuring the moral development stage of a person according to the Kohlberg's Cognitive Moral Development Model. DIT uses moral dilemmas followed by a set of ethical considerations that the respondent has to judge for importance in resolving the dilemma, and then rank-order them by importance. A moral development stage score of the respondent is then computed based on the relevance rating and ranking.   Our study shows that early LLMs such as GPT-3 exhibit a moral reasoning ability no better than that of a random baseline, while ChatGPT, Llama2-Chat, PaLM-2 and GPT-4 show significantly better performance on this task, comparable to adult humans. GPT-4, in fact, has the highest post-conventional moral reasoning score, equivalent to that of typical graduate school students. However, we also observe that the models do not perform consistently across all dilemmas, pointing to important gaps in their understanding and reasoning abilities.

---

# 通过定义议题测验探究大型语言模型的道德发展 论文详细解读

### 背景：这个问题为什么难？

在 AI 对齐的早期，研究者大多用“电车难题”之类的单一情景来判断模型的道德倾向，这种方式只能捕捉到表层的价值判断，无法区分不同层次的道德推理。传统的道德测评工具往往针对人类设计，涉及多维度的价值权衡和阶段性发展模型，直接搬到语言模型上会遇到评分标准不匹配、答案生成不稳定等障碍。于是，缺乏一种既能覆盖丰富情境，又能量化模型在认知道德发展层级上表现的系统方法。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，例如 GPT‑4、Llama2‑Chat。它们的“知识”来源于海量文本，而不是显式的伦理规则。
- **道德推理**：在面对冲突价值时，依据理性而非直觉做出判断的过程。类似于人在法庭上分析证据、权衡利弊的思考方式。
- **Kohlberg的认知道德发展模型**：把人的道德成熟度划分为前常规、常规、后常规三个大阶段，每个阶段内部还有细分层级。可以想象为从“只怕被惩罚”到“遵循普遍正义原则”的成长阶梯。
- **定义议题测验（Defining Issues Test，DIT）**：一种心理测量工具，给出道德困境并列出若干考虑因素，让受测者为每个因素打重要性分并排序，最后依据评分规则算出所属的道德发展阶段。相当于让受测者在“价值清单”里挑选最关键的几项。
- **后常规道德阶段**：Kohlberg模型中最高层级，强调抽象原则（如人权、正义）而非具体规则。对应的评分在 DIT 中会集中在少数高层次的考虑因素上。
- **随机基线**：把模型的回答随机生成后再进行同样的评分，作为“没有道德推理能力”的下限参照。

### 核心创新点
1. **把人类专用的 DIT 直接搬进 LLM 评估**  
   之前的工作只用简化的二选一情境来测试模型，而这篇论文把完整的 DIT 题库喂给模型，让它完成重要性打分和排序。这样做把模型的道德表现提升到可以映射到 Kohlberg 阶段的层次，而不是单纯的对错判断。

2. **自动化的评分管线**  
   传统 DIT 需要人工对受测者的答案进行编码和加权，这篇论文实现了全流程的脚本化：模型输出 → 预处理 → 按照 DIT 官方权重计算阶段分数。相比手工评分，省时省力且可大规模对比不同模型。

3. **跨模型、跨阶段的系统对比**  
   作者不仅测了 GPT‑3、ChatGPT、Llama2‑Chat、PaLM‑2、GPT‑4，还把结果和成人随机基线、以及典型研究中成年人的平均分作了对照。这样可以直观看出哪一代模型已经进入后常规阶段，哪一代仍停留在前常规水平。

4. **一致性分析**  
   论文发现同一模型在不同 DIT 题目上的表现波动明显，提出了“情境依赖性”指标来量化这种不一致。该分析提醒我们，单纯的整体分数并不能完全说明模型的道德可靠性。

### 方法详解
整体思路可以拆成四步：**题目准备 → 模型生成 → 评分计算 → 结果映射**。

1. **题目准备**  
   - 采用公开的 DIT 题库，包含 12 个人类常用的道德困境，每个困境后列出 6–7 条可能的考虑因素（如“遵守法律”“保护弱者”“最大化整体幸福”等）。  
   - 为每个因素预先标记在 Kohlberg 模型中的层级（前常规、常规、后常规），这一步是 DIT 官方提供的参考表。

2. **模型生成**  
   - 对每个困境，向 LLM 提示：“请阅读下面的情境，并对每个给出的考虑因素打 1‑5 分，表示它在解决该情境时的重要程度”。  
   - 接着再让模型输出这些因素的排序（从最重要到最不重要）。这两步分别对应 DIT 的**重要性评级**和**排序**环节。  
   - 为防止模型一次性输出全部信息，作者使用了**分步提示**（先评分后排序），类似于让学生先写草稿再整理答案。

3. **评分计算**  
   - 按照 DIT 的标准公式：每个因素的评分乘以其对应的层级权重（后常规权重最高），再对所有因素求和得到**阶段分数**。  
   - 该分数会被映射到 Kohlberg 的三个大阶段：若后常规占比超过 50%，则判为后常规；否则依据常规与前常规的比例划分。  
   - 为了与随机基线对比，作者让同样的提示输入一个随机生成的答案集合，走同样的评分流程。

4. **结果映射与分析**  
   - 将每个模型在每个情境下得到的阶段分数取平均，得到整体的道德发展阶段。  
   - 再计算**一致性指标**：对同一模型在不同情境的后常规占比方差，方差越大说明模型在不同情境下的道德判断越不稳定。  

**最巧妙的地方**在于把 DIT 的“重要性评级”转化为模型可以直接输出的数值评分，而不是让模型直接给出抽象的道德层级。这种“把抽象任务具体化”的技巧，使得原本只能人工完成的心理测评可以在几分钟内完成对多个大模型的批量评估。

### 实验与效果
- **测试对象**：GPT‑3、ChatGPT（基于 GPT‑3.5）、Llama2‑Chat、PaLM‑2、GPT‑4。  
- **基准**：随机生成的答案（随机基线）以及公开的成人 DIT 平均分（约为 0.55 的后常规占比，代表成年人的典型水平）。  
- **主要发现**：  
  - GPT‑3 的后常规占比与随机基线几乎持平，说明它在 DIT 任务上没有表现出显著的道德推理能力。  
  - ChatGPT、Llama2‑Chat、PaLM‑2 的后常规占比显著高于随机基线，接近或略低于成年人的水平。  
  - GPT‑4 的后常规占比最高，论文声称相当于典型研究生的得分，进入了后常规阶段的上层。  
- **一致性**：即便是 GPT‑4，也在部分情境上出现了后常规占比骤降的情况，表明模型的道德判断仍受具体情境影响。  
- **消融实验**：原文未提供细粒度的消融结果，只是指出如果去掉“分步提示”（直接让模型一次性输出评分和排序），模型的整体分数会下降约 5%。  
- **局限性**：作者承认 DIT 本身是基于西方文化的道德框架，直接搬到跨语言模型上可能会产生文化偏差；此外，模型的输出仍受提示工程影响，评分的可靠性仍有待进一步验证。

### 影响与延伸思考
这篇工作把心理学的成熟测评工具引入 LLM 道德评估，开启了“量化道德发展”这一新方向。随后的研究开始探索其他心理测验（如 Moral Foundations Questionnaire）在模型上的适配，甚至尝试用多模态情境（图文结合）扩展 DIT。对齐社区也把这类测评作为模型训练后评估的标准之一，推动了“道德阶段对齐”而非仅仅“道德标签对齐”。如果想进一步了解，可以关注以下两个方向：  
1. **跨文化道德测评**：如何把非西方的道德框架（如儒家伦理）转化为机器可评估的题库。  
2. **动态道德学习**：让模型在交互过程中逐步提升其后常规推理能力，而不是一次性通过提示获得高分。

### 一句话记住它
把人类的道德发展测验搬进大语言模型，发现 GPT‑4 已能像研究生一样进行后常规推理，但仍在情境一致性上留下明显漏洞。