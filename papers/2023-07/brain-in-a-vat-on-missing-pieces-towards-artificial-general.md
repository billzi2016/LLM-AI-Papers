# Brain in a Vat: On Missing Pieces Towards Artificial General Intelligence in Large Language Models

> **Date**：2023-07-07
> **arXiv**：https://arxiv.org/abs/2307.03762

## Abstract

In this perspective paper, we first comprehensively review existing evaluations of Large Language Models (LLMs) using both standardized tests and ability-oriented benchmarks. We pinpoint several problems with current evaluation methods that tend to overstate the capabilities of LLMs. We then articulate what artificial general intelligence should encompass beyond the capabilities of LLMs. We propose four characteristics of generally intelligent agents: 1) they can perform unlimited tasks; 2) they can generate new tasks within a context; 3) they operate based on a value system that underpins task generation; and 4) they have a world model reflecting reality, which shapes their interaction with the world. Building on this viewpoint, we highlight the missing pieces in artificial general intelligence, that is, the unity of knowing and acting. We argue that active engagement with objects in the real world delivers more robust signals for forming conceptual representations. Additionally, knowledge acquisition isn't solely reliant on passive input but requires repeated trials and errors. We conclude by outlining promising future research directions in the field of artificial general intelligence.

---

# 大脑在罐中：大型语言模型通往通用人工智能的缺失环节 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大语言模型）横空出世之前，AI 评估大多停留在标准化考试或单项能力测评上，这些测试往往只检验模型的记忆和模式匹配能力。随着模型规模膨胀，评测结果看起来异常亮眼，却掩盖了模型在真实世界中“懂得做事”的缺口。根本问题在于：现有评估没有考察模型的主动交互、价值取向和对外部世界的真实认知，这让人难以判断它们是否真的具备通用智能。

### 关键概念速览
**LLM（大型语言模型）**：通过海量文本训练的生成式模型，擅长预测下一个词，就像把大量书籍背下来后随口复述。  
**标准化测试**：固定题库、统一评分的考试方式，例如 SAT、GRE，用来衡量模型在特定知识点上的表现。  
**能力导向基准**（ability‑oriented benchmark）：专门设计的任务集合，旨在评估模型的推理、常识或编程等具体能力。  
**通用人工智能（AGI）**：能够在任意环境中学习、规划并执行任务的智能体，目标是像人一样灵活而不是只会“答题”。  
**价值系统**：决定智能体在多任务情境下优先级的内部准则，类似于人类的道德观或个人目标。  
**世界模型**：智能体对外部现实的内部表征，帮助它预测行动后果，就像我们的大脑里对物体运动的“内在模拟”。  
**主动交互**：智能体通过动手、实验或与环境持续反馈的方式获取信息，而不是被动接受文字输入。  

### 核心创新点
1. **评估盲点的系统梳理 → 对现有 LLM 评测进行全景回顾 → 揭示了“只会答题”与“会做事”之间的鸿沟**。作者把标准化测试和能力基准并列，对比它们在任务广度、任务生成和价值驱动三方面的缺失，明确指出当前评测体系高估了模型的通用性。  
2. **提出四大通用智能特征 → 将“无限任务执行”“上下文任务生成”“价值驱动”“真实世界模型”四个维度具体化 → 为后续 AGI 研究提供了可操作的评估框架**。这些特征不是抽象口号，而是从认知科学和机器人学中抽取的关键能力。  
3. **强调“知行合一”是缺失的核心 → 论证主动与真实环境的交互能够提供更稳健的概念信号 → 为未来把 LLM 与感知/行动系统结合指明方向**。作者用“脑在罐中”的哲学比喻说明，仅靠文字输入的“脑”无法形成完整的世界模型。  
4. **列出未来研究路线 → 从主动学习、价值学习到跨模态世界建模 → 为学术和工业界提供了明确的技术路线图**。这一步把抽象的缺口转化为可落地的研究议题。

### 方法详解
整体思路可以看作三层金字塔：**评估审视 → 能力定义 → 研究路线**。作者先把所有公开的 LLM 评测收集起来，按“任务覆盖度”“任务生成能力”“价值导向”“世界模型”四个维度打分，形成一张矩阵图。接着，以这张矩阵为基准，提出四条通用智能特征，每条特征对应一组具体的能力要求。例如，“无限任务执行”要求模型能够在同一上下文中连续切换不同领域的任务，而不是一次性完成单一指令。  

在此基础上，作者用“知行合一”作为核心缺口的概念框架：  
1. **被动输入**（只读文本） → **概念形成**（统计关联） → **局限**：缺少对因果关系的真实验证。  
2. **主动交互**（与物体、环境的反复试错） → **概念强化**（通过行动产生的误差信号） → **优势**：形成更稳固、可迁移的世界模型。  

为了让读者直观感受，作者把这一过程比作小孩学习走路：只看别人的视频只能学到“走路的姿势”，真正会走是要自己跌倒、站起、感受地面的反馈。文中没有提供具体算法，因为它是一篇观点文章，但通过类比和结构化的概念拆解，读者可以清晰看到作者的逻辑链条：评测 → 缺口 → 需求 → 研究方向。

最巧妙的地方在于把“价值系统”直接嵌入任务生成的框架，而不是把价值当作后置的安全约束。作者认为，只有当模型在生成新任务时已经内化了价值准则，才可能避免“无限任务”带来的失控风险。

### 实验与效果
这篇工作是 **perspective**（观点）论文，没有新实验数据。作者仅列举了已有的公开评测结果，指出它们在四个维度上的得分普遍偏低。文中提到的对比主要是 **LLM 在标准化测试上往往接近或超过人类平均水平**，但在 **跨任务连续推理**、**价值一致性**、**真实世界交互** 上几乎没有报告。作者也承认，缺少统一的跨维度评测基准是当前研究的瓶颈。

### 影响与延伸思考
自从这篇文章出现后，业界对 **“主动学习+语言模型”** 的兴趣明显上升。随后出现的几篇工作（如 **ReAct**、**SayCan**）尝试把 LLM 与工具调用或机器人控制结合，正是对“知行合一”缺口的直接回应。还有一些研究开始构建 **价值驱动的任务生成框架**，尝试让模型在生成新任务时参考预设的价值函数。想进一步跟进，可以关注 **跨模态世界模型**（如 **Gato**、**Perceiver** 系列）以及 **强化学习与语言模型的混合**（如 **RLHF** 的进阶版）方向。

### 一句话记住它
真正的通用智能不只会“答题”，更要会“动手”，把价值观和真实世界模型嵌进每一次行动。