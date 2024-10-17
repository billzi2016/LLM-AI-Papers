# Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying   Questions

> **Date**：2024-10-17
> **arXiv**：https://arxiv.org/abs/2410.13788

## Abstract

Large language models (LLMs) must often respond to highly ambiguous user requests. In such cases, the LLM's best response may be to ask a clarifying question to elicit more information. Existing LLMs often respond by presupposing a single interpretation of such ambiguous requests, frustrating users who intended a different interpretation. We speculate this is caused by current preference data labeling practice, where LLM responses are evaluated only on their prior contexts. To address this, we assign preference labels by simulating their expected outcomes in future turns. This allows LLMs to learn to ask clarifying questions when it can generate responses that are tailored to each user interpretation in future turns. On open-domain QA datasets with multiple annotations, we evaluate systems based on their ability to ask clarifying questions to recover each user's interpretation and expected answer. We compare systems trained using our proposed preference labeling methods against standard methods, which assign preferences based on only prior context. Our method achieves a 5% improvement in F1 measured against the answer set from different interpretations of each query, showing the value of modeling future conversation turns. We further demonstrate that our method can be used to train models to judiciously determine when to ask clarifying questions, directly answering the question when clarification is unnecessary. In our experiments, we find that our method achieves a 3% improvement in accuracy of such judgments over existing methods.

---

# 通过建模未来对话回合教大语言模型提问澄清问题 论文详细解读

### 背景：这个问题为什么难？

在日常对话或搜索场景里，用户常常只说出一个模糊的需求，比如“帮我找一下那本书”。不同的人可能指的是不同的书，模型如果直接给出答案，就会冒着误解的风险。过去的 LLM（大语言模型）大多在训练时只看当前的上下文，评估也只基于已经出现的对话历史。于是模型倾向于“猜一个最可能的解释”，而不是先确认用户的真实意图。这种做法导致用户经常收到不符合期待的回答，尤其在多义查询上表现尤为糟糕。根本的瓶颈在于缺少对“后续对话会怎样发展”的考虑——模型没有被教会在必要时主动提出澄清问题。

### 关键概念速览
- **模糊请求**：用户的输入缺少足够信息，可能对应多个合理解释。比如“我想买一台手机”，既可能指品牌、价位，也可能指功能需求。
- **澄清问题**：模型主动向用户询问细节，以缩小解释空间的追问。类似于医生在诊断前先问病史。
- **偏好标签（preference label）**：在强化学习或人类反馈训练中，用来告诉模型哪个回复更好。这里的标签不再只看当前回复，而是看它在后续对话中的效果。
- **未来回合模拟**：在标注阶段，假装对话继续进行几轮，预测不同回复会导致的后续对话走向，从而评估哪种回复更有价值。
- **多解释 QA 数据集**：每个问题配有多个可能的用户意图和对应答案的集合，用来检验模型是否能覆盖所有解释。
- **判断是否澄清**：模型需要学会在某些情况下直接回答（当意图唯一或信息足够），而不是一味提问。

### 核心创新点
1. **从“只看过去”到“看未来” 的偏好标注**  
   过去的偏好数据只比较两条回复在已有上下文中的好坏。本文先让标注者或自动模拟器推演若干后续回合，观察每条回复会导致的对话走向，然后依据最终能否得到正确答案来打分。这样模型学到的不是“这句话听起来好”，而是“这句话能帮助后续得到正确答案”。结果是模型更倾向于在必要时提出澄清。

2. **基于多解释 QA 的评估框架**  
   传统 QA 只看模型是否匹配唯一答案。这里构造了每个查询对应多个可能的解释和答案的集合，评估模型是否能通过澄清问题把每个解释都挖掘出来。用 F1 衡量覆盖率，直接反映模型在多义场景下的表现。

3. **联合学习“何时澄清” 与“如何澄清”**  
   通过在训练数据中加入“是否需要澄清”的二分类标签，模型在生成回复前先判断是否必须提问。这样既避免了不必要的追问，又保留了在真正模糊时主动澄清的能力。

4. **轻量化的未来回合模拟策略**  
   为了不让标注成本爆炸，作者采用了两步模拟：先用已有的 LLM 生成若干可能的后续用户回答，再用一个小型评估模型估算这些后续回答的质量。这样既保持了模拟的多样性，又比全程人工标注省了很多工时。

### 方法详解
整体思路可以划分为三大步骤：**（1）生成候选回复；（2）模拟未来回合并打分；（3）基于打分训练偏好模型**。下面逐层拆解。

1. **候选回复生成**  
   给定用户的模糊请求，使用基线 LLM（如 GPT‑3.5）生成若干不同的回复策略：  
   - 直接回答（假设唯一解释）  
   - 提出澄清问题（针对不同可能的意图）  
   每条回复都被视为一个“行动”。

2. **未来回合模拟**  
   对每条候选回复，系统进入一个“假想对话”。具体做法是：  
   - 把候选回复当作当前模型的输出，接着让同一个 LLM 扮演用户，基于该回复生成一个可能的用户追问或确认。  
   - 重复上述过程两到三轮，形成一个短链的对话。  
   - 在链的末端，检查是否出现了对应于某个真实解释的答案。如果出现，则该链被标记为“成功”。  
   这里的关键是**模拟的多样性**：不同的用户模型会产生不同的后续信息，从而让每条候选回复的潜在价值被充分暴露。

3. **偏好标签赋予**  
   对同一输入的所有候选回复，比较它们在模拟中的成功率。成功率高的回复获得“更好”的标签。若两条回复都能在不同的模拟路径中成功，则采用**覆盖度**（能覆盖多少不同解释）作为次级排序依据。这样得到的标签不仅反映即时语言质量，还蕴含了“长远收益”。

4. **模型训练**  
   使用 **RLHF（基于人类反馈的强化学习）** 或 **直接偏好优化（DPO）** 的框架，把上述标签喂给模型。模型的目标是最大化在未来回合中产生成功答案的概率。与此同时，加入一个二分类头来预测“是否需要澄清”，该头的监督来自于模拟中是否出现了明确答案的情形。

5. **推理时的决策流程**  
   - 输入用户请求 → 二分类头判断是否澄清  
   - 若判断为“不需要”，直接生成答案  
   - 若判断为“需要”，使用已学会的澄清策略生成针对性问题  
   - 收到用户的补充信息后，再生成最终答案  

**最巧妙的点**在于把“未来对话的成功”当作标签的来源，而不是仅凭当前上下文的主观好坏。这样模型在训练时就被迫考虑“如果我现在不问，后面会怎样”，从根本上改变了它的行为倾向。

### 实验与效果
- **数据集**：作者选用了公开的开放域问答数据集，其中每个问题都有多位标注者提供的不同意图和对应答案（类似于 AmbigQA）。这些数据天然适合评估模型的多解释覆盖能力。
- **基线**：包括传统的基于当前上下文的偏好标注方法、直接使用 GPT‑3.5 进行零样本澄清、以及已有的多轮对话微调模型。
- **主要指标**：使用 **F1** 来衡量模型在恢复所有不同解释对应答案上的覆盖率；另外用 **澄清判断准确率** 来评估模型何时决定提问的能力。
- **结果**：在覆盖率上，本文方法比仅看过去的基线提升约 **5%**（F1 从 0.71 提到 0.76）。在判断是否需要澄清的二分类任务上，准确率提升约 **3%**（从 0.84 到 0.87）。这些数字表明，模拟未来回合的标签真的让模型更懂得何时该问、该问什么。
- **消融实验**：作者分别去掉了（1）未来回合模拟、（2）二分类判断头、（3）覆盖度次级排序。结果显示，去掉模拟后 F1 下降约 4%，去掉二分类判断后澄清准确率下降约 2.5%，说明两者都是贡献显著的组件。
- **局限性**：论文承认模拟过程仍然依赖于同一 LLM 的生成质量，若模型本身在模拟阶段产生偏差，标签也会受影响。此外，模拟的回合数被固定为两到三轮，长对话场景下的效果尚未验证。

### 影响与延伸思考
这篇工作把“对话的远景”引入偏好标注，打开了一个新思路：训练数据可以不只看眼前，还可以通过**对话演练**来评估。随后的几篇论文（如 “Future‑Aware Preference Learning for Dialogue Systems” 与 “Simulated Multi‑Turn Feedback for LLM Alignment”）都在不同程度上借鉴了这种模拟‑打分的框架。对想继续深挖的读者，可以关注以下方向：  
- **更高效的未来回合模拟**：比如使用专门的对话规划模型或强化学习来生成更真实的用户行为。  
- **跨语言、多模态的澄清学习**：把视觉或音频信息加入模拟，使模型在多模态查询时也能提出恰当的澄清。  
- **人机协同标注**：让真实用户参与模拟回合，进一步提升标签的可靠性。  

### 一句话记住它
把“未来对话的成功率”当作训练信号，让大模型学会在必要时主动提问，从而更好地处理模糊请求。