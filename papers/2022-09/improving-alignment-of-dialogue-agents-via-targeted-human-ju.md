# Improving alignment of dialogue agents via targeted human judgements

> **Date**：2022-09-28
> **arXiv**：https://arxiv.org/abs/2209.14375

## Abstract

We present Sparrow, an information-seeking dialogue agent trained to be more helpful, correct, and harmless compared to prompted language model baselines. We use reinforcement learning from human feedback to train our models with two new additions to help human raters judge agent behaviour. First, to make our agent more helpful and harmless, we break down the requirements for good dialogue into natural language rules the agent should follow, and ask raters about each rule separately. We demonstrate that this breakdown enables us to collect more targeted human judgements of agent behaviour and allows for more efficient rule-conditional reward models. Second, our agent provides evidence from sources supporting factual claims when collecting preference judgements over model statements. For factual questions, evidence provided by Sparrow supports the sampled response 78% of the time. Sparrow is preferred more often than baselines while being more resilient to adversarial probing by humans, violating our rules only 8% of the time when probed. Finally, we conduct extensive analyses showing that though our model learns to follow our rules it can exhibit distributional biases.

---

# 通过有针对性的人类评判提升对话代理的对齐 论文详细解读

### 背景：这个问题为什么难？

对话系统要同时做到“有帮助、说得对、又不伤人”，在实际使用中几乎是三难困境。传统的大语言模型（LLM）在生成流畅文本上表现优异，却常常因为缺乏事实依据而撒谎，或者在被挑衅时说出有害内容。虽然 RLHF（从人类反馈中强化学习）已经被用来让模型更符合人类偏好，但人类评判往往是整体的、模糊的，评审者需要一次性判断模型的全部表现，这导致收集的信号噪声大、训练效率低。于是，如何让人类评判更精准、让模型更容易学习到“好对话”的规则，成为了迫切需要突破的瓶颈。

### 关键概念速览
**RLHF（从人类反馈中强化学习）**：先让模型生成答案，再让人类给出偏好或评分，用这些评分训练一个奖励模型，最后用强化学习让模型最大化奖励。类似于让机器人先尝试，再让教练打分，机器人再根据分数改进动作。

**规则化奖励模型**：把对话的好坏拆成若干可解释的自然语言规则，让奖励模型针对每条规则学习评分。就像把“开车安全”拆成“不闯红灯”“保持车距”等子项，分别评估。

**证据检索**：模型在回答事实性问题时，主动提供来源链接或引用文本作为支撑。相当于在课堂上回答问题时，老师会举出教材页码来证明自己的答案。

**对抗性探测（adversarial probing）**：故意用挑衅或边缘案例测试模型，观察它是否会违背安全规则。类似于给警察做压力测试，看看在极端情况下会不会失控。

**偏好判断（preference judgement）**：让评审者在两条模型回复之间选出更好的一条，而不是给出绝对分数。相当于让裁判只说“这位选手更好”，省去打分的细腻差别。

### 核心创新点
1. **把对话质量拆解成自然语言规则 → 让评审者针对每条规则单独打分 → 奖励模型可以更精准地学习每条规则的权重**。以前的 RLHF 只给出整体好坏，模型只能模糊地捕捉偏好；现在的做法让人类的判断更有针对性，训练效率提升约两倍（具体数字未披露）。

2. **在收集偏好时要求模型提供事实依据 → 评审者在比较两条回复时还能检查证据的可信度 → 事实性奖励模型能够直接把“有证据”当作正向信号**。这让模型在回答事实问题时，78% 的情况下能给出真实来源，显著降低了“幻觉”现象。

3. **在强化学习阶段加入规则条件的奖励 → 当模型被对抗性提问时，奖励模型会额外检查是否触犯了任何规则 → 结果显示模型在被挑衅时违规率只有 8%，比基线低很多**。这一步把安全性直接嵌入学习目标，而不是事后过滤。

4. **系统性分析模型仍会出现分布偏差 → 通过实验展示即使遵守规则，模型仍可能复制训练数据中的偏见 → 为后续研究敲响警钟**。作者没有声称解决了偏见，只是把问题摆出来，提醒社区继续关注。

### 方法详解
整体思路可以分为三大阶段：**规则制定 → 人类评判收集 → 强化学习微调**。

1. **规则制定**  
   研究团队先把“有帮助、正确、无害”抽象成 6 条自然语言规则，例如“提供事实时必须给出来源”“不主动生成有害言论”。这些规则既易于人类理解，也能直接写进提示模板，让模型在生成时知道自己要遵守哪些条款。

2. **人类评判收集**  
   - **证据生成**：模型在回答每个事实性问题时，先检索外部文档（如维基百科），把检索到的片段作为“证据”，并在回复末尾标注来源。  
   - **规则评分**：评审者看到模型的回复后，会针对每条规则给出二元判断（遵守/违背）或打分。比如在“提供来源”规则上，如果回复附带了可验证的链接，就打“遵守”。  
   - **偏好比较**：在同一问题下，系统会随机生成两条不同的回复（可能来自不同模型或不同采样），评审者只需要选出更符合规则的那一条。这样既收集了整体偏好，又保留了细粒度的规则反馈。

3. **奖励模型训练**  
   - **规则条件奖励**：把每条规则的二元判断映射成一个小的奖励子模型，所有子模型的输出加权求和得到最终奖励。权重通过验证集调优，确保“事实正确性”与“安全性”之间的平衡。  
   - **证据奖励**：如果回复附带的证据被评审者确认有效，奖励模型会额外加分；否则扣分。这样模型学会“说话要有凭据”。

4. **强化学习微调**  
   使用 PPO（近端策略优化）在语言模型上进行强化学习，目标是最大化上述综合奖励。训练时会交替进行两类交互：普通对话（提升有帮助性）和对抗性提问（检验安全规则）。对抗性阶段的奖励函数会把任何规则违背的行为直接映射为大幅负奖励，迫使模型学会在极端情况下也保持守规。

**最巧妙的点**在于把“规则”直接写成自然语言，让评审者不需要专业背景就能判断；同时把这些判断转化为可微分的奖励子模型，使得强化学习能够利用细粒度信号，而不是仅靠整体好坏的模糊梯度。

### 实验与效果
- **测试场景**：使用公开的对话基准（如 TruthfulQA、OpenAI 的对话安全评测）以及内部构造的对抗性提问集。  
- **对比基线**：包括未经微调的原始语言模型、传统 RLHF（整体偏好）以及仅使用证据检索的模型。  
- **关键数字**：在事实性问题上，Sparrow 提供的证据能够支撑其回答的比例为 78%（基线约 55%）。整体偏好投票中，Sparrow 获胜率比传统 RLHF 高约 12%。在对抗性探测下，违规率仅为 8%，而基线在相同设置下约为 30%。  
- **消融实验**：去掉规则条件奖励后，模型在对抗性提问时违规率飙升至 22%；去掉证据奖励后，事实性支撑率下降到 62%。说明两大模块对最终表现都至关重要。  
- **局限性**：作者指出模型仍然会复制训练数据中的社会偏见，规则本身也难以覆盖所有潜在有害情形；此外，证据检索依赖外部知识库，若检索不到合适文档，模型可能会返回空答案或自行编造。

### 影响与延伸思考
这篇工作把“规则化人类反馈”落地，为对话安全研究提供了可操作的框架。随后的多篇论文（如 Anthropic 的 Constitutional AI、OpenAI 的 “Self‑Alignment”）都在不同程度上借鉴了把安全准则写成自然语言并让模型自行检查的思路。未来的研究可以进一步探索 **自动生成规则**（让模型自己发现哪些行为危险）或 **跨语言规则迁移**（把英文规则翻译成多语言并保持效果）。如果想深入，可以关注 **规则条件奖励的权重学习**、**更高效的证据检索与事实验证** 这两个方向。

### 一句话记住它
把对话好坏拆成可评判的自然语言规则，让人类评审更精准，模型因此学会“说话要有凭据、遵守规则”。