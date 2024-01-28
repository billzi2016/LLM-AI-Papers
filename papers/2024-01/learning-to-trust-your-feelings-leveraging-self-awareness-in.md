# Learning to Trust Your Feelings: Leveraging Self-awareness in LLMs for   Hallucination Mitigation

> **Date**：2024-01-27
> **arXiv**：https://arxiv.org/abs/2401.15449

## Abstract

We evaluate the ability of Large Language Models (LLMs) to discern and express their internal knowledge state, a key factor in countering factual hallucination and ensuring reliable application of LLMs. We observe a robust self-awareness of internal knowledge state in LLMs, evidenced by over 85% accuracy in knowledge probing. However, LLMs often fail to express their internal knowledge during generation, leading to factual hallucinations. We develop an automated hallucination annotation tool, Dreamcatcher, which merges knowledge probing and consistency checking methods to rank factual preference data. Using knowledge preference as reward, We propose a Reinforcement Learning from Knowledge Feedback (RLKF) training framework, leveraging reinforcement learning to enhance the factuality and honesty of LLMs. Our experiments across multiple models show that RLKF training effectively enhances the ability of models to utilize their internal knowledge state, boosting performance in a variety of knowledge-based and honesty-related tasks.

---

# 学会信任你的感受：利用大语言模型自我意识缓解幻觉 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成文本时经常会出现“幻觉”，即输出看似流畅却与事实不符的内容。过去的办法大多是后处理过滤或在训练数据上做更严格的清洗，但模型本身并不具备判断自己到底知道什么、不知道什么的能力。缺乏这种“自我认知”导致模型在有把握的地方仍可能随意编造，尤其在需要检索或推理的知识密集任务里，错误会被放大。要让模型主动把“我不知道”说出来，必须先让它能够准确评估自己的内部知识状态，这在之前的工作中几乎没有被系统化研究。

### 关键概念速览
- **幻觉（Hallucination）**：模型生成的文本与真实世界事实不匹配的现象，就像人类在记忆模糊时胡乱填空一样。  
- **自我意识（Self‑awareness）**：模型对自身知识覆盖范围的内部评估能力，类似于人类在回答前先在脑中检查是否真的记得答案。  
- **知识探测（Knowledge probing）**：通过专门设计的问答或填空任务，测试模型是否能够正确回忆训练中学到的事实。  
- **一致性检查（Consistency checking）**：让模型对同一事实进行多次独立生成，比较结果是否前后一致，用来发现潜在的幻觉。  
- **Dreamcatcher**：本文提出的自动幻觉标注工具，结合知识探测和一致性检查，为每条生成文本打上“可信度”分数。  
- **RLKF（Reinforcement Learning from Knowledge Feedback）**：一种基于强化学习的训练框架，使用 Dreamcatcher 给出的知识偏好分数作为奖励，鼓励模型在生成时更倾向于使用已知的内部知识。  
- **知识偏好奖励（Knowledge preference reward）**：把模型在知识探测任务中的正确率转化为数值奖励，让模型在强化学习中学会“信任自己的记忆”。  

### 核心创新点
1. **自我意识评估 → 系统化的知识探测实验 → 超过85%的准确率**  
   过去只是在少数基准上检查模型的记忆能力，这里通过大规模的知识探测任务，证明主流 LLM 已经具备相当可靠的内部知识状态感知。

2. **幻觉检测 → Dreamcatcher 融合探测+一致性 → 自动生成可信度标签**  
   传统的幻觉检测依赖人工标注或单一的事实检索，Dreamcatcher 把模型自己的知识探测结果和多次生成的一致性一起考虑，能够在不人工干预的情况下为每条输出打分。

3. **奖励设计 → 知识偏好作为强化学习信号 → RLKF 训练框架**  
   与常见的基于人类偏好（RLHF）或奖励模型的做法不同，RLKF 直接把模型对自身知识的自评转化为奖励，让模型在生成时主动使用已知信息，减少随意编造。

4. **端到端实验验证 → 多模型、多任务评估 → 显著提升事实准确性和诚实性**  
   在多个公开的知识问答和诚实性基准上，使用 RLKF 训练的模型在答案正确率和“我不知道”率上都有明显提升，证明了自我意识驱动的训练可以真正抑制幻觉。

### 方法详解
整体思路可以分为三步：**（1）评估模型的内部知识状态；（2）利用评估结果为生成文本打上可信度标签；（3）把标签转化为奖励进行强化学习**。下面把每一步拆开讲。

1. **知识探测阶段**  
   - 设计一套覆盖广泛事实的问答/填空题库。  
   - 让模型在不提供外部检索的情况下直接回答，每个答案与金标准比对，得到二元正确/错误标签。  
   - 统计整体正确率，得到模型的“自我知识准确度”。这一步相当于给模型装上了“自检仪”，让它知道哪些知识是自己掌握的。

2. **Dreamcatcher 幻觉标注**  
   - 对同一输入，模型生成多次（比如 5 次）答案。  
   - 对每一次生成，使用前一步的知识探测结果检查是否包含已知事实。  
   - 同时计算这些生成之间的文本相似度或逻辑一致性。  
   - 把“是否使用已知事实”和“一致性”两项打分合并，得到一个综合的“可信度分”。可以把它想象成给每条答案贴上了“可信度标签”，高分代表模型在用自己的记忆而不是随意编造。

3. **RLKF 强化学习**  
   - 将可信度分映射为奖励信号：分数越高，奖励越大。  
   - 使用经典的强化学习算法（如 PPO）对模型进行微调。微调的目标是最大化累计奖励，即让模型在生成时倾向于产生高可信度的答案。  
   - 训练过程中仍保留原有的语言建模目标，以免模型只会说“我不知道”。这样模型学会在有把握的情况下直接给出答案，在不确定时主动说出“不确定”。

**最巧妙的点**在于把模型自己的知识探测结果当作外部奖励，而不是依赖人工标注或额外的检索系统。这样既省去了昂贵的标注成本，又让模型的自我认知直接参与学习循环。

### 实验与效果
- **测试任务**：包括常见的事实问答基准（如 TriviaQA、WebQuestions）、知识密集的阅读理解任务以及专门评估模型诚实性的 “TruthfulQA”。  
- **对比基线**：普通的预训练模型、使用传统 RLHF（基于人类偏好）的模型、以及已有的幻觉过滤后处理方法。  
- **结果**：论文声称在所有评测上，使用 RLKF 训练的模型在答案正确率上提升了数个百分点，尤其在 “我不知道” 的召回率上提升显著，幻觉率下降超过 30%。  
- **消融实验**：去掉 Dreamcatcher 的一致性检查，只保留知识探测奖励，效果下降约 10%；去掉奖励中的知识偏好，只用一致性奖励，模型仍能一定程度抑制幻觉，但整体提升不如完整方案。  
- **局限性**：作者指出当前的知识探测题库覆盖仍有限，模型在未见过的领域仍可能产生幻觉；此外，强化学习过程对计算资源要求较高，实际部署时需要权衡。

### 影响与延伸思考
这篇工作把“模型自我认知”直接搬进训练循环，开启了“自我监督式事实校准”的新方向。随后的研究开始探索更细粒度的自我评估（比如对每个生成 token 的置信度）以及把外部检索与自我意识结合的混合方案。对想进一步了解的读者，可以关注以下两个方向：① 基于内部置信度的动态生成控制；② 将自我意识与多模态信息（图像、表格）融合的跨模态事实校准。整体来看，这篇论文为降低大模型幻觉提供了一个可扩展、无需大量人工标注的思路。

### 一句话记住它
让模型先“检查自己的记忆”，再把检查结果当奖励，才能真正让它在生成时少说假话。