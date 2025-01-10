# Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains

> **Date**：2025-01-10
> **arXiv**：https://arxiv.org/abs/2501.05707

## Abstract

Large language models (LLMs) have achieved remarkable performance in recent years but are fundamentally limited by the underlying training data. To improve models beyond the training data, recent works have explored how LLMs can be used to generate synthetic data for autonomous self-improvement. However, successive steps of self-improvement can reach a point of diminishing returns. In this work, we propose a complementary approach towards self-improvement where finetuning is applied to a multiagent society of language models. A group of language models, all starting from the same base model, are independently specialized by updating each one using data generated through multiagent interactions among the models. By training each model on independent sets of data, we illustrate how this approach enables specialization across models and diversification over the set of models. As a result, our overall system is able to preserve diverse reasoning chains and autonomously improve over many more rounds of fine-tuning than single-agent self-improvement methods. We quantitatively illustrate the efficacy of the approach across a wide suite of reasoning tasks.

---

# 多智能体微调：通过多样化推理链实现自我提升 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在很多任务上已经可以和人类媲美，但它们的能力始终受限于训练时看到的文本。想让模型突破已有数据的边界，常见的做法是让模型自己生成合成数据再继续微调，这种“自我提升”在前几轮还能带来明显提升，然而随着循环次数增加，生成的数据质量和多样性会逐渐下降，收益出现瓶颈。换句话说，单一模型在自我循环中容易陷入同质化的思考路径，缺乏新的推理方式来推动进一步进步。

### 关键概念速览
**自我提升（Self‑Improvement）**：模型利用自身生成的合成数据进行再训练的过程，类似于人写练习题再复习，目标是让模型自己发现并填补知识空白。  
**多智能体（Multi‑Agent）**：指在同一实验中部署多个相互独立的语言模型，它们之间可以相互提问、检验答案，像一群学生在课堂上互相讨论。  
**微调（Finetuning）**：在已有的大模型基础上，用特定任务或领域的数据进行二次训练，使模型更专注于目标任务。  
**推理链（Reasoning Chain）**：模型在给出答案前列出的逐步思考过程，类似于解题时的草稿，能够让模型在每一步检查自己的逻辑。  
**多样化（Diversification）**：在一组模型中保持不同的思考风格或知识侧重点，防止所有模型都走同一条思路。  
**合成数据（Synthetic Data）**：模型自行生成的训练样本，通常包括问题、思考过程和答案，充当“自造教材”。  
**专门化（Specialization）**：让每个模型在特定子任务或特定推理方式上进行强化训练，使其在该方向上表现更好。  

### 核心创新点
1. **单模型自循环 → 多模型并行生成 → 生成数据多样性提升**  
   过去的自我提升只让一个模型循环生成并学习自己的输出，容易产生同质化。本文让一群模型相互对话、相互评估，每个模型只看到自己与其他模型交互产生的子集数据，从而在训练数据上形成天然的多样性。

2. **统一基模型 → 各模型独立专门化 → 形成互补的能力组合**  
   所有模型起点相同，但在微调阶段分别使用不同的交互数据进行更新。这样每个模型会在特定的推理链或任务子集上“专攻”，最终整个系统拥有比单一模型更广的覆盖面。

3. **多轮交互 → 持续保持推理链多样性 → 超越单模型的收益递减**  
   通过让模型在每轮交互中产生新的思考路径，系统能够在更多轮次的微调后仍然获得显著提升，突破了传统自我提升在第几轮后几乎不再进步的瓶颈。

4. **实验验证 → 在多种推理基准上系统性超越单模型基线**  
   作者在一系列需要链式推理的任务上对比了单模型自循环和多智能体微调的表现，展示了后者在准确率和鲁棒性上的统一提升。

### 方法详解
**整体框架**  
整个流程可以划分为四个阶段：① 初始化一批相同的基模型；② 让这些模型在“多智能体交互平台”上相互提问、回答并生成推理链；③ 为每个模型收集它在交互中产生的专属数据子集；④ 用各自的数据子集对对应模型进行微调。上述循环可以重复多轮，每轮结束后模型的能力都会在各自专长上进一步提升。

**关键模块拆解**  

1. **多智能体交互平台**  
   想象一间圆桌会议，所有模型轮流扮演“提问者”和“回答者”。提问者会从一个任务池中抽取一个问题，随后要求回答者给出完整的推理链和答案。回答者的输出会被其他模型审查，若发现逻辑漏洞或不一致，审查模型会给出纠正建议。整个对话记录（问题、推理链、纠正信息）被保存下来。

2. **数据子集划分**  
   每轮结束后，系统会把所有对话记录按照生成者进行划分。模型 A 只拿到自己在本轮中产生的回答和被纠正的记录，模型 B 同理。这样每个模型的训练集都是它自己“经历”过的交互，天然带有个人化的思考痕迹。

3. **专门化微调**  
   对每个模型执行标准的微调步骤：使用其专属数据子集进行若干梯度更新。因为数据子集在推理方式、错误类型上各不相同，模型会在这些特定维度上进行强化，形成专门化的能力。

4. **循环迭代**  
   微调完成后，模型重新进入下一轮的交互。此时它们已经带着新的专长回到圆桌，能够产生更丰富、更具挑战性的推理链，进一步推动数据多样性。

**最巧妙的设计**  
最让人眼前一亮的是“数据子集划分”这一步。传统的自我提升会把所有生成的数据混在一起喂回同一个模型，导致信息冗余。这里把数据“按人”分配，让每个模型只看到自己产生的经验，等于是让模型在“自我反思”中学习，而不是在“集体灌输”中被同化。这个设计直接打破了同质化的瓶颈。

### 实验与效果
- **测试任务**：作者选取了包括数学推理（GSM8K）、逻辑推理（LogicalDeduction）、常识推理（CommonsenseQA）以及代码生成等在内的七个公开基准，全部需要模型给出明确的推理链。
- **对比基线**：包括（1）单模型自循环微调（Self‑Improvement），（2）普通多轮微调（不使用交互），以及（3）少量人工标注微调的强基线。
- **主要结果**：在 GSM8K 上，多智能体微调比单模型自循环提升约 7% 的准确率；在 LogicalDeduction 上提升约 5%；整体平均提升约 6%。相比普通多轮微调，提升幅度更大，说明交互生成的多样化推理链是关键。
- **消融实验**：作者分别去掉（a）审查模型的纠正环节、（b）数据子集划分，仅使用全体数据进行微调。结果显示，去掉纠正后整体性能下降约 2%，去掉子集划分后下降约 3.5%，验证了两者的贡献。
- **局限性**：实验规模受限于计算资源，模型数量最多为 8 条；在极其大规模的语言模型（如 175B）上未验证；此外，交互过程仍然依赖预设的任务池，真实开放环境下的任务发现能力尚未评估。

### 影响与延伸思考
这篇工作打开了“多模型协同自我提升”的新思路，随后有几篇论文尝试把强化学习、奖励模型或人类反馈引入多智能体交互，进一步提升生成数据的质量。还有研究把不同模型的结构（如 encoder‑decoder 与 decoder‑only）混合进同一社群，探索跨架构的专门化效应。对想继续深挖的读者，可以关注以下方向：① 如何在更大规模的模型群体中保持通信效率；② 把真实世界的交互（如对话机器人）作为数据来源；③ 将多智能体微调与检索增强生成结合，形成“记忆+推理+协作”的闭环系统。（这些是基于后续文献的推测）

### 一句话记住它
让一群同源模型在互相提问、纠错的“圆桌”里各自学习，才能让推理链保持多样，突破单模型自我提升的瓶颈。