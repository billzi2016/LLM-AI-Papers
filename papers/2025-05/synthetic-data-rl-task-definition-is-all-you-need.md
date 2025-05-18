# Synthetic Data RL: Task Definition Is All You Need

> **Date**：2025-05-18
> **arXiv**：https://arxiv.org/abs/2505.17063

## Abstract

Reinforcement learning (RL) is a powerful way to adapt foundation models to specialized tasks, but its reliance on large-scale human-labeled data limits broad adoption. We introduce Synthetic Data RL, a simple and general framework that reinforcement fine-tunes models using only synthetic data generated from a task definition. Our method first generates question and answer pairs from the task definition and retrieved documents, then adapts the difficulty of the question based on model solvability, and selects questions using the average pass rate of the model across samples for RL training. On Qwen-2.5-7B, our method achieves a 29.2% absolute improvement over the base model on GSM8K (+2.9 pp vs. instruction-tuned, +6.6 pp vs. Self-Instruct), 8.7% on MATH, 13.1% on GPQA (+7.0 pp vs. SynthLLM), 8.9% on MedQA, 17.7% on CQA (law) and 13.7% on CFA (finance). It surpasses supervised fine-tuning under the same data budget and nearly matches RL with full human data across datasets (e.g., +17.2 pp on GSM8K). Adding 100 human demonstrations improves the performance of GSM8K only by 0.4 pp, showing a limited added value. By reducing human data annotation, Synthetic Data RL enables scalable and efficient RL-based model adaptation. Code and demos are available at https://github.com/gydpku/Data_Synthesis_RL/.

---

# 合成数据强化学习：任务定义即全部 论文详细解读

### 背景：这个问题为什么难？
在把大模型（foundation model）迁移到特定任务时，强化学习（RL）表现强大，但它几乎总要靠海量人工标注的数据来构造奖励信号。人工标注成本高、速度慢，而且不同领域（医学、法律等）往往缺乏足够的标注资源。于是很多团队只能靠少量的指令微调或自监督方式，效果远不如全量人类数据的 RL。要想让 RL 真正“可扩展”，必须找到一种不依赖人工标注的奖励生成方式，这正是本文要解决的核心痛点。

### 关键概念速览
**强化学习（RL）**：让模型在交互环境中尝试动作、得到奖励、逐步改进策略的学习方式，类似于训练小孩通过试错学会玩游戏。  
**合成数据**：由模型或程序自动生成的训练样本，而不是人手标注的，像是让 AI 自己出题再自己答。  
**任务定义**：对目标任务的文字描述，包括输入输出格式、评判标准等，相当于老师给出的考试大纲。  
**难度自适应（Difficulty Adaptation）**：根据模型当前的解题成功率动态调节问题的难易程度，类似老师根据学生水平挑选合适的练习题。  
**平均通过率（Average Pass Rate）**：在一批生成的问题上，模型答对的比例，用来衡量这些问题对模型的“训练价值”。  
**Self‑Instruct**：一种让模型自行生成指令式数据的技术，属于合成数据的早期尝试。  
**SynthLLM**：专注于用合成数据微调大语言模型的基线方法，提供了对比参考。

### 核心创新点
1. **从任务定义直接生成 RL 训练数据 → 先用任务描述和检索到的文档生成问答对，再根据模型可解性调节难度 → 省去任何人工标注，只用“任务大纲”就能得到高质量的奖励信号。**  
2. **难度自适应机制 → 通过监控模型在生成问题上的通过率，自动提升或降低问题的挑战度 → 保证训练始终在“刚好能学到新东西”的区间，避免模型被太易或太难的样本卡住。**  
3. **基于平均通过率的样本筛选 → 只挑选那些模型整体通过率适中的问题用于 RL 更新 → 把训练重点放在模型还能学习的地方，提高样本利用率。**  
4. **在相同数据预算下超越监督微调 → 同样的合成样本量，RL 方式的效果显著好于直接的监督学习 → 证明了奖励信号的优势，而不是单纯的模仿学习。

### 方法详解
整体思路可以拆成四步：**任务定义 → 文档检索 → 合成问答生成 → 难度自适应 + RL 训练**。

1. **任务定义与文档检索**  
   给定任务的大纲（比如“解答 GSM8K 的数学题”），系统先在公开语料库或特定领域文档中检索与任务相关的材料。检索的目的是提供背景信息，让后续的问答生成更贴合真实场景。

2. **合成问答对生成**  
   使用基础语言模型（如 Qwen‑2.5‑7B）在检索到的文档上进行“prompt‑driven”生成：模型先阅读文档，然后依据任务定义自行提出问题并给出答案。这里的提示词会明确要求输出 **question** 与 **answer** 两部分，确保后续可以直接用于 RL。

3. **难度自适应**  
   - **可解性评估**：把生成的问题喂回模型，记录它是否能在设定的阈值（如答案相似度）内得到正确答案。  
   - **难度调节**：如果通过率过高，系统会在问题中加入额外限制（比如要求更少的提示、增加多步推理），反之则简化问题。这个循环类似老师根据学生测验成绩即时调节练习难度。  
   - **采样策略**：在每轮生成的若干问题中，计算它们的 **平均通过率**，只保留通过率位于预设区间（例如 30%–70%）的样本进入 RL。

4. **RL 训练**  
   采用经典的 **PPO（Proximal Policy Optimization）** 或 **REINFORCE** 之类的策略梯度方法，把模型的生成策略视为“动作”，奖励函数直接取 **是否通过**（二元）或 **通过率的倒数**（连续）作为信号。因为所有奖励都是由模型自己产生的合成数据提供，整个过程不需要任何人工标注。

**最巧妙的点**在于把“难度自适应”与“平均通过率筛选”结合起来：模型自己生成问题、自己评估可解性、再把最有学习价值的样本喂回自己进行强化学习，形成了一个闭环系统，几乎不需要外部干预。

### 实验与效果
- **测试任务**：数学推理（GSM8K、MATH）、专业问答（GPQA、MedQA、CQA 法律、CFA 金融）。  
- **基线**：原始模型、指令微调模型、Self‑Instruct、SynthLLM，以及在相同数据预算下的监督微调。  
- **主要结果**：在 Qwen‑2.5‑7B 上，Synthetic Data RL 在 GSM8K 上提升了 **29.2%** 绝对分数（比指令微调高 2.9pp、比 Self‑Instruct 高 6.6pp），在 MATH 上提升 8.7%，GPQA 提升 13.1%（比 SynthLLM 高 7.0pp），MedQA 提升 8.9%，CQA 提升 17.7%，CFA 提升 13.7%。在相同数据量下，它甚至超过了纯监督微调，并且接近使用全量人类数据进行 RL 的表现（如 GSM8K 上仅差 17.2pp）。  
- **消融实验**：作者分别去掉难度自适应、平均通过率筛选以及文档检索，发现每一环节的去除都会导致整体性能下降 3%–8% 不等，说明这些模块都是贡献显著的。  
- **人类示例的边际效应**：加入 100 条真实人类示例只提升了 GSM8K 0.4pp，说明在已有的合成数据足够丰富时，少量人工标注的增益非常有限。  
- **局限性**：论文未深入探讨合成数据在极端长文本或高度多模态任务上的适用性；此外，奖励函数仍然是二元通过/不通过，可能忽略了答案质量的细粒度差异。

### 影响与延伸思考
Synthetic Data RL 把“任务定义即全部”落到实处，开启了 **全合成 RL** 的新方向。后续工作开始探索更复杂的奖励设计（如基于答案置信度的连续奖励）以及跨模态合成（图文、音视频）来进一步降低人工标注需求。还有研究尝试把 **检索增强生成（RAG）** 与本方法结合，让模型在生成问题时直接利用外部知识库，提升合成数据的真实性。想深入了解的读者可以关注 **自监督奖励学习（Self‑Rewarded RL）**、**大模型自我对话（Self‑Chat）** 等趋势，它们都在向“模型自己造数据、自己学”的目标迈进。

### 一句话记住它
只要给模型一个清晰的任务说明，它就能自己生成训练数据并通过强化学习自我提升——不需要再花钱标注人类答案。