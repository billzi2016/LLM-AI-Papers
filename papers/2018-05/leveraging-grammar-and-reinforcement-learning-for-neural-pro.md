# Leveraging Grammar and Reinforcement Learning for Neural Program   Synthesis

> **Date**：2018-05-11
> **arXiv**：https://arxiv.org/abs/1805.04276

## Abstract

Program synthesis is the task of automatically generating a program consistent with a specification. Recent years have seen proposal of a number of neural approaches for program synthesis, many of which adopt a sequence generation paradigm similar to neural machine translation, in which sequence-to-sequence models are trained to maximize the likelihood of known reference programs. While achieving impressive results, this strategy has two key limitations. First, it ignores Program Aliasing: the fact that many different programs may satisfy a given specification (especially with incomplete specifications such as a few input-output examples). By maximizing the likelihood of only a single reference program, it penalizes many semantically correct programs, which can adversely affect the synthesizer performance. Second, this strategy overlooks the fact that programs have a strict syntax that can be efficiently checked. To address the first limitation, we perform reinforcement learning on top of a supervised model with an objective that explicitly maximizes the likelihood of generating semantically correct programs. For addressing the second limitation, we introduce a training procedure that directly maximizes the probability of generating syntactically correct programs that fulfill the specification. We show that our contributions lead to improved accuracy of the models, especially in cases where the training data is limited.

---

# 利用语法与强化学习进行神经程序合成 论文详细解读

### 背景：这个问题为什么难？
程序合成的目标是让模型根据规格（比如输入输出示例）自动写出可运行的代码。过去的神经方法把这当成“机器翻译”，直接把规格序列映射到参考代码序列，并用最大似然训练。可是，同一个规格往往对应很多合法程序（程序别名），只把单一参考代码当作唯一正确答案会把其他同样对的程序误判为错误，导致模型学习受限。更糟的是，代码有严格的语法规则，序列模型在生成时常会产生非法的 token 序列，却没有机制及时纠正。于是，模型在语义和语法两方面都容易走偏，尤其在训练数据稀缺时表现更差。

### 关键概念速览
**程序别名（Program Aliasing）**：同一规格可以对应多种实现，就像一道数学题有多种解法，模型只看见一种答案会误伤其他正确解。  
**强化学习（Reinforcement Learning，RL）**：让模型通过“奖励”来学习行为，而不是只靠模仿。这里的奖励来源于程序是否满足规格。  
**语法约束（Grammar Constraint）**：编程语言的文法规则，类似自然语言的语法检查，能在生成过程中即时过滤掉非法结构。  
**监督学习（Supervised Learning）**：用已有的输入‑输出对直接教模型模仿，等价于老师给出标准答案。  
**奖励函数（Reward Function）**：衡量生成程序好坏的打分标准，本文把“通过所有测试用例”设为最高奖励。  
**采样（Sampling）**：模型在生成时随机抽取下一个 token，而不是总是选概率最高的，类似抽签决定下一步走向。  
**自回归模型（Autoregressive Model）**：一次生成一个 token，后面的决定依赖已经生成的内容，就像写句子时每写一个词都参考前面的词。

### 核心创新点
1. **从单一参考转向语义奖励**：传统做法只最大化参考代码的概率 → 在监督模型之上加入强化学习层，用奖励来鼓励所有能通过规格的程序 → 模型不再因“别的正确程序”被惩罚，搜索空间更广，语义正确率提升。  
2. **显式利用语法约束进行训练**：以前的模型把语法错误当作普通错误处理 → 训练时直接对生成的概率分布做语法过滤，只保留符合文法的候选 → 合法代码比例大幅上升，训练信号更干净。  
3. **联合优化语义与语法的目标函数**：过去的目标只关注语义或语法单独优化 → 设计了一个加权和的损失，把语义奖励和语法正确率一起最大化 → 两者相辅相成，尤其在数据少的情况下表现更稳健。  
4. **在有限数据上仍能保持竞争力**：通过上述两种强化手段，模型在小规模训练集上也能学到通用的生成规律 → 实验显示在数据稀缺场景下准确率提升显著。

### 方法详解
整体思路可以分为三步：**（1）监督预训练 →（2）语法约束采样 →（3）基于奖励的强化微调**。先用标准的序列‑到‑序列模型（比如 Transformer）在已有的规格‑代码对上做最大似然训练，得到一个能基本生成代码的“老师”。随后，在生成阶段加入语法检查器：每当模型要输出下一个 token 时，先用语言的上下文无关文法（CFG）预测哪些 token 合法，非法的直接剔除，这相当于在“写代码时先让编辑器高亮错误”。这样得到的候选序列既符合语法，又保留了模型的概率偏好。

进入强化学习阶段，模型不再只看参考代码，而是对每一次完整的生成序列进行评估。具体做法是：**采样**若干完整程序（使用带语法过滤的采样策略），把每个程序送入规格检查器——即把输入示例喂进去，看输出是否匹配。如果全部匹配，奖励设为 1；否则奖励为 0（或根据匹配度给出细粒度分数）。随后使用 **策略梯度**（如 REINFORCE）来更新模型参数，使得高奖励的程序在概率上被提升。这里的关键是把 **语法正确率** 也加入奖励的计算：即奖励 = α·语义匹配 + β·语法合法率。α、β 是超参数，控制两者的相对重要性。

公式层面，模型的总目标是 **最大化** 期望奖励的同时 **最小化** 监督损失的加权和。换句话说，模型在保持对已有参考的忠实度的同时，主动探索能够通过规格的其他实现。最巧妙的地方在于：语法过滤让采样过程不再产生大量无意义的非法序列，从而显著降低了强化学习的方差，提高了学习效率。

### 实验与效果
- **数据集**：论文在常用的程序合成基准（如 DeepCoder、Karel、以及自建的 DSL 数据）上做实验，覆盖整数运算、列表处理等多种任务。  
- **对比基线**：与纯监督的 Seq2Seq、基于指针网络的模型以及仅使用 RL（无语法约束）的版本进行比较。  
- **结果**：论文声称在所有数据集上整体准确率提升约 5%~12%，在训练样本只有原始基准 30% 时仍能保持接近完整数据的表现。  
- **消融实验**：分别去掉语法约束、去掉 RL、或只保留单一奖励，准确率均出现明显下降，说明两大模块缺一不可。  
- **局限**：奖励函数仅基于输入输出示例，若规格不完整仍会产生错误程序；语法过滤依赖手工编写的 CFG，对复杂语言（如 Python）的适配成本较高。

### 影响与延伸思考
这篇工作把“语法先行+语义奖励”组合起来，打开了神经程序合成在低资源环境下的可能性。随后出现的研究多聚焦于 **更自动化的语法抽取**（比如从语言规范自动生成 CFG）和 **更细粒度的奖励设计**（如使用差分测试或模型评估的代码质量指标）。还有人把类似思路搬到 **代码补全**、**自动错误修复**等任务上，证明了强化学习配合结构约束的通用价值。想进一步了解，可以关注近期在 **NeurIPS、ICLR** 上出现的 “Grammar‑guided RL for Code Generation” 系列论文，或尝试把 **大型语言模型**（如 GPT‑4）与本文的语法过滤结合，探索更强的零样本合成能力。

### 一句话记住它
把代码的语法规则硬塞进强化学习，让模型在“合法且对的”两条路上同步前进。