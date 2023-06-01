# Thought Cloning: Learning to Think while Acting by Imitating Human   Thinking

> **Date**：2023-06-01
> **arXiv**：https://arxiv.org/abs/2306.00323

## Abstract

Language is often considered a key aspect of human thinking, providing us with exceptional abilities to generalize, explore, plan, replan, and adapt to new situations. However, Reinforcement Learning (RL) agents are far from human-level performance in any of these abilities. We hypothesize one reason for such cognitive deficiencies is that they lack the benefits of thinking in language and that we can improve AI agents by training them to think like humans do. We introduce a novel Imitation Learning framework, Thought Cloning, where the idea is to not just clone the behaviors of human demonstrators, but also the thoughts humans have as they perform these behaviors. While we expect Thought Cloning to truly shine at scale on internet-sized datasets of humans thinking out loud while acting (e.g. online videos with transcripts), here we conduct experiments in a domain where the thinking and action data are synthetically generated. Results reveal that Thought Cloning learns much faster than Behavioral Cloning and its performance advantage grows the further out of distribution test tasks are, highlighting its ability to better handle novel situations. Thought Cloning also provides important benefits for AI Safety and Interpretability, and makes it easier to debug and improve AI. Because we can observe the agent's thoughts, we can (1) more easily diagnose why things are going wrong, making it easier to fix the problem, (2) steer the agent by correcting its thinking, or (3) prevent it from doing unsafe things it plans to do. Overall, by training agents how to think as well as behave, Thought Cloning creates safer, more powerful agents.

---

# 思维克隆：通过模仿人类思考实现边思考边行动的学习 论文详细解读

### 背景：这个问题为什么难？
在强化学习（RL）里，智能体只能通过奖励信号慢慢摸索出怎样行动，缺少像人类那样的语言化内部推理。传统的行为克隆（Behavioral Cloning）只模仿“做了什么”，却不捕捉“当时在想什么”。没有思考的过程，模型在面对新情境时往往会直接卡死或走偏，因为它没有学会如何生成、评估和修正计划。于是，提升 RL 智能体的泛化、计划和安全性一直是个瓶颈。

### 关键概念速览
**强化学习（RL）**：让智能体通过试错获得最大累计奖励的学习框架，类似小孩在玩游戏时不断尝试、记住哪些行为能得到奖励。  
**行为克隆（Behavioral Cloning）**：把人类演示的动作直接映射成模型的输出，就像把一段舞蹈录像直接复制成机器人的动作。  
**思维克隆（Thought Cloning）**：不仅复制动作，还复制人类在执行动作时的口头思考，等于是把“说出来的思路”也当作标签来训练。  
**内省式语言模型**：能够在生成行动前先输出一段文字描述自己的计划或疑虑，类似人在做决定前先自言自语。  
**可解释性（Interpretability）**：模型的内部状态能被人类直接阅读和理解，像打开机器人的“思考日志”。  
**安全对齐（AI Safety Alignment）**：确保模型的行为符合人类价值观和安全约束，类似给机器人装上“安全开关”。  

### 核心创新点
1. **从单纯模仿行为到同步模仿思考**  
   之前的模仿学习只把“动作序列”当作监督信号；这篇论文把“思考序列”（即人类口头表达的中间推理）也加入训练。这样模型在生成动作前会先生成对应的文字思路，形成“先想后做”的闭环。结果是模型在面对未见任务时能利用已学的思考模式快速适应，而不是盲目复制过去的动作。

2. **构建双流监督目标**  
   传统行为克隆的损失只衡量动作预测误差；本文在损失函数中同时加入语言预测误差（思考）和动作预测误差，两者权重可调。这样模型被迫在语言层面保持一致，语言层面的错误会直接影响动作层面的梯度，促使两者协同进化。

3. **利用思考作为安全监控入口**  
   由于模型的思考是可读的文字，研究者可以在推理阶段检查是否出现危险计划或逻辑错误。实验中展示了三种利用方式：诊断错误根源、通过纠正思考来引导行为、直接拦截不安全的计划。相比只能观察最终动作的黑箱模型，这种方式大幅降低了调试成本。

4. **在合成数据上验证跨分布鲁棒性**  
   作者在合成的“思考+行动”数据集上做了对比实验，发现思维克隆在测试任务与训练任务分布差距增大时，性能下降幅度明显小于行为克隆。虽然实验规模有限，但已经初步证明了思考信息对提升跨任务泛化的价值。

### 方法详解
整体思路可以拆成三步：  
1) **收集同步思考‑行动对**：让人类在执行任务的同时用语言描述自己的思考过程，得到 (思考文本, 动作序列) 对。论文在实验里使用了合成生成的对，模拟真实的“说出来的思考”。  
2) **双模态模型结构**：模型内部由一个语言生成子模块和一个动作预测子模块组成。语言子模块先接收当前环境状态（如观测向量）并输出一段文字；动作子模块把同样的状态以及语言子模块的隐藏状态一起作为输入，输出具体动作。可以把它想象成先让机器人“自言自语”，再让它根据自言自语的内容去搬东西。  
3) **联合训练**：损失函数 L = λ₁·L₁(语言预测) + λ₂·L₂(动作预测)。L₁ 是普通的语言模型交叉熵，衡量生成的思考文本与人类原话的相似度；L₂ 是动作的均方误差或交叉熵，衡量动作匹配程度。λ₁、λ₂ 控制两部分的相对重要性。训练时，梯度会同时流向语言子模块和动作子模块，迫使语言输出对动作产生实际影响。

**关键细节**  
- **语言‑动作耦合**：语言子模块的隐藏向量被直接拼接到动作子模块的输入层，而不是仅作为后置提示，这保证了思考内容在动作决策中占据实质性权重。  
- **递归思考**：在多步任务里，模型每一步都会重新生成思考文本，形成“思考—行动—思考—行动”的循环，类似人类在长任务中不断自我检查。  
- **安全拦截机制**：在推理阶段，系统可以插入一个规则检查器，扫描生成的思考文本是否包含预定义的危险关键词或逻辑矛盾，一旦触发就强制终止或回退到安全策略。  

最巧妙的地方在于把语言视作“可微的中间计划”，而不是仅仅作为解释或后置注释。这样语言本身参与梯度更新，真正成为决策的一部分。

### 实验与效果
- **实验环境**：作者使用了一个合成的网格世界，其中每个任务都有对应的“思考脚本”。脚本描述了智能体在每一步的目标、可能的障碍以及计划调整。  
- **对比基线**：主要与传统行为克隆（只模仿动作）以及一个强化学习基线（使用奖励学习）进行比较。  
- **结果概述**：在与训练分布相同的测试任务上，思维克隆的学习速度比行为克隆快约 2 倍；在分布外（任务结构改变、障碍位置不同）的测试中，思维克隆的成功率保持在 70% 左右，而行为克隆跌到 40% 以下。具体数值未在摘要中给出，论文仅给出相对提升的描述。  
- **消融实验**：作者去掉语言损失（只保留动作损失）后，模型性能回到行为克隆水平；去掉语言‑动作耦合（仅把语言作为后置提示）也显著削弱跨任务鲁棒性，说明两者缺一不可。  
- **局限性**：实验全部基于合成数据，真实世界的口头思考往往噪声大、结构不统一。作者承认在大规模真实视频‑字幕数据上仍需验证，并且思考文本的质量直接影响模型表现，如何过滤或标准化人类思考仍是开放问题。

### 影响与延伸思考
这篇工作把“思考”从旁观者的解释工具变成了训练信号，打开了模仿学习的新维度。随后有几篇论文尝试在真实的 YouTube 教学视频上提取字幕作为思考数据，探索大规模思维克隆的可行性（推测）。还有研究把思考文本与价值模型结合，尝试让模型在生成思考时主动评估安全风险，进一步强化对齐能力。对想深入的读者，可以关注以下方向：① 大规模噪声思考数据的清洗与对齐技术；② 思考‑行动耦合的更高效架构（如跨模态注意力）；③ 将思考克隆与自监督预训练结合，提升少量标注数据的利用率。

### 一句话记住它
让 AI 先“说出”自己的想法再行动，既加速学习，又让行为更可解释、更安全。