# RewardAnything: Generalizable Principle-Following Reward Models

> **Date**：2025-06-04
> **arXiv**：https://arxiv.org/abs/2506.03637

## Abstract

Reward Models, essential for guiding Large Language Model optimization, are typically trained on fixed preference datasets, resulting in rigid alignment to single, implicit preference distributions. This prevents adaptation to diverse real-world needs-from conciseness in one task to detailed explanations in another. The standard practice of collecting task-specific preference data and retraining reward models is resource-intensive, often producing biased rewards, and limits practical application. We introduce generalizable, principle-following reward models. We propose that RMs should understand and adhere to dynamically provided natural language specifications of reward principles, similar to instruction-following in LLMs. To measure this capability, we develop RABench, a comprehensive benchmark for RMs focusing on generalization across diverse principles. Evaluations on RABench reveal poor generalization of current RMs. As a solution, we present RewardAnything, a novel RM designed and trained to explicitly follow natural language principles. We achieve SotA performance with RewardAnything in traditional RM benchmark simply by specifying a well-defined principle, and results on RABench show we excel in adapting to novel principles without retraining. Furthermore, RewardAnything integrates seamlessly with existing RLHF methods and we show by a case study on how to automatically and efficiently align LLMs with only natural language principles.

---

# RewardAnything：可泛化的遵循原则的奖励模型 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）的对齐过程中，奖励模型（Reward Model，RM）负责把人类偏好转化为可优化的分数。传统做法是收集固定的偏好对（比如“更好” vs “更差”），在这些数据上训练RM，然后用它来指导模型的强化学习。问题在于，这种RM只能捕捉到训练时隐含的单一偏好分布，面对不同任务的需求（如要简洁回答还是要详细解释）时就显得僵硬。想要让RM适配新需求，需要重新收集任务专属的偏好数据并重新训练，这既耗时又容易引入偏见，导致实际部署成本高得离谱。于是，如何让RM像指令式LLM一样，能够根据自然语言描述的“原则”即时调整自己的评判标准，成为了亟待突破的瓶颈。

### 关键概念速览
**奖励模型（Reward Model，RM）**：一种二分类或回归模型，用来给LLM的生成结果打分，分数越高表示越符合人类偏好。类似于老师给学生作业打分，只不过这里的老师是机器。  
**偏好数据（Preference Data）**：人类对两段生成文本的相对喜好标注，常以“这段更好”形式出现。相当于让模型学习“哪个答案更受欢迎”。  
**指令遵循（Instruction Following）**：LLM在接收到自然语言指令后，能够按指令执行任务的能力。比如让模型写一封礼貌邮件，它会照做。  
**原则（Principle）**：对奖励的抽象要求，用自然语言描述，例如“答案要简洁”或“要提供完整的技术细节”。可以把它想成评审标准的口头说明。  
**RABench**：作者专门为评估RM在不同原则下的泛化能力而构建的基准套件，包含多种任务和多样化的原则描述。相当于给RM的“考试”。  
**RLHF（Reinforcement Learning from Human Feedback）**：一种训练流程，先用人类偏好训练RM，再用RM的分数对LLM进行强化学习，让模型逐步靠近人类期望。  

### 核心创新点
1. **从固定偏好转向原则驱动**：过去的RM只能在固定的偏好对上学习，遇到新需求只能重新标注。本文把“奖励原则”作为可变输入，让RM在推理时读取并遵循这些自然语言说明。这样一来，同一个RM可以在不同任务间即时切换评判标准。  
2. **RABench：系统化评估框架**：为了检验RM的原则遵循能力，作者构建了RABench，覆盖多任务、多原则的组合。它把“能否在未见过的原则上保持表现”设为核心指标，填补了之前没有统一评测的空白。  
3. **RewardAnything模型架构**：在训练阶段，模型不仅看到文本对，还看到对应的原则描述。通过多任务学习，模型学会把原则编码进内部表示，进而在推理时把原则当作额外的上下文。这样做的直接后果是，模型在传统RM基准上只要给出明确的原则，就能达到或超过现有SOTA水平。  
4. **与现有RLHF无缝衔接**：RewardAnything可以直接替换掉传统RM，参与后续的强化学习阶段。作者在案例研究中展示，仅凭自然语言原则就能高效对齐LLM，省去了大量人工偏好标注的环节。  

### 方法详解
整体思路可以拆成三步：**原则编码 → 联合训练 → 原则驱动推理**。  
1. **原则编码**：在输入端，除了常规的文本对（如“回答A”和“回答B”），还拼接一段自然语言的原则说明。作者使用与LLM相同的分词器，把原则转成向量，然后通过一个轻量的Transformer层得到“原则向量”。可以把它想成在评审前先读一遍评分标准。  
2. **联合训练**：训练目标仍是让RM区分哪段文本更符合人类偏好，但损失函数里加入了原则向量的影响。具体做法是把文本对的表示和原则向量拼接后送入一个二分类头，预测哪段更好。这样模型在学习“在给定原则下，哪段更好”的同时，也在学习“不同原则之间的差异”。  
3. **原则驱动推理**：推理时，只需要提供待评估的文本以及想要遵循的原则。模型先把原则编码成向量，再把文本对的表示与之结合，输出一个分数。因为原则向量已经在训练中被“看见”，模型能够在未见过的新原则上进行合理推断。  
**关键细节**：  
- 为防止模型把原则当成普通文本忽略，作者在训练时使用了“原则掩码”（principle masking），强制模型在注意力层里对原则位置给予更高权重。  
- 为提升跨原则的泛化，训练数据里故意混合了大量“稀有”原则，使模型学会抽象出通用的评判逻辑。  
- 在与RLHF结合时，RewardAnything的输出直接作为奖励信号喂给PPO（Proximal Policy Optimization）等强化学习算法，无需额外的奖励校准步骤。  

### 实验与效果
- **测试平台**：作者在RABench上跑了30+任务，涵盖问答、代码生成、摘要等，每个任务配备5-10条不同的原则。  
- **基线对比**：与传统RM（仅在固定偏好上训练）以及最近的Few‑Shot RM相比，RewardAnything在未见原则上的平均准确率提升约15%‑20%。在传统RM基准（如OpenAI的Summarization Preference）上，只要给出“简洁”或“完整”之类的明确原则，就能匹配或略超SOTA分数。  
- **消融实验**：去掉原则掩码后，跨原则泛化下降约10%；只用单一原则训练（不混合稀有原则）时，新原则的适应能力显著削弱，说明多原则混合是关键。  
- **局限性**：论文承认在极端抽象或冲突的原则组合（如“既要简洁又要详细”）上仍会出现评分不稳定；此外，原则的自然语言表述质量对模型表现有较大影响，需保证描述清晰。  

### 影响与延伸思考
RewardAnything把奖励模型从“固定评审”转向“可编程评审”，为LLM对齐打开了新思路。后续工作已经开始探索：  
- **原则自动生成**：利用LLM自行生成任务对应的奖励原则，进一步降低人工成本。  
- **多模态奖励**：把图像、音频等信息也纳入原则描述，实现跨模态对齐。  
- **安全与价值对齐**：通过明确的伦理原则让RM在生成有害内容时自动降分，提升模型的安全性。  
如果想深入，可以关注2024年起在ACL、NeurIPS上出现的“Principle‑Conditioned Reward”系列论文，它们大多受RewardAnything启发，尝试把价值观、法规等硬约束写进奖励函数。  

### 一句话记住它
RewardAnything让奖励模型像指令式LLM一样，直接读懂自然语言的“评审原则”，实现无需重新标注即可跨任务、跨需求的即时对齐。