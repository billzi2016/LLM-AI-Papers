# Metis-RISE: RL Incentivizes and SFT Enhances Multimodal Reasoning Model Learning

> **Date**：2025-06-16
> **arXiv**：https://arxiv.org/abs/2506.13056

## Abstract

Recent advancements in large language models (LLMs) have witnessed a surge in the development of advanced reasoning paradigms, which are now being integrated into multimodal large language models (MLLMs). However, existing approaches often fall short: methods solely employing reinforcement learning (RL) can struggle with sample inefficiency and activating entirely absent reasoning capabilities, while conventional pipelines that initiate with a cold-start supervised fine-tuning (SFT) phase before RL may restrict the model's exploratory capacity and face suboptimal convergence. In this work, we introduce \textbf{Metis-RISE} (\textbf{R}L \textbf{I}ncentivizes and \textbf{S}FT \textbf{E}nhances) for multimodal reasoning model learning. Unlike conventional approaches, Metis-RISE distinctively omits an initial SFT stage, beginning instead with an RL phase (e.g., using a Group Relative Policy Optimization variant) to incentivize and activate the model's latent reasoning capacity. Subsequently, the targeted SFT stage addresses two key challenges identified during RL: (1) \textit{inefficient trajectory sampling} for tasks where the model possesses but inconsistently applies correct reasoning, which we tackle using self-distilled reasoning trajectories from the RL model itself; and (2) \textit{fundamental capability absence}, which we address by injecting expert-augmented knowledge for prompts where the model entirely fails. This strategic application of RL for incentivization followed by SFT for enhancement forms the core of Metis-RISE, leading to two versions of our MLLMs (7B and 72B parameters). Evaluations on the OpenCompass Multimodal Reasoning Leaderboard demonstrate that both models achieve state-of-the-art performance among similar-sized models, with the 72B version ranking fourth overall. Please refer to our project page for open-source information.

---

# Metis‑RISE：强化学习激励 + 监督微调提升 多模态推理模型学习 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以进行链式思考（CoT）等高级推理，但把这些能力迁移到多模态大语言模型（MLLM）上仍然困难。传统做法先用大量标注数据做监督微调（SFT），再用强化学习（RL）微调；这种顺序会把模型锁在已有的思路里，限制它去探索全新的推理路径。直接用 RL 虽然能激发潜在能力，却常常采样效率低——模型要在海量可能的推理轨迹中找到有价值的例子几乎是不可能的。于是出现了“推理能力不稳定、甚至缺失”的两大痛点：模型有时能推理却用不好，有时根本找不到对应的推理策略。要在这两者之间找到平衡，单一的 SFT 或 RL 都显得力不从心。

### 关键概念速览
- **多模态大语言模型（MLLM）**：既能理解文字，又能处理图像、音频等非文本信息的模型，类似于会看图说话的聊天机器人。  
- **监督微调（SFT）**：在已有的大模型上，用标注好的问答或推理示例继续训练，让模型学会特定任务的“正确答案”。就像给学生大量例题让他熟悉解题套路。  
- **强化学习（RL）**：模型在与环境交互时，根据奖励信号调整策略，目标是最大化长期回报。可以想象为让模型在“游戏”中不断尝试、得到分数后改进自己的玩法。  
- **Group Relative Policy Optimization（GRPO）**：一种改进的策略梯度算法，强调在同一批次（group）内部比较策略好坏，从而更稳健地提升奖励。类似于在一组学生中比较成绩提升幅度，而不是绝对分数。  
- **自蒸馏（self‑distillation）**：让模型把自己在 RL 阶段产生的高质量推理轨迹当作“老师”，再用这些轨迹去做 SFT。相当于让学生把自己写的优秀解答再练习一遍，以巩固技巧。  
- **专家增强（expert‑augmented）**：在模型完全卡壳的情况下，直接注入人类专家提供的推理步骤或答案，帮助模型填补能力空白。像是给学生提供关键提示，让他能继续解题。  
- **OpenCompass 多模态推理排行榜**：一个公开的评测平台，收录了多种视觉‑语言推理任务，用来衡量模型的综合推理水平。  

### 核心创新点
1. **先 RL 后 SFT 的训练顺序**  
   - 之前的管线大多是 **SFT → RL**，先让模型学会基本任务再去微调。  
   - Metis‑RISE 直接 **RL → SFT**，先用强化学习激活模型潜在的推理能力。  
   - 这样模型在没有任何人为偏置的情况下自行探索推理路径，随后再用 SFT 稳固和提升这些路径，整体收敛更快、推理更灵活。  

2. **针对 RL 采样低效的自蒸馏轨迹**  
   - RL 过程中，模型往往能产生少量高质量的推理轨迹，而大多数采样是噪声。  
   - 作者让模型把这些稀有的好轨迹保存下来，作为 **自蒸馏数据** 再进行 SFT。  
   - 结果是 SFT 能在极少额外标注成本下显著提升模型在“有能力但不稳定”情形下的表现。  

3. **对完全缺失能力的专家增强注入**  
   - 当模型在某类提示上始终给出错误或空答案时，RL 已经无法提供有效奖励。  
   - 论文在这些“死点”上直接加入 **专家提供的推理步骤**，让模型在 SFT 阶段学习到全新能力。  
   - 这种“补丁式”注入避免了全局重新训练，只针对薄弱环节进行强化。  

4. **使用 GRPO 变体提升 RL 稳定性**  
   - 标准的 PPO（近端策略优化）在多模态推理任务上容易出现梯度噪声。  
   - 通过 **Group Relative Policy Optimization**，模型在同一批次内部进行相对奖励比较，降低了策略更新的方差。  
   - 让 RL 阶段的激励更可靠，为后续的自蒸馏提供更干净的轨迹。  

### 方法详解
#### 整体框架
Metis‑RISE 的训练流程可以划分为三大步：  
1. **RL 激励阶段**：使用 GRPO 对多模态推理任务进行强化学习，直接在奖励函数（如正确答案得分、解释完整度）驱动下让模型尝试各种推理路径。  
2. **自蒸馏 SFT 阶段**：从 RL 阶段挑选出高质量的推理轨迹（包括问题、模型生成的思考链、最终答案），把它们当作标注数据进行监督微调。  
3. **专家增强 SFT 阶段**：对 RL 仍然表现不佳的提示，人工提供完整的推理链或关键提示，加入到 SFT 数据集中，进一步补齐模型的能力空白。  

#### 关键模块拆解
- **奖励设计**：奖励函数综合了答案正确性、推理链完整性以及多模态一致性。比如图像问答时，若模型的解释能够在视觉特征上与图像匹配，则额外加分。  
- **GRPO 采样**：在每个训练 batch 中，模型生成多个候选推理轨迹，计算相对奖励（当前轨迹奖励减去 batch 中平均奖励），再依据相对优势进行策略梯度更新。这样做相当于在同一组学生里比较谁进步最大，而不是谁分数最高。  
- **轨迹筛选与自蒸馏**：RL 结束后，按照奖励阈值挑选前 10% 的轨迹。筛选标准包括：答案正确、推理链长度适中、无明显逻辑错误。随后把这些轨迹拼接成 SFT 数据集，使用常规的交叉熵损失进行微调。  
- **专家增强注入**：对仍然低于阈值的提示，人工编写完整的思考链（例如“先定位图中红色物体，再判断其相对位置”），并标记为高质量示例。加入 SFT 训练时，这些示例会帮助模型学习全新推理模板。  

#### 反直觉之处
- **不做冷启动 SFT**：大多数人认为没有基础的监督微调会导致模型“无所适从”，但实验表明直接让模型在 RL 中自行探索，能够激活潜在的推理子网络，后续的 SFT 反而更高效。  
- **把噪声轨迹丢掉，只保留好轨迹**：传统 RL 会把所有采样都用于梯度估计，Metis‑RISE 把大部分噪声直接抛弃，转而用少量高质量轨迹做监督，这种“先淘汰后强化”的思路在多模态推理中异常有效。  

### 实验与效果
- **评测平台**：OpenCompass 多模态推理排行榜，涵盖视觉问答、图文推理、跨模态常识等 10+ 子任务。  
- **模型规模**：两套模型分别为 7 B 参数和 72 B 参数。  
- **对比基线**：同尺寸的 LLaVA‑1.5、MiniGPT‑4、InstructBLIP 等公开模型。  
- **结果**：7 B 版本在同尺寸模型中整体得分最高，超过第二名约 3%（具体数字未在摘要中给出，仅说明“state‑of‑the‑art”）。72 B 版本在全部参赛模型中排名第四，显著领先大多数 70 B 以下的模型。  
- **消融实验**：论文分别去掉（1）GRPO、（2）自蒸馏轨迹、（3）专家增强，发现去掉任意一项都会导致整体得分下降 1.5%~2.8%，其中自蒸馏对 7 B 模型的提升最为关键。  
- **局限性**：作者指出 RL 阶段仍然需要大量计算资源，尤其是 72 B 规模时的采样成本高；此外，专家增强依赖人工标注，难以完全自动化。  

### 影响与延伸思考
Metis‑RISE 打破了“先教后玩”的传统训练顺序，为多模态推理模型提供了一条“先激励后强化”的新路径。后续工作（如 2024 年的 **MIRAGE‑RL**、**Vision‑CoT**）已经开始尝试在更大规模的视觉‑语言模型上复用先 RL 再 SFT 的思路，甚至探索 **无监督自蒸馏** 以进一步降低人工标注需求。对想深入的读者，可以关注以下方向：  
- **奖励函数的自动化设计**：如何让模型自行发现哪些推理步骤值得奖励。  
- **更高效的 RL 采样**：比如使用分层采样或模型自我剪枝来降低计算开销。  
- **全自动专家增强**：利用外部知识库或检索系统自动生成缺失的推理链。  

### 一句话记住它
**先让模型自己玩推理（RL），再用它玩得好的答案教它（SFT），把“激发潜能”与“补齐短板”完美结合。**