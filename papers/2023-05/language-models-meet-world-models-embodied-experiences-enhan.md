# Language Models Meet World Models: Embodied Experiences Enhance Language   Models

> **Date**：2023-05-18
> **arXiv**：https://arxiv.org/abs/2305.10626

## Abstract

While large language models (LMs) have shown remarkable capabilities across numerous tasks, they often struggle with simple reasoning and planning in physical environments, such as understanding object permanence or planning household activities. The limitation arises from the fact that LMs are trained only on written text and miss essential embodied knowledge and skills. In this paper, we propose a new paradigm of enhancing LMs by finetuning them with world models, to gain diverse embodied knowledge while retaining their general language capabilities. Our approach deploys an embodied agent in a world model, particularly a simulator of the physical world (VirtualHome), and acquires a diverse set of embodied experiences through both goal-oriented planning and random exploration. These experiences are then used to finetune LMs to teach diverse abilities of reasoning and acting in the physical world, e.g., planning and completing goals, object permanence and tracking, etc. Moreover, it is desirable to preserve the generality of LMs during finetuning, which facilitates generalizing the embodied knowledge across tasks rather than being tied to specific simulations. We thus further introduce the classical (EWC) for selective weight updates, combined with low-rank adapters (LoRA) for training efficiency. Extensive experiments show our approach substantially improves base LMs on 18 downstream tasks by 64.28% on average. In particular, the small LMs (1.3B, 6B, and 13B) enhanced by our approach match or even outperform much larger LMs (e.g., ChatGPT).

---

# 语言模型遇见世界模型：具身经验提升语言模型 论文详细解读

### 背景：这个问题为什么难？
大规模语言模型（LM）只在文字数据上训练，虽然能写诗、答题，却常在需要“身体感知”的任务上卡壳——比如判断一个杯子被放到桌子上后是否还能看到，或者规划一次厨房烹饪。传统的提升方法往往是让模型读更多的说明书或常识库，但文字本身并不包含“动作后果”或“空间连贯性”。缺少真实或模拟的身体交互经验，使得模型在物理推理和计划时缺乏必要的感知与行动常识，这正是需要突破的瓶颈。

### 关键概念速览
- **语言模型（LM）**：在海量文本上学习统计规律的神经网络，能够生成或理解自然语言。类似于一个只会说话的“语言机器人”。  
- **世界模型（World Model）**：对外部环境（尤其是物理世界）进行模拟的系统，能够预测动作的后果。可以把它想成一套“虚拟实验室”。  
- **具身经验（Embodied Experience）**：智能体在环境中实际执行动作、观察结果的过程，就像人类在厨房里做饭时获得的感受。  
- **VirtualHome**：一个可编程的家庭场景模拟器，提供家具、物品和可执行的高层次动作。相当于一款“家居版的Minecraft”。  
- **目标导向规划（Goal‑oriented Planning）**：智能体根据预设目标（如“把水杯放进微波炉”）主动搜索并执行一系列动作。类似于人类为完成任务而列清单。  
- **随机探索（Random Exploration）**：智能体不带明确目标，随意在环境中移动、交互，收集多样化的经验。相当于孩子在玩具房里随手玩耍。  
- **弹性权重合并（EWC）**：一种防止模型在新任务上微调时忘记旧知识的技术，通过在重要参数上加惩罚来“记住”过去。可以比作在学习新菜谱时不把老菜谱全忘掉。  
- **低秩适配器（LoRA）**：在大模型上插入小规模的可训练矩阵，只调节少量参数即可实现微调，效率高且不破坏原模型。类似于给原车装一个轻量的外挂引擎。

### 核心创新点
1. **把语言模型和世界模型直接对接 → 在 VirtualHome 中让语言模型控制具身智能体 → 语言模型获得了真实的动作-后果对**。以前的做法只在文字上做推理，这一步让模型“亲自”体验了物理世界的因果链，从而在物体永久性、路径追踪等任务上表现大幅提升。  
2. **双轨经验采集（目标导向 + 随机探索） → 既有结构化的任务完成数据，又有多样化的自由交互数据 → 训练集覆盖了从精细计划到常识性操作的全谱**。单一的任务数据往往只能教会模型完成特定目标，而随机探索补足了模型对环境的广泛感知。  
3. **在微调时加入 EWC + LoRA → 只在不重要的权重上做大幅更新，同时通过低秩适配器保持高效训练 → 语言模型的通用语言能力几乎不被削弱**。这解决了“强化具身知识会导致语言能力退化”的担忧，使得模型在新任务上仍然保持原有的语言优势。  
4. **跨任务评估 → 在 18 项下游任务上整体提升 64.28% → 小模型（1.3B/6B/13B）甚至匹配或超越更大的模型（如 ChatGPT）**。这证明了具身经验的增益不是局部的，而是能够普遍迁移到多种语言推理场景。

### 方法详解
**整体框架**  
这篇论文的流程可以概括为三步：① 在 VirtualHome 中部署一个具身智能体；② 让智能体通过两种策略收集大量交互轨迹；③ 用这些轨迹对预训练语言模型进行受控微调，同时使用 EWC 与 LoRA 保持原有能力。整体思路是“让语言模型在虚拟世界里上课”，而不是单纯在文字上补课。

**步骤 1：具身智能体与世界模型的对接**  
- 选用 VirtualHome 作为模拟环境，因为它提供了丰富的家庭物体和可组合的高层动作（如 `walkTo`, `pickUp`, `open`）。  
- 将语言模型的输出映射为这些高层动作的指令序列。具体做法是把模型的文本生成视作“动作脚本”，再交给 VirtualHome 的执行器。相当于把语言模型当成“指挥官”，模拟器是“执行部队”。  

**步骤 2：经验采集**  
- **目标导向规划**：预设一批日常任务（如“把水果放进冰箱”），使用传统的任务规划器生成完成这些任务的动作序列。语言模型在执行时会观察每一步的状态变化（物体位置、是否成功），并记录成 `(状态, 动作, 下一个状态)` 的三元组。  
- **随机探索**：让智能体在没有明确目标的情况下随机选择可执行的动作，产生大量噪声但多样的交互。这里的关键是覆盖那些在任务规划中很少出现的边缘情形（比如把杯子掉在地上后滚动的轨迹）。  
- 两类轨迹合并后形成一个大规模的“具身经验库”，每条记录都包含自然语言描述（模型生成的指令）和对应的环境反馈。  

**步骤 3：受控微调**  
- **LoRA 适配器**：在语言模型的每层注意力和前馈网络中插入低秩矩阵，只训练这些小矩阵，保持主干权重不动。这样即使是 13B 参数的模型，也只需要几百万可调参数，训练成本大幅降低。  
- **EWC 正则化**：在微调目标函数中加入对重要权重的惩罚项。重要性通过 Fisher 信息矩阵在原始语言模型上预先计算得到，确保模型在学习具身任务时不忘记原有的语言知识。  
- **训练目标**：让模型在给定当前状态的文字描述后，预测下一个最合适的动作指令。等价于让模型学会“从语言到行动”的映射，同时保持对纯文本任务的表现。  

**最巧妙的点**  
- 将 **EWC** 与 **LoRA** 结合使用：EWC 负责“记忆”，LoRA 负责“高效学习”。单独使用任意一种要么会导致语言能力下降，要么训练成本过高。两者的协同让模型在具身知识上快速提升，却几乎不牺牲原有的语言通用性。  
- **双轨经验** 的设计：目标导向提供结构化的高质量标签，随机探索提供覆盖性和鲁棒性。两者相辅相成，使得模型在“计划”和“感知”两方面都得到锻炼。

### 实验与效果
- **评测平台**：作者在 18 项下游任务上做对比，这些任务涵盖了物体永久性推理、路径追踪、日常活动规划、常识问答等多种需要具身常识的场景。  
- **基线对比**：与未经过具身微调的原始语言模型、以及仅使用文字常识库微调的模型相比，本文方法在平均上提升了 **64.28%**。  
- **规模效应**：1.3B、6B、13B 参数的模型经本方法强化后，性能能够追平甚至超越更大规模的模型（如 ChatGPT），说明具身经验的增益在一定程度上可以抵消模型参数的劣势。  
- **消融实验**：论文分别去掉 EWC、LoRA、随机探索或目标导向规划，发现每一项的缺失都会导致整体性能下降 10%~20% 不等，尤其是去掉 EWC 时语言任务的表现出现明显退化。  
- **局限性**：实验全部基于 VirtualHome 这一模拟环境，真实世界的噪声、物理细节（摩擦、光照）未被覆盖；此外，作者未公开具体的任务列表和每项任务的提升幅度，外部复现仍有一定难度。

### 影响与延伸思考
这篇工作打开了“语言模型+具身模拟” 的新思路，随后出现了多篇把机器人仿真、游戏引擎或真实机器人数据直接用于语言模型微调的研究。例如，利用 Minecraft、Habitat 或真实家庭机器人收集的交互日志来提升模型的空间推理能力。未来的方向可能包括：① 将多模态感知（视觉、触觉）加入经验库，让模型在“看见+行动”层面更完整；② 探索跨模拟迁移，验证在一个仿真环境学到的知识能否直接在真实机器人上使用；③ 结合自监督的世界模型学习，让语言模型在没有人工任务标注的情况下自行发现有价值的交互模式。对想深入的读者，可以关注近期的 “Embodied Language Models” 系列论文以及 OpenAI、DeepMind 在强化学习与语言模型融合方面的最新报告。

### 一句话记住它
让语言模型在虚拟世界里“亲自”行动，通过低成本微调和记忆保护，瞬间把“会说话”变成“会做事”。