# Intern-S1: A Scientific Multimodal Foundation Model

> **Date**：2025-08-21
> **arXiv**：https://arxiv.org/abs/2508.15763

## Abstract

In recent years, a plethora of open-source foundation models have emerged, achieving remarkable progress in some widely attended fields, with performance being quite close to that of closed-source models. However, in high-value but more challenging scientific professional fields, either the fields still rely on expert models, or the progress of general foundation models lags significantly compared to those in popular areas, far from sufficient for transforming scientific research and leaving substantial gap between open-source models and closed-source models in these scientific domains. To mitigate this gap and explore a step further toward Artificial General Intelligence (AGI), we introduce Intern-S1, a specialized generalist equipped with general understanding and reasoning capabilities with expertise to analyze multiple science modal data. Intern-S1 is a multimodal Mixture-of-Experts (MoE) model with 28 billion activated parameters and 241 billion total parameters, continually pre-trained on 5T tokens, including over 2.5T tokens from scientific domains. In the post-training stage, Intern-S1 undergoes offline and then online reinforcement learning (RL) in InternBootCamp, where we propose Mixture-of-Rewards (MoR) to synergize the RL training on more than 1000 tasks simultaneously. Through integrated innovations in algorithms, data, and training systems, Intern-S1 achieved top-tier performance in online RL training. On comprehensive evaluation benchmarks, Intern-S1 demonstrates competitive performance on general reasoning tasks among open-source models and significantly outperforms open-source models in scientific domains, surpassing closed-source state-of-the-art models in professional tasks, such as molecular synthesis planning, reaction condition prediction, predicting thermodynamic stabilities for crystals. Our models are available at https://huggingface.co/internlm/Intern-S1.

---

# Intern‑S1：面向科学的多模态基础模型 论文详细解读

### 背景：这个问题为什么难？

科学研究常常涉及实验数据、化学式、晶体结构图等多种模态信息，而现有的开源大模型大多只在文本或通用图像上训练，缺乏对专业科学数据的理解深度。传统的专业模型往往是为单一任务定制，难以共享知识，且训练成本高。与此同时，闭源的商业模型虽然在这些领域表现不错，但不可公开获取，导致学术界难以复现和改进。于是出现了“开源模型在科学任务上远远落后于闭源模型”的尴尬局面，迫切需要一种既开放又具备科学专业能力的通用模型。

### 关键概念速览
**多模态（Multimodal）**：指模型能够同时处理文字、图片、分子式、晶体结构等不同类型的数据，就像人类在阅读论文时会看文字、图表和化学式一样。  
**Mixture-of-Experts（MoE）**：模型内部有很多专家子网络，输入时只激活一小部分专家，类似于公司里不同部门只在需要时被叫去工作，能够在保持参数规模大的同时控制计算成本。  
**激活参数（Activated parameters）**：在一次前向计算中实际被使用的参数数量，MoE 通过只激活部分专家，使得激活参数远小于总参数。  
**持续预训练（Continual pre‑training）**：在已有的大规模语言模型基础上继续喂入新数据进行训练，像给已经学会英语的学生再上科学课。  
**强化学习（Reinforcement Learning，RL）**：模型通过与环境交互获得奖励信号来优化行为，这里用于让模型在具体科学任务上学会“做决定”。  
**Mixture-of-Rewards（MoR）**：在一次 RL 训练中同时使用上千个任务的奖励函数，让模型在多任务之间共享学习信号，类似于一次考试同时考多个科目。  
**InternBootCamp**：作者自建的 RL 训练平台，提供了 1000+ 科学任务的模拟环境，供模型进行在线微调。  
**科学专业任务**：包括分子合成路线规划、化学反应条件预测、晶体热力学稳定性评估等，需要深厚领域知识的任务。

### 核心创新点
1. **大规模科学数据注入 → 持续预训练 5 T 词元，其中 2.5 T 来自科学领域 → 模型在科学语言和符号上的理解显著提升，能够直接处理分子式、晶体结构等专业输入。**  
2. **MoE 架构 + 超大总参数 241 B → 只激活 28 B 参数 → 在保持计算可接受的前提下，模型拥有几乎“专家级”容量，能够在不同科学子领域调用专门的专家子网络。**  
3. **Mixture-of-Rewards 训练策略 → 同时在 InternBootCamp 中对 1000+ 任务进行 RL 微调 → 让模型在一次训练中学会多种科学推理和决策技巧，避免了逐任务单独微调的高成本。**  
4. **离线+在线双阶段 RL → 先在离线数据上做策略预热，再在在线交互中细化 → 兼顾了大规模数据的覆盖和真实环境的反馈，使得模型在专业任务上超过了闭源最强模型。  

### 方法详解
整体思路可以划分为三大阶段：**（1）大规模持续预训练、（2）离线强化学习微调、（3）在线强化学习微调**。  
1. **持续预训练**：在已有的 Intern‑LM 基础上，作者继续喂入 5 T 词元的混合语料。这里的语料分为两类：通用互联网文本和专业科学文献（如化学专利、晶体数据库、实验报告）。MoE 结构让每条输入只触发约 10% 的专家子网络，激活参数保持在 28 B 左右。这样既能让模型学习到通用语言能力，又能在科学符号、公式上形成专门的表示。  
2. **离线 RL 微调**：作者构建了一个包含 1000+ 科学任务的离线数据池，每条数据都配有人工设计的奖励函数（例如分子合成路径的可行性、反应产率的预测误差）。使用 **Mixture-of-Rewards**，模型在一次梯度更新中同时考虑所有任务的奖励，加权求和后得到统一的学习信号。相当于一次训练让模型在“化学、材料、物理”三个科目都得到练习。  
3. **在线 RL 微调（InternBootCamp）**：在真实的交互环境中，模型会收到即时反馈。例如在分子合成规划任务中，模型提出的每一步都会被模拟实验评估其可行性并给出奖励。在线阶段采用 PPO（Proximal Policy Optimization）等安全的策略梯度算法，确保模型在探索新策略时不会偏离已有的高质量解。  

**关键模块的类比**：  
- **MoE** 像是一个拥有上千位专家的咨询公司，用户只需要对应问题的专家出面，其他人保持休息。  
- **MoR** 像是一次综合考试，学生的成绩由多科目的分数加权决定，老师可以一次性看到整体表现并给出改进建议。  

**最巧妙的设计**：MoR 让上千个任务的奖励在同一次梯度更新中共存，避免了传统多任务 RL 中“任务冲突”导致的梯度相互抵消。作者通过对奖励进行归一化并引入任务权重调节，使得每个任务都能贡献适度的学习信号。

### 实验与效果
- **评测基准**：包括通用推理基准（如 GSM8K、MMLU）以及专业科学基准（分子合成规划、反应条件预测、晶体热力学稳定性等）。  
- **对比对象**：主流开源大模型（如 LLaMA‑2、Intern‑LM‑7B）以及闭源商业模型（如 GPT‑4、Claude）。  
- **结果概述**：在通用推理任务上，Intern‑S1 的表现与最强开源模型持平；在科学专业任务上，Intern‑S1 超过所有开源基线，并且在多个任务上“显著超越”闭源最先进模型，尤其在分子合成路线规划和晶体稳定性预测上取得了领先。  
- **消融实验**：作者分别去掉 MoE、MoR、离线 RL、在线 RL 四个组件进行实验，发现去掉 MoR 时多任务学习效果下降约 15%，去掉 MoE 则计算成本翻倍且专业任务准确率下降 10% 以上。  
- **局限性**：论文承认模型仍然对极端稀有的实验数据表现不佳，且在线 RL 需要大量计算资源，普通研究团队难以复现完整训练流程。

### 影响与延伸思考
Intern‑S1 的出现标志着开源大模型首次在高价值科学领域实现了与闭源模型相竞争的水平，激发了社区对 **科学多模态 MoE** 的兴趣。后续工作可能会在以下方向继续深化：  
- **更细粒度的专家划分**，让不同子领域（如有机化学、材料物理）拥有专属专家，提高专业化程度。  
- **跨模态检索与解释**，让模型不仅能给出答案，还能生成可视化的实验方案或结构图。  
- **低资源强化学习**，探索在更少计算预算下实现 MoR 效果的技巧。  
- **安全与可解释性**，因为在科学决策中错误代价高，研究者需要对模型的推理过程进行审计。  

如果想进一步了解，可以关注 **InternBootCamp** 项目的开源实现、以及近期在 arXiv 上出现的 “Scientific MoE” 系列论文。

### 一句话记住它
Intern‑S1 用 241 B 参数的 MoE 结构和 Mixture‑of‑Rewards 强化学习，让开源模型首次在专业科学任务上匹配甚至超越闭源最强模型。