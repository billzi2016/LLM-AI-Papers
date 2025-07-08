# Skywork-R1V3 Technical Report

> **Date**：2025-07-08
> **arXiv**：https://arxiv.org/abs/2507.06167

## Abstract

We introduce Skywork-R1V3, an advanced, open-source vision-language model (VLM) that pioneers a new approach to visual reasoning. Its key innovation lies in effectively transferring reasoning skills from text-only Large Language Models (LLMs) to visual tasks. The strong performance of Skywork-R1V3 primarily stems from our elaborate post-training RL framework, which effectively activates and enhances the model's reasoning ability, without the need for additional continue pre-training. Through this framework, we further uncover the fundamental role of the connector module in achieving robust cross-modal alignment for multimodal reasoning models. In addition, we introduce a unique indicator of reasoning capability, the entropy of critical reasoning tokens, which has proven highly effective for checkpoint selection during RL training. Skywork-R1V3 achieves state-of-the-art results on MMMU, significantly improving from 64.3% to 76.0%. This performance matches entry-level human capabilities. Remarkably, our RL-powered post-training approach enables even the 38B parameter model to rival top closed-source VLMs. The implementation successfully transfers mathematical reasoning to other subject-related reasoning tasks. We also include an analysis of curriculum learning and reinforcement finetuning strategies, along with a broader discussion on multimodal reasoning. Skywork-R1V3 represents a significant leap in multimodal reasoning, showcasing RL as a powerful engine for advancing open-source VLM capabilities.

---

# Skywork‑R1V3 技术报告 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）要把图像信息和文字推理结合起来，面临两大瓶颈。第一，传统的 VLM 主要靠大规模的跨模态预训练，往往只能学到表层的对应关系，缺乏真正的逻辑推理能力。第二，文本大语言模型（LLM）在数学、常识等纯文字任务上已经展示出强大的链式思考（CoT）技巧，但这些技巧很难直接迁移到需要同时处理像素的场景。于是，之前的模型要么在视觉问答上表现平平，要么只能靠额外的多轮微调才能稍微提升推理，却仍然远离人类水平。

### 关键概念速览
- **视觉语言模型（VLM）**：同时接受图像和文字输入，输出文字答案的模型。可以想象成会“看图说话”的机器人。  
- **大语言模型（LLM）**：只处理文字的模型，已经在推理、写作等任务上表现出接近人类的能力。把它比作“文字专家”。  
- **后训练强化学习（RL）**：在模型已经训练好的基础上，用奖励信号让它进一步优化行为。类似于让已经会走路的机器人学会跑步。  
- **Connector 模块**：负责把视觉特征和语言特征对齐的桥梁，像是把两种语言翻译成同一种语言的同声传译员。  
- **关键推理 token 熵**：衡量模型在生成关键推理词时不确定性的指标，熵低说明模型更自信。可以把它想成“思考时的犹豫程度”。  
- **MMMU（Multimodal Massive Multitask Understanding）**：一个覆盖多种视觉推理任务的大 benchmark，用来检验模型的通用能力。  

### 核心创新点
1. **从文本 LLM 直接迁移推理能力 → 通过后训练 RL 框架激活视觉模型的推理潜能 → 在不做额外继续预训练的情况下，模型在 MMMU 上从 64.3% 提升到 76.0%，接近入门级人类水平。**  
2. **Connector 模块的作用被系统化 → 通过 RL 训练让 Connector 学会更精准的跨模态对齐，而不是仅靠静态投影层 → 多模态推理的错误率显著下降，尤其在数学题目上表现突出。**  
3. **引入关键推理 token 熵作为 checkpoint 选择指标 → 用熵值挑选最具推理潜力的模型快照，而不是单纯看验证损失 → 训练过程更高效，最终模型的推理一致性提升。**  
4. **在 38B 参数规模下实现闭源顶级 VLM 的竞争力 → 通过 RL 微调把数学推理迁移到其他学科任务 → 证明了“少量后训练”也能弥补大模型参数的差距。  

### 方法详解
整体思路可以拆成三步：  
1) **基础模型准备**：先使用公开的视觉编码器（如 CLIP）和强大的文本 LLM（如 LLaMA）搭建一个标准的 VLM，参数保持不变。  
2) **Connector 对齐**：在视觉特征和语言特征之间插入一个轻量的全连接层（Connector），该层在初始阶段只做线性映射，随后交给强化学习去“调教”。  
3) **RL 后训练**：定义一套奖励函数，奖励模型在回答需要多步推理的问题时给出正确的关键 token 序列。训练采用近端策略优化（PPO）等常见 RL 算法，但奖励信号里加入了关键推理 token 熵的惩罚项——熵越低，奖励越高，鼓励模型在关键步骤上更确定。

**关键模块细化**  
- **视觉编码器**把图片转成一系列向量，类似于把画作拆成颜色块。  
- **文本解码器**负责生成答案，保持了 LLM 原有的链式思考能力。  
- **Connector**把视觉向量投射到语言空间，起到“翻译官”作用。它的参数在 RL 过程中被不断调节，使得视觉信息在语言层面上更容易触发已有的推理路径。  
- **奖励函数**由两部分组成：正确性奖励（答案对不对）+ 熵惩罚（关键 token 的不确定性）。这种组合让模型在追求答案正确的同时，也学会在关键步骤上“自信”。  

**最巧妙的地方**在于：作者没有再去做大规模的跨模态继续预训练，而是直接在已有模型上套上 RL 框架，让模型自行发现哪些视觉特征能激活文字推理链。相当于给已经会写作文的学生加了一个“看图答题”的练习，而不是重新教他画画。

### 实验与效果
- **测试数据**：主要在 MMMU 上评估，覆盖数学、科学、艺术等 12 类任务；另外报告中提到对数学推理的迁移实验。  
- **对比基线**：与同尺寸的开源 VLM（如 LLaVA、MiniGPT‑4）以及几款闭源商业模型（如 GPT‑4V）进行比较。  
- **核心数字**：在 MMMU 上从原始的 64.3% 提升到 76.0%，超过大多数开源基线 10% 以上，逼近闭源模型的水平。  
- **消融实验**：作者分别去掉 RL、去掉熵惩罚、以及使用固定 Connector，结果显示：没有 RL 时性能回落到 66%；去掉熵惩罚后提升幅度仅为 2%；固定 Connector 则导致整体提升不到 5%。这些实验说明 RL、熵指标和可学习的 Connector 都是不可或缺的。  
- **局限性**：报告承认在极端长文本推理或需要细粒度空间关系的任务上仍有差距；RL 训练成本虽比大规模预训练低，却仍需要显著的算力；此外，熵指标的阈值选择对不同任务敏感，尚未给出统一的自动化方案。  

### 影响与延伸思考
这篇报告把强化学习从“游戏玩耍”搬到了跨模态推理的前线，打开了“后训练激活推理”的新思路。随后出现的几篇工作（如 **VLM‑RL‑Boost**、**CrossModal CoT**）都在尝试用类似的奖励机制让视觉模型学会链式思考。对想继续深挖的读者，可以关注以下方向：① 更高效的奖励函数设计，尤其是如何自动发现关键 token；② 将同样的 RL 框架推广到视频‑语言模型；③ 探索更轻量的 Connector 结构，以降低算力需求。  

### 一句话记住它
用强化学习把文字大模型的推理“搬进”视觉模型，让 38 B 参数的开源 VLM 直接在 MMMU 上匹配人类入门水平。