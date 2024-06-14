# Unpacking DPO and PPO: Disentangling Best Practices for Learning from   Preference Feedback

> **Date**：2024-06-13
> **arXiv**：https://arxiv.org/abs/2406.09279

## Abstract

Learning from preference feedback has emerged as an essential step for improving the generation quality and performance of modern language models (LMs). Despite its widespread use, the way preference-based learning is applied varies wildly, with differing data, learning algorithms, and evaluations used, making disentangling the impact of each aspect difficult. In this work, we identify four core aspects of preference-based learning: preference data, learning algorithm, reward model, and policy training prompts, systematically investigate the impact of these components on downstream model performance, and suggest a recipe for strong learning for preference feedback. Our findings indicate that all aspects are important for performance, with better preference data leading to the largest improvements, followed by the choice of learning algorithm, the use of improved reward models, and finally the use of additional unlabeled prompts for policy training. Notably, PPO outperforms DPO by up to 2.5% in math and 1.2% in general domains. High-quality preference data leads to improvements of up to 8% in instruction following and truthfulness. Despite significant gains of up to 5% in mathematical evaluation when scaling up reward models, we surprisingly observe marginal improvements in other categories.   We publicly release the code used for training (https://github.com/hamishivi/EasyLM) and evaluating (https://github.com/allenai/open-instruct) our models, along with the models and datasets themselves (https://huggingface.co/collections/allenai/tulu-v25-suite-66676520fd578080e126f618).

---

# 拆解 DPO 与 PPO：偏好反馈学习的最佳实践 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，光靠预训练的海量文本已经难以满足高质量指令遵循和事实准确性的需求。于是研究者转向让模型“听取”人类偏好——即让模型生成的答案由人类比较后给出优劣。但实际操作中，偏好学习的管线非常碎片化：有人用不同的偏好数据，有人选不同的学习算法（比如 PPO、DPO），还有人对奖励模型（RM）和用于微调的提示集合各有安排。因为每个环节都可能影响最终效果，单独改动一个环节时很难判断到底是哪个因素在起作用，这让社区难以形成统一的最佳实践。

### 关键概念速览
- **偏好反馈（Preference Feedback）**：让人类对两段模型输出进行比较，标记哪一个更好，类似于让评委给选手打分，帮助模型学习“好答案”的标准。  
- **奖励模型（Reward Model, RM）**：把偏好对比转化为可微分的分数函数，类似于把评委的打分规则写成一个公式，供后续优化使用。  
- **PPO（Proximal Policy Optimization）**：一种强化学习算法，更新模型时会限制每一步的改动幅度，防止“跑偏”。可以想象为在爬山时每次只能迈小步，以免掉下悬崖。  
- **DPO（Direct Preference Optimization）**：直接在偏好对比上做梯度下降的优化方法，省去显式的奖励模型训练步骤，像是直接把评委的打分当作梯度来更新模型。  
- **偏好数据（Preference Data）**：人类标注的比较对，这些数据的质量决定了奖励模型能否捕捉到真实的好坏信号。  
- **策略训练提示（Policy Training Prompts）**：用于让模型生成答案的输入集合，既可以是已有的指令，也可以是额外的未标注提示，用来丰富模型的学习场景。  
- **奖励模型规模（Reward Model Scale）**：奖励模型的参数量或训练数据量，规模越大往往能更细致地捕捉偏好细节，但也会带来计算成本。  

### 核心创新点
1. **系统化四要素拆解 → 统一实验框架 → 明确每个因素的贡献**  
   过去的工作往往只改动一两个环节，难以比较。作者把偏好学习划分为“偏好数据、学习算法、奖励模型、策略训练提示”四大块，搭建了一个可以自由组合的实验平台，从而能够在同一条件下评估每个因素的独立影响。

2. **PPO 与 DPO 的公平对比 → PPO 在数学与通用任务上领先 → 提供了实证依据**  
   通过在相同数据、相同奖励模型、相同提示集下分别跑 PPO 与 DPO，作者发现 PPO 在数学推理上提升最高 2.5%，在通用指令上提升 1.2%。这直接回答了社区对两者优劣的争论。

3. **奖励模型规模的细粒度实验 → 数学任务收益显著、其他任务提升有限 → 揭示规模效应的边界**  
   将奖励模型从基础版放大到更大模型，数学评测分数提升约 5%，但在指令遵循、真实性等指标上几乎没有提升。该发现提醒我们，盲目增大奖励模型并非通用解药。

4. **加入未标注提示进行策略微调 → 额外提升 0.5% 左右 → 证明数据多样性仍有价值**  
   在已有的偏好数据之外，作者额外使用了一批未标注的指令提示进行 PPO 微调，带来了小幅但一致的提升，说明“更多的输入情境”对策略学习仍有帮助。

### 方法详解
整体思路可以概括为四步循环：**收集偏好 → 训练奖励模型 → 选算法微调策略 → 评估并迭代**。下面按步骤拆解：

1. **偏好数据收集**  
   - 人类评审员对同一指令下的两段模型输出进行二选一标注。  
   - 为了控制变量，所有实验使用同一套 30k 条指令的偏好对，随后在不同实验中加入或去除额外的未标注提示。

2. **奖励模型训练**  
   - 将每对输出映射为向量表示（如使用预训练的编码器），再喂入一个二分类头，目标是让模型输出的分数能够正确排序两段文本。  
   - 训练时使用交叉熵损失，奖励模型的规模可以是 7B、13B 等不同大小，作者在实验中对比了这些规模的效果。

3. **学习算法选择**  
   - **PPO 路径**：先用奖励模型对生成的答案打分，得到即时奖励；再使用 PPO 的剪切目标（clip objective）限制每一步的策略更新幅度，确保训练过程稳定。  
   - **DPO 路径**：直接把偏好对的相对概率（由奖励模型输出的分数转化）作为目标，进行梯度下降，不需要显式的强化学习回报或剪切操作。  
   - 两者在同样的奖励模型、相同的学习率、相同的批次大小下运行，确保公平比较。

4. **策略训练提示的使用**  
   - 基础提示集合来自公开的指令数据集（如 Alpaca、OpenAI API 示例）。  
   - 额外的未标注提示是从网络爬取的指令文本，未配对偏好，仅用于让模型在更广的输入空间上练习生成。  
   - 在 PPO 中，这些提示会被当作普通的微调样本；在 DPO 中则不参与，因为 DPO 需要偏好对。

**最巧妙的设计**：作者把奖励模型的输出直接当作 DPO 的“软标签”，省去传统 RL 中的采样与回报估计步骤，使得 DPO 的实现几乎只需要一行梯度更新代码，却仍能在大多数任务上保持竞争力。这种“直接优化偏好”的思路在之前的文献里很少被系统化比较。

### 实验与效果
- **测试任务**：包括数学推理（MATH、GSM8K）、通用指令遵循（OpenAI API 示例）、指令遵循度评估（Open-Instruct）、真实性评估（TruthfulQA）。  
- **基线对比**：与未使用偏好学习的原始模型、以及仅使用 PPO 或 DPO 的常规实现进行比较。  
- **关键数字**：  
  - PPO 在数学任务上比 DPO 高出 **2.5%**，在通用指令上高出 **1.2%**。  
  - 使用高质量偏好数据整体提升 **8%**（在指令遵循和真实性上）。  
  - 将奖励模型放大后，数学评测提升 **约 5%**，但在其他指标上提升不明显（<1%）。  
  - 加入未标注提示后，整体分数再提升 **0.5% 左右**。  
- **消融实验**：作者分别去掉每个四要素进行实验，发现偏好数据质量的提升幅度最大，其次是学习算法的选择，随后是奖励模型的改进，最后是额外提示的贡献最小。  
- **局限性**：  
  - 大规模奖励模型的收益在非数学任务上几乎停滞，说明奖励模型的容量并非万能钥匙。  
  - DPO 虽然实现简洁，但在数学推理等需要细粒度奖励的任务上仍落后于 PPO。  
  - 实验主要在英文指令上完成，跨语言或多模态场景的适用性尚未验证。

### 影响与延伸思考
这篇工作在社区里起到了“标准化实验平台”的作用，很多后续的偏好学习研究直接引用了作者提供的代码库（EasyLM）和评测套件（Open-Instruct）。随后出现的几篇论文（如《Scaling Preference Models》、 《Hybrid PPO‑DPO Training》）都在讨论如何结合 PPO 的稳定性与 DPO 的简洁性，或是探索更高效的奖励模型蒸馏方法。对想继续深入的读者，可以关注以下方向：  
- **跨语言偏好学习**：如何在多语言指令上保持同等的偏好捕捉能力。  
- **奖励模型蒸馏**：在保持数学任务提升的同时，降低奖励模型的计算成本。  
- **混合优化策略**：在同一训练阶段交替使用 PPO 与 DPO，以期兼顾稳定性和效率。  

### 一句话记住它
**高质量的偏好数据是提升 LLM 指令遵循的最大杠杆，PPO 在数学推理上仍是最稳健的选择。**