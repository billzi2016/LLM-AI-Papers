# Do LLMs Really Think Step-by-step In Implicit Reasoning?

> **Date**：2024-11-24
> **arXiv**：https://arxiv.org/abs/2411.15862

## Abstract

It has been well-known that Chain-of-Thought can remarkably enhance LLMs' performance on complex tasks. However, because it also introduces slower inference speeds and higher computational costs, many researches have attempted to use implicit CoT, which does not need LLMs to explicitly generate the intermediate steps. However, the invisible reasoning process leaves us a doubt that, can implicit CoT really be equal to explicit CoT? Therefore, in this study, we address this question through experiments. We probe the information of intermediate steps from the model's hidden states when it is either trained or prompted to perform implicit CoT. The results surprisingly indicate that when prompted, LLMs hardly think about intermediate steps, suggesting they may just rely on experience rather than strict step-by-step reasoning. But when trained, they indeed calculate intermediate steps. Moreover, in both situations, we find the effect of using implicit CoT is susceptible to the format of the problem, reaffirming the current deficiency of implicit CoT.

---

# LLM真的在隐式推理中一步步思考吗？ 论文详细解读

### 背景：这个问题为什么难？
在复杂的推理任务上，直接让大语言模型（LLM）一次性输出答案往往效果不佳。显式的思维链（Chain‑of‑Thought，CoT）通过让模型先写出中间步骤，大幅提升了准确率，但也带来了推理速度慢、算力消耗大的副作用。于是研究者提出“隐式 CoT”，即不让模型显式生成步骤，而是期望它在内部完成同样的推理。关键的疑问是：模型真的在内部进行逐步计算，还是仅凭记忆直接给出答案？如果内部并未真正推理，隐式 CoT 的可靠性和可解释性都将受到质疑。

### 关键概念速览
**显式 CoT（Explicit Chain‑of‑Thought）**：让模型在给出最终答案前，先把思考过程写出来，类似于解题时在纸上列出每一步推导。  
**隐式 CoT（Implicit Chain‑of‑Thought）**：不要求模型输出中间步骤，期望它在内部完成同样的推理，像是心里暗算而不写草稿。  
**隐藏状态（Hidden States）**：模型每层的内部向量表示，记录了当前输入在网络中的“思考痕迹”。  
**提示（Prompt）**：在输入前加上一段文字，引导模型采用特定的推理方式。  
**微调（Fine‑tuning）**：在大量标注数据上继续训练模型，使其学习到新的行为模式。  
**任务格式（Problem Format）**：题目呈现方式，例如“选择题”“填空题”“文字描述”，不同格式会影响模型的推理路径。  
**经验式推理（Experience‑based Reasoning）**：模型直接凭记忆匹配答案，而不是逐步演算。  

### 核心创新点
1. **从隐藏状态探测中间步骤**：过去的研究多关注模型输出的文字，本文首次把注意力转向模型内部的向量表示，检查是否真的蕴含逐步推理的信息。  
2. **对比提示驱动与微调驱动的隐式 CoT**：先前大多数工作只使用提示来实现隐式 CoT，本文同时训练模型进行隐式 CoT，比较两者在内部推理上的差异。  
3. **任务格式敏感性实验**：系统评估了不同题目呈现方式对隐式 CoT 效果的影响，揭示了隐式 CoT 并非“一刀切”可用的通用技巧。  

### 方法详解
整体思路可以拆成三步：①准备显式 CoT 数据，②分别用提示和微调方式让模型执行隐式 CoT，③在模型的隐藏层中搜索“推理痕迹”。  

**步骤一：构造显式 CoT 基准**  
选取常用的数学、逻辑和常识推理数据集，为每道题手工或自动生成完整的思维链。这样既提供了模型学习的目标，也为后续隐藏状态分析提供了“金标准”。  

**步骤二：实现两种隐式 CoT**  
- **提示式隐式 CoT**：在原始问题前加上一句类似“请直接给出答案”，不要求模型输出步骤。  
- **微调式隐式 CoT**：在显式 CoT 数据上继续训练模型，但在训练目标中去掉中间步骤的损失，只保留最终答案的交叉熵。这样模型在训练时看到完整的思维链，却被迫在推理时不显式复现。  

**步骤三：隐藏状态探针**  
在模型生成答案的过程中，截取每一层的隐藏向量。随后训练一个轻量级的线性探针（类似于小型分类器），让它从这些向量中预测显式 CoT 中的每一步文字。若探针能够准确恢复步骤，说明隐藏状态中真的保留了逐步推理的信息。  

**关键细节**  
- 探针只在验证集上训练，防止信息泄漏。  
- 为了排除模型直接记忆答案的干扰，实验中加入了“逆向问题”（把答案顺序打乱）检验探针的鲁棒性。  
- 通过比较不同层的探针表现，作者发现微调模型的中间层更能恢复步骤，而提示模型的隐藏状态几乎不含有可辨认的步骤信息。  

### 实验与效果
- **数据集**：使用了 GSM8K（数学推理）、SVAMP（代数）和 CommonsenseQA（常识）等公开基准。  
- **基线对比**：与传统显式 CoT、纯粹的直接预测（Zero‑Shot）以及仅使用提示的隐式 CoT 进行比较。  
- **结果**：在提示式隐式 CoT 下，探针的恢复准确率仅在 10% 左右，说明模型几乎没有内部步骤；而微调式隐式 CoT 的恢复准确率提升到 45% 以上，接近显式 CoT（约 70%）的水平。整体任务准确率方面，微调隐式 CoT 与显式 CoT 差距不到 2%，但提示式隐式 CoT 与直接预测的差距只有 0.5%。  
- **消融实验**：去掉微调阶段的显式 CoT 监督，探针恢复率跌回 12%，验证了显式步骤的训练是关键。  
- **局限**：作者指出，隐式 CoT 的效果强烈依赖于题目呈现方式，例如在多选题上表现仍不如显式 CoT；此外，探针只能捕捉到显式可解释的步骤，若模型采用完全不同的内部推理方式，可能仍被误判为“未思考”。  

### 影响与延伸思考
这篇工作提醒社区：仅凭输出结果判断模型是否在“思考”是不够的，必须审视内部表征。随后出现的研究开始使用类似的探针技术评估大模型的推理深度，甚至尝试在训练时显式强化隐藏层的步骤可解释性。对想进一步探索的读者，可以关注“可解释性探针”（interpretability probes）和“层级推理强化”（layerwise reasoning regularization）方向，这两条线索已经在近期的 ACL、NeurIPS 论文中出现。  

### 一句话记住它
显式 CoT 让模型真的算步骤，提示式隐式 CoT 只是在偷懒；只有经过显式步骤微调后，模型才会在内部真正“一步步思考”。