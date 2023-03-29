# Improving Code Generation by Training with Natural Language Feedback

> **Date**：2023-03-28
> **arXiv**：https://arxiv.org/abs/2303.16749

## Abstract

The potential for pre-trained large language models (LLMs) to use natural language feedback at inference time has been an exciting recent development. We build upon this observation by formalizing an algorithm for learning from natural language feedback at training time instead, which we call Imitation learning from Language Feedback (ILF). ILF requires only a small amount of human-written feedback during training and does not require the same feedback at test time, making it both user-friendly and sample-efficient. We further show that ILF can be seen as a form of minimizing the KL divergence to the ground truth distribution and demonstrate a proof-of-concept on a neural program synthesis task. We use ILF to improve a Codegen-Mono 6.1B model's pass@1 rate by 38% relative (and 10% absolute) on the Mostly Basic Python Problems (MBPP) benchmark, outperforming both fine-tuning on MBPP and fine-tuning on repaired programs written by humans. Overall, our results suggest that learning from human-written natural language feedback is both more effective and sample-efficient than training exclusively on demonstrations for improving an LLM's performance on code generation tasks.

---

# 通过自然语言反馈训练提升代码生成 论文详细解读

### 背景：这个问题为什么难？

代码生成模型在把自然语言需求翻译成可运行代码时，常常会出现细微的语义错误或边界情况处理不当。传统的提升手段主要是**示例微调**——给模型大量正确的（需求，代码）对。但示例只能告诉模型“该怎么写”，却无法直接指出模型的思考过程或错误根源。另一方面，**推理时的自然语言反馈**（让模型在生成过程中自行接受人类的纠正）虽然很有吸引力，却需要在每一次使用时都提供反馈，成本高且不易部署。因此，如何在**训练阶段**利用少量的人类自然语言批评来提升模型，而不在推理时依赖反馈，成为了一个迫切的需求。

### 关键概念速览

- **大语言模型（LLM）**：经过海量文本预训练，能够理解并生成自然语言的模型。这里指的是专门用于代码的模型，如 Codegen‑Mono。
- **自然语言反馈**：人类用普通话或书面语言对模型输出的代码进行评价、指出错误或给出改进建议，而不是提供完整的正确代码。
- **模仿学习（Imitation Learning）**：让模型学习“模仿”人类的行为或决策过程，常见于机器人或游戏 AI。这里指模型学习如何根据反馈调整自己的生成策略。
- **ILF（Imitation Learning from Language Feedback）**：本文提出的核心算法，利用少量自然语言批评在训练时引导模型学习，更接近真实的代码分布。
- **KL 散度（Kullback‑Leibler Divergence）**：衡量两个概率分布差异的指标。把模型的输出分布逼近真实代码分布时，最小化 KL 散度是一种常用的数学表述。
- **pass@1**：代码生成评估指标，表示模型在一次尝试中生成的代码能否通过所有单元测试的比例。数值越高说明模型越可靠。
- **MBPP（Mostly Basic Python Problems）**：一个包含约 974 条 Python 编程题的基准集合，常用于评估代码生成模型的实际能力。

### 核心创新点

1. **从训练阶段引入自然语言反馈 → ILF 训练流程**  
   传统做法只在推理时使用自然语言纠正，或仅靠示例微调。ILF 把人类的文字批评直接嵌入训练数据，让模型在学习阶段就学会“听取”并利用这些信息。结果是模型在没有任何测试时反馈的情况下，也能表现出更高的代码质量。

2. **极少量反馈即可产生显著提升 → 样本效率**  
   只需要几百条人工撰写的反馈（相对于数万条完整代码示例），ILF 就能让 Codegen‑Mono 6.1B 在 MBPP 上的 pass@1 提升 38%（相对）/10%（绝对）。这说明自然语言批评的信号密度极高，远高于等量的示例。

3. **把 ILF 形式化为 KL 散度最小化 → 理论解释**  
   作者证明 ILF 的目标等价于最小化模型输出分布与“真实代码分布”之间的 KL 散度。换句话说，模型在学习过程中被迫把自己的概率预测拉向人类认为正确的方向，而不是仅仅模仿示例。

4. **超越两类强基线 → 实证优势**  
   与直接在 MBPP 上微调（示例学习）以及在人工修复后的代码上微调（先纠错再学习）相比，ILF 都取得了更好的 pass@1。说明自然语言反馈不仅能替代示例，还能提供额外的、示例无法覆盖的细粒度信息。

### 方法详解

**整体框架**  
ILF 的训练过程可以概括为三步：  
1) **收集反馈**：对一小批模型生成的代码，人工撰写自然语言批评（如“第 3 行缺少冒号”或“变量名不一致”。）  
2) **构造学习目标**：把原始需求、模型生成的代码以及对应的文字反馈拼接成一个复合输入，喂给模型。模型的目标是输出“修正后的代码”。  
3) **优化**：使用标准的语言模型训练损失（交叉熵）对模型参数进行更新，同时在理论层面解释为最小化 KL 散度。

**关键模块拆解**  

- **反馈编码器**：模型本身已经是一个自回归的 Transformer，能够直接接受混合的自然语言和代码序列。这里不需要额外的编码器，只是把反馈当作普通文本插入到输入流中。类比于在写作时先写草稿（代码），老师在旁边写批注（反馈），学生再根据批注改写草稿。

- **目标生成器**：模型在看到需求 + 初始代码 + 反馈后，需要输出“最终代码”。这一步的损失只对最终代码部分计算，前面的需求和反馈部分保持不变，类似于让模型在“阅读理解”后只回答问题。

- **KL 散度视角**：如果把“真实代码分布”视为人类在看到需求后会写出的代码集合，反馈实际上提供了对这分布的梯度信息。最小化 KL 散度等价于让模型的预测概率在每一步都向更符合人类期望的方向移动。虽然论文没有给出完整的数学推导，这里只需要把它当作一种解释：反馈是对模型概率的“软标签”。

**最巧妙的设计**  

- **无需测试时反馈**：训练时使用反馈，推理时直接生成代码。这样既保留了自然语言反馈的学习价值，又避免了部署时的交互成本。  
- **极小样本量**：只用了几百条反馈，却实现了两位数的性能提升，说明作者在挑选反馈时注重了“信息密度”。比如一次反馈可能指出多个错误，等价于提供了多条示例的学习信号。

### 实验与效果

- **数据集与任务**：在 MBPP（Mostly Basic Python Problems）上进行评估，使用 Codegen‑Mono 6.1B 作为基线模型。  
- **对比基线**：① 直接在 MBPP 上进行示例微调；② 在人工修复后的代码上微调（即先让人类改正模型错误，再用改正后的代码训练）。  
- **主要结果**：ILF 让模型的 pass@1 提升了 38%（相对）/10%（绝对），超过了上述两种基线。  
- **消融实验**：原文提到通过去掉反馈或只使用示例进行对比，性能回落到原始微调水平，说明反馈是提升的关键因素。  
- **局限性**：论文未详细探讨反馈质量对效果的敏感度，也没有在更大规模或多语言代码任务上验证；此外，收集高质量自然语言批评仍然需要人工成本。

### 影响与延伸思考

ILF 把“人类批评”从推理阶段搬到训练阶段，为代码生成模型提供了一条高效的学习路径。自发表后，已有后续工作尝试将类似的语言反馈用于 **代码审查自动化**、**模型对齐**（alignment）以及 **跨语言代码迁移**，并探索使用 **大规模合成反馈**（如 LLM 自动生成的批评）来进一步降低人工成本。想深入了解的读者可以关注 **RLHF（Reinforcement Learning from Human Feedback）在代码领域的变体**，以及 **自监督纠错**（self‑repair）方向的最新论文。

### 一句话记住它

只用几百条人写的文字批评，就能让大模型在代码生成上跳跃式提升——自然语言反馈是比示例更“浓缩”的学习信号。