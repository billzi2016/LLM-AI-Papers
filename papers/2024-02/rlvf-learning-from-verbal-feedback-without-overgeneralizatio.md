# RLVF: Learning from Verbal Feedback without Overgeneralization

> **Date**：2024-02-16
> **arXiv**：https://arxiv.org/abs/2402.10893

## Abstract

The diversity of contexts in which large language models (LLMs) are deployed requires the ability to modify or customize default model behaviors to incorporate nuanced requirements and preferences. A convenient interface to specify such model adjustments is high-level verbal feedback, such as "Don't use emojis when drafting emails to my boss." However, while writing high-level feedback is far simpler than collecting annotations for reinforcement learning from human feedback (RLHF), we find that simply prompting a model with such feedback leads to overgeneralization of the feedback to contexts where it is not relevant. We study the problem of incorporating verbal feedback without such overgeneralization, inspiring a new method Contextualized Critiques with Constrained Preference Optimization (C3PO). C3PO uses a piece of high-level feedback to generate a small synthetic preference dataset specifying how the feedback should (and should not) be applied. It then fine-tunes the model in accordance with the synthetic preference data while minimizing the divergence from the original model for prompts where the feedback does not apply. Our experimental results indicate that our approach effectively applies verbal feedback to relevant scenarios while preserving existing behaviors for other contexts. For both human- and GPT-4-generated high-level feedback, C3PO effectively adheres to the given feedback comparably to in-context baselines while reducing overgeneralization by 30%.

---

# RLVF：从口头反馈学习而不产生过度概括 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以在聊天、写作、代码等各种场景里直接使用，但不同用户对同一个模型的细节要求往往千差万别。传统的对齐方式——RLHF（基于人类反馈的强化学习）需要大量标注的偏好对，成本高且难以快速迭代。于是人们尝试直接把“别在给老板的邮件里加表情”之类的高层口头反馈塞进提示里，让模型自行遵守。可惜模型往往把这条规则“一刀切”地套用到所有类似的生成任务，导致在不相关情境下也不再使用表情、换行等原本合理的表现，这种现象被称为**过度概括**。因此，如何让模型只在真正需要的上下文里遵守口头反馈，而不破坏已有的通用能力，成为了一个急需解决的难题。

### 关键概念速览

**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似“会说话的百科全书”。  

**口头反馈（Verbal Feedback）**：用户用一句自然语言描述的行为约束或偏好，例如“写给老板的邮件不要出现表情”。它比标注偏好对更易获得。  

**过度概括（Overgeneralization）**：模型把口头反馈错误地推广到所有上下文，导致在不该受约束的场景里也出现不符合预期的行为。可以想象成老师把“一次作业要手写”这条规矩硬套到所有考试、实验报告上。  

**上下文批评（Contextualized Critiques）**：模型在给定反馈的前提下，对特定输入生成“这句话应该遵守反馈”或“这句话不该遵守反馈”的评价，类似审稿人对稿件的正负评语。  

**合成偏好数据（Synthetic Preference Dataset）**：利用上下文批评自动构造的成对示例（好 vs. 坏），用来告诉模型在何种情境下应当遵守或不遵守反馈。  

**受限偏好优化（Constrained Preference Optimization）**：在微调模型时，同时最大化对合成偏好数据的匹配度，并最小化模型在无关上下文上的改变，常用 KL 散度作为约束手段。  

**微调（Fine‑tuning）**：在已有的大模型基础上，用少量新数据继续训练，使模型在特定任务或约束上表现更好。  

**KL 散度约束**：衡量新模型与原模型分布差异的指标，约束它可以防止模型在不相关情境里跑偏。

### 核心创新点

1. **从高层反馈到合成偏好对的桥接**  
   以前的做法直接把口头反馈塞进提示，模型会自行解释并产生过度概括。本文先让模型生成上下文批评，再把这些批评转化为正负示例，形成一个小规模的合成偏好数据集。这样模型得到的是“在这类输入下应该这样，在那类输入下不应该这样”的明确指示，而不是模糊的全局指令。

2. **微调时加入保持原行为的约束**  
   传统微调只追求在新数据上的表现提升，往往会牺牲模型在其他任务上的能力。本文在优化目标里加入 KL 散度约束，强制模型在反馈不适用的上下文上保持与原模型相近的输出分布，从而抑制过度概括。

3. **上下文批评的自动化生成**  
   作者设计了一个“批评生成器”，它接受原始反馈和任意输入，输出“遵守”或“违背”标签。这个生成器本身是基于同一个 LLM，利用 few‑shot 示例即可工作，省去了人工标注偏好对的成本。

4. **对比实验显示显著降低过度概括**  
   在人类和 GPT‑4 生成的口头反馈两类实验中，本文的方法在遵守反馈的准确率上与直接提示基线持平，却把过度概括率降低了约 30%。这表明约束微调真的起到了“只在该出现的地方出现”的作用。

### 方法详解

**整体思路**：把一句高层口头反馈转化为一套“正负对”，再用受限的偏好微调让模型学会在对应情境下遵守，同时在其他情境保持原样。整个流程可以划分为三步：① 反馈解读与批评生成，② 合成偏好对构造，③ 受限偏好微调。

1. **反馈解读与批评生成**  
   - 输入：原始口头反馈 F（如“不要在给老板的邮件里使用表情”）以及一批随机挑选的上下文句子 X₁…Xₙ。  
   - 过程：使用同一 LLM（称为 Critic）在 few‑shot 提示下，对每个 Xᵢ 生成一个二元标签 Yᵢ∈{遵守, 违背}，并给出简短解释。可以把它想象成让模型先扮演“审稿人”，判断每句话是否符合 F。  
   - 输出：一组 (Xᵢ, Yᵢ) 对，形成了对反馈的上下文化解释。

2. **合成偏好对构造**  
   - 将所有标记为“遵守”的句子收集为正例 P，标记为“违背”的句子收集为负例 N。  
   - 随机配对 (p, n) 生成偏好对，意思是模型应该更倾向输出 p 而不是 n。这样得到的偏好数据集规模虽小，却覆盖了反馈适用与不适用的两端。  
   - 关键在于，这一步把抽象的口头规则具体化为可训练的对比信号。

3. **受限偏好微调（C3PO）**  
   - 目标函数由两部分组成：  
     a) **偏好匹配项**：最大化模型对每个 (p, n) 对的相对得分，使模型更倾向生成正例。实现方式类似 DPO（Direct Preference Optimization），即比较正负两句的对数概率差。  
     b) **KL 散度约束**：对所有未出现于偏好对的输入（即大多数普通上下文），计算新模型与原模型输出分布的 KL 散度，并将其限制在一个小阈值 ε 内。这样模型在这些输入上几乎不改变行为。  
   - 训练时交替优化两项，或使用拉格朗日乘子把约束软化为惩罚项。最终得到的模型在遵守反馈的情境下表现更好，而在其他情境上仍保持原有的通用能力。

**最巧妙的点**：把“高层口头反馈”先交给模型自己生成批评，再把批评转化为可监督的偏好对，这一步把人类难以标注的抽象指令变成了机器可以自行产生的训练信号，极大降低了人工成本。

### 实验与效果

- **测试场景**：作者在多个下游任务上评估，包括邮件撰写、社交媒体回复、代码注释等，每个任务都提供了对应的口头反馈（如“不要使用表情”“代码注释要简洁”）。  
- **基线对比**：与直接在提示中加入同样口头反馈的 In‑Context Learning（ICL）基线以及传统的 RLHF 微调基线进行比较。  
- **主要结果**：  
  - 在遵守反馈的准确率上，C3PO 与 ICL 基线持平，说明约束微调没有牺牲对指令的执行力。  
  - 过度概括率下降约 30%，即模型在不应受约束的情境下保持原有行为的比例显著提升。  
  - 对于人类撰写的反馈和 GPT‑4 自动生成的反馈，两者均表现出相似的改进幅度，证明方法对反馈来源不敏感。  
- **消融实验**：去掉 KL 散度约束后，模型在遵守反馈上仍有提升，但过度概括率几乎回到 ICL 水平，说明约束是抑制概括的关键因素。去掉批评生成直接使用随机正负对也会导致性能下降，验证了上下文批评的必要性。  
- **局限性**：论文未在大规模真实用户交互数据上验证，合成偏好对的规模仍受限于批评生成的质量；如果批评生成出现系统性错误，可能会把错误的约束灌入模型。

### 影响与延伸思考

这篇工作打开了“从自然语言指令到可训练信号”的新路径，后续有研究开始探索更复杂的指令（如多步骤工作流）如何通过类似的批评‑偏好‑约束流程转化为微调数据。还有人把 C3PO 的思路搬到多模态模型上，尝试让视觉‑语言系统遵守“不要在广告中出现特定颜色”的口头约束。对想进一步深入的读者，可以关注以下方向：  
- **指令分解与层级反馈**：如何把一条复合指令拆解成多个子反馈并统一优化。  
- **自动批评质量提升**：利用自监督或对抗训练提升批评生成器的可靠性。  
- **跨任务约束共享**：研究在不同任务之间共享合成偏好对，以实现更通用的约束学习。  

### 一句话记住它

把一句高层口头反馈先让模型自评生成正负示例，再用受限偏好微调，让模型只在该出现的地方遵守指令，避免了全局过度概括。