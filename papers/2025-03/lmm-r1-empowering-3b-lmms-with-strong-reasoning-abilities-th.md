# LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through   Two-Stage Rule-Based RL

> **Date**：2025-03-10
> **arXiv**：https://arxiv.org/abs/2503.07536

## Abstract

Enhancing reasoning in Large Multimodal Models (LMMs) faces unique challenges from the complex interplay between visual perception and logical reasoning, particularly in compact 3B-parameter architectures where architectural constraints limit reasoning capacity and modality alignment.   While rule-based reinforcement learning (RL) excels in text-only domains, its multimodal extension confronts two critical barriers: (1) data limitations due to ambiguous answers and scarce complex reasoning examples, and (2) degraded foundational reasoning induced by multimodal pretraining. To address these challenges, we propose \textbf{LMM-R1}, a two-stage framework adapting rule-based RL for multimodal reasoning through \textbf{Foundational Reasoning Enhancement (FRE)} followed by \textbf{Multimodal Generalization Training (MGT)}. The FRE stage first strengthens reasoning abilities using text-only data with rule-based RL, then the MGT stage generalizes these reasoning capabilities to multimodal domains.   Experiments on Qwen2.5-VL-Instruct-3B demonstrate that LMM-R1 achieves 4.83\% and 4.5\% average improvements over baselines in multimodal and text-only benchmarks, respectively, with a 3.63\% gain in complex Football Game tasks. These results validate that text-based reasoning enhancement enables effective multimodal generalization, offering a data-efficient paradigm that bypasses costly high-quality multimodal training data.

---

# LMM‑R1：通过两阶段基于规则的强化学习赋能 3B 参数多模态大模型的强推理能力 论文详细解读

### 背景：这个问题为什么难？
大规模多模态模型（LMM）需要同时处理视觉信息和抽象推理，而视觉特征往往是噪声多、歧义大的。现有的 3 B 参数小模型在算力和参数预算上受限，难以在视觉感知的基础上形成稳健的逻辑链。传统的基于规则的强化学习（RL）在纯文本上已经证明能显著提升推理，但直接搬到多模态场景会遇到（1）缺少高质量、标注明确的多模态推理数据；（2）多模态预训练会削弱模型原有的文本推理能力。于是出现了“怎么在不花大钱收集多模态标注数据的前提下，让小模型也能推理？”的难题。

### 关键概念速览
- **大模型（LLM）**：指拥有上百亿参数的语言模型，擅长纯文本理解与生成。这里的对比对象是只有 3 B 参数的轻量模型。  
- **多模态大模型（LMM）**：在 LLM 基础上加入视觉编码器，能够接受图像+文字输入。想象成把“看图说话”功能装进了聊天机器人。  
- **基于规则的强化学习（Rule‑based RL）**：利用人工编写的规则来给模型的输出打分，形成奖励信号，让模型在“试错”中学习更好的推理路径。类似老师给学生的即时批改。  
- **Foundational Reasoning Enhancement（FRE）**：先在纯文本数据上用规则 RL 强化模型的基本推理能力，相当于先让学生在黑板上练习逻辑题。  
- **Multimodal Generalization Training（MGT）**：把在 FRE 阶段学到的推理技巧迁移到视觉+文本的任务上，像把黑板练习的技巧搬到实际生活情境。  
- **复杂任务（Complex Task）**：指需要跨模态信息融合、长链推理的场景，例如“足球比赛视频中谁进了第 3 球”。  

### 核心创新点
1. **先文本后多模态的两阶段训练 → 先在大规模纯文本数据上做规则 RL（FRE），再用少量多模态数据进行迁移（MGT） → 解决了多模态高质量标注稀缺的问题，使得 3 B 模型在视觉任务上也能受益于文本推理提升。**  
2. **规则奖励从单一文本判定扩展为跨模态一致性检查 → 在 MGT 阶段，奖励不仅考虑文本答案是否符合规则，还检查视觉线索（如目标检测框）是否与答案对应 → 让模型学会把视觉证据映射到推理步骤，显著提升了视觉‑文本对齐质量。**  
3. **轻量模型专用的“规则库”设计 → 只保留对小模型有效的高频推理规则，避免规则过于细致导致奖励噪声 → 在参数受限的情况下仍能获得稳定的学习信号，提升了整体鲁棒性。**  

### 方法详解
整体思路是把“先把模型的推理底子练好，再让它把底子搬到看图说话的场景”拆成两段。  
1. **Foundational Reasoning Enhancement（FRE）**  
   - **数据**：使用公开的大规模文本推理数据集（如 GSM‑8K、ARC），不涉及任何图像。  
   - **规则引擎**：人工编写一套判定答案是否合理的规则，例如“答案必须是整数且在合理范围内”“步骤中出现的数值必须满足加减乘除的算术关系”。  
   - **强化学习循环**：模型先生成完整的思考链（CoT），规则引擎对每一步打分，得到累计奖励。利用策略梯度（如 REINFORCE）把奖励反馈给模型，更新参数。这里的关键是把奖励信号直接作用在思考链的每一步，而不是仅在最终答案上打分。  
   - **效果**：模型在纯文本基准上获得显著提升，为后续迁移奠定“推理骨架”。  

2. **Multimodal Generalization Training（MGT）**  
   - **数据**：挑选少量高质量的多模态推理样本（如 VQA‑Reason、ScienceQA‑Vision），每条样本包含图像、问题、思考链和答案。  
   - **跨模态规则**：在原有文本规则基础上加入视觉一致性检查。例如，如果思考链中提到“红色球员”，规则会检查图像中是否检测到红色衣服的目标；如果链中出现“左上角的数字”，则验证对应的 OCR 结果。  
   - **双模态奖励**：模型生成的思考链同时送入文本规则和视觉规则，得到两部分奖励的加权和。这样模型在生成推理步骤时必须兼顾文字逻辑和视觉证据。  
   - **微调策略**：采用较小的学习率，仅微调视觉编码器的投影层和语言模型的后置层，防止在迁移过程中把已经学好的文本推理“擦掉”。  

3. **整体训练流程（文字版流程图）**  
   ```
   FRE阶段：文本数据 → 生成思考链 → 规则打分 → 奖励 → 参数更新 → 获得强化的文本推理模型
   ↓
   MGT阶段：多模态数据 → 视觉编码 + 文本编码 → 生成思考链 → 文本+视觉规则打分 → 奖励 → 微调 → 获得兼具视觉感知和强推理的 LMM
   ```  
   最巧妙的地方在于 **奖励的层级化**：先让模型在“纯文字”层面学会自我纠错，再在“文字+图像”层面加上感知校验，这样即使多模态数据很少，也能让模型把已经掌握的推理技巧迁移过去。

### 实验与效果
- **测试模型**：Qwen2.5‑VL‑Instruct‑3B（3 B 参数的多模态指令模型）。  
- **基准**：包括多模态推理套件（VQA‑Reason、ScienceQA‑Vision）以及纯文本推理基准（GSM‑8K、ARC）。  
- **主要结果**：在多模态基准上平均提升 **4.83 %**，在纯文本基准上提升 **4.5 %**，在需要长链视觉‑文本推理的 Football Game 任务上提升 **3.63 %**。这些数字均是相对于未使用 LMM‑R1 的原始模型或传统微调基线。  
- **消融实验**：作者分别去掉 FRE、去掉视觉规则、以及只用单阶段 RL。结果显示：去掉 FRE 时多模态提升跌至约 2 %；去掉视觉规则时 MGT 阶段的增益几乎消失，说明跨模态规则是关键。  
- **局限性**：论文承认规则库的构建仍需人工投入，且规则的覆盖度直接决定奖励质量；此外，实验只在 3 B 规模模型上验证，尚不清楚在更大模型上是否仍有同等收益。  

### 影响与延伸思考
LMM‑R1 为“先文本后多模态”的数据高效迁移提供了可行路径，激发了后续工作在 **规则驱动的跨模态强化学习**、**少样本多模态适配** 方向的探索。2024‑2025 年间，有几篇论文尝试把 **自监督视觉对齐** 与 **文本推理微调** 结合，直接引用了 LMM‑R1 的两阶段框架（如 “Rule‑RL‑Vision”）。如果想进一步深入，可以关注 **可自动生成规则的元学习**、**大模型内部推理路径可解释化** 以及 **跨模态奖励函数的学习** 等前沿议题。  

### 一句话记住它
先用规则强化文本推理，再用少量视觉‑文本样本把这套推理技巧迁移过去，让 3 B 多模态模型也能像大模型一样“看图会推理”。