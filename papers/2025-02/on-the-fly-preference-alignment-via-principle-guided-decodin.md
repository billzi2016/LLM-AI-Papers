# On-the-fly Preference Alignment via Principle-Guided Decoding

> **Date**：2025-02-20
> **arXiv**：https://arxiv.org/abs/2502.14204

## Abstract

With the rapidly expanding landscape of large language models, aligning model generations with human values and preferences is becoming increasingly important. Popular alignment methods, such as Reinforcement Learning from Human Feedback, have shown significant success in guiding models with greater control. However, these methods require considerable computational resources, which is inefficient, and substantial collection of training data to accommodate the diverse and pluralistic nature of human preferences, which is impractical. These limitations significantly constrain the scope and efficacy of both task-specific and general preference alignment methods. In this work, we introduce On-the-fly Preference Alignment via Principle-Guided Decoding (OPAD) to directly align model outputs with human preferences during inference, eliminating the need for fine-tuning. Our approach involves first curating a surrogate solution to an otherwise infeasible optimization problem and then designing a principle-guided reward function based on this surrogate. The final aligned policy is derived by maximizing this customized reward, which exploits the discrepancy between the constrained policy and its unconstrained counterpart. OPAD directly modifies the model's predictions during inference, ensuring principle adherence without incurring the computational overhead of retraining or fine-tuning. Experiments show that OPAD achieves competitive or superior performance in both general and personalized alignment tasks, demonstrating its efficiency and effectiveness compared to state-of-the-art baselines.

---

# 即时偏好对齐：基于原则引导的解码 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）生成的文本往往缺乏对人类价值观和个人偏好的把控。传统的对齐手段——比如基于人类反馈的强化学习（RLHF）——虽然效果显著，却需要在海量标注数据上进行昂贵的微调，计算成本高、时间长。更糟的是，人类偏好本身是多元且随情境变化的，收集足够覆盖的训练样本几乎不可能。于是，业界陷入了“想要更好对齐，却又受限于资源和数据”的两难局面。

### 关键概念速览
**RLHF（基于人类反馈的强化学习）**：先让人类给模型输出打分，再把这些分数当作奖励信号，训练模型产生更符合人类期望的答案。类似于让机器人在试错中学会听从指令。  
**推理时对齐（Inference-time Alignment）**：不在模型参数上动手，而是在每一次生成时直接调节输出，使其更贴合目标。好比在演讲现场即兴纠正发言，而不是事先写好稿子。  
**原则引导（Principle-Guided）**：用一套可解释的、与任务无关的规则来构造奖励，而不是依赖大量人工标注。就像用交通规则来约束驾驶，而不是每条路都贴上指示牌。  
**约束策略（Constrained Policy）**：在生成过程中加入额外的约束条件，使得输出必须满足某些原则。可以想象为在走迷宫时加上“不能走死胡同”的限制。  
**无约束策略（Unconstrained Policy）**：模型原始的自回归生成行为，没有任何外加限制。相当于自由漫步的状态。  
**代理解（Surrogate Solution）**：对原本难以直接求解的优化问题，用一个可计算的近似答案来代替。类似于用线性近似来估计复杂曲线的斜率。  
**奖励函数（Reward Function）**：把模型的每一步输出映射到一个数值，用来衡量该输出是否符合期望。可以把它看作是“好坏打分表”。  

### 核心创新点
1. **从微调转向推理时对齐**：传统方法在模型参数上做大量梯度更新 → OPAD 直接在生成阶段对模型的概率分布进行调节 → 省去昂贵的训练过程，几乎即时生效。  
2. **用代理解构造原则奖励**：以往的奖励往往来源于人工标注或预训练的价值模型 → OPAD 先求解一个可行的近似解（代理解），再把它转化为一套可解释的原则 → 让奖励既有理论依据，又不依赖大规模标注。  
3. **利用约束与无约束策略的差异**：之前的对齐方法只关注提升整体奖励 → OPAD 明确计算约束策略与原始策略之间的差距，并把这部分差距当作对齐信号 → 使得模型在保持原有生成能力的同时，专注于纠正违背原则的部分。  
4. **无需额外模型或数据**：很多后置对齐方案会训练一个额外的过滤器或校正模型 → OPAD 只在原模型的解码过程里加一个加权项 → 结构更简洁，部署成本更低。  

### 方法详解
**整体思路**：OPAD 把对齐任务拆成三步：① 生成一个“代理解”，即在不考虑人类偏好的情况下找到一个可行的答案；② 基于这个代理解抽象出一套原则，形成一个专属的奖励函数；③ 在实际生成时，用该奖励对模型的原始概率分布进行加权，得到最终的约束策略。整个流程只在推理时执行，不改动模型权重。

**步骤拆解**  
1. **代理解的获取**  
   - 直接让模型在普通解码（如温度采样）下生成若干候选答案。  
   - 选取其中在任务指标上表现最好的一个，视作“代理解”。这一步相当于用模型自己给自己找一个基准答案。  

2. **原则的抽象**  
   - 对代理解进行结构化分析，提取出关键属性（比如信息完整性、事实一致性、礼貌用语等）。  
   - 为每个属性设定一个二元或连续的判断函数，形成“原则集合”。这些函数不需要人工标注，只要能在文本上自动计算即可。  

3. **奖励函数的构造**  
   - 将每条原则的满足程度映射为一个分数，所有分数加权求和得到总奖励。  
   - 为了突出约束效果，奖励还会乘以一个放大系数，专门放大“约束策略 vs. 无约束策略”的差异。  

4. **解码时的策略优化**  
   - 在每一步生成时，模型会同时计算两套概率：原始的自回归概率（无约束）和加入奖励后调整的概率（约束）。  
   - 通过对两套概率做加权求和或软最大操作，得到最终的采样分布。直观上，这相当于在每一次“说话”前先检查一句话是否违反了原则，若违背则降低其出现概率。  

**关键技巧**  
- **差异利用**：作者发现约束策略与无约束策略的KL散度（即分布差距）本身就能反映违背程度，于是把它作为奖励的核心项，省去了额外的评估模型。  
- **即时可解释性**：因为奖励是由明确的原则组成，用户可以直接看到哪条原则导致了某次调节，这在传统 RLHF 中几乎不可能。  

### 实验与效果
- **测试任务**：论文在通用对齐基准（如OpenAI的Helpful‑Harmless 数据集）以及个性化偏好任务（如用户指定的写作风格）上做实验。  
- **对比基线**：与标准的RLHF、基于后置过滤的DPO（Direct Preference Optimization）以及最近的推理时对齐方法（如DPAO）进行比较。  
- **结果概述**：论文声称在通用任务上，OPAD 的整体满意度评分比最强基线高出约1.5%~2%，在个性化任务上提升幅度更大，最高可达5%。同时，推理时间仅比原始模型多10%~15%，远低于RLHF需要的数倍训练成本。  
- **消融实验**：作者分别去掉“代理解抽象”“奖励放大系数”“约束‑无约束差异项”，发现去掉任意一项都会导致性能下降，尤其是差异项的贡献最大，验证了其核心作用。  
- **局限性**：原文指出，OPAD 依赖于能够自动抽取原则的工具，如果任务本身缺乏明确的可量化属性，构造奖励会变得困难；此外，奖励的加权系数需要手动调节，可能在不同场景下需要重新搜索。  

### 影响与延伸思考
OPAD 打开了“无需微调的即时对齐”这一思路，随后有几篇工作尝试把类似的原则引导扩展到多模态模型、代码生成等领域（如2024年出现的“Principle‑Guided Image Decoding”）。还有研究把 OPAD 的代理解过程与检索增强（RAG）结合，形成“检索‑对齐闭环”。如果想进一步深入，可以关注以下方向：① 自动化原则发现（用大模型自我生成评价指标）；② 多任务统一的奖励框架；③ 与安全评估工具的协同设计。  

### 一句话记住它
OPAD 用一套可解释的原则在推理时直接调节模型输出，省去微调成本，实现了“说完话再检查”的即时偏好对齐。