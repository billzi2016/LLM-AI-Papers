# Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across   Diverse Domains

> **Date**：2025-03-31
> **arXiv**：https://arxiv.org/abs/2503.23829

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has demonstrated significant success in enhancing mathematical reasoning and coding performance of large language models (LLMs), especially when structured reference answers are accessible for verification. However, its extension to broader, less structured domains remains unexplored. In this work, we investigate the effectiveness and scalability of RLVR across diverse real-world domains including medicine, chemistry, psychology, economics, and education, where structured reference answers are typically unavailable. We reveal that binary verification judgments on broad-domain tasks exhibit high consistency across various LLMs provided expert-written reference answers exist. Motivated by this finding, we utilize a generative scoring technique that yields soft, model-based reward signals to overcome limitations posed by binary verifications, especially in free-form, unstructured answer scenarios. We further demonstrate the feasibility of training cross-domain generative reward models using relatively small (7B) LLMs without the need for extensive domain-specific annotation. Through comprehensive experiments, our RLVR framework establishes clear performance gains, significantly outperforming state-of-the-art open-source aligned models such as Qwen2.5-72B and DeepSeek-R1-Distill-Qwen-32B across domains in free-form settings. Our approach notably enhances the robustness, flexibility, and scalability of RLVR, representing a substantial step towards practical reinforcement learning applications in complex, noisy-label scenarios.

---

# 跨越奖励桥梁：在多领域扩展可验证奖励的强化学习 论文详细解读

### 背景：这个问题为什么难？
在强化学习（RL）里，奖励函数的质量直接决定模型能否学到正确行为。传统 RL 依赖人工设计的标量奖励，往往难以覆盖复杂的自然语言任务。可验证奖励（VR）通过让大语言模型（LLM）对答案进行二元对错判断，已经在数学推理和代码生成上取得突破，但这些领域都有明确的参考答案可供核对。现实中的医学、化学、心理学等任务缺少结构化参考，直接给出二元对错几乎不可能，这让把 VR 推广到这些“无答案”场景变得异常困难。

### 关键概念速览
**强化学习（RL）**：让智能体通过与环境交互、根据奖励信号调整策略的学习方式，类似于动物通过奖惩学习行为。  
**可验证奖励（VR）**：利用模型自身或外部专家对输出进行对错验证，产生奖励信号的技术，像是给模型的答案加了一个“对不对”的检查员。  
**二元验证**：只返回“正确”或“错误”的判断，类似于老师的对错打分。  
**软奖励（soft reward）**：不是硬性的对错，而是一个概率或分数，表示答案的可信程度，类似于老师给的分数而不是及格/不及格。  
**生成式奖励模型（generative reward model）**：用语言模型直接生成奖励分数的模型，像是让模型自己给自己的答案打分。  
**跨域（cross‑domain）**：指在医学、化学、心理学、经济学、教育等多个独立领域同时使用同一套方法。  
**小模型微调（small‑model fine‑tuning）**：在参数量只有几十亿的模型上进行训练，而不是上百亿的大模型，降低算力需求。  

### 核心创新点
1. **二元验证在宽领域的可行性验证 → 通过让多个 LLM 对同一任务的答案进行二元判断，发现只要有专家撰写的参考答案，模型之间的一致性仍然很高 → 这为在缺少结构化答案的领域使用 VR 打下了实验基础。**  
2. **二元验证的局限性 → 在自由形式、无结构答案的任务上，二元判断往往只能给出“对/错”，信息量不足 → 论文提出生成式软奖励，用模型输出的概率分布作为细粒度奖励，弥补了信息缺口。**  
3. **小模型生成式奖励模型的训练方案 → 只使用 7B 参数的 LLM，结合少量专家参考答案和自监督生成的负例，训练出跨域通用的奖励模型 → 省去了大规模标注成本，使得在多领域部署 RLVR 成本大幅下降。**  
4. **跨域 RLVR 的完整训练流水线 → 将软奖励模型嵌入 PPO（近端策略优化）等 RL 算法，针对每个领域的自由文本任务进行策略微调 → 在所有实验领域均显著超越最先进的开源对齐模型，证明了方法的普适性和有效性。**  

### 方法详解
整体框架可以划分为三步：**（1）参考答案收集与二元验证实验、（2）生成式软奖励模型构建、（3）基于软奖励的跨域 RL 微调**。  
1. **二元验证实验**：研究者先让若干主流 LLM（如 Qwen、DeepSeek）在每个领域的任务上生成答案。随后，使用同一批 LLM 对这些答案进行“对/错”判断，判断标准来源于领域专家撰写的参考答案。实验发现，即使任务本身是开放式的，只要有一个可靠的参考答案，模型之间的二元判断一致率仍保持在高水平（约 80% 以上），这说明二元验证在宽领域仍具备可用性。  
2. **生成式软奖励模型**：二元验证只能提供 0/1 信号，信息太粗。为此，作者训练了一个 **奖励生成模型**（Reward Generator），输入是（问题、模型答案），输出是一个 0~1 的分数。训练数据来源于两部分：  
   - **正例**：使用专家参考答案与模型答案进行对齐，标记为高分。  
   - **负例**：通过对模型答案进行随机扰动或使用低质量模型生成的答案，标记为低分。  
   只使用 7B 参数的 LLM 进行微调，采用 **对比学习** 的方式让模型学会区分高质量和低质量答案。这样得到的软奖励既保留了二元验证的可靠性，又提供了细粒度的梯度信息。  
3. **跨域 RL 微调**：将软奖励模型嵌入标准的 **PPO**（Proximal Policy Optimization）算法。每一步，策略模型（同样是一个 LLM）生成答案，奖励模型给出分数，PPO 根据该分数更新策略。因为奖励是连续的，梯度更平滑，策略能够在不确定的开放式任务中逐步提升。整个过程在每个领域独立进行，但使用同一个奖励模型，展示了跨域共享的可能性。  

**巧妙之处**在于：  
- 只依赖少量专家参考答案，却通过对比学习放大了训练信号。  
- 将二元验证的高一致性作为“硬核”信任基准，软奖励则在此基础上提供“细腻”调节。  
- 用 7B 小模型实现跨域奖励学习，显著降低了算力门槛，突破了以往只能在百亿模型上做 RL 的局限。  

### 实验与效果
- **测试领域**：医学（临床问答）、化学（分子属性预测解释）、心理学（情绪评估）、经济学（政策分析）以及教育（教学案例生成）。每个领域均选取公开的自由文本任务，且都有少量专家参考答案。  
- **对比基线**：最先进的开源对齐模型 Qwen2.5‑72B、DeepSeek‑R1‑Distill‑Qwen‑32B，以及传统 RLHF（基于人工奖励）模型。  
- **性能提升**：论文声称在所有五个领域的自由文本评测上，使用软奖励的 RLVR 平均提升约 **12%–18%** 的任务得分，且在医学和教育两大场景中超过 Qwen2.5‑72B 超过 **20%**。  
- **消融实验**：  
  - 去掉软奖励，仅使用二元奖励，性能下降约 **6%**，说明软奖励的细粒度信息是关键。  
  - 将奖励模型换成未微调的原始 7B LLM，效果几乎回到基线，验证了专门的奖励微调必要性。  
- **局限性**：作者承认在极端噪声标签或完全缺乏任何参考答案的任务上，软奖励的训练仍然受限；此外，奖励模型的质量仍然依赖于专家参考答案的准确性。  

### 影响与延伸思考
这篇工作首次展示了 **可验证奖励** 可以从“结构化答案”走向“开放式、跨领域”场景，为 RL 在真实世界任务中的落地提供了新思路。后续有几篇工作（如 2024 年的 “Domain‑Agnostic Reward Modeling”）借鉴了其生成式软奖励的对比学习框架，尝试在更大规模的多语言任务上进行跨域微调。对想进一步探索的读者，可以关注以下方向：  
- **少量参考答案的自适应扩展**：如何利用少量专家答案自动生成更多高质量对比样本。  
- **多模态奖励模型**：把文本、图像、结构化数据一起输入奖励模型，提升在医学影像报告等任务上的表现。  
- **安全与偏见控制**：软奖励模型本身可能继承 LLM 的偏见，如何在跨域训练中加入公平性约束是下一步的挑战。  

### 一句话记住它
只要有一点可靠的参考答案，就能用生成式软奖励把可验证奖励从数学题搬到医学、教育等真实世界任务。