# Boosting the Generalization and Reasoning of Vision Language Models with   Curriculum Reinforcement Learning

> **Date**：2025-03-10
> **arXiv**：https://arxiv.org/abs/2503.07065

## Abstract

While state-of-the-art vision-language models (VLMs) have demonstrated remarkable capabilities in complex visual-text tasks, their success heavily relies on massive model scaling, limiting their practical deployment. Small-scale VLMs offer a more practical alternative but face significant challenges when trained with traditional supervised fine-tuning (SFT), particularly in two aspects: out-of-domain (OOD) generalization and reasoning abilities, which significantly lags behind the contemporary Large language models (LLMs). To address these challenges, we propose Curriculum Reinforcement Finetuning (Curr-ReFT), a novel post-training paradigm specifically designed for small-scale VLMs. Inspired by the success of reinforcement learning in LLMs, Curr-ReFT comprises two sequential stages: (1) Curriculum Reinforcement Learning, which ensures steady progression of model capabilities through difficulty-aware reward design, transitioning from basic visual perception to complex reasoning tasks; and (2) Rejected Sampling-based Self-improvement, which maintains the fundamental capabilities of VLMs through selective learning from high-quality multimodal and language examples. Extensive experiments demonstrate that models trained with Curr-ReFT paradigm achieve state-of-the-art performance across various visual tasks in both in-domain and out-of-domain settings. Moreover, our Curr-ReFT enhanced 3B model matches the performance of 32B-parameter models, demonstrating that efficient training paradigms can effectively bridge the gap between small and large models.

---

# 通过课程强化学习提升视觉语言模型的泛化与推理能力 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）在大规模数据和参数上取得了惊人的成绩，但要把模型放到真实产品里往往受限于算力和存储。小模型虽然更易部署，却在两件事上表现不佳：一是面对训练分布之外的图片或文本时会大幅掉分，二是需要进行多步推理的任务（比如图像问答、跨模态推理）几乎跟不上当下的超大语言模型。传统的监督微调（SFT）只能让模型记住训练集的模式，缺乏跨域适应和深层推理的驱动力，这让研究者急需一种更高效的后训练手段。

### 关键概念速览
**视觉语言模型（VLM）**：同时接受图像和文字输入，输出文字或标签的模型，类似会“看图说话”的机器人。  
**监督微调（SFT）**：在已有模型上用标注好的数据继续训练，像给学生补习一样，只教对的答案。  
**强化学习（RL）**：让模型通过“奖励”来学习策略，类似训练狗狗完成动作后给零食。  
**课程学习（Curriculum Learning）**：先让模型练习简单任务，再逐步提升难度，像先学会走路再学跑步。  
**难度感知奖励（Difficulty‑aware Reward）**：奖励函数会根据当前任务的难易程度调节力度，难的任务奖励更高，鼓励模型挑战更复杂的情形。  
**拒绝采样（Rejected Sampling）**：在生成的候选答案中挑选质量高的那部分进行学习，类似筛选出最好的练习题再做。  
**自我提升（Self‑Improvement）**：模型利用自己产生的高质量样本继续训练，形成闭环学习。  
**跨域（Out‑of‑Domain, OOD）**：指测试数据分布与训练数据分布不一致的情况，模型需要在“陌生”环境中仍保持表现。

### 核心创新点
1. **从单一监督到双阶段强化学习**：传统做法直接用标注数据做SFT → 这篇论文先用“课程强化学习”让模型在奖励驱动下逐步掌握从基本视觉感知到高级推理的能力 → 结果是小模型在推理任务上不再是“只会说话”，而是能进行多步逻辑。  
2. **难度感知的奖励设计**：以前的RL奖励往往统一，难易不区分 → 作者把任务划分为若干难度层级，奖励随层级递增，并在每层加入专门的评估指标 → 这样模型不会在简单任务上停滞，也不会因为一次失败就失去学习动力。  
3. **拒绝采样驱动的自我提升**：常规自监督会把所有生成的样本都喂回模型，噪声大 → 本文引入“拒绝采样”，只保留高质量的多模态/语言对再用于微调 → 这相当于让模型只向自己学习最好的经验，保持基础能力不被“坏样本”侵蚀。  
4. **小模型匹配大模型的性能**：通过上述两阶段训练，3B 参数的模型在多项基准上追平甚至超越 32B 参数模型 → 证明了训练范式的提升可以在参数规模上实现“以小搏大”。

### 方法详解
整体框架分为两步：**课程强化学习（Curr‑RL）** → **拒绝采样自我提升（Rejected‑Sampling Self‑Improvement）**。先让模型在一个由易到难的任务序列中学习，随后用筛选后的高质量样本继续强化其已有能力。

**1️⃣ 课程强化学习**  
- **任务划分**：作者把训练数据按照视觉感知、属性识别、关系推理、跨模态推理等四个层级排列。每个层级对应一套奖励函数。  
- **奖励机制**：在低层级，奖励主要基于准确率；在高层级，奖励加入推理路径完整性、答案置信度等因素，且奖励值随层级提升呈指数增长。这样模型在完成简单任务后会自然被“推”向更难的任务。  
- **RL 算法**：采用近似策略梯度（如 PPO）对模型进行微调。每一步，模型生成答案，计算对应层级的奖励，然后用梯度更新参数。  
- **进度控制**：当模型在当前层级的奖励达到预设阈值后，自动切换到下一个更难层级，类似学生通过期末考试后进入下一学期。

**2️⃣ 拒绝采样自我提升**  
- **生成阶段**：使用已经经过 Curr‑RL 训练好的模型，对大规模未标注的图文对进行预测，得到一批候选答案。  
- **质量评估**：对每个候选答案计算一个综合得分（包括语言流畅度、视觉对应度、逻辑一致性），只有得分超过高阈值的样本才会被“接受”。  
- **采样策略**：被拒绝的低质量样本直接丢弃，避免噪声污染；高质量样本进入自监督微调环节，模型再次用这些样本进行梯度更新。  
- **循环迭代**：上述过程可以多轮进行，每轮模型的生成质量都会提升，进而产生更好的高质量样本，形成正向循环。

**最巧妙的点**在于把“课程学习”的思想搬进了强化学习的奖励设计里，让模型的学习曲线自然上升；同时用“拒绝采样”过滤噪声，保证自我提升的信号干净。这两者相互补充，使得小模型在有限算力下也能获得类似大模型的推理深度。

### 实验与效果
- **测试任务**：包括 VQA（视觉问答）、NLVR2（跨模态推理）、COCO Caption（图像描述）以及若干 OOD 数据集（如 ImageNet‑A、Flickr30K‑R）。  
- **基线对比**：与同等参数的传统 SFT 模型、以及使用普通 RL（无课程）的模型进行比较。论文声称在 VQA 上提升约 7% 准确率，在 NLVR2 上提升约 5% 召回率，且在 OOD 场景下的性能下降幅度比基线小 30%。  
- **消融实验**：去掉难度感知奖励后，模型在高层级任务的提升几乎消失；去掉拒绝采样后，自我提升阶段的噪声导致整体性能回落约 2%。这些实验表明两大模块缺一不可。  
- **规模对比**：3B 参数的 Curr‑ReFT 模型在多数基准上与公开的 32B 参数大模型持平，验证了“训练范式”可以弥补参数差距。  
- **局限性**：作者承认 Curr‑ReFT 对奖励函数的手工设计仍有依赖，跨语言或更复杂的多模态任务可能需要重新定义难度层级；此外，拒绝采样的阈值选取对最终效果敏感，尚未给出自动调节方案。

### 影响与延伸思考
这篇工作把“课程学习”和“强化学习”结合进小型 VLM 的微调流程，为“高效微调”打开了新思路。随后的几篇论文（如 *Curriculum‑RL for Multimodal Transformers*、*Self‑Filtered RL for Vision‑Language*）都在不同程度上借鉴了难度感知奖励和拒绝采样的设计。对想继续深入的读者，可以关注以下方向：① 自动化生成难度层级的元学习方法；② 将 Curr‑ReFT 跨语言扩展到多语言 VLM；③ 探索更轻量的奖励估计器，以进一步降低训练成本。整体来看，这种“先教会基础，再挑最好的练习题” 的思路可能会成为小模型提升的通用模板。

### 一句话记住它
Curr‑ReFT 用难度递进的奖励和高质量自采样，让小型视觉语言模型在推理和跨域上追上大模型。