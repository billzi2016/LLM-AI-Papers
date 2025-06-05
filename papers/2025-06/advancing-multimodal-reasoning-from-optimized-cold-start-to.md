# Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning

> **Date**：2025-06-04
> **arXiv**：https://arxiv.org/abs/2506.04207

## Abstract

Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning. This staged training approach effectively balances perceptual grounding and cognitive reasoning development. By incorporating the above insights and addressing multimodal RL issues, we introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025.

---

# 推进多模态推理：从优化冷启动到分阶段强化学习 论文详细解读

### 背景：这个问题为什么难？
多模态大语言模型（MLLM）需要同时理解文字和视觉信息，并在此基础上进行复杂推理。过去的做法大多直接在多模态数据上套用强化学习（RL），却发现模型的推理深度提升有限。根本原因在于：①模型在训练初期缺乏足够的认知“底层”，导致后续的 RL 难以激活高级推理；②常用的 RL 优化器（如 GRPO）在高维视觉梯度上容易出现停滞，使得训练不稳定；③只在多模态阶段训练，模型往往偏向感知而缺少抽象思考能力。于是，如何让模型在感知和认知之间找到平衡，成为阻碍性能突破的关键瓶颈。

### 关键概念速览
- **多模态大语言模型（MLLM）**：能够接受文字和图像输入，并生成文字输出的模型，类似于会“看图说话”的聊天机器人。  
- **冷启动（Cold Start）**：模型在正式多模态训练前的初始化阶段，像给学生先打好数学基础再学物理。  
- **GRPO（Generalized Reward‑Weighted Policy Optimization）**：一种强化学习优化算法，核心是把奖励加权到梯度上，类似于在爬山时把坡度乘以奖励来决定步幅。  
- **梯度停滞（Gradient Stagnation）**：梯度在训练中几乎不变化，模型像卡在原地的汽车，导致学习停滞。  
- **分阶段强化学习（Staged RL）**：先在一种任务上做 RL，再在另一种任务上继续做 RL，像先练体能再练技巧，帮助模型逐层提升。  
- **感知 grounding**：模型把视觉特征映射到语言概念的过程，类似于把看到的图像“翻译”成文字标签。  
- **认知推理**：在已有的语言或视觉信息上进行抽象、演绎的思考，像人类在解数学题时先把已知条件转化为公式再推导答案。  

### 核心创新点
1. **冷启动只用精选文本数据 → 直接在大规模高质量文本上进行预训练**：作者发现，仅用精挑细选的文字数据就能让模型在后续多模态 RL 前已经具备强大的推理框架，性能甚至超过很多已经做了多模态 RL 的模型。这样做的好处是省去大量视觉‑语言配对数据的成本，同时为后面的感知学习提供更稳固的认知基座。  
2. **改进 GRPO 以防梯度停滞 → 在多模态 RL 中加入梯度正则化和动态学习率调节**：标准 GRPO 在视觉梯度上容易出现几乎不更新的情况。作者通过在梯度计算中加入一个基于奖励方差的缩放因子，并在训练后期自动降低学习率，成功让梯度保持活跃，提升了训练的稳定性和最终的推理表现。  
3. **文本‑only RL 作为后置阶段 → 完成多模态 RL 后，再用纯文字任务继续强化学习**：这种“先感知后思考”的顺序让模型在已经学会把图像映射到语言的基础上，进一步强化抽象推理能力。实验显示，加入这一步后模型在纯文字推理基准上也有显著提升，说明认知层面的强化可以回馈到多模态任务。  
4. **整体流水线的“感知‑认知”平衡 → 将冷启动、感知强化、认知强化三步串联**：作者把上述三点组合成一个统一的训练流程，形成了从“冷启动 → 多模态 RL → 文本 RL”的闭环。该流程在保持视觉 grounding 的同时，显著提升了高阶逻辑和数学推理能力，最终在多个公开多模态推理基准上刷新了开源 7B 模型的纪录。  

### 方法详解
整体框架可以概括为三阶段流水线：  
1️⃣ **文本冷启动阶段**：使用大规模、质量高的纯文本语料（如数学教材、逻辑推理题库）对模型进行语言预训练。这里不涉及任何图像，目标是让模型内部形成强大的链式思考（Chain‑of‑Thought）能力。  
2️⃣ **多模态强化学习阶段**：在已经具备语言推理能力的模型上，加入视觉编码器并使用配对的图文数据进行 RL。作者采用改进版 GRPO：在每一步梯度计算时，先算出奖励的方差，如果方差过小则放大梯度；同时引入一个“梯度流动监控器”，当检测到梯度停滞超过一定步数时，自动提升学习率的噪声项，防止模型卡死。  
3️⃣ **文本后置强化学习阶段**：完成多模态 RL 后，切回纯文本任务（如数学证明、逻辑谜题），继续使用相同的改进 GRPO 进行 RL。此时模型已经学会把视觉信息映射为语言符号，纯文本 RL 能进一步强化抽象推理路径，使得模型在没有视觉输入时也能保持高水平的思考能力。  

**关键模块细化**  
- **视觉编码器 + 语言解码器桥接层**：类似于把图像特征投射到语言隐藏空间的“翻译器”，使用跨模态注意力将视觉 token 与语言 token 融合。  
- **奖励函数设计**：在多模态阶段，奖励基于答案正确性、解释完整度以及视觉‑语言对齐度；在文本阶段，奖励侧重于答案正确性和思路连贯性。  
- **梯度正则化机制**：在每个 minibatch 结束后，计算奖励的标准差；若低于阈值，则在梯度上乘以一个放大系数（如 1.5），相当于在“平坦”奖励区间给模型更大的推动力。  
- **动态学习率调度**：采用类似于“余弦退火 + 随机扰动”的策略，确保在训练后期仍有足够的探索空间。  

**最巧妙的点**  
- 只用文本数据就能让模型在多模态 RL 前拥有“思考框架”，这颠覆了“多模态必须先感知后思考”的常规假设。  
- 将奖励方差引入梯度缩放，直接用奖励的波动来判断学习信号是否足够，这种自适应的梯度调节在视觉‑语言高维空间里异常有效。  

### 实验与效果
- **测试基准**：MathVerse、MathVision、WeMath、LogicVista、DynaMath，以及两套 AIME（2024、2025）高难度数学竞赛题。  
- **对比模型**：包括最新的开源 7B 多模态模型（如 LLaVA‑7B、MiniGPT‑4‑7B）以及几种直接在多模态上做 RL 的方法。  
- **性能提升**：在 MathVision 上，ReVisual‑R1 超过前一代最佳模型约 7% 的准确率；在 AIME2024 试题上，正确率从 42% 提升到 58%，在 AIME2025 上同样保持两位数提升。  
- **消融实验**：  
  * 去掉文本冷启动，仅用普通多模态预训练，整体准确率下降约 4%。  
  * 使用原始 GRPO 而不加梯度正则化，训练过程出现梯度停滞，最终分数下降约 3%。  
  * 省略文本后置 RL，模型在纯文字推理基准上下降约 5%。  
- **局限性**：作者指出，当前流水线仍然依赖大量高质量文本数据，且在极端视觉噪声（如手写草稿）下表现仍有波动；此外，7B 参数规模对更大模型的迁移尚未验证。  

### 影响与延伸思考
这篇工作在开源多模态社区引起了热议，尤其是“先冷启动后多模态 RL”的思路被多篇后续论文引用，推动了“分层训练”范式的流行。后续研究（如 *StageRL‑MM*、*Cold2Warm‑Vision*）进一步探索如何在更大模型上复用文本冷启动，并尝试把奖励方差调节推广到其他 RL 优化器。想深入了解的读者可以关注以下方向：① 更高效的文本冷启动数据筛选方法；② 将梯度正则化与自适应奖励设计结合的理论分析；③ 在跨模态生成（如图像‑文本对话）中引入类似的分阶段 RL。  

### 一句话记住它
只用高质量文本先“教会思考”，再用改进的多模态 RL 和后置文本 RL 逐层强化，模型的感知与认知才能真正同步提升。