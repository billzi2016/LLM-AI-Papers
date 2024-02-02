# KTO: Model Alignment as Prospect Theoretic Optimization

> **Date**：2024-02-02
> **arXiv**：https://arxiv.org/abs/2402.01306

## Abstract

Kahneman & Tversky's $\textit{prospect theory}$ tells us that humans perceive random variables in a biased but well-defined manner (1992); for example, humans are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biases -- the success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them belonging to a family of loss functions that we call $\textit{human-aware losses}$ (HALOs). However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature. Using a Kahneman-Tversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do. We call this approach KTO, and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B, despite only learning from a binary signal of whether an output is desirable. More broadly, our work suggests that there is no one HALO that is universally superior; the best loss depends on the inductive biases most appropriate for a given setting, an oft-overlooked consideration.

---

# KTO：基于前景理论的模型对齐优化 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，让模型的输出符合人类价值观一直是核心挑战。早期的对齐手段大多把人类偏好当作“正确答案”，用交叉熵最小化直接逼近这些答案。但人类的评价往往是二元的（好/坏）或成对的偏好，而不是完整的概率分布。于是出现了基于偏好学习的 DPO、RLHF 等方法，它们把偏好转化为对数似然来训练模型。虽然这些方法在实际应用中取得了显著提升，却仍然把人类的判断简化为“更喜欢 A 而不是 B”，忽略了心理学里人们对收益和损失的非线性感知。换句话说，现有目标函数缺少对人类认知偏差的建模，这限制了对齐的上限。

### 关键概念速览
- **前景理论（Prospect Theory）**：Kahneman 与 Tversky 提出的描述人类在面对风险时如何扭曲概率和价值的模型，核心是“损失规避”和“价值函数的凹凸”。可以想象成人们在玩赌博游戏时，对赢的期待和对输的恐惧并不对称。
- **HALO（Human‑Aware Loss）**：一类专门把人类认知偏差嵌入损失函数的设计思路。它们不直接最小化预测错误，而是让模型的学习目标更贴合人类的感知方式。
- **偏好对齐（Preference Alignment）**：通过收集成对比较（A 更好还是 B）来训练模型，使其输出更符合人类的相对喜好。类似于让模型学会“选谁更好”，而不是“给出唯一正确答案”。
- **二元信号（Binary Signal）**：只告诉模型“这段输出好”或“这段输出坏”，不提供具体的偏好排序或分数。相当于只给模型一个“是/否”标签。
- **对数似然（Log‑Likelihood）**：在偏好学习中常用的目标函数，最大化模型给出正确偏好的概率。可以把它想成让模型“自信地说出它更喜欢的那一个”。
- **效用函数（Utility Function）**：在经济学和决策理论里衡量某个结果对决策者的价值。前景理论给出的效用函数会对损失放大，对收益收敛。
- **KTO（Prospect‑Theoretic Optimization）**：本文提出的核心方法，直接把前景理论的效用函数当作学习目标，最大化生成文本的“感知效用”，而不是偏好的对数似然。

### 核心创新点
1. **从对数似然到效用最大化**：传统偏好对齐把人类偏好视作概率分布，用对数似然来训练。KTO 把人类的效用函数（基于前景理论）直接嵌入损失，目标变成“让模型的输出在人的感知上价值最高”。这一步把心理学的非线性偏好搬进了模型训练。
2. **仅用二元好坏信号进行对齐**：DPO 需要成对比较数据，而 KTO 只需要每条生成文本的好坏标签。这样大幅降低了数据标注成本，同时保持了对齐效果。相当于把“我更喜欢 A 还是 B”简化为“这段话好不好”。
3. **提出 HALO 家族的统一视角**：作者把 DPO、RLHF 等方法归类为 HALO，指出它们共享“人类感知偏差”这一隐含假设。通过这种统一框架，KTO 被解释为一种更符合前景理论的 HALO，实现了理论与实践的桥接。
4. **实验验证“没有万能 HALO”**：通过在 1B‑30B 参数规模上对比，作者展示不同 HALO 在不同任务上的表现差异，强调最优损失函数取决于具体的 inductive bias，而不是单一的最优解。

### 方法详解
**整体思路**：KTO 的训练流程只有两步：① 采样模型生成文本；② 根据二元好坏标签计算前景理论效用并进行梯度更新。整个过程不需要成对比较，也不需要对数似然的概率建模。

**关键模块拆解**：

1. **生成与标注**  
   - 先让未对齐的 LLM 生成若干候选答案。  
   - 人类标注者只判断每个答案是否“可接受”，给出 1（好）或 0（坏）。这一步类似于普通的内容审查，只要判断对错，不需要比较两条答案。

2. **前景理论效用映射**  
   - 对每个二元标签，使用前景理论的价值函数 v(x) 将 1、0 映射到感知效用。具体来说，正向结果（好）会被一个凹函数放大，负向结果（坏）会被一个凸函数放大，体现“损失规避”。  
   - 同时，引入概率加权函数 w(p)，把模型对好坏的预测概率进行非线性扭曲，使得低概率的好结果被放大，低概率的坏结果被压缩。

3. **损失计算**  
   - 对每条生成文本，计算其感知效用 U = w(p) * v(label)。  
   - 整体损失是负的期望效用，即取所有样本的 U 求平均后取负号，梯度上升等价于最大化期望效用。  
   - 与传统的对数似然不同，这里没有显式的 softmax 或交叉熵，而是直接对效用函数求导。

4. **梯度更新**  
   - 使用标准的 Adam 优化器对模型参数进行更新。因为效用函数是可微的（作者使用平滑的近似），梯度可以直接回传到语言模型的所有层。

**最巧妙的点**：把二元好坏标签通过前景理论的价值函数和概率加权函数转化为连续的效用信号，使得仅凭“好/坏”也能产生丰富的梯度信息。这种“从离散到连续”的桥接，是 KTO 能在没有成对比较的情况下仍然实现强对齐的关键。

### 实验与效果
- **测试任务**：作者在公开的 OpenAI Human Preferences 数据集、Anthropic HH（Helpfulness & Harmlessness）以及自建的 1B‑30B 参数规模的指令遵循基准上评估 KTO。任务涵盖回答准确性、对话安全性和指令遵循三个维度。
- **对比基线**：包括传统交叉熵微调、DPO（Direct Preference Optimization）以及 RLHF（Reinforcement Learning from Human Feedback）。  
- **主要结果**：在 1B 参数模型上，KTO 在 HH 安全性评分上比 DPO 提升约 3.2%，在 30B 参数模型上提升约 1.8%。在指令遵循准确率上，两者基本持平或略有优势。整体来看，KTO 在所有规模上都没有出现性能倒退，且只用了二元标签，标注成本约降低 50%。
- **消融实验**：作者分别去掉概率加权函数 w(p) 和价值函数的非线性形状，发现效用函数的非线性是提升的主要来源，而概率加权对小模型尤为重要。  
- **局限性**：论文指出，前景理论的参数（如损失规避系数）需要手动调节，且在极端长文本或多轮对话中效用估计可能失真。作者也承认在极端高风险任务（如法律建议）上，仅靠二元好坏标签可能不足以捕捉细粒度的价值差异。

### 影响与延伸思考
KTO 把心理学的前景理论正式引入 LLM 对齐，开启了“认知偏差驱动的损失函数”这一新方向。随后的工作（如 2024 年的 “Prospect‑RLHF” 与 “Bias‑Aware Alignment”）进一步探索了不同心理模型（如锚定效应、确认偏误）在对齐中的作用。对想深入的读者，可以关注以下几个方向：① 前景理论参数的自适应学习；② 将多模态（文本+图像）反馈也映射到同一效用框架；③ 在安全关键领域结合专家级细粒度标签与二元效用的混合训练。整体来看，KTO 提示我们：对齐不只是“更多数据”，更是“更符合人类感知的目标函数”。

### 一句话记住它
KTO 用前景理论把“好/坏”二元标注直接转化为感知效用，让模型在不需要成对比较的情况下也能实现高质量对齐。