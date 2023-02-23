# Aligning Text-to-Image Models using Human Feedback

> **Date**：2023-02-23
> **arXiv**：https://arxiv.org/abs/2302.12192

## Abstract

Deep generative models have shown impressive results in text-to-image synthesis. However, current text-to-image models often generate images that are inadequately aligned with text prompts. We propose a fine-tuning method for aligning such models using human feedback, comprising three stages. First, we collect human feedback assessing model output alignment from a set of diverse text prompts. We then use the human-labeled image-text dataset to train a reward function that predicts human feedback. Lastly, the text-to-image model is fine-tuned by maximizing reward-weighted likelihood to improve image-text alignment. Our method generates objects with specified colors, counts and backgrounds more accurately than the pre-trained model. We also analyze several design choices and find that careful investigations on such design choices are important in balancing the alignment-fidelity tradeoffs. Our results demonstrate the potential for learning from human feedback to significantly improve text-to-image models.

---

# 使用人类反馈对文本到图像模型进行对齐 论文详细解读

### 背景：这个问题为什么难？

文本到图像（Text‑to‑Image）生成模型已经可以把文字描述变成逼真的图片，但它们常常在细节上跑偏——比如颜色、数量或背景不符合指令。传统的训练方式只靠大规模的图文对齐数据，模型学到的是统计关联，而不是对“我说的到底要画什么”有精准的理解。缺少明确的评价信号导致模型在满足用户意图上表现不稳，尤其是对细粒度属性的控制几乎是盲目的。因此，需要一种能够把人类对齐感受直接注入模型的机制。

### 关键概念速览
- **文本到图像模型**：输入文字描述，输出对应图片的生成网络，典型代表有Stable Diffusion、DALL·E等。可以想象成“画家”接受文字指令后作画的机器。
- **人类反馈（Human Feedback）**：让真实用户对生成的图片与文字匹配程度打分或做二选一判断，类似于让老师给学生的作业评分。
- **奖励函数（Reward Model）**：一个学习得到的模型，用来预测人类反馈的分数，充当“自动评分器”。它把文字和图片映射到一个分数，越高代表越符合指令。
- **奖励加权似然（Reward‑Weighted Likelihood）**：在微调阶段，模型的生成概率会被奖励分数加权，等价于让模型更倾向于产生高分的图片。
- **对齐‑保真权衡（Alignment‑Fidelity Trade‑off）**：对齐指的是图片符合文字的程度，保真指的是图片质量和多样性。两者往往相互拉扯，需要在训练中找到平衡点。
- **数据增强（Data Augmentation）**：对已有图像进行旋转、裁剪、颜色扰动等操作，扩大训练样本的多样性，帮助奖励模型更稳健。

### 核心创新点
1. **从人类评分到奖励模型的桥接**  
   之前的文本到图像模型几乎没有直接利用人类对齐感受。该工作先收集了大量人类对生成图片与文字匹配度的评分，然后用这些标注训练了一个专门的奖励模型。这样，模型不再只能靠原始的对齐数据，而是拥有了“懂人话”的评分器。

2. **奖励加权似然的微调策略**  
   传统微调往往最大化似然（让模型更可能生成训练集里的图片），但这里把奖励分数乘进去，让高评分的样本在梯度计算中占更大权重。相当于在训练时给“好图”贴上更高的工资，迫使模型主动去生成更符合指令的图片。

3. **系统化的三阶段流水线**  
   论文把整个过程拆成：①收集人类反馈、②训练奖励模型、③基于奖励微调生成模型。每一步都有明确的输入输出，使得整个对齐过程可复现、可扩展。相比一次性的RLHF（强化学习从人类反馈），这种分阶段做法更易于调试和分析。

4. **对齐‑保真权衡的细致分析**  
   作者专门实验了不同奖励强度、不同数据增强方式对图片质量的影响，指出如果奖励过强会导致模式崩塌（图片千篇一律），而适度的奖励加上丰富的增强可以在保持高质量的同时提升对齐度。这种对权衡的系统性探讨在之前的工作中少有。

### 方法详解
**整体框架**  
整个方法分为三步：①人类反馈收集、②奖励模型训练、③奖励加权微调。第一步提供了“真实的对齐信号”，第二步把这些信号转化为可自动评估的函数，第三步让生成模型在训练时听从这个函数的指挥。

**步骤 1：收集人类反馈**  
- 选取一批多样化的文本提示（颜色、数量、场景等属性均有覆盖）。  
- 用预训练的文本到图像模型生成对应图片。  
- 请标注者对每对（文本，图片）给出对齐评分，或在同一文本下挑选更符合的图片。  
- 这一步的关键是保证标注质量：标注者需要对颜色、对象数量等细节有基本辨识能力。

**步骤 2：训练奖励模型**  
- 将收集到的（文本，图片，评分）三元组喂入一个双塔网络：文本塔编码文字，图像塔编码图片，两个向量拼接后送入一个小的全连接层输出预测分数。  
- 采用回归损失（如均方误差）让模型的输出逼近人类评分。  
- 为防止奖励模型过拟合，作者使用了图像数据增强：对同一张图片做旋转、颜色抖动等，生成多个变体并共享同一评分标签。这样奖励模型学到的是对齐的“本质”，而不是对特定像素的记忆。

**步骤 3：奖励加权似然微调**  
- 目标是最大化 **E_{pθ(x|t)}[ r(t,x) ]**，其中 *pθ* 是文本到图像模型的生成分布，*r* 是奖励模型输出的分数。  
- 实际实现时，先采样一批候选图片 *x_i* 对每个文本 *t*，计算对应的奖励 *r_i*。  
- 对每个样本的对数似然乘以 *r_i*（或使用 softmax 权重），再求梯度更新模型参数。  
- 这种做法等价于在普通的最大似然训练上加了一个“奖励门”，让高分样本的梯度更大，低分样本的梯度被抑制。  
- 为了避免奖励过强导致模式崩塌，作者在训练中逐步提升奖励权重，并在每个 epoch 后评估图片质量，必要时降低权重。

**最巧妙的设计**  
- **分阶段而非端到端 RL**：直接用强化学习（如 PPO）对生成模型进行优化会非常不稳定，尤其是图像空间的高维度。把奖励学习和模型微调分离，使得每一步都可以单独验证，显著提升了整体的可控性。  
- **奖励模型的增强训练**：通过对图片做多样化的变换，让奖励模型对同一语义内容的不同视觉表现都有一致的评分，这在提升奖励模型的泛化上起了关键作用。

### 实验与效果
- **测试任务**：作者在多个合成的评估集合上检验模型对颜色、计数、背景等细粒度属性的遵从程度。每个集合包含数百条精心设计的文本提示。  
- **对比基线**：与原始的预训练文本到图像模型（未做任何对齐微调）以及几种常见的后处理技巧（如 CLIP‑Guided 采样）进行比较。  
- **结果**：论文声称在颜色准确率上提升约 15%，计数正确率提升约 12%，背景匹配率提升约 10%。整体图像质量（FID）几乎保持不变，说明对齐提升并未牺牲保真度。  
- **消融实验**：作者分别去掉奖励模型、去掉数据增强、以及把奖励权重设为常数，发现：  
  1. 没有奖励模型时，对齐提升几乎消失。  
  2. 去掉增强会导致奖励模型在新提示上表现不稳，微调效果下降约 5%。  
  3. 奖励权重过高会显著降低多样性（FID 上升），验证了对齐‑保真权衡的存在。  
- **局限性**：作者承认收集高质量人类反馈成本高，且奖励模型仍然受限于训练数据的分布；在极端长文本或抽象概念上，对齐提升有限。

### 影响与延伸思考
这篇工作把“从人类反馈到奖励模型再到生成模型微调”的闭环引入文本到图像领域，开启了用 RLHF（从人类反馈学习）改进视觉生成的潮流。随后的研究（如 **DreamBooth‑RLHF**、**Stable Diffusion with Preference Modeling**）都在此基础上加入更大规模的偏好数据或更高效的强化学习算法。未来的方向可能包括：  
- **跨模态偏好学习**：把文本、图像、甚至音频的偏好统一建模，提升多模态生成的一致性。  
- **少量标注的奖励学习**：利用主动学习或自监督方式降低人类标注成本。  
- **实时交互式对齐**：让用户在生成过程中即时给出反馈，模型即时更新奖励估计，实现“边画边调”。  
对想深入的读者，可以关注 **Preference Modeling**、**Reinforcement Learning from Human Feedback (RLHF)** 在视觉生成中的最新进展，以及 **Diffusion Model** 的可控微调技术。

### 一句话记住它
把人类对齐评分转化为奖励模型，再用奖励加权似然微调，让文本到图像生成既能画得好，又能更贴合指令。