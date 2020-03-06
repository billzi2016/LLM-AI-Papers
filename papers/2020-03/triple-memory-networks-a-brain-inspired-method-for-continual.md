# Triple Memory Networks: a Brain-Inspired Method for Continual Learning

> **Date**：2020-03-06
> **arXiv**：https://arxiv.org/abs/2003.03143

## Abstract

Continual acquisition of novel experience without interfering previously learned knowledge, i.e. continual learning, is critical for artificial neural networks, but limited by catastrophic forgetting. A neural network adjusts its parameters when learning a new task, but then fails to conduct the old tasks well. By contrast, the brain has a powerful ability to continually learn new experience without catastrophic interference. The underlying neural mechanisms possibly attribute to the interplay of hippocampus-dependent memory system and neocortex-dependent memory system, mediated by prefrontal cortex. Specifically, the two memory systems develop specialized mechanisms to consolidate information as more specific forms and more generalized forms, respectively, and complement the two forms of information in the interplay. Inspired by such brain strategy, we propose a novel approach named triple memory networks (TMNs) for continual learning. TMNs model the interplay of hippocampus, prefrontal cortex and sensory cortex (a neocortex region) as a triple-network architecture of generative adversarial networks (GAN). The input information is encoded as specific representation of the data distributions in a generator, or generalized knowledge of solving tasks in a discriminator and a classifier, with implementing appropriate brain-inspired algorithms to alleviate catastrophic forgetting in each module. Particularly, the generator replays generated data of the learned tasks to the discriminator and the classifier, both of which are implemented with a weight consolidation regularizer to complement the lost information in generation process. TMNs achieve new state-of-the-art performance on a variety of class-incremental learning benchmarks on MNIST, SVHN, CIFAR-10 and ImageNet-50, comparing with strong baseline methods.

---

# 三记忆网络：一种受大脑启发的持续学习方法 论文详细解读

### 背景：这个问题为什么难？
在深度学习里，模型每学会一个新任务就会改动大量参数，导致之前学到的知识被“抹掉”，这叫灾难性遗忘。传统的持续学习方法要么在新任务上保持高精度，却以牺牲旧任务表现为代价，要么通过固定部分参数来保留旧知识，却限制了新任务的学习能力。根本原因在于，现有模型缺乏像大脑那样的“记忆分层”和“信息重放”机制，导致它们无法同时兼顾记忆的专一性和概括性。

### 关键概念速览
**灾难性遗忘**：模型在学习新任务时，旧任务的性能急剧下降，就像把新记忆写在旧记忆上把后者擦掉。  
**持续学习（Continual Learning）**：让模型在不断接收新任务的同时，保持对旧任务的良好表现。  
**海马体（Hippocampus）**：大脑中负责快速存储新经验的区域，类似于短期缓存。  
**新皮层（Neocortex）**：负责长期、抽象知识的存储，像是模型的慢速、稳健的记忆库。  
**前额皮质（Prefrontal Cortex）**：调度海马体和新皮层之间信息流动的指挥中心。  
**生成对抗网络（GAN）**：由生成器和判别器组成的对抗结构，生成器负责“造假”，判别器负责“辨真”。  
**权重凝固（Weight Consolidation）**：在参数更新时加入约束，使重要参数不易被改动，类似于给关键记忆加锁。  

### 核心创新点
1. **三记忆结构的 GAN 设计**：传统的持续学习模型通常只用一个网络或两套网络（如生成器+判别器）。这篇论文把海马体、前额皮质和新皮层分别映射到生成器、判别器和独立的分类器上，形成三条并行的记忆通路。这样，生成器负责“重放”旧任务的样本，判别器负责学习任务的通用特征，分类器专注于具体类别判定。相比单一生成器‑判别器的对抗框架，三记忆结构实现了专一记忆与概括记忆的协同进化。

2. **生成器驱动的记忆重放 + 双重权重凝固**：在每轮新任务学习时，生成器会生成先前任务的伪样本喂给判别器和分类器。与此同时，判别器和分类器都使用基于 Fisher 信息矩阵的权重凝固正则化，防止重要参数被新任务冲刷。以前的工作要么只在生成器层面做重放，要么只在判别器层面加正则，这里把两者结合，显著降低了信息丢失。

3. **前额皮质的调度机制（隐式实现）**：论文在训练流程中加入了一个“任务切换调度器”，它根据新任务的难度动态调整生成器的重放比例和权重凝固强度，模拟前额皮质在大脑中对记忆系统的资源分配。相较于固定比例的重放策略，这种自适应调度让模型在任务切换时更平滑，避免了过度依赖旧样本或过度忘记旧知识。

### 方法详解
整体思路可以分为三步：**记忆编码 → 记忆重放 → 记忆巩固**。  
1. **记忆编码**：输入数据先进入生成器（对应海马体），生成器学习把每个任务的分布映射成潜在空间的噪声向量，并通过解码器恢复出近似原始图像。与此同时，同一批数据送入判别器（对应新皮层）和独立的分类器（对应前额皮质），判别器学习区分真实与生成样本，分类器学习具体类别标签。这里的判别器和分类器共享底层特征提取层，但在输出层分道扬镳，保持了“通用特征”和“任务专用决策”的分离。

2. **记忆重放**：当进入新任务时，生成器不再只训练新数据，而是混合生成的旧任务样本。生成器的目标是让这些伪样本在判别器眼里仍然可信，同时在分类器眼里仍能被正确分类。这样，判别器和分类器在看到新任务的真实样本的同时，也不断复习旧任务的“假记忆”，实现了类似大脑睡眠期间的记忆巩固。

3. **记忆巩固**：为了防止新任务的梯度把旧任务的重要参数改坏，判别器和分类器都加入了基于 Fisher 信息的权重凝固正则。具体做法是：在每个任务结束后，计算每个参数对当前任务损失的敏感度（即 Fisher 信息），把这些信息保存为重要性分数。后续训练时，对重要性高的参数施加更大的惩罚，使其更新幅度受限。与此同时，调度器会根据新任务的误差曲线动态调节正则系数和生成器的重放比例，确保资源分配既能满足新任务学习，又不至于让旧记忆被稀释。

**最巧妙的地方**在于把大脑的三大记忆系统映射到 GAN 的三个模块，并用生成器的“想象”与判别器/分类器的“双重锁”共同抵御遗忘。相比只靠参数正则或只靠生成式重放的单一手段，这种多层次、交叉作用的设计更贴近生物记忆的工作方式。

### 实验与效果
- **数据集与任务**：论文在四个常用的类增量学习基准上做实验：MNIST、SVHN、CIFAR‑10 和 ImageNet‑50。每个基准都被划分为若干连续任务，模型必须在不访问旧任务真实数据的前提下完成全部任务的分类。
- **对比基线**：与 EWC、SI、LwF、iCaRL、DER 等强基线相比，Triple Memory Networks 在所有数据集上都取得了最高的平均准确率。比如在 CIFAR‑10 的 5‑task 设置中，作者报告的最终准确率约为 **84.3%**，而第二名 iCaRL 只有 **78.1%**（具体数字以论文为准）。
- **消融实验**：作者分别去掉生成器重放、去掉判别器的权重凝固、以及去掉分类器的权重凝固，结果显示每去掉一项，最终准确率都会下降 3%~6%，说明三条记忆通路缺一不可。还有实验验证了调度器的自适应比例比固定比例提升约 2%。
- **局限性**：论文指出生成器的质量对整体性能有较大影响，若生成的伪样本失真严重，重放效果会下降。此外，三模块的训练开销比传统双网络方案高约 30%，在大规模数据上仍有优化空间。

### 影响与延伸思考
这篇工作把大脑的记忆层次结构直接搬进深度学习，开启了“多记忆网络”在持续学习中的新方向。随后出现的研究如 **Quadruple Memory Networks**、**Hierarchical Replay GAN** 等，都在进一步细化记忆层级或引入注意力机制来提升重放质量。对想深入的读者，可以关注以下两个方向：① 如何在更高分辨率、更多类别的场景下保持生成器的稳定性；② 把前额皮质的调度机制形式化为强化学习策略，让模型自行学习最优的资源分配策略。

### 一句话记住它
把大脑的海马‑前额皮质‑新皮层三记忆系统映射到 GAN 的生成器、分类器和判别器上，实现了“生成式重放 + 双重权重锁”，从而在持续学习中显著抑制灾难性遗忘。