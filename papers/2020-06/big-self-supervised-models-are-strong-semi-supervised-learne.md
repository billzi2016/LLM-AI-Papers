# Big Self-Supervised Models are Strong Semi-Supervised Learners

> **Date**：2020-06-17
> **arXiv**：https://arxiv.org/abs/2006.10029

## Abstract

One paradigm for learning from few labeled examples while making best use of a large amount of unlabeled data is unsupervised pretraining followed by supervised fine-tuning. Although this paradigm uses unlabeled data in a task-agnostic way, in contrast to common approaches to semi-supervised learning for computer vision, we show that it is surprisingly effective for semi-supervised learning on ImageNet. A key ingredient of our approach is the use of big (deep and wide) networks during pretraining and fine-tuning. We find that, the fewer the labels, the more this approach (task-agnostic use of unlabeled data) benefits from a bigger network. After fine-tuning, the big network can be further improved and distilled into a much smaller one with little loss in classification accuracy by using the unlabeled examples for a second time, but in a task-specific way. The proposed semi-supervised learning algorithm can be summarized in three steps: unsupervised pretraining of a big ResNet model using SimCLRv2, supervised fine-tuning on a few labeled examples, and distillation with unlabeled examples for refining and transferring the task-specific knowledge. This procedure achieves 73.9% ImageNet top-1 accuracy with just 1% of the labels ($\le$13 labeled images per class) using ResNet-50, a $10\times$ improvement in label efficiency over the previous state-of-the-art. With 10% of labels, ResNet-50 trained with our method achieves 77.5% top-1 accuracy, outperforming standard supervised training with all of the labels.

---

# 大规模自监督模型是强大的半监督学习者 论文详细解读

### 背景：这个问题为什么难？
在视觉任务中，标注一张图片往往要花费几分钟甚至更久，而大规模数据集（比如 ImageNet）拥有上千万张未标记的图片。传统的半监督学习大多依赖于“任务相关”的未标记数据，例如通过一致性正则让模型在不同增强下输出相似结果，但这些方法在 ImageNet 这类高分辨率、类别众多的场景里往往只能略微提升性能。更关键的是，过去的研究倾向于使用相对小的网络结构——因为大模型训练成本高、容易过拟合——导致在标签极少的情况下，模型的表达能力受限，难以充分挖掘海量未标记数据的价值。

### 关键概念速览
**自监督学习**：让模型自己生成监督信号，例如把一张图的两个随机裁剪当作正样本对，模型要学会把它们映射到相近的特征空间。类似于让孩子自己玩拼图，从中学会形状和颜色的关联。

**SimCLR v2**：一种对比学习框架，核心是把同一张图的不同增强视为正例，其他图为负例，并使用更大的投影头和更强的正则。可以把它想成“让模型在噪声中辨认出同一张照片的不同版本”。

**大模型（big network）**：指网络的深度和宽度都显著提升的模型，例如 ResNet‑50 在预训练阶段使用更宽的通道数或更深的层数。它像是拥有更多“大脑神经元”，能捕捉更细腻的视觉模式。

**微调（fine‑tuning）**：在已有的预训练权重上，用少量标记数据继续训练，使模型适应具体任务。相当于把已经学会通用语言的学生，针对某门专业课进行强化训练。

**蒸馏（distillation）**：把大模型的知识转移到小模型里，通常通过让小模型模仿大模型在未标记数据上的预测。类似于老师把自己的解题思路写在黑板上，学生抄下来后自己再练习。

**标签效率（label efficiency）**：在固定的标记预算下，模型能达到的准确率。标签越少，效率越高，说明模型对未标记数据的利用越好。

### 核心创新点
1. **任务无关的自监督预训练 + 大模型**  
   过去的半监督方法往往在预训练阶段就加入任务信息，或者使用相对小的网络。这里先用 SimCLRv2 对一个非常宽深的 ResNet 进行纯自监督预训练，完全不看标签。实验显示，网络越大，在标签极少时提升越明显，突破了“小模型才是半监督利器”的常规认知。

2. **少标签微调直接使用大模型**  
   与常规做法在微调阶段先把模型压缩不同，这里直接在大模型上用几百到几千张标记图片进行微调。结果表明，即使标签只占全部的 1%（约 13 张/类），大模型也能快速收敛到 73.9% 的 top‑1 准确率，远超之前的最优记录。

3. **二次利用未标记数据进行蒸馏**  
   在微调结束后，再把同一批未标记图片喂给大模型，获取它对这些图片的软标签（概率分布），然后用这些软标签训练一个小的 ResNet‑50。这样既保留了大模型的任务特化知识，又大幅降低推理成本，几乎没有精度损失。

4. **统一的三步流程**  
   将“自监督预训练 → 少标签微调 → 蒸馏”串成一条流水线，省去繁杂的多阶段调参。每一步都只需要一次数据遍历，整体计算开销与传统半监督方法相当，却得到更高的标签效率。

### 方法详解
**整体框架**  
整个方案分为三步：  
1）在全量未标记 ImageNet 上，用 SimCLRv2 对一个扩大的 ResNet 进行自监督预训练；  
2）把预训练得到的权重直接加载到同样大小的网络上，用仅有的标记子集（1%~10%）进行监督微调；  
3）利用微调后的大模型对未标记图片生成软标签，再用这些软标签训练一个更小的网络，实现知识蒸馏。

**步骤拆解**  
- **自监督预训练**：每张图片随机做两次强增强（裁剪、颜色抖动、模糊等），送入同一网络得到两个特征向量。再通过一个投影头把向量映射到对比空间，使用 InfoNCE 损失让同图的两个向量相互靠近，其他图片的向量保持距离。SimCLRv2 在此基础上加入了更大的投影头和更强的正则，使得特征更具判别性。这里的网络宽度比标准 ResNet‑50 多出约 2 倍，深度保持不变。

- **少标签微调**：把预训练好的权重直接用于分类头（线性层），只在标记数据上继续训练。因为特征已经非常丰富，分类层只需要少量梯度更新即可适配。训练时仍保留一定的学习率衰减和数据增强，以防过拟合。

- **蒸馏**：微调完成后，冻结大模型，遍历所有未标记图片，记录它在分类头的软概率分布（温度调节后的 logits）。随后，用这些软标签作为目标，训练一个结构更紧凑的 ResNet‑50。损失函数是交叉熵加上 KL 散度，使小模型的输出分布尽量逼近大模型。

**关键细节**  
- **网络规模的选择**：实验发现，网络宽度的提升对低标签比例的收益尤为显著，而深度的增加收益相对平缓。  
- **两次使用未标记数据的策略**：第一次是任务无关的对比学习，第二次是任务特化的软标签蒸馏，形成了“先学通用再学专用”的双重利用。  
- **温度调节**：在蒸馏阶段使用较高的温度，使得软标签分布更平滑，帮助小模型捕捉大模型的细粒度知识。

### 实验与效果
- **数据集**：全部实验在 ImageNet（1.28M 训练图，1000 类）上进行，未标记数据即全体训练集，标记子集分别取 1%（≈13 张/类）和 10%（≈130 张/类）。  
- **主要结果**：使用 1% 标记时，ResNet‑50 在三步流程后达到 73.9% top‑1 准确率，比之前最好的半监督方法提升约 10% 绝对值，标签效率提升约 10 倍。使用 10% 标记时，准确率为 77.5%，已经超过了全标签监督训练的 76.2%（原始 ResNet‑50）。  
- **基线对比**：与 FixMatch、UDA、SimCLR（仅预训练不蒸馏）等方法相比，本文在相同标记比例下均有显著优势。  
- **消融实验**：作者分别去掉“大模型预训练”“蒸馏阶段”“使用 SimCLRv2 投影头”等组件，发现去掉任意一环都会导致准确率下降 2%~4%，验证了每一步的必要性。  
- **局限性**：方法对计算资源要求较高，尤其是大模型的自监督预训练需要数十个 GPU 天的算力；在更小的数据集或非自然图像上未做评估，推广性仍待验证。

### 影响与延伸思考
这篇工作让业界重新审视“模型大小”在半监督学习中的作用，推动了后续的大模型自监督+蒸馏路线。随后出现的 BYOL‑large、MoCo‑v3 等都在更大规模的网络上进行对比学习，并尝试类似的两阶段蒸馏。还有研究把这种“三步走”扩展到目标检测、语义分割等下游任务，证明了跨任务的通用性。想进一步深入，可以关注以下方向：① 更高效的分布式预训练技术，降低大模型门槛；② 在少标签场景下的自适应网络裁剪策略；③ 将大模型的自监督特征与少量标记的元学习结合，提升快速适应新任务的能力。

### 一句话记住它
**把超大自监督模型直接微调再蒸馏，少量标签也能把 ImageNet 的准确率推到几乎全监督的水平。**