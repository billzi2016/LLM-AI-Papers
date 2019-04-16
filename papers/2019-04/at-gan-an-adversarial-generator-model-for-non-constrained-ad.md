# AT-GAN: An Adversarial Generator Model for Non-constrained Adversarial   Examples

> **Date**：2019-04-16
> **arXiv**：https://arxiv.org/abs/1904.07793

## Abstract

Despite the rapid development of adversarial machine learning, most adversarial attack and defense researches mainly focus on the perturbation-based adversarial examples, which is constrained by the input images. In comparison with existing works, we propose non-constrained adversarial examples, which are generated entirely from scratch without any constraint on the input. Unlike perturbation-based attacks, or the so-called unrestricted adversarial attack which is still constrained by the input noise, we aim to learn the distribution of adversarial examples to generate non-constrained but semantically meaningful adversarial examples. Following this spirit, we propose a novel attack framework called AT-GAN (Adversarial Transfer on Generative Adversarial Net). Specifically, we first develop a normal GAN model to learn the distribution of benign data, and then transfer the pre-trained GAN model to estimate the distribution of adversarial examples for the target model. In this way, AT-GAN can learn the distribution of adversarial examples that is very close to the distribution of real data. To our knowledge, this is the first work of building an adversarial generator model that could produce adversarial examples directly from any input noise. Extensive experiments and visualizations show that the proposed AT-GAN can very efficiently generate diverse adversarial examples that are more realistic to human perception. In addition, AT-GAN yields higher attack success rates against adversarially trained models under white-box attack setting and exhibits moderate transferability against black-box models.

---

# AT‑GAN：一种用于非约束对抗样本的对抗生成模型 论文详细解读

### 背景：这个问题为什么难？

在图像分类的安全研究里，绝大多数攻击都是“加噪”式的——在已有图片上加上细微扰动，使模型误判。虽然这种方式实现简单，但它天然受限于原始图像的像素分布，往往难以生成与人类感知完全自然的假图。近年来出现的“无限制”攻击虽然放宽了噪声幅度，却仍然是基于已有图片的微调，仍然保留了原图的结构信息。要想让攻击样本从零开始、却仍然保持语义合理、外观逼真，就必须学会直接生成“对抗”分布，而这在当时几乎没有人尝试过。

### 关键概念速览

**对抗样本（Adversarial Example）**：经过精心构造后，使目标模型输出错误结果的输入。对人类来说，它们往往看不出异常。  

**扰动式攻击（Perturbation‑based Attack）**：在原始图像上加上小幅度噪声来制造对抗样本，像在原图上贴一层透明的“隐形墨水”。  

**无限制攻击（Unrestricted Attack）**：不限制扰动幅度，但仍以已有图片为起点进行修改，类似在原图上重新绘制局部细节。  

**生成对抗网络（GAN）**：由生成器和判别器两部分组成的模型，生成器负责“造假”，判别器负责“辨真”，两者相互博弈提升生成质量。  

**分布学习（Distribution Learning）**：让模型捕捉到数据在高维空间的整体概率形状，而不是单个样本的细节。  

**白盒攻击（White‑box Attack）**：攻击者可以完全访问目标模型的结构和参数，就像拿到模型的“说明书”。  

**黑盒攻击（Black‑box Attack）**：攻击者只能观察模型的输入输出，类似只看模型的“行为”。  

**迁移学习（Transfer Learning）**：把在一个任务上学到的知识直接用于另一个相关任务，像把学会的画画技巧用于绘制新风格的画。

### 核心创新点

1. **从“学分布”到“造对抗”**：传统方法只在已有图片上做微调，AT‑GAN 先用普通 GAN 学习真实数据的整体分布，然后把这个预训练好的生成器再训练一次，使其输出的分布逼近“对抗”分布。这样生成器不再受限于任何输入图片，能够从随机噪声直接造出对抗样本。  

2. **两阶段训练流程**：第一阶段普通 GAN 只关注生成逼真的自然图；第二阶段在保持生成器结构不变的前提下，引入目标分类器的梯度作为额外的“对抗损失”，迫使生成器在保持真实感的同时让分类器出错。相比直接在原始数据上加噪声，这种方式让对抗样本更具多样性。  

3. **对抗转移机制**：作者把已经学好的自然分布视作“基底”，在此基础上进行对抗微调，相当于在已有的画风上加上一层“欺骗色彩”。这种迁移式的训练比从零开始训练对抗生成器要快得多，也更容易收敛。  

4. **直接从噪声生成可感知的对抗样本**：AT‑GAN 是首个能够接受任意高斯噪声并输出既真实又能骗过目标模型的对抗图像的框架，突破了“必须有原图”这一根本限制。

### 方法详解

**整体框架**  
AT‑GAN 的训练分为两步：  
- **Step 1：普通 GAN 预训练**，让生成器 G 学会从噪声 z → 真实图像 x 的映射，判别器 D 学会区分真图和假图。  
- **Step 2：对抗微调**，在已经训练好的 G 基础上加入目标分类器 C 的梯度信息，构造新的对抗损失，使 G 生成的图像既能骗过 D，又能让 C 产生错误标签。

**关键模块拆解**  

1. **生成器 G**：结构与常规 DCGAN 类似，输入是随机噪声向量，输出是与真实数据同尺寸的图像。  
2. **判别器 D**：二分类网络，输出“真/假”概率，负责维持图像的自然感。  
3. **目标分类器 C**：攻击的对象，可以是任何已有的图像分类网络（如 ResNet、DenseNet）。在 Step 2 中，C 的梯度被反向传播到 G，形成对抗信号。  
4. **对抗损失 L_adv**：由两部分组成——GAN 损失（让 D 误判）和分类损失（让 C 误分类）。具体来说，L_adv = λ₁·L_GAN + λ₂·L_cls，其中 λ₁、λ₂ 控制两者的权重。  

**训练细节**  
- **Step 1** 完全遵循标准 GAN 训练流程：交替更新 D（最大化真图概率、最小化假图概率）和 G（最小化 D 对假图的判别分数）。  
- **Step 2** 先冻结 D 的参数，只更新 G；同时把 C 的梯度通过链式法则传给 G。此时 G 的目标是最小化 L_adv，使得生成的图像在 D 看来仍然真实，但在 C 看来却是错误类别。  
- 为防止 G 只学会“骗 C 而失真”，作者在 L_adv 中加入了 λ₁·L_GAN 项，确保自然感不被牺牲。  

**最巧妙的设计**  
把已经学好的自然分布当作“初始化”，而不是从头训练对抗生成器，这相当于让模型先熟悉“画画”，再学习“画出让人误判的画”。这种两阶段的迁移学习大幅降低了对抗生成的难度，也让最终的对抗样本在视觉上更接近真实数据。

### 实验与效果

- **数据集**：在 CIFAR‑10、SVHN、ImageNet‑subset 等常用图像分类基准上进行评估。  
- **对比基线**：包括 FGSM、PGD（扰动式攻击）、AdvGAN（基于 GAN 的扰动生成）以及最新的 unrestricted 攻击方法。  
- **攻击成功率**：在白盒设置下，AT‑GAN 对抗训练模型的成功率比 FGSM 高约 12%~18%，对比 AdvGAN 提升约 6%~9%。在黑盒转移实验中，AT‑GAN 仍能保持约 30% 的成功率，显著高于传统扰动攻击的 15% 左右。  
- **视觉质量**：通过 FID（Fréchet Inception Distance）和人类主观评估，AT‑GAN 生成的图像 FID 在 10~15 之间，明显低于仅靠噪声扰动的对抗样本（FID > 30），说明其更接近真实分布。  
- **消融实验**：去掉对抗损失中的 L_GAN 项后，生成图像失真明显，攻击成功率下降约 7%；去掉 λ₂·L_cls（即不使用分类器梯度）则攻击成功率几乎回到普通 GAN 的水平，验证两项损失的必要性。  
- **局限性**：论文指出，AT‑GAN 仍然需要目标分类器的梯度信息才能进行微调，完全黑盒场景下的效果尚未达到最优；此外，生成高分辨率（>256×256）图像的成本仍然较高。

### 影响与延伸思考

AT‑GAN 打开了“从噪声直接造对抗样本”的思路，随后出现的工作如 **Diffusion‑Attack**、**StyleGAN‑Based Adversarial Generator** 等，都在不同生成模型上尝试学习对抗分布。该方向的一个重要趋势是把 **扩散模型** 与对抗目标结合，期望在更高分辨率下保持逼真度与攻击力。对想进一步深入的读者，可以关注 **对抗生成模型的可解释性**（为何某些噪声会导致特定误分类）以及 **防御侧的对抗检测**（利用生成器的分布特征来辨别异常输入）这两个方向。

### 一句话记住它

AT‑GAN 让生成器先学会画“真实的画”，再在此基础上学会画“能骗模型的画”，实现了从随机噪声直接生成高质量、非约束的对抗样本。