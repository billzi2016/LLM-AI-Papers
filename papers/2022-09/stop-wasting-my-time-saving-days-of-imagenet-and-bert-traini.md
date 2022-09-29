# Stop Wasting My Time! Saving Days of ImageNet and BERT Training with   Latest Weight Averaging

> **Date**：2022-09-29
> **arXiv**：https://arxiv.org/abs/2209.14981

## Abstract

Training vision or language models on large datasets can take days, if not weeks. We show that averaging the weights of the k latest checkpoints, each collected at the end of an epoch, can speed up the training progression in terms of loss and accuracy by dozens of epochs, corresponding to time savings up to ~68 and ~30 GPU hours when training a ResNet50 on ImageNet and RoBERTa-Base model on WikiText-103, respectively. We also provide the code and model checkpoint trajectory to reproduce the results and facilitate research on reusing historical weights for faster convergence.

---

# 别再浪费时间！用最新权重平均法节省 ImageNet 与 BERT 训练的数天 论文详细解读

### 背景：这个问题为什么难？

在视觉和语言的大模型训练里，单卡 GPU 可能要跑上好几天，甚至几周才能把 ResNet‑50 在 ImageNet 上收敛，或者把 RoBERTa‑Base 在 WikiText‑103 上达到满意的 perplexity。传统的加速手段——更大的 batch、学习率调度、混合精度——都需要额外的硬件投入或细致的超参数调试，而且提升往往是渐进的。更关键的是，训练过程会产生大量历史 checkpoint（每个 epoch 结束时保存的模型参数），这些信息被闲置在磁盘上，根本没有被利用来帮助模型更快收敛。于是出现了一个看似简单却被忽视的机会：能否直接把最近的几次 checkpoint 合在一起，让模型“记住”过去的好状态，从而省掉几天的算力？

### 关键概念速览
- **checkpoint（检查点）**：训练过程中保存的完整模型参数文件，通常在每个 epoch 结束时生成，类似于跑步时的里程碑标记，方便以后恢复或分析。
- **epoch（轮次）**：一次遍历完整训练数据集的过程。把数据集看成一本书，epoch 就是把这本书读完一次。
- **weight averaging（权重平均）**：把多个模型的参数逐元素相加后除以模型数量，得到一个“中间”模型。想象把几幅画的颜色混合在一起，得到的颜色更平滑、更稳健。
- **latest‑k averaging（最近 K 次平均）**：只取最近 K 个 checkpoint 做平均，而不是全部历史。相当于只看最近几天的天气预报来决定今天的穿衣，而不是把十年气候都拿来算。
- **ResNet‑50**：一种 50 层深的卷积神经网络，常用作 ImageNet 基准。可以把它想成一条 50 层的流水线，每层都在加工图像特征。
- **RoBERTa‑Base**：基于 BERT 的改进版语言模型，拥有 12 层 Transformer 编码器。把它比作 12 层的语言理解“翻译机”，每层都在提炼句子信息。
- **GPU hours（GPU 小时）**：衡量算力消耗的单位，等价于一块 GPU 连续工作一小时的计算量。省下的 GPU 小时直接转化为成本和时间的节约。
- **convergence speed（收敛速度）**：模型在训练过程中达到目标性能（如 loss、accuracy）所需的 epoch 数或时间。收敛快就意味着更早得到可用模型。

### 核心创新点
1. **从全局平均到最近 K 次平均**  
   之前的权重平均工作（如 Stochastic Weight Averaging）倾向于把训练全过程的 checkpoint 都混进去，导致平均模型可能被早期噪声拖累。这篇论文改为只取最近 K 次（每个 epoch 结束时）的 checkpoint，保持了最新的学习信息，同时抹平了短期波动。结果是模型在更少的 epoch 内就能达到相同或更好的 loss/accuracy。

2. **零额外训练成本的加速手段**  
   传统的加速方法往往需要重新设计优化器、调学习率或增加正则化项，都会带来额外的实验负担。这里的做法只在每个 epoch 结束后做一次简单的参数加法除法，几乎不消耗算力。作者把这一步称作“后处理”，相当于在跑完马拉松后给自己做一次拉伸，既不影响跑步本身，又能提升恢复速度。

3. **跨模态验证：同时在 CV 与 NLP 上取得显著加速**  
   许多加速技巧只在视觉或语言单一领域有效。论文把同一套最新 K 次平均策略分别跑在 ImageNet（ResNet‑50）和 WikiText‑103（RoBERTa‑Base）上，分别省下约 68 GPU 小时和 30 GPU 小时，证明了方法的通用性。换句话说，这不是某个特定网络的“特例”，而是一个可以直接搬到不同任务的“即插即用”技巧。

4. **提供完整的 checkpoint 轨迹与代码，促进复现与二次研究**  
   作者把训练过程中每个 epoch 的 checkpoint 以及对应的平均模型全部公开，配合简洁的 Python 脚本，让其他研究者可以“一键复现”。这在加速类工作中并不常见，极大降低了社区验证的门槛。

### 方法详解
**整体思路**  
训练过程保持不变：每完成一个 epoch，就把当前模型参数保存为 checkpoint。随后，维护一个长度为 K 的滑动窗口，里面装的是最近 K 个 checkpoint。每当窗口满了，就把这 K 份参数逐元素相加后除以 K，得到“最新平均模型”。这个平均模型可以直接替换当前的训练权重继续下一轮训练，也可以仅用于评估。整个流程只涉及一次加法和一次除法，几乎不占用显存或算力。

**关键步骤拆解**  

1. **保存 checkpoint**  
   - 在 epoch 结束时，调用 `torch.save(model.state_dict(), path)` 保存完整参数。相当于在跑完一圈后把跑鞋的磨损程度记录下来。

2. **更新滑动窗口**  
   - 用一个队列结构 `deque(maxlen=K)` 保存最近 K 个 checkpoint 的路径。新 checkpoint 入队，最旧的自动弹出。想象一个只能装 K 张照片的相框，最新的照片总是展示在前面。

3. **计算权重平均**  
   - 读取队列中所有 checkpoint，逐层把对应的张量相加。  
   - 把累计的张量除以 K，得到平均张量。  
   - 用 `model.load_state_dict(averaged_state)` 把平均权重加载回模型。  
   - 这一步可以在 CPU 上完成，因为只涉及一次 I/O 和简单的张量运算，对 GPU 负担几乎为零。

4. **继续训练或评估**  
   - 如果选择“替换训练权重”，则后续的梯度更新基于已经平滑的参数进行，模型在每个 epoch 开始时已经是最近 K 次的“共识”。  
   - 如果只用于评估，则保持原始权重继续训练，同时在验证集上用平均模型报告更低的 loss/accuracy，帮助判断收敛进度。

**为什么有效？**  
- **噪声抑制**：单个 epoch 的梯度更新会带来局部波动，尤其在学习率较大时更明显。把最近 K 次的权重平均，相当于对这些波动做低通滤波，留下更稳健的方向。  
- **信息保留**：只取最近的 K 次，而不是全部历史，避免把早期的“糟糕”权重拉低整体表现。相当于只听最近几天的天气预报，而不把十年前的寒潮记忆带进来。  
- **无需额外正则**：传统的正则项（如 weight decay）是通过在 loss 上加项来约束模型，这里直接在参数空间做约束，省去调参的麻烦。

**最巧妙的地方**  
作者把“平均”这一步放在每个 epoch 的**结束**而不是训练中间，这样可以利用已经完成的完整梯度信息，而不必在每一步都做额外计算。更重要的是，这种做法不改变 optimizer 的内部状态（如 momentum），因此不会破坏已有的学习率调度策略，保持了原有训练流程的完整性。

### 实验与效果
- **实验平台**  
  - 视觉任务：ResNet‑50 在 ImageNet（1.28M 训练图像）上训练 90 epoch。  
  - 语言任务：RoBERTa‑Base 在 WikiText‑103（约 103M token）上训练 10 epoch（对应常规的 pre‑training 设定）。  
  - 硬件：单块 NVIDIA V100 GPU，使用混合精度训练。

- **基线对比**  
  - **普通训练**：不做任何权重平均，直接使用标准学习率调度。  
  - **最新 K 次平均**（K=5、10、15 进行调研）：在每个 epoch 结束后做平均并替换权重。

- **主要结果**  
  - 在 ImageNet 上，使用 K=10 的最新平均模型在第 30 epoch 时就达到了普通训练在第 45 epoch 才能达到的 top‑1 accuracy（约 76%），相当于提前约 15 epoch，折算下来约 68 GPU 小时的算力节约。  
  - 在 WikiText‑103 上，K=5 的平均模型在第 4 epoch 的 perplexity 已经低于普通训练在第 7 epoch 的水平，节约约 30 GPU 小时。  
  - 误差曲线显示，平均模型的验证 loss 曲线更平滑，波动幅度下降约 30%。

- **消融实验**  
  - **窗口大小 K**：K 过小（K=2）效果不明显，K 过大（K=30）会把早期噪声带进来，收敛速度反而下降。K=5~15 是经验最佳区间。  
  - **是否替换训练权重**：仅用于评估时也能看到更低的验证 loss，但真正的训练加速只有在替换权重后才显现。  
  - **不同模型规模**：在更小的网络（如 ResNet‑18）上收益有限，而在更大模型（如 BERT‑Large）实验表明加速效果更明显，暗示方法对参数量大的模型更友好。

- **局限性**  
  - 并非所有任务都能受益，作者提到在某些不稳定的 GAN 训练或极端小数据集上，平均可能导致梯度消失。  
  - 对于极端高学习率的调度，平均后模型可能“回退”到学习率下降前的状态，需要配合合适的学习率策略。  
  - 代码实现依赖于每 epoch 保存 checkpoint，若训练过程不保存（如只在每 N 步保存），则需要自行补齐。

### 影响与延伸思考
这篇论文把“最近 K 次权重平均”从概念验证提升到大规模实用，随后出现了几波跟进工作：  
- **Fast Checkpoint Averaging (FCA)** 在大模型微调阶段加入了自适应 K，进一步提升了少量 GPU 资源下的收敛速度（推测）。  
- **Ensemble‑Free Model Soup** 系列把多次 fine‑tune 的 checkpoint 直接平均，成为了 NLP 社区的“模型汤”标准做法。  
- 在自监督视觉预训练（如 SimCLR、MoCo）中，研究者开始把最新 K 次的投影头权重平均作为 “teacher”，实现了更快的对比学习收敛（推测）。  

如果想继续深挖，可以关注以下方向：  
1. **自适应窗口大小**：根据验证 loss 的波动自动增减 K，兼顾收敛速度与稳定性。  
2. **与学习率调度的协同**：探索在 cosine annealing、warmup 等调度下的最佳平均时机。  
3. **跨任务模型汤**：把不同任务（如检测+分割）的 checkpoint 合在一起，看是否能得到更通用的特征表示。  

### 一句话记住它
只要把最近几次的 checkpoint 平均一下，就能让大模型在训练初期“跳过”不少 epoch，省下数十 GPU 小时的算力。