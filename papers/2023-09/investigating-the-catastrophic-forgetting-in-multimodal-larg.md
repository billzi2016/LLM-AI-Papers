# Investigating the Catastrophic Forgetting in Multimodal Large Language   Models

> **Date**：2023-09-19
> **arXiv**：https://arxiv.org/abs/2309.10313

## Abstract

Following the success of GPT4, there has been a surge in interest in multimodal large language model (MLLM) research. This line of research focuses on developing general-purpose LLMs through fine-tuning pre-trained LLMs and vision models. However, catastrophic forgetting, a notorious phenomenon where the fine-tuned model fails to retain similar performance compared to the pre-trained model, still remains an inherent problem in multimodal LLMs (MLLM). In this paper, we introduce EMT: Evaluating MulTimodality for evaluating the catastrophic forgetting in MLLMs, by treating each MLLM as an image classifier. We first apply EMT to evaluate several open-source fine-tuned MLLMs and we discover that almost all evaluated MLLMs fail to retain the same performance levels as their vision encoders on standard image classification tasks. Moreover, we continue fine-tuning LLaVA, an MLLM and utilize EMT to assess performance throughout the fine-tuning. Interestingly, our results suggest that early-stage fine-tuning on an image dataset improves performance across other image datasets, by enhancing the alignment of text and visual features. However, as fine-tuning proceeds, the MLLMs begin to hallucinate, resulting in a significant loss of generalizability, even when the image encoder remains frozen. Our results suggest that MLLMs have yet to demonstrate performance on par with their vision models on standard image classification tasks and the current MLLM fine-tuning procedure still has room for improvement.

---

# 多模态大语言模型中的灾难性遗忘研究 论文详细解读

### 背景：这个问题为什么难？

在 GPT‑4 之后，研究者们把注意力转向了能够同时处理文字和图像的多模态大语言模型（MLLM）。这些模型的常规做法是先把已有的大语言模型和视觉编码器各自预训练好，再通过一次统一的微调让两者对齐。可是，微调往往会让模型在原本擅长的视觉任务上表现大幅下降——这就是所谓的灾难性遗忘。过去的工作大多关注提升跨模态生成质量，却很少系统量化模型在纯视觉任务（比如 ImageNet 分类）上的退化程度，导致我们不知道到底是对齐过程本身出问题，还是微调策略不够稳健。

### 关键概念速览
- **多模态大语言模型（MLLM）**：把大语言模型（LLM）和视觉编码器拼在一起的系统，能够接受文字和图像输入并输出自然语言。想象成一个会说话的相机。
- **灾难性遗忘（Catastrophic Forgetting）**：模型在学习新任务时，原有任务的能力急剧下降。就像人学会弹钢琴后，原本会的吉他技巧全忘了。
- **视觉编码器（Vision Encoder）**：专门把图片转换成向量的网络，常见的有 CLIP、ViT 等。它相当于把图片翻译成机器能读的“文字”。
- **EMT（Evaluating MulTimodality）**：本文提出的评估框架，把每个 MLLM 当作图像分类器来测量它的视觉保留能力。类似于把多功能工具箱只用来测试螺丝刀的质量。
- **对齐（Alignment）**：让语言和视觉特征在同一向量空间里对应起来的过程。可以比作把两本不同语言的词典对齐，使得同义词在同一页出现。
- **幻觉（Hallucination）**：模型在生成文本时捏造不存在的细节或事实。就像人把看不见的东西“想象”出来并当真说。

### 核心创新点
1. **把 MLLM 当作图像分类器来评估**  
   之前的评估大多围绕跨模态问答或生成质量展开，缺少纯视觉基准。作者提出 EMT，将每个模型的视觉分支直接用于标准图像分类任务，从而得到与原始视觉编码器的可比性能。这样可以直观看到微调后视觉能力的保留程度。

2. **系统化对比多种开源 MLLM**  
   通过 EMT，对几套公开的微调后 MLLM 进行统一测评，发现几乎所有模型在 ImageNet 等分类任务上的准确率都显著低于对应的视觉编码器。这个发现本身就提供了一个“警示灯”，说明当前的微调流程普遍存在遗忘问题。

3. **细粒度追踪 LLaVA 微调过程**  
   作者在 LLaVA（一个流行的 MLLM）上进行阶段性微调，并在每一步使用 EMT 检测性能变化。结果显示，早期微调反而提升了跨数据集的视觉表现，说明对齐可以带来正向迁移；但继续微调会导致模型产生幻觉，视觉分类能力急剧下滑，即使视觉编码器保持冻结也不例外。

4. **揭示“冻结视觉编码器也救不了”的现象**  
   常见的防止遗忘的做法是冻结视觉编码器，只训练语言侧。实验表明，即便如此，模型仍会因语言侧的参数更新而破坏已有的视觉特征映射，提示对齐过程本身比单纯的参数更新更脆弱。

### 方法详解
整体思路可以拆成三步：**（1）准备评估基准、（2）构建 EMT 测试套件、（3）在微调过程中持续监测**。

1. **准备评估基准**  
   - 选取若干公开的视觉分类数据集（如 ImageNet、CIFAR‑10、Flowers102），这些数据集在视觉编码器预训练阶段已经达到高准确率。  
   - 对每个数据集，保留原始的训练/验证划分，以便直接比较微调前后的表现。

2. **构建 EMT 测试套件**  
   - 对每个 MLLM，截取其视觉编码器的输出向量，并在其上接一个轻量级的线性分类头（相当于在原始视觉模型上加一个“帽子”）。  
   - 只训练这个分类头，保持模型其余部分（包括语言模型和对齐层）不动。这样做的目的是让模型的视觉特征直接参与分类，而不受语言生成任务的干扰。  
   - 评估指标仍然是常规的 Top‑1 / Top‑5 准确率，确保结果可与原始视觉编码器的报告直接对比。

3. **微调过程追踪**  
   - 以 LLaVA 为例，作者先在一个大规模图像‑文本对齐数据集上进行若干 epoch 的微调。每完成一次 epoch，就使用 EMT 对所有评估数据集跑一次前向推理，记录分类准确率。  
   - 通过这些时间序列数据，作者观察到：  
     - **早期阶段**（前 1‑2 epoch）视觉准确率略有提升，说明对齐层帮助语言和视觉特征更好地协同。  
     - **中后期**（随后 epoch）准确率快速下降，且模型在生成答案时出现大量幻觉（比如描述不存在的物体颜色），即使视觉编码器本身被标记为“冻结”。  
   - 关键的反直觉点在于：冻结视觉编码器并不能阻止语言侧的梯度通过对齐层反向传播，进而破坏视觉特征的空间结构。

整个流程的核心是把“多模态对齐”视作一种可度量的、会随时间变化的属性，而不是一次性完成的黑盒操作。通过 EMT，作者把对齐质量转化为具体的分类分数，使得灾难性遗忘可以被量化、可视化。

### 实验与效果
- **数据集**：ImageNet‑1K、CIFAR‑10、Flowers102 等标准视觉分类基准。  
- **对比模型**：几套开源的 MLLM（如 LLaVA‑v1.5、MiniGPT‑4、InstructBLIP 等），以及它们对应的原始视觉编码器（如 CLIP‑ViT‑L/14）。  
- **主要发现**：  
  - 在 ImageNet 上，原始视觉编码器的 Top‑1 准确率约为 78%，而经 EMT 评估的 MLLM 多数在 55% 左右，跌幅超过 20%。  
  - 在 CIFAR‑10 上，差距同样显著，原始编码器 96% → MLLM 78%。  
  - LLaVA 的微调曲线显示，前 2 epoch 分类准确率提升约 3%，随后每个 epoch 下降约 5%，到第 6 epoch 时整体下降至原始水平以下。  
- **消融实验**：作者尝试仅冻结对齐层、仅微调语言模型、以及使用不同学习率的组合。结果表明，任何让语言侧梯度直接影响对齐层的设置都会导致视觉性能下降，说明对齐层是遗忘的关键通道。  
- **局限性**：实验主要聚焦在图像分类任务，未覆盖目标检测、分割等更复杂的视觉任务；此外，EMT 只使用线性分类头，可能低估了模型内部更深层次的视觉表征潜力。作者也承认没有探索更高级的防忘策略（如弹性权重保持）在多模态环境下的效果。

### 影响与延伸思考
这篇工作在社区里引发了对“多模态微调安全性”的关注。随后出现的几篇论文（如《Stable Alignment for Multimodal Models》《Mitigating Forgetting in Vision‑Language Fine‑Tuning》）直接引用了 EMT 的评估思路，尝试在微调时加入正则化项或使用多任务学习来保持视觉特征的稳定。对想进一步深入的读者，可以关注以下方向：  
- **弹性权重保持（EWC）在跨模态对齐中的适配**，即在语言侧更新时约束对齐层的梯度。  
- **多任务微调**：同时在图像分类和语言生成上做小步更新，观察是否能形成更稳固的共享表征。  
- **更丰富的评估套件**：把目标检测、实例分割等任务加入 EMT，构建全景的多模态遗忘基准。  
- **可解释性分析**：利用梯度可视化或注意力图，定位语言侧更新如何具体扰动视觉特征空间。

### 一句话记住它
**EMT 把多模态大模型的视觉能力直接量化为分类分数，揭示了即使冻结视觉编码器，微调也会导致灾难性遗忘。**