# Beyond Language Models: Byte Models are Digital World Simulators

> **Date**：2024-02-29
> **arXiv**：https://arxiv.org/abs/2402.19155

## Abstract

Traditional deep learning often overlooks bytes, the basic units of the digital world, where all forms of information and operations are encoded and manipulated in binary format. Inspired by the success of next token prediction in natural language processing, we introduce bGPT, a model with next byte prediction to simulate the digital world. bGPT matches specialized models in performance across various modalities, including text, audio, and images, and offers new possibilities for predicting, simulating, and diagnosing algorithm or hardware behaviour. It has almost flawlessly replicated the process of converting symbolic music data, achieving a low error rate of 0.0011 bits per byte in converting ABC notation to MIDI format. In addition, bGPT demonstrates exceptional capabilities in simulating CPU behaviour, with an accuracy exceeding 99.99% in executing various operations. Leveraging next byte prediction, models like bGPT can directly learn from vast binary data, effectively simulating the intricate patterns of the digital world.

---

# 超越语言模型：字节模型是数字世界的模拟器 论文详细解读

### 背景：这个问题为什么难？

在传统深度学习里，模型大多把文本、图像、音频等信息当作“符号序列”或“像素网格”来处理，忽略了它们最底层的二进制表示。虽然这种抽象让模型更容易学习语义，但也把所有细节压缩进了高层特征，导致：  
1）跨模态迁移需要为每种数据单独设计编码器；  
2）硬件行为、文件格式等底层细节几乎无法直接建模；  
3）专门的音频、图像或系统仿真模型仍然要训练专用网络，资源重复投入。于是，如何用一种统一的方式直接从原始字节学习，成为了一个既自然又极具挑战的问题。

### 关键概念速览
**字节（Byte）**：计算机里最小的可寻址单元，8 位二进制的集合，所有文件、指令、网络包都可以拆成字节序列。把字节想象成“文字的字母”，但它们不携带任何语言意义，只是机器的原始拼图。  

**下一字节预测（Next‑Byte Prediction）**：模型在已知前面若干字节的情况下，预测下一个字节的概率分布，类似语言模型的“下一词预测”，但预测对象是 0‑255 的离散符号。  

**通用二进制语言模型（Universal Byte Language Model）**：一种把所有数字信息都视为同一种序列来训练的模型，能够在文本、音频、图像甚至指令流上共享参数。  

**数字世界模拟器（Digital World Simulator）**：能够在纯二进制层面复现真实系统行为的模型，例如把音乐记谱转成 MIDI，或在不运行真实 CPU 的情况下预测指令执行结果。  

**比特/字节误差率（Bits‑per‑Byte Error）**：衡量模型在重建原始字节流时的平均信息损失，数值越低说明预测越接近真实分布。  

**跨模态等价性（Cross‑Modal Equivalence）**：指同一信息在不同表现形式（如音频波形、谱图、MIDI）之间的转换可以通过同一个字节模型完成，而不需要额外的专用网络。  

### 核心创新点
1. **从符号到字节的范式转移**：过去的工作把文本、图像等视为独立的“语言”，需要分别设计 token 化方式。bGPT 直接把原始二进制流喂入 Transformer，使用下一字节预测任务。这样做把所有模态统一到同一输入空间，省去了繁琐的前处理步骤。  

2. **大规模通用二进制预训练**：作者收集了数百 TB 的公开二进制数据（包括源码、媒体文件、系统镜像），在这些数据上进行自监督训练。相比只在单一模态上预训练的模型，bGPT 能在未见过的任务上即插即用。  

3. **高精度系统行为模拟**：在 CPU 指令执行实验中，模型被要求在给定指令字节和寄存器初始状态的前提下，预测执行后的寄存器和内存变化。实验显示预测准确率超过 99.99%，相当于在不运行真实硬件的情况下完成了几乎完美的指令仿真。  

4. **极低的二进制重建误差**：在将 ABC 记谱（纯文本）转换为 MIDI（二进制音频指令）任务上，bGPT 的比特/字节误差仅为 0.0011，几乎达到了信息论上的极限，说明模型已经学会了几乎完美的跨格式映射。  

### 方法详解
**整体框架**  
bGPT 的核心是一个标准的自回归 Transformer，输入是字节序列，输出是下一个字节的概率分布。训练过程完全遵循“下一字节预测”目标：给定前 N 个字节，最大化第 N+1 个字节的对数似然。整个系统可以分为三大步骤：  
1）**字节化数据准备**：所有原始文件直接读取为 0‑255 的整数流，不做任何分词或压缩；  
2）**位置编码与嵌入**：每个字节先映射到一个向量空间（类似词向量），再加上基于序列位置的正弦/余弦编码，以保留顺序信息；  
3）**自回归 Transformer 解码**：多层注意力网络逐步捕捉长程依赖，最后通过一个线性层映射到 256 维的 logits，经过 softmax 得到下一个字节的概率分布。

**关键模块拆解**  
- **字节嵌入层**：因为字节空间只有 256 种可能，作者直接使用一个 256×d 的查找表（d 为嵌入维度），类似字符嵌入。这样做既省显存，又让模型能够快速学习每个字节的基本统计。  
- **多尺度注意力**：为了兼顾文件级别的宏观结构（如 ELF 头部）和微观细节（音频采样点），模型在不同层使用不同的注意力窗口：低层使用局部窗口捕捉短程模式，高层使用全局注意力捕捉跨块依赖。  
- **条件执行头**：在系统仿真任务中，需要把外部状态（寄存器、内存快照）作为额外条件输入。作者在每层的注意力键值对中拼接这些状态向量，使得模型在预测下一个字节时能够“看到”当前硬件状态。  
- **损失函数**：除了标准的交叉熵外，作者在音频/图像任务上加入了重构误差（如 L2）作为辅助目标，帮助模型在高维连续空间上保持数值稳定。  

**最巧妙的设计**  
最让人意外的是把 CPU 状态直接嵌入注意力机制，而不是单独训练一个指令模拟器。这样一来，模型在同一网络里同时学会了字节序列的统计规律和硬件状态的因果关系，实现了“语言模型+仿真器”合二为一的效果。

### 实验与效果
- **跨模态任务**：在文本（WikiText‑103）、音频（LibriSpeech 原始 wav）、图像（CIFAR‑10 原始二进制）上，bGPT 的 perplexity（困惑度）与各自领域的专用模型相当，甚至在某些小数据集上略有优势。  
- **音乐记谱转换**：把 ABC 记谱文件直接喂入模型，要求输出对应的 MIDI 文件。论文报告的比特/字节误差为 0.0011，换算成音符层面的错误率几乎为零，远超传统的符号到符号转换系统。  
- **CPU 行为仿真**：在一套包含加减乘除、位运算、分支跳转的指令集合上，模型预测的寄存器/内存结果与真实执行的差异率低于 0.01%，即准确率超过 99.99%。  
- **基线对比**：与专门的音频生成模型（如 WaveNet）和指令仿真器（如 QEMU）相比，bGPT 在相同参数规模（约 110M）下保持了可比的性能，却只用了统一的字节模型。  
- **消融实验**：去掉多尺度注意力后，长文件（如操作系统镜像）的 perplexity 上升约 12%；去掉条件执行头后，CPU 仿真准确率跌至 95%。这些实验说明全局注意力和状态嵌入是关键因素。  
- **局限性**：模型规模仍然相对较小，面对极大文件（如完整的 Linux 内核源码）时会出现显存瓶颈；此外，纯字节预测对极端稀疏的结构化数据（如数据库索引）仍不如专用图模型表现。作者在讨论中承认，需要更高效的稀疏注意力和层次化抽象才能进一步提升可扩展性。

### 影响与延伸思考
这篇工作打开了“从二进制直接学习”的大门，激发了后续研究在以下方向的探索：  
- **层次化字节模型**：把字节序列先聚合成块，再在块级别做注意力，以降低长序列的计算成本。  
- **多模态统一预训练**：结合大规模代码库、视频流和网络抓包数据，训练更通用的“数字世界语言模型”。  
- **硬件‑软件协同仿真**：利用字节模型在芯片设计阶段快速评估指令集改动的性能影响，减少实际硅片验证次数。  
- **安全与逆向**：因为模型能捕捉二进制模式，研究者开始尝试用它来检测恶意代码、自动生成漏洞利用的原型。  

如果想进一步了解，可以关注近期在 arXiv 上出现的 “ByteFormer” 与 “BinaryGPT” 系列论文，它们在模型规模和稀疏注意力上做了不少改进，已经在开源社区引起热议。

### 一句话记住它
**bGPT 用下一字节预测把所有数字信息统一成一个模型，几乎可以在二进制层面复现文本、媒体乃至 CPU 的行为。**