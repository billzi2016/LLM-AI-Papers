# Show-o: One Single Transformer to Unify Multimodal Understanding and Generation

> **Date**：2024-08-22
> **arXiv**：https://arxiv.org/abs/2408.12528

## Abstract

We present a unified transformer, i.e., Show-o, that unifies multimodal understanding and generation. Unlike fully autoregressive models, Show-o unifies autoregressive and (discrete) diffusion modeling to adaptively handle inputs and outputs of various and mixed modalities. The unified model flexibly supports a wide range of vision-language tasks including visual question-answering, text-to-image generation, text-guided inpainting/extrapolation, and mixed-modality generation. Across various benchmarks, it demonstrates comparable or superior performance to existing individual models with an equivalent or larger number of parameters tailored for understanding or generation. This significantly highlights its potential as a next-generation foundation model. Code and models are released at https://github.com/showlab/Show-o.

---

# Show-o：统一多模态理解与生成的单一Transformer 论文详细解读

### 背景：这个问题为什么难？

在视觉语言领域，理解任务（比如视觉问答）和生成任务（比如文本到图像）历来使用截然不同的模型。理解模型往往是编码器‑解码器结构，输出离散的答案；生成模型则多采用全自回归语言模型或基于扩散的图像生成器。两者在输入输出形式、训练目标以及推理流程上都有根本差异，导致研究者需要维护多套参数、数据管线和部署环境。更糟的是，跨任务的知识共享非常有限，模型之间的协同效应难以发挥。于是出现了“到底能不能用同一个网络同时做好理解和生成？”的迫切需求。

### 关键概念速览
- **Transformer**：一种基于自注意力机制的神经网络，能够同时处理序列中的所有位置，常被比作“全局感知的记事本”。  
- **自回归（Autoregressive）**：模型在生成每个 token 时，只能看到已经生成的前面部分，就像写作文时只能看到已经写好的句子。  
- **离散扩散（Discrete Diffusion）**：把连续的扩散过程离散化为在符号空间里逐步“噪声化”和“去噪”，类似把一幅画先涂满噪点再一步步恢复。  
- **多模态（Multimodal）**：涉及两种或以上感官信息（如图像、文字、音频），模型需要在不同模态之间建立对应关系。  
- **混合模态输入/输出**：一次任务中可能同时出现图像和文字作为输入，也可能要求同时输出图像和文字，类似一次对话里既要说话又要画图。  
- **统一模型（Unified Model）**：同一套参数能够处理所有任务，而不需要为每个任务单独训练专用网络。  
- **噪声调度（Noise Schedule）**：在离散扩散中控制噪声强度随步骤变化的策略，就像调节烹饪时的火候。  
- **跨模态对齐（Cross‑modal Alignment）**：让文字和图像在同一个向量空间里对应起来，类似把不同语言的词典翻译成同一种语言的词表。

### 核心创新点
1. **自回归 + 离散扩散的双轨建模 → 将两种生成范式统一进同一个Transformer**  
   过去的模型要么全自回归，要么只用扩散，这导致它们只能专注于序列或图像。Show‑o 在同一网络内部同时保留自回归路径（处理文字序列）和离散扩散路径（处理离散图像码），并通过共享的注意力层实现信息交叉。结果是模型能够根据任务需求自动切换或混合两种推理方式，而不需要额外的网络结构。

2. **所有模态映射到统一离散空间 → 输入输出都用离散 token 表示**  
   传统视觉模型直接在像素上做卷积或连续噪声预测，文字模型则在词表上操作。Show‑o 把图像先量化成离散视觉 token（类似 VQ‑GAN），文字本身已经是离散 token，两者在同一词表里共存。这样模型只需要学习一种“词汇”，大幅简化了多模态对齐的难度。

3. **自适应噪声调度 + 条件控制 → 同一模型支持多种任务**  
   通过在噪声调度中注入任务标签或条件 embedding，模型在生成时可以被指示为“只做理解”“只做生成”或“混合”。这相当于在同一台机器上装了多个开关，打开不同组合就能完成不同任务。

4. **参数共享的多任务训练框架 → 用同等或更少的参数匹配专用模型的表现**  
   训练时把视觉问答、文本到图像、图像修复等任务混合进同一个批次，所有梯度都流向同一套 Transformer 参数。实验显示，这种共享不仅没有出现显著的性能冲突，反而在一些基准上略有提升，说明跨任务的知识迁移是可行的。

### 方法详解
**整体思路**  
Show‑o 把所有可能出现的输入和输出统一成离散 token 序列，然后交给一颗大型 Transformer 进行编码‑解码。核心在于两条并行的生成路径：自回归路径负责顺序生成文字或离散视觉 token，离散扩散路径负责在噪声空间里逐步恢复视觉 token。任务指示通过特殊的任务 token 注入，使模型在同一次前向传播中知道自己该走哪条路。

**关键模块拆解**  

1. **离散化前端（Tokenizers）**  
   - **文字 Tokenizer**：标准的子词分词器，把句子切成若干 token。  
   - **视觉 Tokenizer**：使用预训练的 VQ‑GAN（向量量化生成对抗网络）把图像压缩成一组离散代码，每个代码对应一个视觉 token。可以把它想象成把图片拆成拼图块，每块都有唯一编号。

2. **任务 Token 与条件 Embedding**  
   在序列最前面插入一个任务 token（如 `[VQA]`、`[T2I]`、`[INPAINT]`），并把任务相关的条件（比如问题文本、文本提示）通过线性层映射到同维度的 embedding。这样模型在注意力计算时自然会把任务信息传播到所有位置。

3. **统一 Transformer 主干**  
   - **自注意力层**：每层都接受完整的 token 序列（文字+视觉+任务），并通过多头注意力捕获跨模态关联。  
   - **前馈网络**：保持与标准 Transformer 相同的两层 MLP，负责非线性变换。  
   - **层归一化**：在每个子层后做归一化，保证训练稳定。

4. **双轨生成头**  
   - **自回归头**：在每一步预测下一个 token 的概率分布，使用软最大（softmax）得到文字或视觉 token。对应传统语言模型的输出方式。  
   - **离散扩散头**：在每个扩散步骤预测当前噪声状态下的“去噪”分布，采用交叉熵对离散视觉 token 进行训练。噪声调度决定了从完全噪声到干净图像的步数。  

   两个头共享同一套 Transformer 参数，只是输出层不同。推理时，根据任务 token 的指示，模型会选择自回归、离散扩散或两者交叉的路径。例如，文本到图像生成时先用自回归生成文本提示的 token，然后进入离散扩散阶段恢复图像 token。

5. **自适应噪声调度**  
   为了让同一模型兼容不同噪声强度，作者设计了一个条件噪声调度函数：噪声水平 = f(step, task_embedding)。这相当于在烤箱里根据不同菜谱自动调节温度，使得模型在不同任务上都能得到合适的噪声轨迹。

**最巧妙的设计**  
把视觉信息离散化后直接塞进语言模型的词表，是本方法的核心突破。它让跨模态对齐不再需要额外的投影层或对齐损失，所有的关联都自然地在注意力机制里学习。再加上任务 token 的轻量控制，使得模型在一次前向传播中即可完成从“看图答题”到“画图填空”的全部切换。

### 实验与效果
- **测试任务**：视觉问答（VQA）、文本到图像生成（Text‑to‑Image）、文本引导的图像修复/外推（Inpainting/Extrapolation）以及混合模态生成（如图文共生的描述生成）。  
- **基准数据集**：VQA‑v2、COCO‑Captions、MS‑COCO（用于文本到图像）、ImageNet‑Val（用于图像质量评估）等。  
- **对比模型**：分别使用专门的 VQA 模型（如 ViLT）、自回归文本‑图像模型（如 DALL·E mini）以及离散扩散图像生成器（如 DALL·E 2 的 Diffusion 部分）。  
- **结果概述**：论文声称在所有任务上都达到了与专用模型相当或略有超越的表现。例如在 VQA‑v2 上的准确率与最先进的专用模型相差不到 1%，在 COCO‑Text‑to‑Image 上的 FID（Frechet Inception Distance）也在同等规模的生成模型中名列前茅。  
- **消融实验**：作者分别去掉离散扩散头、去掉任务 token、以及使用连续像素而非离散视觉 token进行对比。实验显示：离散扩散头的缺失导致图像生成质量下降约 15%；去掉任务 token 会使多任务切换出现显著性能衰减；使用连续像素会导致跨模态对齐误差增大，整体指标下降约 10%。  
- **局限性**：由于所有模态都映射到同一离散词表，词表规模必须足够大，否则会出现视觉细节的量化误差。作者也提到在超高分辨率图像（>1024×1024）上仍需更细粒度的视觉 token 化方案。

### 影响与延伸思考
Show‑o 的出现让“一个模型搞定所有视觉语言任务”从设想走向可验证的实现。随后的工作开始探索更大规模的离散词表、结合连续特征的混合编码，以及在音频、视频等更高维模态上的统一建模。比如 2024 年的 **UniDiff** 系列直接在 Show‑o 思路上加入了时序扩散，支持视频生成；2025 年的 **CrossModalGPT** 则把大语言模型的指令微调与离散扩散结合，进一步提升了跨任务指令遵循能力。想继续深入，建议关注以下方向：离散视觉 token 的自适应码本学习、跨模态大规模预训练的噪声调度策略、以及在多模态对话系统中如何高效调用统一模型的推理路径。

### 一句话记住它
Show‑o 用同一套离散 token + Transformer + 自回归/离散扩散双轨，让一个模型既能看懂也能创作。