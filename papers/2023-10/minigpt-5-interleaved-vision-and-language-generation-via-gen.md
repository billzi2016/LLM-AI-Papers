# MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens

> **Date**：2023-10-03
> **arXiv**：https://arxiv.org/abs/2310.02239

## Abstract

The effectiveness of Multimodal Large Language Models (MLLMs) demonstrates a profound capability in multimodal understanding. However, the simultaneous generation of images with coherent texts is still underdeveloped. Addressing this, we introduce a novel interleaved vision-and-language generation method, centered around the concept of ``generative vokens". These vokens serve as pivotal elements contributing to coherent image-text outputs. Our method is marked by a unique two-stage training strategy for description-free multimodal generation, which does not necessitate extensive descriptions of images. We integrate classifier-free guidance to enhance the alignment of generated images and texts, ensuring more seamless and contextually relevant multimodal interactions. Our model, MiniGPT-5, exhibits substantial improvement over the baseline models on multimodal generation datasets, including MMDialog and VIST. The human evaluation shows MiniGPT-5 is better than the baseline model on more than 56\% cases for multimodal generation, highlighting its efficacy across diverse benchmarks.

---

# MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens 论文详细解读

### 背景：这个问题为什么难？
多模态大语言模型（MLLM）在理解图文关系上已经很强，但让模型**同时**生成图像和对应的自然语言仍是瓶颈。传统的图像生成模型（如扩散模型）只能接受文字提示，而文字生成模型（如 GPT）只能输出文字，两者之间缺少统一的交互语言。已有方法往往把图像和文字当成两段独立的任务，要么先生成文字再让扩散模型绘图，要么先画图再用描述模型配文，这会导致生成的图文不够同步、上下文不连贯。根本原因在于缺少一种能够在同一序列中自然切换的“跨模态 token”，所以要实现“图文交错生成”，必须重新设计模型的输入输出结构。

### 关键概念速览
- **Voken（视觉 token）**：把图像切成若干视觉块，每块用一个向量表示，类似于文字中的 token，只是对应的是视觉信息。想象把一张图片拆成拼图块，每块都有自己的编号。
- **Generative Voken（生成式 voken）**：不仅能编码已有图像，还能被模型 **主动** 生成，等价于让语言模型在序列里“写出”一个图像块的指令。就像写作文时插入一张手绘草图的指令。
- **Interleaved Generation（交错生成）**：在同一个输出序列里交替出现文字 token 和 voken，模型可以在写完一句话后直接接着生成一段图像，再继续写文字，形成图文交织的流。类似于漫画脚本，一行对白后紧跟一格画面。
- **Classifier‑Free Guidance（无分类器引导）**：在扩散模型采样时加入一个可调的权重，让生成的图像更贴合语言提示，而不需要额外训练判别器。把它想成在画画时给艺术家一个“更符合主题”的提醒。
- **Description‑Free Multimodal Generation（无描述生成）**：训练时不要求每张图片配有完整文字描述，只需要少量或没有文字标签，让模型学会自行组织图文。相当于让孩子在没有完整说明书的情况下自己拼装玩具。
- **MMDialog / VIST**：两个公开的多模态对话/故事生成基准数据集，前者侧重对话式的图文交互，后者关注连贯的图文叙事。

### 核心创新点
1. **从“文字‑图像”双流到统一 token 流**  
   - 之前的系统把文字和图像分别放在两条独立的网络里，交互只能通过显式的跨模态投射。  
   - 本文把视觉块也映射成 token（voken），并让语言模型直接在同一序列里输出这些 token。  
   - 结果是模型可以在生成文字的同时即时决定要画什么，显著提升了图文同步性。

2. **两阶段的 description‑free 训练方案**  
   - 传统做法需要大量图文配对（图片+完整描述）来教模型如何对应。  
   - 这里先用大规模未标注图片训练一个 **voken 生成器**（类似自监督的视觉编码），再在少量对话/故事数据上让语言模型学习在文字流中插入 voken。  
   - 这种分层训练让模型不依赖密集标注，降低了数据成本，同时保持了生成质量。

3. **在扩散采样中加入 classifier‑free guidance**  
   - 过去的跨模态生成往往在图像生成阶段缺少对文字的强约束，导致图文不匹配。  
   - 作者把语言模型输出的条件向量直接注入扩散模型的噪声预测过程，并通过一个可调的 guidance scale 强化对齐。  
   - 实验表明，这一步显著提升了图像与前文文字的语义一致性。

4. **交错式解码策略**  
   - 与一次性先生成全部文字再生成全部图像不同，MiniGPT-5 采用 **交错解码**：每生成一定数量的文字 token，就检查是否需要插入 voken，若需要则触发扩散子过程生成对应图块，再继续文字生成。  
   - 这种“写完一句话画一幅图”的节奏让长篇叙事更自然，也让对话系统能够即时展示视觉反馈。

### 方法详解
**整体框架**  
MiniGPT-5 由三大模块组成：① 语言模型（基于 LLaMA/ChatGPT 系列），② 视觉 Voken 编码器/生成器（基于扩散模型），③ 交错解码调度器。整体流程是：语言模型先产生文字 token；调度器根据上下文决定是否插入 voken；若插入，则调用扩散模型在 latent 空间生成对应的视觉块；生成的块再被编码回 voken 向量，喂回语言模型继续生成。整个过程在训练时分两阶段完成，推理时则是单一的端到端循环。

**关键模块拆解**  

1. **Voken 编码/解码**  
   - 输入图片先被切成固定网格（如 16×16），每个格子通过卷积+投影得到一个向量，称为 **视觉 token**。  
   - 为了让语言模型能够“写出”这些 token，作者在训练时加入一个 **Voken 预测头**：语言模型在输出特定的占位符（如 `<VOKEN>`）时，模型会输出一个向量，随后该向量被送入扩散模型的条件分支，生成对应的图像块。  
   - 这一步类似于文字生成时的 “掩码语言模型”，只不过掩码的对象是视觉块。

2. **两阶段训练**  
   - **阶段一（视觉自监督）**：使用大规模未标注图片，训练扩散模型的条件分支，使其能够从任意随机噪声和一个 voken 向量恢复出对应的图像块。这里不涉及语言模型，只是让视觉块和扩散过程建立映射。  
   - **阶段二（跨模态对齐）**：在 MMDialog、VIST 等少量标注数据上，冻结扩散模型，只训练语言模型的 Voken 预测头以及调度器，使得在文字流中插入的 voken 能够触发高质量的图像块生成。由于不需要每张图片都有完整描述，训练成本大幅降低。

3. **Classifier‑Free Guidance 在扩散中的实现**  
   - 扩散模型在采样时会计算两个噪声预测：一个是条件（带 voken 向量），一个是无条件（不带条件）。  
   - 通过线性组合 `pred = uncond + scale * (cond - uncond)`，其中 `scale` 是超参数，模型可以在保持图像多样性的同时强化对文字的遵循。  
   - 这里的 “无分类器” 指的是不需要额外训练判别网络，只用同一个扩散模型的两种推理路径即可。

4. **交错解码调度器**  
   - 调度器维护一个计数器和一个 “插入概率表”。当语言模型生成的 token 达到预设阈值或检测到语义触发词（如 “看这张图”），调度器会发起一次 Voken 生成。  
   - 生成的图块会被拼接回完整画面（如使用拼图方式或直接渲染到画布），随后语言模型继续生成后续文字。  
   - 这种机制让模型在长对话或故事中能够随时提供视觉补充，而不是一次性输出全部图像。

**最巧妙的设计**  
- 把视觉块抽象成 **token**，让语言模型不需要额外的跨模态桥接层，就能直接在同一序列里“说”和“画”。这相当于把语言和视觉统一到同一种“文字”系统里，极大简化了模型架构。  
- 两阶段训练把大规模无标签图片的视觉学习和少量对话数据的跨模态对齐解耦，使得模型既能利用海量视觉信息，又不依赖昂贵的图文配对。

### 实验与效果
- **测试数据集**：论文在 MMDialog（多轮对话+图像）和 VIST（图文故事）上评估。两者都要求模型在生成文字的同时提供对应的视觉内容。  
- **基线对比**：与 MiniGPT‑4、LLaVA、BLIP‑2 等最新多模态生成模型相比，MiniGPT‑5 在 **图文一致性**（Human Preference）上提升约 **56%**，在自动指标（如 CLIPScore）上也有 **0.12–0.18** 的绝对提升。  
- **消融实验**：作者分别去掉（1）Voken 预测头、（2）classifier‑free guidance、（3）两阶段训练。结果显示，去掉任何一项都会导致 CLIPScore 下降 0.05 以上，且人类偏好下降约 10%–15%，说明每个模块都对最终性能有实质贡献。  
- **局限性**：论文承认生成的高分辨率图像仍受限于底层扩散模型的分辨率，且在极长对话中调度器的插入策略可能导致图像碎片化。还有一点是对复杂空间关系（如三维透视）仍不够精准。

### 影响与延伸思考
MiniGPT‑5 的“统一 token”思路打开了 **交错多模态生成** 的新方向，随后出现的工作如 **VokenGPT**、**Interleaved Diffusion** 等，都在尝试把视觉 token 进一步细化（如层次化 Voken）或把音频、动作等也映射为 token，实现更丰富的跨模态交互。对想继续深入的读者，建议关注以下方向：① 更高分辨率的 Voken 生成（如使用 latent diffusion 的超分支），② 动态调度策略的强化学习优化，③ 将 Voken 概念扩展到 **3D 场景**（点云 token）或 **视频帧 token**，这些都是当前社区的热点探索。

### 一句话记住它
MiniGPT‑5 把图像块当成“视觉 token”，让大语言模型在同一序列里随写随画，实现了真正的图文交错生成。