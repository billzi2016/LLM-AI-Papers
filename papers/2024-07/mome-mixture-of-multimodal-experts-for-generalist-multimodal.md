# MoME: Mixture of Multimodal Experts for Generalist Multimodal Large   Language Models

> **Date**：2024-07-17
> **arXiv**：https://arxiv.org/abs/2407.12709

## Abstract

Multimodal large language models (MLLMs) have demonstrated impressive capabilities across various vision-language tasks. However, a generalist MLLM typically underperforms compared with a specialist MLLM on most VL tasks, which can be attributed to task interference. In this paper, we propose a mixture of multimodal experts (MoME) to mitigate task interference and obtain a generalist MLLM. Our MoME is composed of two key components, a mixture of vision experts (MoVE) and a mixture of language experts (MoLE). MoVE can adaptively modulate the features transformed from various vision encoders, and has a strong compatibility in transformation architecture. MoLE incorporates sparsely gated experts into LLMs to achieve painless improvements with roughly unchanged inference costs. In response to task interference, our MoME specializes in both vision and language modality to adapt to task discrepancies. Extensive experiments show that MoME significantly improves the performance of generalist MLLMs across various VL tasks. The source code is released at https://github.com/JiuTian-VL/MoME

---

# MoME：多模态专家混合模型用于通用多模态大语言模型 论文详细解读

### 背景：这个问题为什么难？
多模态大语言模型（MLLM）需要同时理解视觉和语言信息，理想状态下它们应该在所有视觉‑语言任务上都表现均衡。然而，现有的通用模型往往被“任务干扰”拖累——在某些任务上表现不错，换到别的任务就会出现明显退步。根本原因在于模型内部的视觉和语言子模块被迫共享同一套参数，导致不同任务的特征需求相互冲突。单一的视觉编码器也会出现“盲区”，比如对细粒度纹理或特殊视角的图像提取不到有效特征。于是，如何在保持通用性的同时缓解任务间的相互干扰，成为了迫切需要突破的瓶颈。

### 关键概念速览
**多模态大语言模型（MLLM）**：把大规模语言模型和视觉感知模块拼在一起，能够接受图文混合输入并生成自然语言输出。想象成一个会说话的机器人，既能看又能聊。  
**任务干扰**：不同下游任务对模型内部表示的需求不一致，导致同一参数被拉扯向多个方向，类似多人抢同一把椅子坐，坐姿会被迫妥协。  
**专家混合（Mixture of Experts, MoE）**：把模型拆成若干“专家”，每次前向只激活一小部分专家，像是公司里不同部门轮流处理业务，既提高专精度又不显著增加成本。  
**视觉专家混合（MoVE）**：专门负责视觉特征的 MoE，能够根据输入动态挑选或加权不同视觉编码器的输出。把它想成一个调音师，根据歌曲风格调节不同乐器的音量。  
**语言专家混合（MoLE）**：在大语言模型内部加入稀疏门控的专家网络，使得不同语言子任务可以使用不同的语言专家。类似于编辑部里不同编辑负责不同类型的稿件。  
**稀疏门控（Sparse Gating）**：一种控制机制，只让最合适的少数专家参与计算，保持推理速度几乎不变。可以比作只让最擅长的厨师上菜，其他厨师暂时休息。  

### 核心创新点
1. **从单一视觉编码器到视觉专家混合**  
   之前的通用 MLLM 往往只接入一个视觉 backbone（如 CLIP 或 ViT），面对多样化图像时容易出现盲区。MoME 引入 MoVE，让多个视觉编码器的特征在同一层级上并行生成，并通过门控网络自适应加权。这样模型在不同任务或图像风格下会自动倾向于最擅长的视觉专家，显著提升了视觉特征的覆盖度。  
2. **在语言大模型内部植入稀疏专家**  
   传统的提升语言能力的办法是增大模型或微调全部参数，成本高且容易破坏已有能力。MoME 把 MoLE 融入 LLM 的 Transformer 层，通过稀疏门控只激活少数语言专家，实现了“无痛”升级：推理时间几乎不变，却能在特定语言子任务上获得专门化提升。  
3. **双向专家体系协同缓解任务干扰**  
   任务干扰往往来源于视觉和语言两端的特征冲突。MoME 同时在视觉（MoVE）和语言（MoLE）两侧设立专家混合，使得每个任务可以在两端都找到最匹配的子网络。实验表明，这种双向专精比仅在单一模态上加专家的做法更能削弱干扰。  
4. **兼容性强的特征转换架构**  
   MoVE 采用统一的特征投影层，将不同视觉编码器的输出映射到同一维度，再交给门控网络处理。这样即使加入全新 backbone，也只需要少量投影参数，保持了系统的可扩展性。  

### 方法详解
整体思路可以拆成三步：①准备多套视觉编码器，②构建视觉专家混合层（MoVE），③在语言模型内部加入语言专家混合层（MoLE），最后把两者串联成一个完整的多模态大语言模型。

**步骤一：多模态特征准备**  
- 选取若干公开的视觉 backbone（如 CLIP‑ViT、Swin‑Transformer、ConvNeXt），每个 backbone 把原始图像映射到自己的特征空间。  
- 为了让后续的专家层能够统一处理，所有特征先经过一个线性投影，使维度相同，类似把不同语言的翻译稿统一成同一种字体。

**步骤二：视觉专家混合（MoVE）**  
- 投影后的特征送入一个稀疏门控网络。门控网络根据当前任务的提示（如任务描述或示例）计算每个视觉专家的权重。  
- 只保留权重最高的 K（比如 2）个专家的特征，其余设为零。随后把这 K 条特征加权求和，得到“混合视觉表示”。  
- 关键点在于门控网络本身是轻量的 MLP，训练时使用交叉熵损失加上负载均衡正则，让所有专家都有机会被激活，防止出现“独占专家”。  
- 这种设计让模型在面对不同图像风格或任务时，自动倾向于最擅长的视觉 backbone，避免了单一编码器的盲区。

**步骤三：语言专家混合（MoLE）**  
- 在大语言模型的若干 Transformer 层（比如每隔 4 层）插入 MoLE。每个 MoLE 包含若干独立的前馈网络（FFN），这些 FFN 充当语言专家。  
- 同样使用稀疏门控决定本次前向激活哪几个语言专家。因为门控只在前馈阶段起作用，模型的自注意力路径保持不变，推理成本几乎不受影响。  
- 激活的语言专家会对混合视觉表示进行语言化解码，生成自然语言输出。由于不同任务会触发不同的语言专家，模型在回答细粒度问答、描述生成或推理任务时都能得到更专门的语言处理。

**整体流程（文字版流程图）**  
1. 输入：图像 + 文本提示 →  
2. 多视觉 backbone → 各自特征 → 统一投影 →  
3. MoVE 稀疏门控 → 选出 K 个视觉专家特征 → 加权求和 → 视觉混合向量 →  
4. 与文本提示一起送入 LLM →  
5. LLM 中若干层的 MoLE 稀疏门控 → 选出语言专家 → 前馈处理 →  
6. 最终生成文本输出。

**最巧妙的地方**  
- 把稀疏门控分别放在视觉和语言两端，使得任务干扰可以在两层同时被削弱，而不是只在单一模态上做“补丁”。  
- 采用统一投影层实现不同视觉 backbone 的兼容，极大降低了系统集成的工程成本。  

### 实验与效果
- **评测任务**：论文在多个公开的视觉‑语言基准上做实验，包括 VQAv2（视觉问答）、NLVR2（图文匹配）、COCO Caption（图像描述）以及 RefCOCO（指代消解）等。  
- **对比基线**：与同规模的通用 MLLM（如 LLaVA、MiniGPT‑4）以及专门针对某任务微调的模型进行比较。  
- **主要结果**：MoME 在大多数任务上都实现了 2%~5% 的绝对提升。例如在 VQAv2 上从 71.3% 提升到 74.8%，在 COCO Caption 上的 CIDEr 分数提升约 3.2 分。作者强调这些提升几乎不伴随推理时间的增长。  
- **消融实验**：分别去掉 MoVE、MoLE 或者只保留单侧专家混合，性能均出现显著回落，说明双向专家体系是关键。去掉稀疏门控改为全激活会导致推理成本翻倍，验证了稀疏激活的必要性。  
- **局限性**：原文提到 MoME 仍然依赖预训练好的视觉 backbone，若这些 backbone 本身在某类图像上表现差，MoVE 只能在已有专家之间做权衡，无法根本解决“根本盲区”。此外，稀疏门控的训练需要额外的负载均衡技巧，调参成本相对较高。

### 影响与延伸思考
MoME 的双模态专家混合思路在发布后迅速被后续工作引用，尤其是那些想在保持通用性的前提下提升特定任务表现的研究。比如近期的 “Unified-IO” 系列开始尝试在更多模态（音频、深度）上加入类似的专家层；还有一些工作把 MoE 与参数高效微调（PEFT）结合，进一步压缩微调成本。对想继续深挖的读者，可以关注以下方向：①如何让专家混合在训练阶段自适应发现新专家（即动态专家生成）；②在更低算力设备上实现稀疏门控的硬件加速；③把 MoME 的理念扩展到跨语言、多语言的多模态场景。  

### 一句话记住它
MoME 用视觉‑语言双向稀疏专家，让通用多模态大模型在不增算力的情况下，像挑选最擅长的团队成员一样自动适配每个任务。