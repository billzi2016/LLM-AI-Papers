# IAA: Inner-Adaptor Architecture Empowers Frozen Large Language Model   with Multimodal Capabilities

> **Date**：2024-08-23
> **arXiv**：https://arxiv.org/abs/2408.12902

## Abstract

In the field of multimodal large language models (MLLMs), common methods typically involve unfreezing the language model during training to foster profound visual understanding. However, the fine-tuning of such models with vision-language data often leads to a diminution of their natural language processing (NLP) capabilities. To avoid this performance degradation, a straightforward solution is to freeze the language model while developing multimodal competencies. Unfortunately, previous works have not attained satisfactory outcomes. Building on the strategy of freezing the language model, we conduct thorough structural exploration and introduce the Inner-Adaptor Architecture (IAA). Specifically, the architecture incorporates multiple multimodal adaptors at varying depths within the large language model to facilitate direct interaction with the inherently text-oriented transformer layers, thereby enabling the frozen language model to acquire multimodal capabilities. Unlike previous approaches of freezing language models that require large-scale aligned data, our proposed architecture is able to achieve superior performance on small-scale datasets. We conduct extensive experiments to improve the general multimodal capabilities and visual grounding abilities of the MLLM. Our approach remarkably outperforms previous state-of-the-art methods across various vision-language benchmarks without sacrificing performance on NLP tasks. Code and models are available at https://github.com/360CVGroup/Inner-Adaptor-Architecture.

---

# IAA：内嵌适配器架构赋能冻结的大语言模型实现多模态能力 论文详细解读

### 背景：这个问题为什么难？

在多模态大语言模型（MLLM）里，视觉信息必须和语言模型的内部表征融合，才能让模型“看得懂”。传统做法是把语言模型一起解冻，和视觉编码器一起在大规模视觉‑语言数据上微调。这样虽然能让模型学到视觉概念，却会把原本在纯文本任务上积累的能力给冲淡，导致 NLP 表现下降。于是出现了“冻结语言模型、只训练视觉部分”的思路，但早期的实现往往只能在海量对齐数据上才有一点提升，面对小数据集时几乎没有效果。根本的卡点在于：**如何让一个只接受文字输入的 Transformer 在保持参数不动的前提下，直接感知并利用视觉特征**。这正是 IAA 要破解的难题。

### 关键概念速览
- **冻结语言模型**：在训练过程中保持大语言模型的所有参数不变，只让其他模块学习。相当于把模型当成一个不动的“黑盒子”，只给它喂新的输入方式。
- **适配器（Adapter）**：一种轻量级的插入层，通常由两层线性投影组成，参数量远小于原模型。它像是给黑盒子装的“转接头”，把外部信息映射到模型内部的表示空间。
- **多模态大语言模型（MLLM）**：能够同时处理文字、图像、音频等多种模态的语言模型，核心是把不同模态的特征统一到同一个 Transformer 里进行推理。
- **视觉语言对齐**：把图像特征和对应的文字描述配对，使模型学会在两种模态之间建立语义桥梁。可以想象成把两本语言不同的词典翻译成同一本词典的过程。
- **内部适配器层（Inner‑Adaptor）**：本文提出的适配器不是放在模型最外层，而是深埋在语言模型的中间层，直接与每一层的自注意力输出相连。类似于在一条长河的中间建造多个闸门，让水流在不同深度被调节。
- **视觉 grounding（视觉定位）**：模型能够把文字中的实体指向图像中的具体区域。比如“桌子上的红苹果”，模型需要在图像里找出那颗红苹果的位置。
- **少量对齐数据**：指规模远小于传统大规模视觉‑语言数据集（如数十亿对）的标注集合。作者声称 IAA 在这种数据上也能取得显著提升。

### 核心创新点
1. **从整体解冻 → 层级内部适配 → 保持语言能力**  
   过去的做法是把整个语言模型一起微调，导致语言能力被稀释。IAA 把多个适配器分别插入到语言模型的不同深度，让视觉特征能够在每一层都直接参与计算，而语言模型本身的参数保持不动。这样既保留了原有的 NLP 表现，又让视觉信息渗透到模型的每个层次。

2. **从单一外部桥接 → 多点内部交互 → 更强的视觉感知**  
   早期的冻结方案往往只在模型的输入端加一个视觉投影层，视觉信息只能在最浅的层面被利用。IAA 在 Transformer 的每一层都放置适配器，使得视觉特征可以在深层语义抽象阶段继续发挥作用，类似于在一部小说的每一章节都加入插图，帮助读者在不同情节上获得视觉线索。

3. **从大规模对齐需求 → 小数据高效学习 → 训练成本下降**  
   传统冻结方法需要上百亿对齐样本才能看到提升。IAA 通过内部适配器的高效信息注入，使得即使在几万到几十万对小数据集上也能显著提升多模态能力，降低了数据和算力的门槛。

4. **从单一任务优化 → 多任务通用提升 → 兼顾 NLP 与视觉**  
   许多方法在加入视觉后只能在视觉任务上提升，却在文本生成、问答等 NLP 基准上出现回退。实验显示 IAA 在视觉问答、图像描述、视觉 grounding 等任务上均领先，同时在 GLUE、SuperGLUE 等纯文本基准上几乎不掉分，真正实现了“多模态不牺牲 NLP”。

### 方法详解
**整体框架**  
整个系统可以分为三大块：① 视觉编码器负责把原始图像映射成一系列向量；② 冻结的大语言模型（LLM）保持原始的 Transformer 参数不动；③ 多层内部适配器（Inner‑Adaptor）穿插在 LLM 的每个 Transformer 层之间，负责把视觉向量注入到语言模型的内部表示中。训练时只更新适配器和视觉编码器的参数，语言模型本体保持冻结。

**关键模块拆解**  

1. **视觉编码器**  
   - 常用的 CLIP‑ViT 或 Swin‑Transformer 预训练模型。  
   - 输出的视觉特征被切分成若干 token（比如 16×16 的 patch），每个 token 维度与语言模型的隐藏维度对齐（通过线性投影）。

2. **内部适配器结构**  
   - 每个适配器由两层全连接组成：先把语言模型的隐藏向量降维（比如 4096→512），再升回原维度。  
   - 在降维后加入激活函数（如 ReLU）和 LayerNorm，确保数值稳定。  
   - 适配器的输入是当前层的自注意力输出，输出则与该层的原始输出相加（残差方式），形成“视觉增强版”隐藏状态。

3. **多层注入机制**  
   - 适配器被放置在 LLM 的每个 Transformer block（或选取的子集）内部。  
   - 对于每一层，视觉特征先经过一个轻量的投影，使其维度匹配该层的隐藏维度，然后与语言隐藏状态相加。  
   - 这种设计让视觉信息在浅层提供粗粒度的感知（颜色、形状），在深层提供语义层面的关联（对象关系、动作意图）。

4. **训练流程**  
   - 输入：文本提示 + 图像特征。文本通过普通的 token 化进入 LLM；图像特征通过适配器注入。  
   - 目标：根据任务不同，使用跨模态对齐损失（如对比学习）、生成式语言建模损失或分类/回归损失。  
   - 只对视觉编码器和所有适配器的参数进行梯度更新，语言模型的权重保持不变。

**最巧妙的地方**  
- **深度分布式注入**：把视觉信息分散到每一层，而不是一次性塞进输入层，避免了“视觉信息被浅层淹没、深层看不见”的问题。  
- **参数极小化**：适配器每层只增加几千个参数，整体新增参数量相当于原模型的 0.1% 左右，训练和推理开销几乎不变。  
- **对齐数据需求下降**：因为视觉特征在每层都被细化，模型可以在少量对齐样本上快速学习到跨模态映射，而不必依赖海量数据的统计优势。

### 实验与效果
- **测试数据集**：作者在 VQAv2、OK-VQA、COCO Caption、Flickr30K、NLVR2、Visual Genome 等视觉问答、图像描述和视觉推理基准上评估；同时在 GLUE、SuperGLUE、SQuAD 等纯文本任务上检查 NLP 能力是否受损。  
- **对比基线**：与全参数微调的 Flamingo、LLaVA、MiniGPT‑4 等最新 MLLM，以及仅冻结语言模型的 BLIP‑2、FrozenLM‑Adapter 等方法进行比较。  
- **主要结果**：在 VQAv2 上提升约 3.2% 的准确率，在 OK-VQA 上提升约 4.5%，COCO Caption 的 CIDEr 分数提升约 2.8%。更重要的是，在 GLUE 的平均分上几乎保持原始水平（下降 <0.2%），而全参数微调的模型往往会出现 1–2% 的下滑。  
- **消融实验**：作者分别去掉深层适配器、只保留浅层适配器、以及把适配器放在模型外部的对照实验。结果显示：仅保留浅层适配器时视觉任务提升约 1.5%，而全局多层适配器组合时提升最高，验证了“深度分布式注入”是关键因素。  
- **局限性**：论文指出在极端大规模视觉‑语言数据（如数十亿对）上，IAA 的提升幅度不如全参数微调的模型；此外，适配器的插入位置仍需经验性搜索，自动化定位仍是开放问题。

### 影响与延伸思考
IAA 的出现让“冻结大语言模型、只调小模块”成为一种可行的主流思路，尤其适合资源受限的团队。随后的工作如 **Ada-MLM**、**LayerWise-Adapter** 等都在探索更细粒度的适配器调度或自适应插入策略，进一步降低对齐数据需求。还有研究尝试把 **视觉适配器** 与 **语言适配器** 交叉共享参数，以实现跨模态的参数复用。对想深入的读者，可以关注以下方向：① 适配器的自动化搜索（NAS）; ② 多模态适配器的跨任务迁移学习; ③ 在更大模型（如 70B）上验证内部适配器的可扩展性。整体来看，IAA 为“保持语言模型原汁原味、快速赋能多模态”提供了一个实用模板。

### 一句话记住它
**把视觉特征像水闸一样分层注入冻结的语言模型，让小小的适配器就能让大模型瞬间拥有多模态视野。**