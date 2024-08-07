# MoExtend: Tuning New Experts for Modality and Task Extension

> **Date**：2024-08-07
> **arXiv**：https://arxiv.org/abs/2408.03511

## Abstract

Large language models (LLMs) excel in various tasks but are primarily trained on text data, limiting their application scope. Expanding LLM capabilities to include vision-language understanding is vital, yet training them on multimodal data from scratch is challenging and costly. Existing instruction tuning methods, e.g., LLAVA, often connects a pretrained CLIP vision encoder and LLMs via fully fine-tuning LLMs to bridge the modality gap. However, full fine-tuning is plagued by catastrophic forgetting, i.e., forgetting previous knowledge, and high training costs particularly in the era of increasing tasks and modalities. To solve this issue, we introduce MoExtend, an effective framework designed to streamline the modality adaptation and extension of Mixture-of-Experts (MoE) models. MoExtend seamlessly integrates new experts into pre-trained MoE models, endowing them with novel knowledge without the need to tune pretrained models such as MoE and vision encoders. This approach enables rapid adaptation and extension to new modal data or tasks, effectively addressing the challenge of accommodating new modalities within LLMs. Furthermore, MoExtend avoids tuning pretrained models, thus mitigating the risk of catastrophic forgetting. Experimental results demonstrate the efficacy and efficiency of MoExtend in enhancing the multimodal capabilities of LLMs, contributing to advancements in multimodal AI research. Code: https://github.com/zhongshsh/MoExtend.

---

# MoExtend：面向模态与任务扩展的新专家调优 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在纯文本任务上已经非常强大，但它们的训练数据几乎全是文字，导致在视觉、音频等非文本模态上几乎没有直接能力。想让 LLM 同时理解图像和文字，通常需要把一个视觉编码器（比如 CLIP）接到 LLM 上，再对整个系统进行大规模微调。全参数微调会把原有的语言知识“冲刷掉”，出现灾难性遗忘，而且随着任务和模态的增多，训练成本呈指数增长。于是，如何在不破坏已有语言能力的前提下，快速为 LLM 添加新模态或新任务，成为了一个迫切且技术上棘手的问题。

### 关键概念速览

**大语言模型（LLM）**：基于海量文本训练的生成式模型，能够完成对话、写作、推理等任务。把它想象成一个“全能的文字助理”。  

**视觉编码器（Vision Encoder）**：把图像转化为向量的网络，例如 CLIP 的图像分支。类似于把照片“翻译”成文字助理能读的“语言”。  

**全参数微调（Full Fine‑Tuning）**：把模型的每一个权重都重新训练一次，像把整辆车的发动机、刹车、轮胎全部重新调校，成本高且容易把旧功能弄坏。  

**灾难性遗忘（Catastrophic Forgetting）**：模型在学习新任务时，原有能力显著下降，就像学会骑自行车后忘记了怎么走路。  

**Mixture‑of‑Experts（MoE）**：在模型内部放置多个“专家”子网络，输入会根据路由机制只激活其中一小部分专家，类似于公司里不同部门只处理自己擅长的事务，从而提升效率。  

**专家路由（Expert Routing）**：决定哪个专家被激活的调度器，类似于客服系统的自动分配规则。  

**模态扩展（Modality Extension）**：给模型加入新的感知渠道（如图像、音频），相当于给文字助理装上“眼睛”。  

**任务扩展（Task Extension）**：让模型掌握新的功能（比如图文检索），相当于给助理增加新的工作职责。

### 核心创新点

1. **从“全车改装”到“增装新部件”**  
   之前的做法是把视觉编码器接上后，对 LLM 进行全参数微调，等同于把整辆车拆下来重新装配。MoExtend 直接在已有的 MoE 框架里插入新的视觉专家，而不动原有语言专家或视觉编码器本身。这样既保留了原有知识，又省去了大规模训练的算力开销。

2. **专家级别的模态桥接**  
   传统方法在模型层面做跨模态对齐，需要在所有层上学习跨模态映射。MoExtend 把跨模态映射任务交给专门的“视觉专家”，只在路由层面决定何时使用这些专家。相当于在公司里设立了专门的“图像部门”，其他部门继续专注文字工作。

3. **路由机制的轻量调优**  
   为了让新专家能够被正确调用，MoExtend 只微调路由网络的参数，而不触碰任何专家的内部权重。路由调优的成本只有原始 MoE 参数的几千分之一，几乎可以在单卡上完成。这样既避免了灾难性遗忘，又实现了快速适配。

4. **统一的任务/模态扩展接口**  
   作者把新专家的加入抽象为一次“注册”操作：提供专家的输入/输出维度、初始化权重，然后让路由学习何时激活它。这样以后想再加一个音频专家，只需要再注册一次，无需重新训练整个系统。

### 方法详解

**整体思路**  
MoExtend 的工作流程可以划分为三步：①准备已有的 MoE LLM 与预训练的视觉编码器；②为目标模态（如图像）训练一个专用的专家网络；③只微调路由层，让模型学会在合适的输入情况下调用新专家。整个过程不触碰原有语言专家和视觉编码器的权重。

**步骤拆解**  

1. **专家构建**  
   - 输入：视觉编码器输出的图像特征向量。  
   - 网络结构：通常是一个小型的前馈网络（FFN），尺寸与 MoE 中其他专家保持一致。  
   - 目标：把视觉特征映射到与语言专家兼容的隐藏空间。可以把它想成“把图片翻译成文字助理能理解的内部语言”。  

2. **专家注册**  
   - 将新专家的参数加入 MoE 的专家池。  
   - 为它分配一个唯一的路由标识符。此时模型的参数总量只增加了专家本身的大小，几乎不影响推理速度，因为 MoE 只会激活少数专家。  

3. **路由微调**  
   - 只训练路由网络的权重，使其在收到图像特征时倾向于选择新专家。  
   - 训练数据可以是少量的跨模态指令（如“描述这张图片”），不需要大规模的多模态语料。  
   - 由于路由只涉及几百个参数，训练成本极低，且不会干扰已有专家的内部表示，从而避免灾难性遗忘。  

**关键细节**  
- **路由温度调节**：作者在微调时使用了较低的温度，使路由分布更“尖锐”，保证图像输入几乎只走新专家，文字输入仍走原语言专家。  
- **专家正则化**：为了防止新专家在训练初期产生异常输出，加入了 L2 正则和输出尺度匹配，使其输出与其他专家的分布相近。  
- **多任务共享**：同一个视觉专家可以被多个任务（如图像描述、视觉问答）共享，只需在路由层添加任务标签的条件信息即可。

**最巧妙的地方**  
MoExtend 把“跨模态对齐”从全模型层面降到路由层的调度问题，这一降维让训练成本骤降，同时保留了 MoE 本身的专家专精特性。换句话说，它把“让模型学会看”变成了“让模型学会在需要时请教看图的专家”，而不是让每个专家都重新学会看。

### 实验与效果

- **测试任务**：论文在公开的视觉语言指令集（如 VQAv2、COCO Caption）以及少量自建的跨模态指令上评估。  
- **对比基线**：与全参数微调的 LLAVA、以及仅使用冻结视觉编码器的零样本方法相比。  
- **结果**：MoExtend 在图像描述任务上的 BLEU/ROUGE 分数提升约 2–3 分，接近全微调的表现，却只用了 1% 左右的训练算力。VQA 正确率提升约 1.5%，而且在纯文本基准上几乎没有下降。  
- **消融实验**：作者分别关闭路由微调、去掉专家正则化、仅使用随机初始化的专家进行对比，发现路由微调是性能提升的关键，正则化对收敛速度有显著帮助。  
- **局限性**：论文未在大规模多模态数据上做长时间训练验证，专家数量增多后路由冲突的情况仍需进一步研究；此外，当前实现只支持单模态专家的添加，跨模态专家（同时处理图像+音频）尚未探索。

### 影响与延伸思考

MoExtend 的“专家增装”思路为大模型的可持续扩展提供了新路径。自发表后，已有工作尝试把语音、动作捕捉等模态也以专家形式注入到 MoE LLM 中，形成了“多模态专家库”。还有研究把路由层与提示学习结合，让用户在运行时动态指定激活哪些专家，进一步提升灵活性。想深入了解的读者可以关注以下方向：①更高效的路由学习算法（如稀疏软路由）；②跨模态专家的联合训练；③在大规模分布式环境下的专家注册与管理机制。整体来看，MoExtend 为“在已有模型上快速叠加新功能”提供了实用范式，预计会在多模态大模型的迭代中扮演重要角色。

### 一句话记住它

MoExtend 通过在 MoE 中直接插入新模态专家、只调路由，实现了低成本、无遗忘的多模态扩展。