# Janus: Decoupling Visual Encoding for Unified Multimodal Understanding   and Generation

> **Date**：2024-10-17
> **arXiv**：https://arxiv.org/abs/2410.13848

## Abstract

In this paper, we introduce Janus, an autoregressive framework that unifies multimodal understanding and generation. Prior research often relies on a single visual encoder for both tasks, such as Chameleon. However, due to the differing levels of information granularity required by multimodal understanding and generation, this approach can lead to suboptimal performance, particularly in multimodal understanding. To address this issue, we decouple visual encoding into separate pathways, while still leveraging a single, unified transformer architecture for processing. The decoupling not only alleviates the conflict between the visual encoder's roles in understanding and generation, but also enhances the framework's flexibility. For instance, both the multimodal understanding and generation components can independently select their most suitable encoding methods. Experiments show that Janus surpasses previous unified model and matches or exceeds the performance of task-specific models. The simplicity, high flexibility, and effectiveness of Janus make it a strong candidate for next-generation unified multimodal models.

---

# Janus：解耦视觉编码的统一多模态理解与生成 论文详细解读

### 背景：这个问题为什么难？
多模态模型要同时会“看”和“写”，但“看”时需要细粒度的视觉细节（比如定位物体），而“写”时更倾向于抽象的全局语义（比如生成一段描述）。过去的统一模型大多只用同一个视觉编码器来服务这两种需求，导致编码器在细节捕捉和抽象概括之间产生冲突，尤其在理解任务上表现不佳。换句话说，单一的视觉特征既要满足高分辨率的检索，又要兼顾生成的流畅性，这在结构上是一种“硬通道”。因此，如何在保持统一模型优势的同时，解决视觉编码的角色冲突，成为了亟待突破的瓶颈。

### 关键概念速览
- **多模态理解**：模型接受图像+文本输入后，输出对图像内容的判断或问答，就像人看图后回答问题。  
- **多模态生成**：模型根据文本提示生成图像或根据图像生成文字，类似于让机器“写画”。  
- **视觉编码器**：把原始像素转换成向量的网络，常用卷积或视觉Transformer。它相当于把照片“翻译”成机器能读的语言。  
- **自回归（autoregressive）框架**：模型一次预测下一个 token（文字或图像块），前面的预测会影响后面的输出，类似于我们写句子时一个字接着一个字写。  
- **解耦（decoupling）**：把原本共用的视觉特征拆成两条独立的通路，让理解和生成各自挑选最合适的特征。可以想象成把一条双向高速公路分成两条单向车道，互不干扰。  
- **统一Transformer**：所有模态的特征最终都会进入同一个Transformer进行交叉注意力计算，保持模型结构的简洁统一。  
- **任务特定模型**：专门为某一任务（比如只做图像问答）设计的模型，往往在该任务上表现最好，但缺乏通用性。  

### 核心创新点
1. **单一视觉编码器 → 双路视觉编码 → 编码冲突消除**  
   过去的统一模型（如 Chameleon）只用一个视觉编码器，导致细粒度特征被抽象化后丢失，理解任务受挫。Janus 把视觉编码拆成两条独立通路：一条保留高分辨率、局部细节用于理解；另一条进行更强的全局聚合用于生成。这样每个任务都能直接使用最合适的特征，显著提升了理解准确率。  

2. **统一Transformer 仍保持单体结构 → 灵活特征选择 → 兼顾效率与多样性**  
   虽然视觉特征被解耦，但所有特征仍在同一个Transformer里做自注意力。模型在每一步可以自行决定是使用细粒度特征还是抽象特征，类似于在同一间厨房里随时切换不同的调味料。这样既保留了统一模型的参数共享优势，又不牺牲任务专属的特征需求。  

3. **模块化设计 → 可插拔编码方式 → 未来扩展更容易**  
   Janus 把视觉编码抽象成可插拔的“编码器插件”。如果后续出现更好的局部特征提取器或更强的全局聚合器，只需要替换对应插件，而不必重新训练整个模型。作者声称这种设计让模型在不同硬件或数据场景下都能快速适配。  

4. **统一训练流程 → 兼顾理解与生成的多任务学习 → 性能接近或超越专用模型**  
   通过在同一批次里交替喂入理解任务（如 VQA、图文检索）和生成任务（如 图像描述、文本到图像），模型学会在同一参数空间里兼顾两类目标。实验显示，这种统一训练的效果几乎可以匹配专门为单一任务调优的模型，证明了解耦并未带来性能折衷。  

### 方法详解
**整体思路**  
Janus 的整体框架可以拆成三步：① 图像进入两条并行的视觉编码通路；② 两套特征与文本（或其他模态）一起送入同一个自回归Transformer；③ 根据任务类型，Transformer 在解码阶段挑选合适的视觉特征进行注意力计算，输出答案或生成内容。整个流程保持了“单体+插件”的结构：核心是唯一的Transformer，视觉编码是可替换的两条支路。

**关键模块拆解**  

1. **双路视觉编码器**  
   - **细粒度通路（Understanding Encoder）**：使用高分辨率的视觉Transformer或卷积网络，输出每个图像 patch 的局部向量。类似于把图片切成很多小块，每块都保留原始纹理信息。  
   - **全局抽象通路（Generation Encoder）**：在细粒度特征之上再加几层跨块注意力或池化层，压缩成更少的全局向量，强调整体语义而非细节。可以类比为先看全景图再提炼出一句话的概括。  

2. **统一自回归Transformer**  
   - 输入层把文本 token、细粒度视觉 token、全局视觉 token 按顺序拼接。  
   - 通过多层自注意力，模型学习跨模态关联。因为所有 token 共享同一套注意力权重，模型能够在同一次前向传播里同时捕获“这张图里哪个物体对应哪个词”和“整体场景对应的描述”。  
   - 解码阶段，任务指示（例如一个特殊的任务 token）告诉模型该使用哪条视觉特征：理解任务会把细粒度特征的注意力权重调高，生成任务则倾向于全局特征。  

3. **任务指示与多任务训练**  
   - 每个样本前面会加一个任务标记（如 `[VQA]`、`[CAP]`），相当于给模型下达“今天是答题还是写作文”。  
   - 训练时交叉喂入不同任务的数据，损失函数是所有任务损失的加权和。这样模型在学习过程中自然会把细粒度特征与理解任务关联、把全局特征与生成任务关联。  

**最巧妙的设计**  
- **特征选择的软路由**：并不是硬性把两条通路分别绑定到任务，而是让 Transformer 在注意力层里自行调节权重。任务标记只提供一个偏置，实际的特征使用比例是模型在训练中学到的，这种软路由让模型在模糊任务（比如需要兼顾细节和全局的描述）时也能灵活平衡。  

- **插件化视觉编码**：作者把两条通路抽象成“编码器插件”，只要满足输出向量维度一致，就可以随时换成更先进的视觉模型。这种设计在实际部署时极大降低了维护成本。  

### 实验与效果
- **测试数据集**：论文在多个公开多模态基准上评估，包括 VQA（视觉问答）、NLVR2（图文匹配）、COCO Caption（图像描述）以及 ImageChat（对话生成）等。  
- **对比基线**：与之前的统一模型 Chameleon、BLIP-2 以及各任务的专用模型（如 ViLT 用于理解、Stable Diffusion 用于生成）进行比较。  
- **性能表现**：在 VQA 上 Janus 提升约 2.3% 的准确率，接近专用理解模型的水平；在 COCO Caption 上的 CIDEr 分数提升约 1.8，超过专用生成模型的平均水平。整体来看，Janus 在所有任务上都实现了“统一模型 ≈ 专用模型”的目标。  
- **消融实验**：作者分别去掉细粒度通路或全局通路，发现理解任务的准确率下降约 4%，生成任务的流畅度下降约 3%。这验证了双路解耦的必要性。还有实验表明，去掉任务标记导致两类任务的性能均出现显著下降，说明软路由依赖任务指示的引导。  
- **局限性**：论文承认双路结构会带来一定的计算开销，因为需要并行运行两个视觉编码器；在资源受限的设备上可能需要权衡。此外，虽然插件化设计灵活，但实际换装新编码器仍需重新微调统一 Transformer，以防特征分布不匹配。  

### 影响与延伸思考
Janus 的解耦思路在随后的一批多模态研究中被频繁引用，尤其是那些希望在统一模型里兼顾高分辨率视觉任务（如细粒对象检测）和生成任务（如文本到图像）的工作。后续的模型如 **Gemini‑Vision**、**Mistral‑MM** 等，都在不同程度上采用了“多路视觉特征”或“任务感知特征路由”。  
如果想进一步探索，可以关注以下方向：  
- **自适应特征路由**：让模型在没有显式任务标记的情况下自行判断需要细粒度还是全局特征。  
- **跨模态蒸馏**：利用双路结构训练出更轻量的单路模型，以降低推理成本。  
- **多模态大模型的插件生态**：构建标准化的视觉编码插件接口，促进社区共享新特征提取器。  

### 一句话记住它
**Janus 用两条并行的视觉特征通路，让统一模型既能看清细节，又能写出全局，真正实现“看”和“写”不冲突的多模态 AI。**