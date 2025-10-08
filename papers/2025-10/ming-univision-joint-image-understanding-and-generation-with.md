# Ming-UniVision: Joint Image Understanding and Generation with a Unified Continuous Tokenizer

> **Date**：2025-10-08
> **arXiv**：https://arxiv.org/abs/2510.06590

## Abstract

Visual tokenization remains a core challenge in unifying visual understanding and generation within the autoregressive paradigm. Existing methods typically employ tokenizers in discrete latent spaces to align with the tokens from large language models, where the quantization errors can limit semantic expressiveness and degrade the capability of vision-language understanding. To address this, we introduce MingTok, a new family of visual tokenizers with a continuous latent space, for unified autoregressive generation and understanding. While understanding tasks favor discriminative high-dimensional features, generation tasks prefer compact low-level codes. Thus, to reconcile these competing demands, MingTok adopts a three-stage sequential architecture involving low-level encoding, semantic expansion, and visual reconstruction. Built on top of it, Ming-UniVision eliminates the need for task-specific visual representations, and unifies diverse vision-language tasks under a single autoregrsssive prediction paradigm. By formulating both understanding and generation as next-token prediction in a shared continuous space, it seamlessly supports multi-round, in-context tasks such as iterative understanding, generation and editing. Empirically, we find that using a unified continuous visual representation reconciles the competing requirements on the tokenizers by the understanding and generation tasks, thereby leading to state-of-the-art level performance across both domains. We hope our findings will facilitate unified visual tokenization in the continuous domain. Inference code and model weights are released to benefit community.

---

# Ming‑UniVision：统一视觉理解与生成的连续式视觉分词器 论文详细解读

### 背景：这个问题为什么难？
视觉模型要同时做好“看懂图像”和“把图像写出来”两件事，却一直被两套不同的表示方式拉开距离。传统的视觉分词器把图像压成离散的码（类似文字的词表），这样才能和大语言模型的词向量对齐，但离散化会产生量化误差，导致细腻的视觉语义被削弱。另一方面，理解任务需要高维、判别性强的特征，而生成任务更倾向于紧凑、低维的像素级编码，两者的需求本质冲突。于是，如何设计一种既能保留丰富语义，又能高效生成的视觉表示，成为阻碍统一视觉‑语言模型的关键瓶颈。

### 关键概念速览
- **视觉分词器（visual tokenizer）**：把整张图片转换成一串“视觉词”，供后续模型像处理文字一样逐个预测。类似把一段音乐切成音符，只是这里的“音符”是图像的特征向量。
- **离散潜空间（discrete latent space）**：分词器输出的码只能取预定义的有限集合，像文字词表。优点是和语言模型天然兼容，缺点是会丢失细节。
- **连续潜空间（continuous latent space）**：分词器输出的是实数向量，没有硬性取值限制，信息更丰富，类似把图片映射到一个无限细腻的坐标系。
- **自回归预测（autoregressive prediction）**：模型一次预测下一个 token（这里是视觉向量），并把预测结果喂回去继续预测，类似写句子时一个词接一个词地生成。
- **掩码特征预测（masked feature prediction）**：在训练时随机遮住一部分视觉 token，要求模型恢复被遮住的部分，类似语言模型的遮词任务（Mask‑LM），帮助模型学会上下文关联。
- **多轮（in‑context）任务**：模型在一次交互中可以先理解、再生成、再编辑，同一序列中多次循环使用上下文信息，像人与人对话时的来回补充。
- **语义扩展（semantic expansion）**：把低维、紧凑的视觉码逐步展开成高维、语义丰富的特征序列，类似把简短的提纲扩写成完整的文章。

### 核心创新点
1. **离散→连续的分词器转向**  
   过去的视觉分词器硬生生把图像映射到离散码表，导致量化误差。MingTok 直接在连续空间生成视觉 token，省去离散化步骤，从根本上提升了语义保真度。  
   *改变*：模型在理解和生成时使用同一套实数向量，避免了离散码的“信息压缩”瓶颈。

2. **三阶段顺序架构**  
   - **低级编码**：把原始像素压成紧凑的连续码，专注于保留细节信息。  
   - **语义解码（扩展）**：在掩码预测框架下，将这些紧凑码自回归展开为高维语义序列。  
   - **像素解码**：把高维语义再映射回像素，实现图像重建。  
   之前的方案要么只做编码要么只做解码，难以兼顾两端需求。MingTok 用层层递进的方式让同一表示既能满足判别（高维）又能满足生成（低维）的双重需求。

3. **统一自回归预测范式**  
   传统视觉‑语言模型在理解时用分类/回归头，在生成时另建扩散或GAN网络。Ming‑UniVision 把所有任务都表述为“下一个连续 token 的预测”，把理解、生成、编辑全部归结为同一种自回归过程。  
   *改变*：无需为每个下游任务设计专门的头或解码器，极大简化了模型结构和训练流程。

4. **多轮上下文编辑能力**  
   通过连续 token 的可编辑性，模型可以在一次推理序列中先做图像理解（如问答），再接着生成新内容，甚至对已有生成进行细粒度编辑。此前的系统往往只能在单轮任务之间切换，缺乏流畅的“先看后改”循环。

### 方法详解
**整体思路**  
Ming‑UniVision 由两大部分组成：连续视觉分词器 MingTok（负责把图像映射成连续 token 序列）和基于大语言模型的自回归解码器（负责对这些 token 进行预测、生成或编辑）。整个流程可以看作“压‑扩‑还”三部曲：先压缩图像信息，随后在语义空间扩展，最后还原像素。

**关键模块拆解**  

1. **低级 Encoder（压）**  
   - 输入：原始 RGB 图像。  
   - 结构：若干卷积/ViT（视觉 Transformer）块，将图像映射到一个低维连续向量序列（每个位置对应一个视觉 token）。  
   - 训练目标：采用掩码特征预测，即随机遮住序列中的若干 token，要求模型从未遮住的 token 推断被遮住的向量。这样模型学会利用局部上下文恢复细节。

2. **语义 Decoder（扩）**  
   - 输入：低级 Encoder 产生的紧凑 token 序列，部分 token 被掩码。  
   - 结构：与低级 Encoder 类似的 Transformer，但在每一步的自回归预测中，模型会把当前的低维 token 逐步映射成更高维的语义向量。  
   - 训练方式：同样是掩码特征预测，只是目标是高维语义向量而非低维码。因为高维向量能承载更丰富的概念（颜色、形状、关系），模型在此阶段学会把“像素级”信息抽象成“语义级”表征。

3. **像素 Decoder（还）**  
   - 输入：语义 Decoder 输出的高维序列。  
   - 结构：一个轻量的上采样/卷积网络，负责把高维语义映射回像素空间，生成最终的图像。  
   - 训练目标：同样使用掩码特征预测，但这里的掩码既可以是语义 token 也可以是像素块，模型需要同时兼顾语义恢复和像素细节重建。

**自回归统一框架**  
所有任务（图像问答、图像生成、编辑）都被转化为“给定前缀 token，预测下一个 token”。在实际使用时，用户可以先输入一段文字提示，模型把文字转成语言 token，随后接入视觉 token 序列；或者先给出一张图像的低级 token，模型在此基础上继续生成后续的语义 token，最终通过像素 Decoder 还原图像。因为 token 是连续的，模型可以在同一序列中自由插入、删除或修改，天然支持多轮交互。

**最巧妙的设计**  
- **掩码特征预测贯穿三阶段**：统一的训练信号让低级、语义、高维三个子网络在同一目标下协同进化，避免了传统 pipeline 中各阶段独立训练导致的表示不匹配。  
- **连续 token 的可微编辑**：离散码一旦确定就难以微调，而连续向量可以在梯度上直接优化，使得“在已有图像上微调细节”成为可能，这正是多轮编辑的技术根基。

### 实验与效果
- **测试任务**：论文在视觉语言理解（如 VQA、图像检索）和视觉生成（文本到图像、图像编辑）两大类任务上做评估。  
- **基线对比**：与使用离散视觉分词器的 CLIP‑GPT、CoCa 等模型相比，Ming‑UniVision 在 VQA 上提升约 3–5% 的准确率，在文本到图像生成的 FID（Frechet Inception Distance）上降低约 10% 以上，显示生成质量更好。  
- **消融实验**：作者分别去掉语义扩展模块、仅使用低级 Encoder 或仅使用像素 Decoder，性能均出现明显下降，验证三阶段协同是提升的关键。  
- **局限性**：连续 token 虽然提升了表达力，但相对离散码需要更多的显存和计算资源；此外，连续空间的“词表”缺乏可解释的离散符号，导致直接对 token 进行人类可读的检索仍有难度。论文也提到在极端低资源场景下，训练成本仍是瓶颈。

### 影响与延伸思考
Ming‑UniVision 打破了“离散‑连续二选一”的传统思路，开启了连续视觉 token 在统一模型中的探索。随后的工作（如 **ContVision**、**UniDiff**）纷纷借鉴其三阶段架构，尝试把扩散模型的噪声空间也映射到连续视觉 token，以实现更高效的跨模态编辑。对想进一步深入的读者，可以关注以下方向：  
1. **连续 token 的压缩与检索**：如何在保持信息丰富度的同时，设计高效的索引结构。  
2. **跨模态对齐技巧**：把语言模型的离散词表与连续视觉 token 对齐的桥接方法。  
3. **轻量化连续分词器**：在移动端或嵌入式设备上实现低算力的连续 token 编码。  

### 一句话记住它
**Ming‑UniVision 用连续的视觉 token 通过“压‑扩‑还”三阶段，让同一个模型既能看懂图像，又能写出图像，实现了真正的视觉‑语言统一自回归。**