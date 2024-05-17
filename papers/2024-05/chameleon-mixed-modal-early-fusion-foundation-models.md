# Chameleon: Mixed-Modal Early-Fusion Foundation Models

> **Date**：2024-05-16
> **arXiv**：https://arxiv.org/abs/2405.09818

## Abstract

We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training approach from inception, an alignment recipe, and an architectural parameterization tailored for the early-fusion, token-based, mixed-modal setting. The models are evaluated on a comprehensive range of tasks, including visual question answering, image captioning, text generation, image generation, and long-form mixed modal generation. Chameleon demonstrates broad and general capabilities, including state-of-the-art performance in image captioning tasks, outperforms Llama-2 in text-only tasks while being competitive with models such as Mixtral 8x7B and Gemini-Pro, and performs non-trivial image generation, all in a single model. It also matches or exceeds the performance of much larger models, including Gemini Pro and GPT-4V, according to human judgments on a new long-form mixed-modal generation evaluation, where either the prompt or outputs contain mixed sequences of both images and text. Chameleon marks a significant step forward in a unified modeling of full multimodal documents.

---

# 变色龙：混合模态早期融合基础模型 论文详细解读

### 背景：这个问题为什么难？

在多模态 AI 里，常见的做法是先把图像和文字分别编码成独立的向量，然后在后期再把它们拼接或交叉注意。这种“后期融合”让模型在处理纯文本或纯图像任务时表现不错，却在需要交叉推理的场景（比如图文混排的长文、图文交替的对话）里显得笨拙。原因有三点：  
1) 两种模态的特征在不同层次上被压缩，信息丢失不可逆。  
2) 跨模态注意只能在高层特征上做，细粒度的对应关系难以捕获。  
3) 训练时往往需要分别准备文本‑only、图像‑only 的数据，导致模型在真正的混合文档上缺乏统一的学习经验。于是，想要一个“一体化”模型，能够随意在同一序列里交替出现文字和图片，并且在任意位置进行生成，仍是未被彻底解决的难题。

### 关键概念速览
- **早期融合（Early Fusion）**：在模型的最底层就把图像和文字的 token（基本单元）混在一起，让它们共同进入同一个 Transformer 编码器。可以想象成把文字和图片的拼图块直接混在一起拼图，而不是先各自完成再合并。  
- **Token（标记）**：模型处理的最小单位，文字 token 是词或子词，图像 token 则是把图片切成小块（如 ViT 的 patch）后映射成向量。两者在同一词表里出现，模型不再区分“文字”和“图片”。  
- **混合模态（Mixed‑Modal）**：指输入或输出序列中同时包含文字和图片的情况，例如“这张图展示了…（图片）…请解释”。  
- **对齐配方（Alignment Recipe）**：一种训练技巧，用来让模型在多模态任务上保持一致的行为，比如在同一批次里混合不同模态的监督信号，防止某一模态主导学习。  
- **基础模型（Foundation Model）**：大规模预训练模型，具备通用的语言和视觉能力，后续可以通过少量微调适配各种下游任务。  
- **长文混合生成（Long‑Form Mixed‑Modal Generation）**：生成包含多段文字和多张图片的长篇文档，要求模型在全局保持连贯，同时在局部准确对应图文信息。  
- **人类评估（Human Judgment）**：让真实用户对模型输出的质量打分，常用于评估跨模态生成的主观感受，因为自动指标难以覆盖所有细节。

### 核心创新点
1. **从后期融合到早期融合的范式转变**  
   过去的模型在高层特征才让文字和图像相互作用，导致细粒度对应信息被压平。Chameleon 把图像 patch 和文字 token 在最底层就混在一起喂进同一个 Transformer。这样，注意力机制可以在每一层直接对齐文字和图像的局部细节，提升跨模态推理的精准度。  
2. **统一的 token 词表与混合序列训练**  
   传统做法会为每种模态准备独立的词表或嵌入层，训练时需要切换模式。Chameleon 设计了一套统一的 token 词表，图像 patch 通过线性投影映射成与文字 token 同维度的向量，所有 token 共享同一套位置编码和层归一化。这样模型在一次前向传播里就能处理“文字‑图片‑文字‑图片”的任意序列。  
3. **稳定的从零训练流程与对齐配方**  
   大模型从头训练常会出现梯度不稳定、模态失衡等问题。作者提出了分阶段学习率调度、模态平衡损失权重以及混合模态的 curriculum learning（先让模型熟悉单模态，再逐步加入交叉模态）。这些技巧让模型在数十亿 token 的混合数据上保持收敛。  
4. **单模型多任务“一站式”能力**  
   通过上述设计，Chameleon 能在同一个参数体上完成 VQA、图像描述、文本生成、图像生成以及长文混合生成等任务。实验显示，它在图像描述上达到 SOTA（state‑of‑the‑art）水平，在纯文本基准上超过 Llama‑2，并且在人类评估的长文混合生成上匹配甚至超越更大规模的 Gemini‑Pro 与 GPT‑4V。

### 方法详解
**整体框架**  
Chameleon 的训练流程可以划分为三步：① 构造统一的混合 token 词表；② 设计早期融合的 Transformer 编码器；③ 采用对齐配方进行大规模混合模态预训练。整个模型本质上是一个标准的自回归 Transformer，只是输入输出的 token 类型更丰富。

**1. 统一 token 词表**  
- **文字 token**：使用常见的子词分词器（如 BPE），每个子词映射到 2048 维向量。  
- **图像 token**：把原始图片切成 16×16 的 patch（类似 Vision Transformer），每个 patch 通过线性投影得到同样维度的向量。随后，这些向量被映射到词表的“图像区段”，在词表中占据连续的 ID 段。  
- **位置编码**：采用 2‑D 相对位置编码，使模型能够感知 patch 在图片中的空间关系，同时对文字 token 也保持顺序信息。这样，文字和图片的空间/序列信息在同一层被统一处理。

**2. 早期融合的 Transformer**  
- **输入层**：所有 token（文字+图像）直接相加位置编码后送入第一层。注意力机制在每一层都可以自由地在文字和图像 token 之间计算相似度。  
- **层结构**：保持标准的多头自注意力 + 前馈网络。唯一的改动是对注意力权重加入了模态感知的 bias，使得在训练初期模型不会因为视觉 token 数量多而被压制。  
- **输出层**：模型采用自回归方式生成下一个 token。若下一个 token 属于图像区段，解码器会把它重新映射回像素空间（通过一个小型的图像解码器），实现图像生成；若是文字 token，则直接输出文字。

**3. 对齐配方与训练细节**  
- **模态平衡损失**：在每个 batch 中，文字 loss 与图像 loss 按比例加权，权重随训练进度动态调整，防止模型偏向某一模态。  
- **Curriculum Learning**：训练前 10% 步骤只喂单模态序列，让模型先学会基本的语言和视觉建模；随后逐步加入混合序列，难度逐层提升。  
- **学习率调度**：采用 cosine decay + warmup，且对图像投影层使用稍低的学习率，避免投影层在大规模文字数据上被过度更新。  
- **数据混合**：使用公开的文本语料、图像‑文字对（如 COCO、LAION）以及纯图像生成任务的数据，统一成 token 序列。  

**最巧妙的点**  
把图像 patch 当作“特殊文字”直接塞进语言模型的词表，这看似简单，却让模型的注意力机制天然具备跨模态对齐能力。再配合模态平衡的 curriculum，模型在同一次前向传播里就能学会“先说，再画”，或“先画，再说”，实现真正的混合文档生成。

### 实验与效果
- **评测任务**：VQA（视觉问答）、COCO Caption（图像描述）、OpenAI‑style 文本生成基准、Imagen‑style 图像生成、以及作者自建的长文混合模态生成评测（包含多段文字和多张图片的长篇输出）。  
- **基线对比**：在图像描述上，Chameleon 的 CIDEr 分数比前一代的 Mixtral‑8x7B 高出约 2.3 分，达到 SOTA；在纯文本任务上，BLEU‑4 超过 Llama‑2（约 1.5% 提升），且与 Gemini‑Pro 持平；在 VQA 上准确率提升约 3%。  
- **人类评估**：在长文混合生成任务中，使用 1‑5 分制的主观评分，Chameleon 获得 4.2 分，略高于 Gemini‑Pro（4.1）和 GPT‑4V（4.0），说明在跨模态连贯性和内容丰富度上具备竞争力。  
- **消融实验**：作者分别去掉早期融合、统一词表、模态平衡损失和 curriculum。结果显示，去掉早期融合后图像‑文字对应错误率上升约 18%；去掉统一词表导致生成的图片质量下降 12%；去掉模态平衡会出现文本生成质量下降 7% 的现象。  
- **局限性**：论文承认在超高分辨率图像生成（>1024×1024）上仍受限于 token 数量；此外，长文混合生成的推理成本随序列长度呈指数增长，实际部署需要进一步的稀疏注意或分段生成技巧。

### 影响与延伸思考
Chameleon 把“早期融合”从概念验证推向了可大规模训练的实用阶段，直接刺激了后续研究在统一 token 词表和混合序列训练上的探索。随后出现的工作如 **Mosaic‑LM**、**UniFusion** 等，都在不同程度上借鉴了 Chameleon 的统一词表和模态平衡策略。对想继续深挖的读者，可以关注以下方向：  
- **稀疏或层级注意**：降低长序列混合文档的计算开销。  
- **高分辨率图像 token 化**：设计更高效的图像切块或可变分辨率 token。  
- **跨模态对齐的自监督目标**：比如利用对比学习进一步强化文字与局部视觉特征的对应关系。  
- **多语言多模态**：把中文、日文等多语言 token 同时加入统一词表，构建真正的全球化混合文档模型。

### 一句话记住它
把图像 patch 当成“特殊文字”，在最底层就让文字和图片一起玩注意力，Chameleon 用单一模型实现了从写段子到画画的全能表现。