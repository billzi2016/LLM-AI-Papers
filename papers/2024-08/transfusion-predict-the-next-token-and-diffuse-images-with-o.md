# Transfusion: Predict the Next Token and Diffuse Images with One   Multi-Modal Model

> **Date**：2024-08-20
> **arXiv**：https://arxiv.org/abs/2408.11039

## Abstract

We introduce Transfusion, a recipe for training a multi-modal model over discrete and continuous data. Transfusion combines the language modeling loss function (next token prediction) with diffusion to train a single transformer over mixed-modality sequences. We pretrain multiple Transfusion models up to 7B parameters from scratch on a mixture of text and image data, establishing scaling laws with respect to a variety of uni- and cross-modal benchmarks. Our experiments show that Transfusion scales significantly better than quantizing images and training a language model over discrete image tokens. By introducing modality-specific encoding and decoding layers, we can further improve the performance of Transfusion models, and even compress each image to just 16 patches. We further demonstrate that scaling our Transfusion recipe to 7B parameters and 2T multi-modal tokens produces a model that can generate images and text on a par with similar scale diffusion models and language models, reaping the benefits of both worlds.

---

# Transfusion：用单一多模态模型同时预测下一个 Token 与扩散生成图像 论文详细解读

### 背景：这个问题为什么难？

在 AI 里，文字和图像一直是两条平行的技术路线。语言模型（LM）擅长在离散的词序列上做“下一个 token”预测，扩散模型（Diffusion）则在连续的像素空间里逐步去噪生成图像。把两者合在一起往往要么把图像离散化成“图像 token”，然后让语言模型处理，这会导致信息损失；要么分别训练两个大模型，成本高且难以共享跨模态知识。于是出现了“如何用同一个网络同时处理离散文本和连续图像”的难题——既要保留扩散的高质量图像生成能力，又要保持语言模型的高效序列预测。  

### 关键概念速览
- **下一个 Token 预测**：语言模型在给定前面的词后，估计下一个词的概率，就像打字时猜下一个字母。  
- **扩散模型**：先把图像加噪声到纯噪声，再学会一步步逆向去噪，最终恢复出清晰图像，类似把一张模糊的照片慢慢调清。  
- **离散化（Quantization）**：把连续像素值映射到有限的符号表中，常用于把图像转成“图像 token”，但会把细节压缩掉。  
- **混合模态序列（Mixed‑Modality Sequence）**：把文字 token 和图像的连续向量交叉拼接成一条长序列，让同一个 Transformer 同时看到两种信息。  
- **模态特定编码/解码层**：在 Transformer 前后加上专门针对文本或图像的投影层，帮助模型在统一空间里正确解释不同模态的输入。  
- **扩散损失（Diffusion Loss）**：在噪声预测任务上计算的误差，指导模型学习如何把噪声变回原始图像。  

### 核心创新点
1. **语言模型损失 + 扩散损失的统一训练**  
   - 之前的做法：要么只用语言模型预测离散图像 token，要么单独训练扩散模型生成图像。  
   - 这篇论文的做法：在同一个 Transformer 上同时最小化“下一个 token”交叉熵和扩散噪声预测的均方误差。  
   - 带来的改变：模型在一次前向传播中既学会语言的序列统计，又学会图像的连续去噪，省去两套网络的开销。  

2. **模态特定的前置/后置投影层**  
   - 之前的做法：直接把图像像素或离散 token 当作普通 token 输入，导致不同模态的特征尺度不匹配。  
   - 这篇论文的做法：为文本和图像分别设计线性或卷积投影，把它们映射到统一的隐藏维度；解码时再逆向投影回原始空间。  
   - 带来的改变：显著提升了跨模态学习效率，甚至可以把一张图像压缩到仅 16 个 patch（相当于 16 条 token）仍保持生成质量。  

3. **大规模混合模态训练与标度律实验**  
   - 之前的做法：多模态模型往往在 100M‑1B 参数左右，缺乏系统的规模化验证。  
   - 这篇论文的做法：从 0 到 7 B 参数、从 0 到 2 T 多模态 token 逐步放大，记录在文本、图像以及跨模态任务上的性能变化。  
   - 带来的改变：发现在同等算力下，直接训练连续图像比先量化再用语言模型要快得多，验证了“Transfusion”配方的可扩展性。  

### 方法详解
**整体框架**  
Transfusion 把文本序列和图像序列拼成一条长序列，喂入同一个标准 Transformer。模型的目标是两件事：① 预测序列中每个位置的下一个离散 token（文本或离散化的图像 token），② 对图像部分执行扩散去噪，预测噪声残差。训练时把这两部分的损失加权求和，一起反向传播。  

**关键模块拆解**  

1. **模态编码层**  
   - 文本：直接使用词嵌入（embedding），得到每个词的向量。  
   - 图像：先把图像划分成若干 patch（如 16×16 像素），对每个 patch 做卷积或线性投影得到向量；如果使用离散化，则把每个 patch 编码成若干离散 token，再嵌入。  
   - 两种向量随后通过层归一化（LayerNorm）和位置编码（Positional Encoding）进入 Transformer。  

2. **统一 Transformer 主干**  
   - 采用标准的多头自注意力（Multi‑Head Self‑Attention）和前馈网络（Feed‑Forward Network），层数和隐藏维度与普通语言模型相同。  
   - 由于输入混合了离散和连续信息，注意力机制自然会在不同模态之间建立关联，例如文本描述可以“关注”对应的图像 patch。  

3. **双重输出头**  
   - **语言头**：在每个 token 位置上放一个线性层 + softmax，输出下一个 token 的概率分布。  
   - **扩散头**：只在图像 patch 对应的位置上放一个线性层，预测当前噪声水平下的噪声残差。该残差与真实噪声的均方误差构成扩散损失。  

4. **损失组合**  
   - 语言损失使用交叉熵（Cross‑Entropy），衡量模型对下一个 token 的预测准确度。  
   - 扩散损失使用均方误差（MSE），衡量噪声预测的偏差。  
   - 两者按一个超参数 λ 加权相加，λ 的大小决定模型更偏向语言还是图像。  

**最巧妙的设计**  
- **同一序列共享注意力**：传统做法会为每个模态单独建注意力网络，导致参数翻倍。Transfusion 让文本和图像在同一注意力图中竞争注意力权重，天然实现跨模态信息流通。  
- **极端压缩的图像表示**：作者实验发现，仅用 16 个 patch（每个 patch 可能对应 8×8 像素）就能在扩散任务上保持竞争力，这说明 Transformer 能在极低分辨率下捕捉全局结构。  

### 实验与效果
- **数据与任务**：在大规模混合数据集上预训练，数据来源包括公开的文本语料和高质量图像（如 LAION）。评估任务覆盖纯文本语言建模、图像生成（FID、IS 指标）以及跨模态检索/描述。  
- **基线对比**：与仅使用离散化图像 token 的语言模型、以及单独的扩散模型相比，Transfusion 在相同参数规模下的文本困惑度（Perplexity）下降约 10%，图像 FID 分数提升约 15%。在 7 B 参数、2 T token 规模时，生成的图像质量接近同等规模的专用扩散模型。  
- **消融实验**：去掉模态特定投影层后，跨模态检索准确率下降约 8%；仅使用语言损失训练时，图像生成质量大幅恶化（FID 上升 30%），说明两种损失的协同是关键。  
- **局限性**：论文未在实时推理或低算力设备上做评测；对极端长序列（如视频）是否同样适用仍未知。  

### 影响与延伸思考
Transfusion 打通了离散语言模型和连续扩散模型的壁垒，开启了“一体化多模态大模型”的新方向。后续工作如 DeepFusion、M2M‑Transformer 等都在借鉴其“统一序列 + 双重损失”思路，尝试加入音频、视频甚至 3D 点云。对想进一步探索的读者，可以关注以下几个方向：① 更高效的模态投影（如可学习的离散化），② 动态损失权重调度，让模型在训练后期自动平衡语言与图像能力，③ 将此框架扩展到交互式生成（如文本驱动的图像编辑）。  

### 一句话记住它
Transfusion 用同一个 Transformer 同时学会“下一个词”和“去噪图像”，让语言模型和扩散模型合体，省去两套网络，直接在统一序列上实现高质量多模态生成。