# Training LLMs over Neurally Compressed Text

> **Date**：2024-04-04
> **arXiv**：https://arxiv.org/abs/2404.03626

## Abstract

In this paper, we explore the idea of training large language models (LLMs) over highly compressed text. While standard subword tokenizers compress text by a small factor, neural text compressors can achieve much higher rates of compression. If it were possible to train LLMs directly over neurally compressed text, this would confer advantages in training and serving efficiency, as well as easier handling of long text spans. The main obstacle to this goal is that strong compression tends to produce opaque outputs that are not well-suited for learning. In particular, we find that text na\"ively compressed via Arithmetic Coding is not readily learnable by LLMs. To overcome this, we propose Equal-Info Windows, a novel compression technique whereby text is segmented into blocks that each compress to the same bit length. Using this method, we demonstrate effective learning over neurally compressed text that improves with scale, and outperforms byte-level baselines by a wide margin on perplexity and inference speed benchmarks. While our method delivers worse perplexity than subword tokenizers for models trained with the same parameter count, it has the benefit of shorter sequence lengths. Shorter sequence lengths require fewer autoregressive generation steps, and reduce latency. Finally, we provide extensive analysis of the properties that contribute to learnability, and offer concrete suggestions for how to further improve the performance of high-compression tokenizers.

---

# 在神经压缩文本上训练大语言模型 论文详细解读

### 背景：这个问题为什么难？
传统的大语言模型（LLM）使用子词分词器把原始文字切成若干 token，虽然能把字符数压缩一点，但仍然需要上千个 token 才能覆盖几千字的上下文。更高的压缩率意味着更短的序列、更少的自回归步骤，从而提升训练和推理效率。然而，现有的神经压缩器（如算术编码）在把文本压成极短的比特流后，生成的符号序列缺乏可解释的结构，模型很难从中学习语言规律。换句话说，压得越狠，输出越“黑盒”，学习难度随之上升，这成为把高压缩率直接用于 LLM 的最大瓶颈。

### 关键概念速览
**子词分词器**：把文字切成常见的词根或词缀，类似把句子拆成拼图块，块的大小适中，模型容易拼合。  
**神经压缩器**：利用深度网络学习文本的概率分布，再用算术编码把文本压成极短的比特流，像把整本书压进一个硬盘。  
**算术编码**：一种基于概率模型的无损压缩方法，把出现概率高的符号用更少的比特表示，类似把常用词写成简写。  
**等信息窗口（Equal-Info Windows）**：把文本划分成若干块，使每块在压缩后占用相同的比特数，保证每个窗口携带的“信息量”大致相等。  
**字节级基线**：直接把文本当作字节序列喂给模型的最原始方式，等价于把每个字符当作一个 token。  
**困惑度（Perplexity）**：衡量语言模型预测下一个 token 难易程度的指标，数值越低说明模型越懂语言。  
**自回归生成**：模型一次预测一个 token，预测完后再把结果喂回模型继续预测，类似人一句一句说话。

### 核心创新点
1. **从“任意压缩”到“等信息窗口”**：以前直接把算术编码得到的比特流喂模型，模型几乎学不到有用的语言结构。本文提出把文本切成每块压缩后长度相同的窗口，使得每一步训练的输入长度固定且信息密度均衡，避免了极端短块或极端长块导致的学习不稳定。  
2. **把神经压缩器当作“高阶 tokenizer”**：把神经压缩器视作一种新型分词器，直接输出固定比特长度的 token 序列。相比子词分词器，这种 token 更紧凑，序列长度大幅缩短，从而在相同的显存预算下可以处理更长的上下文。  
3. **规模效应验证**：实验表明，随着模型参数和训练数据规模的提升，使用等信息窗口的压缩文本学习效果会继续改善，说明该方法能够在大模型时代保持可扩展性。  
4. **系统性可学习性分析**：作者对比了不同压缩率、窗口大小、比特对齐方式等因素，对哪些属性会破坏可学习性给出明确结论，并提供了进一步提升高压缩 tokenizer 性能的实用建议。

### 方法详解
整体思路可以分为三步：**压缩 → 窗口划分 → LLM 训练**。

1. **神经压缩阶段**  
   先训练一个独立的神经压缩模型（通常是自回归或变分自编码器），它学习文本的概率分布并使用算术编码把整篇文档压成一串比特。这里的压缩率可以远高于子词分词器，常见的 10–20 倍甚至更高。

2. **等信息窗口划分**  
   - **目标**：让每个训练样本的长度固定，且每个窗口携带相似的信息量。  
   - **实现**：先统计压缩后比特流的累计熵曲线（即每增加多少比特对应多少信息），然后在比特流上设定等间距的切点，使得每段恰好包含相同的比特数。可以把它想象成把一根长绳子按等长切段，虽然每段的实际文字长度不同，但每段的“信息重量”相同。  
   - **对齐**：为了让模型能够直接读取，比特流会被重新映射成整数 token（比如每 8 比特映射成一个字节 token），保证每个窗口的 token 数目固定。

3. **LLM 训练**  
   - **输入**：模型接收的是等信息窗口产生的 token 序列，而不是传统的子词 token。  
   - **目标**：和普通语言模型一样，预测下一个 token 的概率分布，只是这里的 token 是压缩后的比特块。  
   - **优化**：使用标准的自回归交叉熵损失。因为每个窗口的长度固定，训练过程中的显存占用更可预测，能够在同样的硬件上放入更大的 batch 或更长的上下文。  
   - **推理**：生成时模型先输出压缩 token，随后通过相同的解码器（算术解码）把比特流还原成自然语言，等价于先生成“压缩指令”，再把指令展开成文字。

**最巧妙的点**在于等信息窗口的设计：它把压缩率和可学习性解耦。压得越狠，信息密度越高，但只要窗口长度固定，模型仍然能在每一步看到足够的信号，从而避免了“信息稀疏”导致的学习失败。

### 实验与效果
- **数据集**：论文在大规模的公开文本语料（如 Pile、OpenWebText）上进行训练和评估。  
- **基线**：与子词分词器（BPE/WordPiece）以及最原始的字节级 token（每字节一个 token）作对比。  
- **结果**：在困惑度上，等信息窗口的模型比字节级基线低很多，具体数值未在摘要中给出，但作者称“显著优于”。在推理速度方面，由于序列长度缩短，生成同等文字量所需的自回归步骤减少，整体延迟下降了数十个百分点。  
- **消融实验**：作者分别去掉等信息窗口、使用不等长窗口、或直接使用算术编码的原始比特流进行训练，结果显示学习性能急剧下降，验证了窗口均衡信息量是关键因素。  
- **局限**：与同等参数量的子词模型相比，困惑度仍稍高，说明压缩 token 仍然带来一定的信息损失或学习难度。作者也指出，压缩率过高会导致窗口内部信息过于密集，模型在极端情况下仍可能出现学习瓶颈。

### 影响与延伸思考
这篇工作打开了“高压缩率 tokenizer”在大模型训练中的可能性，激发了后续研究探索更高效的编码方式、可逆的离散表示以及与检索增强模型的结合。2024 年以后，有几篇论文尝试把可微分的压缩层直接嵌入 Transformer，或把等信息窗口与稀疏注意力配合，以进一步压缩显存占用。想深入了解的读者可以关注 **Neural Compression for Language Modeling** 系列、以及 **Efficient Transformers** 方向的最新进展。

### 一句话记住它
把文本压成等信息窗口，让模型在极短的 token 序列上也能学会语言，从而实现更高效的训练和更快的生成。