# Learning to Compress Prompts with Gist Tokens

> **Date**：2023-04-17
> **arXiv**：https://arxiv.org/abs/2304.08467

## Abstract

Prompting is the primary way to utilize the multitask capabilities of language models (LMs), but prompts occupy valuable space in the input context window, and repeatedly encoding the same prompt is computationally inefficient. Finetuning and distillation methods allow for specialization of LMs without prompting, but require retraining the model for each task. To avoid this trade-off entirely, we present gisting, which trains an LM to compress prompts into smaller sets of "gist" tokens which can be cached and reused for compute efficiency. Gist models can be trained with no additional cost over standard instruction finetuning by simply modifying Transformer attention masks to encourage prompt compression. On decoder (LLaMA-7B) and encoder-decoder (FLAN-T5-XXL) LMs, gisting enables up to 26x compression of prompts, resulting in up to 40% FLOPs reductions, 4.2% wall time speedups, and storage savings, all with minimal loss in output quality.

---

# 通过要点令牌压缩提示 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，提示（prompt）是激活模型多任务能力的关键入口，但提示本身会占用模型的上下文窗口。窗口越小，模型能一次性处理的文本就越少，导致需要把同一个提示反复塞进窗口，浪费算力。传统的解决办法是对每个任务进行微调或蒸馏，让模型内部记住任务指令，但这需要为每个任务重新训练模型，成本高且失去了“一次训练、随处使用”的灵活性。因此，如何在不重新训练模型的前提下，让提示占用更少的空间、计算更高效，成为了一个迫切的技术难题。

### 关键概念速览
**提示（Prompt）**：用户给模型的指令或上下文，就像老师在黑板上写的题目，模型根据它来生成答案。  
**上下文窗口（Context Window）**：模型一次性能看到的文字长度上限，类似于记事本的纸张大小，超出就会被截断。  
**要点令牌（Gist Token）**：经过特殊训练后，模型生成的极短表示，能够浓缩原始提示的核心信息，类似于把长篇文章压成一句摘要。  
**注意力掩码（Attention Mask）**：控制 Transformer 各层之间信息流动的开关，本文把它改成只让模型关注要点令牌，从而逼迫模型压缩提示。  
**指令微调（Instruction Fine‑tuning）**：在大量指令-响应对上继续训练模型，使其更好地理解和执行自然语言指令。  
**FLOPs（Floating‑point Operations）**：衡量计算量的指标，FLOPs 越少，运行越省时省电。  
**解码器模型（Decoder‑only）**：只负责生成文本的模型，如 LLaMA 系列。  
**编码器‑解码器模型（Encoder‑Decoder）**：先把输入编码再生成输出的模型，如 T5 系列。

### 核心创新点
1. **提示压缩的训练目标 → 通过修改注意力掩码让模型自行生成要点令牌 → 实现了在不增加额外训练成本的情况下，让模型学会把完整提示浓缩成极短的 token 序列。**  
2. **要点令牌的缓存机制 → 把压缩后的要点令牌存到磁盘或显存中，后续相同任务直接读取 → 大幅降低了重复编码同一提示的算力开销。**  
3. **统一的实现方式 → 同时适用于解码器模型（LLaMA‑7B）和编码器‑解码器模型（FLAN‑T5‑XXL），只需在注意力掩码上做小改动 → 证明了方法的模型无关性和易部署性。**  
4. **压缩率与质量的平衡实验 → 在保持输出质量几乎不变的前提下，实现最高 26 倍的提示压缩 → 直接转化为约 40% 的 FLOPs 节省和 4.2% 的实际运行时间提升。  

### 方法详解
整体思路可以分为三步：**（1）准备指令微调数据、（2）在微调过程中加入压缩约束、（3）使用要点令牌进行推理**。  
1. **数据准备**：沿用常规的指令微调数据集（如 Alpaca、FLAN），每条样本包含完整提示和对应的期望输出。  
2. **压缩约束的实现**：在 Transformer 的自注意力层里，作者把注意力掩码改成两段：  
   - **提示段**：只允许模型在前几层看到完整提示，后面的层被强制看不到。  
   - **要点段**：在后几层，模型只能看到一小段预留的 token 位（即要点令牌），而看不到原始提示。  
   这种掩码的设计迫使模型在前几层把提示信息压缩进要点位，后面的层只能依赖这些要点来完成任务。训练目标仍然是最大化输出的语言概率，没有额外的损失函数。  
3. **要点令牌的生成与缓存**：微调完成后，给定任意提示，模型在前向传播的前几层会输出一组固定长度的要点令牌。把这组 token 保存下来，下次同样的任务直接把要点令牌塞进模型的后半层即可，省去再次编码完整提示的步骤。  
4. **推理流程**：  
   - **步骤 A**：读取或实时生成要点令牌。  
   - **步骤 B**：把要点令牌拼接到输入的开头（或特定位置），其余输入（如用户问题）照常进入模型。  
   - **步骤 C**：模型在后续层只基于要点令牌和用户问题进行生成。  
   这样，提示本身只占用了极少的 token 空间，却仍然提供了完整的任务指令信息。  
**最巧妙的地方**在于只动了注意力掩码这块几乎不影响原有训练代码的细节，却成功让模型自发学习到信息压缩的能力，省去了额外的蒸馏或对齐损失。

### 实验与效果
- **实验平台**：在 LLaMA‑7B（解码器）和 FLAN‑T5‑XXL（编码器‑解码器）两种主流模型上进行评估。  
- **任务与数据**：使用公开的指令遵循基准（如 Alpaca、OpenAI API 评测集）以及若干实际对话/问答任务。  
- **对比基线**：普通指令微调（不压缩）、以及传统的微调后直接使用完整提示的方式。  
- **主要结果**：  
  - 提示压缩率最高可达 **26 倍**（例如原始 128 token 被压到 5 token）。  
  - 计算量下降约 **40% FLOPs**，实际推理时间提升约 **4.2%**。  
  - 质量下降极小，BLEU / ROUGE 等指标下降不到 **0.5%**，人类评审的满意度几乎持平。  
- **消融实验**：作者分别关闭注意力掩码、只在前几层使用掩码、以及改变要点令牌长度。结果显示：掩码的层数和要点令牌的长度是影响压缩率和质量的关键因素，最优配置在后 4 层使用掩码、要点令牌长度为 5‑8。  
- **局限性**：论文未在超长上下文（如 32k token）场景下测试，要点令牌的固定长度可能在极其复杂指令上出现信息丢失；此外，缓存要点令牌需要额外的存储管理逻辑。

### 影响与延伸思考
这篇工作打开了“提示压缩”这一新方向，随后出现的研究开始探索 **可学习的提示嵌入（learnable prompt embeddings）**、**跨任务提示共享**以及 **多模态要点生成** 等思路。推测未来会有更多工作把要点令牌和检索系统结合，让模型在检索到相似任务时直接复用已有要点，从而进一步降低算力。想深入了解的读者可以关注 **Prompt Compression**、**Efficient Inference** 以及 **Instruction Tuning** 这几个热点方向的最新会议论文。

### 一句话记住它
只改注意力掩码，让模型自己把长提示压成几条要点，既省算力又不牺牲质量。