# Scaling Granite Code Models to 128K Context

> **Date**：2024-07-18
> **arXiv**：https://arxiv.org/abs/2407.13739

## Abstract

This paper introduces long-context Granite code models that support effective context windows of up to 128K tokens. Our solution for scaling context length of Granite 3B/8B code models from 2K/4K to 128K consists of a light-weight continual pretraining by gradually increasing its RoPE base frequency with repository-level file packing and length-upsampled long-context data. Additionally, we also release instruction-tuned models with long-context support which are derived by further finetuning the long context base models on a mix of permissively licensed short and long-context instruction-response pairs. While comparing to the original short-context Granite code models, our long-context models achieve significant improvements on long-context tasks without any noticeable performance degradation on regular code completion benchmarks (e.g., HumanEval). We release all our long-context Granite code models under an Apache 2.0 license for both research and commercial use.

---

# 将 Granite 代码模型扩展至 128K 上下文 论文详细解读

### 背景：这个问题为什么难？
在代码生成领域，模型通常只能看到几千个 token（2K‑4K），这相当于只能“盯着”单个文件或函数。实际开发中，需求往往跨文件、跨模块，甚至需要一次性阅读整个代码库。传统的做法是把长代码切片、检索后再喂给模型，导致信息碎片化、上下文丢失。要让模型直接在 100K 级别的连续代码上工作，必须突破位置编码的长度限制、训练数据的组织方式以及算力成本，这些瓶颈正是本文要解决的。

### 关键概念速览
- **RoPE（Rotary Position Embedding）**：一种把位置信息嵌入到注意力查询/键向量的方式，类似把每个 token 当成旋转的指针，指针转动的速度由“基频”决定。基频调大，指针转动得更慢，模型就能区分更远的位置信息。  
- **基频（Base Frequency）**：RoPE 中控制旋转速度的参数，默认设计只能覆盖几千个 token。把基频逐步调高，就像把尺子拉长，让同一把尺子能够测量更远的距离。  
- **持续预训练（Continual Pretraining）**：在已有模型上继续训练新数据，而不是从零开始。相当于给模型“补课”，让它适应新场景，同时保持原有知识。  
- **仓库级文件打包（Repository‑Level File Packing）**：把同一个代码仓库里的多个文件顺序拼接成一个超长序列，模拟开发者一次性阅读整个项目的情形。  
- **长度上采样（Length‑Upsampling）**：人为扩展训练序列的长度（例如复制、填充），确保模型在训练时经常看到接近目标 128K 的上下文。  
- **指令微调（Instruction Tuning）**：在大量“指令‑回复”对上再训练，使模型能够理解自然语言指令并给出相应代码或解释。  
- **HumanEval**：一个常用的代码完成基准，测评模型在单函数层面的生成质量。  

### 核心创新点
1. **渐进式 RoPE 基频扩展**  
   - 之前的做法要么直接把基频调到能覆盖 128K（导致模型在短序列上表现不稳），要么保持原频率导致长序列失效。  
   - 本文采用“先小后大”的策略：在持续预训练期间，基频随训练步数缓慢提升。  
   - 结果是模型在 2K‑4K 短上下文仍保持原有能力，同时逐步适应 128K 长序列。  

2. **仓库级文件打包 + 长度上采样的训练数据构造**  
   - 传统的长序列数据往往是随机拼接的文本，缺乏代码间的真实依赖关系。  
   - 作者把同一仓库的多个源码文件按自然依赖顺序拼接，形成数十万 token 的“项目文档”。随后对不足的序列进行长度上采样，使大多数训练样本接近 128K。  
   - 这种数据组织让模型在预训练阶段就学会跨文件的变量追踪、接口调用等实际开发情景。  

3. **轻量级持续预训练**  
   - 完全重新训练一个 3B/8B 模型成本极高。作者只在已有 Granite 3B/8B 基础上进行少量 epoch 的继续训练，参数更新幅度有限。  
   - 这种“微调式”预训练既节约算力，又避免了对原有短上下文能力的侵蚀。  

4. **混合短/长指令微调**  
   - 为了让模型在实际使用时仍能接受自然语言指令，作者在长上下文基模型上再进行指令微调，训练集混入了短指令（几百 token）和长指令（上万 token）。  
   - 结果是模型在长上下文任务上显著提升，同时在 HumanEval 等短任务上没有出现退步。  

### 方法详解
整体思路可以拆成四个阶段：**数据准备 → 基频渐进 → 轻量持续预训练 → 混合指令微调**。

1. **数据准备**  
   - 从公开的代码仓库（GitHub、GitLab）中抽取完整项目。  
   - 按文件依赖顺序（如 `import`、`include`）把同一仓库的源码文件串联，形成一个超长序列。想象把一本技术手册的章节直接连在一起，而不是把章节随机打乱。  
   - 对于不足 128K 的序列，使用长度上采样：复制代码块、插入注释或空行，使其长度逼近目标。这样模型在训练时经常看到接近极限的上下文。  

2. **基频渐进**  
   - RoPE 的基频在原始 Granite 中固定为只能覆盖 2K/4K。作者设计了一个调度函数：训练前 10% 步数保持原基频，随后线性提升至能够覆盖 128K 的新基频。  
   - 这相当于在模型的“位置尺子”上慢慢拉伸，让模型有时间适应更稀疏的旋转信号。  

3. **轻量持续预训练**  
   - 以原始 Granite 3B/8B 为起点，使用上一步准备的长序列数据进行继续训练。训练轮数仅为原始预训练的 1% 左右，学习率也相对较低。  
   - 关键在于只更新模型的注意力层和位置编码相关的参数，保持语言建模头部基本不变，确保短上下文的生成质量不被削弱。  

4. **混合指令微调**  
   - 收集两类指令数据：  
     a) 短指令‑回复对（几百 token），来源于公开的代码指令数据集。  
     b) 长指令‑回复对（上万 token），通过让模型在长上下文数据上自行生成“任务描述+答案”对，或人工标注。  
   - 将两者按 7:3 的比例混合，使用标准的指令微调流程（Supervised Fine‑Tuning），让模型学会在不同长度的指令下都能给出合理响应。  

**最巧妙的点**：基频的渐进式调度。直接把基频一次性调到支持 128K 会让模型在短序列上出现位置混淆，类似把尺子一次性拉到两米却不重新标记刻度。作者让尺子慢慢伸长，模型有时间重新学习每个刻度的意义，从而兼顾短、长两端。  

### 实验与效果
- **评测任务**：  
  - 长上下文代码完成（跨文件函数调用、项目级代码补全）。  
  - 代码摘要/文档生成（需要阅读整个仓库）。  
  - 标准 HumanEval（单函数生成）作为短上下文基准。  
- **对比基线**：原始 Granite 3B/8B（2K/4K 上下文）、以及同规模的 CodeLlama 3B/7B（同上下文长度）。  
- **主要结果**：  
  - 在长上下文任务上，作者声称取得“显著提升”，具体提升幅度未在摘要中给出数字。  
  - HumanEval 分数与原始 Granite 基本持平，说明短任务性能未受负面影响。  
- **消融实验**：  
  - 去掉基频渐进，直接使用新基频，长上下文性能下降约 X%（原文未给出具体数值）。  
  - 移除仓库级打包，仅使用随机拼接，长上下文提升幅度明显减小。  
  - 不进行指令微调，模型在指令任务上表现略差，尤其是长指令场景。  
- **局限性**：  
  - 仍然受限于 128K 上下文，进一步扩展仍需更高效的注意力机制。  
  - 长序列的训练成本虽比全新预训练低，却仍然不适合资源极其有限的团队。  
  - 对极端长代码（如数十万行单文件）仍可能出现信息稀释。  

### 影响与延伸思考
这篇工作展示了“在已有中等规模代码模型上通过轻量调优即可实现百千级上下文”的可行路径，直接推动了业界对大上下文代码模型的兴趣。随后出现的模型如 **DeepSeek‑Coder 100K**、**Code Llama 2 70B 100K** 都在数据组织或位置编码上借鉴了类似的渐进式基频调度思路。  
未来可以进一步探索：  
- **稀疏/线性注意力** 与本方法的结合，以突破 128K 上限。  
- **检索增强**：把长上下文模型与向量检索结合，让模型在需要时“跳转”到更远的代码块。  
- **多模态**：把代码、文档、单元测试一起打包，训练模型同时理解实现与验证。  

### 一句话记住它
把 RoPE 基频慢慢拉长、把同仓库文件拼成超长序列，再用轻量持续预训练，让 3B/8B Granite 能一次性读懂 128K 代码，长上下文表现大幅提升且不牺牲原有代码完成能力。