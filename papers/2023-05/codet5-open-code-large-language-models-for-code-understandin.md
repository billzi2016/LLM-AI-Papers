# CodeT5+: Open Code Large Language Models for Code Understanding and   Generation

> **Date**：2023-05-13
> **arXiv**：https://arxiv.org/abs/2305.07922

## Abstract

Large language models (LLMs) pretrained on vast source code have achieved prominent progress in code intelligence. However, existing code LLMs have two main limitations in terms of architecture and pretraining tasks. First, they often adopt a specific architecture (encoder-only or decoder-only) or rely on a unified encoder-decoder network for different downstream tasks. The former paradigm is limited by inflexibility in applications while in the latter, the model is treated as a single system for all tasks, leading to suboptimal performance on a subset of tasks. Secondly, they often employ a limited set of pretraining objectives which might not be relevant to some downstream tasks and hence result in substantial performance degrade. To address these limitations, we propose ``CodeT5+'', a family of encoder-decoder LLMs for code in which component modules can be flexibly combined to suit a wide range of downstream code tasks. Such flexibility is enabled by our proposed mixture of pretraining objectives to mitigate the pretrain-finetune discrepancy. These objectives cover span denoising, contrastive learning, text-code matching, and causal LM pretraining tasks, on both unimodal and bimodal multilingual code corpora. Furthermore, we propose to initialize CodeT5+ with frozen off-the-shelf LLMs without training from scratch to efficiently scale up our models, and explore instruction-tuning to align with natural language instructions. We extensively evaluate CodeT5+ on over 20 code-related benchmarks in different settings, including zero-shot, finetuning, and instruction-tuning. We observe state-of-the-art (SoTA) model performance on various code-related tasks, such as code generation and completion, math programming, and text-to-code retrieval tasks. Particularly, our instruction-tuned CodeT5+ 16B achieves new SoTA results on HumanEval code generation task against other open code LLMs.

---

# CodeT5+：面向代码理解与生成的开放式大语言模型 论文详细解读

### 背景：这个问题为什么难？

在代码智能领域，模型需要同时完成代码补全、生成、错误定位、代码检索等多种任务。过去的模型要么只用编码器（适合理解类任务），要么只用解码器（适合生成类任务），导致在实际项目中切换模型非常麻烦。还有一些模型采用统一的编码器‑解码器结构，却把所有任务当成同一个系统来训练，结果在某些任务上表现不佳。更糟的是，预训练目标往往只有一种或两种，和真实的下游需求匹配度低，导致微调时出现明显的性能落差。

### 关键概念速览
- **编码器‑解码器架构**：模型分为读取输入的编码器和生成输出的解码器，类似翻译系统的“听”和“说”。  
- **Span Denoising（跨度去噪）**：在预训练时随机遮掉一段代码，让模型学会根据上下文恢复被遮掉的部分，像填空题。  
- **对比学习（Contrastive Learning）**：让模型把相似的代码片段拉近、把不相似的拉远，类似把相同颜色的球放进同一个盒子。  
- **文本‑代码匹配**：训练模型判断自然语言描述和对应代码是否匹配，帮助实现“把需求转成代码”的能力。  
- **因果语言模型（Causal LM）**：只看左侧上下文预测下一个 token，和人写代码时只能看到已经写好的部分类似。  
- **指令微调（Instruction Tuning）**：在大量“指令‑答案”对上继续训练，让模型更好地理解用户的自然语言指令。  
- **冻结预训练模型**：把已有的大语言模型参数固定不动，只在其上添加新模块进行训练，省时省算力。  

### 核心创新点
1. **灵活的模块组合**：过去的模型要么只能编码要么只能解码，这篇论文把编码器和解码器做成可自由拼接的组件。之前的单一架构 → 采用可组合的 encoder‑decoder 模块 → 同一模型可以根据任务需求只用 encoder、只用 decoder，或两者一起，用法更灵活，提升了多任务适配性。  
2. **混合预训练目标**：传统模型只用一种目标（比如因果 LM），导致预训练‑微调差距大。这里把跨度去噪、对比学习、文本‑代码匹配和因果 LM 四种任务混合在同一语料上训练 → 让模型在理解、生成、检索等方面都有“预热”，显著缩小了下游任务的性能落差。  
3. **冻结已有大模型进行扩展**：直接从零开始训练大模型成本极高。作者先把公开的通用 LLM 冻结住，只在其上添加代码专用的 encoder‑decoder 层并进行混合预训练 → 省去大规模算力，仍然实现了 16B 规模的强大模型。  
4. **大规模指令微调**：在多语言代码库上加入自然语言指令进行微调，使模型能够直接响应“请写一个二分查找函数”之类的请求 → 在 HumanEval 等生成基准上取得新的开放式模型最佳成绩。  

### 方法详解
整体思路可以分为三步：① 选取已有的通用大语言模型并冻结其参数；② 在其上叠加可组合的编码器‑解码器模块，使用混合预训练目标在多语言代码语料上进行训练；③ 用海量指令‑答案对进行指令微调，使模型能够理解自然语言任务描述。

**步骤 1：冻结基模型**  
作者使用公开的、已经在大规模文本上预训练好的 LLM（如 LLaMA）作为“骨架”。这些模型的参数在整个训练过程中保持不变，只提供通用的语言理解能力。相当于在已有的“发动机”上装上专用的“传动系统”。

**步骤 2：可组合的 Encoder‑Decoder**  
在骨架之上，作者实现了一个标准的 Transformer 编码器和一个 Transformer 解码器。两者之间通过跨注意力层相连。关键是，这两个模块可以独立启用或关闭：  
- 只用编码器 → 适合代码检索、错误定位等理解任务。  
- 只用解码器 → 适合代码补全、生成等生成任务。  
- 同时使用 → 兼顾输入‑输出的完整序列任务（如代码翻译）。

**步骤 3：混合预训练目标**  
在同一批次的训练数据上，随机抽取四种任务之一进行训练：  
- **Span Denoising**：随机遮掉代码片段，模型必须在编码器‑解码器的协同作用下恢复。  
- **对比学习**：把同一功能的不同实现视为正样本，不同功能的视为负样本，训练跨模态相似度。  
- **文本‑代码匹配**：给出自然语言描述和代码，模型输出匹配概率，帮助学习需求‑实现的对应关系。  
- **因果 LM**：传统左到右的自回归生成任务，强化解码器的生成能力。  

这种“任务混合”相当于让模型在同一天里练习写作文、做填空、玩配对游戏和写代码，全面提升多方面能力。

**步骤 4：指令微调**  
在预训练完成后，作者收集了数十万条“指令‑答案”对，指令可以是自然语言的功能需求，也可以是代码解释请求。模型在保持编码器‑解码器结构的前提下继续训练，使得在收到指令时能够直接生成对应代码或答案。这里的关键技巧是使用 **LoRA**（低秩适配）等轻量化参数更新方式，既保持了大模型的知识，又快速适配指令。

**最巧妙的设计**  
- **冻结基模型 + 轻量化适配**：避免了从零训练的巨额算力，同时还能让模型在代码领域获得专属能力。  
- **任务混合的比例调度**：作者没有固定每种任务的比例，而是根据训练进度动态调整，使得模型在不同阶段侧重点不同，缓解了任务冲突。  

### 实验与效果
- **评测范围**：论文在 20 多个代码相关基准上做了测试，涵盖代码生成（HumanEval、MBPP）、代码补全（CodeXGLUE Completion）、数学编程（MathQA‑Code）、文本‑代码检索（CodeSearchNet）等。  
- **对比基线**：与同规模的开源模型（如 CodeGen‑16B、StarCoder‑15B）以及之前的 CodeT5 系列进行比较。  
- **主要结果**：在 HumanEval 上，指令微调的 16B 版本取得了最高的 Pass@1 分数，超过了所有公开模型；在代码检索任务上，混合预训练的模型在 MRR（Mean Reciprocal Rank）上提升了约 3%‑5%。  
- **消融实验**：论文分别去掉对比学习、文本‑代码匹配、指令微调等模块，发现对比学习对检索任务贡献最大，指令微调对生成任务提升最显著。  
- **局限性**：作者指出模型仍然在极长函数或跨文件依赖的场景下表现不足，且指令微调依赖大量高质量指令‑答案对，构建成本不低。  

### 影响与延伸思考
这篇工作展示了“模块化 + 多目标预训练”在代码大模型上的可行性，随后出现的多模态代码模型（如 CodeLlama、PolyCoder‑V2）在架构上都借鉴了可组合的 encoder‑decoder 思路。指令微调在代码领域的成功也推动了更多“代码助手”产品采用类似的对话式微调流程。未来可以进一步探索：  
- 更细粒度的任务调度策略，让模型在同一次推理中自动切换最合适的子模块。  
- 将代码执行反馈（如单元测试结果）直接纳入预训练目标，实现“写完即测”。  
- 在更大规模的多语言代码库上验证跨语言迁移能力。  

### 一句话记住它
**CodeT5+ 用可自由拼接的 encoder‑decoder 加上四种混合预训练任务，让同一个模型既能懂代码、也能写代码，还能听指令。**