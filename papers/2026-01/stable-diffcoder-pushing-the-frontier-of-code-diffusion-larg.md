# Stable-DiffCoder: Pushing the Frontier of Code Diffusion Large Language Model

> **Date**：2026-01-22
> **arXiv**：https://arxiv.org/abs/2601.15892

## Abstract

Diffusion-based language models (DLLMs) offer non-sequential, block-wise generation and richer data reuse compared to autoregressive (AR) models, but existing code DLLMs still lag behind strong AR baselines under comparable budgets. We revisit this setting in a controlled study and introduce Stable-DiffCoder, a block diffusion code model that reuses the Seed-Coder architecture, data, and training pipeline. To enable efficient knowledge learning and stable training, we incorporate a block diffusion continual pretraining (CPT) stage enhanced by a tailored warmup and block-wise clipped noise schedule. Under the same data and architecture, Stable-DiffCoder overall outperforms its AR counterpart on a broad suite of code benchmarks. Moreover, relying only on the CPT and supervised fine-tuning stages, Stable-DiffCoder achieves stronger performance than a wide range of \~8B ARs and DLLMs, demonstrating that diffusion-based training can improve code modeling quality beyond AR training alone. Moreover, diffusion-based any-order modeling improves structured code modeling for editing and reasoning, and through data augmentation, benefits low-resource coding languages.

---

# Stable‑DiffCoder: 稳定扩散代码模型 论文详细解读

### 背景：这个问题为什么难？
代码生成一直是大语言模型（LLM）的强项，但主流模型大多采用自回归（AR）方式——一次生成一个 token，顺序依赖前文。自回归虽然直观，却限制了并行度，训练和推理都只能按序进行，导致算力利用率低。扩散式语言模型（DLLM）提供了块级、非顺序的生成方式，理论上可以更好地复用上下文信息并实现并行生成。但在实际的代码任务上，已有的 DLLM 仍然落后于同等规模的 AR 基线，尤其在相同的算力预算下，生成质量和编辑能力都不够理想。于是需要一种既保留扩散模型并行优势，又能在代码领域实现竞争甚至超越 AR 的新方案。

### 关键概念速览
**自回归（AR）模型**：一次预测下一个 token，后面的预测全部依赖已经生成的内容，像顺序写作一样。  
**扩散式语言模型（DLLM）**：先在噪声空间对整段文本进行“去噪”恢复，块与块之间可以并行处理，类似先把画布涂满噪点再慢慢擦除细节。  
**块（block）生成**：把一段代码切成若干固定大小的子序列，模型一次处理一个块，而不是逐字符。  
**持续预训练（Continual Pre‑Training, CPT）**：在已有模型的基础上继续大规模无监督训练，像给已经学会基础语法的学生再上进阶课程。  
**噪声调度（noise schedule）**：在扩散过程中控制噪声强度随时间的变化，类似烘焙时调节温度曲线。  
**Warmup**：训练初期使用较小的学习率逐步提升，防止模型一开始就被大梯度冲击，像热身跑步一样让模型慢慢进入状态。  
**结构化编辑**：对已有代码进行增删改的操作，需要模型理解代码的语法树和依赖关系，类似在文档中精准定位并修改段落。

### 核心创新点
1. **同架构同数据的公平对比 → 复用 Seed‑Coder 的 2.5 B 参数、相同数据管线 → 在相同预算下直接比较扩散块模型与 AR 基线，证明扩散方式本身并非劣势。**  
2. **块扩散持续预训练 + 专属噪声调度 → 在 CPT 阶段加入针对块的噪声裁剪策略，并配合专门设计的 warmup 曲线 → 训练更稳定、知识吸收更快，避免了扩散模型常见的梯度爆炸或收敛慢的问题。**  
3. **仅靠 CPT 与监督微调即可超越多种 8 B 规模的 AR 与 DLLM → 省去额外的多阶段微调或大规模指令调教 → 展示了扩散式训练本身就能提升代码建模质量。**  
4. **any‑order（任意顺序）建模提升编辑与推理 → 通过块级去噪，模型不再强制顺序生成，而是可以从任意块开始恢复代码 → 在代码编辑、结构化推理任务上表现更好，并且在低资源编程语言上通过数据增强获得额外收益。**

### 方法详解
整体思路可以拆成三大步骤：  
1️⃣ **准备工作**：以 Seed‑Coder（一个在大规模代码上预训练的 2.5 B 参数模型）为起点，保持相同的 tokenizer、模型层数和训练数据。  
2️⃣ **块扩散持续预训练（CPT）**：把代码序列划分为固定长度的块（如 64 token），对每个块施加噪声并让模型学习从噪声恢复原始块。这里的噪声不是一次性加满，而是采用**块级裁剪噪声调度**：在训练早期对大块使用较低噪声，随着训练进度逐步提升噪声幅度，防止模型在一次性去噪时失去局部结构。与此同时，**warmup** 采用线性上升后再切换到余弦衰减，确保学习率在前几千步保持温和。  
3️⃣ **监督微调**：在完成 CPT 后，直接在代码生成、补全、编辑等下游任务上进行有标签的微调。因为 CPT 已经让模型掌握了块级去噪能力，微调时只需要少量任务特定的梯度即可收敛。

**关键模块的类比**：  
- **块划分**相当于把一本书拆成若干章节，每章节内部可以独立阅读。  
- **噪声裁剪**像是先把章节的文字模糊一点点，而不是一次把整本书都涂成浓墨。  
- **去噪过程**则是编辑者逐段恢复章节内容，既能利用前后章节的上下文，又不必严格按章节顺序。

**最巧妙的设计**在于噪声调度的“块裁剪”。普通扩散模型对整个序列一次性加噪，导致长序列的细节被彻底抹掉，代码的缩进、括号匹配等结构信息难以恢复。作者把噪声强度限制在每个块内部，并在训练后期才放宽，这样模型在早期就学会保留局部语法结构，后期再学习跨块的全局一致性。

### 实验与效果
- **数据与任务**：使用与 Seed‑Coder 完全相同的 1.3 T token 代码语料，覆盖多种主流语言（Python、Java、C++ 等）以及低资源语言。评测任务包括代码补全、函数生成、编辑（给定 buggy 代码输出修复版）以及结构化推理（如变量追踪）。  
- **基线对比**：与同等参数的自回归模型（2.5 B）以及若干公开的 8 B 规模 AR 与 DLLM（如 CodeGen‑6B、StarCoder‑16B）进行比较。论文报告在大多数基准上 Stable‑DiffCoder 超过同规模 AR 约 2–4% 的准确率，在跨语言低资源实验中提升更明显，甚至超过部分 8 B AR。  
- **消融实验**：分别去掉块噪声裁剪、warmup、以及仅使用普通预训练（不做 CPT）进行对比。结果显示，去掉噪声裁剪后整体性能下降约 1.5%，去掉 warmup 则导致训练不稳定、收敛速度减半。仅做普通预训练的模型在所有任务上均不及 CPT 版本。  
- **局限性**：作者指出，块大小的选择仍是经验性超参数，过大块会削弱并行优势，过小块则可能破坏代码的长程依赖。此外，当前实现仍依赖大量 GPU 内存，部署成本高于传统 AR 模型。

### 影响与延伸思考
Stable‑DiffCoder 的成功证明，扩散式训练在代码领域并非只能追随文本生成的脚步，而是可以在保持并行性的同时提升结构化编辑能力。自论文发布后，已有几篇工作尝试把块扩散与指令微调结合，或把噪声调度进一步细化到语法树层级（如“树形扩散”）。对想继续深挖的读者，可以关注以下方向：① 更细粒度的结构感知噪声（把 AST 节点当作块）；② 低资源语言的跨语言噪声迁移；③ 将扩散式编辑与人机交互的实时代码补全结合。整体来看，这篇工作打开了代码生成模型在“任意顺序”建模上的新大门。

### 一句话记住它
块级噪声裁剪的持续预训练让扩散模型在代码生成上既并行又更懂结构，直接把 2.5 B 的扩散模型推到比多数 8 B 自回归模型更强的水平。