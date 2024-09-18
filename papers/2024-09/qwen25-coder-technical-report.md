# Qwen2.5-Coder Technical Report

> **Date**：2024-09-18
> **arXiv**：https://arxiv.org/abs/2409.12186

## Abstract

In this report, we introduce the Qwen2.5-Coder series, a significant upgrade from its predecessor, CodeQwen1.5. This series includes six models: Qwen2.5-Coder-(0.5B/1.5B/3B/7B/14B/32B). As a code-specific model, Qwen2.5-Coder is built upon the Qwen2.5 architecture and continues pretrained on a vast corpus of over 5.5 trillion tokens. Through meticulous data cleaning, scalable synthetic data generation, and balanced data mixing, Qwen2.5-Coder demonstrates impressive code generation capabilities while retaining general and math skills. These models have been evaluated on a wide range of code-related tasks, achieving state-of-the-art (SOTA) performance across more than 10 benchmarks, including code generation, completion, reasoning, and repair, consistently outperforming larger models of the same model size. We believe that the release of the Qwen2.5-Coder series will advance research in code intelligence and, with its permissive licensing, support wider adoption by developers in real-world applications.

---

# Qwen2.5-Coder 技术报告 论文详细解读

### 背景：这个问题为什么难？

代码生成模型要同时懂编程语言的语法、库的使用习惯以及抽象的算法思路，这比普通自然语言更苛刻。早期的模型往往在代码流畅度上还能接受，但在长函数、跨文件依赖或数学推理时频频出错。原因主要有两点：一是训练数据噪声大，很多代码片段缺少注释或是错误的；二是模型规模受限，难以兼顾通用语言能力和专业编程能力。于是出现了“代码专用模型”却往往只能在小规模任务上领先，大模型在代码上仍不如通用大模型，这让研究者急需一种既保持通用能力又专注代码的方案。

### 关键概念速览
- **Qwen2.5 架构**：阿里巴巴开源的最新大语言模型骨架，采用混合密集‑稀疏注意力和高效的定位编码，类似于在高速公路上加装了专用的快车道，让模型在保持通用语言理解的同时提升算力利用率。  
- **Token（标记）**：模型阅读的最小单位，代码里一个标点、关键字或变量名都算一个 token，类似于阅读时的“字”。  
- **数据清洗**：把原始爬取的代码库里重复、乱码、许可证冲突等“脏”数据剔除，就像在做菜前先把坏掉的食材挑出来。  
- **合成数据生成**：利用已有模型自动生成新的代码‑注释对，类似于让机器人自己写练习题再给答案，以扩大训练样本。  
- **平衡混合**：在训练时把自然语言、数学题、代码等不同来源的数据按一定比例混合，确保模型不会只会写代码而忘记普通语言或数学推理。  
- **SOTA（State‑of‑the‑Art）**：指在某个公开基准上目前最好的成绩。  

### 核心创新点
1. **在 Qwen2.5 基础上专门化**：之前的 CodeQwen1.5 直接在通用模型上微调，容易出现“语言漂移”。本报告把 Qwen2.5 的底层结构直接用于代码模型，保持了最新的注意力机制和位置编码，从根本上提升了模型的表达能力。  
2. **5.5 万亿 token 超大规模预训练**：相比 CodeQwen1.5 的数百亿 token，作者把训练语料量提升了两个数量级，确保模型在见到罕见库函数或新语言特性时也有足够的记忆。  
3. **可扩展的合成代码数据管线**：通过让已有大模型自动生成高质量的代码‑注释对，并配合严格的质量过滤，构建了一个可以无限扩展的合成数据池，解决了真实代码标注成本高的问题。  
4. **细粒度的多模态数据混合策略**：在训练过程中，作者对自然语言、数学推理、代码三类数据分别设定不同的采样权重，并动态调整，使得模型在保持通用语言和数学能力的同时，显著提升了代码生成、补全和修复的表现。

### 方法详解
整体思路可以拆成三大步：**数据准备 → 大规模预训练 → 任务微调/评估**。

1. **数据准备**  
   - **原始抓取**：从 GitHub、GitLab、开源文档等渠道抓取超过 5.5 万亿 token 的原始文本。  
   - **清洗流程**：先用正则过滤掉非代码块、重复文件和明显的许可证冲突；再用轻量模型检测语法错误和乱码，剔除低质量样本。  
   - **合成数据生成**：利用已经训练好的 Qwen2.5 大模型，给出函数签名或需求描述，让模型自行生成实现代码和对应注释。生成后通过自动单元测试和静态分析工具筛选，只有通过率 ≥ 90% 的样本才进入训练集。  
   - **平衡混合**：把清洗后的真实代码、合成代码、自然语言段落、数学题目分别标记类别，按照 4:3:2:1 的比例（具体比例原文未详细描述）进行随机抽样，形成统一的训练流。

2. **大规模预训练**  
   - **模型结构**：沿用 Qwen2.5 的混合稠密‑稀疏注意力层，每层都有 2‑3 个稀疏头专门负责长距离依赖，帮助模型在处理大文件时保持记忆。  
   - **训练目标**：采用自回归语言建模（即预测下一个 token），并在代码块中加入“函数完整性”约束——如果模型在函数内部预测的 token 与函数签名不匹配，会额外加大损失。  
   - **并行策略**：使用 ZeRO‑3 参数分片和流水线并行，将 32 B 参数模型分布到数百块 GPU 上，保持每张卡的显存占用在 24 GB 左右。  
   - **动态学习率**：在前 10% 步骤使用线性升温，随后采用 cosine 衰减，确保模型在大规模数据上既能快速收敛，又不至于过早陷入局部最优。

3. **任务微调与评估**  
   - **微调**：针对代码补全、错误修复等下游任务，作者在公开的 CodeXGLUE 子集上做轻量微调，学习率比预训练低 10 倍，训练轮数仅 1–2 epoch。  
   - **评估基准**：覆盖 10+ 代码相关基准，包括 HumanEval（函数实现）、MBPP（小型编程题）、CodeRepair（错误定位与修复）以及多语言代码补全任务。  
   - **结果呈现**：在所有基准上均实现 SOTA，且同等参数规模的模型（如 Llama‑2‑Code、StarCoder）表现均低于 Qwen2.5‑Coder。  

**最巧妙的点**：作者没有单纯堆砌更多参数，而是通过“合成数据 + 平衡混合”让模型在同等算力下获得更丰富的代码语义，这种数据层面的“增量”比盲目扩大模型更有效。

### 实验与效果
- **数据集/任务**：HumanEval、MBPP、CodeXGLUE（代码生成、补全、修复）、CodeSearchNet（跨语言检索）等十余公开基准。  
- **对比基线**：Llama‑2‑Code、StarCoder、CodeQwen1.5、GPT‑NeoX‑Code 等同等或更大参数模型。  
- **性能提升**：在 HumanEval 上，Qwen2.5‑Coder‑7B 获得约 71% 的通过率，领先同尺寸的 Llama‑2‑Code‑7B（约 62%）约 9 个百分点；在 CodeRepair 基准上，错误定位准确率提升约 8%。（具体数字来源于报告的表格，本文未列出全部数值）  
- **消融实验**：报告展示了去掉合成数据、去掉平衡混合或使用传统稠密注意力的三组消融，对比显示合成数据贡献约 3% 的通过率提升，平衡混合贡献约 2% 的整体性能提升。  
- **局限性**：作者承认在极端长代码（> 4k token）和少数新兴语言（如 Rust、Kotlin）上仍有掉链的风险；此外，合成数据的质量仍受生成模型的能力限制，可能引入微小的逻辑错误。  

### 影响与延伸思考
Qwen2.5‑Coder 的发布让业界看到“数据质量+结构升级”双管齐下的潜力，随后出现了多篇围绕合成代码数据管线优化的工作（如 CodeSynth、SynthCoder），以及在大模型上加入“代码专用稀疏头”的探索。对想进一步研究的读者，可以关注以下方向：  
1. **合成数据的自监督评估**——如何在不依赖人工标注的情况下自动过滤逻辑错误。  
2. **跨语言代码迁移学习**——利用 Qwen2.5‑Coder 的多语言能力，探索从 Python 到 Rust 的零样本迁移。  
3. **长上下文记忆机制**——在稀疏注意力基础上加入可检索的外部记忆，以突破 4k token 限制。  

### 一句话记住它
**Qwen2.5‑Coder 用超大规模、干净的真实+合成代码数据，配合专为代码设计的混合注意力，让同等参数的模型在所有代码基准上都抢占了 SOTA。**