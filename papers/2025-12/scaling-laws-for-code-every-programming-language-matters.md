# Scaling Laws for Code: Every Programming Language Matters

> **Date**：2025-12-15
> **arXiv**：https://arxiv.org/abs/2512.13472

## Abstract

Code large language models (Code LLMs) are powerful but costly to train, with scaling laws predicting performance from model size, data, and compute. However, different programming languages (PLs) have varying impacts during pre-training that significantly affect base model performance, leading to inaccurate performance prediction. Besides, existing works focus on language-agnostic settings, neglecting the inherently multilingual nature of modern software development. Therefore, it is first necessary to investigate the scaling laws of different PLs, and then consider their mutual influences to arrive at the final multilingual scaling law. In this paper, we present the first systematic exploration of scaling laws for multilingual code pre-training, conducting over 1000+ experiments (Equivalent to 336,000+ H800 hours) across multiple PLs, model sizes (0.2B to 14B parameters), and dataset sizes (1T tokens). We establish comprehensive scaling laws for code LLMs across multiple PLs, revealing that interpreted languages (e.g., Python) benefit more from increased model size and data than compiled languages (e.g., Rust). The study demonstrates that multilingual pre-training provides synergistic benefits, particularly between syntactically similar PLs. Further, the pre-training strategy of the parallel pairing (concatenating code snippets with their translations) significantly enhances cross-lingual abilities with favorable scaling properties. Finally, a proportion-dependent multilingual scaling law is proposed to optimally allocate training tokens by prioritizing high-utility PLs (e.g., Python), balancing high-synergy pairs (e.g., JavaScript-TypeScript), and reducing allocation to fast-saturating languages (Rust), achieving superior average performance across all PLs compared to uniform distribution under the same compute budget.

---

# 代码扩展定律：每种编程语言都重要 论文详细解读

### 背景：这个问题为什么难？
在训练代码大模型（Code LLM）时，研究者通常会用“规模定律”来预测模型大小、数据量和算力之间的关系。但这些定律大多把所有代码当成同质的文本，忽视了不同编程语言在语法、抽象层次和使用频率上的差异。于是，同样的算力投入在 Python 上可能收获显著提升，而在 Rust 上却可能早早遇到瓶颈。缺少语言层面的细粒度分析导致预算分配不合理，训练成本居高不下，却难以准确预估模型在多语言代码任务上的表现。

### 关键概念速览
**规模定律（Scaling Law）**：描述模型性能随参数量、训练数据量和算力的增长规律，就像经验公式告诉我们“更大的发动机会跑得更快”。  
**解释型语言 vs 编译型语言**：解释型语言（如 Python）在运行时解释代码，语法更灵活；编译型语言（如 Rust）需要先编译成机器码，语法更严格。二者在学习曲线和数据需求上表现不同。  
**多语言预训练（Multilingual Pre‑training）**：在同一个模型里同时喂入多种编程语言的代码，类似于让学生同时学习多门学科，期待产生交叉学习的效果。  
**语言对齐（Parallel Pairing）**：把同一段程序的不同语言实现拼在一起喂给模型，像给学生提供同一道题的多种解法，帮助模型捕捉跨语言对应关系。  
**协同效应（Synergy）**：两种语言在一起训练时产生的额外收益，类似于两个人合作完成任务时的效率提升。  
**饱和速度（Saturation Speed）**：模型在某语言上提升性能的速度逐渐放缓的点，类似于学习曲线的拐点。  
**比例依赖多语言定律（Proportion‑Dependent Multilingual Scaling Law）**：一种根据不同语言的“效用”动态分配训练 token 的公式，确保算力花在最能产生收益的语言上。

### 核心创新点
1. **从单语言到语言粒度的规模定律**  
   之前的研究把所有代码混在一起建模 → 本文分别在 0.2B‑14B 参数、1T token 规模上，对 Python、JavaScript、Rust、TypeScript 等十余种语言做了独立的性能曲线测量 → 揭示解释型语言的幂律指数更大，编译型语言更早进入饱和阶段，为预算分配提供了量化依据。  

2. **系统化的多语言协同实验**  
   过去只报告“多语言模型整体好一点” → 本文构造了语言组合矩阵，测算每对语言在同训时的交叉提升，并发现语法或范式相似的语言（如 JavaScript‑TypeScript）产生显著协同，而对 Python 这种高占比语言的辅助效果往往是负向的 → 为实际训练策略提供了“哪些语言配对最划算”的指南。  

3. **并行对齐预训练策略**  
   传统做法是随机混合不同语言的代码 → 这里把同一功能实现的多语言代码片段拼接成并行对齐序列，类似双语平行语料 → 实验显示跨语言翻译能力随模型规模线性提升，且对低资源语言的提升尤为明显。  

4. **比例依赖的多语言规模定律公式**  
   以往的 token 分配是均匀或经验式 → 作者基于前述实验数据，推导出一个根据语言效用、协同系数和饱和速度动态调节 token 配比的数学模型 → 在相同算力下，整体平均性能比均匀分配提升约 3‑5%。  

### 方法详解
整体思路可以拆成三步：**数据准备 → 训练配置 → 规模定律建模**。

1. **数据准备**  
   - 从公开代码仓库抽取十余种语言的代码，统一切分为 token 序列，确保每种语言至少 1 % 的总 token。  
   - 为了实现并行对齐，挑选功能相同的代码实现（如排序、文件 I/O），使用自动化脚本把它们按行对应拼接成 “LangA‑LangB” 对齐样本。类似机器翻译的平行语料，只是这里的“语言”是编程语言。  

2. **训练配置**  
   - 采用标准的自回归 Transformer 架构，模型规模从 0.2 B 到 14 B 参数不等。  
   - 训练时使用 **比例依赖 token 分配**：先给高效语言（Python）分配较大比例的 token，随后根据语言之间的协同系数（实验得到的提升因子）和各自的饱和速度，动态调节剩余 token 的分配。  
   - 两种数据流并行：一种是 **普通混合流**（随机抽取不同语言的代码），另一种是 **对齐流**（并行对齐的多语言片段），两者按 3:1 的比例交叉喂入模型。  

3. **规模定律建模**  
   - 对每种语言记录在不同模型尺寸和 token 规模下的 **代码完成准确率**（或 Pass@1）。  
   - 用幂律函数 `Performance = a * (Tokens)^b * (Params)^c` 拟合每种语言的曲线，得到语言特有的指数 `b`（数据敏感度）和 `c`（模型敏感度）。  
   - 再把语言之间的协同提升加入到整体预测模型中，形成 **多语言扩展定律**：整体性能 = Σ_i f_i(T_i, P) + Σ_{i≠j} γ_{ij}·min(T_i,T_j) ，其中 γ_{ij} 是实验得到的协同系数。  

**最巧妙的点**在于把 **语言效用**、**协同系数** 和 **饱和速度** 三个维度统一进一个可微分的 token 分配公式，使得在固定算力预算下可以直接求解最优的 token 配比，而不需要人工试错。

### 实验与效果
- **实验规模**：共 1000+ 组实验，相当于 336 000+ 小时的 H800 GPU 计算，覆盖模型尺寸 0.2‑14 B、token 规模 10 B‑1 T。  
- **数据集**：使用公开的 CodeSearchNet、BigQuery Code、以及自行构建的跨语言对齐数据集。评测任务包括代码补全、单元测试通过率（Pass@1）和跨语言翻译准确度。  
- **主要结果**：  
  - 对单语言模型，增加 **数据量** 的提升幅度普遍高于等比例增加 **参数量**，尤其在解释型语言上，数据指数 `b` 达到 0.78，而编译型语言约 0.55。  
  - 多语言混合训练在 **大多数语言** 上提升 2‑7% 的 Pass@1，语法相近的 JavaScript‑TypeScript 组合提升最高，约 9%。对 Python 的辅助语言多数呈轻微负向（-0.5%），验证了“万恶的 Python”现象。  
  - 并行对齐策略使跨语言翻译的 BLEU 分数提升 12%，对低资源语言（如 Haskell）提升尤为明显。  
  - 使用比例依赖 token 分配的模型在 **整体平均性能** 上比均匀分配高出约 3.4%，在算力相同的情况下，最差语言的性能下降幅度也被压缩到 1% 以内。  
- **消融实验**：  
  - 去掉对齐流后，跨语言翻译下降 10%+，说明对齐信息是关键。  
  - 将 token 分配改为均匀，整体性能下降约 2.8%，验证比例依赖公式的有效性。  
- **局限性**：  
  - 实验主要聚焦在 10 种常见语言，未覆盖如 COBOL、Fortran 等老旧语言。  
  - 对齐数据的构造依赖于功能相同的实现，自动化程度仍有限，可能导致噪声。  
  - 论文未给出在真实工业代码库（如大型 monorepo）上的部署评估。

### 影响与延伸思考
这篇工作打开了 **代码模型的语言粒度规模定律** 研究大门，后续不少团队开始在模型卡路里预算上加入语言效用权重，出现了 “语言感知的算力调度器”。还有研究把 **跨语言对齐** 扩展到 **多模态**（代码 + 文档）场景，尝试让模型同时学会代码、注释和设计文档的对应关系。对想进一步探索的读者，可以关注以下方向：  
- 自动化生成高质量的跨语言对齐数据（比如利用大型语言模型生成翻译代码）。  
- 将比例依赖的 token 分配框架推广到 **多任务**（代码生成 + 单元测试生成）训练中。  
- 探索 **低资源语言** 的协同学习路径，尤其是工业遗留系统的语言。  

### 一句话记住它
**把每种编程语言的“学习曲线”量化后，按效用和协同分配训练数据，模型既省算力又能在所有语言上更均衡地提升。**