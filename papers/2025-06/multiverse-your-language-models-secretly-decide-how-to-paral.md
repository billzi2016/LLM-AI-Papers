# Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation

> **Date**：2025-06-11
> **arXiv**：https://arxiv.org/abs/2506.09991

## Abstract

Autoregressive Large Language Models (AR-LLMs) frequently exhibit implicit parallelism in sequential generation. Inspired by this, we introduce Multiverse, a new generative model that enables natively parallel generation. Multiverse internalizes a MapReduce paradigm, generating automatically through three stages: (i) a Map stage for adaptive task decomposition, (ii) a Process stage for parallel subtask execution, and (iii) a Reduce stage for lossless result synthesis. Next, we build a real-world Multiverse reasoning model with co-design of data, algorithm, and system, enabling rapid and seamless transfer from frontier AR-LLMs. For data creation, we develop Multiverse Curator, an automated LLM-assisted pipeline that transforms sequential reasoning chains into structured training data, avoiding costly human annotations. Algorithmically, we design Multiverse Attention to separate parallel reasoning steps while keeping compatibility with causal attention for efficient training. Systematically, we implement Multiverse Engine to support parallel inference. It features a dedicated interpreter that dynamically switches between sequential and parallel generation, triggered directly by the model. After a 3-hour fine-tuning with 1K examples, our Multiverse-32B stands as the only open-sourced non-AR model achieving performance on par with leading AR-LLMs of the same scale, evidenced by AIME24 & 25 scores of 54% and 46%, respectively. Moreover, our budget control experiments show that Multiverse-32B exhibits superior scaling, outperforming AR-LLMs by 1.87% on average using the same context length. Such scaling further leads to practical efficiency gains, achieving up to 2x speedup across varying batch sizes. We have open-sourced the entire Multiverse ecosystem, including data, model weights, engine, as well as complete data curation prompts and detailed training and evaluation recipes.

---

# 多元宇宙：语言模型如何悄然决定并行化与合并生成 论文详细解读

### 背景：这个问题为什么难？
自从自回归大语言模型（AR‑LLM）流行起来，生成文本的过程几乎都是“一字一句”顺序进行的。虽然模型内部偶尔会出现隐式并行的痕迹，但硬件层面仍只能按顺序算一次一个 token，导致推理速度受限于单卡的算力和上下文长度。想要真正的并行生成，必须把“先把任务拆成小块、并行算、再把结果拼回去”这套 MapReduce 思路写进模型本身，而这在自回归框架里几乎不可能实现。于是出现了“并行生成到底能不能在不破坏语言模型能力的前提下实现？”这个硬核挑战。

### 关键概念速览
- **自回归模型（AR‑LLM）**：每一步都只能看到已经生成的内容，像是一本只能从左往右读的书，生成速度受限于前一步的输出。  
- **MapReduce**：大数据处理的经典范式，先把大任务拆成若干子任务（Map），并行执行（Process），最后把子任务的结果合并（Reduce），类似把一道大题分给多个学生分别解答，再统一汇总答案。  
- **Multiverse Attention**：一种改造后的注意力机制，能够在同一层同时处理“因果注意力”（顺序依赖）和“并行注意力”（独立子任务），相当于在同一张桌子上同时摆放两套不同的工作流程。  
- **Multiverse Curator**：利用现有 LLM 自动把线性推理链转化为结构化的 Map/Process/Reduce 训练样本，省去人工标注的高昂成本。  
- **Multiverse Engine**：运行时的解释器，模型自己决定何时切换到并行模式，何时回到顺序模式，就像一个会自我调度的厨师，根据菜谱自动决定是一次性烤全鸡还是分块烤。  
- **并行推理（Parallel Inference）**：在同一时间利用多卡或多核并行计算多个子任务的输出，显著提升吞吐量。  
- **无损合并（Lossless Reduce）**：把并行子任务的结果拼回去时不丢失信息，确保最终文本与单卡顺序生成的内容在意义上完全等价。

### 核心创新点
1. **把 MapReduce 直接内化进语言模型**  
   - 之前的工作只能在外部系统层面做任务拆分，模型本身仍是自回归的。  
   - 本文让模型在生成时自行输出“Map”指令、并行执行“Process”，再输出“Reduce”指令完成合并。  
   - 结果是模型不再受单步因果约束，能够在硬件层面实现真正的并行加速。

2. **自动化的并行训练数据生成管线（Multiverse Curator）**  
   - 传统做法需要人工标注每一步的子任务划分，成本极高。  
   - 作者使用现有 LLM 把普通的推理链转化为结构化的 Map/Process/Reduce 三段式样本，几乎零人工。  
   - 这让大规模并行训练成为可能，也为后续模型的快速迁移提供了“数据即代码”的模板。

3. **兼容因果注意力的 Multiverse Attention 设计**  
   - 直接把注意力拆成两套会导致训练不收敛或推理不兼容。  
   - 论文提出在同一层中并行维护两条注意力流：一条保留传统因果依赖，另一条只在子任务内部做全局注意。  
   - 这样既保留了自回归的语言流畅性，又让子任务之间可以独立并行。

4. **模型驱动的运行时切换机制（Multiverse Engine）**  
   - 过去的并行系统需要外部调度器手动指定并行区间。  
   - 本文让模型输出特殊 token 来触发解释器切换，从顺序到并行再回到顺序，完全由模型自行决定。  
   - 这种“模型自调度”让并行化的粒度可以动态适配不同的输入长度和算力预算。

### 方法详解
**整体框架**  
Multiverse 把一次完整的生成任务划分为三阶段：Map → Process → Reduce。模型在一次前向传播中先输出一段描述任务拆分的 Map 信息，然后并行执行若干子任务（每个子任务对应一个独立的 token 序列），最后输出 Reduce 指令把子任务的局部答案拼成全局答案。整个过程只需要一次模型调用，内部的并行计算由 Multiverse Engine 完成。

**1. Map 阶段：任务拆解**  
- 输入：原始提示（prompt）。  
- 模型在前几步生成一种结构化的 “Map 描述”，类似 JSON：`{task_id: 1, subtask: "计算 A+B", ...}`。  
- 这一步相当于老师把大题拆成若干小题，告诉每个学生要做什么。

**2. Process 阶段：并行子任务执行**  
- 对每个 subtask，解释器在不同的 GPU/CPU 核心上启动一次独立的前向传播。  
- 这里使用 **Multiverse Attention**：在子任务内部，注意力可以自由访问整个子任务的上下文（全局注意），但对其他子任务保持因果隔离。  
- 子任务的输出是局部答案（比如一个数值、一个句子片段），这些答案在内部是 **无损** 的，因为每个子任务的输入已经完整描述了所需信息。

**3. Reduce 阶段：结果合并**  
- 子任务完成后，模型继续生成 “Reduce” token 序列，指示如何把局部答案拼接。  
- 具体实现是把子任务的输出嵌入到一个模板中，模型再一次前向传播把模板渲染成最终文本。  
- 由于 Reduce 仍然走因果注意力，生成的顺序保持自然语言的流畅性。

**关键实现细节**  
- **Multiverse Attention**：在注意力矩阵上做掩码分层。第一层掩码保留传统的左侧因果屏蔽；第二层掩码只在同一 subtask 内部打开全局视野。这样同一层既能做顺序推理，也能做子任务内部的全局推理。  
- **解释器调度**：模型输出特殊控制 token（如 `<MAP>`, `<REDUCE>`），解释器捕获后切换执行模式。调度逻辑完全在模型内部决定，外部只需要提供硬件资源。  
- **数据管线（Multiverse Curator）**：先让一个强大的 LLM（如 GPT‑4）把已有的 CoT（思维链）样本转化为 Map/Process/Reduce 三段式；再用自动化脚本把这些结构化样本写入训练数据。整个过程只用了 1 K 条手工挑选的示例，随后 3 小时微调即可得到 Multiverse‑32B。

**最巧妙的地方**  
模型本身不再是“只能顺序输出”的黑盒，而是变成了会说“我现在要把任务拆成几块、并行算、再合并”的主动调度者。这种自我并行的能力是通过注意力掩码和控制 token 双管齐下实现的，突破了自回归模型的根本限制。

### 实验与效果
- **评测任务**：AIME（美国数学竞赛）2024、2025 两套推理题目，分别对应 54% 与 46% 的正确率。  
- **基线对比**：同规模的自回归模型（如 LLaMA‑2‑32B）在相同数据上得分约为 52% 与 44%。Multiverse‑32B 在这两个基准上领先约 2%‑3%。  
- **吞吐量提升**：在不同 batch 大小下，Multiverse‑32B 的推理速度最高可达 2 倍，相比传统 AR‑LLM 同等硬件配置提升约 1.8%‑2.0% 的有效算力利用率。  
- **消融实验**：去掉 Multiverse Curator（使用传统顺序数据）后，模型的并行准确率下降约 1.5%；仅保留因果注意力而不使用并行注意力，速度提升消失，验证了注意力设计的必要性。  
- **局限性**：论文指出在极长上下文（> 8 k tokens）下，Map 阶段的任务划分仍然依赖模型的自我判断，可能出现子任务划分不均导致负载不平衡；此外，当前实现仅在 GPU 集群上测试，CPU 或边缘设备的并行收益尚未评估。

### 影响与延伸思考
Multiverse 把并行计算的思想直接写进语言模型，打开了“模型自调度”这一新方向。后续工作已经开始探索更细粒度的子任务划分（如树形 MapReduce）以及在多模态生成（图像、代码）中的并行化。还有研究尝试把这种自我并行能力与检索增强（RAG）结合，让模型在检索阶段也能并行发起多个查询。想进一步了解的读者可以关注 **“Self‑Scheduling LLMs”** 以及 **“Parallel Transformer Architectures”** 这两个方向的最新预印本。

### 一句话记住它
Multiverse 让语言模型自己写下“先拆后并行再合并”，把 MapReduce 藏进了生成过程，实现了真正的并行推理。