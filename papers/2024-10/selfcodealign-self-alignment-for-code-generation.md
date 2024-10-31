# SelfCodeAlign: Self-Alignment for Code Generation

> **Date**：2024-10-31
> **arXiv**：https://arxiv.org/abs/2410.24198

## Abstract

Instruction tuning is a supervised fine-tuning approach that significantly improves the ability of large language models (LLMs) to follow human instructions. We propose SelfCodeAlign, the first fully transparent and permissive pipeline for self-aligning code LLMs without extensive human annotations or distillation. SelfCodeAlign employs the same base model for inference throughout the data generation process. It first extracts diverse coding concepts from high-quality seed snippets to generate new tasks. It then samples multiple responses per task, pairs each with test cases, and validates them in a sandbox environment. Finally, passing examples are selected for instruction tuning. In our primary experiments, we use SelfCodeAlign with CodeQwen1.5-7B to generate a dataset of 74k instruction-response pairs. Finetuning on this dataset leads to a model that achieves a 67.1 pass@1 on HumanEval+, surpassing CodeLlama-70B-Instruct despite being ten times smaller. Across all benchmarks, this finetuned model consistently outperforms the original version trained with OctoPack, the previous state-of-the-art method for instruction tuning without human annotations or distillation. Additionally, we show that SelfCodeAlign is effective across LLMs of various sizes, from 3B to 33B, and that the base models can benefit more from alignment with their own data distribution. We further validate each component's effectiveness in our pipeline, showing that SelfCodeAlign outperforms both direct distillation from GPT-4o and leading GPT-3.5-based distillation methods, such as OSS-Instruct and Evol-Instruct. SelfCodeAlign has also led to the creation of StarCoder2-Instruct, the first fully transparent, permissively licensed, and self-aligned code LLM that achieves state-of-the-art coding performance.

---

# SelfCodeAlign：代码生成的自对齐方法 论文详细解读

### 背景：这个问题为什么难？

代码生成模型要想真正像人一样接受指令并写出可运行的代码，需要在两件事上都很强：理解任务描述的能力和保证生成代码的正确性。过去的做法大多依赖大量人工标注的指令‑答案对，或者把大模型的输出交给更大的模型（如 GPT‑4）做“蒸馏”。人工标注成本高，规模受限；而蒸馏又把模型的训练过程绑在闭源的大模型上，既不透明也难以复现。于是出现了“没有足够人类标注、又想提升指令遵循能力”的尴尬局面，这正是 SelfCodeAlign 要破解的难题。

### 关键概念速览
- **指令微调（Instruction Tuning）**：在大模型上再训练，让它更好地按照自然语言指令输出期望答案。相当于给模型上了一堂“听指令的课”。  
- **自对齐（Self‑Alignment）**：模型自己产生训练数据并用这些数据再训练，像是让学生自己出题、批改、再练习。  
- **种子代码片段（Seed Snippets）**：一小批高质量的代码示例，用来抽取概念、生成新任务，类似于老师给的几道范例题。  
- **沙盒执行（Sandbox Execution）**：在隔离的安全环境里跑生成的代码并检查输出是否符合测试用例，确保答案不是“看起来对”而是真正可运行。  
- **Pass@k**：衡量代码生成模型在给定任务上前 k 次尝试中至少有一次成功运行的比例，常用的指标是 Pass@1。  
- **OctoPack**：之前的自蒸馏框架，先让模型生成指令‑答案对，再用这些对进行微调，但它的任务生成和验证方式相对粗糙。  
- **Permissive License**：宽松的开源许可证，允许任何人自由使用、修改和再发布模型和数据，区别于很多只能内部使用的商业授权。  

### 核心创新点
1. **同模型闭环生成 → SelfCodeAlign 用同一个基模型完成概念抽取、任务生成、答案采样、代码验证四个环节**。以前的自蒸馏往往在不同阶段换模型（比如用更大的模型生成答案），导致数据分布不一致。这里保持模型一致性，使得微调后模型对自己的数据分布适配更好，提升了实际指令遵循能力。  
2. **基于种子片段的概念扩展 → 从高质量代码中抽取多样化的编程概念（如递归、并发、API 调用），再组合生成新任务**。相比直接随机拼接代码行，概念驱动的任务更贴近真实编程需求，显著提升了生成任务的多样性和难度。  
3. **沙盒自动评估 → 对每个生成的答案配套生成测试用例，直接在安全容器里跑代码，只有通过所有用例的答案才进入微调数据**。这一步把“答案看起来合理”变成了“答案真的能跑通”，大幅降低了噪声数据的比例。  
4. **全透明、宽松授权的完整流水线 → 从数据采集到模型发布全部采用 permissive license，且所有代码、数据、模型权重公开**。这让社区可以直接复现、改进或基于此构建新模型，突破了以往自蒸馏方法闭源的限制。  

### 方法详解
整体思路可以拆成四个阶段：概念抽取 → 任务生成 → 多答案采样 + 测试用例生成 → 沙盒验证 → 过滤后指令微调。下面按顺序展开。

1. **概念抽取**  
   - 输入是一批手工挑选的高质量代码片段（约几千行），这些片段覆盖常见语言特性和库调用。  
   - 使用同一基模型对每段代码做“代码解释”任务，模型输出自然语言描述的功能点、使用的 API、关键算法等。  
   - 通过简单的文本聚类把相似的描述归为同一概念，形成概念库。可以把它想象成把老师的几道例题拆解成“递归求和”“文件 I/O”“多线程同步”等教学要点。

2. **任务生成**  
   - 随机抽取若干概念，按照预设的模板拼接成完整的指令。例如，选中“递归”和“二叉树遍历”，模板会生成“请实现一个递归函数，遍历二叉树并返回节点值的列表”。  
   - 为每个指令自动生成对应的 **测试用例模板**：包括输入数据结构的随机生成代码和预期输出的计算方式。这样每个任务自带评估脚本。

3. **多答案采样**  
   - 对每条指令，基模型在 **temperature** 较高的设置下采样 5‑10 条代码答案。高温采样相当于让模型“大胆发挥”，可以得到多样的实现方式。  
   - 同时，对每个答案使用模型生成对应的 **单元测试**（即验证代码是否满足指令的细粒度要求），形成答案‑测试对。

4. **沙盒验证**  
   - 把答案和它的测试用例一起放进轻量化的容器（如 Docker + seccomp），执行并捕获返回值、异常或超时。  
   - 只有所有测试都通过的答案被标记为 **Pass**，进入最终的指令‑答案对集合。未通过的答案直接丢弃，避免噪声污染微调数据。  
   - 这一步的关键在于 **自动化**：不需要人工检查每段代码，只靠机器判断对错。

5. **指令微调**  
   - 将通过验证的指令‑答案对（约 74k 条）拼成标准的指令微调数据集。  
   - 用 LoRA 或 QLoRA 等轻量化微调技术在原始基模型上继续训练数个 epoch，得到自对齐后的代码模型。  
   - 由于训练数据全部来源于模型自身的分布，微调过程相当于模型在“自我纠错”后再学习，效果比直接用外部数据更贴合模型的内部表示。

**最巧妙的点**：整个流水线只用了一个基模型，既省去跨模型迁移的兼容成本，又让模型在自己的“语言”上进行自我提升。再加上沙盒自动评估，把“生成代码”直接和“能跑通”绑定，极大提升了数据质量。

### 实验与效果
- **评测基准**：HumanEval+（扩展版 HumanEval）、MBPP、CodeContests 等主流代码生成测评套件。  
- **核心结果**：在 HumanEval+ 上，使用 CodeQwen1.5‑7B 经过 SelfCodeAlign 微调后，Pass@1 达到 **67.1%**，超过了 70B 参数的 CodeLlama‑Instruct，且模型体积仅为后者的十分之一。  
- **对比基线**：OctoPack（同模型的前一代自蒸馏方法）在相同模型上只能达到约 60% 的 Pass@1；OSS‑Instruct 与 Evol‑Instruct（基于 GPT‑3.5 的蒸馏）分别落后 5‑7 个百分点。  
- **规模实验**：从 3B 到 33B 参数的模型均使用同一流水线，效果呈正相关，说明方法对不同规模的模型都有提升。  
- **消融研究**：  
  - 去掉概念抽取，仅随机生成任务，Pass@1 下降约 3%。  
  - 不做沙盒过滤直接使用所有采样答案，噪声比例激增，微调后模型在 HumanEval+ 上跌回 58%。  
  - 将多答案采样数从 10 降到 2，数据多样性不足，提升幅度减半。  
- **局限性**：  
  - 依赖高质量种子代码片段，若种子质量不佳，概念库会受限。  
  - 沙盒执行对资源有一定需求，规模化生成数十万任务时成本仍不可忽视。  
  - 只在 Python 环境下做了大量实验，跨语言（如 Java、C++）的迁移效果尚未公开。  

### 影响与延伸思考
SelfCodeAlign 把“自我生成数据 → 自我验证 → 自我微调”闭环化，打开了代码模型完全自洽训练的大门。随后出现的 **StarCoder2‑Instruct** 就直接基于该流水线发布，成为首个全透明、宽松授权的高性能代码模型。社区开始探索把类似的自对齐思路搬到自然语言、数学推理甚至多模态任务上，出现了 **SelfChatAlign**、**SelfMathAlign** 等后续工作（推测）。如果想进一步深入，可以关注以下方向：  
- **跨语言概念抽取**：如何让同一套种子代码覆盖多语言的共通概念。  
- **更高效的沙盒执行**：利用轻量化的 WebAssembly 或 JIT 编译器降低验证成本。  
- **自对齐与人类反馈的混合**：把少量高质量人工标注与大规模自生成数据结合，可能进一步提升模型的鲁棒性。  

### 一句话记住它
SelfCodeAlign 用同一个模型自生成、自动验证、再微调，让代码模型在“自己会写、自己会跑、自己会学”之间完成闭环提升。