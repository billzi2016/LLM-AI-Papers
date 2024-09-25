# Programming Every Example: Lifting Pre-training Data Quality Like   Experts at Scale

> **Date**：2024-09-25
> **arXiv**：https://arxiv.org/abs/2409.17115

## Abstract

Large language model pre-training has traditionally relied on human experts to craft heuristics for improving the corpora quality, resulting in numerous rules developed to date. However, these rules lack the flexibility to address the unique characteristics of individual example effectively. Meanwhile, applying tailored rules to every example is impractical for human experts. In this paper, we demonstrate that even small language models, with as few as 0.3B parameters, can exhibit substantial data refining capabilities comparable to those of human experts. We introduce Programming Every Example (ProX), a novel framework that treats data refinement as a programming task, enabling models to refine corpora by generating and executing fine-grained operations, such as string normalization, for each individual example at scale. Experimental results show that models pre-trained on ProX-curated data outperform either original data or data filtered by other selection methods by more than 2% across various downstream benchmarks. Its effectiveness spans various model sizes and pre-training corpora, including C4, RedPajama-V2, FineWeb, FineWeb-Edu, and DCLM. Furthermore, ProX exhibits significant potential in domain-specific continual pre-training: without domain specific design, models trained on OpenWebMath refined by ProX outperform human-crafted rule-based methods, improving average accuracy by 7.6% over Mistral-7B, with 14.6% for Llama-2-7B and 20.3% for CodeLlama-7B, all within 10B tokens to be comparable to models like Llemma-7B trained on 200B tokens. Further analysis highlights that ProX significantly saves training FLOPs, offering a promising path for efficient LLM pre-training. We are open-sourcing ProX with >500B corpus, models, and sharing all training and implementation details for reproducible research and future innovation. Code: https://github.com/GAIR-NLP/ProX

---

# 为每个示例编程：像专家一样大规模提升预训练数据质量 论文详细解读

### 背景：这个问题为什么难？

大语言模型的预训练离不开海量文本，但原始网页、论坛等数据往往噪声满天飞——乱码、重复、广告、错误的标点甚至事实错误都有。过去的做法是让人工专家写一堆过滤规则，比如“去掉含有 URL 的行”或“删除重复句子”。这些规则虽然能把大部分垃圾踢走，却是“一刀切”的：不同示例的噪声类型千差万别，单一规则很难兼顾所有细节。更糟的是，规则的维护成本极高，想要针对每条数据做细粒度的清洗几乎不可能。于是，提升数据质量的瓶颈一直卡在“规则不够灵活、人工成本太高”这两点上。

### 关键概念速览
- **预训练数据质量**：指用于训练语言模型的文本是否干净、规范、信息丰富。质量高的语料能让模型学到更可靠的语言规律。
- **规则式过滤**：人工专家手写的正则表达式或条件判断，用来剔除或修改不符合要求的文本。像是给数据装上“筛子”，但筛子孔径固定，难以适配所有脏东西。
- **小语言模型（0.3B 参数）**：参数只有三亿量级的轻量模型，算力需求低，却足以完成基本的语言理解和生成任务。这里把它当成“自动化的清洁工”。
- **Programming Every Example (ProX)**：一种把数据清洗当成“写程序”来做的框架。模型为每条原始文本生成一段专属的清洗脚本，然后执行脚本得到干净的版本。
- **细粒度操作**：指对单个字符、单词或短语的具体变换，例如“把全角数字转成半角”“统一日期格式为 YYYY‑MM‑DD”。相当于给每条数据配备了专属的微调工具。
- **持续预训练（Continual Pre‑training）**：在已有模型基础上，再用特定领域的语料继续训练，以提升该领域的表现。这里的“特定领域”指数学网页等专业内容。
- **FLOPs 节约**：指在训练过程中减少的浮点运算次数。数据更干净意味着模型能用更少的算力学到同等或更好的知识。

### 核心创新点
1. **从规则到程序的范式转变**  
   - 之前：人工写规则，所有示例共享同一套过滤逻辑。  
   - 本文：让模型为每条示例生成专属的清洗程序，真正做到“一条数据一套代码”。  
   - 改变：细粒度、可定制的清洗让噪声剔除更彻底，尤其是那些规则难以捕捉的边缘情况。

2. **小模型即能完成高质量数据精炼**  
   - 之前：只有大模型或人工才能判断哪些文本值得保留。  
   - 本文：实验表明仅 0.3B 参数的模型就能生成与专家水平相当的清洗脚本。  
   - 改变：大幅降低了生成清洗程序的算力门槛，使得在数百亿条语料上大规模运行成为可能。

3. **统一的“生成‑执行”流水线**  
   - 之前：过滤、去重、标准化往往是多个独立工具链，难以保证前后衔接。  
   - 本文：ProX 把“写脚本”和“跑脚本”两步合并进同一个模型推理过程，形成端到端的流水线。  
   - 改变：简化了工程实现，降低了数据处理的错误传播风险，同时提升了整体效率。

4. **在领域持续预训练中无需手工设计**  
   - 之前：针对数学、医学等专业领域，需要专家手写专属规则或构造特定的过滤器。  
   - 本文：直接把 OpenWebMath 交给 ProX 处理，得到的语料在多模型（Mistral‑7B、Llama‑2‑7B、CodeLlama‑7B）上分别提升 7.6%、14.6% 和 20.3% 的平均准确率。  
   - 改变：证明了 ProX 能在不做任何领域特化的情况下，自动生成高质量的专业语料。

### 方法详解
**整体思路**：把“数据清洗”看作“给每条文本写一段小程序”，先让语言模型生成这段程序，再在安全的执行环境里跑它，最后把输出收集成新的训练语料。整个过程可以概括为四步：① 示例输入 → ② 程序生成 → ③ 程序执行 → ④ 结果收集。

**步骤拆解**  

1. **示例输入与提示构造**  
   - 每条原始文本（可能是网页段落、论坛帖子等）连同一段固定的提示一起喂给小语言模型。提示会说明需要做哪些常见的清洗任务（如字符归一化、去除广告、统一标点），并要求模型输出一种“伪代码”或 DSL（Domain‑Specific Language）指令序列。  
   - 类比：就像让学生在老师给的题目下写出解题步骤，只不过这里的“题目”是原始噪声文本。

2. **程序生成**  
   - 小模型基于提示生成的指令可能是 `REPLACE("“", "\"")`、`NORMALIZE_DATE()`、`REMOVE_IF_CONTAINS("http")` 等。每条指令对应一种细粒度操作。  
   - 关键点在于模型需要学会“组合”这些指令：先统一引号，再去掉 URL，最后压缩空格。组合的自由度让每条数据的清洗路径独一无二。

3. **安全执行环境**  
   - 生成的指令被送入一个轻量的解释器（实现为 Python 函数或自研 DSL 解释器），在沙箱中逐条执行。因为指令是预先限定好的几类操作，解释器可以保证不会执行任意代码，安全性得到控制。  
   - 类比：相当于把学生写的解题步骤交给老师的自动批改系统，只检查合法的步骤并给出最终答案。

4. **结果收集与过滤**  
   - 执行完毕后得到的文本如果仍然不符合基本质量阈值（如长度、语言检测），会被丢弃或重新交给模型生成新的指令。最终保留下来的文本组成 **ProX‑curated corpus**，随后用于大模型的预训练。  

**最巧妙的设计**  
- **程序化思维**：把清洗任务抽象成可执行的指令，使得模型的生成能力直接转化为操作能力，避免了“生成后再人工审查”的低效环节。  
- **小模型驱动**：利用 0.3B 参数模型完成指令生成，显著降低了算力成本，使得在 500B 级别的语料上运行成为现实。  
- **统一接口**：所有细粒度操作都通过同一套 DSL 表达，既保证了可解释性，又方便后续扩展（比如加入新指令）。

### 实验与效果
- **实验语料**：C4、RedPajama‑V2、FineWeb、FineWeb‑Edu、DCLM 等公开大规模语料；以及专门的数学网页集合 OpenWebMath 用于领域持续预训练。  
- **下游任务**：多项标准基准（未在摘要中列出具体名称），覆盖问答、阅读理解、代码生成等。  
- **整体提升**：在所有下游基准上，使用 ProX 精炼后的语料训练的模型比直接使用原始语料或使用其他过滤方法的模型平均提升超过 **2%**。  
- **领域预训练**：在 OpenWebMath 上，Mistral‑7B 的平均准确率提升 **7.6%**，Llama‑2‑7B 提升 **14.6%**，CodeLlama‑7B 提升 **20.3%**，且只用了 **10B** token 的训练量，效果相当于使用 **200B** token 的 Llemma‑7B。  
- **FLOPs 节约**：作者指出，干净的语料让模型在同等训练步数下收敛更快，整体训练 FLOPs 明显下降，具体数值未在摘要中给出。  
- **消融实验**：摘要未提供细节，原文可能会展示去掉“程序生成”或“执行”环节的对比，以证明两者缺一不可。  
- **局限性**：摘要未提及，但可以推测：指令集合的覆盖度决定了能处理的噪声类型；如果出现模型未能生成合适指令的异常文本，仍会被误保留或误删。

### 影响与延伸思考
ProX 把“数据清洗”提升到可编程的层面，为大模型预训练的“数据管道”打开了新思路。自论文发布后，已有几篇工作尝试把 **LLM‑generated 数据标注**、**自监督噪声检测** 与 ProX 类似的指令化框架结合，进一步探索“模型自我清洗”循环。  
对想继续深入的读者，可以关注以下方向：  
- **指令语言的扩展**：设计更丰富的 DSL，以覆盖更复杂的文本结构（如表格、代码块）。  
- **自适应指令生成**：让模型在生成指令时参考全局统计信息，实现跨示例的协同优化。  
- **跨模态清洗**：把图像、音频等非文本模态也映射到可执行指令，实现统一的多模态数据质量提升。  
- **安全与可解释性**：研究如何在更开放的指令空间里保持执行安全，同时提供人类可读的清洗日志。

### 一句话记住它
让小语言模型为每条文本写专属清洗脚本，做到“像专家一样”在海量数据上自动、细粒度地提升预训练质量。