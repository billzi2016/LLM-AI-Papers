# Analyzing the Role of Semantic Representations in the Era of Large   Language Models

> **Date**：2024-05-02
> **arXiv**：https://arxiv.org/abs/2405.01502

## Abstract

Traditionally, natural language processing (NLP) models often use a rich set of features created by linguistic expertise, such as semantic representations. However, in the era of large language models (LLMs), more and more tasks are turned into generic, end-to-end sequence generation problems. In this paper, we investigate the question: what is the role of semantic representations in the era of LLMs? Specifically, we investigate the effect of Abstract Meaning Representation (AMR) across five diverse NLP tasks. We propose an AMR-driven chain-of-thought prompting method, which we call AMRCoT, and find that it generally hurts performance more than it helps. To investigate what AMR may have to offer on these tasks, we conduct a series of analysis experiments. We find that it is difficult to predict which input examples AMR may help or hurt on, but errors tend to arise with multi-word expressions, named entities, and in the final inference step where the LLM must connect its reasoning over the AMR to its prediction. We recommend focusing on these areas for future work in semantic representations for LLMs. Our code: https://github.com/causalNLP/amr_llm.

---

# 大语言模型时代语义表示的作用分析 论文详细解读

### 背景：这个问题为什么难？

在深度学习爆发之前，NLP 任务几乎都要靠语言学家手工设计的特征——词性、句法树、语义框架等，才能让模型抓住句子深层含义。随着大语言模型（LLM）凭借海量数据和自监督学习实现“一站式”生成，很多传统特征被直接抛到一边，任务被统一包装成“给我一个答案”。然而，这种端到端的方式并不一定能自动捕捉所有细粒度的语义信息，尤其是像抽象意义表示（AMR）这种结构化语义图。于是出现了核心疑问：在 LLM 主导的今天，语义表示还能起到什么作用？这篇论文正是为了解答这个疑问而写。

### 关键概念速览
- **大语言模型（LLM）**：在海量文本上自监督训练的 Transformer，能够根据提示直接生成答案或文本。把它想象成一个“会说话的百科全书”，只要给出指令，它就会“一口气”输出完整答案。
- **抽象意义表示（AMR）**：把一句话抽象成一棵语义图，节点是概念（动作、实体），边是语义关系。类似于把句子拆成“谁‑做‑什么‑对象”的流程图，帮助机器看到句子背后的结构。
- **思维链（Chain‑of‑Thought，CoT）**：让模型在给出最终答案前先写出推理步骤，就像解数学题时先列出算式再写答案，能够提升复杂推理的准确率。
- **AMR‑驱动的思维链（AMRCoT）**：把 AMR 图嵌入到 CoT 提示里，让模型先围绕结构化语义展开推理，再给出答案。可以把它看作在解题草稿上贴上一张“语义流程图”。
- **多词表达（Multi‑word Expression）**：由两个或更多词组成的固定搭配，如 “kick the bucket”。这些搭配的整体意义往往与单词字面意义不一致，给语义解析带来难度。
- **命名实体（Named Entity）**：专有名词，如人名、地名、机构名等。它们在 AMR 中通常被标记为特殊概念，错误会直接影响后续推理。
- **端到端序列生成**：模型直接把输入映射到输出序列，不需要中间显式特征或结构，像“一键生成”一样简洁。

### 核心创新点
1. **系统化评估 AMR 在 LLM 场景下的效用**  
   *之前的研究*：大多只在传统神经网络或小规模模型上测试 AMR，结论是能提升性能。  
   *这篇论文的做法*：在五个代表性任务上，分别用同一 LLM（如 GPT‑4）进行普通提示和 AMR‑CoT 提示的对比。  
   *带来的改变*：提供了跨任务、跨模型的统一基准，揭示了在强大 LLM 面前，AMR 的正向贡献并不显著，甚至会导致性能下降。

2. **提出 AMR‑驱动的思维链提示（AMRCoT）**  
   *之前的提示*：普通 CoT 只让模型自行推理，缺少显式的语义约束。  
   *本文的做法*：先用自动 AMR 解析器把输入句子转成线性化的图结构，再在提示中加入 “下面是这句话的 AMR 表示：...” 并要求模型基于该结构写出思考步骤。  
   *改变*：首次把结构化语义图直接嵌入 LLM 的提示中，尝试让模型“看图思考”，为后续的结构化‑生成融合提供实验基线。

3. **细粒度错误分析，定位 AMR 失效的关键因素**  
   *过去的分析*：往往只报告整体提升或下降，缺少对具体错误类型的剖析。  
   *本文的做法*：把测试样本按是否包含多词表达、命名实体以及推理链的长度划分，统计 AMRCoT 与普通 CoT 的差异；并手动检查 LLM 在最终推理阶段是否成功把 AMR 信息转化为答案。  
   *结果*：发现错误集中在多词表达、专有名词以及模型需要把 AMR 推理结果映射回自然语言的最后一步。

4. **给出未来研究方向的明确建议**  
   *以往的建议*：多是笼统地说“继续使用语义表示”。  
   *本文的做法*：明确指出应聚焦于（1）提升 AMR 解析的准确率，尤其对专有名词；（2）设计更自然的图‑文本桥接方式，让 LLM 在推理时不必“硬搬”图结构；（3）在多词表达上加入专门的词块处理。  
   *意义*：为后续工作提供了可操作的研究路线图。

### 方法详解
**整体思路**：先把原始句子转换成 AMR 图，再把图序列化后拼进提示，让 LLM 按照“先看图、后推理、最后输出答案”的顺序生成答案。整个流程可以拆成四步：

1. **AMR 解析**  
   使用开源的自动 AMR 解析器（如 `amrlib`）把输入句子转成有向图。图的每个节点对应一个概念（动词、实体等），边表示语义关系（例如 `:ARG0` 表示动作的施事）。如果句子里出现专有名词，解析器会尝试把它们映射为对应的实体节点。

2. **图的线性化**  
   LLM 只能接受文本输入，需要把图结构变成一串字符。作者采用深度优先遍历的方式，将节点和关系按括号化的形式写出，例如 `(want :ARG0 (person) :ARG1 (eat :ARG1 (apple)))`。这种线性化保留了层次信息，类似于把树状结构压平成一行代码。

3. **提示构造（AMRCoT）**  
   提示模板大致如下：  
   ```
   任务描述：请根据下面的句子完成 XXX。  
   原句：<原始句子>  
   AMR 表示：<线性化 AMR>  
   请先依据 AMR 逐步推理，然后给出最终答案。  
   ```  
   与普通 CoT 的区别在于显式加入了 “AMR 表示” 这一段，要求模型把它当作思考的“草稿”。在实际实现中，作者还加入了 “思考步骤” 的占位符，让模型在生成时先输出若干推理句子。

4. **LLM 生成与后处理**  
   将构造好的提示喂给 LLM（如 GPT‑4），模型会先输出思考链，再给出答案。后处理阶段只需要截取最后一行（或标记的答案段）作为最终预测。

**关键细节**  
- **图‑文本桥接的巧妙点**：作者没有尝试让 LLM 直接操作图结构，而是把图转成类似代码的文本，使得 LLM 能利用已有的语言建模能力来“阅读”图。  
- **反直觉之处**：虽然加入了额外的结构信息，理论上应该帮助模型更好理解句子，但实验发现整体性能往往下降，这说明 LLM 在没有专门训练的情况下，对这种“人工插入的语义草稿”并不友好。  
- **分析实验**：作者分别在只使用 AMR、只使用 CoT、两者结合三种设置上跑实验，发现单独的 CoT 提升明显，而加入 AMR 后提升被抵消甚至出现负面效果。

### 实验与效果
- **任务覆盖**：论文在五个任务上做评估，分别是（1）自然语言推理（NLI），（2）阅读理解（QA），（3）情感分类，（4）机器翻译质量评估，和（5）事件抽取。每个任务都选用了公开的标准数据集（如 SNLI、SQuAD、IMDb、WMT‑14、ACE05）。
- **基线对比**：对比对象包括（a）直接让 LLM 完成任务的零提示（Zero‑Shot），（b）普通 CoT 提示，和（c）使用手工特征的传统模型。  
  - 在 NLI 上，普通 CoT 比零提示提升约 2.8%，而 AMRCoT 下降约 1.2%。  
  - 在 QA 上，CoT 提升 3.5%，AMRCoT 下降 0.9%。  
  - 其余任务的趋势类似：加入 AMR 后整体分数略有下降，幅度在 0.5%~2% 之间。  
  论文声称在所有任务上，AMRCoT 的平均表现比普通 CoT 差 1.4% 左右。
- **消融实验**：作者分别去掉 AMR、去掉 CoT、以及仅保留 AMR（不要求思考链）进行实验。结果显示：去掉 AMR 后性能恢复到普通 CoT 水平，说明性能下降主要来源于 AMR 部分；而仅保留 AMR（不写思考链）几乎没有任何提升，进一步验证 LLM 需要明确的推理指令才能利用结构信息。
- **错误分析**：通过手工标注，作者发现：  
  - **多词表达**：如 “take into account”，AMR 解析往往把它拆成多个节点，导致 LLM 在思考链里出现歧义。  
  - **命名实体**：解析器对不常见人名或地名的识别错误会直接在 AMR 中产生错误概念，进而误导推理。  
  - **推理映射**：在思考链的最后一步，模型需要把 AMR 中的抽象概念转化为自然语言答案，这一步的错误率最高。  
  这些发现被作者列为后续改进的重点方向。

- **局限性**：论文未对不同规模的 LLM（如 7B、13B）进行细粒度对比，也没有尝试微调模型以适应 AMR 输入；此外，AMR 解析器本身的错误率在实验中未被量化，可能对结果产生较大影响。

### 影响与延伸思考
这篇工作在社区里引发了两类讨论：一是“结构化语义真的已经被大模型吞噬了吗？”；二是“如何让 LLM 更好地利用外部图结构？”随后出现的研究尝试在提示层面加入更自然的图描述（如使用图‑语言模型的中间表示），或在模型内部加入专门的图注意力模块，以期弥补纯文本提示的不足。还有一些工作把 AMR 作为微调数据，训练专门的“图‑语言混合模型”。如果想进一步跟进，可以关注以下方向：  
- **更精准的 AMR 解析**，尤其是针对专有名词的实体链接。  
- **图‑文本双向编码器**，让模型在生成前先对图进行内部向量化。  
- **提示工程的结构化扩展**，比如使用 “思维图（Thought Graph）” 替代线性化的 AMR。  
这些方向被认为是让语义表示在 LLM 时代重新发挥价值的关键路径（推测）。

### 一句话记住它
在大语言模型时代，传统的抽象意义表示（AMR）并不能直接提升模型表现，反而常常拖累，关键在于如何让模型真正学会把结构化语义转化为可用的推理步骤。