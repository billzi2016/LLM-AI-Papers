# Re-Reading Improves Reasoning in Large Language Models

> **Date**：2023-09-12
> **arXiv**：https://arxiv.org/abs/2309.06275

## Abstract

To enhance the reasoning capabilities of off-the-shelf Large Language Models (LLMs), we introduce a simple, yet general and effective prompting method, Re2, i.e., \textbf{Re}-\textbf{Re}ading the question as input. Unlike most thought-eliciting prompting methods, such as Chain-of-Thought (CoT), which aim to elicit the reasoning process in the output, Re2 shifts the focus to the input by processing questions twice, thereby enhancing the understanding process. Consequently, Re2 demonstrates strong generality and compatibility with most thought-eliciting prompting methods, including CoT. Crucially, Re2 facilitates a "bidirectional" encoding in unidirectional decoder-only LLMs because the first pass could provide global information for the second pass. We begin with a preliminary empirical study as the foundation of Re2, illustrating its potential to enable "bidirectional" attention mechanisms. We then evaluate Re2 on extensive reasoning benchmarks across 14 datasets, spanning 112 experiments, to validate its effectiveness and generality. Our findings indicate that, with the exception of a few scenarios on vanilla ChatGPT, Re2 consistently enhances the reasoning performance of LLMs through a simple re-reading strategy. Further analyses reveal Re2's adaptability, showing how it can be effectively integrated with different LLMs, thought-eliciting prompting, and ensemble strategies. Our code is available at \url{https://github.com/Tebmer/Rereading-LLM-Reasoning/}

---

# 重读提升大语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在直接回答事实类问题时已经相当强大，但在需要多步推理的场景——比如数学计算、逻辑谜题或跨段落的阅读理解——仍会出现“跳步”或“漏掉关键信息”。传统的提示（prompt）往往只在输出端让模型写出思考过程（如 CoT），却没有系统地帮助模型在输入阶段更充分地捕捉全局信息。于是模型在一次阅读后就进入生成，缺少对问题的二次审视，导致推理链条不够稳固。解决这个瓶颈的关键在于：如何让模型在不改变模型结构的前提下，获得类似“双向注意”（bidirectional attention）的优势。

### 关键概念速览
**LLM（大语言模型）**：基于海量文本训练的自回归模型，常见的实现是只看左侧已生成内容的解码器。  
**CoT（思维链）**：让模型在给出答案前先把推理步骤写出来，类似于人在解题时先列出草稿。  
**Prompt（提示）**：向模型提供的文字指令或上下文，用来引导模型产生期望的输出。  
**单向解码器**：只能从左到右逐词生成的模型，天然缺少对后文的直接感知。  
**双向注意**：模型在处理某个位置时，同时考虑左侧和右侧信息，类似于人阅读时可以回头检查前文。  
**Re2（重读）**：把同一道题目在同一次交互中出现两遍，让模型在第二遍时已经拥有第一次阅读的全局上下文。  
**Ensemble（集成）**：把多个模型或多次推理的结果合并，以期获得更稳健的答案。

### 核心创新点
1. **从输出转向输入的思考激发**：之前的思维链方法主要让模型在答案前写出推理步骤，而 Re2 把“思考”搬到输入端，通过让模型先读一遍问题再读第二遍，使得第二遍的注意力能够利用第一次已经形成的全局信息。这样即使是单向解码器，也能在第二轮“看到”后面的内容。  
2. **通用性与兼容性**：Re2 并不是一个独立的推理框架，而是一个可以叠加在任何已有提示上的插件。实验表明，无论是纯粹的零提示、CoT、还是更复杂的自洽提示（self‑consistency），把问题重复一次都能带来提升。  
3. **“双向”注意的隐式实现**：在一次交互中，模型的第一遍阅读生成了完整的上下文向量，这些向量在第二遍作为输入的一部分被再次消费。于是模型在第二遍的自注意力层里，实际上能够访问到全局信息，模拟了双向注意的效果，而不需要改动模型内部结构。  
4. **轻量级实现**：只需在提示里把原始问题复制一遍，或在 API 调用时把同一段文字拼接两次，无需额外的模型微调、额外的计算图或专门的训练数据，几乎可以“一键”开启。

### 方法详解
**整体思路**：Re2 的核心操作只有一步——把同一道题目在同一次输入中出现两次。具体流程如下：

1. **构造原始提示**：根据任务选择合适的基线提示，例如直接提问、CoT 提示或其他思维链模板。  
2. **重读拼接**：把构造好的提示中的问题文本复制一遍，放在原问题之后。常见的写法是 `Question: <原问题>\nQuestion: <原问题>`，或者在 CoT 场景下 `Question: <原问题>\nAnswer: Let's think step by step.\n... \nQuestion: <原问题>`。  
3. **一次性送入模型**：将完整的拼接文本一次性喂给解码器模型，让模型从头到尾生成答案。模型在生成过程中会先处理第一遍的内容，形成内部的上下文表示；随后在第二遍出现时，这些表示已经在注意力缓存中，可被再次引用。  
4. **输出解析**：模型的最终输出仍然是答案或思维链，和普通提示没有区别。只是在内部的注意力流动上，多了一层“回看”机制。

**类比**：想象你在阅读一道数学题时，先快速浏览一遍获取整体结构，然后再回到题目开头重新审视细节，这样可以发现之前忽略的条件。Re2 正是把这种“先读后读”的学习策略搬进了语言模型的交互流程。

**关键细节**：
- **位置编码**：因为模型是自回归的，第二遍的文本仍然拥有递增的位置编码，模型不会把它当成重复的噪声，而是把它当作后续的上下文。  
- **注意力缓存**：大多数实现会在生成时缓存前面的键值对，第二遍的注意力查询可以直接访问这些缓存，实现信息的跨段传播。  
- **兼容多种提示**：在 CoT 场景，只需要把思考链的开头保留，后面再加一次完整问题即可；在自洽（self‑consistency）等集成方法中，仍然可以在每一次采样前使用重读。  
- **最巧妙的点**：不需要改动模型的内部结构，也不需要额外的训练，只是利用了模型本身的自注意力机制和缓存特性，让一次前向传播就相当于两次阅读。

### 实验与效果
- **评测范围**：作者在 14 个公开的推理基准上做了实验，涵盖数学、逻辑、常识和多步阅读理解等任务，总计 112 组实验组合。  
- **对比基线**：包括直接零提示、标准 CoT、以及更高级的自洽集成方法。  
- **提升幅度**：论文声称，除少数在原始 ChatGPT（vanilla ChatGPT）上表现不佳的情况外，Re2 在几乎所有设置下都提升了模型的正确率。具体提升幅度在不同数据集上从几百分点到两位数不等。  
- **消融实验**：作者分别去掉第一次阅读、只保留第二遍、以及在不同位置插入重读，结果显示：只有完整的两遍顺序才能最大化提升，说明“先读后读”的顺序是关键。  
- **兼容性验证**：把 Re2 与 CoT、Self‑Consistency、以及不同规模的 LLM（如 GPT‑3.5、Claude、LLaMA 系列）组合后，均观察到正向增益，证明方法的通用性。  
- **局限性**：在某些对话式模型（如原生 ChatGPT）上，重读有时会导致生成冗长或重复的答案，作者把这归因于模型的对话历史管理方式。除此之外，方法本身对计算成本几乎没有影响，因为只增加了输入长度。

### 影响与延伸思考
这篇工作把“输入层面的二次阅读”推向前台，提醒社区：提升推理不一定要在输出端加花样，也可以在输入端做结构化的安排。随后出现的几篇论文尝试把问题拆分成多段、或在同一次交互中加入“回顾”指令，都是对 Re2 思路的延伸。还有研究把重读与检索增强（retrieval‑augmented）结合，让模型在第二遍阅读时加入外部文档的摘要，进一步提升事实推理的准确性。想继续深入，可以关注以下方向：  
- **动态重读**：让模型自行决定是否需要第二遍以及何时重读，而不是固定两遍。  
- **跨模态重读**：在视觉‑语言任务中，把图像描述先读一次，再在文字提示中重复，以实现跨模态的双向注意。  
- **训练层面的重读**：在微调阶段显式加入重读步骤，观察是否能让模型内部形成更稳固的推理表征。

### 一句话记住它
让大语言模型先读一次再读一次，同样的题目就能在不改模型的情况下获得“双向注意”，显著提升推理能力。