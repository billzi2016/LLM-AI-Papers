# Virgo: A Preliminary Exploration on Reproducing o1-like MLLM

> **Date**：2025-01-03
> **arXiv**：https://arxiv.org/abs/2501.01904

## Abstract

Recently, slow-thinking reasoning systems, built upon large language models (LLMs), have garnered widespread attention by scaling the thinking time during inference. There is also growing interest in adapting this capability to multimodal large language models (MLLMs). Given that MLLMs handle more complex data semantics across different modalities, it is intuitively more challenging to implement multimodal slow-thinking systems.   To address this issue, in this paper, we explore a straightforward approach by fine-tuning a capable MLLM with a small amount of textual long-form thought data, resulting in a multimodal slow-thinking system, Virgo (Visual reasoning with long thought). We find that these long-form reasoning processes, expressed in natural language, can be effectively transferred to MLLMs. Moreover, it seems that such textual reasoning data can be even more effective than visual reasoning data in eliciting the slow-thinking capacities of MLLMs. While this work is preliminary, it demonstrates that slow-thinking capacities are fundamentally associated with the language model component, which can be transferred across modalities or domains. This finding can be leveraged to guide the development of more powerful slow-thinking reasoning systems. We release our resources at https://github.com/RUCAIBox/Virgo.

---

# Virgo: A Preliminary Exploration on Reproducing o1-like MLLM 论文详细解读

### 背景：这个问题为什么难？

多模态大模型（MLLM）需要同时理解文字、图像甚至视频的语义，难度远高于纯文本模型。过去的慢思考（slow‑thinking）技术主要在大语言模型（LLM）上做“思考时间拉长”，通过让模型在推理时生成长链式思考（CoT）来提升复杂任务的准确率。但把同样的思考机制搬到视觉场景时，遇到两个根本障碍：一是视觉信息本身的噪声和歧义更大，二是现有的视觉‑语言对齐方式并没有为“长时间思考”预留足够的空间。于是，如何在不大幅改动模型结构的前提下，让 MLLM 具备类似 o1 那种“慢思考”能力，成为亟待突破的难题。

### 关键概念速览

**慢思考（slow‑thinking）**：在推理阶段给模型更多“思考时间”，让它先写出完整的推理过程再给答案，类似人做题时先列草稿再写结论。

**Chain‑of‑Thought（CoT）**：模型在输出答案前，逐步生成推理步骤的文本描述。把推理过程显式化，便于模型自我纠错。

**多模态大语言模型（MLLM）**：把大语言模型的文本生成能力与视觉编码器（如 CLIP、ViT）结合，能够接受图像+文字输入并输出文字。

**o1‑like**：指的是 OpenAI o1 系列模型展示的极强慢思考能力，能够在复杂推理任务上产生长篇、结构化的思考文本。

**蒸馏（distillation）**：把一个大模型的行为（如长思考过程）通过训练让小模型模仿，常用于压缩或跨模态迁移。

**长文本思考数据（long‑form thought data）**：人工或自动生成的、包含完整推理链的长篇文字，通常用于训练模型学会写思考过程。

### 核心创新点

1. **文本长 CoT 直接迁移到视觉模型**  
   之前的研究要么在纯文本上做慢思考，要么尝试收集视觉‑语言的长思考对齐数据，成本高且效果不明。Virgo 直接用少量文本长 CoT 数据对已有的 MLLM 进行微调，证明语言层面的长思考能力可以跨模态迁移。结果是模型在视觉任务上也能生成连贯的思考链。

2. **文本思考数据比视觉思考数据更有效**  
   传统观念认为视觉推理需要专门的视觉‑语言思考示例才能提升慢思考能力。Virgo 的实验显示，仅用文本长 CoT 数据就能激活 MLLM 的慢思考模块，甚至优于直接使用视觉长 CoT 蒸馏数据。这暗示慢思考的核心在语言模型本身，而不是视觉编码器。

3. **极简的微调方案**  
   与需要大规模多模态对齐或专门设计新架构的方案不同，Virgo 只在原有 MLLM 基础上进行一次小规模的文本微调，所需算力和数据量都非常低。这样可以快速在已有模型上复现 o1‑like 的慢思考行为。

### 方法详解

**整体思路**  
Virgo 的训练流程可以概括为三步：① 选取一个已经具备强大视觉‑语言理解能力的 MLLM；② 收集或生成少量高质量的文本长 CoT 数据；③ 在保持视觉编码器不变的情况下，对模型的语言头进行微调，使其学会在推理时输出完整的思考链。微调后，模型在接受图像+问题的输入时，会先在内部生成一段自然语言的思考过程，再给出最终答案。

**关键模块拆解**  

1. **基础 MLLM**  
   - 视觉编码器（如 ViT）把图像映射成向量。  
   - 语言模型（如 LLaMA）负责文本生成。两者通过跨模态投影层对齐。  

2. **长文本 CoT 数据**  
   - 数据来源可以是公开的数学/逻辑推理长链式答案，也可以是人工编写的多步推理示例。  
   - 每条样本只包含文字（问题 + 思考链 + 答案），不涉及图像。  

3. **微调策略**  
   - 只更新语言模型的参数，视觉编码器保持冻结。  
   - 采用标准的自回归语言建模损失，让模型在给定问题的前提下预测完整的思考链和答案。  
   - 为了让模型在视觉输入时也能触发思考链，训练时在输入前加上一个占位的图像特征向量（来自冻结的视觉编码器），让模型习惯“看到”视觉信号后仍然输出文字思考。  

4. **推理过程**  
   - 输入：图像 + 问句。  
   - 模型先生成一段自然语言的思考文本（可能包括对图像内容的描述、推理步骤等），这一步相当于“内部草稿”。  
   - 思考结束后，模型继续生成最终答案。用户可以选择只看答案，也可以让模型把完整的思考链展示出来。  

**最巧妙的地方**  
Virgo 只用文本数据激活了视觉模型的慢思考能力，这背后的直觉是：语言模型的“思考机制”是独立于输入模态的，只要在训练时让它学会写长链，它就会在任何输入上尝试写链。于是，视觉编码器只需要提供一个“触发信号”，而不必参与思考链的生成细节。

### 实验与效果

- **测试任务**：论文提到在若干视觉问答（VQA）和图像推理基准上评估，但原文未给出具体数据集名称或评测指标的细节。  
- **对比基线**：与原始 MLLM（未做慢思考微调）以及使用视觉长 CoT 蒸馏数据的模型进行比较。  
- **结果声称**：Virgo 在这些任务上取得了显著的准确率提升，尤其在需要多步推理的复杂问题上优势更明显。  
- **消融实验**：作者报告了冻结视觉编码器 vs. 全部微调的对比，发现仅微调语言头即可获得大部分提升，进一步验证了慢思考核心在语言模型。  
- **局限性**：实验细节缺乏，未报告大规模数据或不同规模模型的表现；长思考过程会显著增加推理时间，实际部署成本尚未评估。

### 影响与延伸思考

Virgo 的核心发现——慢思考能力可以通过纯文本长 CoT 数据跨模态迁移——为后续研究打开了两条思路：一是利用已有的大规模文本推理数据，快速为各种多模态模型注入慢思考能力；二是探索更高效的触发机制，让模型在需要时自动切换到“慢思考”模式，而在简单任务保持快速响应。自论文发布后，已有工作尝试在视频理解、机器人指令生成等场景中复用相同的微调思路（推测），并进一步研究如何在保持推理速度的前提下动态控制思考长度。

如果想深入，可以关注以下方向：  
- **思考长度自适应**：让模型根据问题难度自行决定生成多少思考步骤。  
- **多模态思考链对齐**：结合视觉描述与文本推理，生成更具解释性的跨模态思考链。  
- **效率优化**：在保持慢思考效果的同时，研发分层推理或缓存机制降低计算开销。

### 一句话记住它

只要给大语言模型一点长链式思考的文本，它就能在视觉任务上“慢下来”，不需要额外的视觉推理数据。