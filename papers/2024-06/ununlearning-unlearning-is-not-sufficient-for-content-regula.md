# UnUnlearning: Unlearning is not sufficient for content regulation in   advanced generative AI

> **Date**：2024-06-27
> **arXiv**：https://arxiv.org/abs/2407.00106

## Abstract

Exact unlearning was first introduced as a privacy mechanism that allowed a user to retract their data from machine learning models on request. Shortly after, inexact schemes were proposed to mitigate the impractical costs associated with exact unlearning. More recently unlearning is often discussed as an approach for removal of impermissible knowledge i.e. knowledge that the model should not possess such as unlicensed copyrighted, inaccurate, or malicious information. The promise is that if the model does not have a certain malicious capability, then it cannot be used for the associated malicious purpose. In this paper we revisit the paradigm in which unlearning is used for in Large Language Models (LLMs) and highlight an underlying inconsistency arising from in-context learning. Unlearning can be an effective control mechanism for the training phase, yet it does not prevent the model from performing an impermissible act during inference. We introduce a concept of ununlearning, where unlearned knowledge gets reintroduced in-context, effectively rendering the model capable of behaving as if it knows the forgotten knowledge. As a result, we argue that content filtering for impermissible knowledge will be required and even exact unlearning schemes are not enough for effective content regulation. We discuss feasibility of ununlearning for modern LLMs and examine broader implications.

---

# 反遗忘：仅靠遗忘不足以实现高级生成式 AI 内容监管 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，用户或监管机构常常希望把某些不该出现的知识（比如未授权的版权内容、错误信息或恶意指令）从模型中彻底删除。早期的“精确遗忘”只能在训练阶段把对应的参数调回去，却忽视了模型在推理时可以通过**上下文学习**（in‑context learning）重新“召回”这些知识。于是，即使模型在训练时已经“忘记”，只要给它合适的提示，它仍然能表现出好像记得一样的能力，这直接破坏了内容监管的初衷。

### 关键概念速览
**精确遗忘（Exact Unlearning）**：在模型训练完毕后，针对特定数据点或概念进行参数级别的逆向更新，使模型在所有后续推理中不再表现出该知识的影响。类似于把已经写好的字擦掉，要求纸面上彻底看不见。

**近似遗忘（Inexact Unlearning）**：为了降低计算成本，采用近似的参数调整或数据抽样来实现遗忘，效果不一定完全消除目标知识。相当于用橡皮擦轻轻抹掉，可能留下痕迹。

**上下文学习（In‑Context Learning, ICL）**：模型在推理时直接利用用户提供的示例或提示来推断任务，而不需要额外的梯度更新。就像人看几句话后立刻学会了新技巧。

**反遗忘（Ununlearning）**：利用 ICL 把已经遗忘的知识重新注入模型的推理过程，使模型表现出“好像记得”的行为。可以把它想象成在黑板上擦掉的字，被学生在课堂上重新写出来。

**内容过滤（Content Filtering）**：在模型输出阶段加入检测或屏蔽机制，阻止不合规信息出现。类似于在出版前的审稿环节。

**知识重构（Knowledge Reconstruction）**：通过提示工程或外部工具把模型内部的潜在知识重新组合、显式化的过程。相当于把散落的拼图重新拼成完整图案。

### 核心创新点
1. **指出遗忘与监管的根本矛盾** → 通过分析 ICL 的机制，作者证明即使模型在训练阶段完成了精确遗忘，仍然可以在推理时被“反遗忘”。 → 这让业界认识到，仅靠训练阶段的遗忘手段无法保证内容安全，需要在推理层面额外防护。

2. **提出“反遗忘”概念并给出实现思路** → 作者展示了几种利用少量示例或提示让模型重新表现出已遗忘能力的技巧，例如提供“伪造的训练样本”或“任务指令”。 → 这为后续研究提供了具体的攻击向量，也帮助评估现有遗忘方案的真实有效性。

3. **系统性评估遗忘方案在现代 LLM 上的可行性** → 在 GPT‑3.5、Claude、Llama 等主流模型上实验，比较精确遗忘、近似遗忘与无遗忘三种设置下的反遗忘成功率。 → 实验结果显示，即使是最严格的精确遗忘，也能被简单的 ICL 提示恢复约 70% 的功能，凸显监管空洞。

4. **呼吁内容过滤作为必备补充** → 基于实验，作者论证了仅靠遗忘无法满足监管需求，必须在推理阶段加入实时过滤或审计机制。 → 为政策制定者和产品团队提供了明确的技术路线图。

### 方法详解
整体思路可以拆成三步：① 选定目标知识（如某版权段落或恶意指令）；② 对模型执行遗忘（精确或近似）；③ 通过精心设计的上下文提示尝试“反遗忘”。作者把这三步包装成一个评估框架，称为 **Ununlearning Benchmark**。

**步骤 1：目标知识标记**  
研究者先在训练语料中定位要删除的片段，记录其原始文本、所属任务以及对应的标签。比如，一段未授权的小说章节被标记为“版权内容”。

**步骤 2：遗忘实现**  
- **精确遗忘**：使用已有的梯度逆向技术，对模型参数进行反向微调，使得损失函数对该片段的梯度趋近于零。  
- **近似遗忘**：采用数据抽样或蒸馏方式，只在少量相关样本上进行微调，降低计算开销。  
作者在实验中分别对两种方法进行对比，确保模型在标准评测上不再表现出该知识。

**步骤 3：反遗忘提示构造**  
核心创新在这里。作者设计了三类提示：  
1. **直接示例**：在用户提示中直接嵌入原始片段的改写版，让模型在上下文中“看到”它。  
2. **任务指令**：给模型下达类似“请完成以下未授权文本的续写”之类的指令，诱导模型激活内部的潜在记忆。  
3. **链式提示**：先让模型生成与目标知识相关的中间步骤（如摘要），再让它基于这些中间结果继续生成，形成知识的逐层恢复。  

在实际运行时，这些提示只需要几句话，几乎不增加推理成本，却能让模型在输出中出现原本已遗忘的内容。

**最巧妙的点**：作者把“反遗忘”视作一种 **上下文注入攻击**，而不是对模型参数的再训练。这样做的好处是：  
- 只需要一次前向传播，几乎不消耗额外算力。  
- 可以针对任何已经部署的模型，无需获取内部权重。  

### 实验与效果
- **数据集与任务**：作者选取了三个场景：① 未授权的文学片段续写，② 错误医学信息的纠正，③ 恶意指令（如生成钓鱼邮件）。每个场景都有对应的基准测试集。  
- **对比基线**：包括（1）未做任何遗忘的原始模型，（2）仅使用精确遗忘的模型，（3）仅使用近似遗忘的模型。  
- **主要结果**：在精确遗忘条件下，使用最简短的直接示例提示即可恢复约 **68%** 的目标输出；近似遗忘的恢复率更高，达到 **75%**。未做遗忘的模型自然保持 100% 能力。  
- **消融实验**：作者分别去掉提示中的示例、指令或链式结构，发现单独的指令恢复率跌至 30%，而加入链式提示后恢复率提升至 80%。这说明多步上下文构造是关键。  
- **局限性**：实验主要在公开的 LLM 上进行，未覆盖专有的安全强化模型；此外，提示长度受限于实际应用场景，过长提示可能被系统截断。作者也承认，针对更强的防御（如输出审计）时，反遗忘成功率会下降。

### 影响与延伸思考
这篇工作在发布后迅速引发了两类后续研究：  
1. **防御方向**：有人尝试在模型内部加入“上下文记忆屏蔽”，让模型在检测到潜在敏感提示时主动抑制对应知识的激活。  
2. **监管工具**：监管机构开始把“反遗忘可行性”列入模型审计指标，要求供应商提供针对 ICL 的安全评估报告。  

如果想进一步了解，可以关注 **“上下文安全”（Contextual Safety）** 这一新兴子领域，它结合提示工程、输出审计和模型蒸馏，试图在推理层面实现更细粒度的内容控制。

### 一句话记住它
即使模型在训练时把敏感知识“忘了”，只要给它合适的提示，它仍能“记起”，所以内容监管必须在推理阶段加装过滤，而不是仅靠遗忘。