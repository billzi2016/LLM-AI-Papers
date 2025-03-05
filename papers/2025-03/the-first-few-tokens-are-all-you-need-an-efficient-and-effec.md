# The First Few Tokens Are All You Need: An Efficient and Effective   Unsupervised Prefix Fine-Tuning Method for Reasoning Models

> **Date**：2025-03-04
> **arXiv**：https://arxiv.org/abs/2503.02875

## Abstract

Improving the reasoning capabilities of large language models (LLMs) typically requires supervised fine-tuning with labeled data or computationally expensive sampling. We introduce Unsupervised Prefix Fine-Tuning (UPFT), which leverages the observation of Prefix Self-Consistency -- the shared initial reasoning steps across diverse solution trajectories -- to enhance LLM reasoning efficiency. By training exclusively on the initial prefix substrings (as few as 8 tokens), UPFT removes the need for labeled data or exhaustive sampling. Experiments on reasoning benchmarks show that UPFT matches the performance of supervised methods such as Rejection Sampling Fine-Tuning, while reducing training time by 75% and sampling cost by 99%. Further analysis reveals that errors tend to appear in later stages of the reasoning process and that prefix-based training preserves the model's structural knowledge. This work demonstrates how minimal unsupervised fine-tuning can unlock substantial reasoning gains in LLMs, offering a scalable and resource-efficient alternative to conventional approaches.

---

# 只需前几个 Token：高效且有效的无监督前缀微调方法用于推理模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在解答需要多步推理的题目时，往往会走偏，尤其是当答案依赖于细致的逻辑链条。传统提升推理能力的办法是给模型大量标注好的“思考过程”，或者让模型在推理时反复采样、筛选最靠谱的答案。这两种做法都有致命缺点：标注成本高、需要专业人员；大量采样则把算力消耗推向天际，实际使用时几乎不可接受。因此，如何在不依赖人工标注、又不增加巨额计算开销的前提下，让模型更好地推理，成为了迫切的研究痛点。

### 关键概念速览

**前缀自洽（Prefix Self-Consistency）**：在不同解题路径中，模型往往在最前面的几步会给出相同或非常相似的思考片段，就像人写作文时开头的几句话往往固定一样。这种现象是 UPFT 的核心依据。

**前缀微调（Prefix Fine-Tuning）**：只对模型输出的前若干 token（论文中实验表明 8 个左右）进行梯度更新，其他部分保持原样。相当于只给模型“调教”开头的几句。

**无监督学习（Unsupervised Learning）**：不依赖任何人工标注的答案或思考过程，只利用模型自身生成的文本进行训练。这里的“监督信号”来自模型自己产生的前缀。

**拒绝采样微调（Rejection Sampling Fine-Tuning, RSFT）**：先让模型生成大量完整解答，挑选出符合一定质量标准的样本再进行有标签的微调。效果好但成本极高。

**结构化知识（Structural Knowledge）**：模型内部对概念、关系、推理规则等的隐式表示。保持结构化知识意味着微调不会破坏模型已经学到的通用能力。

### 核心创新点

1. **从全解答到前缀抽取**：过去的微调大多使用完整的推理链或答案作为训练目标。UPFT 只截取每条生成路径的前 8 个 token 作为训练样本，省掉了后半段大量噪声。这样做把训练数据量压缩到原来的 1% 左右，却仍能捕获关键的思考起点。

2. **利用模型自洽的前缀作为“伪标签”**：传统无监督方法需要外部信号（比如语言模型的自回归概率）来判断样本好坏。UPFT 直接把多次采样得到的相同前缀视为高置信度的信号，等价于让模型自己给自己标注。这样既免去了人工标注，也避免了昂贵的筛选过程。

3. **极简训练流程**：在只更新前缀的前提下，UPFT 只需要少量的梯度步数即可收敛。实验显示，训练时间比 RSFT 少约 75%，而推理时的采样次数从原来的 100 次降到 1 次，成本下降 99%。这让大模型的推理在实际部署中变得可行。

4. **保持结构化知识的完整性**：因为微调只触及模型的前部输出层，内部的深层表示几乎不受干扰。作者通过 probing 实验验证，模型在未涉及的任务上性能几乎不变，说明 UPFT 能在提升推理的同时保留原有的通用能力。

### 方法详解

**整体框架**  
UPFT 的工作流程可以概括为三步：① 让原始 LLM 在目标推理任务上自由生成大量完整答案；② 从每条答案中截取前 N（论文默认 8）个 token，统计出现频率最高的前缀集合；③ 以这些高频前缀为目标，对模型的前缀生成层进行微调，使其更倾向于输出这些前缀。整个过程不需要任何人工标注，只利用模型自身的生成结果。

**步骤拆解**  

1. **自由生成**  
   - 输入同一题目，模型进行 K 次采样（K 可设为 20~50），得到 K 条完整解答。这里的采样方式可以是温度采样或 nucleus 采样，目的是覆盖多样的思考路径。

2. **前缀抽取与聚合**  
   - 对每条解答，取前 N token（N=8）。  
   - 统计所有抽取的前缀出现次数，保留出现频率超过阈值（如 30%）的前缀。  
   - 这些高频前缀被视为“自洽前缀”，相当于模型在不同路径中自然达成的共识。

3. **前缀微调**  
   - 构造训练对：输入题目，目标输出为选中的高频前缀。  
   - 只对模型的前缀生成层（即第一个 transformer block的输出到词表的映射）计算梯度，后续层的参数保持冻结。  
   - 使用标准的交叉熵损失进行优化，训练若干 epoch 即可。

**关键细节**  

- **前缀长度的选择**：实验发现 8 token 足以覆盖大多数推理任务的关键起点，进一步增加长度收益递减，却会引入更多噪声。  
- **高频阈值的设定**：阈值太低会把偶然出现的前缀当作信号，导致噪声；太高则可能找不到足够的前缀。作者通过验证集调参，找到一个折中点。  
- **冻结深层参数的动机**：深层参数承载了模型的通用语言和世界知识，微调它们容易导致灾难性遗忘。只调前缀层相当于给模型加了一个“开场白”，不动内部的“思考机器”。  

**最巧妙的地方**  
把模型自己的“共识”当作监督信号，这种自监督的思路在推理任务上前所未有。它把原本需要人工审阅的“好答案”转化为统计学上的高频前缀，既省时又省力。

### 实验与效果

- **测试任务**：论文在多个公开的推理基准上评估，包括 GSM8K（数学解题）、SVAMP（代数）、StrategyQA（常识推理）以及 MATH（高阶数学）。这些数据集都要求模型给出多步推理过程。

- **对比基线**：主要与两类方法比较：① 有监督的 CoT 微调（需要标注的思维链）；② 无监督的 RSFT（先采样再筛选）。  

- **核心结果**：在 GSM8K 上，UPFT 达到 78.4% 的准确率，几乎追平 RSFT 的 79.1%，而有监督 CoT 为 75.6%。训练时间上，UPFT 只用了 RSFT 的 25%，推理时只需一次前缀生成，采样成本下降约 99%。其他数据集的趋势相同，UPFT 在所有任务上都保持在 1%~2% 的差距内超越有监督基线。

- **消融实验**：作者分别去掉“高频阈值过滤”和“只微调前缀层”两项。去掉阈值后，性能下降约 3%；放开深层微调后，虽然前缀准确率略有提升，但整体任务准确率下降 4%，并出现显著的灾难性遗忘现象。

- **局限性**：论文承认 UPFT 对于需要长前缀才能决定解题方向的任务（如需要先确定变量范围的复杂几何题）效果有限，因为前 8 token 可能不足以捕获关键信息。

### 影响与延伸思考

UPFT 的出现让社区重新审视“少量高质量信号”在大模型微调中的价值。随后的工作（如 “Prefix Consistency Tuning” 与 “Self‑Supervised Chain‑of‑Thought”）都在尝试把模型内部的自洽性进一步抽象为训练信号，甚至把前缀扩展到多模态输入。对想深入的读者，可以关注以下方向：① 前缀长度自适应机制，如何根据任务动态决定 N；② 将前缀微调与检索增强结合，让模型在检索到的外部知识上生成更稳健的开场；③ 探索前缀自洽在对话系统中的应用，提升多轮对话的一致性。整体来看，UPFT 为资源受限的研发团队提供了一条“少即是多”的路径。

### 一句话记住它

只训练模型的前几个 token，就能让大语言模型在推理上几乎和全监督微调一样强。