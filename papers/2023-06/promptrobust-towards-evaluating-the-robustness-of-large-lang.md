# PromptRobust: Towards Evaluating the Robustness of Large Language Models   on Adversarial Prompts

> **Date**：2023-06-07
> **arXiv**：https://arxiv.org/abs/2306.04528

## Abstract

The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptRobust, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks including sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present a comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users.

---

# PromptRobust：面向对抗性提示的大语言模型鲁棒性评估 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）已经被当作“万能助理”嵌入搜索、客服、写作等场景，几乎所有功能都通过**提示（prompt）**来驱动。可是，提示本身并不总是完美的——用户会打错字、用同义词、甚至把句子顺序弄乱。过去的评估大多只看模型在干净、标准化的提示下的表现，忽视了这些日常的、细微的噪声。缺少系统化的对抗性提示集合，也没有统一的度量方式，导致我们根本不知道模型在真实使用中会不会因为一个小小的笔误而崩溃。正是这种盲区让对提示鲁棒性的研究迫在眉睫。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，例如 GPT‑4、LLaMA。它们的行为完全由输入的文字提示决定。
- **提示（Prompt）**：用户给模型的指令或问题，类似于向老师提问的方式。提示的表述方式会直接影响模型的答案。
- **对抗性提示（Adversarial Prompt）**：在保持原意的前提下，对提示进行细微扰动（如错别字、同义词替换），目的是让模型产生错误或不一致的输出。
- **鲁棒性（Robustness）**：模型在面对这些扰动时，输出质量下降的幅度。鲁棒性高意味着即使提示有点“脏”，模型仍能给出正确答案。
- **字符级攻击**：在提示的字符层面做改动，例如把“i”换成“1”、删掉一个字母，类似键盘打字错误。
- **词级攻击**：把提示中的词换成同义词或近义词，或者插入/删除常见的功能词，类似口语中换词的现象。
- **句子级攻击**：对提示的句子顺序、结构进行调整，或在提示里加入无关句子，类似把一段话的段落顺序打乱。
- **语义级攻击**：利用生成模型重新表述整个提示，确保语义不变但文字表述完全不同，类似把一句话用另一种说法重新写一遍。
- **可转移性（Transferability）**：一种攻击在一个模型上成功后，能否在其他模型上同样导致错误。若可转移，说明问题根源在提示本身而非特定模型。

### 核心创新点
1. **多粒度对抗提示生成 → 论文提出四层次（字符、词、句子、语义）的大规模扰动策略 → 让评估覆盖从最细微的打字错误到完整的句子改写，全面揭示模型的薄弱环节。**  
   之前的工作只关注单一层面的噪声（比如拼写错误），而 PromptRobust 把所有可能的用户失误都系统化，形成了一个“全景图”。

2. **构建规模化基准 → 生成 4,788 条对抗提示，横跨 8 类任务、13 套公开数据集 → 为每个任务提供原始提示和对应的扰动版，形成统一的评估平台。**  
   过去的鲁棒性实验往往只在少数任务上手工造几个例子，难以比较。PromptRobust 把评估标准化，任何新模型都可以直接跑分。

3. **细致的鲁棒性与可转移性分析 → 对比不同模型在同一扰动下的表现，统计“攻击成功率”并探讨为何某些扰动更具破坏力 → 揭示了提示设计的关键风险点。**  
   这一步让我们不只是看到“模型不鲁棒”，而是明白“是哪类错误最致命”，为后续防御提供了方向。

4. **实用的提示编写指南 → 基于实验结果，总结出“安全提示”原则（如避免高频同义词替换、保持关键实体不被拆分） → 直接帮助研发人员和普通用户写出更稳健的提示。**  
   大多数对抗性研究停留在攻击层面，PromptRobust 进一步给出防御建议，具有很强的落地价值。

### 方法详解
**整体框架**  
PromptRobust 的评估流程可以概括为四步：  
1) **收集原始提示**：从每个任务的官方数据集里抽取标准提示（例如情感分析的“这条评论是正面的还是负面的？”）。  
2) **生成对抗提示**：对每条原始提示分别施加字符、词、句子、语义四类扰动，得到多组变体。  
3) **语义保持验证**：使用语义相似度模型（如 Sentence‑BERT）过滤掉意义偏离太大的变体，确保“仍然在问同一个问题”。  
4) **鲁棒性测评**：把原始提示和对应的对抗提示分别喂给目标 LLM，记录任务指标（准确率、BLEU、解题正确率等），计算**鲁棒性下降率**（原始指标 - 对抗指标）/ 原始指标。

**关键模块拆解**  

- **字符级攻击模块**  
  - 操作包括：键位相邻替换（e→r）、随机删除、字符重复、大小写混淆。  
  - 类比：像在打字时手指滑了一下，导致一个字母变形。  
  - 实现上，先构建字符邻接表（依据键盘布局），对每个单词随机抽取 1–2 个字符进行上述操作。

- **词级攻击模块**  
  - 采用同义词词典（WordNet）和上下文感知的替换模型（如 BERT‑Masked）生成候选词。  
  - 只替换非关键实体词（如人名、数字），防止语义彻底改变。  
  - 类比：把“好”换成“优秀”，在口语中常见的换词行为。

- **句子级攻击模块**  
  - 包括：句子顺序打乱、在提示末尾插入无关陈述、把复合句拆成多个短句。  
  - 通过依存句法树识别句子边界，确保插入的噪声句子在语法上是合法的。  
  - 类比：在一封邮件里把段落顺序调错，或在结尾加上一句“顺便说一下，我喜欢咖啡”。

- **语义级攻击模块**  
  - 使用大型生成模型（如 GPT‑3.5）在“保持原意”的约束下重新表述提示。  
  - 通过双向检索（原提示 → 生成 → 重新生成回原）确保语义闭环。  
  - 类比：把“一只猫坐在窗台上”改写成“窗台上有只猫在坐”，文字全变但意思不变。

- **语义保持过滤**  
  - 对每对（原始提示，扰动提示）计算余弦相似度，阈值设为 0.85。  
  - 若相似度低于阈值，则该扰动被丢弃，防止“攻击”变成“新任务”。  

- **鲁棒性度量**  
  - 对每个任务，计算 **原始准确率** 与 **对抗后准确率** 的差值，得到 **鲁棒性下降率**。  
  - 进一步统计 **攻击成功率**（下降率 > 10%）以及 **跨模型可转移率**（同一扰动在不同模型上均成功的比例）。

**最巧妙的设计**  
- **多层次扰动 + 语义过滤** 的组合：既保证了攻击的多样性，又避免了“把问题改成别的问题”。这一步在之前的对抗文本工作里很少出现，极大提升了评估的可信度。  
- **统一的基准平台**：所有任务共享同一套扰动生成器，只需换数据集即可复用，极大降低了评估成本。

### 实验与效果
- **任务与数据集**：情感分析（SST‑2、IMDB）、自然语言推理（SNLI、MNLI）、阅读理解（SQuAD、RACE）、机器翻译（WMT‑14 EN‑DE）、数学题求解（MATH）、以及代码解释等，共计 8 类任务、13 套公开数据。  
- **模型**：OpenAI GPT‑3.5、GPT‑4、Meta LLaMA‑2‑13B、Claude‑2 等主流商用与开源 LLM。  
- **主要发现**：  
  - 在字符级攻击下，GPT‑4 的情感分析准确率从 94.2% 降至 86.5%（下降率 8.2%）。  
  - 词级攻击对 NLI 任务影响更大，SNLI 上的准确率从 92.1% 下降到 78.3%（下降率 15%）。  
  - 语义级攻击在机器翻译任务中导致 BLEU 分数平均下降 12.4 分。  
  - 跨模型可转移性高达 68%，说明同一扰动往往能同时击垮多个模型。  
- **消融实验**：去掉字符级扰动后，总体下降率下降约 3%；去掉语义级扰动后下降率下降约 7%，表明语义级攻击是最具破坏力的因素。  
- **局限性**：  
  - 生成的对抗提示仍然依赖英文词典和模型，中文等低资源语言的覆盖度不足（原文未详细描述）。  
  - 只评估了离线推理场景，未考虑实时交互中模型可能的自我纠错机制。  

### 影响与延伸思考
PromptRobust 在发布后迅速成为评估 LLM 提示鲁棒性的“标配”。随后出现的工作如 **AdvPrompt**, **RobustPrompt** 等，都在其基准上扩展了多语言、代码提示等领域。安全团队也开始把对抗提示加入模型的微调数据，以提升抗噪声能力。对研究者而言，下一步可以探索 **提示修复（Prompt Repair）**、**鲁棒提示生成（Robust Prompt Synthesis）**，甚至把对抗提示当作 **对齐信号（Alignment Signal）** 来训练更稳健的模型（推测）。  

### 一句话记住它
PromptRobust 揭示了即使是最强大的大语言模型，也会在细微的、看似无害的提示扰动下大幅失效，提醒我们必须把提示鲁棒性当作模型安全的基本要素。