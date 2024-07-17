# SELF-GUIDE: Better Task-Specific Instruction Following via   Self-Synthetic Finetuning

> **Date**：2024-07-16
> **arXiv**：https://arxiv.org/abs/2407.12874

## Abstract

Large language models (LLMs) hold the promise of solving diverse tasks when provided with appropriate natural language prompts. However, prompting often leads models to make predictions with lower accuracy compared to finetuning a model with ample training data. On the other hand, while finetuning LLMs on task-specific data generally improves their performance, abundant annotated datasets are not available for all tasks. Previous work has explored generating task-specific data from state-of-the-art LLMs and using this data to finetune smaller models, but this approach requires access to a language model other than the one being trained, which introduces cost, scalability challenges, and legal hurdles associated with continuously relying on more powerful LLMs. In response to these, we propose SELF-GUIDE, a multi-stage mechanism in which we synthesize task-specific input-output pairs from the student LLM, then use these input-output pairs to finetune the student LLM itself. In our empirical evaluation of the Natural Instructions V2 benchmark, we find that SELF-GUIDE improves the performance of LLM by a substantial margin. Specifically, we report an absolute improvement of approximately 15% for classification tasks and 18% for generation tasks in the benchmark's metrics. This sheds light on the promise of self-synthesized data guiding LLMs towards becoming task-specific experts without any external learning signals.

---

# SELF‑GUIDE：通过自合成微调实现更好的任务特定指令遵循 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）面前，只要给出合适的自然语言提示，就能让模型尝试完成各种任务。但实际使用时，单纯靠提示往往只能得到“勉强可用”的答案，准确率远低于在同一任务上做充分微调的模型。微调的前提是要有大量标注好的训练数据，可是很多细分任务根本没有公开的大规模标注集。于是出现了“让更大的模型帮忙生成数据再训练小模型”的思路，却需要额外的、往往更昂贵的上游模型作为数据源，成本高、版权风险大，难以大规模推广。换句话说，如何在没有外部数据和外部模型的帮助下，让一个 LLM 自己变成某个任务的专家，成了亟待突破的瓶颈。

### 关键概念速览
**提示（Prompt）**：把任务描述、示例或问题以自然语言的形式喂给模型，让模型据此生成答案。相当于给模型下指令，但模型只能“猜”出最合适的输出，往往不够精准。  
**微调（Finetuning）**：在已有的大模型基础上，用特定任务的标注数据继续训练，使模型的参数向该任务的最优解靠拢。类似于把通用工具改装成专用工具。  
**自合成（Self‑Synthetic）**：模型自己生成训练所需的输入‑输出对，而不是依赖外部模型或人工标注。可以把模型想象成既是老师也是学生。  
**指令遵循（Instruction Following）**：模型在接收到明确的任务指令后，能够按照指令的意图产生符合预期的输出。好比人类在听到“请写一封邀请函”后，能写出符合格式和语气的信件。  
**多阶段机制（Multi‑stage Mechanism）**：把整个学习过程拆成若干连续的步骤，每一步的产出都为下一步提供更好的起点。类似于分层训练营，先打基础再强化。  
**自然指令（Natural Instructions）**：一种任务集合，涵盖分类、生成、抽取等多种任务类型，每个任务都有自然语言的说明和示例，用来评估模型的指令遵循能力。  

### 核心创新点
1. **之前的方法 → 本文的做法 → 带来的改变**  
   过去的自监督或自指令方法大多依赖一个比目标模型更强的外部 LLM 来生成伪标签，然后再用这些伪标签微调目标模型。SELF‑GUIDE 把生成和微调的角色合二为一，让学生模型自己产生任务特定的输入‑输出对，再用这些对直接微调自身。这样省掉了外部模型的使用成本，也避免了跨模型版权争议。  

2. **之前的方法 → 本文的做法 → 带来的改变**  
   传统的微调往往一次性使用固定的标注集，缺乏对模型自身能力的动态反馈。SELF‑GUIDE 引入了“自我评估—自我纠错”的循环：模型先生成答案，再用同模型的另一种提示检查答案的合理性，只有通过检查的对才进入微调池。这样可以在生成过程中筛除低质量样本，提高自合成数据的整体质量。  

3. **之前的方法 → 本文的做法 → 带来的改变**  
   以往的任务特化往往只针对单一任务或少数任务进行微调，难以在多任务基准上统一提升。SELF‑GUIDE 在 Natural Instructions V2 这样的大规模多任务集合上一次性完成自合成与微调，展示了跨任务的通用提升能力，证明了自合成微调可以在“多任务”场景下同样有效。  

### 方法详解
#### 整体框架
SELF‑GUIDE 把学习过程划分为三个阶段：① **任务指令采样**，② **自合成数据生成与过滤**，③ **自我微调**。整个流程像是让模型先“读任务说明”，再“自己出题并答题”，最后“把答对的题目当作教材再学一遍”。  

#### 步骤拆解  
1. **任务指令采样**  
   - 从 Natural Instructions V2 中抽取每个任务的自然语言描述、输入格式和示例。  
   - 把这些信息包装成一个统一的 Prompt，喂给学生模型，让它了解当前要解决的任务是什么。  

2. **自合成数据生成**  
   - **输入生成**：模型在指令的引导下，随机生成若干符合任务输入约束的样本。例如分类任务会让模型自行造出若干句子；生成任务会让模型自行构造情境描述。  
   - **答案生成**：同一模型在第二轮 Prompt 中被要求对刚才生成的输入给出答案。这里使用了“Chain‑of‑Thought”式的提示，让模型先思考再输出，以提升答案质量。  

3. **自我过滤**  
   - 为了防止模型把错误答案也当作教材，SELF‑GUIDE 设计了一个**自评 Prompt**：让模型再次审视自己的答案，判断是否符合任务要求（比如是否满足分类标签的定义、是否满足生成文本的长度或风格约束）。  
   - 只有通过自评的输入‑输出对才会被保留进入微调数据池。  

4. **自我微调**  
   - 将过滤后的自合成对拼成一个标准的微调数据集。  
   - 使用常规的监督学习方式（交叉熵损失）继续训练学生模型，训练轮数和学习率与普通微调相同，只是数据来源不同。  
   - 训练结束后，模型即可在该任务上直接使用指令进行推理，表现比单纯提示要好得多。  

#### 关键细节与巧思  
- **双向 Prompt 设计**：一次用于生成输入，一次用于生成答案，还要一次用于自评，形成了“三段式” Prompt 链，确保每一步都有明确的目标。  
- **自评阈值**：作者没有使用硬性的对错判定，而是让模型给出置信度分数，只保留置信度高于某阈值的样本，这相当于让模型自己决定哪些答案值得信任。  
- **多轮迭代**：虽然论文的核心实验只用了单轮自合成微调，但框架天然支持多轮循环——每轮微调后模型的生成质量提升，后续轮次可以产生更高质量的自合成数据，形成“自我提升的闭环”。  

### 实验与效果
- **评测基准**：使用 Natural Instructions V2，这是一套覆盖分类、生成、抽取等 61 类任务的多任务指令集合，提供统一的评测指标。  
- **对比基线**：  
  - 纯提示（zero‑shot / few‑shot）方式的直接推理。  
  - 传统微调：在同样的学生模型上使用公开的少量标注数据进行微调。  
  - 外部合成：利用更大模型（如 GPT‑4）生成伪标签再微调的方案（论文中称为 “teacher‑generated data”）。  
- **性能提升**：在分类任务上，SELF‑GUIDE 相比纯提示提升约 15% 的绝对分数；在生成任务上提升约 18%。相对传统微调，提升幅度略低于外部合成方案，但省去了外部模型的使用成本。  
- **消融实验**：论文对自评过滤、双向 Prompt、以及是否使用多轮迭代做了消融。结果显示，去掉自评过滤后性能下降约 5%，说明过滤是提升质量的关键环节。  
- **局限性**：作者承认自合成数据的质量仍受模型本身能力限制，尤其在极端长文本或高度专业化任务上，生成的样本可能仍不够可靠。此外，自我微调会消耗额外的计算资源，虽然比租用更大模型便宜，但仍不是零成本。  

### 影响与延伸思考
SELF‑GUIDE 把“自我生成‑自我学习”落地为一种可操作的微调流程，打开了在缺乏标注数据的场景下让 LLM 自主进化的大门。随后的研究（如 Self‑Instruct、Iterative Self‑Training 等）进一步扩展了这种闭环思路，尝试在更大规模、多语言甚至跨模态任务上实现自我提升。对想深入的读者，可以关注以下方向：① 更精细的自评机制（比如引入外部校验器或置信度校准）；② 多轮迭代的收敛行为与理论分析；③ 将 SELF‑GUIDE 与 RLHF（基于人类反馈的强化学习）结合，探索“自我生成 + 人类校正”的混合路径。  

### 一句话记住它
让模型自己出题、自己批改，再用合格的答案微调自身，省去外部大模型，直接把 LLM 变成任务专家。