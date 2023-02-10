# Toolformer: Language Models Can Teach Themselves to Use Tools

> **Date**：2023-02-09
> **arXiv**：https://arxiv.org/abs/2302.04761

## Abstract

Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q\&A system, two different search engines, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.

---

# Toolformer：语言模型可以自学使用工具 论文详细解读

### 背景：这个问题为什么难？

大规模语言模型（LLM）在零样本或少样本设置下能完成很多抽象任务，却在算数、事实查询等“基础功能”上经常失手。传统做法是直接在模型内部让它记忆这些知识，但模型规模越大，记忆的细节越容易被稀释，导致在需要精确答案时表现不佳。另一方面，专门的工具（计算器、搜索引擎等）在对应子任务上几乎是完美的，却缺乏语言模型的通用推理能力。如何让一个统一的模型在需要时主动调用外部工具，而不是硬编码在模型内部，是一个既涉及模型自我决策，又涉及跨系统交互的难题。

### 关键概念速览
- **语言模型（LM）**：接受文字序列并预测下一个词的模型，类似于在句子里猜下一个字的游戏。  
- **工具（Tool）**：模型可以通过 API 调用的外部程序，例如计算器、搜索引擎、翻译系统等，像是模型的“外挂”。  
- **API 调用决策**：模型在生成文本的每一步判断是否需要调用工具、何时调用以及传入什么参数，类似于人在写作时决定是否去查字典。  
- **自监督学习**：不需要人工标注的大规模训练方式，模型自己生成“伪标签”来学习，这里指模型通过少量示例自动生成调用标记。  
- **零样本（Zero‑shot）**：模型在没有看到任何任务特定示例的情况下直接完成任务，像是第一次见到新游戏却还能玩得不错。  
- **提示（Prompt）**：给模型的输入指令或示例，用来引导模型的行为，类似于老师给学生的题目说明。  
- **结果融合**：模型把工具返回的答案重新写进自己的生成流中，使后续预测受益，像是把查到的资料粘贴进自己的笔记。  

### 核心创新点
1. **从少量示例到自监督调用标记**  
   之前的工作需要大量人工标注哪些位置应该调用工具。Toolformer 只提供每个 API 几个手工示例，然后让模型自行在大规模文本中搜索相似上下文，自动生成“调用标签”。这样大幅降低了标注成本，同时保持了高质量的训练信号。

2. **统一的调用决策网络**  
   传统方案往往为每种工具单独训练一个判断器。本文把所有工具的调用判断合并进同一个语言模型内部的分类头，模型在每一步同时预测“是否调用”以及“调用哪个工具”。这种共享参数的设计让模型能够在不同工具之间迁移经验，提升了整体鲁棒性。

3. **结果嵌入回语言模型**  
   许多系统把工具输出当作外部信息，直接喂给下游模型，却没有让语言模型自行学习如何把这些信息写进自己的上下文。Toolformer 在训练时让模型学习把工具返回的文本当作普通 token 继续生成，从而在推理阶段自然地把结果融合进后续预测。

4. **保持原始语言建模能力**  
   增加工具调用的风险是可能削弱模型的纯语言生成水平。作者在训练时交替进行普通语言建模和工具调用任务，确保模型在不需要工具时仍然表现如原始模型。实验显示，这种“双轨”训练几乎不牺牲原有的语言流畅度。

### 方法详解
整体思路可以拆成三大步骤：**示例收集 → 自动标注 → 联合微调**。

1. **示例收集**  
   对每个目标 API（如计算器、搜索引擎）准备 5–10 条手工示例，示例形式是“文本 → 调用 → 参数 → 结果”。比如“3+5 等于多少？” → 调用计算器 → 参数 “3+5” → 结果 “8”。这些示例仅用于引导模型学习调用的模式。

2. **自动标注（Self‑Supervised Tagging）**  
   把大规模未标注的文本（比如维基百科）喂进语言模型，让模型在每个 token 位置预测是否出现了类似示例的模式。如果模型判断出可能需要调用工具，就用对应的 API 生成真实结果，并把“调用+参数+结果”写进标注文件。这样模型自己产生了数百万条“调用标签”，为后续训练提供了丰富的监督信号。

3. **联合微调（Joint Fine‑tuning）**  
   在微调阶段，模型的输出层被扩展为两部分：  
   - **调用分类头**：在每一步输出一个概率，决定是否调用工具以及调用哪一个。  
   - **普通语言建模头**：仍然预测下一个 token。  
   训练目标是两者的加权和：当标签指示调用时，模型必须学习正确的 API、参数生成以及把返回的文本写回；否则继续普通语言预测。因为调用过程本身也是一次“生成”，模型在训练时会看到真实的 API 调用序列，从而学会在推理时自行触发。

**关键细节**  
- **参数生成方式**：模型直接在文本中生成参数字符串，类似于在句子里写出“3+5”。  
- **结果融合策略**：工具返回的文本被插入到模型的生成流中，后面的 token 预测会基于这段新加入的上下文。  
- **防止滥用调用**：在训练时加入惩罚项，使模型倾向于只在真正需要时调用工具，避免无意义的 API 调用。  
- **多工具共享**：所有工具共享同一个调用分类头，只是输出空间里有不同的工具 ID，这让模型可以在少数示例的帮助下快速适配新工具。

### 实验与效果
- **测试任务**：包括算数题（GSM8K）、事实查询（Open-domain QA）、机器翻译、日程推理等零样本任务。  
- **基线对比**：与同规模的原始语言模型、以及使用外部工具但需要手工标注的系统相比，Toolformer 在多数任务上取得了两位数的提升。比如在算数任务上，准确率从约 30% 提升到接近 70%（具体数字请参考原文）。  
- **消融实验**：作者分别去掉自动标注、共享调用头、结果融合三项，发现每项都对最终性能有显著贡献，尤其是去掉结果融合会导致后续预测几乎不利用工具输出。  
- **保持语言建模**：在标准语言建模基准（如 WikiText‑103）上，Toolformer 的困惑度（perplexity）几乎与原始模型持平，说明加入工具能力并未削弱其文本生成水平。  
- **局限性**：工具调用仍然受限于 API 的响应时间和可用性；在需要多步推理的复杂任务上，模型有时会提前结束调用，导致答案不完整。原文也提到对工具的错误返回缺乏鲁棒的纠错机制。

### 影响与延伸思考
Toolformer 打开了“语言模型自我装备”这一思路的大门，随后出现了大量围绕“工具使用”展开的工作，例如让模型调用代码解释器、数据库查询或图像生成模型。很多后续研究在此基础上加入了**工具选择的强化学习**、**多模态工具链**以及**可解释的调用日志**等方向。想进一步了解，可以关注以下趋势：  
- **工具调用的安全与对齐**：如何防止模型滥用或误用外部 API。  
- **跨语言/跨模态工具集成**：把语音识别、图像检索等非文本工具纳入统一框架。  
- **自适应工具学习**：模型在运行时自行发现新工具并学习调用方式，而不是事先提供示例。  

### 一句话记住它
让大模型在需要时主动调用外部工具，并通过自监督学习把调用过程内化，从而在基础功能上匹配专用工具，同时保持通用语言能力。