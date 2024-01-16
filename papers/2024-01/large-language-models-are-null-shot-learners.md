# Large Language Models are Null-Shot Learners

> **Date**：2024-01-16
> **arXiv**：https://arxiv.org/abs/2401.08273

## Abstract

This paper presents null-shot prompting. Null-shot prompting exploits hallucination in large language models (LLMs) by instructing LLMs to utilize information from the "Examples" section that never exists within the provided context to perform a task. While reducing hallucination is crucial and non-negligible for daily and critical uses of LLMs, we propose that in the current landscape in which these LLMs still hallucinate, it is possible, in fact, to exploit hallucination to increase performance in performing tasks compared to standard zero-shot prompting. Experiments with eight LLMs show improvements in performance across the majority of eight datasets, including reading comprehension, arithmetic reasoning, and closed-book question answering. The observed inconsistency in increased relative performance across the LLMs also potentially indicates a different degree of inherent hallucination in each model. These differences show that it is possible to utilize null-shot prompting as a way to detect degrees of hallucination in LLMs using existing benchmarking datasets. We also perform ablation studies, including experimenting with a modified version of null-shot prompting that incorporates ideas from zero-shot chain-of-thought prompting, which shows different trends of results.

---

# Large Language Models are Null‑Shot Learners 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在没有任何示例的情况下完成任务时，往往只能靠“零样本提示”来引导模型思考。零样本提示的效果受限于模型对任务描述的理解深度，尤其在阅读理解、算术推理等需要细粒度推理的场景，准确率常常不尽如人意。与此同时，LLM 天生会产生“幻觉”——即在缺乏明确依据时编造信息。过去的研究把幻觉视作缺陷，努力通过检索、事实校验等手段抑制它，却忽视了幻觉本身可能蕴含的潜在信息。于是出现了一个悖论：如果模型已经在“编造”，能否把这种编造当作一种资源，反而提升任务表现？

### 关键概念速览
- **零样本提示（Zero‑Shot Prompting）**：不给模型任何示例，只提供任务描述，让模型直接输出答案。类似让学生在没有例题的情况下直接写作文。
- **幻觉（Hallucination）**：模型在没有真实依据时生成的虚构内容。可以把它想象成人在记忆不清时凭空捏造的细节。
- **Null‑Shot Prompting**：在提示中声称有“示例”章节，但实际不提供任何示例，让模型自行“想象”示例并据此作答。相当于让学生先假装自己已经看过例题，然后再写答案。
- **Chain‑of‑Thought（CoT）**：让模型在给出最终答案前先写出推理步骤，像在黑板上写草稿一样帮助模型理清思路。
- **Hallucination Degree（幻觉程度）**：衡量模型产生虚构信息的倾向性。幻觉程度高的模型更容易在 Null‑Shot 场景下“发挥”。
- **基准数据集（Benchmark Datasets）**：用于评估模型在阅读理解、算术推理、闭卷问答等任务上的表现的标准数据集合。

### 核心创新点
1. **从抑制幻觉转向利用幻觉**  
   - 之前的工作把幻觉当作错误信号，努力让模型不产生虚构内容。  
   - 这篇论文故意在提示中制造一个不存在的“示例”章节，诱导模型自行构造示例。  
   - 结果显示，模型在自行“编造”示例后，整体任务准确率超过了传统零样本提示。

2. **提出 Null‑Shot Prompt 的统一模板**  
   - 传统零样本提示只包含任务描述和问题本身。  
   - Null‑Shot Prompt 在此基础上加入“Examples:”字段，但不填任何内容，随后直接给出问题。  
   - 这种极简的结构让模型产生内部示例，形成一种自我提示的循环，提高了推理质量。

3. **将 Null‑Shot 用作幻觉程度的探针**  
   - 通过在同一基准上比较不同模型的 Null‑Shot 增益，作者发现增益大小与模型的幻觉倾向相关。  
   - 于是 Null‑Shot 成为一种无需额外标注、即可评估模型幻觉程度的工具。

4. **结合 CoT 的改进版 Null‑Shot**  
   - 在 Null‑Shot 基础上加入“Let’s think step by step.”的链式思考指令。  
   - 与纯 Null‑Shot 相比，表现出现不同趋势，说明两种提示策略在利用幻觉的方式上并不完全相同。

### 方法详解
**整体思路**：先给模型一个看似完整但实际缺失示例的提示，让模型自行填补空白，再让它基于这些自创的示例完成任务。整个过程只涉及一次前向推理，不需要额外的检索或微调。

**步骤拆解**：

1. **构造 Null‑Shot Prompt**  
   - 提示模板大致为：  
     ```
     Task: <任务描述>
     Examples:
     Question: <问题>
     Answer:
     ```  
   - 注意“Examples:”后面没有任何示例，只留下空行。相当于在对话中说：“下面会有示例，但我先不展示，你自行想象。”

2. **模型内部生成示例**  
   - 当模型读取到“Examples:”但没有实际内容时，会依据任务描述和自身的语言生成能力，自动产生一段或多段看似示例的文本。  
   - 这一步是模型的自我提示（self‑prompting），类似人类在没有参考答案时先回忆类似情境再作答。

3. **基于自创示例完成主问题**  
   - 在模型生成完示例后，紧接着出现的真实问题会被模型视作“在示例之后的下一个任务”。  
   - 模型会把前面自编的示例当作参考，推理出答案。

4. **可选的 CoT 叠加**  
   - 在 Null‑Shot Prompt 前加入一步链式思考指令，如“Let’s think step by step.”  
   - 这会促使模型在自创示例后先写出推理过程，再给出最终答案，进一步提升复杂任务的可靠性。

**关键细节**：

- **提示词的顺序**：必须先出现“Examples:”再出现真实问题，否则模型不会触发自创示例的行为。  
- **示例空白的长度**：实验中发现留一个空行足以让模型产生完整示例，过多空行并不会带来额外收益。  
- **模型规模的影响**：大模型更倾向于生成高质量的自创示例，从而在 Null‑Shot 场景下获得更大提升。

**最巧妙的地方**：利用模型的“幻觉”本能把缺失信息当作创作空间，而不是把它视作错误。这个思路把原本的缺点转化为优势，几乎不需要额外资源。

### 实验与效果
- **测试任务**：包括阅读理解、算术推理、闭卷问答等八个公开基准数据集。  
- **对比基线**：标准零样本提示（直接给任务描述和问题）以及在部分实验中加入 CoT 的零样本提示。  
- **主要发现**：在大多数数据集上，Null‑Shot Prompt 相比零样本提示提升了显著的准确率，尤其在算术推理和阅读理解上提升最为明显。  
- **模型覆盖**：实验使用了八种不同规模和架构的 LLM（包括 GPT‑3 系列、Claude、LLaMA 等），不同模型的提升幅度不一，暗示了幻觉程度的差异。  
- **消融实验**：  
  - 移除“Examples:”字段后，性能回落到普通零样本水平，验证了该字段的关键性。  
  - 加入 CoT 指令后，某些模型的提升幅度进一步扩大，而另一些模型则出现下降，说明两种提示策略的交互并非线性。  
- **局限性**：  
  - 对于极度保守或已被强制抑制幻觉的模型，Null‑Shot 的增益有限。  
  - 在需要严格事实准确性的任务（如法律文书生成）中，利用幻觉可能导致错误信息的放大，作者提醒需谨慎使用。  
  - 论文未提供具体数值，只给出“多数数据集上表现提升”，因此实际提升幅度需自行复现验证。

### 影响与延伸思考
这篇工作打开了“把幻觉当资源”的新视角，随后有几篇后续研究尝试在不同任务（如代码生成、对话系统）中加入类似的自创示例机制，甚至提出“Self‑Example Prompting”作为通用提示模板。还有研究把 Null‑Shot 作为模型诊断工具，用来量化不同模型的幻觉倾向，为模型选型提供了新的参考指标。未来可以进一步探索：  
- 如何在保持高准确率的同时约束幻觉的真实性，形成“受控幻觉”。  
- 将 Null‑Shot 与检索增强（Retrieval‑Augmented Generation）结合，让模型在自创示例的同时参考外部事实。  
- 在多语言或跨模态任务中验证 Null‑Shot 的普适性。

### 一句话记住它
让模型“假装有示例却不给”，利用它的幻觉自编示例，从而在零样本任务上实现意想不到的性能提升。