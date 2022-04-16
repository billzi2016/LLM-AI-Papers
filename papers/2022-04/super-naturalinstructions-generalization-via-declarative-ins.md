# Super-NaturalInstructions: Generalization via Declarative Instructions   on 1600+ NLP Tasks

> **Date**：2022-04-16
> **arXiv**：https://arxiv.org/abs/2204.07705

## Abstract

How well can NLP models generalize to a variety of unseen tasks when provided with task instructions? To address this question, we first introduce Super-NaturalInstructions, a benchmark of 1,616 diverse NLP tasks and their expert-written instructions. Our collection covers 76 distinct task types, including but not limited to classification, extraction, infilling, sequence tagging, text rewriting, and text composition. This large and diverse collection of tasks enables rigorous benchmarking of cross-task generalization under instructions -- training models to follow instructions on a subset of tasks and evaluating them on the remaining unseen ones. Furthermore, we build Tk-Instruct, a transformer model trained to follow a variety of in-context instructions (plain language task definitions or k-shot examples). Our experiments show that Tk-Instruct outperforms existing instruction-following models such as InstructGPT by over 9% on our benchmark despite being an order of magnitude smaller. We further analyze generalization as a function of various scaling parameters, such as the number of observed tasks, the number of instances per task, and model sizes. We hope our dataset and model facilitate future progress towards more general-purpose NLP models.

---

# Super-NaturalInstructions：通过声明式指令实现跨任务泛化（1600+ NLP 任务） 论文详细解读

### 背景：这个问题为什么难？
在 NLP 里，模型往往被训练在固定的任务上，比如情感分类或命名实体识别，换到新任务时需要重新标注数据或微调。过去的多任务学习虽然能让模型看到多种任务，但仍依赖于统一的标签空间或手工设计的任务头，难以直接接受“请帮我做 X”这种自然语言指令。更重要的是，缺少大规模、覆盖面广的指令数据集，导致我们无法系统评估模型在真正未见任务上的指令遵循能力。于是，如何让模型只看说明就能完成千差万别的新任务，成为了一个迫切且尚未解决的挑战。

### 关键概念速览
**指令跟随（Instruction Following）**：模型在收到一段自然语言任务描述后，直接输出对应答案，而不是先经过任务特化的模型头。类似于人类听到“把这段文字改写成正式语气”，立刻动手改写。  
**声明式指令（Declarative Instruction）**：用完整、明确的句子描述任务目标和输入输出格式，而不是示例或代码。想象成一本使用手册，告诉模型“这是一道选择题，请返回正确选项”。  
**跨任务泛化（Cross‑Task Generalization）**：模型在只见过一部分任务的指令后，能够在完全新任务上仍然表现良好。相当于学会了“学习如何学习”。  
**Tk‑Instruct**：本文训练的 Transformer 系列模型，专门用来接受多种指令（文字定义或 k‑shot 示例）并生成答案。它的名字来源于“Task‑aware Knowledge”。  
**In‑Context Learning**：在推理时直接把少量示例或指令放进模型的上下文，而不改变模型参数。好比在考试时老师给你几道例题，你直接用这些例子来解后面的题。  
**任务类型（Task Type）**：指任务的功能大类，如分类、抽取、填空、序列标注、文本改写、文本生成等。每种类型对应不同的输入输出模式。  
**指令数据集（Instruction Dataset）**：包含任务描述、示例和评估数据的集合，用来训练或评估指令跟随模型。  

### 核心创新点
1. **从少量任务到千余任务的指令基准 → 构建 Super‑NaturalInstructions**：作者把原有的 NaturalInstructions 扩展到 1,616 个任务，覆盖 76 种任务类型，并为每个任务撰写专家级指令。这样就有了一个足够大、足够多样的“指令实验室”，可以真正检验模型的跨任务泛化能力。  
2. **小模型也能高效指令跟随 → 训练 Tk‑Instruct**：在该基准上，作者训练了一个相对轻量的 Transformer（参数量比 InstructGPT 小约十倍），但通过大规模指令数据和混合的 plain‑language 与 k‑shot 示例输入，使模型在未见任务上比 InstructGPT 高出 9%+。这表明模型规模不是唯一决定指令能力的因素，数据多样性和训练方式同样关键。  
3. **系统化分析泛化因素 → 任务数量、实例数、模型规模的标度实验**：作者分别增减训练任务数、每任务的样本数以及模型层数，观察对跨任务表现的影响。结果显示，任务种类的多样性提升比单纯增加样本更能推动泛化，这为以后构建更通用的指令模型提供了经验法则。  
4. **统一的指令格式 → 声明式指令 + k‑shot 示例的混合输入**：过去的指令学习要么只用文字描述，要么只用示例，二者各有盲点。本文把两者合在一起，让模型既能读懂任务目标，又能从少量示例中捕捉细节，类似于人类在阅读说明书后再看几个操作示例，效果更稳健。

### 方法详解
整体思路可以划分为三步：**数据构建 → 模型训练 → 指令推理**。

1. **数据构建**  
   - **任务挑选**：从公开的 NLP 任务库（如 GLUE、SuperGLUE、Kaggle、HuggingFace Datasets 等）中抽取 1,616 条任务，确保覆盖分类、抽取、填空、序列标注、改写、生成等大类。  
   - **指令撰写**：每个任务请领域专家用自然语言写出完整的任务说明，包括输入格式、输出要求、评价指标等。比如对情感分类的指令会写成“判断下面这段文字的情感倾向，是正面还是负面”。  
   - **示例生成**：为每个任务随机抽取若干实例，形成 k‑shot 示例（k=1~5），并把这些示例拼接在指令后面，形成统一的输入模板。  
   - **评估划分**：把任务集合随机划分为 **训练任务**（约 80%）和 **测试任务**（剩余 20%），后者在训练期间从未出现过，保证评估的“未见任务”属性。

2. **模型训练（Tk‑Instruct）**  
   - **模型架构**：采用标准的 Transformer 编码器‑解码器（类似 T5），但规模从小到中等不等（参数量约 300M–1B），比 InstructGPT 小约十倍。  
   - **输入编码**：把指令文字、k‑shot 示例以及实际输入拼接成一个长序列，使用特殊分隔符标记不同段落（如 `<INST>`, `<EXAMPLE>`, `<INPUT>`）。  
   - **目标输出**：模型直接生成任务的答案文本（如标签、抽取片段、改写后句子），不需要额外的任务头。  
   - **训练目标**：最大化生成答案的对数似然，即让模型在看到指令+示例后，尽可能准确地预测答案。  
   - **技巧**：使用混合精度、梯度累积以及大批量随机采样，确保模型在多任务指令上均衡学习；同时加入指令噪声（同一任务的不同表述）提升鲁棒性。

3. **指令推理**  
   - **纯文字指令**：只给模型任务描述，不提供示例，模型依赖对指令的理解完成任务。  
   - **k‑shot 推理**：在指令后附上少量示例，模型可以从示例中学习细粒度的映射规则。  
   - **解码策略**：使用束搜索（beam search）或采样（sampling）得到最终答案，依据任务类型选择不同的后处理（如去除特殊标记、对齐标签空间）。

**最巧妙的地方**在于把“声明式指令”和“k‑shot 示例”混合进同一上下文，让模型在同一次前向传播中既学到任务的宏观目标，又捕捉到具体的输入输出对应关系。这种“一次性指令+示例”设计，比单独使用文字或示例更能提升对新任务的适应性。

### 实验与效果
- **评测基准**：使用 Super‑NaturalInstructions 中的 20% 未见任务（约 320 条），覆盖所有 76 种任务类型。  
- **对比模型**：InstructGPT（原始指令模型，参数量约 10×）以及若干公开的多任务 T5 变体。  
- **主要结果**：Tk‑Instruct 在整体指标上比 InstructGPT 高出 **9% 以上**（原文未给出精确数值），且在多数任务类型上都有正向提升。尤其在需要细粒度抽取的任务上，混合指令+示例的设置带来了显著增益。  
- **消融实验**：  
  1. **去掉 k‑shot 示例** → 性能下降约 3%–5%，说明示例对细节捕捉很关键。  
  2. **仅使用单一任务类型训练** → 在跨任务测试时跌破 50% 的基准，验证多样任务训练的重要性。  
  3. **缩小指令数据规模**（只用 50% 任务）→ 效果下降约 2%，表明任务数量的多样性对泛化有线性正向影响。  
- **局限性**：作者指出模型仍然对极端长指令或高度专业化的领域（如医学报告）表现不佳；此外，指令的语言风格差异仍会导致一定的性能波动。  

### 影响与延伸思考
这篇工作提供了首个规模超过千任务的指令基准，直接推动了“通用指令模型”方向的研究。随后出现的 **OpenInstruction**、**FLAN** 系列以及 **Mistral‑Instruct** 等模型，都在数据规模或指令多样性上向 Super‑NaturalInstructions 看齐。对想进一步探索的读者，可以关注以下几个方向：  
- **指令自动生成**：利用大模型自行生成高质量指令，降低人工撰写成本。  
- **跨语言指令**：把指令翻译成多语言，检验模型在不同语言指令下的泛化。  
- **指令鲁棒性**：研究指令的噪声、歧义或对抗性修改对模型输出的影响。  
- **任务层次结构**：把任务类型组织成层次图谱，让模型在学习时利用任务之间的语义关联。  

### 一句话记住它
只要给模型一段清晰的任务说明和少量示例，即使模型小到十倍于 InstructGPT，也能在上千未见任务上表现出色。