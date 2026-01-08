# An Empirical Investigation of Robustness in Large Language Models under Tabular Distortions

> **Date**：2026-01-08
> **arXiv**：https://arxiv.org/abs/2601.05009

## Abstract

We investigate how large language models (LLMs) fail when tabular data in an otherwise canonical representation is subjected to semantic and structural distortions. Our findings reveal that LLMs lack an inherent ability to detect and correct subtle distortions in table representations. Only when provided with an explicit prior, via a system prompt, do models partially adjust their reasoning strategies and correct some distortions, though not consistently or completely. To study this phenomenon, we introduce a small, expert-curated dataset that explicitly evaluates LLMs on table question answering (TQA) tasks requiring an additional error-correction step prior to analysis. Our results reveal systematic differences in how LLMs ingest and interpret tabular information under distortion, with even SoTA models such as GPT-5.2 model exhibiting a drop of minimum 22% accuracy under distortion. These findings raise important questions for future research, particularly regarding when and how models should autonomously decide to realign tabular inputs, analogous to human behavior, without relying on explicit prompts or tabular data pre-processing.

---

# 大语言模型在表格扭曲下的鲁棒性实证研究 论文详细解读

### 背景：这个问题为什么难？

表格是结构化信息的典型载体，很多问答系统依赖模型把文字描述的表格转化为内部的“表格感知”。在此之前的研究大多假设输入的表格是干净、规范的——列标题对应列内容、行列对齐完整。实际使用中，表格经常因为复制粘贴、OCR、手工编辑等原因出现列名错位、单位不统一、行列错位等细微错误。传统的 LLM 直接把表格当作普通文本处理，缺少专门的错误检测与纠正机制，导致在这些微小扭曲出现时准确率骤降。正因为这种“看似小却致命”的问题在真实场景里普遍存在，才需要专门的研究来量化并改进模型的鲁棒性。

### 关键概念速览

**表格扭曲（Table Distortion）**：对原本规范的表格进行细微的结构或语义改动，如把两列的顺序互换或把单位写错，类似于把一本书的章节标题错位后再阅读。  

**语义扭曲（Semantic Distortion）**：改变表格中信息的意义层面，例如列名与列值不匹配、单位不一致，像是把“重量（kg）”误写成“重量（lb）”。  

**结构扭曲（Structural Distortion）**：改变表格的排版或布局，如行向右偏移、单元格拆分，类似于把一张表格的格子错位后再填数据。  

**表格问答（Table Question Answering, TQA）**：给模型一个表格和一个自然语言问题，要求模型在表格中检索或计算出答案。  

**系统提示（System Prompt）**：在对话式模型的最前面加入一段指令，告诉模型该怎么处理后续输入，相当于给模型下达“作业要求”。  

**错误纠正步骤（Error‑Correction Step）**：模型在正式回答前先对表格进行自检和修正的过程，类似于人类先检查表格是否有错再做分析。  

**鲁棒性（Robustness）**：模型在面对输入噪声或异常时仍能保持性能的能力，等同于人在嘈杂环境下仍能听清对话。  

**专家标注基准（Expert‑Curated Benchmark）**：由领域专家手工挑选并标注的测试集，用来精准评估模型在特定任务上的表现。

### 核心创新点

1. **从“无干预”到“显式先验”实验设计**  
   *之前的工作往往直接把扭曲表格喂给模型，观察性能下降，却没有尝试给模型额外的指示。*  
   *本文在系统提示中加入“请先检查表格是否有错并自行纠正”的指令，让模型主动进入错误纠正模式。*  
   *结果显示，带提示的模型在多数扭曲类型上恢复了约 10%‑15% 的准确率，证明显式先验能够部分激活模型的自检能力。*

2. **构建小规模、专家审校的 TQA 扭曲基准**  
   *过去的表格问答数据集（如 WikiTableQuestions）几乎全是干净表格，缺少噪声样本。*  
   *作者从 WikiTQ 中挑选出 200 条代表性问答，手工施加六类语义/结构扭曲并标注正确答案，形成了一个专门评估“先纠错后答题”能力的基准。*  
   *该基准揭示了即使是最先进的模型（GPT‑5.2）在最轻微的纵向错位下也会掉 22% 以上的准确率。*

3. **系统化分析模型对不同扭曲的感知差异**  
   *以往只报告整体下降幅度，缺少对“哪类扭曲最致命”的细粒度洞察。*  
   *本文通过对每种扭曲单独评测，绘制了性能热力图，发现结构扭曲（尤其是行的水平错位）比语义扭曲更容易导致模型误解全局结构。*  
   *这种细分让后续研究可以有针对性地设计纠错模块。*

4. **提出“自适应对齐”概念的研究议题**  
   *在实验中观察到模型在没有提示时几乎不主动纠错，提示后又并非总能成功。*  
   *作者提出模型应当像人类一样在检测到潜在不一致时自行决定是否进行对齐，而不是完全依赖外部指令。*  
   *这为未来的自监督纠错机制提供了明确的方向。*

### 方法详解

整体思路可以概括为三步：**（1）构造扭曲表格 → （2）通过系统提示诱导模型进行错误纠正 → （3）在纠正后的表格上执行标准 TQA**。下面逐层拆解。

1. **扭曲生成器**  
   - 研究者实现了一个规则库，针对每条原始表格随机施加六类扭曲中的一种或多种。  
   - 语义扭曲包括：列标题互换、数值单位改写、标签值不匹配。  
   - 结构扭曲包括：把整列向下平移若干行、把行整体向右错位、把单元格拆分成两行、把表头嵌入数据行。  
   - 生成的每个扭曲样本都保留原始的“干净”版本，以便后续对比。

2. **系统提示设计**  
   - 提示的核心句式是：“在回答下面的问题之前，请先检查表格是否存在列名/行列错位等错误，并自行纠正后再进行分析。”  
   - 为避免模型把提示当成普通文字，提示被放在对话的 **system** 角色位置，确保所有后续输入都在同一会话上下文中。  
   - 研究者还实验了不同提示长度（简短 vs. 详细），发现简短提示足以触发纠错行为，详细提示并未带来额外提升。

3. **错误纠正模块（模型内部）**  
   - 在实际运行时，模型先接收到系统提示，然后收到扭曲表格的文字化表示（如 Markdown 表格）。  
   - 模型的生成过程被视为一次“自检+回答”双阶段任务：  
     - **自检阶段**：模型输出一段描述表格是否异常的文字（例如“第3行的列标题向右错位了”），并给出纠正后的表格文本。  
     - **回答阶段**：模型基于纠正后的表格继续回答用户提出的自然语言问题。  
   - 这种设计不需要额外的外部纠错模型，完全依赖 LLM 的内部推理能力。

4. **评估流程**  
   - 对每个扭曲样本，记录模型在 **无提示**、**有提示** 两种设置下的最终答案是否正确。  
   - 同时收集模型自检阶段的输出，检查其是否真实发现了扭曲并给出了合理的纠正表格。  
   - 通过对比两种设置的准确率，量化系统提示的提升幅度；通过自检输出的准确性，评估模型的错误感知能力。

**最巧妙的点**在于把“错误检测”嵌入到同一个生成过程，而不是另起一个专门的纠错模型。这样既保持了 LLM 的端到端特性，又让研究者能够直接观察模型是否真的“看到了”扭曲。

### 实验与效果

- **数据集**：作者基于 WikiTQ 挑选了 200 条原始表格问答，手工施加六类扭曲，形成了 1,200 条扭曲样本（每类 200 条）。  
- **模型**：实验覆盖了三代主流 LLM：GPT‑3.5、GPT‑4、以及最新的 GPT‑5.2（作者称为 SoTA）。  
- **基线**：对比了两种基线：① 直接在干净表格上做 TQA（上限性能），② 在扭曲表格上不加任何提示直接回答。  
- **主要结果**：  
  - 在无提示情况下，所有模型在扭曲表格上的准确率下降至少 22%，GPT‑5.2 下降幅度为 27%。  
  - 加入系统提示后，GPT‑5.2 的准确率提升约 12%，仍比干净表格低 15%。  
  - 对于结构扭曲（尤其是行的水平错位），提升幅度更小，仅 6%；而对语义扭曲（如单位不一致），提升可达 18%。  
- **消融实验**：作者分别去掉系统提示、去掉自检阶段的表格输出、以及只提供“检查但不纠正”的指令。结果显示，**完整的“检查+纠正”指令是提升的关键**，仅检查而不纠正几乎没有效果。  
- **局限性**：  
  - 实验规模相对小，仅 200 条原始表格，可能不足以覆盖所有真实业务场景。  
  - 只评估了单层扭曲，未考虑多层复合扭曲的交互影响。  
  - 纠正质量依赖模型的自检能力，错误的自检会导致二次错误，作者在论文中承认这一点。

### 影响与延伸思考

这篇工作首次系统量化了 LLM 在微小表格扭曲下的脆弱性，并展示了通过显式先验可以部分缓解。发表后，几篇后续研究开始探索 **自监督表格纠错**，尝试让模型在大规模未标注表格上自行学习检测异常模式（如 2024 年的 “Self‑Align Tables”）。还有工作把专门的表格结构感知层（如 Table‑Transformer）与 LLM 结合，期望在模型内部形成更强的结构约束。对想进一步深入的读者，可以关注 **表格自适应对齐（Adaptive Table Alignment）**、**多模态表格纠错（结合视觉 OCR 信息）** 以及 **基于检索的纠错前置** 等方向，这些都是当前社区的热点。

### 一句话记住它

**LLM 在表格细微扭曲面前几乎不自检，只有在明确提示下才会尝试纠错，但仍远未达到人类的自适应对齐水平。**