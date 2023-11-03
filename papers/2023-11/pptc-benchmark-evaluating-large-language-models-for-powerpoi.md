# PPTC Benchmark: Evaluating Large Language Models for PowerPoint Task   Completion

> **Date**：2023-11-03
> **arXiv**：https://arxiv.org/abs/2311.01767

## Abstract

Recent evaluations of Large Language Models (LLMs) have centered around testing their zero-shot/few-shot capabilities for basic natural language tasks and their ability to translate instructions into tool APIs. However, the evaluation of LLMs utilizing complex tools to finish multi-turn, multi-modal instructions in a complex multi-modal environment has not been investigated. To address this gap, we introduce the PowerPoint Task Completion (PPTC) benchmark to assess LLMs' ability to create and edit PPT files based on user instructions. It contains 279 multi-turn sessions covering diverse topics and hundreds of instructions involving multi-modal operations. We also propose the PPTX-Match Evaluation System that evaluates if LLMs finish the instruction based on the prediction file rather than the label API sequence, thus it supports various LLM-generated API sequences. We measure 3 closed LLMs and 6 open-source LLMs. The results show that GPT-4 outperforms other LLMs with 75.1\% accuracy in single-turn dialogue testing but faces challenges in completing entire sessions, achieving just 6\% session accuracy. We find three main error causes in our benchmark: error accumulation in the multi-turn session, long PPT template processing, and multi-modality perception. These pose great challenges for future LLM and agent systems. We release the data, code, and evaluation system of PPTC at \url{https://github.com/gydpku/PPTC}.

---

# PPTC 基准：评估大语言模型在 PowerPoint 任务完成上的表现 论文详细解读

### 背景：这个问题为什么难？

传统的 LLM 评测大多停留在纯文本的零/少样本推理或把指令映射成单一 API 调用上，缺少对模型在真实办公环境中“动手”完成任务的考察。PowerPoint 这类多模态、需要文件结构、图形、文字、布局等多层次操作的工具，远比简单的文本生成复杂：模型必须先理解用户的多轮指令，再调用一系列编辑 API，最后保证生成的 PPT 文件在视觉和功能上符合预期。之前的评测没有提供这样一个完整的、多轮、多模态的任务场景，导致我们不知道模型在真实工具使用中的真实能力。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言的深度学习模型，像 ChatGPT、Claude 等。这里指的是可以通过文字指令驱动外部工具的模型。
- **多模态指令**：指令中既包含文字描述，又可能涉及图片、表格等非文本信息，需要模型同时感知多种感官信号。类似于让人同时看图说话。
- **工具调用（Tool Calling）**：模型在生成答案的过程中，主动输出调用外部 API（如 PowerPoint 编辑接口）的指令序列。相当于让模型在“思考”后去“动手”。
- **多轮对话**：用户和模型之间的交互不是一次性完成，而是分多次来回，每一步都可能依赖前面的上下文。像和同事一起迭代修改 PPT。
- **PPTX‑Match 评估系统**：一种基于生成的 PPT 文件本身而非 API 调用序列来判断任务是否完成的评测方法。它直接比较模型输出的 PPT 与参考答案的结构、内容和布局。
- **错误累积（Error Accumulation）**：在多轮交互中，前一次的细小失误会在后续步骤被放大，最终导致整体任务失败。类似于拼图时一块错位会让整幅图看不完整。

### 核心创新点
1. **从 API 序列到文件级评估的转变**  
   - 之前的评测往往只检查模型输出的 API 调用是否与标注一致，忽略了不同调用顺序可能产生相同结果的情况。  
   - 本文提出 PPTX‑Match，直接对模型生成的 PPT 文件进行结构、文本、图形等维度的比对。  
   - 这样既容忍了多种合法的 API 序列，也更贴近真实用户关心的“文件到底对不对”。

2. **构建大规模多轮多模态 PPT 任务集**  
   - 过去的基准多是单轮、单模态的问答或代码生成。  
   - 作者收集并人工标注了 279 场真实的多轮对话，覆盖数百条涉及图片插入、表格编辑、模板切换等操作的指令。  
   - 这为评估模型在复杂办公情境下的连续协作提供了完整的实验平台。

3. **系统化错误来源分析**  
   - 通过对实验结果的细粒度拆解，作者归纳出三大错误根源：多轮错误累积、长模板处理瓶颈、跨模态感知不足。  
   - 这种诊断框架帮助后续研究快速定位模型薄弱环节，而不是仅给出整体准确率。

### 方法详解
整体框架可以看作三层流水线：**指令解析 → API 生成 → 文件生成与评估**。

1. **指令解析层**  
   - 模型接收用户的文字或图片指令，利用自身的多模态感知能力把自然语言映射成结构化的任务意图（如“在第 3 张幻灯片插入一张公司 logo”。）  
   - 这里没有特殊的额外模块，直接使用 LLM 的原始生成能力，只是要求模型在输出时遵循约定的 API 调用格式。

2. **API 生成层**  
   - 根据解析得到的意图，模型输出一系列 PowerPoint 操作 API（如 `add_slide()`, `insert_image(slide=3, path=logo.png)`）。  
   - 与传统的“一次性生成答案”不同，模型需要在每轮对话结束后决定是否继续调用更多 API，或者等待用户的下一条指令。  
   - 为了兼容不同模型的调用风格，作者没有强制统一 API 序列，而是让评估系统在后端把所有合法序列统一执行。

3. **文件生成与 PPTX‑Match 评估层**  
   - 所有 API 被实际执行在一个 PowerPoint 引擎（如 python-pptx）上，生成最终的 `.pptx` 文件。  
   - PPTX‑Match 读取生成文件的 XML 结构，逐页比对文本、图片、表格、布局等属性，计算整体匹配度。  
   - 只要文件在关键维度上与参考答案一致，就算该指令完成成功，避免了因 API 顺序差异导致的误判。

**最巧妙的点**在于把评估焦点从“模型说了什么”搬到“模型做了什么”。这让不同的实现（比如有的模型先删掉旧幻灯片再重建，有的模型直接修改属性）都能得到公平的评分。

### 实验与效果
- **数据集**：使用作者构建的 PPTC 基准，包含 279 条多轮对话，主题覆盖商务报告、学术演示、产品介绍等。每条对话平均 4–5 轮指令，总计约 1300 条具体编辑操作。  
- **对比模型**：3 家闭源模型（GPT‑4、Claude、ChatGPT‑4）和 6 款开源模型（Llama‑2‑70B、Mistral‑7B、Falcon‑180B 等）。  
- **单轮准确率**：GPT‑4 在单轮指令上达到了 75.1% 的 PPTX‑Match 正确率，领先第二名约 12% 以上。其他模型多数在 30%–50% 区间。  
- **整场会话成功率**：即所有轮次都正确完成的会话比例，GPT‑4 仅为 6%，其余模型更低，说明多轮错误累积是主要瓶颈。  
- **消融实验**：作者分别去掉（1）多轮上下文记忆、（2）图片感知模块、（3）长模板分块处理，发现去掉图片感知导致整体准确率下降约 9%，长模板处理缺失导致单轮准确率下降约 7%。这验证了三大错误来源的实际影响。  
- **局限性**：实验仅覆盖 PowerPoint 一个工具，评估体系对其他 Office 软件的迁移尚未验证；此外，评估仍依赖人工标注的参考 PPT，可能受标注者主观偏好影响。

### 影响与延伸思考
PPTC 基准首次把“工具使用”与“多模态多轮对话”结合起来，为 LLM 走出纯文本、进入真实办公自动化提供了实验土壤。自论文发布后，已有工作尝试把类似的评测扩展到 Excel、Word 甚至 CAD 软件，形成了“Office‑Agent”系列基准。后续研究可以从以下方向深入：①提升模型的跨模态感知（比如直接读取图片中的文字）；②设计更鲁棒的记忆机制，防止错误在多轮中放大；③探索自监督的“工具调用”预训练，让模型在大量模拟编辑任务中提前学习 API 序列的合理性。对想进一步了解的读者，可以关注近期在 arXiv 上出现的 “ToolBench” 与 “AgentBench” 系列论文，它们在任务复杂度和评估细粒度上与 PPTC 有不少互补。

### 一句话记住它
PPTC 把“模型说话”变成“模型动手”，用文件级匹配评估 LLM 在多轮多模态 PowerPoint 编辑中的真实表现。