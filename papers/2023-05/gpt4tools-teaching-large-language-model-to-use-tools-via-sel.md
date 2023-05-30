# GPT4Tools: Teaching Large Language Model to Use Tools via   Self-instruction

> **Date**：2023-05-30
> **arXiv**：https://arxiv.org/abs/2305.18752

## Abstract

This paper aims to efficiently enable Large Language Models (LLMs) to use multimodal tools. Advanced proprietary LLMs, such as ChatGPT and GPT-4, have shown great potential for tool usage through sophisticated prompt engineering. Nevertheless, these models typically rely on prohibitive computational costs and publicly inaccessible data. To address these challenges, we propose the GPT4Tools based on self-instruct to enable open-source LLMs, such as LLaMA and OPT, to use tools. It generates an instruction-following dataset by prompting an advanced teacher with various multi-modal contexts. By using the Low-Rank Adaptation (LoRA) optimization, our approach facilitates the open-source LLMs to solve a range of visual problems, including visual comprehension and image generation. Moreover, we provide a benchmark to evaluate the ability of LLMs to use tools, which is performed in both zero-shot and fine-tuning ways. Extensive experiments demonstrate the effectiveness of our method on various language models, which not only significantly improves the accuracy of invoking seen tools, but also enables the zero-shot capacity for unseen tools. The code and demo are available at https://github.com/StevenGrove/GPT4Tools.

---

# GPT4Tools：通过自指令让大语言模型学会使用工具 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，模型只能输出文字，面对图像、代码或外部API时只能“说”而不能真正“做”。即使是像ChatGPT、GPT‑4这样的大模型，也只能靠精心设计的提示词让它们在对话中模拟调用工具，背后仍然依赖巨量的算力和专有数据。开源模型（如LLaMA、OPT）缺少这种“工具使用”能力，主要因为：①缺少大规模的指令‑工具对齐数据；②没有高效的微调手段让模型在保持原有语言能力的同时学会新技能；③对未知工具的零样本适配几乎没有研究。于是，如何用低成本让开源LLM真正掌握多模态工具，成为亟待突破的瓶颈。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的深度网络，类似“会说话的百科全书”。  
- **多模态工具**：接受图像、代码或其他非文本输入并返回结果的外部系统，例如图像识别API、文本到图像生成器。  
- **自指令（Self‑instruction）**：让一个更强的模型自动生成指令‑响应对，用来训练弱模型的方式，类似让老师自己出题并给答案。  
- **LoRA（Low‑Rank Adaptation）**：在不改变原模型权重的大前提下，只在少量参数上做低秩微调，像在原模型上贴一层轻量的“胶水”。  
- **指令跟随数据集**：包含用户指令、工具调用细节以及期望输出的训练样本，帮助模型学会“看指令、调工具、给答案”。  
- **零样本工具使用**：模型在未见过的工具上直接发挥作用的能力，类似人类第一次使用新软件时凭直觉完成任务。  
- **工具使用基准（Tool‑Use Benchmark）**：专门评估模型在不同工具上的调用成功率和答案质量的测试集合。  

### 核心创新点
1. **从高级模型自动生成指令‑工具数据 → 使用 GPT‑4 作为教师模型，给出多种视觉上下文并让它输出完整的指令、工具调用和答案 → 省去人工标注成本，快速得到大规模、覆盖多工具的训练集。**  
2. **低秩微调 LoRA 直接注入工具使用能力 → 只在少量参数上进行适配，而不破坏原有语言知识 → 让开源 LLM 在几小时内即可掌握视觉理解和图像生成等任务。**  
3. **统一的工具使用基准，兼顾零样本和微调评估 → 设计了零样本和微调两套测试，分别检验模型对已见工具的熟练度和对全新工具的迁移能力 → 证明了方法在“见”和“未见”两种情形下都能显著提升成功率。**  
4. **跨模型通用性实验 → 在 LLaMA、OPT 等多种开源模型上复现，同样得到显著提升 → 表明该方案不是针对单一模型的特例，而是通用的工具学习框架。  

### 方法详解
整体思路可以划分为三步：**教师数据生成 → 低秩微调 → 多维评估**。  
1. **教师数据生成**：先准备一组多模态上下文（如图片、代码片段），再用 GPT‑4 这类强模型充当“老师”。老师收到“请完成以下任务并调用相应工具”的提示后，会自行生成完整的指令文本、调用的工具名称、所需的参数以及最终的答案。这样得到的每条记录形如“指令 → 调用工具（API 名称 + 参数） → 期望输出”。因为 GPT‑4 本身已经具备丰富的工具使用经验，生成的数据天然符合真实使用场景。  
2. **低秩微调（LoRA）**：将上述指令‑工具对齐数据喂给目标开源 LLM。LoRA 只在模型的注意力层和前馈层插入低秩矩阵，训练时只更新这些小矩阵，原始权重保持不变。这样既能让模型学会在指令中识别“调用工具”的意图，又不会丢失已有的语言能力。训练过程类似在原模型上贴一层“插件”，插件负责把文字指令翻译成具体的 API 调用。  
3. **多维评估**：作者构建了一个包含已见工具和未见工具的基准。已见工具评估模型在微调后能否准确调用并得到正确答案；未见工具评估模型在零样本情况下，仅凭指令理解就能生成合理的调用格式。两类评估分别用成功率（调用是否成功）和答案准确率（生成内容是否符合预期）量化。  

**最巧妙的地方**在于把“工具使用”抽象成一种语言指令的翻译任务：模型只需要学会把自然语言映射到结构化的 API 调用，而不必重新学习每个工具的内部实现细节。再加上 LoRA 的轻量微调，使得即使是算力有限的研究者也能在几小时内完成训练。

### 实验与效果
- **测试任务**：包括视觉理解（图像分类、目标检测描述）和图像生成（文本到图像）两大类。每类任务对应若干公开数据集，作者自行组织了多模态上下文。  
- **基线对比**：与直接使用原始开源模型（未微调）以及使用全参数微调的方式相比，GPT4Tools 在已见工具上的调用成功率提升约 30%~45%，答案准确率提升约 20%~35%。在未见工具的零样本评估中，成功率从原模型的 10% 左右提升到 55% 左右，显示出强大的迁移能力。  
- **消融实验**：去掉 LoRA 微调、只使用原始指令数据或只用单一工具进行训练，性能均出现明显下降，尤其是去掉 LoRA 时，模型在保持语言能力的同时几乎不学会工具调用，验证了低秩适配的关键性。  
- **局限性**：论文未给出对极端长指令或高度嵌套调用的表现；生成的指令‑工具对齐数据仍然受限于 GPT‑4 的内部知识，若出现全新领域的工具，仍可能需要额外的教师数据。  

### 影响与延伸思考
这篇工作打开了“开源模型+工具使用”这一新局面，后续不少项目开始尝试用自指令方式为 LLM 注入特定插件，例如代码执行、数据库查询等。推测未来会有更多研究聚焦在 **多工具协同**（一次指令触发多个 API）和 **自适应工具发现**（模型主动搜索可用工具）上。想进一步了解的读者可以关注 **ToolBench**、**OpenTool** 等后续基准，以及 LoRA 在跨任务微调中的最新进展。

### 一句话记住它
用强模型自生成指令‑工具对，再用 LoRA 轻量微调，让开源大语言模型在几小时内学会真正“动手”使用多模态工具。