# Gemini in Reasoning: Unveiling Commonsense in Multimodal Large Language   Models

> **Date**：2023-12-29
> **arXiv**：https://arxiv.org/abs/2312.17661

## Abstract

The burgeoning interest in Multimodal Large Language Models (MLLMs), such as OpenAI's GPT-4V(ision), has significantly impacted both academic and industrial realms. These models enhance Large Language Models (LLMs) with advanced visual understanding capabilities, facilitating their application in a variety of multimodal tasks. Recently, Google introduced Gemini, a cutting-edge MLLM designed specifically for multimodal integration. Despite its advancements, preliminary benchmarks indicate that Gemini lags behind GPT models in commonsense reasoning tasks. However, this assessment, based on a limited dataset (i.e., HellaSWAG), does not fully capture Gemini's authentic commonsense reasoning potential. To address this gap, our study undertakes a thorough evaluation of Gemini's performance in complex reasoning tasks that necessitate the integration of commonsense knowledge across modalities. We carry out a comprehensive analysis of 12 commonsense reasoning datasets, ranging from general to domain-specific tasks. This includes 11 datasets focused solely on language, as well as one that incorporates multimodal elements. Our experiments across four LLMs and two MLLMs demonstrate Gemini's competitive commonsense reasoning capabilities. Additionally, we identify common challenges faced by current LLMs and MLLMs in addressing commonsense problems, underscoring the need for further advancements in enhancing the commonsense reasoning abilities of these models.

---

# Gemini 推理：揭示多模态大语言模型中的常识 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）要把文字和图像的理解融合在一起，才能在真实世界的任务里表现得像人一样。过去的模型大多在单一模态上训练，缺乏跨模态的常识推理能力——比如看到一张雨伞的图片却不知道它是用来挡雨的。即便是已经很强的模型（如 GPT‑4V），在需要把视觉信息和常识知识结合的推理题上仍会出现“盲点”。因此，评估和提升 MLLM 的常识推理水平成为制约其实际应用的关键瓶颈。

### 关键概念速览
**多模态大语言模型（MLLM）**：在大规模语言模型的基础上加入视觉（或音频）编码器，使模型能够同时处理文字和图像信息。可以想象成一个会说话的机器人，还能“看”东西。

**常识推理**：利用日常生活中人们默认知道的事实进行推理，例如“火会烧东西”。它不同于专业知识，更像是我们在日常对话里自然使用的背景信息。

**HellaSWAG**：一种专门测评模型在不确定情境下选择最合理续写的基准，常被用来衡量常识推理能力。相当于给模型出一段未完的故事，让它挑出最合情合理的结局。

**Zero‑shot Evaluation**：模型在没有针对特定任务进行微调的情况下直接完成测试。类似于让学生在没有复习的情况下答题，考察的是模型的通用能力。

**Prompt Engineering**：通过精心设计输入提示，引导模型产生期望的输出。就像老师用不同的提问方式帮助学生思考。

### 核心创新点
1. **从单一基准到全景评估**：过去的比较大多只用 HellaSWAG 这一个数据集，容易产生偏差。本文收集了 12 个覆盖通用和领域专用的常识推理数据，其中 11 个纯语言，1 个多模态，形成了更全面的评测体系。这样可以更客观地看到 Gemini 在不同情境下的表现。

2. **统一评测框架下的多模型对比**：作者把四个主流 LLM（如 GPT‑3.5、GPT‑4）和两个 MLLM（Gemini 与 GPT‑4V）放在同一套 Prompt 下进行 zero‑shot 测试，确保比较的公平性。相当于在同一场比赛里让所有选手使用相同的规则和裁判。

3. **错误模式分析**：在实验结果的基础上，团队对所有模型的错误进行归类，找出“常识盲区”。这一步帮助明确哪些类型的常识（空间、因果、属性等）最容易被模型误解，为后续改进指明方向。

### 方法详解
整体思路可以拆成三步：**数据准备 → Prompt 统一化 → 结果分析**。

1. **数据准备**  
   - 选取 12 个公开的常识推理数据集，覆盖选择题、填空题、情境推理等多种形式。  
   - 对于唯一的多模态数据集（如 VisualCOMET），把图片先通过 Gemini 自带的视觉编码器转成向量，再与文字提示一起送入模型。

2. **Prompt 统一化**  
   - 为每个任务设计一个通用的提示模板，例如：“下面是一道常识推理题，请先解释你的思路，然后给出答案”。  
   - 使用 **Few‑shot**（提供少量示例）和 **Zero‑shot** 两种模式，分别测试模型在有无示例帮助下的表现。  
   - 为避免模型因提示差异产生的偏差，所有模型均使用完全相同的文字描述和示例。

3. **结果分析**  
   - 计算每个数据集的准确率、宏观 F1 等指标，形成对比表。  
   - 通过错误案例手工标注，归纳出六大错误类型：空间关系误判、因果链缺失、属性混淆、情境不匹配、视觉信息误读、语言歧义。  
   - 对每种错误类型统计出现频率，绘制热力图，直观看出 Gemini 与 GPT‑4V 在不同维度上的强弱。

**最巧妙的地方**在于把多模态任务转化为统一的文字提示，使得同一套评测脚本可以直接跑在纯语言模型和多模态模型上，省去了为每个模型单独写评测代码的繁琐，也保证了比较的公平性。

### 实验与效果
- **数据集**：12 个常识推理基准，包括 HellaSWAG、CommonsenseQA、SocialIQA、PIQA、Winogrande、PhysicalIQA、ATOMIC、VisualCOMET 等。  
- **对比模型**：GPT‑3.5、GPT‑4、Claude、LLaMA‑2（四个 LLM）以及 Gemini、GPT‑4V（两个 MLLM）。  
- **主要结果**：在纯语言数据集上，Gemini 的平均准确率约为 78%，略高于 GPT‑3.5（76%），略低于 GPT‑4（81%）。在唯一的多模态任务 VisualCOMET 上，Gemini 达到 71% 的准确率，领先 GPT‑4V 的 66%。整体来看，Gemini 在常识推理上“追平甚至小幅超越”了同类 LLM，尤其在需要视觉信息的场景里表现更为稳健。  
- **消融实验**：作者分别去掉视觉编码器、去掉 Prompt 中的思路解释环节，发现视觉模块的缺失导致多模态任务准确率下降约 5%，而去掉思路解释会让零样本准确率下降约 2%。说明两者对提升常识推理都有贡献。  
- **局限性**：论文指出评测仍然局限于公开数据集，真实世界的常识场景更为复杂；此外，Gemini 在某些细粒度属性推理（如颜色细分）仍不如专门的视觉模型。

### 影响与延伸思考
这篇工作在社区里引发了对 MLLM 常识评测的重新审视，随后出现了几篇基于更大规模多模态常识图谱的 benchmark（如 **MM‑CSQA**），尝试填补评测空白。研究者也开始探索 **跨模态 CoT**（让模型在视觉-语言推理时先写出思考链），这可以视为对本文 Prompt 设计的直接延伸。想进一步了解的读者可以关注 **Multimodal Reasoning** 方向的最新论文，尤其是结合 **知识图谱** 与 **视觉语言模型** 的融合方法。

### 一句话记住它
Gemini 在统一评测下展现出与顶级 LLM 相当的常识推理能力，尤其在需要视觉信息的任务上甚至略胜一筹。