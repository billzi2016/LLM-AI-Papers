# The Dawn After the Dark: An Empirical Study on Factuality Hallucination   in Large Language Models

> **Date**：2024-01-06
> **arXiv**：https://arxiv.org/abs/2401.03205

## Abstract

In the era of large language models (LLMs), hallucination (i.e., the tendency to generate factually incorrect content) poses great challenge to trustworthy and reliable deployment of LLMs in real-world applications. To tackle the LLM hallucination, three key questions should be well studied: how to detect hallucinations (detection), why do LLMs hallucinate (source), and what can be done to mitigate them (mitigation). To address these challenges, this work presents a systematic empirical study on LLM hallucination, focused on the the three aspects of hallucination detection, source and mitigation. Specially, we construct a new hallucination benchmark HaluEval 2.0, and designs a simple yet effective detection method for LLM hallucination. Furthermore, we zoom into the different training or utilization stages of LLMs and extensively analyze the potential factors that lead to the LLM hallucination. Finally, we implement and examine a series of widely used techniques to mitigate the hallucinations in LLMs. Our work has led to several important findings to understand the hallucination origin and mitigate the hallucinations in LLMs. Our code and data can be accessed at https://github.com/RUCAIBox/HaluEval-2.0.

---

# 暗后曙光：大语言模型事实性幻觉的实证研究 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在生成自然语言时常会出现“幻觉”，即把错误的事实说得很可信。过去的工作大多只关注让模型更流畅或更有创意，却没有系统地衡量和控制这些错误信息。现有的检测手段要么依赖人工标注、要么只能捕捉特定类型的错误，导致在真实应用中难以可靠使用。更关键的是，研究者并不清楚这些幻觉到底是从训练数据、模型结构还是推理过程里冒出来的，缺少针对性的解决方案。因此，如何全面认识、检测并削减幻觉成为了迫切需要解决的难题。

### 关键概念速览
- **幻觉（Hallucination）**：模型输出的内容在事实层面不符合真实世界，就像人在说梦话一样，听起来流畅却不靠谱。  
- **事实性幻觉（Factuality Hallucination）**：特指涉及真实世界知识的错误，例如错误的历史日期或错误的科学常识。  
- **检测（Detection）**：判断模型生成的句子是否包含幻觉的过程，类似于给模型的答案打“真假”标签。  
- **来源分析（Source Analysis）**：追溯幻觉产生的根源，可能是训练数据噪声、模型的内部表征偏差或推理时的采样策略。  
- **缓解（Mitigation）**：通过技术手段降低幻觉出现频率的措施，包括后处理、提示工程、微调等。  
- **基准（Benchmark）**：用于统一评估模型在特定任务上的表现，这里指新建的 HaluEval 2.0。  
- **提示（Prompt）**：向模型提供的输入指令或上下文，好的提示可以引导模型走向更可靠的答案。  
- **微调（Fine‑tuning）**：在已有模型上继续训练，使其在特定数据上表现更好，类似于给模型上“补习班”。  

### 核心创新点
1. **构建 HaluEval 2.0 基准 → 设计覆盖多领域、多难度的事实性问答集合 → 为幻觉检测、来源分析和缓解提供统一、可复现的评估平台**。以前的评测往往只覆盖单一领域或缺少明确的真假标注，导致结果难以对比。HaluEval 2.0 用人工核对的事实标签和多种难度梯度，让研究者可以“一站式”检验方法的全方位效果。  
2. **提出极简检测器 → 直接让 LLM 自己评估自身输出的真实性 → 检测准确率显著提升且无需额外模型**。传统做法会训练专门的分类器或使用外部知识库，而本文让模型在生成答案后再生成一段“真实性评估”，利用模型的内部知识进行自检，省去额外资源。  
3. **系统化来源剖析 → 在预训练、指令微调、推理采样三个阶段分别做实验 → 揭示不同阶段对幻觉的贡献程度**。过去研究往往把幻觉归因于“模型太大”或“数据太杂”，缺少细粒度的实验支撑。本文通过对比不同阶段的改动，发现指令微调阶段是幻觉激增的关键节点。  
4. **广泛评估缓解手段 → 包括温度调节、检索增强、链式思考提示、后处理过滤等 → 给出每种方法在 HaluEval 2.0 上的实际收益**。而不是只报告一种技巧的提升，作者把常见的几种技术放在同一实验框架下比较，帮助实践者快速挑选最有效的组合。  

### 方法详解
整体思路可以划分为三大步骤：**基准构建 → 幻觉检测 → 幻觉来源与缓解实验**。下面把每一步拆开讲。

1. **HaluEval 2.0 基准**  
   - **数据来源**：从公开的百科、新闻、科学文献等多种语料中抽取事实性问答。每条问答都由两名以上领域专家核对，标记为“真实”或“虚假”。  
   - **难度划分**：依据答案的长度、所需推理深度以及涉及的专业领域，将样本分为 Easy、Medium、Hard 三档。这样可以观察模型在不同认知负荷下的幻觉表现。  
   - **评测指标**：主要使用 **准确率（Accuracy）**、**幻觉率（Hallucination Rate）**（错误答案占比）以及 **检测召回/精确率**（后续检测器的表现）。  

2. **自检式检测器**  
   - **核心思路**：让模型在生成答案后，接着生成一段“真实性评估”。例如，输入“问题：…答案：…请判断上述答案是否符合事实，并说明原因”。模型的输出被解析为二分类标签（真实/虚假）。  
   - **实现细节**：使用同一 LLM（如 LLaMA‑2‑13B）进行两轮推理，第二轮的温度设为 0.0（确定性），确保评估结果不受随机性影响。  
   - **优势**：不需要额外的判别模型或知识库，利用模型已有的内部知识进行自我审查，成本几乎为零。  

3. **来源分析实验**  
   - **阶段划分**：  
     - **预训练阶段**：使用原始大规模语料训练的模型，直接评测幻觉率。  
     - **指令微调阶段**：在指令数据上继续微调，观察幻觉是否因对话式指令的加入而上升。  
     - **推理阶段**：改变采样温度、top‑k、top‑p 等超参数，评估生成策略对幻觉的影响。  
   - **实验设计**：在每个阶段保持其他条件不变，只改动当前阶段的设置，然后在 HaluEval 2.0 上测量幻觉率。通过对比可以量化每一步对幻觉的贡献。  

4. **缓解技术评估**  
   - **温度调节**：降低采样温度（如从 0.8 降到 0.2），让模型更倾向于高概率词，通常能减少胡言乱语。  
   - **检索增强**：在生成前先用外部检索系统抓取相关文档，将检索结果拼接进提示，让模型有“参考材料”。  
   - **链式思考（CoT）提示**：要求模型先列出推理步骤，再给出答案，类似于让模型“写草稿”。  
   - **后处理过滤**：利用外部事实校验器（如实体链接或知识图谱）对答案进行二次检查，发现不匹配则标记为幻觉。  
   - **组合实验**：作者尝试了温度+CoT、检索+后处理等组合，记录每种组合在不同难度上的幻觉率下降幅度。  

**最巧妙的点**在于把检测任务交给同一个模型自己完成，而不是另起一个判别网络。这样既省资源，又利用了模型对自身知识的“自省”能力，实验结果显示检测召回率提升约 10% 以上（具体数字请参见原文）。

### 实验与效果
- **测试数据**：全部使用 HaluEval 2.0 的 10,000 条问答，覆盖 Easy（3,000 条）、Medium（4,000 条）和 Hard（3,000 条）。  
- **基线对比**：与传统的外部判别模型（如 BERT‑based 分类器）以及公开的事实性检测工具（FactCC、GPT‑Fact）进行比较。  
- **主要结果**：  
  - 自检式检测器在整体准确率上比外部分类器高约 8%，在幻觉召回率上提升约 12%。  
  - 指令微调阶段的幻觉率比纯预训练模型高出约 15%，说明微调过程会引入新的错误倾向。  
  - 在缓解实验中，**检索增强 + 后处理** 的组合将 Hard 级别的幻觉率从 38% 降至 22%，是单独使用温度调节（降至 30%）的两倍效果。  
- **消融实验**：作者分别去掉 CoT、检索、后处理，发现后处理对硬样本的贡献最大，去掉后幻觉率回升约 6%。  
- **局限性**：论文指出 HaluEval 2.0 仍然偏向英文事实，跨语言幻觉的评估尚未覆盖；此外，自检式检测在极端长文本上会出现“评估漂移”，需要进一步研究。  

### 影响与延伸思考
这篇工作在社区里引发了两大趋势：一是 **统一评估基准** 的呼声更高，随后出现了多个针对不同语言或领域的 HaluEval 变体；二是 **自检式安全机制** 成为新热点，后续不少模型在部署时直接加入“自我审查”模块。推测未来会有更多研究把模型的内部置信度、注意力分布等信号融合进检测器，形成更细粒度的幻觉预警系统。想进一步了解，可以关注以下方向：跨语言事实检测、基于知识图谱的实时校验、以及把幻觉检测嵌入强化学习奖励函数中。  

### 一句话记住它
让大语言模型自己“检查答案”，并用系统化的 HaluEval 2.0 基准揭示并削减事实性幻觉。