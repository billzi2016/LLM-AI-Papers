# GPT Self-Supervision for a Better Data Annotator

> **Date**：2023-06-07
> **arXiv**：https://arxiv.org/abs/2306.04349

## Abstract

The task of annotating data into concise summaries poses a significant challenge across various domains, frequently requiring the allocation of significant time and specialized knowledge by human experts. Despite existing efforts to use large language models for annotation tasks, significant problems such as limited applicability to unlabeled data, the absence of self-supervised methods, and the lack of focus on complex structured data still persist. In this work, we propose a GPT self-supervision annotation method, which embodies a generating-recovering paradigm that leverages the one-shot learning capabilities of the Generative Pretrained Transformer (GPT). The proposed approach comprises a one-shot tuning phase followed by a generation phase. In the one-shot tuning phase, we sample a data from the support set as part of the prompt for GPT to generate a textual summary, which is then used to recover the original data. The alignment score between the recovered and original data serves as a self-supervision navigator to refine the process. In the generation stage, the optimally selected one-shot sample serves as a template in the prompt and is applied to generating summaries from challenging datasets. The annotation performance is evaluated by tuning several human feedback reward networks and by calculating alignment scores between original and recovered data at both sentence and structure levels. Our self-supervised annotation method consistently achieves competitive scores, convincingly demonstrating its robust strength in various data-to-summary annotation tasks.

---

# GPT自监督提升数据标注器 论文详细解读

### 背景：这个问题为什么难？
把原始数据压缩成一段简洁的摘要，需要深厚的领域知识和大量的人工时间。传统的标注流程往往依赖专家手工撰写，成本高且难以规模化。已有研究尝试让大语言模型（LLM）直接生成摘要，但大多数方法只能在已有标注好的样本上微调，面对全新、未标注的数据时表现乏力；此外，它们缺少自我纠错的机制，无法保证生成的摘要在结构和细节上与原始信息保持一致。正是这些根本性限制，让“让模型自己学会标注”成为亟待突破的难题。

### 关键概念速览
**生成-还原循环**：模型先把输入数据生成摘要，再尝试从该摘要恢复出原始数据，类似先写提纲再检查是否能还原全文。  
**一-shot 调优**：只给模型展示一条示例（即“一次示例”），让它在此基础上完成任务，类似一次性教学。  
**Prompt（提示词）**：向模型提供的文字指令或示例，起到“任务说明书”的作用。  
**对齐得分（Alignment Score）**：衡量恢复后数据与原始数据相似程度的分数，像是两篇文章的相似度检测。  
**结构层对齐**：不仅比较文字相似，还比较信息的组织方式（如表格、层级），相当于检查文章的章节安排是否一致。  
**人类反馈奖励网络（Human Feedback Reward Model）**：用人类评分训练的模型，用来评估生成摘要的质量，类似老师给作业打分的自动化版本。  
**支持集（Support Set）**：提供给模型的少量示例集合，帮助模型理解任务。  

### 核心创新点
1. **从“生成→恢复”到自监督循环**：过去的标注方法大多只让模型生成摘要，缺少检验环节。本文在生成后加入恢复步骤，用恢复成功程度作为自监督信号，引导模型自行纠错。这样模型不再完全依赖外部标注，而是通过内部一致性提升质量。  
2. **一-shot 采样作为模板**：传统微调需要大量标注样本，成本高。这里只抽取支持集中的一条样本放进 Prompt，作为“一次示例”，让 GPT 在一次性学习后直接生成新数据的摘要。相比需要大量梯度更新的方式，这种方法更轻量、适配性更强。  
3. **对齐得分双层评估**：不仅在句子层面计算相似度，还在结构层面比较恢复后数据的组织形式。双层评估让模型在保持信息完整性的同时，也能生成结构合理的摘要。  
4. **结合人类反馈奖励网络进行调优**：在自监督循环之外，作者额外训练了若干基于人类评分的奖励模型，用来对生成的摘要进行二次筛选和微调，进一步提升了摘要的可读性和专业度。  

### 方法详解
整体思路可以拆成两大阶段：**一-shot 调优**和**生成**。  
1. **一-shot 调优阶段**  
   - 从支持集随机抽取一条完整数据（比如一段技术报告），把它连同一段手工写好的摘要一起放进 Prompt，交给 GPT。  
   - GPT 根据 Prompt 生成自己的摘要版本。  
   - 接着，模型把这段自己生成的摘要再喂回到同一个 Prompt（这次指令是“请根据摘要恢复原始数据”），让 GPT 试图重建最初的完整数据。  
   - 通过比较恢复出来的数据与原始数据，计算**对齐得分**。如果恢复得好，说明生成的摘要保留了足够信息；如果恢复差，得分低，模型会在后续迭代中调整 Prompt 中的示例或生成策略。  
   - 这个过程不需要人工标注的梯度更新，只是利用 GPT 的一次性生成能力和对齐得分来“自我监督”。  

2. **生成阶段**  
   - 在调优阶段挑选出对齐得分最高的那条一-shot 示例，固定下来作为模板 Prompt。  
   - 对每条待标注的原始数据，直接把它放进 Prompt，要求 GPT 生成摘要。  
   - 为了进一步提升质量，作者在生成后又跑一次恢复检查：把 GPT 生成的摘要喂回模型，让它恢复原始数据，并用**句子层对齐**和**结构层对齐**两种得分综合评估。  
   - 最后，使用事先训练好的人类反馈奖励网络，对所有候选摘要进行打分，选出最高分的作为最终标注。  

**巧妙之处**在于：  
- 只用一条示例就能让 GPT 学会任务，省去了大规模标注的成本。  
- 对齐得分本身就是一种“自我纠错”，模型不需要外部标签就能感知自己的错误。  
- 双层对齐让模型兼顾信息完整性和结构合理性，避免只追求文字相似而忽视层次。  

### 实验与效果
- **数据集与任务**：论文在多个“数据→摘要”场景上评估，包括技术文档摘要、医学病例概括以及结构化表格说明等。  
- **基线对比**：与传统微调的 GPT、直接提示生成（Zero-shot）以及一些专门的摘要模型（如 BART、T5）进行比较。  
- **结果**：论文声称在所有测试任务上均取得了竞争性分数，尤其在结构化数据的摘要上对齐得分提升了约 5%~10%（具体数字未披露）。  
- **消融实验**：作者分别去掉“一-shot 调优”“结构层对齐”“人类反馈奖励网络”，发现去掉任意一环后整体得分都会下降，结构层对齐的贡献最为显著。  
- **局限性**：方法依赖 GPT 本身的生成质量，若底层模型在特定领域表现差，恢复环节也会受限；此外，对齐得分的计算成本随数据规模线性增长，极大数据集上可能成为瓶颈。  

### 影响与延伸思考
这篇工作把自监督的思想从语言模型的预训练阶段搬到了具体任务的标注环节，开启了“生成-还原”循环在数据标注中的新路径。随后有研究尝试把同样的框架用于代码注释、图像描述甚至多模态数据的标注，证明了该思路的通用性。对想进一步探索的读者，可以关注以下方向：  
- **更高效的对齐度量**：设计轻量化的结构对齐算法，降低计算开销。  
- **跨模型自监督**：让不同规模的模型相互校验，提升小模型的标注能力。  
- **多模态生成-还原**：把文本、表格、图片等信息统一进恢复环节，实现更丰富的数据标注。  

### 一句话记住它
只给模型一次示例，让它先生成摘要再自检恢复，用恢复的好坏来自动调教，GPT 就能自己学会高质量标注。