# Quiet-STaR: Language Models Can Teach Themselves to Think Before   Speaking

> **Date**：2024-03-14
> **arXiv**：https://arxiv.org/abs/2403.09629

## Abstract

When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is implicit in almost all written text. For example, this applies to the steps not stated between the lines of a proof or to the theory of mind underlying a conversation. In the Self-Taught Reasoner (STaR, Zelikman et al. 2022), useful thinking is learned by inferring rationales from few-shot examples in question-answering and learning from those that lead to a correct answer. This is a highly constrained setting -- ideally, a language model could instead learn to infer unstated rationales in arbitrary text. We present Quiet-STaR, a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions. We address key challenges, including 1) the computational cost of generating continuations, 2) the fact that the LM does not initially know how to generate or use internal thoughts, and 3) the need to predict beyond individual next tokens. To resolve these, we propose a tokenwise parallel sampling algorithm, using learnable tokens indicating a thought's start and end, and an extended teacher-forcing technique. Encouragingly, generated rationales disproportionately help model difficult-to-predict tokens and improve the LM's ability to directly answer difficult questions. In particular, after continued pretraining of an LM on a corpus of internet text with Quiet-STaR, we find zero-shot improvements on GSM8K (5.9%$\rightarrow$10.9%) and CommonsenseQA (36.3%$\rightarrow$47.2%) and observe a perplexity improvement of difficult tokens in natural text. Crucially, these improvements require no fine-tuning on these tasks. Quiet-STaR marks a step towards LMs that can learn to reason in a more general and scalable way.

---

# Quiet‑STaR：语言模型可以自学先思考再输出 论文详细解读

### 背景：这个问题为什么难？

在自然语言生成里，模型往往直接预测下一个词，缺少“思考”过程。早期的推理方法（如 CoT）只能在特定的问答或代码任务上通过示例让模型写出推理链，范围被严格限定在少数几类题目。若想让模型在任意文本中自行补全隐含的推理步骤，面临两大难题：一是生成完整思考会显著增加计算开销；二是模型本身并不知道何时该停下来“思考”，也不清楚如何把思考结果用于后续预测。于是，如何让大模型在不额外微调特定任务的情况下，学会在每个 token 前产生有用的内部解释，成为了瓶颈。

### 关键概念速览
- **Self‑Taught Reasoner（STaR）**：一种让模型从少量带有推理过程的示例中学习，只有产生正确答案的推理才会被保留并用于后续训练。类似于学生只记住老师讲对的解题步骤。
- **思考（Thought）**：模型在生成下一个词之前插入的文字片段，用来解释或预测接下来会出现的内容。可以把它想象成写作时的“草稿注释”。
- **并行采样（Parallel Sampling）**：一次性为整段文本生成所有 token 的思考，而不是逐个 token 递归生成，类似于一次性写完整篇文章的提纲再填细节，极大提升效率。
- **思考起止标记（Thought Start/End Tokens）**：模型学习的特殊符号，标记思考的开始和结束，帮助模型在生成流中明确何时进入思考模式。
- **扩展教师强制（Extended Teacher Forcing）**：训练时强制模型在已知的正确文本上输出对应的思考片段，类似于老师在课堂上先给出答案的推导过程，让学生模仿。
- **困惑度（Perplexity）**：衡量语言模型对真实文本的预测能力的指标，数值越低说明模型越“懂”语言。

### 核心创新点
1. **从少量 QA 推理到全局文本思考**  
   之前的 STaR 只能在问答示例里学习推理链，局限在特定任务。Quiet‑STaR 把思考扩展到任意文本的每个 token，模型在阅读整段文字时会主动生成解释。这样一来，模型不再依赖任务标签，而是学会在自然语言流中自发产生内部推理。

2. **并行生成思考的采样算法**  
   传统逐 token 生成思考会把计算成本乘以思考次数，几乎不可行。作者提出一种 tokenwise 并行采样：一次性为所有待预测 token 生成对应的思考片段，并用可学习的起止标记把思考嵌入生成序列。相当于一次性给每个词配上“注释”，大幅降低了时间开销。

3. **可学习的思考边界标记**  
   为了让模型知道何时进入思考、何时返回正文，论文引入了两个专门的 token（<THINK>、</THINK>）。模型在训练时学习在合适的位置插入这些标记，类似于写作时主动加上“[思考] … [/思考]”。这让模型在推理时不再是盲目生成，而是有结构的内部对话。

4. **扩展教师强制让思考与答案同步**  
   在训练阶段，作者把真实文本的每个 token 对应的思考片段作为教师信号强制喂入模型，确保思考与后续 token 紧密关联。这样模型学到的思考不仅是随意的解释，而是对下一个词的有用预测，提升了整体生成质量。

### 方法详解
整体思路可以分为三步：**（1）思考标记化、（2）并行思考采样、（3）扩展教师强制训练**。

1. **思考标记化**  
   首先在原始语料中插入两个特殊 token，分别表示思考的开始和结束。每当模型在生成时遇到 <THINK>，它进入“内部思考模式”，随后输出一段自由文本（思考），直到出现 </THINK> 为止。思考的长度不固定，模型自行决定何时结束。

2. **并行思考采样**  
   对于一段待预测的文本，模型一次性生成所有 token 的思考序列。实现方式是把原始输入复制成 N 份（N 为待预测 token 数），在每份的对应位置插入 <THINK>，并让模型并行输出 N 条思考。随后，模型把每条思考的输出拼回原始序列，形成“思考+正文”的完整流。这样做的好处是 GPU 可以一次性处理所有位置的注意力计算，避免了逐步递归的时间累积。

3. **扩展教师强制**  
   训练时，作者利用已有的互联网文本作为“老师”。对于每个真实 token，预先用人工或自动方式生成一个对应的思考（比如解释前文概念、预测下一个词的语义），并把它放在 <THINK> 与 </THINK> 之间。模型在前向传播时被强制输出这些思考，而不是自行猜测。损失函数同时包含思考的交叉熵和正文 token 的交叉熵，两者加权求和，使模型学会把思考当作提升正文预测的工具。

**最巧妙的点**在于把思考当作一种“可学习的中间表示”，并通过并行采样把它的生成成本压到和普通语言模型相当的水平。传统上，思考链是线性的、耗时的；这里的并行化让思考几乎是“隐形”的计算负担。

### 实验与效果
- **测试任务**：作者在 GSM8K（数学文字题）和 CommonsenseQA（常识问答）上做了零样本评估，还在自然文本上测量了困惑度的变化。
- **主要结果**：在 GSM8K 上，准确率从 5.9% 提升到 10.9%；在 CommonsenseQA 上，从 36.3% 提升到 47.2%。这两个提升都是在没有针对任务进行任何微调的情况下实现的。对自然文本的困惑度也出现了显著下降，尤其是对那些原本预测困难的 token。
- **消融实验**：论文分别去掉并行采样、思考标记或扩展教师强制，发现去掉任何一个模块都会导致性能回落，尤其是去掉思考标记后，模型几乎恢复到普通语言模型的水平，说明思考边界的显式学习是关键。
- **局限性**：作者指出，思考的质量仍然依赖于预训练语料的多样性，某些专业领域的隐式推理仍然难以捕获；此外，思考生成虽然并行化，但在极长序列上仍会带来一定的显存压力。

### 影响与延伸思考
Quiet‑STaR 为“在语言模型内部自发思考”提供了可扩展的实现路径，随后的工作开始探索更细粒度的思考控制（如层级思考、跨句思考）以及把思考用于检索增强、代码生成等场景。推测未来会有更多研究把思考标记与强化学习结合，让模型学会在需要时主动开启思考，而在简单情形下直接输出。想进一步了解，可以关注 **Self‑Consistency**、**Tree‑of‑Thoughts** 等后续提出的多步推理框架，以及 **Meta‑Prompting** 系列对思考触发机制的探索。

### 一句话记住它
Quiet‑STaR 让大语言模型在每个词前并行生成“思考注释”，把隐式推理显式化，从而在零样本任务上显著提升准确率。