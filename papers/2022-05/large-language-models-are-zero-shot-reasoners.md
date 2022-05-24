# Large Language Models are Zero-Shot Reasoners

> **Date**：2022-05-24
> **arXiv**：https://arxiv.org/abs/2205.11916

## Abstract

Pretrained large language models (LLMs) are widely used in many sub-fields of natural language processing (NLP) and generally known as excellent few-shot learners with task-specific exemplars. Notably, chain of thought (CoT) prompting, a recent technique for eliciting complex multi-step reasoning through step-by-step answer examples, achieved the state-of-the-art performances in arithmetics and symbolic reasoning, difficult system-2 tasks that do not follow the standard scaling laws for LLMs. While these successes are often attributed to LLMs' ability for few-shot learning, we show that LLMs are decent zero-shot reasoners by simply adding "Let's think step by step" before each answer. Experimental results demonstrate that our Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP), symbolic reasoning (Last Letter, Coin Flip), and other logical reasoning tasks (Date Understanding, Tracking Shuffled Objects), without any hand-crafted few-shot examples, e.g. increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with large InstructGPT model (text-davinci-002), as well as similar magnitudes of improvements with another off-the-shelf large model, 540B parameter PaLM. The versatility of this single prompt across very diverse reasoning tasks hints at untapped and understudied fundamental zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive capabilities may be extracted by simple prompting. We hope our work not only serves as the minimal strongest zero-shot baseline for the challenging reasoning benchmarks, but also highlights the importance of carefully exploring and analyzing the enormous zero-shot knowledge hidden inside LLMs before crafting finetuning datasets or few-shot exemplars.

---

# 大语言模型是零样本推理者 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理里，算术、符号推理等需要多步逻辑的任务一直是模型的软肋。传统的语言模型往往只能直接给出答案，缺乏中间推理过程，导致在需要“系统2”思考的场景下准确率极低。出现的 few‑shot 方法通过在提示中加入几例完整的思考链（Chain of Thought，CoT）才能显著提升，但这依赖人工挑选、排版示例，成本高且对不同任务的迁移性差。于是社区迫切想知道：在没有任何示例的情况下，模型本身是否已经蕴含了可被激活的推理能力？

### 关键概念速览
**大语言模型（LLM）**：在海量文本上预训练得到的模型，拥有上百亿甚至上千亿参数，能够生成连贯的自然语言。  
**零样本（Zero‑Shot）**：模型在没有看到任何任务特定示例的情况下直接完成任务。  
**Few‑Shot 学习**：在提示中加入少量（通常 1‑5 条）任务示例，帮助模型理解任务格式。  
**思维链（Chain of Thought，CoT）**：让模型在给出最终答案前先写出逐步推理过程，类似人做数学题时的草稿。  
**提示工程（Prompt Engineering）**：通过设计输入文字的方式，引导模型产生期望的输出。  
**InstructGPT / text‑davinci‑002**：OpenAI 提供的指令式大模型，能够理解并遵循自然语言指令。  
**PaLM**：Google 开发的 5400 亿参数的大模型，同样支持指令式交互。  

### 核心创新点
1. **从 Few‑Shot 到 Zero‑Shot 的思维链迁移**  
   之前的 CoT 需要在提示里放入完整的思考示例，本文直接在答案前加上一句 “Let's think step by step”。这句自然语言指令本身就能触发模型自行展开推理链。  
2. **统一单一提示模板跨任务使用**  
   过去每个任务往往要手工调试示例格式，本文的 “一步步思考” 句子在算术、符号、时间和对象追踪等七类基准上均表现提升，证明同一个提示足以激活模型的通用推理机制。  
3. **把零样本表现提升到可比拟的 Few‑Shot 水平**  
   在 MultiArith 上从 17.7% 提升到 78.7%，在 GSM8K 上从 10.4% 提升到 40.7%，这些数字接近或超过了部分 few‑shot 基线，显示仅靠一句提示就能释放模型潜在的系统2能力。  

### 方法详解
整体思路非常简洁：在每个测试样本的答案位置前，插入一条固定的自然语言指令 “Let's think step by step”。模型收到的完整输入相当于 “题目 + 指令”。随后模型会自行生成一段文字，先列出推理步骤，最后给出答案。整个过程不需要任何手工标注的示例，也不需要额外的模型结构改动。

**步骤拆解**  
1. **准备原始题目**：从基准数据集抽取题干，例如 “John has 3 apples, Mary gives him 2 more. How many does he have now?”  
2. **拼接提示**：在题目后面直接加上 “Let's think step by step.”，形成完整的输入字符串。  
3. **模型生成**：将该字符串喂入 LLM，模型在解码时会先输出推理过程，如 “First, John has 3 apples. Then Mary gives him 2 more, so 3 + 2 = 5.”，随后输出答案 “5”。  
4. **答案抽取**：从模型输出的最后一行或特定标记处抓取数值，作为最终预测。

**为何一句话能起作用？**  
LLM 在预训练阶段已经见过大量包含“let's think step by step”之类的教学或解释性文本，这类句子在语料中往往伴随详细的推理过程。模型把这句话当作“请展开思考”的信号，从而切换到内部的推理子网络。换句话说，这是一种“软触发”，不需要显式的示例，只要提供了正确的触发词。

**最巧妙的地方**  
- **极简触发词**：只用一句英文短语即可激活多步推理，省去繁琐的 few‑shot 示例设计。  
- **跨语言、跨任务的通用性**：实验表明即使在中文或混合语言的题目上，同样的英文指令仍然有效，说明模型对指令的理解已经超越语言表层。  

### 实验与效果
- **测试任务**：算术（MultiArith、GSM8K、AQUA‑RAT、SVAMP）、符号推理（Last Letter、Coin Flip）、时间理解（Date Understanding）以及对象追踪（Tracking Shuffled Objects）。  
- **基线对比**：零样本直接回答（不加指令）与本文的 Zero‑shot‑CoT。以 InstructGPT（text‑davinci‑002）为例，MultiArith 准确率从 17.7% 提升到 78.7%，GSM8K 从 10.4% 提升到 40.7%；在 PaLM 540B 上也出现类似幅度的提升。  
- **消融实验**：作者尝试了不同的触发词（如 “think carefully”, “solve step by step”），发现 “Let's think step by step” 效果最稳健，说明具体措辞对激活推理链有细微影响。  
- **局限性**：对极其长的推理链或需要外部工具（如表格检索）的任务仍然表现不佳；此外，模型有时会产生冗余或错误的中间步骤，需要后处理才能保证答案的可靠性。  

### 影响与延伸思考
这篇工作让研究者重新审视了 LLM 的“隐藏能力”，推动了零样本推理的热潮。随后出现的 **Self‑Consistency**、**Least‑to‑Most Prompting** 等方法，都在此基础上进一步探索如何让模型自行生成并筛选多个推理路径。对想继续深入的读者，可以关注以下方向：  
- **自动化提示搜索**：利用元学习或强化学习寻找最有效的触发词。  
- **多模态推理**：把同样的零样本思维链概念扩展到图像、代码等非文本输入。  
- **可信度评估**：设计机制判断模型生成的中间步骤是否合理，从而过滤错误答案。  

### 一句话记住它
只要在答案前加上一句 “Let's think step by step”，大语言模型就能在零样本条件下自行展开多步推理，效果媲美专门设计的 few‑shot 示例。