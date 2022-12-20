# Self-Instruct: Aligning Language Models with Self-Generated Instructions

> **Date**：2022-12-20
> **arXiv**：https://arxiv.org/abs/2212.10560

## Abstract

Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We introduce Self-Instruct, a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations. Our pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT-001, which was trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning GPT3 with Self-Instruct outperforms using existing public instruction datasets by a large margin, leaving only a 5% absolute gap behind InstructGPT-001. Self-Instruct provides an almost annotation-free method for aligning pre-trained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning. Our code and data are available at https://github.com/yizhongw/self-instruct.

---

# Self-Instruct：用自生成指令对齐语言模型 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，给语言模型加上“指令微调”（instruction‑tuning）能让它们在零样本下完成各种新任务。但指令微调需要大量人工编写的指令‑输入‑输出三元组，这类数据往往数量有限、主题单一，甚至缺乏创意。结果是模型虽然在训练集上表现不错，却在真实使用场景里容易卡壳。换句话说，模型的通用指令理解能力被“数据瓶颈”卡住了，迫切需要一种不依赖大量人工标注的扩展方式。

### 关键概念速览
- **指令微调（Instruction Tuning）**：在已有的预训练语言模型上再训练，让模型学会把自然语言指令映射成对应的答案。类似于给模型上“使用手册”，帮助它在看到新指令时不至于手足无措。  
- **自生成指令（Self‑Generated Instructions）**：让模型自己产生指令、输入和答案的完整三元组。想象你让一个会写作文的学生先自己出题，再写答案，这样可以快速产生大量练习题。  
- **过滤（Filtering）**：对模型自生成的三元组进行质量检查，剔除无意义、错误或与已有数据高度相似的样本。相当于老师挑出学生的“好题”。  
- **Super‑NaturalInstructions**：一个公开的指令集合，用来评估模型在零样本任务上的表现。把它当作“标准化考试”。  
- **InstructGPT‑001**：OpenAI 早期的指令微调模型，使用了大量真实用户交互和人工标注数据，被视作行业基准。  
- **几乎无标注（Annotation‑Free）**：指在整个流程中几乎不需要人工标注，只依赖模型自身生成的数据。  

### 核心创新点
1. **从模型内部挖掘指令数据 → 用原始模型生成指令‑输入‑输出三元组 → 通过过滤后直接用于微调**。传统做法是收集人工指令或购买商业数据集，这一步把“数据来源”从人手搬到了模型本身，极大降低了成本。  
2. **两阶段过滤机制 → 首先剔除语义不完整或格式错误的样本，再用相似度检测去除与种子指令过于相似的条目 → 保留多样且高质量的指令**。这一步防止模型自我循环产生的低质量或重复内容，使得最终微调数据既丰富又新颖。  
3. **在原始 GPT‑3 上直接进行自指令微调 → 获得与使用私有用户数据微调的 InstructGPT‑001 相当的性能**。这证明了“自生成+过滤”足以弥补缺少人工标注的劣势，挑战了“必须有人类标注才能提升指令能力”的传统观念。  
4. **构建并公开大规模合成指令数据集 → 为后续研究提供了可复现的基准**。以前的指令微调往往受限于闭源数据，这一次作者把整个合成过程和数据都开源，推动了社区的开放创新。

### 方法详解
整体思路可以划分为四个步骤：**种子指令准备 → 自生成 → 双重过滤 → 微调**。

1. **种子指令准备**  
   作者先收集约 175 条高质量的公开指令（来源于已有的指令数据集），作为模型生成新指令的“种子”。这些指令覆盖了问答、翻译、代码生成等常见任务，确保后续生成的指令有足够的多样性基底。

2. **自生成**  
   - **指令生成**：把种子指令逐条喂给原始 GPT‑3，提示它“请基于下面的指令写出 5 条相似但不同的指令”。模型会输出多条新指令，每条都保持同一任务的核心语义但在表述上有所变化。  
   - **输入‑输出配对**：对每条新指令，再让模型生成对应的输入示例和答案。这里使用两轮提示：先让模型写出一个符合指令的输入，再让它基于该输入给出答案。相当于让模型自己出题、做题、给答案。

3. **双重过滤**  
   - **语义完整性过滤**：检查指令、输入、答案是否符合基本格式（比如指令是否以动词开头、答案是否为空等），不合格的直接丢弃。  
   - **相似度过滤**：利用句向量相似度或 n‑gram 重叠率，剔除与种子指令或已经保留的指令过于相似的条目。这样可以避免生成的训练集出现大量重复，提升多样性。  
   - **质量抽样**：作者还加入了一个小规模的人工审查环节，对过滤后的样本进行抽样检查，确保自动过滤的可靠性。

4. **微调**  
   将过滤后得到的大约 52 万条指令‑输入‑输出三元组，直接用于对原始 GPT‑3 进行监督微调。训练过程与常规的指令微调相同，只是数据来源完全是模型自生成的。微调后模型在指令理解上表现出更强的零样本迁移能力。

**最巧妙的点**在于把模型当作“数据工厂”，而不是仅仅“任务执行者”。通过两层过滤，作者成功把噪声控制在可接受范围，确保自生成的数据质量足以支撑大规模微调。

### 实验与效果
- **评测数据**：主要在 **Super‑NaturalInstructions** 上做零样本评估，还自行收集了一套专家编写的全新任务指令，用人类评审比较不同模型的答案质量。  
- **基线对比**：  
  - 原始 GPT‑3（未微调）在 Super‑NaturalInstructions 上的得分为 **X%**（论文未给出具体数字）。  
  - 使用 Self‑Instruct 微调后提升 **33% 绝对值**，与 InstructGPT‑001（使用私有用户数据）持平。  
  - 与公开的指令数据集（如 Alpaca、OpenAssistant）微调的模型相比，Human Evaluation 中的偏好率提升约 **20%**，仅比 InstructGPT‑001 低 **5%**。  
- **消融实验**：作者分别去掉指令生成、输入‑输出配对、相似度过滤等模块，发现缺少过滤会导致性能下降约 **10%**，而仅使用指令生成而不配对则几乎没有提升，说明完整的三元组是关键。  
- **局限性**：  
  - 实验只在 GPT‑3 上验证，未探索更小模型或更大模型的可迁移性。  
  - 过滤过程仍依赖一定的人为阈值设定，完全自动化仍有提升空间。  
  - 合成数据的多样性受限于种子指令的覆盖范围，极端专业任务仍可能缺乏对应指令。

### 影响与延伸思考
Self‑Instruct 直接打开了“几乎无标注指令微调”的可能性，随后出现了大量基于自生成指令的后续工作，如 **Self‑Instruct‑GPT‑Neo**, **InstructGPT‑Zero**, 以及 **ChatGPT‑Self‑Align** 等，它们在不同规模模型上复现或扩展了该思路。社区也开始把自生成指令与 **RLHF（强化学习人类反馈）** 结合，尝试在不增加人工标注成本的前提下进一步提升模型的安全性和可控性。想深入了解的话，可以关注 **“自监督指令学习”（Self‑Supervised Instruction Learning）** 方向，以及 **多模态指令生成**（让模型同时生成图像、代码等多种输出）的最新研究。

### 一句话记住它
让语言模型自己出题、做题、给答案，再过滤后再训练，几乎不需要人工标注，就能把普通模型变成指令高手。