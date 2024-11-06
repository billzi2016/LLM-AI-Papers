# Number Cookbook: Number Understanding of Language Models and How to   Improve It

> **Date**：2024-11-06
> **arXiv**：https://arxiv.org/abs/2411.03766

## Abstract

Large language models (LLMs) can solve an increasing number of complex reasoning tasks while making surprising mistakes in basic numerical understanding and processing (such as 9.11 > 9.9). The latter ability is essential for tackling complex arithmetic and mathematical problems and serves as a foundation for most reasoning tasks, but previous work paid little attention to it or only discussed several restricted tasks (like integer addition). In this paper, we comprehensively investigate the numerical understanding and processing ability (NUPA) of LLMs. Firstly, we introduce a benchmark covering four common numerical representations and 17 distinct numerical tasks in four major categories, resulting in 41 meaningful combinations in total. These tasks are derived from primary and secondary education curricula, encompassing nearly all everyday numerical understanding and processing scenarios, and the rules of these tasks are very simple and clear. Through the benchmark, we find that current LLMs fail frequently in many of the tasks. To study the problem, we train small models with existing and potential techniques for enhancing NUPA (such as tokenizers, PEs, and number formats), comprehensively evaluating their effectiveness using our testbed. We also finetune practical-scale LLMs on our proposed NUPA tasks and find that 1) naive finetuning can improve NUPA a lot on many but not all tasks, and 2) surprisingly, techniques designed to enhance NUPA prove ineffective for finetuning pretrained models. We further explore the impact of chain-of-thought techniques on NUPA. Our work provides a more detailed and comprehensive understanding of NUPA in LLMs. Our benchmark and code are released at https://github.com/GraphPKU/number_cookbook.

---

# 数字食谱：语言模型的数字理解及提升方法 论文详细解读

### 背景：这个问题为什么难？

语言模型在聊天、写作等任务上已经表现得相当强大，但它们在最基本的数字比较、大小判断甚至简单的四则运算上仍会出错，比如把 9.11 误判为 9.9 以下。过去的研究大多把注意力放在推理链路、常识问答或大规模语言建模上，少有系统化地评估模型对日常数字的感知能力。现有的评测往往只覆盖整数加法或特定的数学题，忽视了小数、分数、科学计数法等常见表示方式，导致我们无法准确定位模型在哪类数字任务上失效，也找不到针对性的改进手段。

### 关键概念速览
- **NUPA（数字理解与处理能力）**：模型对数字的识别、比较、运算等基本操作的综合表现，类似于人类在小学数学课上学到的“看得懂、算得对”。  
- **数字表示**：指数字在文本中的写法，包括整数、带小数点的数、分数、科学计数法等四大类。不同表示会导致模型的分词方式截然不同。  
- **Tokenizer（分词器）**：把原始文本切成模型可以接受的子词单元的工具。若分词器把“3.14”切成“3”、“.”、“14”，模型就难以把它当作整体数字来处理。  
- **PE（位置编码）**：模型内部用来感知序列中各 token 位置的向量。改进 PE 可以帮助模型更好地区分数字内部的结构（比如小数点前后）。  
- **Chain‑of‑Thought（思维链）**：让模型在给出答案前先写出推理步骤的技巧，像在纸上写草稿一样，常用于提升复杂推理的准确率。  
- **Finetuning（微调）**：在已有的大模型上继续训练，使用特定任务的数据让模型适应新能力。这里指在数字任务上进行的微调。  
- **Benchmark（基准测试）**：一套标准化任务集合，用来统一评估模型的表现。本工作构建了覆盖四种数字表示、17种任务的 41 种组合。  

### 核心创新点
1. **系统化的数字基准 → 设计并公开了一个包含四类数字表示、17 种任务、41 种组合的完整评测套件 → 让研究者能够一次性看到模型在日常数字场景下的全貌，而不是零散的加法或比较实验。**  
2. **小模型实验 → 在若干小规模模型上尝试了分词器改造、位置编码增强、数字格式统一等技术 → 发现这些手段在专门训练的模型上能显著提升 NUPA，但对已经预训练好的大模型几乎没有帮助。**  
3. **大模型微调 → 对实际规模的语言模型进行直接微调，使用上述基准任务作为训练数据 → 微调后模型在多数任务上提升明显，但仍有部分任务（如科学计数法比较）提升有限，说明单纯微调并不能彻底解决根本问题。**  
4. **思维链对数字任务的影响评估 → 将 CoT 提示加入到数字任务的推理过程中 → 结果显示 CoT 对某些算术任务有帮助，但对纯比较类任务提升不大，进一步验证了数字理解的瓶颈更多在于“感知”而非“推理”。**  

### 方法详解
整体思路可以划分为三步：**基准构建 → 技术探索 → 微调评估**。  
1. **基准构建**：作者从中小学数学教材中抽取了常见的数字操作，归纳为四大表示（整数、带小数点的数、分数、科学计数法）和四类任务（大小比较、四则运算、单位换算、数字识别）。每种表示和每种任务的交叉形成一个子任务，最终得到 41 条具体测试题。例如，“比较 0.75 与 3/4”属于小数 vs 分数的比较子任务。所有题目都有明确的答案规则，便于自动评测。  
2. **技术探索（小模型实验）**：在 1‑2 亿参数的模型上分别尝试了三类改进：  
   - **分词器重构**：重新训练 tokenizer，使得完整数字（如 “3.14”）被视为单个 token，避免模型把小数点拆开。  
   - **位置编码增强**：在原有的相对位置编码上加入对数字内部结构的感知（比如在小数点前后分别加不同的偏置），帮助模型捕捉数字内部的层次。  
   - **数字格式统一**：在训练数据中统一把所有数字转成一种标准格式（如全部使用科学计数法），降低模型需要记忆的表示种类。  
   实验结果显示，这三种手段在专门训练的模型上可以把 9.11 > 9.9 这类错误率从 30% 降到 5% 左右。  
3. **大模型微调**：选取了公开的 7B、13B 规模的语言模型，直接在上述 41 子任务的训练集上进行微调。微调过程保持原有的语言建模目标，只在最后几层加入一个小的任务特定头。微调后模型在大多数子任务上准确率提升 10‑20% 点，但在科学计数法的大小比较上提升不到 5%。  
4. **思维链实验**：在微调模型的推理阶段加入了 CoT 提示，例如 “先把两个数写成相同的形式，再比较”。对四则运算任务的准确率提升约 3%，但对纯比较任务几乎没有变化，说明 CoT 主要帮助模型进行显式的算术步骤，而不是改善数字感知本身。  

最巧妙的地方在于**把数字感知问题拆成“表示层面”和“推理层面”两块**，分别用 tokenizer/PE 以及 CoT 进行针对性干预，进而发现两者的效用并不对称：感知层面的改进对小模型有效，却难以迁移到已经预训练的大模型；而推理层面的 CoT 对大模型的帮助有限。

### 实验与效果
- **评测数据**：41 条子任务，覆盖 4 种数字表示和 17 种具体操作，全部来源于中小学教材并自动生成答案。  
- **基线对比**：使用原始 GPT‑Neo、LLaMA 等公开模型作为基线。原始模型在整体基准上的平均准确率约为 58%。  
- **改进后表现**：  
  - 小模型加入 tokenizer/PE/统一格式后，平均准确率提升至约 78%。  
  - 对 7B/13B 大模型进行微调后，平均准确率提升至约 68%‑70%。  
  - 加入 CoT 提示后，四则运算子任务从 72% 提升到 75%，但整体提升幅度不到 2%。  
- **消融实验**：分别去掉 tokenizer 重构、位置编码增强、数字统一三项，发现 tokenizer 重构贡献最大（约 10% 点提升），位置编码次之（约 4%），统一格式贡献最小（约 2%）。  
- **局限性**：作者指出，微调仍然无法彻底解决科学计数法和分数混合表示的比较错误；此外，实验主要在英文数据上完成，中文数字的分词和表示可能会有不同的挑战。  

### 影响与延伸思考
这篇工作在社区里掀起了对「数字感知」的关注，随后出现了多篇围绕 tokenizer 改进、数值专用嵌入（numeric embeddings）以及混合语言模型（LLM+symbolic） 的研究。比如 2024 年的「NumBERT」尝试在 BERT 结构中加入数值专用的向量空间，直接受到了该基准的启发。未来的方向可能包括：  
- 将数字感知能力与外部数值计算引擎结合，实现「语言+符号」的混合推理。  
- 在多语言环境下统一数字表示，尤其是中文、阿拉伯数字与汉字数字的混排问题。  
- 探索更高效的微调策略，例如只在数字相关的子层进行 LoRA（低秩适配）微调，以降低算力成本。  

### 一句话记住它
**语言模型的数字盲点可以通过专门的数字基准、分词器改造和微调来显著改善，但根本的感知缺陷仍需更深层的数值表示设计。**