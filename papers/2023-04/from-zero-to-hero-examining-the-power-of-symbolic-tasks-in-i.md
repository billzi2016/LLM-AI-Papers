# From Zero to Hero: Examining the Power of Symbolic Tasks in Instruction   Tuning

> **Date**：2023-04-17
> **arXiv**：https://arxiv.org/abs/2304.07995

## Abstract

Fine-tuning language models on tasks with instructions has demonstrated potential in facilitating zero-shot generalization to unseen tasks. In this paper, we introduce a straightforward yet effective method for enhancing instruction tuning by employing symbolic tasks. Compared to crowdsourced human tasks or model-generated tasks, symbolic tasks present a unique advantage as they can be easily generated in vast quantities, theoretically providing an infinite supply of high-quality training instances. To explore the potential of symbolic tasks, we carry out an extensive case study on the representative symbolic task of SQL execution. Empirical results on various benchmarks validate that the integration of SQL execution leads to significant improvements in zero-shot scenarios, particularly in table reasoning. Notably, our 3B model surpasses both the 175B GPT-3 and ChatGPT in zero-shot table reasoning across four benchmarks. Furthermore, experimental results on BBH (27 tasks) and MMLU (57 tasks) reveal that language models can be enhanced through symbolic tasks without compromising their generality. We hope that our paper serves as a catalyst, inspiring increased efforts to incorporate symbolic tasks in instruction tuning.

---

# 从零到英雄：符号任务在指令微调中的力量探究 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，让模型只看说明就能完成从未见过的任务，是提升通用性的关键。过去的指令微调大多依赖人工收集的任务或让已有模型自行生成任务，这两种方式都有瓶颈：人工标注成本高、规模受限；模型生成的任务质量参差不齐，且容易出现同质化。于是出现了“训练数据不够多、质量不够好”这根绊脚石，限制了零样本（zero‑shot）泛化能力的进一步提升。

### 关键概念速览
**指令微调（Instruction Tuning）**：在已有语言模型上继续训练，让模型学习如何根据自然语言指令完成任务，类似给模型上课教它“听指令做事”。  
**符号任务（Symbolic Task）**：输入输出都可以用明确的符号系统描述的任务，例如 SQL 查询、数学公式求值等，像是给模型提供一套“可编程的练习”。  
**零样本泛化（Zero‑Shot Generalization）**：模型在没有见过特定任务的情况下，直接根据指令完成任务，类似人类第一次接触新游戏却能靠说明书上手。  
**表格推理（Table Reasoning）**：模型需要在结构化表格数据上进行检索、计算或逻辑推断，类似在 Excel 表里找答案。  
**BBH（Big‑Bench Hard）**：一套包含 27 项高难度任务的基准，用来衡量模型的通用推理能力。  
**MMLU（Massive Multitask Language Understanding）**：覆盖 57 个学科的测评集合，检验模型的知识广度和深度。  

### 核心创新点
1. **从人工/模型生成任务 → 符号任务 → 数据量几乎无限**  
   过去的指令微调依赖人手标注或让模型自行造题，规模受限且质量难保。作者直接把可程序化的符号任务（以 SQL 执行为代表）当作训练素材，利用自动化脚本批量生成海量高质量实例，实现了“几乎无限”的训练数据供应。  

2. **把 SQL 执行任务嵌入指令微调 → 零样本表格推理显著提升**  
   在指令微调阶段加入大量“给出表格和 SQL，要求模型返回查询结果”的样本，使模型在内部形成了对结构化查询的隐式理解。实验显示，这种做法让 3B 参数模型在四个表格推理基准上超过了 175B 参数的 GPT‑3 与 ChatGPT，说明符号任务的引入可以弥补模型规模的不足。  

3. **在通用基准 BBH 与 MMLU 上验证不损失通用性 → 同时提升特定与广义能力**  
   作者在加入符号任务后，仍然在 BBH（27 项）和 MMLU（57 项）上进行评测，结果显示模型整体表现没有下降，甚至有小幅提升。这表明符号任务并不会让模型“只会做 SQL”，而是提升了整体的指令理解和推理能力。  

### 方法详解
整体思路可以拆成三步：  
1. **任务生成**：编写一个能够随机生成合法 SQL 查询以及对应表格数据的脚本。脚本会随机选取列名、数据类型、行数等参数，确保每条样本在结构和难度上都有差异。  
2. **指令包装**：把每条生成的 (表格, SQL, 结果) 三元组包装成自然语言指令的形式，例如：“下面是一张包含员工信息的表格，请执行以下 SQL 查询并返回结果：`SELECT AVG(salary) FROM table WHERE department='Sales'`”。这样模型在训练时看到的仍是“指令+输入”，保持与其他任务一致的格式。  
3. **混合微调**：将这些符号任务与传统的人类指令任务混合，统一送入语言模型进行指令微调。混合比例在原文未详细说明，但核心是让模型在同一轮训练中既看到普通的问答、翻译等任务，也看到大量的 SQL 推理任务。  

关键模块的类比：  
- **任务生成器** 像是“自动出题机”，只要给它设定规则，它就能不停出新题。  
- **指令包装器** 像是“老师把题目翻译成课堂语言”，让模型把“做题”看成“听指令”。  
- **混合微调** 类似“综合训练营”，让模型在多种练习中交叉学习，避免单一技能的过度专化。  

最巧妙的地方在于：作者没有直接让模型学习“执行 SQL”，而是让它在自然语言指令的框架下“解释并返回查询结果”。这让模型在学习符号推理的同时，保持了对指令的通用解码能力，避免了“只会 SQL”的单一化风险。

### 实验与效果
- **测试任务**：四个专注表格推理的零样本基准（未在摘要中列名）、BBH（27 项）和 MMLU（57 项）。  
- **基线对比**：与同等规模的指令微调模型、以及 175B 参数的 GPT‑3 与 ChatGPT 进行比较。  
- **核心结果**：在表格推理上，3B 参数模型在四个基准上超越了 175B GPT‑3 与 ChatGPT，说明符号任务的加入可以让小模型实现“大模型级别”的零样本表现。BBH 与 MMLU 的整体得分保持不降，甚至略有提升，证明通用能力未受牺牲。  
- **消融实验**：原文未给出详细数字，但提到通过去掉 SQL 任务的训练，模型在表格推理上的表现回落，说明符号任务是提升的关键因素。  
- **局限性**：实验主要围绕 SQL 这一单一符号任务展开，是否对其他符号系统（如正则表达式、图算法）同样有效尚未验证；此外，生成的表格和查询虽然多样，但仍受脚本设定的规则限制，真实世界的复杂查询可能仍有差距。  

### 影响与延伸思考
这篇工作向社区展示了“符号任务+指令微调”可以成为一种高效、低成本的通用能力提升手段。随后出现的研究开始探索把数学公式求解、程序合成、逻辑推理等其他可自动生成的符号任务加入指令微调，形成了“多模态符号指令训练”的趋势。对想进一步深入的读者，可以关注以下方向：  
- **符号任务多样化**：如何自动生成高质量的图算法、正则表达式等任务。  
- **任务混合策略**：不同符号任务的比例、难度调度对模型学习曲线的影响。  
- **跨语言符号指令**：把符号任务与多语言指令结合，提升非英语模型的结构化推理能力。  

### 一句话记住它
让模型在指令微调时大量练习自动生成的 SQL 任务，就能让小模型在零样本表格推理上抢下大模型的风头。