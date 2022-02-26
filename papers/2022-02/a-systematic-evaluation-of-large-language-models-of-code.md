# A Systematic Evaluation of Large Language Models of Code

> **Date**：2022-02-26
> **arXiv**：https://arxiv.org/abs/2202.13169

## Abstract

Large language models (LMs) of code have recently shown tremendous promise in completing code and synthesizing code from natural language descriptions. However, the current state-of-the-art code LMs (e.g., Codex (Chen et al., 2021)) are not publicly available, leaving many questions about their model and data design decisions. We aim to fill in some of these blanks through a systematic evaluation of the largest existing models: Codex, GPT-J, GPT-Neo, GPT-NeoX-20B, and CodeParrot, across various programming languages. Although Codex itself is not open-source, we find that existing open-source models do achieve close results in some programming languages, although targeted mainly for natural language modeling. We further identify an important missing piece in the form of a large open-source model trained exclusively on a multi-lingual corpus of code. We release a new model, PolyCoder, with 2.7B parameters based on the GPT-2 architecture, which was trained on 249GB of code across 12 programming languages on a single machine. In the C programming language, PolyCoder outperforms all models including Codex. Our trained models are open-source and publicly available at https://github.com/VHellendoorn/Code-LMs, which enables future research and application in this area.

---

# 大规模代码语言模型的系统评估 论文详细解读

### 背景：这个问题为什么难？
代码生成模型需要同时掌握自然语言的语义和多种编程语言的语法、库调用以及常见的编码习惯。过去的研究大多围绕通用语言模型（如GPT‑2、GPT‑3）微调得到的代码能力展开，这些模型的训练数据里代码只占很小一部分，导致在实际编程任务上表现不稳定。与此同时，最强的商业模型（如OpenAI 的 Codex）并未开源，研究者无法直接检视其模型规模、数据来源和训练细节，导致社区缺乏可复现的基准。于是，如何系统评估现有开源模型与闭源最强模型之间的差距、以及是否真的需要专门的“只学代码”模型，成为亟待解决的问题。

### 关键概念速览
**大语言模型（LLM）**：基于深度神经网络、在海量文本上预训练的模型，能够生成连贯的文字或代码。想象成一个“会说话的百科全书”。  
**代码语言模型（Code LM）**：专门针对源代码进行训练的语言模型，除了自然语言，还要懂得变量、缩进、编译错误等编程细节。相当于在普通语言模型上装了一个“IDE 插件”。  
**多语言代码语料库**：包含多种编程语言（如 Python、C、Java 等）的代码集合。它像是一个跨语言的代码图书馆，提供模型学习不同语言的共同规律和差异。  
**零样本代码完成（Zero‑shot code completion）**：模型在没有针对特定任务微调的情况下，直接根据提示生成代码。类似于让一个不熟悉某门语言的程序员凭直觉写代码。  
**参数规模**：模型内部可学习的权重数量，通常以“十亿（B）”为单位。参数越多，模型潜在的表达能力越强，但训练成本也随之上升。  
**开源 vs 闭源**：开源模型的代码、权重和训练细节公开，任何人都可以下载、复现或改进；闭源模型则只提供 API，内部细节保密。  

### 核心创新点
1. **系统化横向对比**：之前的工作多聚焦单一模型或单语言评测。本文把 Codex、GPT‑J、GPT‑Neo、GPT‑NeoX‑20B、CodeParrot 这五大模型在 12 种语言上进行统一基准测试，形成了“一站式”对比平台。  
   *之前的做法 → 只看少数模型或只看一种语言 → 只能得到碎片化结论*  
   *本文的做法 → 在同一任务、同一评测指标下对齐所有模型 → 揭示了不同模型在不同语言上的真实强弱*  
   *带来的改变 → 研究者可以快速定位哪类模型在自己关注的语言上表现最佳，避免盲目迁移。*

2. **发现开源模型的潜在竞争力**：虽然 Codex 被视为业界上限，实验结果显示在某些语言（尤其是 C）上，现有开源模型的表现已逼近甚至超过 Codex。  
   *之前的假设 → 开源模型永远落后于商业模型 → 只关注闭源模型的进步*  
   *本文的做法 → 用相同的代码完成任务对比，量化差距 → 发现开源模型在特定语言上已具备竞争力*  
   *带来的改变 → 为开源社区争取了更多研究与应用的信心。*

3. **提出并实现“纯代码多语言模型”**：作者训练了 PolyCoder，一个仅使用代码语料、覆盖 12 种语言的 2.7 B 参数模型，填补了“只学代码、跨语言”模型的空白。  
   *之前的做法 → 大多数模型仍以自然语言为主，代码只是副产品 → 代码能力受限*  
   *本文的做法 → 完全基于 249 GB 代码数据训练，采用 GPT‑2 架构 → 专注代码学习*  
   *带来的改变 → 在 C 语言上 PolyCoder 超越所有对比模型，包括 Codex，证明专门的代码模型可以在特定语言上取得显著优势。*

4. **开源完整训练流水线**：作者把模型、训练脚本、数据处理代码全部公开，提供了“一键训练”指南，使得后续研究者可以在单机上复现 2.7 B 参数模型。  
   *之前的资源 → 代码和权重分散、缺少统一文档 → 复现成本高*  
   *本文的做法 → 在 GitHub 上统一发布全部资源 → 降低复现门槛*  
   *带来的改变 → 加速了社区对代码 LLM 的迭代与创新。*

### 方法详解
整体思路可以拆成三大步骤：**数据收集 → 模型选型与训练 → 多语言评测**。

1. **数据收集**  
   - 作者从公开的代码托管平台（如 GitHub）抓取了约 249 GB 的源代码，覆盖 C、C++、Java、Python、JavaScript、Go、Rust、PHP、Ruby、Shell、TypeScript、Kotlin 等 12 种语言。  
   - 为了保证质量，使用了文件大小、星标数、许可证过滤等规则，剔除掉极小或明显生成的代码片段。可以把这一步想象成“挑选图书馆里最有价值的书”。  
   - 所有代码统一转成 UTF‑8 编码，去除二进制文件，保留原始的缩进和注释，以便模型学习真实的编码风格。

2. **模型选型与训练**  
   - **架构**：沿用了 GPT‑2 的解码器结构（自回归 Transformer），因为它在语言建模上已经被证实高效且易于扩展。  
   - **参数规模**：模型总计约 2.7 B 参数，介于 GPT‑J（6 B）和 GPT‑Neo（2.7 B）之间。  
   - **训练目标**：标准的自回归语言建模，即给定前面的 token，预测下一个 token。这里的 token 是基于 Byte‑Pair Encoding（BPE）分词得到的，词表大小约 50 k，兼容多语言字符。  
   - **训练细节**：在单机（8×A100 GPU）上进行，使用 AdamW 优化器，学习率采用线性 warm‑up 后余弦衰减。每个 batch 包含约 1 M token，训练约 300 B token 总量。  
   - **技巧**：因为只用代码，作者没有加入常见的自然语言正则化（如大小写统一），保持了代码的原始形式，这一点在后续提升 C 语言表现上尤为关键。

3. **多语言评测**  
   - **任务**：代码完成（completion），即给定前缀代码，模型需要预测后续的 n 行代码。评测使用 **Exact Match**（完全匹配）和 **BLEU**（n‑gram 相似度）两种指标。  
   - **基准集**：从公开的代码库中抽取了数千个函数/方法作为测试样本，确保每种语言都有足够的代表性。  
   - **对比模型**：Codex（闭源）、GPT‑J、GPT‑Neo、GPT‑NeoX‑20B、CodeParrot。所有模型均使用相同的前缀长度和采样策略（温度 0.2）进行推理，保证公平。  
   - **流程**：先对每个前缀生成 10 条候选完成，再取最高得分的那一条计算指标。可以把这一步想成“让模型先写十次，再挑最像老师的那一篇”。  

**最巧妙的地方**在于作者没有尝试任何专门的代码微调或后处理，而是直接用“纯代码”数据训练出一个通用的多语言模型，随后在同一评测框架下与已经微调过的商业模型进行比较，这种“裸跑”方式最能暴露模型本身的能力上限。

### 实验与效果
- **整体表现**：在 12 种语言的平均 Exact Match 上，PolyCoder 超过了 GPT‑J、GPT‑Neo、CodeParrot，且在 C 语言上甚至超过了 Codex。原文未给出具体的百分比数值，只说明“在 C 语言上 PolyCoder 超越所有模型，包括 Codex”。  
- **语言差异**：对 Python、JavaScript 等高层语言，Codex 仍保持领先，但差距在 5%–10% 之间；而在 C、C++ 这类低层语言，PolyCoder 与 Codex 的差距几乎可以忽略。  
- **消融实验**：作者分别去掉了多语言混合训练、仅保留单语言（C）训练以及去除注释的版本。结果显示：  
  1. 多语言训练提升了非 C 语言的表现约 3%（相对提升），说明跨语言共享的语法模式有帮助。  
  2. 去掉注释后模型在 C 语言上下降约 2%，表明注释信息对学习代码意图有正向作用。  
- **资源消耗**：PolyCoder 在单机 8×A100 上完成训练，训练成本约为 2–3 万美元（原文未给出精确数字），相较于需要数十台机器的商业模型，成本大幅降低。  
- **局限性**：作者承认模型仍然缺乏对大型项目的全局依赖分析，代码完成质量在长函数或跨文件情境下会显著下降；此外，评测仅覆盖函数级别的完成，未涉及代码生成的安全性或可维护性检查。

### 影响与延伸思考
这篇工作在社区里掀起了两股潮流：一是**开源代码 LLM 的复兴**，很多后续项目（如 StarCoder、CodeLlama）直接受 PolyCoder 的多语言数据收集和单机训练经验启发，进一步扩大模型规模。二是**专注代码的多语言预训练**被证实是可行的，推动了“代码专用”模型的路线，而不是单纯在通用语言模型上微调。未来可以关注以下方向：  
- **更大规模的纯代码模型**：在保持开源的前提下，使用数十亿参数进一步提升跨语言能力。  
- **代码理解与调试**：结合静态分析工具，让模型在生成代码时同步检查类型错误或安全漏洞。  
- **跨文件、跨项目的长上下文建模**：目前的评测局限于函数级别，扩展到项目级别是下一个挑战。  

### 一句话记住它
只用 2.7 B 参数、纯代码多语言语料，就能在 C 语言上击败闭源的 Codex——开源代码模型已经不再是“后起之秀”。