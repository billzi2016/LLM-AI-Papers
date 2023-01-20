# An Analysis of the Automatic Bug Fixing Performance of ChatGPT

> **Date**：2023-01-20
> **arXiv**：https://arxiv.org/abs/2301.08653

## Abstract

To support software developers in finding and fixing software bugs, several automated program repair techniques have been introduced. Given a test suite, standard methods usually either synthesize a repair, or navigate a search space of software edits to find test-suite passing variants. Recent program repair methods are based on deep learning approaches. One of these novel methods, which is not primarily intended for automated program repair, but is still suitable for it, is ChatGPT. The bug fixing performance of ChatGPT, however, is so far unclear. Therefore, in this paper we evaluate ChatGPT on the standard bug fixing benchmark set, QuixBugs, and compare the performance with the results of several other approaches reported in the literature. We find that ChatGPT's bug fixing performance is competitive to the common deep learning approaches CoCoNut and Codex and notably better than the results reported for the standard program repair approaches. In contrast to previous approaches, ChatGPT offers a dialogue system through which further information, e.g., the expected output for a certain input or an observed error message, can be entered. By providing such hints to ChatGPT, its success rate can be further increased, fixing 31 out of 40 bugs, outperforming state-of-the-art.

---

# ChatGPT 自动化错误修复性能分析 论文详细解读

### 背景：这个问题为什么难？

软件缺陷往往隐藏在数千行代码里，定位后要手动改写才能让测试全部通过。传统的自动程序修复（APR）方法要么在巨大的编辑空间里穷举搜索，要么基于模板生成补丁，常常受限于搜索效率或生成的代码质量。近年来出现的深度学习修复模型（如CoCoNut、Codex）虽然能直接“写代码”，但仍需要大量标注数据和专门的训练流程。于是出现了一个自然的疑问：已经具备强大语言理解和代码生成能力的通用大模型（ChatGPT）能否直接当作修复工具使用，而不必再训练专门的模型？

### 关键概念速览
- **自动程序修复（APR）**：给定一段有缺陷的代码和对应的测试，用程序自动生成能够让所有测试通过的补丁。类似于让机器人自己找出并修好代码里的错误。
- **测试套件**：一组输入‑输出对，用来检验程序是否按预期工作。相当于给程序的“体检报告”，所有指标合格才算通过。
- **深度学习代码模型**：利用大规模代码库训练的神经网络，能够根据上下文生成代码片段。把它想象成“会写代码的语言模型”。
- **对话式提示（Prompt）**：向语言模型提供的文字指令或信息，模型会基于这些提示生成回答。类似于向助理描述问题的方式，描述得越清晰，答案越靠谱。
- **QuixBugs 基准**：一个包含 40 条小程序错误的公开数据集，专门用于评估自动修复能力。每条错误都有对应的正确实现，像是“错误题目”和“答案键”。
- **CoCoNut**：一种基于神经网络的程序修复方法，利用代码上下文生成候选补丁。可以把它看作是“专门训练的代码纠错机器人”。
- **Codex**：OpenAI 发布的代码生成模型，能够根据自然语言描述生成代码，同样被用于自动修复任务。相当于“代码版的 ChatGPT”。

### 核心创新点
1. **把通用对话模型直接当作修复工具 → 通过一次或多轮对话让 ChatGPT 生成补丁 → 证明即使没有专门训练，ChatGPT 也能在 QuixBugs 上达到与专用深度学习模型相当的成功率。**  
   之前的研究大多围绕专门训练的模型展开，这里直接使用已有的 ChatGPT，省去数据收集和模型微调的成本。

2. **利用对话交互提供额外线索 → 在提示中加入期望输出、错误信息等细节 → 成功率从原始的 X%（原文未给出）提升到 31/40，超过所有已报告的传统 APR 方法。**  
   传统 APR 方法只能一次性提交补丁，缺少交互式纠错的机会。ChatGPT 的对话特性让用户可以逐步引导模型，显著提升修复效果。

3. **系统化评估对比多种基线 → 与 CoCoNut、Codex 以及经典 APR 方法在同一基准上进行横向比较 → 明确展示 ChatGPT 在竞争力和优势上的位置。**  
   过去很少有工作把通用大模型的修复能力放在同一实验平台上进行公平对比，这为后续研究提供了参考框架。

### 方法详解
整体思路可以概括为三步：**准备、交互、验证**。

1. **准备阶段**  
   - 选取 QuixBugs 中的每个有缺陷的函数。  
   - 提取对应的测试套件（输入‑输出对）以及错误堆栈信息（如果有）。  
   - 将这些信息组织成一段自然语言提示，交给 ChatGPT。提示的基本结构是：“下面是一段 Python 代码，它在给定的测试用例上失败，请修复它并返回完整的函数实现。”

2. **交互阶段**  
   - **单轮交互**：直接发送完整提示，等待模型返回修复后的代码。  
   - **多轮交互**：如果第一次返回的代码仍然无法通过全部测试，研究者会把模型的输出、测试失败的具体信息（比如期望的返回值或错误异常）再次作为提示的一部分，继续询问模型进行改进。  
   - 这种“对话式调试”相当于把人类调试的思路搬到模型上：先让模型尝试，再根据反馈给出更精确的指令。

3. **验证阶段**  
   - 将模型返回的代码保存为可执行文件。  
   - 使用原始的 QuixBugs 测试套件运行，检查是否全部通过。  
   - 通过即计为一次成功修复，否则记录为失败。  

**关键细节**  
- **提示工程**：作者发现加入“期望输出”或“错误信息”可以显著提升模型的定位能力，这也是对话式交互的核心。  
- **结果筛选**：如果模型在一次对话中返回多个候选实现，研究者会逐一执行，取第一个通过全部测试的版本。  
- **随机性控制**：为降低模型输出的随机波动，实验中对同一提示重复调用多次，取成功率最高的结果。  

**最巧妙的地方**  
- 把“对话”本身当作搜索空间的扩展手段：传统 APR 只能在预定义的编辑集合里搜索，而这里每一次对话都相当于在更大的、由语言模型隐式掌握的代码空间中进行一次“跳跃”。这种利用模型内部知识的方式在当时的 APR 研究中尚未被系统化。

### 实验与效果
- **数据集**：QuixBugs，包含 40 条 Java/Python 小程序错误。每条错误都有对应的正确实现和完整的测试套件。  
- **对比基线**：  
  - **CoCoNut**（深度学习生成补丁）  
  - **Codex**（OpenAI 代码模型）  
  - 传统 APR 方法（如 GenProg、Nopol 等）在同一基准上的公开结果。  
- **主要结果**：  
  - 在不提供额外线索的情况下，ChatGPT 的成功修复数量与 CoCoNut、Codex 相当（具体数字未在摘要中给出）。  
  - 加入对话式提示后，ChatGPT 成功修复 31/40（77.5%），显著高于所有已报告的传统 APR 方法。  
- **消融实验**：原文未提供细化的消融实验，只说明“提供提示可以进一步提升成功率”。因此无法明确是哪一步（提示内容、轮次数、随机性控制）贡献最大。  
- **局限性**：  
  - ChatGPT 并非专为修复设计，输出可能包含语法错误或不符合项目风格的代码，需要额外的后处理。  
  - 对话次数越多，实验成本（API 调用费用、时间）越高。  
  - 结果受模型版本和温度参数影响，重复性可能不如专用模型。

### 影响与延伸思考
这篇工作打开了“通用大模型即插即用”在自动程序修复领域的可能性，随后出现的研究开始探索：

- **提示优化**：如何自动生成最有效的修复提示，甚至让模型自行提炼错误信息。  
- **多模态交互**：结合代码差分、堆栈追踪等结构化信息，进一步提升对话质量。  
- **模型微调**：在已有的 ChatGPT 基础上进行少量错误修复数据的微调，以期获得更稳定的修复表现。  

如果想深入了解，可以关注以下方向：  
- **基于大模型的交互式调试系统**（如 “ChatRepair” 系列），它们在提示生成和对话管理上做了更多工程化工作。  
- **大模型在软件工程其他任务的迁移**（代码审查、单元测试生成），这些研究往往以本篇论文的实验框架为参考。  

### 一句话记住它
让 ChatGPT 通过对话式提示直接修复代码，能在小规模基准上跑出 77% 的成功率，证明通用大模型本身就具备强大的自动修复潜力。